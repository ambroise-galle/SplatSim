"""BotanicGarden → scène canonique du pipeline (même contrat qu'Oxford et DiTer++).

- repère fixe = repère de la trajectoire de référence (première pose du VLP-16) ; repère `base` = repère `velodyne`
  (origine des rayons), T_base_lidar = identité ;
- trajectoire = GT (pose datée en fin de tour, recalée de GT_DT_S = −25 ms : mesuré au gyroscope Xsens) mise en forme
  par le gyroscope entre les poses (`botanic_lidar.shaped_trajectory`) ;
- balayages VLP-16 : points ≥ `--min-range`, deskewés au temps de chaque point, ligne = anneau trié par élévation,
  colonne = azimut relatif à la coupure (modèle `stage0/vlp16.json`) ;
- caméras couleur `rgb_left` (rgb0, dossier c54d7a) et `rgb_right` (rgb1, c54d79) : images pleine résolution
  1920×1200 REDRESSÉES (k1, k2) en sténopé de même K, une vue sur `--every` (défaut 4 → 10 Hz, les vues du bag VLIO),
  masque des pixels valides ; T_base_cam depuis `calib_chain.yaml` (T_rgb0_vlp16, T_rgb0_rgb1).

  venv/bin/python -m splatpipe.stage0.ingest_botanic --seq 1018_00 --out data_local/botanicgarden/scene_1018_00
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re

import numpy as np

from splatpipe.stage0.botanic_lidar import NAME, load_raw, raw_files, shaped_trajectory, deskew

GT_DT_S = -0.025
CAMS = {"rgb_left": ("rgb0", "c54d7a"), "rgb_right": ("rgb1", "c54d79")}


def read_chain(path: str) -> dict:
    s = open(path).read()
    out = {}
    for m in re.finditer(r"(T_\w+):\s*!!opencv-matrix.*?data:\s*\[(.*?)\]", s, re.S):
        out[m.group(1)] = np.array([float(x) for x in m.group(2).replace("\n", " ").split(",")]).reshape(4, 4)
    return out


def read_intr(path: str) -> dict:
    import yaml

    txt = open(path).read().replace("%YAML:1.0", "")
    d = yaml.safe_load(txt)
    p, q = d["projection_parameters"], d["distortion_parameters"]
    return dict(fx=p["fx"], fy=p["fy"], cx=p["cx"], cy=p["cy"], k1=q["k1"], k2=q["k2"], p1=q.get("p1", 0.0),
                p2=q.get("p2", 0.0))


def _yaml(d) -> str:
    import yaml

    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True, default_flow_style=None, width=120)


def main():
    import cv2

    from splatpipe.common.lidar_model import StructuredSpinningLidarModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_local/botanicgarden")
    ap.add_argument("--seq", default="1018_00")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-range", type=float, default=1.0)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--only-camera", action="store_true")
    args = ap.parse_args()
    S = f"{args.root}/{args.seq}"
    raw = f"{S}/lidar_raw"
    out = args.out
    for sub in ("lidar/vlp16_deskewed", "lidar_model", "traj", "poses", "imu", "reports"):
        os.makedirs(f"{out}/{sub}", exist_ok=True)

    # ------------------------------------------------------------------ modèle, trajectoire
    model = StructuredSpinningLidarModel.load_json(f"{S}/stage0/{NAME}.json")
    model.save_json(f"{out}/lidar_model/{NAME}.json")
    traj, fit = shaped_trajectory(f"{args.root}/gt_pose/gt_{args.seq}_qnew.tum", f"{raw}/imu_data.npz", GT_DT_S)
    traj.body = "base"
    traj.meta.update({"body": "base = repère velodyne", "gt_dt_s": GT_DT_S,
                      "imu_fit": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fit.items()}})
    traj.save(f"{out}/traj/spline.npz")
    os.system(f"cp {raw}/imu_data.npz {out}/imu/imu_xsens.npz")

    # ------------------------------------------------------------------ balayages
    ring_to_row = np.full(int(model.row_ring_ids.max()) + 1, -1)
    ring_to_row[model.row_ring_ids] = np.arange(model.n_rows)
    a0 = float(model.column_azimuths_rad[0] - np.sign(model.column_azimuths_rad[1] - model.column_azimuths_rad[0])
               * 0.5 * abs(model.column_azimuths_rad[1] - model.column_azimuths_rad[0]))
    sgn = 1.0 if model.spinning_direction == "COUNTER_CLOCKWISE" else -1.0
    dlt = 2 * np.pi / model.n_cols
    stamps, n_pts, n_near = [], 0, 0
    files = [] if args.only_camera else raw_files(raw)
    for i, f in enumerate(files):
        sw = load_raw(f)
        if sw["t_abs"].min() < traj.t_start or sw["t_abs"].max() > traj.t_end:
            continue
        rho = np.linalg.norm(sw["xyz"], axis=1)
        k = rho >= args.min_range
        n_near += int((~k).sum())
        sw = {a: (v[k] if isinstance(v, np.ndarray) and len(v) == len(rho) else v) for a, v in sw.items()}
        az = np.arctan2(sw["xyz"][:, 1], sw["xyz"][:, 0])
        col = (np.floor(np.mod(sgn * (az - a0), 2 * np.pi) / dlt).astype(np.int64)) % model.n_cols
        row = ring_to_row[sw["ring"]]
        t_start, t_end = float(sw["t_abs"].min()), float(sw["t_abs"].max())
        xyz = deskew(sw, traj)
        T0, T1 = traj.matrices(np.array([t_start, t_end]))[0]
        st = sw["stamp_ns"]
        np.savez_compressed(
            f"{out}/lidar/vlp16_deskewed/{st}.npz", xyz=xyz.astype(np.float32), intensity=sw["intensity"],
            ret=np.zeros(len(xyz), np.int8), row=row.astype(np.uint8), col=col.astype(np.int16),
            t_rel=(sw["t_abs"] - t_start).astype(np.float32), t_start=np.float64(t_start), t_end=np.float64(t_end),
            stamp_ns=np.int64(st), T_map_lidar_start=T0, T_map_lidar_end=T1, frame=np.array("map"),
            min_range_m=np.float32(args.min_range), n_points_file=np.int64(len(rho)),
            n_near_removed=np.int64(int((~k).sum())), n_invalid_time_removed=np.int64(0))
        stamps.append(st)
        n_pts += len(xyz)
        if i % 300 == 0:
            print(f"[lidar] {i}/{len(files)}", flush=True)
    if args.only_camera:
        stamps = [int(os.path.basename(f)[:-4]) for f in os.listdir(f"{out}/lidar/vlp16_deskewed")]

    # ------------------------------------------------------------------ caméras
    chain = read_chain(f"{args.root}/calib/extrinsics/calib_chain.yaml")
    T_vlp_rgb0 = np.linalg.inv(chain["T_rgb0_vlp16"])
    T_base = {"rgb_left": T_vlp_rgb0, "rgb_right": T_vlp_rgb0 @ chain["T_rgb0_rgb1"]}
    poses, cams_cfg, img_sets = {}, {}, {}
    for cam, (cal, folder) in CAMS.items():
        K = read_intr(f"{args.root}/calib/camera_intrinsics/dalsa_{cal}.yaml")
        rows = list(csv.DictReader(open(f"{S}/frames/frames_{folder}.csv")))[::args.every]
        first = cv2.imread(rows[0]["path"], cv2.IMREAD_UNCHANGED)
        h, w = first.shape[:2]
        Km = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1.0]])
        D = np.array([K["k1"], K["k2"], K["p1"], K["p2"], 0.0])
        mx, my = cv2.initUndistortRectifyMap(Km, D, None, Km, (w, h), cv2.CV_32FC1)
        os.makedirs(f"{out}/rawimgs/{cam}", exist_ok=True)
        os.makedirs(f"{out}/cameras/{cam}", exist_ok=True)
        valid = cv2.erode(cv2.remap(np.full((h, w), 255, np.uint8), mx, my, cv2.INTER_NEAREST, borderValue=0),
                          np.ones((3, 3), np.uint8))
        cv2.imwrite(f"{out}/cameras/{cam}/valid_mask.png", valid)
        t_img = np.array([int(r["stamp_ns"]) for r in rows]) / 1e9
        inside = (t_img >= traj.t_start) & (t_img <= traj.t_end)
        Tmb, _ = traj.matrices(t_img[inside])
        pc = {}
        with open(f"{out}/cameras/{cam}/index.csv", "w", newline="") as fo:
            wcsv = csv.writer(fo)
            wcsv.writerow(["stamp_ns", "rel_path"])
            for r, T in zip([r for r, ok in zip(rows, inside) if ok], Tmb):
                st = int(r["stamp_ns"])
                dst = f"{out}/rawimgs/{cam}/{st}.jpg"
                if not os.path.exists(dst):
                    img = cv2.imread(r["path"], cv2.IMREAD_UNCHANGED)
                    cv2.imwrite(dst, cv2.remap(img, mx, my, cv2.INTER_LINEAR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                Twc = (T @ T_base[cam]).tolist()
                pc[str(st)] = {"t_start": st / 1e9, "t_end": st / 1e9, "T_map_cam_start": Twc, "T_map_cam_end": Twc}
                wcsv.writerow([st, f"rawimgs/{cam}/{st}.jpg"])
        poses[cam] = pc
        img_sets[cam] = sorted(int(s) for s in pc)
        cams_cfg[cam] = {"label": f"DALSA C1930 {cal}", "model": "PINHOLE", "width": int(w), "height": int(h),
                         "fx": K["fx"], "fy": K["fy"], "cx": K["cx"], "cy": K["cy"],
                         "valid_mask": f"cameras/{cam}/valid_mask.png",
                         "undistorted_from": {"model": "PINHOLE radtan", "k1": K["k1"], "k2": K["k2"]},
                         "rate_hz": 40.0 / args.every, "shutter": "global", "T_base_cam": T_base[cam].tolist(),
                         "index": f"cameras/{cam}/index.csv", "source": {
                             "intrinsics": f"calib/camera_intrinsics/dalsa_{cal}.yaml (image_height du fichier = 1080, "
                                           "images réelles 1200 : cy ≈ 600 confirme 1200)",
                             "extrinsics": "calib/extrinsics/calib_chain.yaml : inv(T_rgb0_vlp16) [· T_rgb0_rgb1]",
                             "images": f"imagezip pleine résolution, dossier {folder}, horodatage botanic_frames.py"}}
        print(f"[{cam}] {len(pc)} images", flush=True)
    json.dump({"description": "T_map_cam = caméra → repère de la trajectoire de référence",
               "created_utc": datetime.datetime.now(datetime.UTC).isoformat(), "spline": "traj/spline.npz",
               "global_shutter": True, "cameras": poses}, open(f"{out}/poses/cameras.json", "w"))
    sensors = {"contract_version": "0.1", "dataset": f"BotanicGarden {args.seq}", "base_frame": "base",
               "base_definition": "repère velodyne (VLP-16)", "map_frame": "repère de la GT (première pose VLP-16)",
               "cameras": cams_cfg,
               "lidars": {NAME: {"model": "Velodyne VLP-16 (16 × 1800, 10 Hz)", "rate_hz": 10.0,
                                 "T_base_lidar": np.eye(4).tolist(), "beam_model": f"lidar_model/{NAME}.json",
                                 "min_range_m": args.min_range}},
               "imu": {"model": "Xsens (/imu/data)", "R_base_imu": fit["R_sensor_imu"].tolist(),
                       "data": "imu/imu_xsens.npz"},
               "time_offsets": {"gt_s": GT_DT_S}}
    with open(f"{out}/sensors.yaml", "w") as fo:
        fo.write("# sensors.yaml — contrat de données v0.1 (splatpipe.stage0.ingest_botanic)\n")
        fo.write(_yaml(sensors))
    st_sorted = sorted(stamps)
    splits = {"contract_version": "0.1", "cameras": {}, "lidars": {NAME: {
        "unit": "stamp_ns d'en-tête du nuage (fin du tour)", "rule": "tours triés ; test_in = rang % 10 == 0",
        "n_train": len(st_sorted) - len(st_sorted[::10]), "n_test_in": len(st_sorted[::10]), "n_test_out": 0,
        "train": [s for i, s in enumerate(st_sorted) if i % 10], "test_in": st_sorted[::10], "test_out": []}}}
    for cam, ims in img_sets.items():
        splits["cameras"][cam] = {"rule": "images triées ; test_in = rang % 10 == 0", "train": [s for i, s in enumerate(ims) if i % 10],
                                  "test_in": ims[::10]}
    with open(f"{out}/splits.yaml", "w") as fo:
        fo.write(_yaml(splits))
    rep = {"n_tours": len(stamps), "points_par_tour": n_pts / max(len(stamps), 1),
           "frac_proche_retire": n_near / max(n_pts + n_near, 1), "images": {c: len(v) for c, v in img_sets.items()},
           "gt_dt_s": GT_DT_S, "imu_fit_residu_deg": fit["residu_rot_rel_deg_rms"]}
    json.dump(rep, open(f"{out}/reports/stage0_ingest.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
