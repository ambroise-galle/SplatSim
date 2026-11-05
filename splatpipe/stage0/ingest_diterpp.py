"""DiTer++ — mise au format canonique du pipeline d'une tranche extraite (Lawn, Go2 : Ouster OS1-64 + RealSense).

Même contrat que la scène Oxford (`sensors.yaml`, `traj/spline.npz`, `lidar/<capteur>_deskewed/<stamp>.npz`,
`poses/cameras.json`, `cameras/<cam>/index.csv`, `splits.yaml`), avec :
- repère `base` = origine des rayons de l'Ouster = os_sensor + (0, 0, h) (h estimé, cf. `diterpp_lidar.py model`) ;
  T_base_lidar = identité ; repère fixe = carte TLS (`Lawn_GT.pcd`) ;
- trajectoire = GT recalée de +48,7 ms (la pose datée au début du tour est celle du milieu du tour), mise en forme
  par le gyroscope du corps entre les poses, orientations éventuellement lissées (variante choisie par
  `diterpp_lidar.py deskew`) ;
- balayages : points ≥ `--min-range` (corps et arceau du robot en deçà de 0,8 m), deskewés au temps de leur colonne de
  tir, ligne = ring, colonne = colonne de tir ;
- caméra `front` : sténopé + plumb_bob (calibration des auteurs), T_base_cam = Tz(−h) · T_lidar_camera, temps image
  + `--cam-dt`.

  .venv/bin/python -m splatpipe.stage0.ingest_diterpp --dir data_local/diterpp/lawn \
      --out data_local/diterpp/lawn/scene_upper_day_604_784
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os

import numpy as np

from splatpipe.stage0.diterpp_lidar import (GT_DT_S, NAME, build_trajectory, deskew_points, fit_imu, gt_trajectory,
                                            load_sweep)
from splatpipe.stage0.diterpp_overlay import K_AUTEURS, T_LIDAR_CAMERA, fold_radius

CAM = "front"


def _yaml(d) -> str:
    import yaml

    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True, default_flow_style=None, width=120)


def camera_block(ci: dict, T_base_cam: np.ndarray, args) -> dict:
    K = K_AUTEURS
    blk = {"label": "front (RealSense couleur)", "width": ci["width"], "height": ci["height"],
           "fx": K["fx"], "fy": K["fy"], "cx": K["cx"], "cy": K["cy"]}
    if args.undistort:
        blk.update({"model": "PINHOLE", "valid_mask": f"cameras/{CAM}/valid_mask.png",
                    "undistorted_from": {"model": "OPENCV (plumb_bob)", "k1": K["D"][0], "k2": K["D"][1],
                                         "p1": K["D"][2], "p2": K["D"][3], "k3": K["D"][4]}})
    else:
        blk.update({"model": "OPENCV", "k1": K["D"][0], "k2": K["D"][1], "p1": K["D"][2], "p2": K["D"][3],
                    "k3": K["D"][4], "distortion_valid_radius_norm": fold_radius(K["D"])})
    blk.update({"rate_hz": 30.0, "shutter": "global (supposé)", "T_base_cam": T_base_cam.tolist(),
                "index": f"cameras/{CAM}/index.csv", "time_offset_s": args.cam_dt,
                "source": {"intrinsics": "document des auteurs DiTer++ (plumb_bob)",
                           "extrinsics": "T_lidar_camera des auteurs (lidar = os_sensor)",
                           "formula": "T_base_cam = inv(T_os_base) · T_lidar_camera"}})
    return blk


def undistort_images(src_dir: str, dst_dir: str, mask_path: str, K: dict):
    """Redresse chaque image en sténopé idéal de même K (OpenCV) ; masque = pixels couverts par l'image d'origine
    (érodé d'1 px). Distorsion faible : ~4 px aux coins."""
    import cv2

    if os.path.islink(dst_dir):
        os.unlink(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    Km = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1.0]])
    D = np.array(K["D"], float)
    files = sorted(glob.glob(f"{src_dir}/*.jpg"))
    h, w = cv2.imread(files[0]).shape[:2]
    mx, my = cv2.initUndistortRectifyMap(Km, D, None, Km, (w, h), cv2.CV_32FC1)
    ones = np.full((h, w), 255, np.uint8)
    valid = cv2.remap(ones, mx, my, cv2.INTER_NEAREST, borderValue=0)
    valid = cv2.erode(valid, np.ones((3, 3), np.uint8))
    cv2.imwrite(mask_path, valid)
    for i, f in enumerate(files):
        img = cv2.imread(f)
        cv2.imwrite(f"{dst_dir}/{os.path.basename(f)}", cv2.remap(img, mx, my, cv2.INTER_LINEAR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        if i % 1000 == 0:
            print(f"[redressement] {i}/{len(files)}", flush=True)
    print(f"[redressement] masque valide : {float((valid > 0).mean()):.4f} des pixels", flush=True)


def main():
    from splatpipe.common.lidar_model import StructuredSpinningLidarModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data_local/diterpp/lawn")
    ap.add_argument("--slice", default="upper_day_604_784")
    ap.add_argument("--gt", default="Lawn_upper_day_gt.tum")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-range", type=float, default=0.8)
    ap.add_argument("--smooth-rot", type=float, default=0.0, help="σ (s) du lissage des orientations GT ; 0 = aucun")
    ap.add_argument("--cam-dt", type=float, default=0.0, help="décalage ajouté au temps des images (s)")
    ap.add_argument("--anchors", default="", help="TUM des poses recalées sur le TLS (diterpp_lidar refine), "
                    "au temps physique du milieu des tours, à la place de la GT")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--undistort", action="store_true",
                    help="images redressées en sténopé idéal (même K) dans rawimgs/front + masque des pixels valides")
    ap.add_argument("--only-camera", action="store_true", help="ne refait que la partie caméra (et sensors.yaml)")
    args = ap.parse_args()
    src = f"{args.dir}/{args.slice}"
    out = args.out
    for sub in ("lidar/ouster_deskewed", "lidar_model", "traj", "poses", f"cameras/{CAM}", "imu", "reports"):
        os.makedirs(f"{out}/{sub}", exist_ok=True)

    # ------------------------------------------------------------------ modèle de faisceaux
    model = StructuredSpinningLidarModel.load_json(f"{src}/stage0/{NAME}.json")
    h = float(model.notes["origine"].split("(0, 0, ")[1].split(")")[0])
    model.save_json(f"{out}/lidar_model/{NAME}.json")
    for f in (f"{NAME}_estimation.json", f"{NAME}_near_range.npz"):
        os.system(f"cp '{src}/stage0/{f}' '{out}/lidar_model/'")
    T_os_base = np.eye(4)
    T_os_base[2, 3] = h                      # p_os = T_os_base · p_base

    # ------------------------------------------------------------------ trajectoire (repère base)
    gt1 = gt_trajectory(args.anchors, 0.0) if args.anchors else gt_trajectory(f"{args.dir}/{args.gt}", GT_DT_S)
    imu = np.load(f"{src}/imu_body.npz")
    fit = fit_imu(gt1, imu["t"], imu["gyro"])
    traj_os = build_trajectory(gt1, imu["t"], imu["gyro"], fit, smooth_rot_s=args.smooth_rot)
    traj = traj_os.transformed(np.eye(4), T_os_base)
    traj.body = "base"
    traj.meta.update({"body": "base = origine des rayons Ouster = os_sensor + (0, 0, h)", "h_m": h,
                      "ancres": args.anchors or f"{args.gt} (+{GT_DT_S} s)",
                      "imu_fit": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fit.items()}})
    traj.save(f"{out}/traj/spline.npz")
    np.savez_compressed(f"{out}/imu/imu_body.npz", **{k: imu[k] for k in imu.files})

    # ------------------------------------------------------------------ balayages
    files = [] if args.only_camera else sorted(glob.glob(f"{src}/lidar/*.npz"))
    if args.limit:
        files = files[:args.limit]
    stamps, stats = [], {"n_points": 0, "n_near": 0}
    for i, f in enumerate(files):
        st = int(os.path.basename(f)[:-4])
        sw = load_sweep(f)
        rho = np.linalg.norm(sw["xyz"] - T_os_base[:3, 3], axis=1)
        k = rho >= args.min_range
        n_near = int((~k).sum())
        sw = {kk: (v[k] if isinstance(v, np.ndarray) and len(v) == len(rho) else v) for kk, v in sw.items()}
        t_start, t_end = st / 1e9 + sw["t_cols"][0], st / 1e9 + sw["t_cols"][-1]
        if t_start < traj.t_start or t_end > traj.t_end:
            continue
        xyz = deskew_points(sw, st, traj_os)               # repère carte (points physiques, origine sans effet)
        T0, T1 = traj.matrices(np.array([t_start, t_end]))[0]
        np.savez_compressed(
            f"{out}/lidar/ouster_deskewed/{st}.npz", xyz=xyz.astype(np.float32), intensity=sw["intensity"],
            reflectivity=sw["reflectivity"], ret=np.zeros(len(xyz), np.int8), row=sw["row"].astype(np.uint8),
            col=sw["col"].astype(np.int16), t_rel=(sw["t_rel"] - sw["t_cols"][0]).astype(np.float32),
            t_start=np.float64(t_start), t_end=np.float64(t_end), stamp_ns=np.int64(st), T_map_lidar_start=T0,
            T_map_lidar_end=T1, frame=np.array("tls"), min_range_m=np.float32(args.min_range),
            n_points_file=np.int64(sw["n_file"]), n_near_removed=np.int64(n_near), n_invalid_time_removed=np.int64(0))
        stamps.append(st)
        stats["n_points"] += len(xyz)
        stats["n_near"] += n_near
        if i % 200 == 0:
            print(f"[lidar] {i}/{len(files)}", flush=True)

    if args.only_camera:                         # balayages déjà écrits : on garde leur liste pour splits.yaml
        stamps = [int(os.path.basename(f)[:-4]) for f in glob.glob(f"{out}/lidar/ouster_deskewed/*.npz")]

    # ------------------------------------------------------------------ caméra
    imgs = sorted(int(os.path.basename(p)[:-4]) for p in glob.glob(f"{src}/cam/*.jpg"))
    raw_dir = f"{out}/rawimgs/{CAM}"
    if args.undistort:
        undistort_images(f"{src}/cam", raw_dir, f"{out}/cameras/{CAM}/valid_mask.png", K_AUTEURS)
    elif not os.path.lexists(raw_dir):
        os.makedirs(os.path.dirname(raw_dir), exist_ok=True)
        os.symlink(os.path.relpath(f"{src}/cam", os.path.dirname(raw_dir)), raw_dir)
    T_base_cam = np.linalg.inv(T_os_base) @ T_LIDAR_CAMERA
    t_img = np.array(imgs) / 1e9 + args.cam_dt
    inside = (t_img >= traj.t_start) & (t_img <= traj.t_end)
    Tmb, _ = traj.matrices(t_img[inside])
    poses = {}
    with open(f"{out}/cameras/{CAM}/index.csv", "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["stamp_ns", "rel_path"])
        for st, T in zip(np.array(imgs)[inside], Tmb):
            Twc = (T @ T_base_cam).tolist()
            ts = int(st) / 1e9 + args.cam_dt
            poses[str(int(st))] = {"t_start": ts, "t_end": ts, "T_map_cam_start": Twc, "T_map_cam_end": Twc}
            w.writerow([int(st), f"rawimgs/{CAM}/{int(st)}.jpg"])
    json.dump({"description": "T_map_cam = caméra → carte TLS ; obturateur supposé global (non vérifié)",
               "created_utc": datetime.datetime.now(datetime.UTC).isoformat(), "spline": "traj/spline.npz",
               "global_shutter": True, "cam_dt_s": args.cam_dt, "cameras": {CAM: poses}},
              open(f"{out}/poses/cameras.json", "w"))

    # ------------------------------------------------------------------ sensors.yaml / splits.yaml
    ci = json.load(open(f"{src}/camera_info.json"))
    K = K_AUTEURS
    sensors = {
        "contract_version": "0.1",
        "dataset": f"DiTer++ Lawn upper_day, tranche {args.slice}",
        "base_frame": "base",
        "base_definition": f"origine des rayons Ouster = os_sensor + (0, 0, {h:.4f}) m (axes de os_sensor)",
        "map_frame": "tls (Lawn_GT.pcd, Leica RTC360)",
        "cameras": {CAM: camera_block(ci, T_base_cam, args)},
        "lidars": {NAME: {
            "model": "Ouster OS1-64 (64 × 1024, 10 Hz)", "rate_hz": 10.0, "T_base_lidar": np.eye(4).tolist(),
            "beam_model": f"lidar_model/{NAME}.json", "min_range_m": args.min_range,
            "source": "nuages organisés du pilote ouster-ros (os_sensor) ; modèle estimé (diterpp_lidar.py model)"}},
        "imu": {"model": "IMU du corps Go2 (/agent1/imu/data, 500 Hz)", "frame": "imu",
                "R_base_imu": fit["R_sensor_imu"].tolist(), "time_offset_s": fit["dt_s"], "data": "imu/imu_body.npz",
                "source": "rotation et décalage estimés contre les rotations relatives de la GT (fit_imu)"},
        "time_offsets": {"gt_s": GT_DT_S, "camera_s": args.cam_dt, "imu_s": fit["dt_s"]},
    }
    with open(f"{out}/sensors.yaml", "w") as f:
        f.write("# sensors.yaml — contrat de données v0.1 (splatpipe.stage0.ingest_diterpp)\n"
                "# T_a_b : p_a = T_a_b · p_b (4×4, lignes). Distances en m. Caméra : repère optique OpenCV.\n")
        f.write(_yaml(sensors))
    st_sorted = sorted(stamps)
    img_sorted = [int(s) for s in np.array(imgs)[inside]]
    splits = {"contract_version": "0.1",
              "cameras": {"unit": "stamp_ns de l'image front", "rule": "images triées ; test_in = rang % 10 == 0",
                          "n_train": len(img_sorted) - len(img_sorted[::10]), "n_test_in": len(img_sorted[::10]),
                          "n_test_out": 0, "train": [s for i, s in enumerate(img_sorted) if i % 10],
                          "test_in": img_sorted[::10], "test_out": []},
              "lidars": {NAME: {"unit": "stamp_ns d'en-tête du nuage (début du tour)",
                                "rule": "tours triés ; test_in = rang % 10 == 0",
                                "n_train": len(st_sorted) - len(st_sorted[::10]), "n_test_in": len(st_sorted[::10]),
                                "n_test_out": 0, "train": [s for i, s in enumerate(st_sorted) if i % 10],
                                "test_in": st_sorted[::10], "test_out": []}}}
    with open(f"{out}/splits.yaml", "w") as f:
        f.write(_yaml(splits))
    rep = {"n_sweeps": len(stamps), "n_images": len(img_sorted), "points_par_tour_median":
           stats["n_points"] / max(len(stamps), 1), "frac_proche_retire": stats["n_near"] / max(stats["n_points"] + stats["n_near"], 1),
           "h_m": h, "imu_fit": sensors["imu"], "smooth_rot_s": args.smooth_rot, "cam_dt_s": args.cam_dt,
           "ancres": args.anchors or "GT"}
    json.dump(rep, open(f"{out}/reports/stage0_ingest.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
