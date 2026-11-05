"""Export d'une zone au format COLMAP (texte) pour un trainer 3DGS « images seules » (baseline B0 du plan).

Poses = nos poses caméra (`poses/cameras.json`, T_map_cam caméra → carte), SANS structure-from-motion : les deux
splats (B0 et le nôtre) vivent dans le même repère carte et se comparent image par image.
- Images : les 3 caméras, fenêtre temporelle autour d'une pose (`--center-pose`, `--half-window-s`), une sur
  `--every` ; liens symboliques `images/<cam>_<stamp>.jpg`.
- Caméras : celles de `sensors.yaml` — OPENCV_FISHEYE (fx fy cx cy k1–k4, Oxford) ou PINHOLE (images déjà
  redressées, DiTer++).
- Points : voxels de la carte à ≤ `--radius` m, couleurs de `map_colors_lift.npz`, sous-échantillonnés.
Convention COLMAP : images.txt contient R et t monde → caméra (R_cw = Rᵀ, t_cw = −Rᵀ·c), quaternion wxyz.

  ~/venv/bin/python -m splatpipe.train.export_colmap --root data_local/scene --out ~/b0_cour \
      --center-pose cam0:1710338243391100552
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np


def main():
    import yaml
    from scipy.spatial.transform import Rotation

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_local/scene")
    ap.add_argument("--out", required=True)
    ap.add_argument("--center-pose", required=True, help="cam:stamp")
    ap.add_argument("--half-window-s", type=float, default=16.0)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--radius", type=float, default=40.0)
    ap.add_argument("--max-points", type=int, default=1_500_000)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    sensors = yaml.safe_load(open(f"{root}/sensors.yaml"))["cameras"]
    poses = json.load(open(f"{root}/poses/cameras.json"))["cameras"]
    cam_c, st_c = args.center_pose.split(":")
    t_c = int(st_c) / 1e9
    center = np.asarray(poses[cam_c][st_c]["T_map_cam_start"], float)[:3, 3]
    os.makedirs(f"{args.out}/images", exist_ok=True)
    os.makedirs(f"{args.out}/sparse/0", exist_ok=True)

    cams = tuple(sensors)
    with open(f"{args.out}/sparse/0/cameras.txt", "w") as f:
        for i, c in enumerate(cams, 1):
            K = sensors[c]
            if K.get("model", "OPENCV_FISHEYE") == "PINHOLE":
                f.write(f"{i} PINHOLE {K['width']} {K['height']} {K['fx']} {K['fy']} {K['cx']} {K['cy']}\n")
            else:
                f.write(f"{i} OPENCV_FISHEYE {K['width']} {K['height']} {K['fx']} {K['fy']} {K['cx']} {K['cy']} "
                        f"{K['k1']} {K['k2']} {K['k3']} {K['k4']}\n")

    n_img = 0
    with open(f"{args.out}/sparse/0/images.txt", "w") as f:
        for i, c in enumerate(cams, 1):
            with open(f"{root}/cameras/{c}/index.csv") as fi:
                rel = {str(int(r["stamp_ns"])): os.path.basename(r["rel_path"]) for r in csv.DictReader(fi)}
            ks = [k for k in sorted(poses[c], key=int) if abs(int(k) / 1e9 - t_c) <= args.half_window_s]
            for k in ks[:: args.every]:
                src = f"{root}/rawimgs/{c}/{rel.get(k, '')}"
                if k not in rel or not os.path.exists(src):
                    continue
                T = np.asarray(poses[c][k]["T_map_cam_start"], float)
                R_cw = T[:3, :3].T
                t_cw = -R_cw @ T[:3, 3]
                qx, qy, qz, qw = Rotation.from_matrix(R_cw).as_quat()
                name = f"{c}_{k}.jpg"
                dst = f"{args.out}/images/{name}"
                if not os.path.lexists(dst):
                    os.symlink(src, dst)
                n_img += 1
                f.write(f"{n_img} {qw} {qx} {qy} {qz} {t_cw[0]} {t_cw[1]} {t_cw[2]} {i} {name}\n\n")

    from splatpipe.stage4.overfit_core import _ply_header

    off, n, props = _ply_header(f"{root}/map/map_5cm_static.ply")
    dt = np.dtype([(name, "<" + t) for name, t in props])
    mm = np.memmap(f"{root}/map/map_5cm_static.ply", dtype=dt, mode="r", offset=off, shape=(n,))
    xyz = np.stack([mm["x"], mm["y"], mm["z"]], 1).astype(np.float64)
    col = np.load(f"{root}/reports/stage4/map_colors_lift.npz")
    rgb, wgt = col["rgb"].astype(np.float32), col["weight"]
    sel = np.flatnonzero((((xyz - center) ** 2).sum(1) <= args.radius ** 2) & (wgt > 0))
    if len(sel) > args.max_points:
        sel = np.random.default_rng(0).choice(sel, args.max_points, replace=False)
    c8 = np.clip(rgb[sel] * 255, 0, 255).astype(np.uint8)
    with open(f"{args.out}/sparse/0/points3D.txt", "w") as f:
        for j, (p, cc) in enumerate(zip(xyz[sel], c8), 1):
            f.write(f"{j} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {cc[0]} {cc[1]} {cc[2]} 0\n")
    print(json.dumps({"images": n_img, "points": int(len(sel)), "centre": center.round(2).tolist(), "out": args.out}))


if __name__ == "__main__":
    main()
