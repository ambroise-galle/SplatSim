"""Contrôle visuel calibration caméra ↔ LiDAR d'une scène canonique (caméras PINHOLE redressées) : points LiDAR
deskewés des balayages à ± `--win` s de l'image, projetés avec T_map_cam (z-buffer par pixel), colorés par
profondeur, sur la photo. Score de bords : gradient moyen de la photo sur les sauts de profondeur du nuage projeté,
relatif au gradient moyen (> 1 = bords alignés).

  venv/bin/python -m splatpipe.stage0.scene_overlay --root data_local/botanicgarden/scene_1018_00 --n 4
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np


def main():
    import cv2
    import yaml

    from splatpipe.stage0.diterpp_overlay import edge_score
    from splatpipe.train.data import lidar_paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--win", type=float, default=0.3)
    ap.add_argument("--max-range", type=float, default=40.0)
    a = ap.parse_args()
    sensors = yaml.safe_load(open(f"{a.root}/sensors.yaml"))
    poses = json.load(open(f"{a.root}/poses/cameras.json"))["cameras"]
    _, ddir = lidar_paths(a.root)
    sw_files = sorted(glob.glob(f"{a.root}/{ddir}/*.npz"))
    sw_t = np.array([int(os.path.basename(f)[:-4]) / 1e9 for f in sw_files])
    out_dir = f"{a.root}/reports/overlay"
    os.makedirs(out_dir, exist_ok=True)
    rep = []
    for cam, cfg in sensors["cameras"].items():
        rows = list(csv.DictReader(open(f"{a.root}/{cfg['index']}")))
        for r in rows[len(rows) // (a.n + 1)::len(rows) // (a.n + 1)][:a.n]:
            st = r["stamp_ns"]
            T = np.asarray(poses[cam][st]["T_map_cam_start"], float)
            img = cv2.imread(f"{a.root}/{r['rel_path']}")
            H, W = img.shape[:2]
            sel = np.flatnonzero(np.abs(sw_t - int(st) / 1e9) <= a.win)
            P = np.concatenate([np.load(sw_files[i])["xyz"] for i in sel]).astype(np.float64)
            pc = (P - T[:3, 3]) @ T[:3, :3]
            k = (pc[:, 2] > 0.5) & (np.linalg.norm(pc, axis=1) < a.max_range)
            pc = pc[k]
            u = (cfg["fx"] * pc[:, 0] / pc[:, 2] + cfg["cx"]).astype(int)
            v = (cfg["fy"] * pc[:, 1] / pc[:, 2] + cfg["cy"]).astype(int)
            ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u, v, z = u[ok], v[ok], pc[ok, 2]
            o = np.argsort(-z)                                   # le plus proche écrit en dernier
            dep = np.full((H, W), np.inf)
            dep[v[o], u[o]] = z[o]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
            # bords sur une image de profondeur densifiée (dilatation 5 px) : le nuage VLP-16 est clairsemé
            dd = np.where(np.isfinite(dep), dep, 0).astype(np.float32)
            dd = cv2.dilate(dd, np.ones((5, 5), np.uint8))
            dd[dd == 0] = np.inf
            score = edge_score(gray, dd)
            cmap = cv2.applyColorMap(np.clip(255 * (1 - np.log(z[o]) / np.log(a.max_range)), 0, 255).astype(np.uint8)[:, None],
                                     cv2.COLORMAP_JET)[:, 0]
            ov = img.copy()
            for x, y, c in zip(u[o], v[o], cmap):
                cv2.circle(ov, (int(x), int(y)), 2, tuple(int(cc) for cc in c), -1)
            sheet = np.concatenate([img, ov], 1)
            sheet = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2))
            cv2.putText(sheet, f"{cam} {st} | score bords {score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)
            cv2.imwrite(f"{out_dir}/{cam}_{st}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
            rep.append({"cam": cam, "stamp": st, "score_bords": round(score, 3), "n_points": int(len(z))})
            print(json.dumps(rep[-1]), flush=True)
    json.dump(rep, open(f"{out_dir}/overlay.json", "w"), indent=1)


if __name__ == "__main__":
    main()
