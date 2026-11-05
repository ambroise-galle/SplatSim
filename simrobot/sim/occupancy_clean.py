"""Carte d'occupation LiDAR (`lidar_occupation.npz`) : retrait des traînées des personnes qui accompagnaient le robot
(desktop). Ce sont des voxels de 0,15 à 2 m au-dessus du sol, à moins de 3 m du trajet, sans aucune gaussienne visible
(opacité > 0,2, hors ciel) à moins de 15 cm : présents dans les balayages, absents du splat, car ils bougeaient. Dans le
LiDAR simulé, ils formaient des rideaux de points le long du sentier. On met leur μ (et leurs impacts) à zéro ;
`OccupancyMap` écarte alors ces voxels au chargement.

  cd ~/dino && PYTHONPATH=. venv/bin/python simrobot/sim/occupancy_clean.py --scene out/simrobot/scenes/HOB-1_full \
      --labels out/simrobot/labels/HOB-1_full/gaussiennes.npz --out lidar_occupation_nettoyee.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--occupation", default="lidar_occupation.npz")
    ap.add_argument("--out", required=True, help="nom du fichier nettoyé (dans la scène)")
    ap.add_argument("--band", default="0.15,2.0", help="tranche de hauteur au-dessus du sol (m)")
    ap.add_argument("--near", type=float, default=3.0, help="distance maximale au trajet (m)")
    ap.add_argument("--radius", type=float, default=0.15, help="rayon de recherche d'une gaussienne visible (m)")
    a = ap.parse_args()
    from scipy.spatial import cKDTree

    from terrain_labels import load_splat

    t0 = time.time()
    meta = json.load(open(f"{a.scene}/scene.json"))
    P, op = load_splat(a.scene, meta)
    lab = np.load(a.labels)["label"]
    vis = (op > 0.2) & (lab != 2)
    kd = cKDTree(P[vis])
    del P
    O = dict(np.load(f"{a.scene}/{a.occupation}"))
    res, ny, below = float(O["res"]), int(O["ny"]), float(O["below"])
    lo, cs = O["lo"], O["col_start"]
    col = np.repeat(np.arange(len(cs) - 1), np.diff(cs))
    vz = O["vz"].astype(np.int64)
    h = (vz + 0.5) * res - below
    b0, b1 = map(float, a.band.split(","))
    mu = O["mu"].astype(np.float32)
    cand = np.where((h > b0) & (h < b1) & (mu > 0))[0]
    x = lo[0] + (col[cand] // ny + 0.5) * res
    y = lo[1] + (col[cand] % ny + 0.5) * res
    path = np.array(json.load(open(f"{a.scene}/trajets.json"))["dlio"]["p"])[:, :2]
    near = cKDTree(path).query(np.stack([x, y], 1))[0] < a.near
    cand, x, y = cand[near], x[near], y[near]
    z = O["zb"].ravel()[col[cand]] + (vz[cand] + 0.5) * res
    d, _ = kd.query(np.stack([x, y, z], 1), distance_upper_bound=a.radius)
    ghost = cand[~np.isfinite(d)]
    O["mu"] = O["mu"].copy()
    O["mu"][ghost] = 0
    if "hits" in O:
        O["hits"] = O["hits"].copy()
        O["hits"][ghost] = 0
    stats = {"voxels": int(len(mu)), "candidats_tranche_pres_du_trajet": int(len(cand)), "retires": int(len(ghost)),
             "part_des_candidats": round(len(ghost) / max(len(cand), 1), 3),
             "duree_s": round(time.time() - t0)}
    O["fantomes"] = json.dumps(stats)
    np.savez(f"{a.scene}/{a.out}", **O)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
