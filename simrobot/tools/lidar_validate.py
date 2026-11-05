"""Validation du LiDAR simulé sur de VRAIS rayons : balayages Hesai mis de côté à la construction de la carte
d'occupation (`--holdout`), rejoués dans la simulation (même origine, même direction) ; on compare la distance simulée
à la distance mesurée.

- « juste »     : |simulée − mesurée| < `tol` ;
- « trop tôt »  : la simulation s'arrête avant (elle bloque ce que le vrai rayon a traversé : végétation trop opaque) ;
- « trop loin » : la simulation passe au travers (ou ne revient pas) alors que le vrai rayon s'est arrêté.

  PYTHONPATH=. venv/bin/python simrobot/tools/lidar_validate.py --scene out/simrobot/scenes/HOB-1_full \
      --scans data_local/simrobot/grandtour/HOB-1/lidar/dlio_hesai_points_undistorted.tar \
      --odometry data_local/simrobot/grandtour/HOB-1/lidar/dlio_map_odometry.tar
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
sys.path.insert(0, os.path.dirname(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--scans", required=True)
    ap.add_argument("--odometry", required=True)
    ap.add_argument("--n", type=int, default=60, help="nombre de balayages mis de côté à rejouer")
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--max-range", type=float, default=50.0)
    ap.add_argument("--priors", default="", help="longueurs a priori à essayer, ex. 0.025,0.1,0.3,1")
    ap.add_argument("--floors", default="0", help="décalages du plancher à essayer (m), ex. 0,-0.05,-0.1")
    ap.add_argument("--no-old", action="store_true", help="sans l'ancienne carte")
    ap.add_argument("--extra", default="", help="autres cartes d'occupation à comparer (fichiers de la scène, virgules)")
    ap.add_argument("--sol-corrige", action="store_true",
                    help="plancher = sol corrigé par les étiquettes (terrain_etiquettes.npz : dH) pour les cartes --extra")
    a = ap.parse_args()
    from gaussgym_export import FLIP
    from lidar_occupancy import TarZarr
    from lidar_sim import OccupancyMap, VoxelMap
    from lidar_terrain import quat_xyzw_to_R, zarr_group
    from terrain import height_at

    meta = json.load(open(f"{a.scene}/scene.json"))
    T = dict(np.load(f"{a.scene}/terrain.npz"))
    centre = np.array(meta["centre_monde_m"])
    Rr, tr = np.array(meta["recalage_lidar"]["R"]), np.array(meta["recalage_lidar"]["t"])
    odo = zarr_group(a.odometry, "dlio_map_odometry")
    po, qo = odo("pose_pos").astype(float), odo("pose_orien").astype(float)
    Ro = quat_xyzw_to_R(qo / np.linalg.norm(qo, axis=1, keepdims=True))
    occ = np.load(f"{a.scene}/lidar_occupation.npz")
    held = occ["held_scans"]
    held = held[np.linspace(0, len(held) - 1, min(a.n, len(held))).astype(int)]
    maps = {} if a.no_old else {"ancienne carte 10 cm": VoxelMap(f"{a.scene}/lidar_voxels.npz", T)}
    for pr in ([float(x) for x in a.priors.split(",")] if a.priors else [None]):
        for fo in [float(x) for x in a.floors.split(",")]:
            maps[f"occupation prior={pr} plancher={fo:+.2f}"] = OccupancyMap(f"{a.scene}/lidar_occupation.npz", T,
                                                                              prior=pr, floor_offset=fo)
    Tc = dict(T)
    if a.sol_corrige:
        Tc["H"] = (T["H"] + np.load(f"{a.scene}/terrain_etiquettes.npz")["dH"]).astype(np.float32)
    for f in [x for x in a.extra.split(",") if x]:
        maps[f"{f} plancher=-0.05" + (" sol corrigé" if a.sol_corrige else "")] = OccupancyMap(
            f"{a.scene}/{f}", Tc, floor_offset=-0.05)
    Z = TarZarr(a.scans, "dlio_hesai_points_undistorted")
    res = {k: {"d": [], "h": [], "r": [], "rs": [], "el": [], "az": []} for k in maps}
    self_d = []                                                   # simulation contre elle-même (autre tirage)
    for n, k in enumerate(held):
        pts = Z.row("points", k).reshape(-1, 3).astype(np.float64)
        nv = int(np.asarray(Z.row("valid", k)).reshape(-1)[0])
        pts = pts[:nv]
        pts = pts[np.isfinite(pts).all(1) & (np.abs(pts).sum(1) > 0)]
        to_phys = lambda x: (x @ Rr.T + tr) @ FLIP.T - centre     # noqa: E731
        P = to_phys(pts @ Ro[k].T + po[k])
        o = to_phys(po[k][None])[0]
        v = P - o
        r = np.linalg.norm(v, axis=1)
        keep = (r > 0.5) & (r < a.max_range)
        P, v, r = P[keep], v[keep], r[keep]
        D = v / r[:, None]
        h = P[:, 2] - height_at(T, P[:, :2])
        for name, M in maps.items():
            rs, _ = M.cast(o, D, max_range=a.max_range + 5, noise=0.0, seed=int(k))
            res[name]["d"].append(np.where(np.isfinite(rs), rs - r, np.inf))
            res[name]["h"].append(h)
            res[name]["r"].append(r)
            res[name]["rs"].append(np.where(np.isfinite(rs), rs, a.max_range + 5))
            res[name]["el"].append(np.degrees(np.arcsin(np.clip(D[:, 2], -1, 1))))
            res[name]["az"].append(np.degrees(np.arctan2(D[:, 1], D[:, 0])) + 1000 * n)   # secteurs par balayage
            if name == list(maps)[-1]:
                r2, _ = M.cast(o, D, max_range=a.max_range + 5, noise=0.0, seed=int(k) + 7919)
                self_d.append(np.where(np.isfinite(r2) & np.isfinite(rs), r2 - rs, np.inf))
    out = {"balayages": int(len(held)), "rayons": int(sum(len(x) for x in res[next(iter(maps))]["d"]))}
    for name in maps:
        d, h, r = (np.concatenate(res[name][k]) for k in ("d", "h", "r"))
        def frac(m):
            return {"juste": round(float((np.abs(d[m]) < a.tol).mean()), 3),
                    "trop_tot": round(float((d[m] < -a.tol).mean()), 3),
                    "trop_loin": round(float((d[m] > a.tol).mean()), 3), "n": int(m.sum())}
        out[name] = {"tous": frac(np.ones(len(d), bool)),
                     "impact au sol (<0,3 m)": frac(h < 0.3),
                     "impact 0,3–2 m (sous-bois, troncs)": frac((h >= 0.3) & (h < 2)),
                     "impact > 2 m": frac(h >= 2),
                     "distance < 10 m": frac(r < 10), "distance ≥ 10 m": frac(r >= 10)}
    sd = np.concatenate(self_d)
    out["plafond (simulation contre elle-même)"] = {"juste": round(float((np.abs(sd) < a.tol).mean()), 3)}
    # distributions : quantiles des distances, et médiane par secteur (2° d'élévation × 10° d'azimut, par balayage)
    for name in maps:
        r, rs = np.concatenate(res[name]["r"]), np.concatenate(res[name]["rs"])
        el, az = np.concatenate(res[name]["el"]), np.concatenate(res[name]["az"])
        key = np.floor(el / 2).astype(np.int64) * 1000000 + np.floor(az / 10).astype(np.int64)
        uk, inv = np.unique(key, return_inverse=True)
        o_ = np.argsort(inv, kind="stable")
        starts = np.searchsorted(inv[o_], np.arange(len(uk)))
        med_r = np.array([np.median(x) for x in np.split(r[o_], starts[1:])])
        med_s = np.array([np.median(x) for x in np.split(rs[o_], starts[1:])])
        rel = (med_s - med_r) / med_r
        out[name]["distribution"] = {
            "quantiles_reels_m": np.percentile(r, [10, 25, 50, 75, 90]).round(2).tolist(),
            "quantiles_simules_m": np.percentile(rs, [10, 25, 50, 75, 90]).round(2).tolist(),
            "secteurs": int(len(uk)),
            "mediane_secteur_ecart_relatif_median": round(float(np.median(rel)), 3),
            "mediane_secteur_a_20pc_pres": round(float((np.abs(rel) < 0.2).mean()), 3)}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
