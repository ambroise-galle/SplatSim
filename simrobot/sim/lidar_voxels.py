"""Carte de voxels pour le LiDAR simulé (`lidar_sim.py`), tirée de la carte LiDAR DLIO de la mission.

Voxels de `res` m alignés sur la grille du terrain (repère physique). Chaque voxel occupé garde :
- le centre de ses points (décalage quantifié dans le voxel) et la normale du plan local : le rayon est intersecté
  avec ce petit plan (surfel), pas avec la face du cube, d'où une distance précise sans marches de 10 cm ;
- l'intensité moyenne mesurée (Hesai, normalisée sur 0–255) ;
- une probabilité de retour : surface plane 0,95, autre voxel d'au moins 3 points 0,85, voxel épars (1–2 points,
  feuillage lointain, bruit) 0,35. Un rayon peut traverser le feuillage clairsemé, comme un vrai LiDAR.
Stockage en colonnes (CSR) : `col_start[col] … col_start[col+1]` = voxels de la colonne, triés par hauteur `vz`.

Nettoyage : les traînées des personnes qui accompagnaient le robot (0,15–2 m au-dessus du sol, à moins de 3 m du
trajet, absentes du splat) sont retirées, comme pour les obstacles (stems.py), ainsi que le bruit sous le sol.

  PYTHONPATH=. venv/bin/python simrobot/sim/lidar_voxels.py --scene out/simrobot/scenes/HOB-1_full \
      --lidar-dir data_local/simrobot/grandtour/HOB-1/lidar --splat out/simrobot/scenes/HOB-1_full/merged.ply
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.dirname(__file__))
from gaussgym_export import FLIP  # noqa: E402
from terrain import height_at  # noqa: E402


def read_ply_xyzi(path):
    with open(path, "rb") as f:
        props, n = [], 0
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                props.append((line.split()[-1], {"float": "<f4", "double": "<f8", "uchar": "u1"}[line.split()[1]]))
            elif line == "end_header":
                break
        V = np.fromfile(f, dtype=np.dtype(props), count=n)
    I = V["intensity"] if "intensity" in V.dtype.names else np.zeros(n, np.float32)
    return np.stack([V["x"], V["y"], V["z"]], 1).astype(np.float64), I.astype(np.float32)


def ghost_mask(P, h, terrain, path_xy, Q, hq, op, sc, band=(0.15, 2.0), near=3.0, splat_max=300.0, cover_min=100.0):
    """Points des traînées de personnes : dans la tranche `band`, à moins de `near` m du trajet, là où le splat est
    reconstruit (sol ≥ cover_min/m²) mais vide dans la tranche (< splat_max/m², voisinage 3×3 de 10 cm)."""
    from scipy.ndimage import uniform_filter
    from scipy.spatial import cKDTree

    lo, res = np.asarray(terrain["lo"], float), float(terrain["res"])
    nx, ny = terrain["H"].shape

    def grid_density(xy, sel):
        ij = np.floor((xy[sel] - lo) / res).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
        g = np.bincount(ij[ok, 0] * ny + ij[ok, 1], minlength=nx * ny).reshape(nx, ny).astype(np.float32)
        return uniform_filter(g, size=3) / (res * res)

    dens = grid_density(Q[:, :2], (hq > band[0]) & (hq < band[1]) & (op > 0.2) & (sc < 0.3))
    cover = uniform_filter(grid_density(Q[:, :2], (hq > -0.3) & (hq < 0.3) & (op > 0.2)), size=15)
    ij = np.clip(np.floor((P[:, :2] - lo) / res).astype(np.int64), 0, [nx - 1, ny - 1])
    cand = (h > band[0]) & (h < band[1])
    cand[cand] &= (dens[ij[cand, 0], ij[cand, 1]] < splat_max) & (cover[ij[cand, 0], ij[cand, 1]] >= cover_min)
    if cand.any():
        cand[cand] &= cKDTree(path_xy).query(P[cand, :2])[0] < near
    return cand


def build(P, I, terrain, res=None):
    """Points (repère physique) + intensités → dict de la carte de voxels (cf. docstring du module)."""
    lo, H = np.asarray(terrain["lo"], float), terrain["H"]
    res = float(terrain["res"]) if res is None else res
    nx, ny = int(round(H.shape[0] * float(terrain["res"]) / res)), int(round(H.shape[1] * float(terrain["res"]) / res))
    z0 = float(np.floor(P[:, 2].min() / res) * res)
    ix = np.floor((P[:, 0] - lo[0]) / res).astype(np.int64)
    iy = np.floor((P[:, 1] - lo[1]) / res).astype(np.int64)
    iz = np.floor((P[:, 2] - z0) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy, iz, P, I = ix[ok], iy[ok], iz[ok], P[ok], I[ok]
    nz = int(iz.max()) + 1
    key = (ix * ny + iy) * nz + iz
    uk, inv, n = np.unique(key, return_inverse=True, return_counts=True)
    nv = len(uk)
    c = np.stack([np.bincount(inv, P[:, k], nv) for k in range(3)], 1) / n[:, None]
    d = P - c[inv]
    cov = np.empty((nv, 3, 3), np.float32)
    for a in range(3):
        for b in range(a, 3):
            cov[:, a, b] = cov[:, b, a] = np.bincount(inv, d[:, a] * d[:, b], nv) / n
    w, v = np.linalg.eigh(cov)                                  # valeurs propres croissantes
    w = np.maximum(w, 1e-10)
    normal = v[:, :, 0]
    # la carte accumulée a ~3 cm d'épaisseur : à 10 cm, λ3/λ2 ~ 0,1–0,3 sur une vraie surface (seuil 0,08 : 12 % de
    # voxels « plans » seulement, et des troncs traversés une fois sur quatre avec p = 0,5)
    planar = (n >= 3) & (w[:, 0] < 0.3 * w[:, 1]) & (w[:, 1] > 0.1 * w[:, 2])
    linear = (n >= 3) & ~planar & (w[:, 1] < 0.1 * w[:, 2])
    phit = np.where(n <= 2, 0.35, np.where(planar, 0.95, 0.85))
    normal[~planar] = 0
    imean = np.bincount(inv, I, nv) / n
    i99 = float(np.percentile(I, 99)) or 1.0
    col = (uk // nz).astype(np.int64)
    vz = (uk % nz).astype(np.uint16)
    vcx, vcy = col // ny, col % ny
    centre_vox = np.stack([lo[0] + (vcx + 0.5) * res, lo[1] + (vcy + 0.5) * res, z0 + (vz + 0.5) * res], 1)
    cen_q = np.clip(np.round((c - centre_vox) / (res / 2) * 127), -127, 127).astype(np.int8)
    col_start = np.zeros(nx * ny + 1, np.int64)
    np.add.at(col_start, col + 1, 1)
    col_start = np.cumsum(col_start).astype(np.int64)
    return {"lo": lo, "res": np.float64(res), "nx": np.int64(nx), "ny": np.int64(ny), "z0": np.float64(z0),
            "col_start": col_start, "vz": vz, "cen": cen_q,
            "nrm": np.clip(np.round(normal * 127), -127, 127).astype(np.int8),
            "inten": np.clip(imean / i99 * 255, 0, 255).astype(np.uint8),
            "phit": np.clip(np.round(phit * 255), 0, 255).astype(np.uint8),
            "stats": json.dumps({"voxels": int(nv), "points": int(len(P)), "plans": round(float(planar.mean()), 3),
                                 "lineaires": round(float(linear.mean()), 3), "epars": round(float((n <= 2).mean()), 3),
                                 "intensite_p99": i99})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--lidar-dir", required=True)
    ap.add_argument("--splat", default=None, help="PLY du splat (repère du PLY de la scène) : retrait des traînées")
    ap.add_argument("--res", type=float, default=None)
    a = ap.parse_args()
    t0 = time.time()
    meta = json.load(open(f"{a.scene}/scene.json"))
    T = dict(np.load(f"{a.scene}/terrain.npz"))
    centre = np.array(meta["centre_monde_m"])
    R, t = np.array(meta["recalage_lidar"]["R"]), np.array(meta["recalage_lidar"]["t"])
    P, I = read_ply_xyzi(glob.glob(f"{a.lidar_dir}/*_dlio.ply")[0])
    P = (P @ R.T + t) @ FLIP.T - centre
    h = P[:, 2] - height_at(T, P[:, :2])
    keep = h > -0.3                                              # bruit sous le sol
    stats = {"points": int(len(P)), "sous_le_sol": int((~keep).sum())}
    if a.splat:
        from stems import load_splat_points

        Q, op, sc = load_splat_points(a.splat, np.array(meta["ply_to_physics"]))
        hq = Q[:, 2] - height_at(T, Q[:, :2].astype(np.float64))
        path = np.array(json.load(open(f"{a.scene}/trajets.json"))["dlio"]["p"])[:, :2]
        g = ghost_mask(P, h, T, path, Q, hq, op, sc)
        stats["trainees_retirees"] = int((g & keep).sum())
        keep &= ~g
        del Q, op, sc, hq
    V = build(P[keep], I[keep], T, a.res)
    np.savez(f"{a.scene}/lidar_voxels.npz", **V)
    stats.update(json.loads(str(V["stats"])))
    stats["taille_Mo"] = round(os.path.getsize(f"{a.scene}/lidar_voxels.npz") / 1e6, 1)
    stats["duree_s"] = round(time.time() - t0, 1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
