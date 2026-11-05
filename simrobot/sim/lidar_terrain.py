"""Terrain de collision tiré de la CARTE LiDAR GrandTour (DLIO), dans le repère monde de la simulation.

Repères :
- mission = repère des poses GaussGym (`transforms.json`), axe z vers le BAS ;
- `dlio_map` = repère de la carte LiDAR et de l'odométrie DLIO, aligné sur la gravité, z haut ;
- monde = FLIP · mission (180° autour de x), z haut ; à quelques mm / 0,02° près, c'est `dlio_map`.

1. Recalage carte → mission : (R, t) et bras de levier l (base DLIO → caméra avant), par moindres carrés alternés
   sur les positions appariées par horodatage : c_i = R (p_i + R_i l) + t. Résidu attendu : quelques mm.
2. Sol : par cellule, bas de colonne (quantile 10 %) puis médiane de la COUCHE de sol (−5 / +35 cm autour) ; les
   pics isolés (> 25 cm de la médiane locale 7×7) sont remplacés. (Pièges : la médiane de toute la colonne inclut
   troncs et canopée ; le quantile bas seul est trop bas de ~10 cm, l'épaisseur de la carte accumulée.)
3. Obstacles : cellules avec assez de points LiDAR entre `obs_zmin` et `obs_zmax` au-dessus du sol (troncs, souches,
   rochers). L'herbe et le lierre restent sous `obs_zmin`.
4. Trous : diffusion (inpainting), pas de plateaux au plus proche.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import tarfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from gaussgym_export import FLIP  # noqa: E402


def zarr_group(tar_path, group):
    import numcodecs

    tf = tarfile.open(tar_path)
    names = tf.getnames()

    def arr(key):
        za = json.load(tf.extractfile(f"{group}/{key}/.zarray"))
        chunks = sorted(n for n in names if n.startswith(f"{group}/{key}/") and not n.split("/")[-1].startswith("."))
        raw = b"".join(tf.extractfile(n).read() for n in chunks)
        c = numcodecs.get_codec(za["compressor"]) if za["compressor"] else None
        a = np.frombuffer(c.decode(raw) if c else raw, dtype=np.dtype(za["dtype"]))
        return a.reshape(-1, *za["shape"][1:])[:za["shape"][0]]

    return arr


def quat_xyzw_to_R(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def umeyama(A, B):
    """R, t minimisant |R A + t − B|² (sans échelle)."""
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((B - mb).T @ (A - ma))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, mb - R @ ma


def read_xyz_ply(path):
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
    return np.stack([V["x"], V["y"], V["z"]], 1).astype(np.float64)


def register(tiles, lidar_dir):
    """(R, t) : dlio_map → mission, bras de levier, résidus."""
    odo = zarr_group(glob.glob(f"{lidar_dir}/dlio_map_odometry.tar")[0], "dlio_map_odometry")
    to, po, qo = odo("timestamp").astype(float), odo("pose_pos").astype(float), odo("pose_orien").astype(float)
    tc, pc = [], []
    for t in tiles:
        for f in json.load(open(f"{t}/transforms.json"))["frames"]:
            if "hdr_front" in f["file_path"]:
                s, ns = str(f["timestamp"]).split("_")
                tc.append(int(s) + int(ns) * 1e-9)
                pc.append(np.array(f["transform_matrix"], float)[:3, 3])
    tc, pc = np.array(tc), np.array(pc)
    ok = (tc > to[0]) & (tc < to[-1])
    tc, pc = tc[ok], pc[ok]
    p = np.stack([np.interp(tc, to, po[:, k]) for k in range(3)], 1)
    qi = np.stack([np.interp(tc, to, qo[:, k]) for k in range(4)], 1)
    Ri = quat_xyzw_to_R(qi / np.linalg.norm(qi, axis=1, keepdims=True))
    lever = np.zeros(3)
    for _ in range(20):
        R, t = umeyama(p + Ri @ lever, pc)
        resid = (pc - t) @ R - p                              # = R_i l
        lever = np.linalg.lstsq(Ri.reshape(-1, 3), resid.reshape(-1), rcond=None)[0]
    err = np.linalg.norm((p + Ri @ lever) @ R.T + t - pc, axis=1)
    ang = np.degrees(np.arccos(np.clip((np.trace(FLIP @ R) - 1) / 2, -1, 1)))
    return R, t, {"n_paires": int(len(tc)), "bras_levier_m": lever.round(3).tolist(),
                  "residu_m": {"median": round(float(np.median(err)), 4), "p90": round(float(np.percentile(err, 90)), 4)},
                  "ecart_a_FLIP_deg": round(float(ang), 3), "R": R.round(6).tolist(), "t": t.round(4).tolist()}


def lidar_points_world(lidar_dir, R, t):
    """Carte DLIO → repère monde (FLIP · mission)."""
    return (read_xyz_ply(glob.glob(f"{lidar_dir}/*_dlio.ply")[0]) @ R.T + t) @ FLIP.T


def ground_and_obstacles(P, lo, res, nx, ny, q_sol=0.5, obs_zmin=0.3, obs_zmax=1.3, obs_min_points=8,
                         obs_min_frac=0.2, obs_min_slices=8, obs_height=1.0, free_xy=None, free_radius=0.35,
                         trunk_margin=0.30, method="troncs", splat=None):
    """P : points LiDAR (repère de la grille, z haut). → dict terrain (H avec piliers d'obstacle, sol, couvert, obstacle).

    `method="troncs"` (défaut) : obstacles de stems.build_obstacles (troncs par cercles et continuité verticale,
    obstacles bas vus par le LiDAR ET le splat `splat` = (Q, opacité, échelle)). `method="cellules"` : ancienne méthode
    ci-dessous, gardée pour comparaison — elle manquait la plupart des troncs et gardait des murs invisibles (traînées
    des personnes accumulées dans la carte).

    Obstacle = cellule dont au moins `obs_min_frac` des points sont dans la bande [obs_zmin, obs_zmax] au-dessus du sol,
    répartis sur au moins `obs_min_slices` tranches de 10 cm (colonne : tronc, souche, rocher). Un seuil en nombre de
    points seul ne suffit pas (HÖB-1 : 10 % des cellules du chemin parcouru dépassent 14 points dans la bande —
    personnes, branches basses, bruit de carte). Les cellules à moins de `free_radius` du trajet réel `free_xy` sont
    forcées libres (le robot y est passé)."""
    import cv2

    ij = np.floor((P[:, :2] - lo) / res).astype(int)
    inb = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    P, ij = P[inb], ij[inb]
    cell = ij[:, 0] * ny + ij[:, 1]
    order = np.lexsort((P[:, 2], cell))
    cs, zs = cell[order], P[order, 2]
    cnt = np.bincount(cs, minlength=nx * ny)
    first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    has = cnt >= 3
    # bas de colonne (quantile 10 %) puis médiane de la seule COUCHE de sol [z10 − 5 cm, z10 + 35 cm] : la médiane
    # de toute la colonne inclut troncs et canopée (marches de > 1 m entre cellules voisines sur HÖB-1)
    z10 = np.full(nx * ny, np.nan)
    z10[has] = zs[first[has] + np.floor(0.1 * (cnt[has] - 1)).astype(int)]
    rel = P[:, 2] - z10[cell]
    lay = (rel > -0.05) & (rel < 0.35)
    zl = np.full(nx * ny, np.nan)
    o2 = np.lexsort((P[lay, 2], cell[lay]))
    c2, z2 = cell[lay][o2], P[lay, 2][o2]
    n2 = np.bincount(c2, minlength=nx * ny)
    f2 = np.concatenate([[0], np.cumsum(n2)[:-1]])
    h2 = n2 >= 3
    zl[h2] = z2[f2[h2] + np.floor(q_sol * (n2[h2] - 1)).astype(int)]
    low = np.where(h2, zl, z10).reshape(nx, ny)
    valide = has.reshape(nx, ny)
    # pics isolés (bruit sous le sol, branche basse) : remplacés par la médiane locale 7×7
    from scipy.ndimage import median_filter

    fill = np.where(valide, low, np.nanmedian(low))
    med = median_filter(fill, size=7)
    spike = valide & (np.abs(low - med) > 0.25)
    low = np.where(spike, med, low)
    hgt = P[:, 2] - np.nan_to_num(low, nan=1e6).ravel()[cell]
    band = (hgt > obs_zmin) & (hgt < obs_zmax)
    n_band = np.bincount(cell[band], minlength=nx * ny).reshape(nx, ny)
    n_sl = int(round((obs_zmax - obs_zmin) / 0.1))
    sl = np.zeros((nx * ny, n_sl), bool)
    sl[cell[band], np.clip(((hgt[band] - obs_zmin) / 0.1).astype(int), 0, n_sl - 1)] = True
    slices = sl.sum(1).reshape(nx, ny)
    frac = n_band / np.maximum(cnt.reshape(nx, ny), 1)
    obstacle = valide & (n_band >= obs_min_points) & (frac >= obs_min_frac) & (slices >= obs_min_slices)
    if free_xy is not None:
        from scipy.spatial import cKDTree

        xx, yy = np.meshgrid(lo[0] + res * (np.arange(nx) + 0.5), lo[1] + res * (np.arange(ny) + 0.5), indexing="ij")
        dpath, _ = cKDTree(free_xy[:, :2]).query(np.stack([xx.ravel(), yy.ravel()], 1))
        obstacle &= dpath.reshape(nx, ny) >= free_radius
    obstacle = cv2.morphologyEx(obstacle.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    sol = low
    vs = np.isfinite(sol)
    zmin, zmax = np.nanmin(sol), np.nanmax(sol)
    norm = np.where(vs, (sol - zmin) / (zmax - zmin), 0).astype(np.float32)
    filled = cv2.inpaint((norm * 65535).astype(np.float32), (~vs).astype(np.uint8), 15, cv2.INPAINT_TELEA) / 65535
    H = np.where(vs, sol, filled * (zmax - zmin) + zmin).astype(np.float32)
    # seconde passe sur grappes de pics (> 7×7) et bords d'inpainting : > 50 cm de la médiane 15×15
    med15 = median_filter(H, size=15)
    H = np.where(np.abs(H - med15) > 0.5, med15, H).astype(np.float32)
    # lissage léger 3×3 : à 10 cm, une marche entre cellules voisines devient une arête vive du hfield où les roues calent
    from scipy.ndimage import uniform_filter

    H = uniform_filter(H, size=3).astype(np.float32)
    if method == "troncs":
        from stems import build_obstacles

        troncs, src, low, stats = build_obstacles(P, {"H": H, "lo": lo, "res": res}, splat=splat, free_xy=free_xy)
        print(json.dumps({"obstacles": stats}), flush=True)
        return {"H": H, "sol": H, "lo": lo, "res": res, "couvert": vs, "obstacle": low, "troncs": troncs,
                "troncs_source": src}
    troncs = obstacle_cylinders(obstacle, H, lo, res)
    if free_xy is not None and len(troncs):
        # couloir libre : aucun tronc dont le BORD est à moins de trunk_margin du trajet réel (le robot y est passé)
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(free_xy[:, :2]).query(troncs[:, :2])
        troncs = troncs[dist - troncs[:, 2] >= trunk_margin]
    return {"H": H, "sol": H, "lo": lo, "res": res, "couvert": vs, "obstacle": obstacle, "troncs": troncs}


def obstacle_cylinders(obstacle, H, lo, res, small_area=0.5, step=0.2):
    """Obstacles → cylindres verticaux (x, y, rayon, z_base) pour MuJoCo. Composante compacte (≤ `small_area` m²,
    tronc, souche) : un cylindre de même aire ; composante étendue (buisson, tronc couché) : un cylindre de rayon
    0,6·step par case de `step` m occupée. Des piliers dans le champ de hauteur, eux, se font escalader par les roues
    (paroi verticale à frottement 1) et retournent le robot."""
    import cv2

    n, lab, stats, cent = cv2.connectedComponentsWithStats(obstacle.astype(np.uint8), connectivity=8)
    out = []
    k = int(round(step / res))
    for c in range(1, n):
        area = stats[c, cv2.CC_STAT_AREA] * res * res
        ii, jj = np.nonzero(lab == c)
        if area <= small_area:
            x, y = lo[0] + (ii.mean() + 0.5) * res, lo[1] + (jj.mean() + 0.5) * res
            out.append((x, y, max(0.06, np.sqrt(area / np.pi)), float(H[ii, jj].min())))
        else:
            for bi, bj in {(i // k, j // k) for i, j in zip(ii, jj)}:
                m = (ii // k == bi) & (jj // k == bj)
                x, y = lo[0] + (ii[m].mean() + 0.5) * res, lo[1] + (jj[m].mean() + 0.5) * res
                out.append((x, y, 0.6 * step, float(H[ii[m], jj[m]].min())))
    return np.array(out, np.float32).reshape(-1, 4)
