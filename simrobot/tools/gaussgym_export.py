"""Morceaux de splat GaussGym / GrandTour (splatfacto) → PLY 3DGS métrique dans le repère de la mission GrandTour
(z haut), assemblables : chaque gaussienne est gardée par le morceau dont une caméra est la plus proche (partition de
Voronoï sur les positions caméra de tous les morceaux), ce qui supprime les doublons aux frontières.

Repère : splatfacto stocke p_dp = s · (R p_mission + t) (`dataparser_transforms.json`). On inverse : positions,
échelles (− log s), quaternions ET harmoniques sphériques (degrés 1-3 tournés, sinon la couleur dépendante de la vue
serait fausse dans le nouveau repère).

  # un morceau, pour SuperSplat (recentré, y haut, rayon 20 m) :
  python simrobot/tools/gaussgym_export.py --tiles <dir_morceau> --out tile.ply --yup --radius 20
  # assemblage d'une mission (tous les morceaux d'un dossier) :
  python simrobot/tools/gaussgym_export.py --tiles <dir_mission>/* --out mission.ply
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gaussgym_check import read_splat_ply  # noqa: E402

# constantes des harmoniques sphériques réelles (convention 3DGS / gsplat)
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435]


def sh_basis(d):
    """Bases SH réelles degrés 1..3 (15 colonnes, ordre 3DGS) aux directions unitaires d (M, 3)."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    return np.stack([
        -C1 * y, C1 * z, -C1 * x,
        C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * zz - xx - yy), C2[3] * x * z, C2[4] * (xx - yy),
        C3[0] * y * (3 * xx - yy), C3[1] * x * y * z, C3[2] * y * (4 * zz - xx - yy),
        C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy), C3[5] * z * (xx - yy),
        C3[6] * x * (xx - 3 * yy)], 1)


def sh_rotation(R):
    """Matrice 15×15 M telle que rest' = M @ rest quand la scène tourne de R (couleur'(d) = couleur(Rᵀ d))."""
    rng = np.random.default_rng(0)
    d = rng.normal(size=(2000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    B, Br = sh_basis(d), sh_basis(d @ R)            # Br[i] = Y(Rᵀ d_i)
    M = np.zeros((15, 15))
    for lo, hi in [(0, 3), (3, 8), (8, 15)]:          # bloc par degré
        X, *_ = np.linalg.lstsq(B[:, lo:hi], Br[:, lo:hi], rcond=None)   # Br = B @ X
        M[lo:hi, lo:hi] = X
    return M


def quat_from_R(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def quat_mul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def tile_cameras(tile):
    fr = json.load(open(f"{tile}/transforms.json"))["frames"]
    return np.array([np.array(f["transform_matrix"], float)[:3, 3] for f in fr])   # repère mission, mètres


def load_tile_mission(tile):
    """Gaussiennes d'un morceau dans le repère mission (mètres) : means, sh (N,16,3), opac, scales, quats."""
    dp = json.load(open(f"{tile}/splatfacto/dataparser_transforms.json"))
    A = np.array(dp["transform"], float)
    R, t, s = A[:, :3], A[:, 3], dp["scale"]
    G = read_splat_ply(f"{tile}/splatfacto/splat.ply")
    Ri = R.T                                          # dp → mission
    G["means"] = (G["means"] / s - t) @ Ri.T
    G["scales"] = G["scales"] / s
    q = G["quats"] / np.linalg.norm(G["quats"], axis=1, keepdims=True)
    G["quats"] = quat_mul(quat_from_R(Ri)[None], q)
    M = sh_rotation(Ri)
    G["sh"][:, 1:] = np.einsum("ij,njc->nic", M, G["sh"][:, 1:])
    return G


# Le repère « mission » des poses GaussGym / GrandTour (transforms.json) a l'axe z vers le BAS (axe haut des images
# ≈ −z, vérifié sur HÖB-1, ROOT-1, LMB-1 ; caméras 0,7 m SOUS le sol en z). Le repère de la carte LiDAR DLIO
# (`dlio_map`, aligné sur la gravité, z haut) s'en déduit par 180° autour de x (recalage : résidu 3 mm, cf.
# simrobot/sim/lidar_terrain.py). Repère « monde » de la simulation = mission tourné par FLIP.
FLIP = np.diag([1.0, -1.0, -1.0])


def tile_cameras_world(tile):
    return tile_cameras(tile) @ FLIP.T


def load_tile_world(tile):
    """Gaussiennes d'un morceau dans le repère monde (z haut, mètres)."""
    G = load_tile_mission(tile)
    rotate_all(G, FLIP, np.zeros(3))
    return G


def rotate_all(G, R, center):
    G["means"] = (G["means"] - center) @ R.T
    G["quats"] = quat_mul(quat_from_R(R)[None], G["quats"])
    M = sh_rotation(R)
    G["sh"][:, 1:] = np.einsum("ij,njc->nic", M, G["sh"][:, 1:])


def write_ply(path, G, keep_sh=True, sh_degree=None, normals=True):
    """PLY 3DGS. `sh_degree` < degré du splat : bandes supérieures retirées ; `normals=False` : pas de nx, ny, nz
    (3 flottants inutiles par gaussienne)."""
    n = len(G["means"])
    op = np.clip(G["opac"], 1e-6, 1 - 1e-6)          # sigmoïde saturée à 1,0 en float32 → logit infini
    k = G["sh"].shape[1] if sh_degree is None else (sh_degree + 1) ** 2
    rest = G["sh"][:, 1:k].transpose(0, 2, 1).reshape(n, -1) if keep_sh else np.zeros((n, 0), np.float32)
    cols = [G["means"]] + ([np.zeros((n, 3))] if normals else []) + [G["sh"][:, 0], rest,
            np.log(op / (1 - op))[:, None], np.log(G["scales"]), G["quats"]]
    data = np.concatenate(cols, 1).astype(np.float32)
    names = ["x", "y", "z"] + (["nx", "ny", "nz"] if normals else []) + ["f_dc_0", "f_dc_1", "f_dc_2"] + \
            [f"f_rest_{i}" for i in range(rest.shape[1])] + \
            ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" +
                 "".join(f"property float {k}\n" for k in names) + "end_header\n").encode())
        f.write(data.tobytes())


def main():
    from scipy.spatial import cKDTree

    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=0.0, help="rayon de découpe autour du centre des caméras (0 = tout)")
    ap.add_argument("--min-opacity", type=float, default=0.05)
    ap.add_argument("--yup", action="store_true",
                    help="recentrer et passer (x, y, z) → (x, z, −y) depuis le repère mission (z BAS) : PLY y vers le "
                         "bas, convention COLMAP, que SuperSplat affiche à l'endroit")
    ap.add_argument("--no-sh", action="store_true", help="ne garder que la couleur de base (fichier 4× plus léger)")
    a = ap.parse_args()

    tiles = [t.rstrip("/") for t in a.tiles if os.path.exists(f"{t}/splatfacto/splat.ply")]
    cams = [tile_cameras(t) for t in tiles]
    owner = np.concatenate([np.full(len(c), i) for i, c in enumerate(cams)])
    tree = cKDTree(np.concatenate(cams))
    center = np.concatenate(cams).mean(0)
    parts, stats = [], []
    for i, t in enumerate(tiles):
        G = load_tile_mission(t)
        keep = G["opac"] >= a.min_opacity
        if len(tiles) > 1:                            # partition de Voronoï sur les caméras
            _, j = tree.query(G["means"])
            keep &= owner[j] == i
        if a.radius > 0:
            keep &= np.linalg.norm(G["means"][:, :2] - center[:2], axis=1) <= a.radius
        parts.append({k: v[keep] for k, v in G.items()})
        stats.append({"morceau": os.path.basename(t), "gardees": int(keep.sum()), "total": int(len(keep))})
        print(json.dumps(stats[-1]), flush=True)
    G = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if a.yup:
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)    # (x, y, z) → (x, z, −y)
        rotate_all(G, R, center)
    write_ply(a.out, G, keep_sh=not a.no_sh)
    meta = {"morceaux": stats, "n": int(len(G["means"])), "centre_mission_m": center.round(3).tolist(),
            "y_haut": a.yup, "rayon_m": a.radius, "sh": not a.no_sh}
    json.dump(meta, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "morceaux"}))


if __name__ == "__main__":
    main()
