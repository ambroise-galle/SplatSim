"""Terrain physique tiré des maillages GaussGym (NKSR) : champ de hauteur régulier, repère mission (z haut, mètres).

Les maillages GaussGym ne couvrent que le sol, dans un couloir d'environ 6 m autour du chemin (pas de troncs, pas de
canopée) : hauteur = max des points échantillonnés sur les triangles de la cellule ; les cellules hors maillage sont
bouchées par la cellule couverte la plus proche (masque `couvert` conservé pour l'affichage).
"""

from __future__ import annotations

import numpy as np


def read_mesh_ply(path):
    """Maillage Open3D binaire (x, y, z double [+ normales] + couleurs uchar ; faces uchar + 3 uint)."""
    with open(path, "rb") as f:
        props, nv, nf = [], 0, 0
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                nv = int(line.split()[-1])
            elif line.startswith("element face"):
                nf = int(line.split()[-1])
            elif line.startswith("property") and "list" not in line:
                props.append((line.split()[-1], {"double": "<f8", "float": "<f4", "uchar": "u1"}[line.split()[1]]))
            elif line == "end_header":
                break
        V = np.fromfile(f, dtype=np.dtype(props), count=nv)
        F = np.fromfile(f, dtype=np.dtype([("n", "u1"), ("v", "<u4", 3)]), count=nf)
    return np.stack([V["x"], V["y"], V["z"]], 1), F["v"].astype(np.int64)


def sample_triangles(V, F, spacing, rng):
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    k = np.clip(np.ceil(area / (spacing * spacing / 4)), 1, 50).astype(int)
    idx = np.repeat(np.arange(len(F)), k)
    u, v = rng.random((2, len(idx)))
    m = u + v > 1
    u[m], v[m] = 1 - u[m], 1 - v[m]
    t = tri[idx]
    return np.concatenate([V, t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])])


def heightfield(meshes, res=0.1, pad=3.0):
    """→ dict(H (nx, ny) hauteurs aux CENTRES de cellule, lo (x, y) du coin, res, couvert (nx, ny) bool)."""
    from scipy.ndimage import distance_transform_edt

    rng = np.random.default_rng(0)
    P = np.concatenate([sample_triangles(V, F, res, rng) for V, F in meshes])
    lo = P[:, :2].min(0) - pad
    ij = np.floor((P[:, :2] - lo) / res).astype(int)
    n = ij.max(0) + 1 + int(pad / res)
    H = np.full(n, -np.inf)
    np.maximum.at(H, (ij[:, 0], ij[:, 1]), P[:, 2])
    cov = np.isfinite(H)
    _, (ii, jj) = distance_transform_edt(~cov, return_indices=True)
    return {"H": H[ii, jj].astype(np.float32), "lo": lo, "res": res, "couvert": cov}


def height_at(terrain, xy):
    """Hauteur bilinéaire du champ aux points xy (N, 2)."""
    H, lo, res = terrain["H"], terrain["lo"], terrain["res"]
    f = (np.atleast_2d(xy) - lo) / res - 0.5
    i0 = np.clip(np.floor(f).astype(int), 0, np.array(H.shape) - 2)
    t = np.clip(f - i0, 0, 1)
    a = H[i0[:, 0], i0[:, 1]] * (1 - t[:, 0]) + H[i0[:, 0] + 1, i0[:, 1]] * t[:, 0]
    b = H[i0[:, 0], i0[:, 1] + 1] * (1 - t[:, 0]) + H[i0[:, 0] + 1, i0[:, 1] + 1] * t[:, 0]
    return a * (1 - t[:, 1]) + b * t[:, 1]
