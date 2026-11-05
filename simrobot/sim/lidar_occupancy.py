"""Carte d'OCCUPATION pour le LiDAR simulé, construite en rejouant les vrais balayages Hesai XT32 de la mission
(GrandTour `dlio_hesai_points_undistorted`, redressés par DLIO, une pose `dlio_map_odometry` par balayage).

Pour chaque point mesuré, le rayon capteur → point est parcouru dans une grille de voxels de `res` m :
- chaque voxel reçoit la LONGUEUR de rayon parcourue dedans (jusqu'à l'impact) ;
- le voxel de l'impact reçoit +1 « touché » et l'intensité mesurée (moyenne glissante).
Coefficient d'extinction μ = touchés / longueur (estimateur des inventaires forestiers par LiDAR, cf. densité de
surface foliaire par voxel). En simulation, un rayon qui parcourt ℓ dans le voxel s'y arrête avec 1 − exp(−μ ℓ) :
un buisson souvent traversé est perméable, un tronc touché à chaque passage est opaque, et un rayon qui frôle un
coin de voxel (ℓ petit) compte peu, à la mesure comme à la simulation.
Piège évité : « touchés / (touchés + traversés) » par voxel (probabilité par voxel, première version) rendait les
troncs transparents (probabilité médiane 0,08 par tronc, traversée ~85 %) : les rayons qui frôlent un coin et les
écarts de recalage entre passages (~2–3 cm) comptaient comme des traversées complètes.
`--holdout N` : un balayage sur N est mis de côté (non utilisé), pour valider la simulation sur de vrais rayons.

Grille : en x, y alignée sur le terrain (sous-multiple de sa résolution) ; en z, colonnes qui suivent le sol (base =
sol − `below`, hauteur `below` + `above`), pour garder une grille dense tenant en mémoire. Au-delà, pas de voxels.
Sortie `lidar_occupation.npz` : voxels touchés au moins une fois, rangés par colonne (CSR), probabilité et intensité
sur 8 bits, base z de chaque colonne.

  PYTHONPATH=. venv/bin/python simrobot/sim/lidar_occupancy.py --scene out/simrobot/scenes/HOB-1_full \
      --scans hesai=/mnt/data/dino/gt_lidar/dlio_hesai_points_undistorted.tar \
              livox=/mnt/data/dino/gt_lidar/livox_points_undistorted.tar \
              velodyne=/mnt/data/dino/gt_lidar/velodyne_points_undistorted.tar \
      --odometry /mnt/data/dino/gt_lidar/dlio_map_odometry.tar
  # puis (Hesai seul construit à part avec --scans hesai=…, renommé lidar_occupation_hesai.npz) :
  PYTHONPATH=. venv/bin/python simrobot/sim/lidar_occupancy.py comble --all …/lidar_occupation.npz \
      --hesai …/lidar_occupation_hesai.npz --out …/lidar_occupation_comble.npz
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tarfile
import time

import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.dirname(__file__))
from gaussgym_export import FLIP  # noqa: E402

# boîtier (box_base) → LiDAR (hesai_lidar), métadonnées GrandTour (dlio_hesai_points_undistorted.yaml)
T_BOX_HESAI_Q = (-0.004419949690193552, -0.7093301791430758, 0.7048516039326548, 0.003921407292290494)   # w, x, y, z
T_BOX_HESAI_T = (-0.04461184951630229, 0.3022381420105506, -0.01253994548350209)


# Livox Mid-360 (montée à l'envers) et Velodyne VLP-16 (intégrée à l'ANYmal), tf.yaml de GrandTour
# (p_parent = R p_enfant + t)
T_BOX_LIVOX = ((0.9999743041039169, -0.006935301181255335, 0.0015623437878433857, -0.0009229362387923458),
               (-0.30263164816469024, 0.04875595821861217, 0.1098417136717083))
T_BASE_VELODYNE = ((2.220446049250313e-16, 0.0, 0.0, -1.0), (-0.31, -1.376676550535194e-16, -0.1585))
T_BASE_BOX = ((4.0026794885936924e-05, -0.9999999991989279, 0.0, 0.0),
              (-0.0764038, -0.036122437224394885, 0.2802761091673168))
# capteur → (groupe zarr, distance minimale : en deçà, le robot lui-même, décalage d'horodatage (s))
SENSORS = {"hesai": ("dlio_hesai_points_undistorted", 0.3, 0.0), "livox": ("livox_points_undistorted", 0.5, 0.0),
           "velodyne": ("velodyne_points_undistorted", 0.8, 0.05)}
# Corrections rigides (repère hesai, appliquées après les extrinsèques) estimées par ICP de 40 balayages sur la carte
# DLIO de HÖB-1 (vecteur de rotation, translation m). Écart médian à la carte : Velodyne 3,8 → 2,4 cm (0,92°, ~2 cm :
# capteur sur la base de l'ANYmal, pas sur le boîtier) ; Livox 1,7 → 1,6 cm.
CORRECTIONS = {"velodyne": ((0.00257, -0.009916, -0.012281), (-0.00178, 0.01202, 0.01864)),
               "livox": ((-0.001443, 0.00024, -0.003852), (-0.00161, 0.00251, 0.0036))}


def T44(q, t):
    M = np.eye(4)
    M[:3, :3] = quat_wxyz_to_R(*q)
    M[:3, 3] = t
    return M


def T_hesai_sensor(name, invert=True, corrected=True):
    """Capteur → repère hesai_lidar (celui dont DLIO donne la pose), par le boîtier (box_base) et la base de l'ANYmal.
    invert : les transformations de tf.yaml sont lues comme parent ← enfant inversé (--check sur HÖB-1 : Livox 1,7 cm
    de la carte DLIO contre 12–19 cm avec l'autre lecture, Velodyne 3,8 contre 17–24 cm) ; corrected : correction ICP."""
    from scipy.spatial.transform import Rotation

    f = (lambda M: np.linalg.inv(M)) if invert else (lambda M: M)
    box_hesai = f(T44(T_BOX_HESAI_Q, T_BOX_HESAI_T))
    if name == "hesai":
        return np.eye(4)
    if name == "livox":
        M = np.linalg.inv(box_hesai) @ f(T44(*T_BOX_LIVOX))
    elif name == "velodyne":
        M = np.linalg.inv(box_hesai) @ np.linalg.inv(f(T44(*T_BASE_BOX))) @ f(T44(*T_BASE_VELODYNE))
    else:
        raise ValueError(name)
    if corrected and name in CORRECTIONS:
        C = np.eye(4)
        C[:3, :3] = Rotation.from_rotvec(CORRECTIONS[name][0]).as_matrix()
        C[:3, 3] = CORRECTIONS[name][1]
        M = C @ M
    return M


def quat_wxyz_to_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class TarZarr:
    """Lecture par blocs d'un groupe zarr v2 rangé dans un tar (sans tout charger : 3,2 Go pour HÖB-1)."""

    def __init__(self, path, group):
        self.tf = tarfile.open(path)
        self.mem = {m.name.lstrip("./"): m for m in self.tf.getmembers() if m.isfile()}
        self.group = group

    def meta(self, key):
        return json.load(self.tf.extractfile(self.mem[f"{self.group}/{key}/.zarray"]))

    def keys(self):
        return sorted({n.split("/")[1] for n in self.mem if n.startswith(self.group + "/") and n.count("/") >= 2})

    def chunk(self, key, ci):
        import numcodecs

        za = self.meta(key)
        name = f"{self.group}/{key}/{ci}" + ".0" * (len(za["shape"]) - 1)
        raw = self.tf.extractfile(self.mem[name]).read()
        c = numcodecs.get_codec(za["compressor"]) if za["compressor"] else None
        a = np.frombuffer(c.decode(raw) if c else raw, dtype=np.dtype(za["dtype"]))
        n0 = min(za["chunks"][0], za["shape"][0] - ci * za["chunks"][0])
        return a.reshape(za["chunks"])[:n0]

    def row(self, key, k):
        """Ligne k (numéro de balayage global) d'un tableau ; blocs de tailles différentes selon le tableau (points :
        256 balayages, intensité : 512) — un bloc en cache par tableau et par instance (plusieurs capteurs)."""
        if not hasattr(self, "_cache"):
            self._cache, self._meta = {}, {}
        if key not in self._meta:
            self._meta[key] = self.meta(key)["chunks"][0]
        c0 = self._meta[key]
        ci = k // c0
        if self._cache.get(key, (None,))[0] != ci:
            self._cache[key] = (ci, self.chunk(key, ci))
        return self._cache[key][1][k - ci * c0]


@njit(cache=True, parallel=True, fastmath=True)
def accumulate(O, P, I, lo0, lo1, res, zb, nk, hits, length, inten, min_range=0.3, upd_int=True):
    """O (s, 3) origines par balayage, P (s, n, 3) points (NaN = absent), I (s, n) intensités 0–255.
    hits : impacts par voxel ; length : longueur de rayon parcourue dans chaque voxel (m)."""
    nx, ny = zb.shape
    S, N = P.shape[0], P.shape[1]
    for q in prange(S * N):
        s = q // N
        px, py, pz = P[s, q % N, 0], P[s, q % N, 1], P[s, q % N, 2]
        if not (px == px):
            continue
        ox, oy, oz = O[s, 0], O[s, 1], O[s, 2]
        dx, dy, dz = px - ox, py - oy, pz - oz
        L = math.sqrt(dx * dx + dy * dy + dz * dz)
        if L < min_range:
            continue
        dx /= L
        dy /= L
        dz /= L
        hx = int(math.floor((px - lo0) / res))
        hy = int(math.floor((py - lo1) / res))
        if hx >= 0 and hx < nx and hy >= 0 and hy < ny:
            hk = int(math.floor((pz - zb[hx, hy]) / res))
            if hk >= 0 and hk < nk:
                if hits[hx, hy, hk] < 65535:
                    hits[hx, hy, hk] += 1
                if upd_int:                              # intensité du seul LiDAR simulé (Hesai) : échelles différentes
                    inten[hx, hy, hk] = np.uint8((np.int32(inten[hx, hy, hk]) * 7 + np.int32(I[s, q % N])) >> 3) \
                        if hits[hx, hy, hk] > 1 else np.uint8(I[s, q % N])
                elif inten[hx, hy, hk] == 0:
                    inten[hx, hy, hk] = 60
        # longueur parcourue dans chaque voxel jusqu'à l'impact (parcours 2D des colonnes, tranches en hauteur)
        ix = int(math.floor((ox - lo0) / res))
        iy = int(math.floor((oy - lo1) / res))
        sx = 1 if dx > 0 else -1
        sy = 1 if dy > 0 else -1
        tdx = res / abs(dx) if dx != 0 else 1e30
        tdy = res / abs(dy) if dy != 0 else 1e30
        tmx = ((lo0 + (ix + (1 if dx > 0 else 0)) * res) - ox) / dx if dx != 0 else 1e30
        tmy = ((lo1 + (iy + (1 if dy > 0 else 0)) * res) - oy) / dy if dy != 0 else 1e30
        t0 = 0.0
        while t0 < L:
            t1 = min(tmx, tmy, L)
            if ix >= 0 and ix < nx and iy >= 0 and iy < ny:
                base = zb[ix, iy]
                za = oz + dz * t0 - base
                zc = oz + dz * t1 - base
                k0 = int(math.floor(min(za, zc) / res))
                k1 = int(math.floor(max(za, zc) / res))
                if k0 < 0:
                    k0 = 0
                if k1 > nk - 1:
                    k1 = nk - 1
                for k in range(k0, k1 + 1):
                    if dz != 0:
                        ta = (base + k * res - oz) / dz
                        tb = (base + (k + 1) * res - oz) / dz
                        ell = min(t1, max(ta, tb)) - max(t0, min(ta, tb))
                    else:
                        ell = t1 - t0
                    if ell > 0:
                        length[ix, iy, k] += ell
            if tmx < tmy:
                t0 = tmx
                tmx += tdx
                ix += sx
            else:
                t0 = tmy
                tmy += tdy
                iy += sy


def aggregate(V, f=2, prior=0.1):
    """Carte d'occupation agrégée par f × f × f (μ = Σ touchés / Σ longueur : agrégation exacte de l'estimateur).
    Les colonnes de base z identiques par bloc f × f (grille du terrain à f × res) gardent leurs niveaux alignés."""
    nx, ny, nk, res = int(V["nx"]), int(V["ny"]), int(V["nk"]), float(V["res"])
    cs = V["col_start"]
    col = np.repeat(np.arange(len(cs) - 1), np.diff(cs))
    ix, iy, k = col // ny, col % ny, V["vz"].astype(np.int64)
    nx2, ny2, nk2 = nx // f, ny // f, nk // f
    ok = (ix // f < nx2) & (iy // f < ny2) & (k // f < nk2)
    key = ((ix[ok] // f) * ny2 + iy[ok] // f) * nk2 + k[ok] // f
    uk, inv = np.unique(key, return_inverse=True)
    h = np.bincount(inv, V["hits"][ok].astype(np.float64), len(uk))
    ell = np.bincount(inv, V["length"][ok].astype(np.float64), len(uk))
    it = np.bincount(inv, V["inten"][ok].astype(np.float64) * V["hits"][ok], len(uk)) / np.maximum(h, 1)
    col2, k2 = uk // nk2, (uk % nk2).astype(np.uint16)
    col_start = np.zeros(nx2 * ny2 + 1, np.int64)
    np.add.at(col_start, col2 + 1, 1)
    out = {k_: V[k_] for k_ in ("lo", "below") if k_ in V}
    out.update({"res": np.float64(res * f), "nx": np.int64(nx2), "ny": np.int64(ny2), "nk": np.int64(nk2),
                "zb": np.ascontiguousarray(V["zb"][::f, ::f][:nx2, :ny2]), "col_start": np.cumsum(col_start),
                "vz": k2, "hits": np.minimum(h, 65535).astype(np.uint16), "length": ell.astype(np.float32),
                "mu": np.minimum(h / (ell + prior), 1000).astype(np.float16), "inten": it.astype(np.uint8),
                "prior_m": np.float64(prior), "held_scans": V["held_scans"], "holdout": V["holdout"]})
    return out


def fill(all_npz, hesai_npz, out_npz, min_length=0.05, prior=0.1):
    """Carte « Hesai + comblement » : μ du Hesai là où il a parcouru au moins `min_length` m dans le voxel, sinon μ de
    tous les capteurs (comptes des trois LiDAR). Validation HÖB-1 sur balayages Hesai mis de côté : mélanger les trois
    capteurs partout dégrade la prédiction du Hesai (44,2 → 40,1 % juste ; faisceaux et sensibilités différents,
    Velodyne recalée à 2,4 cm) ; le comblement est quasi neutre (43,5 %), rapproche le 90e centile des distances du réel
    (27,5 → 25,8 m ; réel 24,3) et ajoute 39 % de voxels, à 93 % jamais traversés par le Hesai (couverture hors
    trajet). Sortie compacte (μ seulement)."""
    A = dict(np.load(all_npz))
    Hs = np.load(hesai_npz)
    nk = int(A["nk"])

    def keys(V):
        cs = V["col_start"]
        return np.repeat(np.arange(len(cs) - 1), np.diff(cs)) * nk + V["vz"].astype(np.int64)

    kA, kH = keys(A), keys(Hs)
    pos = np.searchsorted(kA, kH)
    assert np.all(kA[pos] == kH), "la carte Hesai doit être incluse dans celle des trois capteurs (mêmes balayages)"
    hH = np.zeros(len(kA), np.float32)
    lH = np.zeros(len(kA), np.float32)
    hH[pos], lH[pos] = Hs["hits"], Hs["length"]
    use_h = lH >= min_length
    mu = np.where(use_h, hH / (lH + prior), A["hits"] / (A["length"] + prior))
    A["mu"] = np.minimum(mu, 1000).astype(np.float16)
    A["prior_m"] = np.float64(prior)
    A.pop("hits")
    A.pop("length")
    np.savez(out_npz, **A)
    return float(use_h.mean())


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "comble":
        ap = argparse.ArgumentParser(description="carte Hesai + comblement par les autres LiDAR (cf. fill)")
        ap.add_argument("cmd")
        ap.add_argument("--all", required=True, help="carte des trois LiDAR (comptes bruts)")
        ap.add_argument("--hesai", required=True, help="carte du Hesai seul (comptes bruts, mêmes balayages Hesai)")
        ap.add_argument("--out", required=True)
        ap.add_argument("--min-length", type=float, default=0.05)
        a = ap.parse_args()
        print(json.dumps({"voxels_mu_hesai": round(fill(a.all, a.hesai, a.out, a.min_length), 3)}))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--scans", required=True, nargs="+",
                    help="capteur=tar, ex. hesai=…/dlio_hesai_points_undistorted.tar livox=… velodyne=…")
    ap.add_argument("--odometry", required=True)
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--below", type=float, default=1.0, help="m sous le sol du terrain")
    ap.add_argument("--above", type=float, default=12.0, help="m au-dessus du sol du terrain")
    ap.add_argument("--prior", type=float, default=0.1,
                    help="longueur a priori (m) de μ = touchés / (longueur + prior), réglée sur les balayages mis de côté")
    ap.add_argument("--holdout", type=int, default=10,
                    help="un balayage Hesai sur N mis de côté (validation) ; les balayages des autres capteurs à moins "
                         "de 0,06 s de ces instants sont écartés aussi ; 0 = aucun")
    ap.add_argument("--every", type=int, default=1, help="un balayage sur N")
    ap.add_argument("--max-scans", type=int, default=0, help="par capteur")
    ap.add_argument("--time-offset", nargs="*", default=[], help="capteur=décalage (s) de l'horodatage de balayage")
    ap.add_argument("--check", action="store_true", help="écart de quelques balayages à la carte DLIO, puis sortie")
    a = ap.parse_args()
    t_start = time.time()
    from lidar_terrain import quat_xyzw_to_R, zarr_group
    from scipy.spatial.transform import Rotation, Slerp

    meta = json.load(open(f"{a.scene}/scene.json"))
    T = dict(np.load(f"{a.scene}/terrain.npz"))
    centre = np.array(meta["centre_monde_m"])
    Rr, tr = np.array(meta["recalage_lidar"]["R"]), np.array(meta["recalage_lidar"]["t"])
    odo = zarr_group(a.odometry, "dlio_map_odometry")
    to, po, qo = odo("timestamp").astype(float), odo("pose_pos").astype(float), odo("pose_orien").astype(float)
    qo = qo / np.linalg.norm(qo, axis=1, keepdims=True)
    Ro = quat_xyzw_to_R(qo)
    slerp = Slerp(to, Rotation.from_quat(qo))                     # pose du Hesai à un instant quelconque
    offsets = {k: float(v) for k, v in (x.split("=") for x in a.time_offset)}
    srcs = {k: TarZarr(v, SENSORS[k][0]) for k, v in (x.split("=") for x in a.scans)}
    to_phys = lambda x: (x @ Rr.T + tr) @ FLIP.T - centre         # noqa: E731

    def valid_mask(Z, k, pts):
        v = np.asarray(Z.row("valid", k)).reshape(-1)
        # `valid` = nombre de points valides du balayage (forme [balayages, 1]) ; sinon masque par point
        v = (np.arange(len(pts)) < int(v[0])) if v.size == 1 else v.astype(bool)[:len(pts)]
        return v & np.isfinite(pts).all(1) & (np.abs(pts).sum(1) > 0)

    def pose_map(name, Z, k, invert=True, dt=0.0):
        """Capteur → dlio_map pour le balayage k : pose DLIO du Hesai (même indice pour le Hesai, interpolée à
        l'horodatage du balayage pour les autres) composée avec les extrinsèques."""
        if name == "hesai":
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = Ro[k], po[k]
            return M
        ts = float(np.asarray(Z.row("timestamp", k)).reshape(-1)[0]) + offsets.get(name, SENSORS[name][2]) + dt
        if ts < to[0] or ts > to[-1]:
            return None
        M = np.eye(4)
        M[:3, :3] = slerp([ts]).as_matrix()[0]
        M[:3, 3] = [np.interp(ts, to, po[:, i]) for i in range(3)]
        return M @ T_hesai_sensor(name, invert)

    def scan_physics(name, Z, k, **kw):
        pts = Z.row("points", k).reshape(-1, 3).astype(np.float64)
        ok = valid_mask(Z, k, pts)
        M = pose_map(name, Z, k, **kw)
        if M is None:
            return None
        P = pts[ok] @ M[:3, :3].T + M[:3, 3]
        return to_phys(P), to_phys(M[:3, 3][None])[0], ok

    if a.check:
        from lidar_voxels import read_ply_xyzi
        from scipy.spatial import cKDTree
        import glob

        Pm, _ = read_ply_xyzi(glob.glob(f"{os.path.dirname(a.odometry)}/*_dlio.ply")[0])
        Pm = to_phys(Pm)
        tree, cache = None, {}
        for name, Z in srcs.items():
            n = Z.meta("points")["shape"][0]
            for inv in ((True,) if name == "hesai" else (False, True)):
                for dt in ((0.0,) if name == "hesai" else (-0.05, 0.0, 0.05)):
                    ds = []
                    for k in (n // 5, n // 2, 4 * n // 5):
                        r = scan_physics(name, Z, k, invert=inv, dt=dt)
                        if r is None:
                            continue
                        P, o, _ = r
                        key = tuple(np.round(o[:2]))
                        if key not in cache:
                            sel = np.linalg.norm(Pm[:, :2] - o[:2], axis=1) < 25
                            cache[key] = cKDTree(Pm[sel])
                        d, _ = cache[key].query(P[::10])
                        ds.append(np.median(d))
                    print(f"{name:9s} convention {'inverse' if inv else 'directe'} décalage {dt:+.2f} s : écart médian "
                          f"à la carte DLIO {np.round(ds, 3)} m", flush=True)
        return

    res = a.res
    f = int(round(float(T["res"]) / res))
    H = np.repeat(np.repeat(T["H"], f, 0), f, 1)
    zb = (H - a.below).astype(np.float32)
    nx, ny = zb.shape
    nk = int(round((a.below + a.above) / res))
    lo = np.asarray(T["lo"], float)
    print(json.dumps({"grille": [nx, ny, nk], "Go": round(nx * ny * nk * 7 / 1e9, 2)}), flush=True)
    hits = np.zeros((nx, ny, nk), np.uint16)
    length = np.zeros((nx, ny, nk), np.float32)
    inten = np.zeros((nx, ny, nk), np.uint8)
    # balayages Hesai mis de côté, et instants correspondants (écartés pour les autres capteurs)
    n_h = srcs["hesai"].meta("points")["shape"][0] if "hesai" in srcs else len(to)
    allh = np.arange(0, n_h, a.every)
    held = allh[(allh % a.holdout) == a.holdout // 2] if a.holdout else allh[:0]
    held_t = to[held] if len(held) else np.zeros(0)
    i99 = 42.0                                                    # 99e centile de l'intensité du Hesai (carte DLIO)
    st = {"capteurs": {}}
    for name, Z in srcs.items():
        zp = Z.meta("points")
        scans = np.arange(0, zp["shape"][0], a.every)
        if name == "hesai":
            scans = np.setdiff1d(scans, held)
        elif len(held_t):
            ts = np.array([float(np.asarray(Z.row("timestamp", k)).reshape(-1)[0]) for k in scans])
            j = np.searchsorted(held_t, ts)
            near = np.minimum(np.abs(held_t[np.clip(j, 0, len(held_t) - 1)] - ts),
                              np.abs(held_t[np.clip(j - 1, 0, len(held_t) - 1)] - ts))
            scans = scans[near > 0.06]          # ±0,25 s écartait la moitié des balayages (1 Hesai mis de côté / s)
        if a.max_scans:
            scans = scans[:a.max_scans]
        npt, n_scans, n_pts = zp["shape"][1], 0, 0
        for b0 in range(0, len(scans), 16):                       # lots de 16 balayages (parallélisme sur les points)
            bs = scans[b0:b0 + 16]
            Pb = np.full((len(bs), npt, 3), np.nan, np.float64)
            Ob = np.zeros((len(bs), 3))
            Ib = np.zeros((len(bs), npt), np.float32)
            for j, k in enumerate(bs):
                r = scan_physics(name, Z, k)
                if r is None:
                    continue
                P, o, ok = r
                Pb[j, :len(P)] = P
                Ob[j] = o
                if name == "hesai":
                    Ib[j, :len(P)] = Z.row("intensity", k).reshape(-1)[:len(ok)][ok]
                n_pts += len(P)
            accumulate(Ob, Pb, np.clip(Ib / i99 * 255, 0, 255).astype(np.uint8), lo[0], lo[1], res, zb, nk,
                       hits, length, inten, SENSORS[name][1], name == "hesai")
            n_scans += len(bs)
            if (b0 // 16) % 32 == 0:
                print(json.dumps({"capteur": name, "balayages": n_scans, "points": n_pts,
                                  "s": round(time.time() - t_start)}), flush=True)
        st["capteurs"][name] = {"balayages": n_scans, "points": n_pts}
    # voxels touchés au moins une fois → CSR par colonne
    idx = np.flatnonzero(hits.ravel())
    h, ell = hits.ravel()[idx].astype(np.float32), length.ravel()[idx]
    mu = h / (ell + a.prior)                         # 1/m ; longueur a priori : un voxel touché une fois n'est pas opaque
    p = 1 - np.exp(-mu * res)                        # opacité d'une traversée de res (statistiques, rendu)
    col, k = idx // nk, (idx % nk).astype(np.uint16)
    col_start = np.zeros(nx * ny + 1, np.int64)
    np.add.at(col_start, col + 1, 1)
    col_start = np.cumsum(col_start)
    out = {"lo": lo, "res": np.float64(res), "nx": np.int64(nx), "ny": np.int64(ny), "nk": np.int64(nk),
           "zb": zb, "col_start": col_start, "vz": k, "mu": np.minimum(mu, 1000).astype(np.float16),
           "below": np.float64(a.below), "prior_m": np.float64(a.prior),
           # comptes bruts (réglage de la longueur a priori sans rejouer les balayages : lidar_validate.py --priors)
           "hits": np.minimum(h, 65535).astype(np.uint16), "length": ell.astype(np.float32),
           "inten": inten.ravel()[idx], "holdout": np.int64(a.holdout), "held_scans": held.astype(np.int32),
           "capteurs": np.array(list(srcs))}
    hz = k.astype(np.float32) * res - a.below
    st.update({"voxels": int(len(idx)), "duree_s": round(time.time() - t_start)})
    for lo_, hi_ in [(-0.2, 0.2), (0.2, 0.6), (0.6, 2.0), (2.0, 6.0), (6.0, 12.0)]:
        mm = (hz >= lo_) & (hz < hi_)
        if mm.any():
            st[f"p_median_{lo_}_{hi_}m"] = round(float(np.median(p[mm])), 3)
    out["stats"] = json.dumps(st)
    np.savez(f"{a.scene}/lidar_occupation.npz", **out)
    st["taille_Mo"] = round(os.path.getsize(f"{a.scene}/lidar_occupation.npz") / 1e6, 1)
    print(json.dumps(st), flush=True)


if __name__ == "__main__":
    main()
