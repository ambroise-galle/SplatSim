"""LiDAR rotatif simulé : rayons lancés dans la carte de voxels de la mission (`lidar_voxels.npz`) et sur le sol du
terrain physique, compilé avec numba (parallèle, sans GIL).

Lancer de rayons (`cast`) :
- parcours 2D des colonnes de la grille (Amanatides-Woo) ; dans chaque colonne, recherche binaire des voxels
  traversés entre les hauteurs d'entrée et de sortie du rayon ;
- impact = intersection avec le petit plan du voxel (centre + normale) s'il est bien défini, sinon point du rayon le
  plus proche du centre des points ; précision sous le voxel, sans marches ;
- probabilité de retour par voxel (feuillage clairsemé traversable) : tirage indépendant par voxel, donc le plus proche
  des voxels « retenus » = traitement séquentiel le long du rayon ;
- plancher : sol du terrain physique (bilinéaire), là où la carte n'a pas vu le sol ;
- bruit de distance gaussien.

Capteur (`Lidar`) : balayage TOURNANT. Chaque appel à `advance` tire les colonnes d'azimut couvertes pendant le pas
de temps, depuis la pose courante du robot, d'où la déformation due au mouvement, comme sur un vrai capteur. Un tour
complet donne un balayage (points en repère physique, distances, intensités).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True, inline="always")
def _height(x, y, H, lo0, lo1, res):
    nx, ny = H.shape
    fx = (x - lo0) / res - 0.5
    fy = (y - lo1) / res - 0.5
    i0 = min(max(int(math.floor(fx)), 0), nx - 2)
    j0 = min(max(int(math.floor(fy)), 0), ny - 2)
    tx = min(max(fx - i0, 0.0), 1.0)
    ty = min(max(fy - j0, 0.0), 1.0)
    a = H[i0, j0] * (1 - tx) + H[i0 + 1, j0] * tx
    b = H[i0, j0 + 1] * (1 - tx) + H[i0 + 1, j0 + 1] * tx
    return a * (1 - ty) + b * ty


@njit(cache=True, inline="always")
def _xs(st):                                             # xorshift64 (constantes uint64 : sinon numba passe en flottant)
    st ^= st << np.uint64(13)
    st ^= st >> np.uint64(7)
    st ^= st << np.uint64(17)
    return st


@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def cast(O, Rm, D, lo0, lo1, res, nx, ny, z0, col_start, vz, cen, nrm, phit, inten, H, max_range, min_range, noise,
         seed, ground_int, out_r, out_i, out_p):
    """O (3,) origine commune, Rm (3, 3) rotation capteur → physique, D (n, 3) directions unitaires dans le repère du
    capteur. → out_r (distance, NaN sans retour), out_i (intensité 0–255), out_p (point, repère physique)."""
    n = D.shape[0]
    half = 0.5 * res
    ox, oy, oz = O[0], O[1], O[2]
    for i in prange(n):
        dx = Rm[0, 0] * D[i, 0] + Rm[0, 1] * D[i, 1] + Rm[0, 2] * D[i, 2]
        dy = Rm[1, 0] * D[i, 0] + Rm[1, 1] * D[i, 1] + Rm[1, 2] * D[i, 2]
        dz = Rm[2, 0] * D[i, 0] + Rm[2, 1] * D[i, 1] + Rm[2, 2] * D[i, 2]
        st = _xs(np.uint64(seed * 7477965 + i * 2891336453 + 88172645463325252))
        ix = int(math.floor((ox - lo0) / res))
        iy = int(math.floor((oy - lo1) / res))
        sx = 1 if dx > 0 else -1
        sy = 1 if dy > 0 else -1
        tdx = res / abs(dx) if dx != 0 else 1e30
        tdy = res / abs(dy) if dy != 0 else 1e30
        tmx = ((lo0 + (ix + (1 if dx > 0 else 0)) * res) - ox) / dx if dx != 0 else 1e30
        tmy = ((lo1 + (iy + (1 if dy > 0 else 0)) * res) - oy) / dy if dy != 0 else 1e30
        t0 = 0.0
        best = 1e30
        bi = 0
        ga = _height(ox, oy, H, lo0, lo1, res)
        ga_ok = True
        while t0 < max_range:
            t1 = min(tmx, tmy, max_range)
            if ix >= 0 and ix < nx and iy >= 0 and iy < ny:
                col = ix * ny + iy
                za = oz + dz * t0
                zb = oz + dz * t1
                # plancher : sol du terrain physique, évalué seulement près du sol (le reste du temps, le rayon est
                # au-dessus de la cellule : une lecture au lieu de deux interpolations)
                if min(za, zb) < H[ix, iy] + 0.3:
                    if not ga_ok:
                        ga = _height(ox + dx * t0, oy + dy * t0, H, lo0, lo1, res)
                    gb = _height(ox + dx * t1, oy + dy * t1, H, lo0, lo1, res)
                    ga_ok = True
                else:
                    gb = -1e30
                    ga_ok = False
                if za >= ga and zb < gb:
                    tg = t0 + (za - ga) / ((za - ga) - (zb - gb)) * (t1 - t0)
                    if tg < best and tg > min_range:
                        best = tg
                        bi = -1
                s = col_start[col]
                e = col_start[col + 1]
                if e > s:
                    zlo = min(za, zb)
                    zhi = max(za, zb)
                    klo = int(math.floor((zlo - z0) / res))
                    khi = int(math.floor((zhi - z0) / res))
                    a, b = s, e                                  # premier voxel de hauteur ≥ klo
                    while a < b:
                        m = (a + b) >> 1
                        if np.int64(vz[m]) < klo:
                            a = m + 1
                        else:
                            b = m
                    j = a
                    while j < e and np.int64(vz[j]) <= khi:
                        zv0 = z0 + vz[j] * res
                        if dz != 0:
                            tz0 = (zv0 - oz) / dz
                            tz1 = (zv0 + res - oz) / dz
                            tin = max(t0, min(tz0, tz1))
                            tout = min(t1, max(tz0, tz1))
                        else:
                            tin = t0
                            tout = t1
                        if tin <= tout:
                            cx = lo0 + (ix + 0.5) * res + cen[j, 0] * (half / 127.0)
                            cy = lo1 + (iy + 0.5) * res + cen[j, 1] * (half / 127.0)
                            cz = zv0 + half + cen[j, 2] * (half / 127.0)
                            qx, qy, qz = cx - ox, cy - oy, cz - oz
                            # point du rayon le plus proche du centre des points, borné au trajet dans le voxel
                            th = min(max(qx * dx + qy * dy + qz * dz, tin), tout)
                            if nrm[j, 0] != 0 or nrm[j, 1] != 0 or nrm[j, 2] != 0:
                                nxv, nyv, nzv = nrm[j, 0] / 127.0, nrm[j, 1] / 127.0, nrm[j, 2] / 127.0
                                den = dx * nxv + dy * nyv + dz * nzv
                                if abs(den) > 0.2:                   # plan local du voxel (surfel)
                                    tp = (qx * nxv + qy * nyv + qz * nzv) / den
                                    if tp > tin - res and tp < tout + res:
                                        th = tp
                            if th < best and th > min_range:
                                st = _xs(st)
                                if (st & np.uint64(255)) < np.uint64(phit[j]):
                                    best = th
                                    bi = j
                        j += 1
                if best <= t1 + res:
                    break
            elif (ix < 0 and sx < 0) or (ix >= nx and sx > 0) or (iy < 0 and sy < 0) or (iy >= ny and sy > 0):
                break                                            # sorti de la grille en s'éloignant
            else:
                ga_ok = False
            ga = gb
            if tmx < tmy:
                t0 = tmx
                tmx += tdx
                ix += sx
            else:
                t0 = tmy
                tmy += tdy
                iy += sy
        if best < max_range:
            st = _xs(st)
            u1 = (st & np.uint64(0xFFFF)) / 65536.0 + 1e-6
            u2 = ((st >> np.uint64(16)) & np.uint64(0xFFFF)) / 65536.0
            r = best + noise * math.sqrt(-2.0 * math.log(u1)) * math.cos(6.283185307 * u2)
            out_r[i] = r
            out_i[i] = ground_int if bi < 0 else inten[bi]
            out_p[i, 0] = ox + dx * r
            out_p[i, 1] = oy + dy * r
            out_p[i, 2] = oz + dz * r
        else:
            out_r[i] = np.nan
            out_i[i] = 0
            out_p[i, 0] = np.nan
            out_p[i, 1] = np.nan
            out_p[i, 2] = np.nan


@njit(cache=True, parallel=True, fastmath=True, nogil=True)
def cast_mu(O, Rm, D, lo0, lo1, res, nx, ny, nk, zb, col_start, vz, mu, inten, H, hlo0, hlo1, hres, max_range,
            min_range, noise, seed, ground_int, out_r, out_i, out_p, ntx, nty, kmask, tmask, tzmin, tzmax):
    """Comme `cast`, sur la carte d'OCCUPATION (lidar_occupancy.py) : colonnes de voxels qui suivent le sol (base zb),
    coefficient d'extinction μ par voxel (Beer-Lambert). Un seul seuil par rayon, E = −ln(u) : le rayon s'arrête là
    où l'épaisseur optique cumulée Σ μ·ℓ atteint E (même loi qu'un tirage par voxel).

    Parcours à DEUX niveaux : blocs de 8 × 8 colonnes (tuiles), sautés d'un coup quand le rayon y passe dans des
    tranches de hauteur vides et loin du sol (84 % des pas de colonne sur HÖB-1) ; colonnes de `res` sinon. Dans une
    colonne, le masque `kmask` évite la recherche des voxels quand les tranches traversées sont vides."""
    n = D.shape[0]
    ox, oy, oz = O[0], O[1], O[2]
    tres = 8 * res
    for i in prange(n):
        dx = Rm[0, 0] * D[i, 0] + Rm[0, 1] * D[i, 1] + Rm[0, 2] * D[i, 2]
        dy = Rm[1, 0] * D[i, 0] + Rm[1, 1] * D[i, 1] + Rm[1, 2] * D[i, 2]
        dz = Rm[2, 0] * D[i, 0] + Rm[2, 1] * D[i, 1] + Rm[2, 2] * D[i, 2]
        st = _xs(np.uint64(seed * 7477965 + i * 2891336453 + 88172645463325252))
        st = _xs(st)
        E = -math.log(((st >> np.uint64(11)) & np.uint64(0xFFFFFFFF)) / 4294967296.0 + 1e-12)
        tau = 0.0
        best = 1e30
        bi = 0
        ga = 0.0
        gb = 0.0
        ga_ok = False
        sx = 1 if dx > 0 else -1
        sy = 1 if dy > 0 else -1
        # --- niveau 1 : tuiles de 8 × 8 colonnes
        tx = int(math.floor((ox - lo0) / tres))
        ty = int(math.floor((oy - lo1) / tres))
        Tdx = tres / abs(dx) if dx != 0 else 1e30
        Tdy = tres / abs(dy) if dy != 0 else 1e30
        Tmx = ((lo0 + (tx + (1 if dx > 0 else 0)) * tres) - ox) / dx if dx != 0 else 1e30
        Tmy = ((lo1 + (ty + (1 if dy > 0 else 0)) * tres) - oy) / dy if dy != 0 else 1e30
        T0 = 0.0
        while T0 < max_range and best >= 1e29:
            T1 = min(Tmx, Tmy, max_range)
            if tx >= 0 and tx < ntx and ty >= 0 and ty < nty:
                tile = tx * nty + ty
                za = oz + dz * T0
                zc = oz + dz * T1
                zlo, zhi = min(za, zc), max(za, zc)
                near = zlo < tzmax[tile] + 1.3                   # sol proche (base = sol − 1 m)
                kl = int(math.floor((zlo - tzmax[tile]) / res))
                kh = int(math.floor((zhi - tzmin[tile]) / res))
                l0 = max(kl, 0) >> 3
                l1 = min(kh, nk - 1) >> 3
                occ_t = False
                if l1 >= l0 and l0 < 64:
                    w = min(l1, 63) - l0 + 1
                    bt = (tmask[tile] >> np.uint64(l0)) & ((np.uint64(1) << np.uint64(w)) - np.uint64(1)) \
                        if w < 64 else tmask[tile]
                    occ_t = bt != np.uint64(0)
                if near or occ_t:
                    # --- niveau 2 : colonnes de la tuile, de T0 à T1
                    te = T0 + 1e-7
                    ix = int(math.floor((ox + dx * te - lo0) / res))
                    iy = int(math.floor((oy + dy * te - lo1) / res))
                    ix = min(max(ix, tx * 8), tx * 8 + 7)
                    iy = min(max(iy, ty * 8), ty * 8 + 7)
                    tdx = res / abs(dx) if dx != 0 else 1e30
                    tdy = res / abs(dy) if dy != 0 else 1e30
                    tmx = ((lo0 + (ix + (1 if dx > 0 else 0)) * res) - ox) / dx if dx != 0 else 1e30
                    tmy = ((lo1 + (iy + (1 if dy > 0 else 0)) * res) - oy) / dy if dy != 0 else 1e30
                    t0 = T0
                    ga_ok = False
                    while t0 < T1:
                        t1 = min(tmx, tmy, T1)
                        if ix >= 0 and ix < nx and iy >= 0 and iy < ny:
                            col = (((ix >> 3) * nty + (iy >> 3)) << 6) + ((ix & 7) << 3) + (iy & 7)
                            za = oz + dz * t0
                            zc = oz + dz * t1
                            base = zb[col]
                            tg = 1e30
                            if min(za, zc) < base + 1.3:             # plancher : sol du terrain physique
                                if not ga_ok:
                                    ga = _height(ox + dx * t0, oy + dy * t0, H, hlo0, hlo1, hres)
                                gb = _height(ox + dx * t1, oy + dy * t1, H, hlo0, hlo1, hres)
                                ga_ok = True
                                if za >= ga and zc < gb:
                                    tg = t0 + (za - ga) / ((za - ga) - (zc - gb)) * (t1 - t0)
                            else:
                                ga_ok = False
                            klo = int(math.floor((min(za, zc) - base) / res))
                            khi = int(math.floor((max(za, zc) - base) / res))
                            c0 = max(klo, 0) >> 3
                            c1 = min(khi, nk - 1) >> 3
                            occ = False
                            if c1 >= c0 and c0 < 64:
                                w = min(c1, 63) - c0 + 1
                                bits = (kmask[col] >> np.uint64(c0)) & ((np.uint64(1) << np.uint64(w)) - np.uint64(1)) \
                                    if w < 64 else kmask[col]
                                occ = bits != np.uint64(0)
                            s = col_start[col]
                            e = col_start[col + 1]
                            if occ and e > s:
                                a, b = s, e                          # premier voxel de hauteur ≥ klo
                                while a < b:
                                    m = (a + b) >> 1
                                    if np.int64(vz[m]) < klo:
                                        a = m + 1
                                    else:
                                        b = m
                                j0 = a
                                a, b = j0, e                         # premier voxel de hauteur > khi
                                while a < b:
                                    m = (a + b) >> 1
                                    if np.int64(vz[m]) <= khi:
                                        a = m + 1
                                    else:
                                        b = m
                                j1 = a
                                for q in range(j1 - j0):
                                    j = j0 + q if dz >= 0 else j1 - 1 - q    # ordre du rayon
                                    zv0 = base + np.int64(vz[j]) * res
                                    if dz != 0:
                                        tz0 = (zv0 - oz) / dz
                                        tz1 = (zv0 + res - oz) / dz
                                        tin = max(t0, min(tz0, tz1))
                                        tout = min(t1, max(tz0, tz1))
                                    else:
                                        tin = t0
                                        tout = t1
                                    if tin >= tg:
                                        break
                                    if tout > tin:
                                        m_ = np.float64(mu[j])
                                        dtau = m_ * (min(tout, tg) - tin)
                                        if tau + dtau >= E:
                                            ts = tin + (E - tau) / m_
                                            if ts > min_range:
                                                best = ts
                                                bi = j
                                                break
                                        tau += dtau
                            if best >= 1e29 and tg < 1e29 and tg > min_range:
                                best = tg
                                bi = -1
                            if best < 1e29:
                                break
                        ga = gb if ga_ok else ga
                        if tmx < tmy:
                            t0 = tmx
                            tmx += tdx
                            ix += sx
                        else:
                            t0 = tmy
                            tmy += tdy
                            iy += sy
            elif (tx < 0 and sx < 0) or (tx >= ntx and sx > 0) or (ty < 0 and sy < 0) or (ty >= nty and sy > 0):
                break
            if Tmx < Tmy:
                T0 = Tmx
                Tmx += Tdx
                tx += sx
            else:
                T0 = Tmy
                Tmy += Tdy
                ty += sy
        if best < max_range:
            st = _xs(st)
            u1 = (st & np.uint64(0xFFFF)) / 65536.0 + 1e-6
            u2 = ((st >> np.uint64(16)) & np.uint64(0xFFFF)) / 65536.0
            r = best + noise * math.sqrt(-2.0 * math.log(u1)) * math.cos(6.283185307 * u2)
            out_r[i] = r
            out_i[i] = ground_int if bi < 0 else inten[bi]
            out_p[i, 0] = ox + dx * r
            out_p[i, 1] = oy + dy * r
            out_p[i, 2] = oz + dz * r
        else:
            out_r[i] = np.nan
            out_i[i] = 0
            out_p[i, 0] = np.nan
            out_p[i, 1] = np.nan
            out_p[i, 2] = np.nan


class OccupancyMap:
    """Carte d'occupation (lidar_occupancy.py) : voxels de 5 cm, coefficient d'extinction mesuré."""

    def __init__(self, path, terrain, prior=None, floor_offset=0.0, min_opacity=0.01):
        """prior : longueur a priori (m) pour recalculer μ = touchés / (longueur + prior) depuis les comptes bruts ;
        floor_offset : décalage vertical du plancher (sol du terrain), négatif = plus bas ;
        min_opacity : voxels plus transparents ignorés (HÖB-1, < 1 % : 10 % des voxels, 0,2 % de l'épaisseur optique ;
        ils remplissaient les masques et forçaient le parcours fin)."""
        V = np.load(path)
        self.lo = V["lo"].astype(np.float64)
        self.res = float(V["res"])
        self.nx, self.ny, self.nk = int(V["nx"]), int(V["ny"]), int(V["nk"])
        if prior is not None and "hits" in V:
            mu = (V["hits"].astype(np.float32) / (V["length"] + np.float32(prior))).astype(np.float32)
        else:
            mu = np.asarray(V["mu"], np.float32)
        # réordonne les colonnes par tuiles de 8 × 8 (accès mémoire groupés pendant le parcours des rayons : un rayon
        # qui avance en x sautait ny colonnes en mémoire à chaque pas)
        nx, ny = self.nx, self.ny
        self.ntx, self.nty = (nx + 7) // 8, (ny + 7) // 8
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        tcol = (((ix >> 3) * self.nty + (iy >> 3)) << 6) + ((ix & 7) << 3) + (iy & 7)     # (nx, ny) → index tuilé
        ncol_t = self.ntx * self.nty * 64
        cs = V["col_start"].astype(np.int64)
        vz, inten = np.asarray(V["vz"]), np.asarray(V["inten"])
        if min_opacity > 0:
            keep = (1 - np.exp(-mu * self.res)) >= min_opacity
            col_of = np.repeat(np.arange(len(cs) - 1), np.diff(cs))[keep]
            vz, inten, mu = vz[keep], inten[keep], mu[keep]
            cs = np.concatenate([[0], np.cumsum(np.bincount(col_of, minlength=len(cs) - 1))])
        cnt = np.zeros(ncol_t, np.int64)
        cnt[tcol.ravel()] = np.diff(cs)
        self.col_start = np.concatenate([[0], np.cumsum(cnt)])
        order = np.argsort(tcol.ravel(), kind="stable")                  # colonnes d'origine dans l'ordre tuilé
        starts, lens = cs[:-1][order], np.diff(cs)[order]
        idx = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens) + np.arange(lens.sum())
        self.vz = np.ascontiguousarray(vz[idx], np.uint16)
        self.mu = np.ascontiguousarray(mu[idx], np.float32)
        self.inten = np.ascontiguousarray(inten[idx], np.uint8)
        zb = np.zeros(ncol_t, np.float32)
        zb[tcol.ravel()] = V["zb"].ravel()
        self.zb = zb
        # masque par colonne des tranches de 8 voxels occupées (bit l = au moins un voxel de hauteur k avec k >> 3 = l)
        col_of = np.repeat(np.arange(ncol_t), np.diff(self.col_start))
        self.kmask = np.zeros(ncol_t, np.uint64)
        np.bitwise_or.at(self.kmask, col_of, np.left_shift(np.uint64(1), np.minimum(self.vz >> 3, 63).astype(np.uint64)))
        # par tuile (8 × 8 colonnes) : union des masques, bases z min / max (parcours à deux niveaux)
        self.tmask = np.bitwise_or.reduce(self.kmask.reshape(-1, 64), axis=1)
        self.tzmin = zb.reshape(-1, 64).min(1).astype(np.float32)
        self.tzmax = zb.reshape(-1, 64).max(1).astype(np.float32)
        self.H = np.ascontiguousarray(terrain["H"], np.float64) + floor_offset
        self.hlo = np.asarray(terrain["lo"], np.float64)
        self.hres = float(terrain["res"])

    def cast(self, O, D, max_range=60.0, min_range=0.3, noise=0.02, seed=0, ground_int=60, R=None, out=None):
        if out is None:
            out = (np.empty(len(D), np.float32), np.empty(len(D), np.uint8), np.empty((len(D), 3), np.float32))
        cast_mu(np.asarray(O, np.float64), np.eye(3) if R is None else np.ascontiguousarray(R, np.float64),
                np.ascontiguousarray(D, np.float64), self.lo[0], self.lo[1], self.res, self.nx, self.ny, self.nk,
                self.zb, self.col_start, self.vz, self.mu, self.inten, self.H, self.hlo[0], self.hlo[1], self.hres,
                max_range, min_range, noise, seed, ground_int, *out, self.ntx, self.nty, self.kmask, self.tmask,
                self.tzmin, self.tzmax)
        return out[0], out[1]


def load_map(scene_dir, terrain, name="lidar_occupation.npz"):
    """Carte d'occupation mesurée `name` si elle existe (par défaut : Hesai + comblement Livox / Velodyne ;
    `lidar_occupation_hesai.npz` : Hesai seul, ~30 % plus rapide), sinon l'ancienne carte de voxels."""
    import os

    if os.path.exists(f"{scene_dir}/{name}"):
        V = np.load(f"{scene_dir}/{name}")
        if "mu" in V:
            # plancher abaissé de 5 cm : le sol du terrain est la MÉDIANE de la couche de sol, les rayons rasants le
            # touchaient trop tôt (validation sur balayages mis de côté : 36 % → 32 % de « trop tôt »)
            return OccupancyMap(f"{scene_dir}/{name}", terrain, floor_offset=-0.05)
    return VoxelMap(f"{scene_dir}/lidar_voxels.npz", terrain)


class VoxelMap:
    def __init__(self, path, terrain):
        V = np.load(path)
        self.lo = V["lo"].astype(np.float64)
        self.res = float(V["res"])
        self.nx, self.ny = int(V["nx"]), int(V["ny"])
        self.z0 = float(V["z0"])
        self.col_start = V["col_start"].astype(np.int64)
        # entiers compacts (uint16 / int8 / uint8) : 10 octets par voxel, ~60 Mo pour HÖB-1 (Mac 8 Go)
        self.vz = np.ascontiguousarray(V["vz"], np.uint16)
        self.cen = np.ascontiguousarray(V["cen"], np.int8)
        self.nrm = np.ascontiguousarray(V["nrm"], np.int8)
        self.phit = np.ascontiguousarray(V["phit"], np.uint8)
        self.inten = np.ascontiguousarray(V["inten"], np.uint8)
        self.H = np.ascontiguousarray(terrain["H"], np.float64)
        assert abs(float(terrain["res"]) - self.res) < 1e-9 and np.allclose(terrain["lo"], self.lo), \
            "carte de voxels et terrain sur des grilles différentes"

    def cast(self, O, D, max_range=60.0, min_range=0.3, noise=0.02, seed=0, ground_int=60, R=None, out=None):
        """Directions D dans le repère physique (ou dans celui du capteur si R est donné). `out` = (r, i, p) : tampons
        à remplir sans allocation (vues contiguës)."""
        if out is None:
            out = (np.empty(len(D), np.float32), np.empty(len(D), np.uint8), np.empty((len(D), 3), np.float32))
        cast(np.asarray(O, np.float64), np.eye(3) if R is None else np.ascontiguousarray(R, np.float64),
             np.ascontiguousarray(D, np.float64), self.lo[0], self.lo[1], self.res, self.nx, self.ny, self.z0,
             self.col_start, self.vz, self.cen, self.nrm, self.phit, self.inten, self.H, max_range, min_range, noise,
             seed, ground_int, *out)
        return out[0], out[1]


class Lidar:
    """LiDAR rotatif monté sur le robot (repère châssis : x avant, y gauche, z haut). Par défaut 32 lignes de −16° à
    +15° (gabarit Hesai XT32), 1 024 colonnes, 10 tours/s, portée 60 m (la forêt masque au-delà)."""

    def __init__(self, vmap, rings=32, el_min=-16.0, el_max=15.0, cols=1024, rate=10.0, max_range=60.0, noise=0.02,
                 mount=(0.0, 0.0, 0.3)):
        self.vmap, self.rings, self.cols, self.rate = vmap, rings, cols, rate
        self.max_range, self.noise = max_range, noise
        self.mount = np.asarray(mount, float)
        el = np.radians(np.linspace(el_max, el_min, rings))        # ligne 0 = la plus haute (haut de l'image)
        az = -np.arange(cols) * 2 * np.pi / cols                   # rotation horaire vue de dessus, colonne 0 = avant
        E, A = np.meshgrid(el, az, indexing="ij")
        self.dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)   # (rings, cols, 3)
        # rangées par colonne : une tranche de colonnes = un bloc contigu (ni copie ni indexation à chaque appel)
        self.dirs_cr = np.ascontiguousarray(self.dirs.transpose(1, 0, 2).reshape(-1, 3))
        self.cursor = 0.0
        self.done_cols = 0                                         # colonnes déjà tirées (curseur entier)
        self.seed = 0
        self.reset_scan()

    def reset_scan(self):
        n = self.rings * self.cols                                 # tampons rangés par colonne (cols, rings)
        self.r_cr = np.full(n, np.nan, np.float32)
        self.i_cr = np.zeros(n, np.uint8)
        self.p_cr = np.full((n, 3), np.nan, np.float32)

    def advance(self, dt, pos, R, min_slice=1 / 16):
        """Tire les colonnes balayées pendant `dt` depuis la pose (pos, R : châssis → physique). → balayage complet
        (dict) quand le tour se termine, sinon None. Les colonnes sont tirées par tranches d'au moins `min_slice` tour
        (6 ms à 10 Hz, ~1 cm de déplacement à 2 m/s) : chaque appel numba coûte un temps fixe, et 25 petits appels par
        tour faisaient passer un tour de 7 à 12 ms."""
        c0 = int(self.done_cols)
        self.cursor += dt * self.rate * self.cols
        c1 = int(self.cursor)
        if c1 - c0 < min_slice * self.cols and c1 // self.cols == c0 // self.cols:
            return None
        self.done_cols = c1
        done = None
        O = pos + R @ self.mount
        while c0 < c1:
            ce = min(c1, (c0 // self.cols + 1) * self.cols)
            a, b = (c0 % self.cols) * self.rings, ((ce - 1) % self.cols + 1) * self.rings
            self.seed += 1
            self.vmap.cast(O, self.dirs_cr[a:b], self.max_range, noise=self.noise, seed=self.seed, R=R,
                           out=(self.r_cr[a:b], self.i_cr[a:b], self.p_cr[a:b]))
            if ce % self.cols == 0:                                 # tour terminé → (rings, cols)
                done = {"r": self.r_cr.reshape(self.cols, self.rings).T.copy(),
                        "i": self.i_cr.reshape(self.cols, self.rings).T.copy(),
                        "pts": self.p_cr.reshape(self.cols, self.rings, 3).transpose(1, 0, 2).copy(),
                        "origin": O.astype(np.float32)}
                self.reset_scan()
            c0 = ce
        if self.done_cols >= 1 << 40:                               # recentrage (jamais atteint en pratique)
            k = (self.done_cols // self.cols) * self.cols
            self.done_cols -= k
            self.cursor -= k
        return done
