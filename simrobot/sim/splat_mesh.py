"""Géométrie de collision tirée des gaussiennes étiquetées (exploration, desktop) — troncs inclinés et sol.

- **Troncs** (`axes_troncs`) : pour chaque tronc détecté au LiDAR (`terrain.npz`, sources 0 et 1), les gaussiennes
  « tree » voisines sont coupées en tranches de 25 cm le long de l'axe courant. Dans chaque tranche, on ajuste un
  cercle de rayon FIXÉ (celui du LiDAR) : un arc partiel, seul côté vu des caméras, ne biaise pas alors le centre.
  La tranche est gardée si ≥ 25 % de ses points sont à moins de 4 cm du cercle (branches, baliveaux et troncs voisins
  sont aussi « tree » : l'erreur sur tous les points vaut 16 cm en médiane et rejetait 3 tranches sur 4).
  Une droite est ajustée sur les centres des tranches (moindres carrés pondérés, Huber), et l'opération est répétée
  trois fois. Résultat : axe incliné + segments de capsule qui suivent les centres (tronc courbe).
  Le même ajustement appliqué aux points de la carte LiDAR DLIO sert de comparaison.
- **Troncs couchés** (`troncs_couches`) : composantes connexes (voxels de 10 cm) des gaussiennes « tree » de 5 cm à
  1 m au-dessus du sol, loin des troncs debout. Une composante allongée (ACP : λ1 ≥ 6 λ2, longueur ≥ 0,8 m) devient
  une capsule couchée posée au sol, de diamètre = hauteur du dessus vu (≤ 60 cm, au-delà : fourré de baliveaux).
  Beaucoup de troncs couchés (< 30 cm de haut) n'avaient AUCUNE géométrie : la détection LiDAR d'obstacles bas
  demande des points de 0,3 à 1,2 m. Les autres étaient une chaîne de cylindres verticaux de 2 m.
- **Sol** (`sol_gaussiennes`) : par cellule de 5 cm, médiane pondérée (opacité × visibilité) de la hauteur des
  gaussiennes étiquetées sol (earth, path, road, dirt track, water, grass, rock), à ±40 cm du sol LiDAR. Les troncs,
  les buissons et les flottants sont exclus par leur étiquette.

Évaluation (`evaluer`) sur de vrais balayages Hesai mis de côté à la construction de la carte d'occupation :
- troncs : écart radial |distance à l'axe − r| des points LiDAR autour de chaque tronc, de 0,3 à 3 m ;
- sol : lancer des vrais rayons dont l'impact est au sol (gaussienne la plus proche, à moins de 10 cm, étiquetée sol)
  dans le champ de hauteur de chaque modèle ; erreur verticale = (distance mesurée − distance prédite) × sin(plongée)
  (> 0 : le modèle est trop haut).

  cd ~/dino && PYTHONPATH=. venv/bin/python simrobot/sim/splat_mesh.py --scene out/simrobot/scenes/HOB-1_full \
      --labels out/simrobot/labels/HOB-1_full/gaussiennes.npz --dlio data_local/simrobot/grandtour/HOB-1/lidar \
      --scans /mnt/data/dino/gt_lidar/... --odometry ...
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

TREE = 4
GROUND = (13, 52, 6, 91, 21, 9, 34)


def load_gaussians(scene, labels, meta):
    with open(f"{scene}/merged.ply", "rb") as f:
        props, n = [], 0
        while True:
            s = f.readline().decode().strip()
            if s.startswith("element vertex"):
                n = int(s.split()[-1])
            elif s.startswith("property"):
                props.append((s.split()[-1], "<f4"))
            elif s == "end_header":
                break
        V = np.fromfile(f, dtype=np.dtype(props), count=n)
    P = np.stack([V["x"], V["y"], V["z"]], 1).astype(np.float64) @ np.asarray(meta["ply_to_physics"]).T
    op = (1 / (1 + np.exp(-V["opacity"]))).astype(np.float32)
    sc = np.exp(np.stack([V[f"scale_{i}"] for i in range(3)], 1)).max(1).astype(np.float32)
    L = np.load(labels)
    return P, op, sc, L["label"], L["weight"].astype(np.float32)


# ------------------------------------------------------------------------------------------------ troncs inclinés

def circle_fixed_r(xy, w, c0, r, iters=10):
    """Centre d'un cercle de rayon fixé r ajusté sur des points (arc partiel possible) ; Gauss-Newton pondéré + Huber."""
    c = np.array(c0, float)
    for _ in range(iters):
        d = xy - c
        dist = np.hypot(d[:, 0], d[:, 1]) + 1e-9
        res = dist - r
        hub = np.minimum(1.0, 0.03 / np.maximum(np.abs(res), 1e-9))
        J = -d / dist[:, None]
        ww = w * hub
        A = (J * ww[:, None]).T @ J + 1e-6 * np.eye(2)
        step = np.linalg.solve(A, -(J * ww[:, None]).T @ res)
        c += np.clip(step, -0.1, 0.1)
        if np.hypot(*step) < 1e-4:
            break
    d = xy - c
    res = np.hypot(d[:, 0], d[:, 1]) - r
    return c, float(np.sqrt(np.average(res ** 2, weights=w)))


def fit_axis(pts, w, seed, r, z0, slice_h=0.25, h_rng=(0.3, 3.3), min_pts=12, iters=3):
    """Axe d'un tronc à partir de points voisins (gaussiennes ou LiDAR). Renvoie dict ou None.
    seed : (x, y) du tronc vertical LiDAR ; z0 : sol sous le tronc."""
    base = np.array([seed[0], seed[1], z0])
    u = np.array([0.0, 0.0, 1.0])
    out = None
    for it in range(iters):
        rel = pts - base
        t = rel @ u                                              # abscisse le long de l'axe (≈ hauteur)
        perp = rel - t[:, None] * u
        dist = np.linalg.norm(perp, axis=1)
        keep = (t > h_rng[0]) & (t < h_rng[1]) & (dist < r + (0.6 if it == 0 else 0.25))
        if keep.sum() < 3 * min_pts:
            return out
        # repère orthonormé perpendiculaire à u
        e1 = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        cen, ts, ws, rms = [], [], [], []
        for k in range(int(h_rng[0] / slice_h), int(np.ceil(h_rng[1] / slice_h))):
            m = keep & (t >= k * slice_h) & (t < (k + 1) * slice_h)
            if m.sum() < min_pts:
                continue
            xy = np.stack([perp[m] @ e1, perp[m] @ e2], 1)
            # l'arc vu doit couvrir au moins ~60° pour contraindre le centre
            ang = np.sort(np.arctan2(xy[:, 1], xy[:, 0]))
            arc = 2 * np.pi - np.max(np.diff(np.r_[ang, ang[0] + 2 * np.pi]))
            if arc < np.radians(60):
                continue
            c, _ = circle_fixed_r(xy, w[m], (0.0, 0.0), r)
            # jugé sur les points proches du cercle (branches, baliveaux et troncs voisins sont aussi « tree »)
            res = np.abs(np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]) - r)
            inl = res < 0.04
            if np.hypot(*c) > r + 0.5 or inl.sum() < min_pts or inl.mean() < 0.25:
                continue
            e = float(np.sqrt(np.mean(res[inl] ** 2)))
            cen.append(base + c[0] * e1 + c[1] * e2 + ((k + 0.5) * slice_h) * u)
            ts.append((k + 0.5) * slice_h)
            ws.append(float(np.sqrt(inl.sum())))
            rms.append(e)
        if len(cen) < 4:
            return out
        C, ws = np.array(cen), np.array(ws)
        # droite pondérée : x, y affines en z (troncs à moins de ~45° de la verticale)
        A = np.stack([np.ones(len(C)), C[:, 2]], 1)
        W = np.diag(ws)
        coef = np.linalg.lstsq(np.sqrt(W) @ A, np.sqrt(W) @ C[:, :2], rcond=None)[0]   # (2, 2)
        pred = A @ coef
        resid = np.linalg.norm(C[:, :2] - pred, axis=1)
        hub = np.minimum(1.0, 0.05 / np.maximum(resid, 1e-9))
        coef = np.linalg.lstsq(np.sqrt(np.diag(ws * hub)) @ A, np.sqrt(np.diag(ws * hub)) @ C[:, :2], rcond=None)[0]
        d = np.array([coef[1, 0], coef[1, 1], 1.0])
        u = d / np.linalg.norm(d)
        base = np.array([coef[0, 0] + coef[1, 0] * z0, coef[0, 1] + coef[1, 1] * z0, z0])
        out = {"base": base, "u": u, "tilt_deg": float(np.degrees(np.arccos(u[2]))), "n_tranches": len(C),
               "centres": C, "rms_cercle": float(np.median(rms)), "h_max": float(max(ts) + slice_h / 2)}
    return out


def axes_troncs(trunks, H_at, pts, w, max_tilt=35.0):
    """trunks : (n, 4) x, y, r, z ; pts/w : nuage et poids. → liste d'ajustements (None si échec)."""
    from scipy.spatial import cKDTree

    kd = cKDTree(pts[:, :2])
    fits = []
    for x, y, r, z in trunks:
        idx = kd.query_ball_point([x, y], r + 1.6)
        if len(idx) < 40:
            fits.append(None)
            continue
        idx = np.asarray(idx)
        z0 = float(H_at(np.array([[x, y]]))[0])
        f = fit_axis(pts[idx], w[idx], (x, y), float(r), z0)
        if f is not None and f["tilt_deg"] > max_tilt:
            f = None
        fits.append(f)
    return fits


# ------------------------------------------------------------------------------------------------ troncs couchés

def troncs_couches(P, h, op, sc, lab, standing, H_at, vox=0.1, min_len=0.8, lin=6.0, max_top=0.6):
    """Capsules couchées depuis les gaussiennes « tree » basses éloignées des troncs debout."""
    from scipy import ndimage
    from scipy.spatial import cKDTree

    m = (lab == TREE) & (h > 0.05) & (h < 1.0) & (op > 0.3) & (sc < 0.15)
    if len(standing):
        d, j = cKDTree(standing[:, :2]).query(P[m, :2])
        far = d > standing[j, 2] + 0.35
        idx = np.where(m)[0][far]
    else:
        idx = np.where(m)[0]
    Q = P[idx]
    hh = h
    lo = Q.min(0)
    ijk = np.floor((Q - lo) / vox).astype(np.int64)
    shape = ijk.max(0) + 1
    occ = np.zeros(shape, bool)
    cnt = np.zeros(shape, np.int32)
    np.add.at(cnt, tuple(ijk.T), 1)
    occ = cnt >= 3
    labs, nlab = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    comp = labs[tuple(ijk.T)]
    caps = []
    order = np.argsort(comp)
    bounds = np.searchsorted(comp[order], np.arange(1, nlab + 2))
    for c in range(1, nlab + 1):
        sel = order[bounds[c - 1]:bounds[c]]
        if len(sel) < 60:
            continue
        X = Q[sel]
        mu = X.mean(0)
        ev, evec = np.linalg.eigh(np.cov((X - mu).T))
        d = evec[:, 2]
        if ev[2] < lin * ev[1] or abs(d[2]) > 0.5:                # allongé et plutôt couché (< 30°)
            continue
        t = (X - mu) @ d
        t0, t1 = np.percentile(t, [2, 98])
        if t1 - t0 < min_len:
            continue
        # tronc couché posé au sol : diamètre = hauteur du dessus vu (q90 des gaussiennes au-dessus du sol LiDAR),
        # axe à un rayon au-dessus du sol à chaque bout. Le centre des gaussiennes, sur la face vue, est trop haut.
        top = float(np.percentile(hh[idx[sel]], 90))
        if top > max_top:                                         # fourré de baliveaux, pas un tronc couché
            continue
        rad = float(max(top / 2, 0.05))
        a, b = mu + t0 * d, mu + t1 * d
        a[2] = H_at(a[None, :2])[0] + rad
        b[2] = H_at(b[None, :2])[0] + rad
        caps.append({"a": a, "b": b, "r": rad, "n": int(len(sel)), "long_m": float(t1 - t0)})
    return caps


# ------------------------------------------------------------------------------------------------ sol

def sol_points(Pts, T, res=0.05, min_n=3):
    """Champ de hauteur : médiane des points par cellule (cellules de moins de min_n points : NaN)."""
    lo = np.asarray(T["lo"], float)
    nx = int(round(T["H"].shape[0] * float(T["res"]) / res))
    ny = int(round(T["H"].shape[1] * float(T["res"]) / res))
    ix = np.floor((Pts[:, 0] - lo[0]) / res).astype(np.int64)
    iy = np.floor((Pts[:, 1] - lo[1]) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    key, z = ix[ok] * ny + iy[ok], Pts[ok, 2]
    o = np.lexsort((z, key))
    key, z = key[o], z[o]
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    cnt = np.diff(np.r_[starts, len(key)])
    Hm = np.full(nx * ny, np.nan, np.float32)
    good = cnt >= min_n
    Hm[key[starts[good]]] = z[starts[good] + cnt[good] // 2]
    return Hm.reshape(nx, ny), res


def sol_gaussiennes(P, h, op, w, lab, T, res=0.05, band=0.4, min_w=5.0):
    """Champ de hauteur 5 cm : médiane pondérée des gaussiennes étiquetées sol, à ±band du sol LiDAR."""
    lo = np.asarray(T["lo"], float)
    nx = int(round(T["H"].shape[0] * float(T["res"]) / res))
    ny = int(round(T["H"].shape[1] * float(T["res"]) / res))
    m = np.isin(lab, GROUND) & (np.abs(h) < band) & (op > 0.2)
    ix = np.floor((P[m, 0] - lo[0]) / res).astype(np.int64)
    iy = np.floor((P[m, 1] - lo[1]) / res).astype(np.int64)
    z = P[m, 2]
    ww = (op[m] * w[m]).astype(np.float64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    key = ix[ok] * ny + iy[ok]
    z, ww = z[ok], ww[ok]
    o = np.lexsort((z, key))
    key, z, ww = key[o], z[o], ww[o]
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    csum = np.cumsum(ww)
    tot = np.add.reduceat(ww, starts)
    before = np.r_[0, csum[starts[1:] - 1]]
    half = before + tot / 2
    # médiane pondérée : premier élément dont la somme cumulée atteint la moitié du poids de sa cellule
    pos = np.searchsorted(csum, half)
    Hs = np.full(nx * ny, np.nan, np.float32)
    good = tot >= min_w
    Hs[key[starts[good]]] = z[np.minimum(pos[good], len(z) - 1)]
    return Hs.reshape(nx, ny), res


# ------------------------------------------------------------------------------------------------ évaluation

def held_out_points(scene, scans, odometry, n=40, max_range=40.0):
    from gaussgym_export import FLIP
    from lidar_occupancy import TarZarr
    from lidar_terrain import quat_xyzw_to_R, zarr_group

    meta = json.load(open(f"{scene}/scene.json"))
    centre = np.array(meta["centre_monde_m"])
    Rr, tr = np.array(meta["recalage_lidar"]["R"]), np.array(meta["recalage_lidar"]["t"])
    odo = zarr_group(odometry, "dlio_map_odometry")
    po, qo = odo("pose_pos").astype(float), odo("pose_orien").astype(float)
    Ro = quat_xyzw_to_R(qo / np.linalg.norm(qo, axis=1, keepdims=True))
    held = np.load(f"{scene}/lidar_occupation.npz")["held_scans"]
    held = held[np.linspace(0, len(held) - 1, min(n, len(held))).astype(int)]
    Z = TarZarr(scans, "dlio_hesai_points_undistorted")
    to_phys = lambda x: (x @ Rr.T + tr) @ FLIP.T - centre                 # noqa: E731
    O, Pts = [], []
    for k in held:
        pts = Z.row("points", k).reshape(-1, 3).astype(np.float64)
        nv = int(np.asarray(Z.row("valid", k)).reshape(-1)[0])
        pts = pts[:nv]
        pts = pts[np.isfinite(pts).all(1) & (np.abs(pts).sum(1) > 0)]
        P = to_phys(pts @ Ro[k].T + po[k])
        o = to_phys(po[k][None])[0]
        r = np.linalg.norm(P - o, axis=1)
        keep = (r > 0.5) & (r < max_range)
        Pts.append(P[keep])
        O.append(np.repeat(o[None], keep.sum(), 0))
    return np.concatenate(O), np.concatenate(Pts)


def radial_residual(Pn, base, u, r, h_rng=(0.3, 3.0)):
    rel = Pn - base
    t = rel @ u
    d = np.linalg.norm(rel - np.outer(t, u), axis=1)
    m = (t > h_rng[0]) & (t < h_rng[1]) & (d < r + 0.3)
    return np.abs(d[m] - r)


def cast_heightfield(O, D, Hgrid, lo, res, max_t=40.0, step=0.02):
    """Premier franchissement d'un champ de hauteur par des rayons (marche à pas fixe + interpolation)."""
    nx, ny = Hgrid.shape
    n = len(O)
    t = np.full(n, np.nan)
    prev = None
    active = np.ones(n, bool)
    for s in np.arange(0.3, max_t, step):
        p = O + s * D
        i = np.clip(((p[:, 0] - lo[0]) / res).astype(np.int64), 0, nx - 1)
        j = np.clip(((p[:, 1] - lo[1]) / res).astype(np.int64), 0, ny - 1)
        hz = Hgrid[i, j]
        below = p[:, 2] <= hz
        hit = active & below & np.isfinite(hz)
        if prev is not None:
            # interpolation linéaire entre le pas précédent (au-dessus) et celui-ci (dessous)
            a0 = prev[hit]
            a1 = (p[hit, 2] - hz[hit])
            t[hit] = s - step * a1 / np.where(np.abs(a1 - a0) > 1e-9, a1 - a0, 1e-9)
        else:
            t[hit] = s
        active &= ~hit
        prev = p[:, 2] - hz
        if not active.any():
            break
    return t


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "planche":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--scene", required=True)
        ap.add_argument("--labels", required=True)
        ap.add_argument("--mesh", required=True)
        ap.add_argument("--out", required=True)
        a = ap.parse_args()
        return planche(a.scene, a.labels, a.mesh, a.out)
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--dlio", required=True, help="dossier contenant la carte DLIO (*_dlio.ply)")
    ap.add_argument("--scans", required=True)
    ap.add_argument("--odometry", required=True)
    ap.add_argument("--n-scans", type=int, default=40)
    ap.add_argument("--out", required=True, help="dossier des résultats")
    a = ap.parse_args()
    from scipy.spatial import cKDTree

    from gaussgym_export import FLIP
    from lidar_voxels import read_ply_xyzi
    from terrain import height_at

    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(f"{a.scene}/scene.json"))
    T = dict(np.load(f"{a.scene}/terrain.npz"))
    H_at = lambda xy: height_at(T, xy)                                     # noqa: E731
    P, op, sc, lab, w = load_gaussians(a.scene, a.labels, meta)
    h = P[:, 2] - H_at(P[:, :2])
    path = np.array(json.load(open(f"{a.scene}/trajets.json"))["dlio"]["p"])[:, :2]
    kd_path = cKDTree(path)
    tr, src = T["troncs"], T["troncs_source"]
    trunks = tr[src < 2]
    dpath = kd_path.query(trunks[:, :2])[0]
    near = dpath < 12.0                                                    # zone reconstruite par le splat
    trunks = trunks[near]
    trunks_idx = np.where(src < 2)[0][near]                                # indices dans terrain.npz
    print(f"{len(trunks)} troncs à moins de 12 m du trajet", flush=True)

    # --- troncs couchés (position et étendue depuis les gaussiennes « tree »)
    standing = tr[src < 2]
    caps = troncs_couches(P, h, op, sc, lab, standing, H_at)
    # --- axes depuis les gaussiennes « tree »
    g = (lab == TREE) & (op > 0.3) & (sc < 0.12) & (h > 0.2) & (h < 3.6)
    fits_g = axes_troncs(trunks, H_at, P[g], (op[g] * np.sqrt(w[g])).astype(np.float64))
    near_g = (np.abs(h) < 0.6) & (op > 0.2)
    kd_g = cKDTree(P[near_g])
    lab_g = lab[near_g]
    # --- mêmes axes depuis la carte LiDAR DLIO (comparaison)
    centre = np.array(meta["centre_monde_m"])
    Rr, trl = np.array(meta["recalage_lidar"]["R"]), np.array(meta["recalage_lidar"]["t"])
    Pl, _ = read_ply_xyzi(glob.glob(f"{a.dlio}/*_dlio.ply")[0])
    Pl = (Pl @ Rr.T + trl) @ FLIP.T - centre
    hl = Pl[:, 2] - H_at(Pl[:, :2])
    ml = (hl > 0.2) & (hl < 3.6)
    fits_l = axes_troncs(trunks, H_at, Pl[ml], np.ones(int(ml.sum())))
    # sol LiDAR 5 cm conditionné par les étiquettes : points de la carte DLIO dont la gaussienne la plus proche (≤ 10 cm)
    # est étiquetée sol, médiane par cellule de 5 cm dans ±15 cm du sol LiDAR 10 cm
    mg = np.abs(hl) < 0.15
    dd, jj = kd_g.query(Pl[mg], distance_upper_bound=0.1)
    okg = np.isfinite(dd)
    okg[okg] = np.isin(lab_g[jj[okg]], GROUND)
    G5 = Pl[mg][okg]
    Hl5, _ = sol_points(G5, T, res=0.05)
    print(f"sol LiDAR étiqueté : {okg.mean():.1%} des points de la couche de sol gardés, "
          f"{np.isfinite(Hl5).mean():.1%} des cellules de 5 cm", flush=True)
    # hauteur des troncs couchés depuis la carte LiDAR : les surfaces du splat sont ~10 cm trop hautes (cf. sol)
    kl = cKDTree(Pl[(hl > -0.05) & (hl < 0.8), :2])
    Pl_low = Pl[(hl > -0.05) & (hl < 0.8)]
    hl_low = hl[(hl > -0.05) & (hl < 0.8)]
    caps_l = []
    for c in caps:
        a_, b_ = c["a"].copy(), c["b"].copy()
        ab = b_[:2] - a_[:2]
        idx = np.asarray(kl.query_ball_point(((a_ + b_) / 2)[:2], np.linalg.norm(ab) / 2 + 0.3))
        if len(idx) < 10:
            continue
        tt = np.clip(((Pl_low[idx, :2] - a_[:2]) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        dseg = np.linalg.norm(Pl_low[idx, :2] - (a_[:2] + tt[:, None] * ab), axis=1)
        m = dseg < max(c["r"], 0.1)
        if m.sum() < 10:
            continue
        top = float(np.percentile(hl_low[idx][m], 90))
        if top < 0.05:                                             # rien qui dépasse du sol LiDAR
            continue
        r = top / 2
        a_[2] = H_at(a_[None, :2])[0] + r
        b_[2] = H_at(b_[None, :2])[0] + r
        caps_l.append({**c, "a": a_, "b": b_, "r": r, "r_gauss": c["r"]})
    print(f"troncs couchés : {len(caps)} trouvés dans les gaussiennes, {len(caps_l)} recalés en hauteur sur le LiDAR ; "
          f"diamètre gaussiennes {np.median([2 * c['r'] for c in caps]) * 100:.0f} cm → LiDAR "
          f"{np.median([2 * c['r'] for c in caps_l]) * 100 if caps_l else 0:.0f} cm (médianes)", flush=True)
    caps = caps_l
    del Pl, G5, Pl_low
    print(f"axes : gaussiennes {sum(f is not None for f in fits_g)}, LiDAR {sum(f is not None for f in fits_l)} "
          f"({time.time() - t0:.0f} s)", flush=True)

    low = tr[src == 2]
    replaced = np.zeros(len(low), bool)
    for c in caps:
        a_, b_ = c["a"][:2], c["b"][:2]
        ab = b_ - a_
        tt = np.clip(((low[:, :2] - a_) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        dseg = np.linalg.norm(low[:, :2] - (a_ + tt[:, None] * ab), axis=1)
        replaced |= dseg < c["r"] + low[:, 2]
    print(f"troncs couchés : {len(caps)} capsules, longueur médiane "
          f"{np.median([c['long_m'] for c in caps]) if caps else 0:.2f} m ; obstacles bas recouverts {int(replaced.sum())}",
          flush=True)

    # --- sol
    Hs, sres = sol_gaussiennes(P, h, op, w, lab, T)
    del P
    print(f"sol gaussiennes : {np.isfinite(Hs).mean():.1%} des cellules de 5 cm ({time.time() - t0:.0f} s)", flush=True)

    # --- évaluation sur balayages mis de côté
    O, Q = held_out_points(a.scene, a.scans, a.odometry, n=a.n_scans)
    print(f"{len(Q)} points mis de côté ({time.time() - t0:.0f} s)", flush=True)
    kq = cKDTree(Q[:, :2])
    res_t = {"vertical (LiDAR actuel)": [], "incliné (gaussiennes)": [], "incliné (LiDAR)": []}
    rows = []
    for (x, y, r, z), fg, fl in zip(trunks, fits_g, fits_l):
        idx = kq.query_ball_point([x, y], r + 1.2)
        if len(idx) < 30:
            continue
        Pn = Q[idx]
        z0 = float(H_at(np.array([[x, y]]))[0])
        rv = radial_residual(Pn, np.array([x, y, z0]), np.array([0, 0, 1.0]), r)
        if len(rv) < 30:
            continue
        row = {"vertical": float(np.median(rv)), "n": len(rv)}
        res_t["vertical (LiDAR actuel)"].append(rv)
        for name, f in (("incliné (gaussiennes)", fg), ("incliné (LiDAR)", fl)):
            if f is None:
                res_t[name].append(rv)                                   # repli : cylindre vertical
                row[name] = None
            else:
                rr = radial_residual(Pn, f["base"], f["u"], r)
                res_t[name].append(rr if len(rr) else rv)
                row[name] = float(np.median(rr)) if len(rr) else None
                row[name + "_tilt"] = f["tilt_deg"]
        rows.append(row)
    summary = {"troncs_evalues": len(rows), "axes_ajustes_gaussiennes": sum(f is not None for f in fits_g),
               "axes_ajustes_LiDAR": sum(f is not None for f in fits_l)}
    # troncs couchés : le dessus de la capsule face aux points LiDAR mis de côté au-dessus de son axe
    tops = []
    for c in caps:
        a_, b_ = c["a"], c["b"]
        ab = b_[:2] - a_[:2]
        idx = np.asarray(kq.query_ball_point(((a_ + b_) / 2)[:2], np.linalg.norm(ab) / 2 + c["r"] + 0.2))
        if len(idx) < 20:
            continue
        Pn = Q[idx]
        tt = np.clip(((Pn[:, :2] - a_[:2]) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        dseg = np.linalg.norm(Pn[:, :2] - (a_[:2] + tt[:, None] * ab), axis=1)
        hn = Pn[:, 2] - H_at(Pn[:, :2])
        m = (dseg < 0.5 * c["r"]) & (tt > 0.05) & (tt < 0.95) & (hn > 0.02) & (hn < 2 * c["r"] + 0.3)
        if m.sum() < 10:
            continue
        top = float(np.percentile(hn[m], 90))                           # dessus mesuré au-dessus du sol
        tops.append((top - 2 * c["r"], top))
    if tops:
        tops = np.array(tops)
        summary["troncs_couches"] = {"capsules": len(caps), "evaluees": len(tops),
                                     "diametre_capsule_cm_q10_50_90": (np.percentile([2 * c["r"] for c in caps], [10, 50, 90]) * 100).round(1).tolist(),
                                     "dessus_LiDAR_moins_dessus_capsule_cm_q10_50_90": (np.percentile(tops[:, 0], [10, 50, 90]) * 100).round(1).tolist(),
                                     "dessus_LiDAR_au_dessus_du_sol_cm_q50": round(float(np.median(tops[:, 1])) * 100, 1)}
    for name, L_ in res_t.items():
        v = np.concatenate(L_)
        summary[name] = {"ecart_radial_median_cm": round(float(np.median(v)) * 100, 1),
                         "p90_cm": round(float(np.percentile(v, 90)) * 100, 1), "points": int(len(v))}
    tilts_g = np.array([f["tilt_deg"] for f in fits_g if f is not None])
    tilts_l = np.array([f["tilt_deg"] for f in fits_l if f is not None])
    summary["inclinaison_gaussiennes_q50_90_deg"] = np.percentile(tilts_g, [50, 90]).round(1).tolist()
    summary["inclinaison_LiDAR_q50_90_deg"] = np.percentile(tilts_l, [50, 90]).round(1).tolist()
    # accord des deux axes inclinés quand les deux existent
    both = [(fg, fl) for fg, fl in zip(fits_g, fits_l) if fg is not None and fl is not None]
    if both:
        dang = [np.degrees(np.arccos(np.clip(fg["u"] @ fl["u"], -1, 1))) for fg, fl in both]
        dbase = [np.linalg.norm(fg["base"][:2] - fl["base"][:2]) for fg, fl in both]
        summary["accord_axes_gaussiennes_LiDAR"] = {"n": len(both), "angle_q50_90_deg": np.percentile(dang, [50, 90]).round(1).tolist(),
                                                    "base_q50_90_cm": (np.percentile(dbase, [50, 90]) * 100).round(1).tolist()}
    # troncs penchés (> 5° selon le LiDAR incliné) : là où le modèle incliné doit gagner
    pen = [r_ for r_ in rows if (r_.get("incliné (LiDAR)_tilt") or 0) > 5]
    if pen:
        summary["troncs_penches_>5deg"] = {"n": len(pen), **{k: round(float(np.median([r_[k] if r_.get(k) is not None else r_["vertical"]
                                                                                      for r_ in pen])) * 100, 1)
                                                            for k in ("vertical", "incliné (gaussiennes)", "incliné (LiDAR)")}}

    # sol : rayons dont l'impact est au sol (|h| < 0,3 m, cellules à moins de 6 m du trajet), élévation < −3°
    D = Q - O
    rng = np.linalg.norm(D, axis=1)
    D /= rng[:, None]
    hq = Q[:, 2] - H_at(Q[:, :2])
    dq = kd_path.query(Q[:, :2])[0]
    gm = (np.abs(hq) < 0.4) & (dq < 6.0) & (D[:, 2] < -0.05)
    cand = np.where(gm)[0]
    dd, jj = kd_g.query(Q[cand], distance_upper_bound=0.1)
    okg = np.isfinite(dd)
    okg[okg] = np.isin(lab_g[jj[okg]], GROUND)
    sel = cand[okg]
    summary["sol_impacts_candidats"] = int(len(cand))
    summary["sol_impacts_etiquetes_sol"] = int(len(sel))
    rs = np.random.default_rng(0)
    sel = rs.choice(sel, min(len(sel), 300000), replace=False)
    lo = np.asarray(T["lo"], float)
    t_lidar = cast_heightfield(O[sel], D[sel], T["H"], lo, float(T["res"]))
    Hs_f = np.where(np.isfinite(Hs), Hs, np.repeat(np.repeat(T["H"], 2, 0), 2, 1)[:Hs.shape[0], :Hs.shape[1]])
    t_gauss = cast_heightfield(O[sel], D[sel], Hs_f, lo, sres)
    H10_up = np.repeat(np.repeat(T["H"], 2, 0), 2, 1)[:Hl5.shape[0], :Hl5.shape[1]]
    t_l5 = cast_heightfield(O[sel], D[sel], np.where(np.isfinite(Hl5), Hl5, H10_up), lo, 0.05)
    off = 0.0
    summary["sol"] = {"rayons": int(len(sel))}
    for name, tt in (("LiDAR 10 cm (actuel)", t_lidar), ("gaussiennes sol 5 cm", t_gauss),
                     ("LiDAR 5 cm, points étiquetés sol", t_l5)):
        err = (rng[sel] - tt) * (-D[sel, 2])                            # erreur verticale (> 0 : modèle trop haut)
        ok = np.isfinite(err)
        e = err[ok]
        dsel = dq[sel][ok]
        summary["sol"][name] = {"rayons_touchant": int(ok.sum()),
                                "erreur_verticale_mediane_cm": round(float(np.median(e)) * 100, 1),
                                "dispersion_mediane_cm": round(float(np.median(np.abs(e - np.median(e)))) * 100, 1),
                                "a_3cm": round(float((np.abs(e) < 0.03).mean()), 3),
                                **{f"mediane_cm_{lo_}-{hi_}m": round(float(np.median(e[(dsel >= lo_) & (dsel < hi_)])) * 100, 1)
                                   for lo_, hi_ in ((0, 1), (1, 3), (3, 6))},
                                **{f"a_3cm_{lo_}-{hi_}m": round(float((np.abs(e[(dsel >= lo_) & (dsel < hi_)]) < 0.03).mean()), 3)
                                   for lo_, hi_ in ((0, 1), (1, 3), (3, 6))}}
    summary["duree_s"] = round(time.time() - t0)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    np.savez_compressed(f"{a.out}/splat_mesh.npz", sol_gauss=Hs, sol_res=sres, sol_lidar5=Hl5,
                        troncs=trunks, axes_g=np.array([np.r_[f["base"], f["u"], f["tilt_deg"]] if f else np.full(7, np.nan)
                                                        for f in fits_g]),
                        axes_l=np.array([np.r_[f["base"], f["u"], f["tilt_deg"]] if f else np.full(7, np.nan) for f in fits_l]),
                        couches=np.array([np.r_[c["a"], c["b"], c["r"]] for c in caps]) if caps else np.zeros((0, 7)),
                        bas_recouverts=replaced)
    json.dump({"resume": summary, "troncs": rows}, open(f"{a.out}/splat_mesh_eval.json", "w"), ensure_ascii=False, indent=1)
    # géométrie pour le simulateur : axe LiDAR s'il existe (le plus juste), sinon axe des gaussiennes ; garde-fous :
    # axes LiDAR et gaussiennes en désaccord de plus de 5° → vertical ; plus de 15° d'inclinaison seulement si les deux
    # concordent à 3° ; et le tronc doit rester à ≥ 25 cm du trajet de l'ANYmal entre 0,1 et 0,8 m (il y est passé),
    # sinon on garde le cylindre vertical (déjà contrôlé contre le trajet par stems.py)
    inc_idx, inc_base, inc_u, inc_src = [], [], [], []
    rejet = {"desaccord": 0, "trop_penche": 0, "trajet": 0}
    for k, (fg, fl) in enumerate(zip(fits_g, fits_l)):
        f, s_ = (fl, 0) if fl is not None else (fg, 1)
        if f is None:
            continue
        ang = (np.degrees(np.arccos(np.clip(fg["u"] @ fl["u"], -1, 1))) if fg is not None and fl is not None else None)
        if ang is not None and ang > 5:
            rejet["desaccord"] += 1
            continue
        if f["tilt_deg"] > 15 and (ang is None or ang > 3):
            rejet["trop_penche"] += 1
            continue
        v = f["u"] / f["u"][2]
        pts = f["base"] + np.linspace(0.1, 0.8, 8)[:, None] * v
        if (kd_path.query(pts[:, :2])[0] - trunks[k, 2]).min() < 0.25:
            rejet["trajet"] += 1
            continue
        inc_idx.append(trunks_idx[k])
        inc_base.append(f["base"])
        inc_u.append(f["u"])
        inc_src.append(s_)
    print(f"axes rejetés (cylindre vertical gardé) : {rejet}", flush=True)
    low_idx = np.where(src == 2)[0]
    np.savez_compressed(f"{a.scene}/geometrie_etiquetee.npz", incl_idx=np.array(inc_idx, np.int64),
                        incl_base=np.array(inc_base), incl_u=np.array(inc_u), incl_src=np.array(inc_src, np.int8),
                        couches=np.array([np.r_[c["a"], c["b"], c["r"]] for c in caps]) if caps else np.zeros((0, 7)),
                        bas_remplaces=low_idx[replaced],
                        note="troncs inclinés (incl_src 0 : axe LiDAR, 1 : axe des gaussiennes « tree ») et troncs "
                             "couchés (capsules a, b, r) ; bas_remplaces : obstacles bas de terrain.npz recouverts")
    print(f"geometrie_etiquetee.npz : {len(inc_idx)} troncs inclinés ({sum(inc_src)} par les gaussiennes), "
          f"{len(caps)} troncs couchés, {int(replaced.sum())} obstacles bas remplacés", flush=True)



# ------------------------------------------------------------------------------------------------ planche visuelle

def capsule_points(a, b, r, n=1500, rng=None):
    """Points à la surface d'une capsule (segment a–b, rayon r)."""
    rng = rng or np.random.default_rng(0)
    d = b - a
    L = np.linalg.norm(d)
    u = d / max(L, 1e-9)
    e1 = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    t = rng.uniform(0, L, n)
    th = rng.uniform(0, 2 * np.pi, n)
    return a + np.outer(t, u) + r * (np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2))


def planche(scene, labels, mesh_npz, out, n_each=4, W=960, H=640):
    """Vues de capture où l'on voit des troncs penchés et des troncs couchés : ancienne géométrie (orange) et
    nouvelle (cyan) projetées sur le rendu du splat."""
    from PIL import Image, ImageDraw

    from splat_labels import font, load_gaussians as load_g, render_rgb, views

    meta = json.load(open(f"{scene}/scene.json"))
    T = dict(np.load(f"{scene}/terrain.npz"))
    M = np.load(mesh_npz)
    M2P = np.asarray(meta["ply_to_physics"])                               # physique = M2P · PLY
    V = views(scene, W, H, stride=2, holdout=10 ** 9)
    cams_c = np.array([np.linalg.inv(v["viewmat"])[:3, 3] @ M2P.T for v in V])   # centres caméra, physique

    def project(v, Xphys):
        Xp = Xphys @ M2P                                                    # physique → PLY (M2P orthogonale)
        pc = Xp @ v["viewmat"][:3, :3].T + v["viewmat"][:3, 3]
        ok = pc[:, 2] > 0.2
        uv = pc[:, :2] / np.maximum(pc[:, 2:3], 1e-6) * [v["K"][0, 0], v["K"][1, 1]] + [v["K"][0, 2], v["K"][1, 2]]
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        return uv, ok

    def best_view(center, dmin=1.5, dmax=8.0):
        d = np.linalg.norm(cams_c - center, axis=1)
        best, score = None, 1e9
        for k in np.where((d > dmin) & (d < dmax))[0]:
            uv, ok = project(V[k], center[None])
            if ok[0]:
                s = np.hypot(uv[0, 0] - W / 2, uv[0, 1] - H / 2) + 30 * d[k]
                if s < score:
                    best, score = k, s
        return best

    items = []
    ax_l, ax_g, tr = M["axes_l"], M["axes_g"], M["troncs"]
    tilt = np.nan_to_num(ax_l[:, 6], nan=0)
    # troncs penchés de 6 à 25° selon le LiDAR ET les gaussiennes (axes concordants à 3° près), les plus proches
    # d'une caméra de capture
    agree = np.degrees(np.arccos(np.clip(np.nan_to_num(np.sum(ax_l[:, 3:6] * ax_g[:, 3:6], 1)), -1, 1))) < 3
    dcam = np.array([np.linalg.norm(cams_c - (tr[i, :3] if False else np.r_[tr[i, :2], tr[i, 3]]), axis=1).min()
                     for i in range(len(tr))])
    cand = np.where((tilt > 6) & (tilt < 25) & agree)[0]
    for i in cand[np.argsort(dcam[cand])]:
        base, u = ax_g[i, :3], ax_g[i, 3:6]
        x, y, r, _ = tr[i]
        old = [capsule_points(np.array([x, y, base[2]]), np.array([x, y, base[2] + 2.0]), r)]
        new = [capsule_points(base, base + 3.0 * u / u[2], r)]
        if best_view(base + 1.0 * u) is None:
            continue
        items.append(("tronc penché %.0f°" % tilt[i], base + 1.0 * u, old, new))
        if len(items) >= n_each:
            break
    caps = M["couches"]
    low = T["troncs"][T["troncs_source"] == 2]
    for c in sorted(caps, key=lambda c: -np.linalg.norm(c[3:6] - c[:3]))[:n_each]:
        a_, b_, r = c[:3], c[3:6], c[6]
        ab = b_[:2] - a_[:2]
        tt = np.clip(((low[:, :2] - a_[:2]) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        near = np.linalg.norm(low[:, :2] - (a_[:2] + tt[:, None] * ab), axis=1) < r + low[:, 2]
        old = [capsule_points(np.r_[o[:2], o[3]], np.r_[o[:2], o[3] + 2.0], o[2], 400) for o in low[near]]
        items.append(("tronc couché %.1f m, Ø %.0f cm" % (np.linalg.norm(b_ - a_), 200 * r), (a_ + b_) / 2, old,
                      [capsule_points(a_, b_, r)]))
    G = load_g(f"{scene}/merged.ply")
    rows, f = [], font(18)
    for title, center, old, new in items:
        k = best_view(center)
        if k is None:
            continue
        img, _ = render_rgb(G, V[k], W, H)
        tiles = []
        for pts_list, col in ((old, (255, 150, 30)), (new, (40, 230, 255))):
            im = Image.fromarray(img.copy())
            d = ImageDraw.Draw(im)
            for pts in pts_list:
                uv, ok = project(V[k], pts)
                for x_, y_ in uv[ok]:
                    d.ellipse((x_ - 1.5, y_ - 1.5, x_ + 1.5, y_ + 1.5), fill=col)
            d.text((8, 6), title + (" — avant" if col[0] == 255 else " — gaussiennes étiquetées"), fill=(255, 255, 255), font=f)
            tiles.append(im)
        rows.append(tiles)
    S = Image.new("RGB", (2 * W + 6, len(rows) * (H + 6)), (20, 20, 24))
    for r_, (a_, b_) in enumerate(rows):
        S.paste(a_, (0, r_ * (H + 6)))
        S.paste(b_, (W + 6, r_ * (H + 6)))
    S = S.resize((S.width // 2, S.height // 2))
    S.save(out, quality=88)
    print(out, S.size)


if __name__ == "__main__":
    main()
