"""Étiquettes du splat → terrain de collision (desktop). Produit `<scène>/terrain_etiquettes.npz`, lu par server.py :

- `sol` (nx, ny) : classe ADE20K du sol par cellule du terrain (10 cm). C'est la classe majoritaire, pondérée par
  l'opacité, des gaussiennes à ±15 cm du sol ; une cellule sans gaussienne prend la classe de la plus proche. Elle sert
  à l'adhérence et à la résistance au roulement des roues (`surfaces.json`).
- Pour chaque obstacle de `terrain.npz` (`troncs`), `obs_type` :
  - 4 « sans gaussienne » : moins de 10 gaussiennes visibles non « sol » dans son volume (rayon + 5 cm, 0,3–1,5 m) → retiré. Ce
    sont surtout des troncs LiDAR hors de la zone reconstruite (598 sur HÖB-1) : des collisions invisibles ;
  - troncs debout visibles (sources 0, 1) : 0 rigide ;
  - obstacles bas (source 2), d'après les gaussiennes de 0,15 à 1 m dans le rayon + 10 cm : 2 « relief » si « earth »
    domine (et « tree » < 40 %) ; 0 rigide si boisé (« plant » ne domine pas) ET opaque au LiDAR (q90 du μ des voxels
    ≥ 8 m⁻¹ : souche, tronc couché, rocher) ; 1 « végétation » sinon : traversable, freine selon `obs_s` =
    μ_moyen / 0,55, borné à [0,3 ; 3], × (1 + 0,5 × part boisée).
- `relief_dh` (nx, ny) : hauteur ajoutée au champ de hauteur sous les reliefs (haut des voxels LiDAR opaques).
- `veg_s`, `veg_h` (nx, ny) : végétation en nappe (« plant » domine de 0,15 à 1 m) ; freinage relatif = μ de la carte
  d'occupation (0,1–1 m) / 0,55 × hauteur / 0,6 m.
- `dH` (nx, ny, ≤ 0) : corrections du sol. (a) Bosses de plus de 20 cm au-dessus du sol lissé (médiane 1,5 m) sans
  gaussienne visible à 30 cm, ou toutes leurs gaussiennes 15 cm plus bas : aplanies. (b) Sous la végétation, le sol
  LiDAR (médiane de la couche de sol) est tiré vers le haut : 11 % des points mis de côté y sont plus de 8 cm sous lui,
  contre 3,8 % sur sol nu. Il est abaissé de (q05 des points DLIO de la cellule − q05 médian du sol nu), borné à 15 cm.
  Les corrections sont adoucies (flou gaussien σ = 15 cm) pour ne pas créer de marches.
  (c) Plateaux plus étroits que 2 m et plus hauts que 40 cm (H − ouverture morphologique 2,1 m, qui garde les pentes)
  sans gaussienne « sol » visible à ±25 cm : le LiDAR n'y a vu que le haut d'une végétation dense, et le « sol » y
  montait jusqu'à 4 m (« tente » verte près du départ 22). Ramenés à l'ouverture.
  (d) Le long du trajet de l'ANYmal (±30 cm), le sol est borné à ±8 cm de la hauteur de son capteur moins sa hauteur
  médiane au-dessus du sol (des plateaux de 1,4 et 2 m y restaient, puis des trous, sur le circuit du mode jeu).
  Lissage final : min(correction, flou σ = 15 cm), pour garder la correction entière au centre.

  cd ~/dino && PYTHONPATH=. venv/bin/python simrobot/sim/terrain_labels.py --scene out/simrobot/scenes/HOB-1_full \
      --labels out/simrobot/labels/HOB-1_full/gaussiennes.npz --dlio data_local/simrobot/grandtour/HOB-1/lidar \
      --scans /mnt/data/dino/gt_lidar/dlio_hesai_points_undistorted.tar --odometry /mnt/data/dino/gt_lidar/dlio_map_odometry.tar
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

TREE, PLANT, EARTH, SKY = 4, 17, 13, 2
RIGIDE, VEGETATION, RELIEF, REMPLACE, SANS = 0, 1, 2, 3, 4
TYPES = ("rigide", "végétation", "relief", "remplacé par une capsule", "sans gaussienne")
MU_REF = 0.55                  # extinction LiDAR moyenne d'un buisson médian (0,1–1 m) : freinage relatif s = 1


def load_splat(scene, meta):
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
    P = np.stack([V["x"], V["y"], V["z"]], 1) @ np.asarray(meta["ply_to_physics"]).T
    return P, (1 / (1 + np.exp(-V["opacity"]))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--labels", required=True, help="gaussiennes.npz de splat_labels.py lift")
    ap.add_argument("--occupation", default="lidar_occupation.npz")
    ap.add_argument("--terrain", default="terrain.npz")
    ap.add_argument("--dlio", default=None, help="dossier de la carte DLIO (*_dlio.ply) : sol abaissé sous la végétation")
    ap.add_argument("--scans", default=None, help="balayages Hesai (tar) : évaluation sur les balayages mis de côté")
    ap.add_argument("--odometry", default=None)
    ap.add_argument("--max-arbre", type=float, default=0.4, help="part « tree » au-delà de laquelle ce n'est pas un relief")
    ap.add_argument("--mu-solide", type=float, default=8.0,
                    help="μ (q90 des voxels, m⁻¹) au-delà duquel un obstacle bas boisé est solide (souche, tronc couché)")
    ap.add_argument("--min-support", type=int, default=10, help="gaussiennes minimum dans le volume d'un obstacle")
    a = ap.parse_args()
    t0 = time.time()
    from scipy import ndimage
    from scipy.spatial import cKDTree

    meta = json.load(open(f"{a.scene}/scene.json"))
    T = np.load(f"{a.scene}/{a.terrain}")
    H, lo, res = T["H"], T["lo"], float(T["res"])
    nx, ny = H.shape
    lab = np.load(a.labels)["label"]
    P, op = load_splat(a.scene, meta)
    ix = np.floor((P[:, 0] - lo[0]) / res).astype(np.int64)
    iy = np.floor((P[:, 1] - lo[1]) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (lab < 254)
    ix, iy, P, op, lab = ix[ok], iy[ok], P[ok], op[ok], lab[ok]
    h = P[:, 2] - H[ix, iy]
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    path = np.array(json.load(open(f"{a.scene}/trajets.json"))["dlio"]["p"])[:, :2]
    dpath = cKDTree(path).query(np.stack([lo[0] + (gx.ravel() + 0.5) * res,
                                          lo[1] + (gy.ravel() + 0.5) * res], 1))[0].reshape(nx, ny)

    # --- classe du sol par cellule : histogramme pondéré par l'opacité, ±15 cm du sol
    g = (h > -0.15) & (h < 0.15)
    cell = ix[g] * ny + iy[g]
    hist = np.zeros((nx * ny, 256), np.float32)
    np.add.at(hist, (cell, lab[g].astype(np.int64)), op[g])
    vu = hist.sum(1) > 0
    sol = np.where(vu, hist.argmax(1), 255).astype(np.uint8).reshape(nx, ny)
    del hist
    near = ndimage.distance_transform_edt(sol == 255, return_distances=False, return_indices=True)
    sol_plein = sol[near[0], near[1]]

    # --- présence du splat par cellule : gaussiennes visibles (opacité > 0,2, hors ciel) de −0,3 à +2 m, voisinage 3×3
    col = (h > -0.3) & (h < 2.0) & (op > 0.2) & (lab != SKY)
    cnt = np.bincount(ix[col] * ny + iy[col], minlength=nx * ny).reshape(nx, ny).astype(np.float32)
    cnt3 = ndimage.uniform_filter(cnt, 3) * 9
    top = np.full(nx * ny, -np.inf, np.float32)
    np.maximum.at(top, ix[col] * ny + iy[col], h[col].astype(np.float32))
    top = top.reshape(nx, ny)
    # soutien d'un obstacle : gaussiennes visibles NON « sol » (le sol du splat est ~10 cm au-dessus du sol LiDAR :
    # compter le sol validait des obstacles posés sur de la litière nue)
    obj = col & ~np.isin(lab, (13, 52, 6, 91, 21, 9, 34))
    kd_vis = cKDTree(P[obj, :2])
    h_vis = h[obj]

    # gaussiennes « sol » visibles à ±25 cm du sol LiDAR, voisinage 3×3 : le sol LiDAR y est vu par les caméras
    gs = np.isin(lab, (13, 52, 6, 91, 21, 9, 34)) & (op > 0.3) & (np.abs(h) < 0.25)
    cg3 = ndimage.uniform_filter(np.bincount(ix[gs] * ny + iy[gs], minlength=nx * ny).reshape(nx, ny).astype(np.float32),
                                 3) * 9

    # --- végétation : cellules où « plant » domine de 0,15 à 1 m ; hauteur (haut des gaussiennes « plant » opaques,
    # moins les ~7 cm dont le splat surélève les surfaces, cf. note 11)
    b = (h > 0.15) & (h < 1.0) & (op > 0.2)
    ca = np.bincount(ix[b] * ny + iy[b], minlength=nx * ny).reshape(nx, ny)
    cp = np.bincount(ix[b & (lab == PLANT)] * ny + iy[b & (lab == PLANT)], minlength=nx * ny).reshape(nx, ny)
    veg = (ca >= 5) & (cp >= 0.5 * ca)
    vh = np.zeros(nx * ny, np.float32)
    pm = (lab == PLANT) & (op > 0.5) & (h > 0.05) & (h < 1.5)
    np.maximum.at(vh, ix[pm] * ny + iy[pm], h[pm].astype(np.float32))
    veg_h = np.where(veg, np.clip(ndimage.median_filter(vh.reshape(nx, ny), 3) - 0.07, 0, 1.2), 0).astype(np.float32)

    # --- composition autour des obstacles (gaussiennes de 0,15 à 1 m au-dessus du sol)
    cls = np.full(256, 3, np.int64)
    cls[TREE], cls[PLANT], cls[EARTH] = 0, 1, 2
    comp = np.zeros((nx * ny, 4), np.float32)
    np.add.at(comp, (ix[b] * ny + iy[b], cls[lab[b]]), op[b])
    comp = comp.reshape(nx, ny, 4)
    del P

    # --- carte d'occupation LiDAR : μ par voxel de 5 cm, hauteur au-dessus du sol
    O = np.load(f"{a.scene}/{a.occupation}")
    ores, ony = float(O["res"]), int(O["ny"])
    olo = O["lo"]
    cs = O["col_start"]
    below = float(O["below"])
    vz = O["vz"].astype(np.int64)
    mu = O["mu"].astype(np.float32)
    # densité de végétation par cellule de 10 cm : μ moyen de 0,1 à 1 m (voxels absents = 0), 4 colonnes de 5 cm
    f_ = int(round(res / ores))
    colv = np.repeat(np.arange(len(cs) - 1), np.diff(cs))
    hv = (vz + 0.5) * ores - below
    bv = (hv > 0.1) & (hv < 1.0)
    cv = (colv[bv] // ony) // f_ * ny + (colv[bv] % ony) // f_
    okc = cv < nx * ny
    mu_cell = (np.bincount(cv[okc], weights=mu[bv][okc], minlength=nx * ny)[:nx * ny]
               / (f_ * f_ * 0.9 / ores)).reshape(nx, ny).astype(np.float32)
    del colv, hv, bv, cv
    veg_s = np.where(veg, np.clip(mu_cell / MU_REF, 0, 3) * np.clip(veg_h / 0.6, 0, 1), 0).astype(np.float32)

    def column_voxels(x0, x1, y0, y1):
        """Voxels des colonnes 5 cm dans le rectangle [x0, x1[ × [y0, y1[ (repère physique) : x, y, h, μ."""
        i0, i1 = int(np.floor((x0 - olo[0]) / ores)), int(np.ceil((x1 - olo[0]) / ores))
        j0, j1 = int(np.floor((y0 - olo[1]) / ores)), int(np.ceil((y1 - olo[1]) / ores))
        out = []
        for i in range(max(i0, 0), min(i1, int(O["nx"]))):
            c0 = i * ony
            for j in range(max(j0, 0), min(j1, ony)):
                s, e = cs[c0 + j], cs[c0 + j + 1]
                if e > s:
                    k = vz[s:e]
                    out.append(np.stack([np.full(e - s, olo[0] + (i + 0.5) * ores),
                                         np.full(e - s, olo[1] + (j + 0.5) * ores),
                                         (k + 0.5) * ores - below, mu[s:e]], 1))
        return np.concatenate(out) if out else np.zeros((0, 4))

    tr, src = T["troncs"], T["troncs_source"]
    n = len(tr)
    obs_type = np.zeros(n, np.int8)
    obs_parts = np.zeros((n, 3), np.float32)
    obs_mu = np.zeros(n, np.float32)
    obs_q90 = np.zeros(n, np.float32)
    obs_s = np.zeros(n, np.float32)
    obs_haut = np.zeros(n, np.float32)
    obs_support = np.zeros(n, np.int32)
    relief_dh = np.zeros((nx, ny), np.float32)
    veg_obs = np.zeros((nx, ny), bool)
    mu_opaque = -np.log(1 - 0.3) / ores                            # ≥ 30 % d'arrêt sur 5 cm
    for i in range(n):
        x, y, r, _ = tr[i]
        idx = kd_vis.query_ball_point([x, y], r + 0.05)
        obs_support[i] = int(((h_vis[idx] > 0.3) & (h_vis[idx] < 1.5)).sum()) if idx else 0
        R = r + 0.1
        cx, cy, rr = (x - lo[0]) / res, (y - lo[1]) / res, R / res
        i0, i1 = int(max(0, cx - rr - 1)), int(min(nx, cx + rr + 2))
        j0, j1 = int(max(0, cy - rr - 1)), int(min(ny, cy + rr + 2))
        m = (gx[i0:i1, j0:j1] + 0.5 - cx) ** 2 + (gy[i0:i1, j0:j1] + 0.5 - cy) ** 2 <= rr ** 2
        c = comp[i0:i1, j0:j1][m].sum(0)
        f = c / c.sum() if c.sum() > 0 else np.zeros(4)
        obs_parts[i] = f[:3]
        Vx = column_voxels(x - R, x + R, y - R, y + R)
        Vx = Vx[(Vx[:, 0] - x) ** 2 + (Vx[:, 1] - y) ** 2 <= R ** 2]
        vb = Vx[(Vx[:, 2] > 0.1) & (Vx[:, 2] < 1.0)]
        n_vol = np.pi * R ** 2 * 0.9 / ores ** 3                     # voxels du volume (absents = μ nul)
        obs_mu[i] = float(vb[:, 3].sum() / n_vol) if len(vb) else 0.0
        vin = Vx[((Vx[:, 0] - x) ** 2 + (Vx[:, 1] - y) ** 2 <= (r + 0.05) ** 2) & (Vx[:, 2] > 0.2) & (Vx[:, 2] < 1.0)]
        obs_q90[i] = float(np.percentile(vin[:, 3], 90)) if len(vin) else 0.0
        opaque = Vx[(Vx[:, 2] > 0.0) & (Vx[:, 2] < 1.2) & (Vx[:, 3] >= mu_opaque)]
        obs_haut[i] = float(np.percentile(opaque[:, 2], 90)) if len(opaque) else 0.0
        if obs_support[i] < a.min_support:                         # rien de visible : pas de collision invisible
            obs_type[i] = SANS
            continue
        if src[i] != 2:
            continue                                               # tronc debout visible : rigide
        dom = int(np.argmax(f)) if c.sum() > 0 else 0
        if dom == 2 and f[0] < a.max_arbre:
            obs_type[i] = RELIEF
            # haut du relief par cellule de 10 cm : voxels opaques les plus hauts
            ci = np.floor((opaque[:, 0] - lo[0]) / res).astype(int)
            cj = np.floor((opaque[:, 1] - lo[1]) / res).astype(int)
            top_r = np.zeros((i1 - i0, j1 - j0), np.float32)
            inb = (ci >= i0) & (ci < i1) & (cj >= j0) & (cj < j1)
            np.maximum.at(top_r, (ci[inb] - i0, cj[inb] - j0), opaque[inb, 2].astype(np.float32))
            sub = relief_dh[i0:i1, j0:j1]
            sub[m] = np.maximum(sub[m], top_r[m])
        elif dom != 1 and obs_q90[i] >= a.mu_solide:
            obs_type[i] = RIGIDE                                   # boisé et opaque : souche, tronc couché, rocher
        else:
            obs_type[i] = VEGETATION                               # se traverse en freinant, plus fort si boisé
            obs_s[i] = float(min(3.0, np.clip(obs_mu[i] / MU_REF, 0.3, 3.0) * (1 + 0.5 * f[0])))
            sub = veg_obs[i0:i1, j0:j1]
            sub[m] = True

    # --- sol : bosses sans gaussiennes aplanies ; sol abaissé sous la végétation (impacts LiDAR les plus bas)
    Hs = ndimage.median_filter(H, size=15)
    bump = (H + relief_dh - Hs) > 0.2
    sans_g = bump & ((cnt3 < 5) | (top < relief_dh - 0.15))
    dH = np.where(sans_g, np.minimum(Hs - H, 0.0), 0.0).astype(np.float32)
    # plateaux du sol LiDAR plus étroits que 2 m et plus hauts que 40 cm (ouverture morphologique, qui garde les pentes) sans
    # gaussienne « sol » visible à leur hauteur : le LiDAR n'y a vu que le haut d'une végétation dense (jusqu'à 4 m)
    Ho = ndimage.grey_opening(H, size=(21, 21))
    plateau = ((H - Ho) > 0.4) & (cg3 < 3)
    dH = np.where(plateau, np.minimum(dH, Ho - H), dH).astype(np.float32)
    sans_g |= plateau
    relief_dh[sans_g] = 0.0
    stats_veg = {}
    vegm = veg | veg_obs
    if a.dlio:
        from gaussgym_export import FLIP
        from lidar_voxels import read_ply_xyzi

        centre = np.array(meta["centre_monde_m"])
        Rr, trl = np.array(meta["recalage_lidar"]["R"]), np.array(meta["recalage_lidar"]["t"])
        Pl, _ = read_ply_xyzi(glob.glob(f"{a.dlio}/*_dlio.ply")[0])
        Pl = (Pl @ Rr.T + trl) @ FLIP.T - centre
        li = np.floor((Pl[:, 0] - lo[0]) / res).astype(np.int64)
        lj = np.floor((Pl[:, 1] - lo[1]) / res).astype(np.int64)
        okl = (li >= 0) & (li < nx) & (lj >= 0) & (lj < ny)
        li, lj, zl = li[okl], lj[okl], Pl[okl, 2]
        del Pl
        hl = zl - H[li, lj]
        mm = np.abs(hl) < 0.4
        key, hh = li[mm] * ny + lj[mm], hl[mm]
        o = np.lexsort((hh, key))
        key, hh = key[o], hh[o]
        st = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
        nn = np.diff(np.r_[st, len(key)])
        q05 = np.full(nx * ny, np.nan, np.float32)
        good = nn >= 20
        q05[key[st[good]]] = hh[st[good] + (nn[good] * 0.05).astype(np.int64)]
        q05 = q05.reshape(nx, ny)
        bare = np.isin(sol_plein, (13, 52, 6)) & ~vegm & (cnt3 >= 5) & (dpath < 6) & np.isfinite(q05)
        q05_ref = float(np.median(q05[bare]))
        dveg = np.where(vegm & np.isfinite(q05), np.clip(q05 - q05_ref, -0.15, 0.0), 0.0).astype(np.float32)
        dveg = np.where(vegm, ndimage.median_filter(dveg, 3), 0.0).astype(np.float32)
        dH = np.minimum(dH, dveg)
        stats_veg = {"q05_sol_nu_cm": round(q05_ref * 100, 1), "cellules_abaissees": int((dveg < -0.01).sum()),
                     "abaissement_median_cm": round(float(np.median(dveg[dveg < -0.01])) * 100, 1)
                     if (dveg < -0.01).any() else 0}
    # le long du trajet de l'ANYmal, le sol est connu : hauteur du capteur (pose DLIO) − sa hauteur médiane au-dessus du
    # sol LiDAR. Des plateaux y montaient jusqu'à 2 m (sans doute des personnes arrêtées sur le sentier) ; le sol ne doit
    # pas y dépasser ce niveau de plus de 10 cm
    Dp = np.array(json.load(open(f"{a.scene}/trajets.json"))["dlio"]["p"])
    pi_ = np.clip(((Dp[:, 0] - lo[0]) / res).astype(int), 0, nx - 1)
    pj_ = np.clip(((Dp[:, 1] - lo[1]) / res).astype(int), 0, ny - 1)
    off = float(np.median(Dp[:, 2] - H[pi_, pj_]))
    zp = np.full((nx, ny), np.inf, np.float32)
    np.minimum.at(zp, (pi_, pj_), (Dp[:, 2] - off).astype(np.float32))
    zp = ndimage.grey_erosion(zp, size=(7, 7))                     # ±30 cm autour du trajet
    sur_trajet = np.isfinite(zp) & (H + dH > zp + 0.10)
    dH = np.where(sur_trajet, np.minimum(dH, zp + 0.05 - H), dH).astype(np.float32)
    # corrections adoucies : un abaissement par cellule faisait des marches (jusqu'à 15 cm) au bord des zones abaissées,
    # où le robot restait coincé. min(brut, flou) : rampes autour, mais correction entière au centre (le flou seul
    # laissait 20–30 cm des plateaux et creusait leurs bords)
    dH = np.minimum(np.minimum(dH, ndimage.gaussian_filter(dH, sigma=1.5)), 0.0).astype(np.float32)
    # puis, sur le trajet de l'ANYmal (±30 cm), sol borné à ±8 cm de son niveau mesuré : la correction des plateaux y
    # laissait aussi des trous (16 cm à s = 50 m du circuit), où une roue de 13 cm restait coincée
    tr_ok = np.isfinite(zp)
    dH = np.where(tr_ok, np.clip(H + dH, zp - 0.08, zp + 0.08) - H, dH).astype(np.float32)
    if a.scans and a.odometry:
        import splat_mesh

        Oh, Q = splat_mesh.held_out_points(a.scene, a.scans, a.odometry, n=40)
        qi = np.clip(((Q[:, 0] - lo[0]) / res).astype(int), 0, nx - 1)
        qj = np.clip(((Q[:, 1] - lo[1]) / res).astype(int), 0, ny - 1)
        for name, msk in (("végétation", vegm), ("sol nu", np.isin(sol_plein, (13, 52, 6)) & ~vegm & (cnt3 >= 5))):
            sel = msk[qi, qj] & (dpath[qi, qj] < 6)
            h0 = Q[sel, 2] - H[qi[sel], qj[sel]]
            h1 = h0 - dH[qi[sel], qj[sel]]
            k = np.abs(h0) < 0.6
            stats_veg[f"points_mis_de_cote_sous_le_sol_de_plus_de_8cm ({name})"] = {
                "avant": round(float((h0[k] < -0.08).mean()), 3), "apres": round(float((h1[k] < -0.08).mean()), 3),
                "au_dessus_de_plus_de_8cm_avant": round(float((h0[k] > 0.08).mean()), 3),
                "au_dessus_de_plus_de_8cm_apres": round(float((h1[k] > 0.08).mean()), 3)}
    stats = {"obstacles": int(n),
             "types": {TYPES[t]: {"troncs": int(((obs_type == t) & (src < 2)).sum()),
                                  "obstacles_bas": int(((obs_type == t) & (src == 2)).sum())} for t in (0, 1, 2, 4)},
             "sans_gaussienne_a_moins_de_6m": int(((obs_type == SANS)
                                                   & (dpath[np.clip(((tr[:, 0] - lo[0]) / res).astype(int), 0, nx - 1),
                                                            np.clip(((tr[:, 1] - lo[1]) / res).astype(int), 0, ny - 1)] < 6)).sum()),
             "vegetation_freinage_s_q10_50_90": np.percentile(obs_s[obs_type == VEGETATION], [10, 50, 90]).round(2).tolist()
             if (obs_type == VEGETATION).any() else None,
             "vegetation_en_nappe": {"cellules": int(veg.sum()), "freinantes_s>0.05": int((veg_s > 0.05).sum()),
                                     "hauteur_mediane_cm": round(float(np.median(veg_h[veg])) * 100, 1)},
             "bosses_sans_gaussienne_aplanies": {"cellules": int(sans_g.sum()), "a_moins_de_6m": int((sans_g & (dpath < 6)).sum()),
                                                 "dont_plateaux_ouverture_2m": int(plateau.sum()),
                                                 "plateaux_de_plus_de_1m": int((plateau & ((H - Ho) > 1)).sum())},
             "sol_sous_vegetation": stats_veg,
             "trajet_ANYmal": {"hauteur_capteur_au_dessus_du_sol_m": round(off, 3), "cellules_abaissees": int(sur_trajet.sum())},
             "sol_cellules_vues": round(float(vu.mean()), 3),
             "duree_s": round(time.time() - t0, 1)}
    np.savez_compressed(f"{a.scene}/terrain_etiquettes.npz", sol=sol_plein, sol_vu=(sol != 255), obs_type=obs_type,
                        obs_parts=obs_parts, obs_mu=obs_mu, obs_q90=obs_q90, obs_s=obs_s, obs_haut=obs_haut,
                        obs_support=obs_support, relief_dh=relief_dh, dH=dH, veg_s=veg_s, veg_h=veg_h,
                        splat_present=(cnt3 >= 5), bosses_aplanies=sans_g,
                        types=np.array(TYPES), stats=json.dumps(stats, ensure_ascii=False))
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
