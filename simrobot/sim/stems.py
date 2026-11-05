"""Troncs d'arbres tirés de la carte LiDAR par ajustement de cercles par tranches (méthode des inventaires forestiers
par LiDAR terrestre / mobile), puis contrôlés contre le splat.

1. Points entre `h_min` et `h_max` au-dessus du sol, découpés en tranches de `slice_h`.
2. Par tranche : occupation sur une grille de `cell` m, composantes connexes, cercle ajusté sur les points de chaque
   composante (Kasa puis Gauss-Newton pondéré). Cercle gardé si rayon plausible, résidu faible, arc couvert ≥ ~120°
   (souvent seule la face tournée vers le sentier est scannée) et composante compacte autour du cercle.
3. Cercles reliés d'une tranche à l'autre (centres à moins de `link` m). Tronc = cercles sur ≥ `min_slices` tranches,
   dont une au-dessus de `min_top` m. Cela écarte :
   - les traînées des personnes qui marchaient à côté du robot : hachures sous ~1,9 m, pas des cercles ;
   - les buissons : nuage diffus ;
   - les troncs couchés : pas de cercle répété en hauteur.
4. Géométrie : centre à hauteur de robot (droite ajustée sur les centres, évaluée à `h_eval`), rayon médian des
   tranches basses, inclinaison.

Pièges de l'ancienne méthode (cellules de 10 cm, « ≥ 20 % des points de la colonne entre 0,3 et 1,3 m ») :
- un grand tronc a des points jusqu'à 20 m de haut : la tranche basse ne pèse que quelques % de sa colonne, et la
  plupart des troncs étaient manqués ;
- les objets bas et étendus (troncs couchés, buissons, personnes accumulées dans la carte) passaient le critère :
  murs invisibles.
"""

from __future__ import annotations

import numpy as np


def fit_circle(xy, iters=8):
    """Cercle (cx, cy, r) : Kasa algébrique puis Gauss-Newton géométrique à poids de Huber. → (cx, cy, r, rms)."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.stack([x, y, np.ones_like(x)], 1)
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-8))
    for _ in range(iters):
        dx, dy = x - cx, y - cy
        d = np.maximum(np.hypot(dx, dy), 1e-9)
        res = d - r
        s = 1.4826 * np.median(np.abs(res)) + 1e-4
        w = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(res), 1e-9))
        J = np.stack([-dx / d, -dy / d, -np.ones_like(d)], 1)
        JW = J * w[:, None]
        try:
            step = np.linalg.solve(JW.T @ J + 1e-9 * np.eye(3), -JW.T @ res)
        except np.linalg.LinAlgError:
            break
        cx, cy, r = cx + step[0], cy + step[1], abs(r + step[2])
        if np.abs(step).max() < 1e-5:
            break
    d = np.hypot(x - cx, y - cy)
    return cx, cy, r, float(np.sqrt(np.mean((d - r) ** 2)))


def ransac_circles(pts, rng, r_min=0.06, r_max=0.6, band=0.025, n_hyp=300, min_inl=30, min_bins=5, max_circles=2):
    """Cercles noyés dans un amas plus grand (anneau de tronc soudé à un buisson ou à des branches) : hypothèses sur
    3 points tirés au hasard, inliers à ±`band` m, arc couvert ≥ `min_bins` × 30°. → liste de (cx, cy, r, rms, n, arc)."""
    out = []
    pts = pts.astype(np.float64)
    for _ in range(max_circles):
        if len(pts) < min_inl:
            break
        idx = rng.integers(0, len(pts), size=(n_hyp, 3))
        a, b, c = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
        d = 2 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1]) + c[:, 0] * (a[:, 1] - b[:, 1]))
        ok = np.abs(d) > 1e-9
        a, b, c, d = a[ok], b[ok], c[ok], d[ok]
        sa, sb, sc_ = (a ** 2).sum(1), (b ** 2).sum(1), (c ** 2).sum(1)
        ux = (sa * (b[:, 1] - c[:, 1]) + sb * (c[:, 1] - a[:, 1]) + sc_ * (a[:, 1] - b[:, 1])) / d
        uy = (sa * (c[:, 0] - b[:, 0]) + sb * (a[:, 0] - c[:, 0]) + sc_ * (b[:, 0] - a[:, 0])) / d
        r = np.hypot(a[:, 0] - ux, a[:, 1] - uy)
        ok = (r >= r_min) & (r <= r_max)
        if not ok.any():
            break
        ux, uy, r = ux[ok], uy[ok], r[ok]
        dist = np.abs(np.hypot(pts[None, :, 0] - ux[:, None], pts[None, :, 1] - uy[:, None]) - r[:, None])
        score = (dist < band).sum(1)
        k = int(np.argmax(score))
        if score[k] < min_inl:
            break
        inl = dist[k] < band
        q = pts[inl]
        cx, cy, rr, rms = fit_circle(q)
        az = np.arctan2(q[:, 1] - cy, q[:, 0] - cx)
        bins = np.bincount(((az + np.pi) / (np.pi / 6)).astype(int) % 12, minlength=12)
        nb = int((bins >= 2).sum())
        # l'intérieur d'un tronc est vide : peu de points à moins de 0,6 r du centre
        inside = (np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) < 0.6 * rr).sum()
        if nb >= min_bins and r_min <= rr <= r_max and inside < 0.15 * inl.sum():
            out.append((cx, cy, rr, rms, int(inl.sum()), nb))
        pts = pts[~inl]
    return out


def slice_circles(xy, lo, shape, cell=0.03, min_cells=6, r_min=0.04, r_max=0.8, min_bins=4, rms_abs=0.03,
                  rms_rel=0.25, ransac_max_ext=3.0, seed=0):
    """Cercles d'une tranche : liste de (cx, cy, r, rms, n_points, arc_bins)."""
    import cv2

    ij = np.floor((xy - lo) / cell).astype(np.int64)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
    xy, ij = xy[ok], ij[ok]
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    cnt = np.bincount(flat, minlength=shape[0] * shape[1])
    occ = (cnt >= 2).reshape(shape).astype(np.uint8)        # une cellule à 1 point = bruit
    occ = cv2.dilate(occ, np.ones((3, 3), np.uint8))           # referme les anneaux scannés de façon clairsemée
    n, lab, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    lp = lab.ravel()[flat]
    keep = lp > 0
    xy, lp = xy[keep], lp[keep]
    order = np.argsort(lp, kind="stable")
    xy, lp = xy[order], lp[order]
    starts = np.searchsorted(lp, np.arange(1, n + 1))
    out = []
    rng = np.random.default_rng(seed)
    max_ext = int(np.ceil(2 * r_max / cell)) + 4
    for c in range(1, n):
        w, hgt = stats[c, cv2.CC_STAT_WIDTH], stats[c, cv2.CC_STAT_HEIGHT]
        if stats[c, cv2.CC_STAT_AREA] < min_cells or max(w, hgt) * cell > ransac_max_ext:
            continue
        pts = xy[starts[c - 1]:starts[c]]
        if len(pts) < 8:
            continue
        if max(w, hgt) > max_ext:                            # amas trop grand pour un seul tronc : RANSAC
            out += ransac_circles(pts, rng)
            continue
        cx, cy, r, rms = fit_circle(pts.astype(np.float64))
        # anneaux épais : la carte DLIO accumule plusieurs passages (résidu médian ~3 cm pour r = 0,1–0,4 m)
        if not (r_min <= r <= r_max) or rms > max(rms_abs, rms_rel * r):
            continue
        # composante compacte autour du cercle (pas un cercle tangent à un buisson ou une traînée)
        ext = max(w, hgt) * cell
        if ext > 2 * r + 0.2 or np.percentile(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy), 95) > r + 0.1:
            out += ransac_circles(pts, rng)                  # anneau soudé à autre chose : on le cherche dedans
            continue
        az = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        bins = np.bincount(((az + np.pi) / (np.pi / 6)).astype(int) % 12, minlength=12)
        nb = int((bins >= 2).sum())
        if nb < min_bins and r > 0.07:                       # petits troncs : souvent un disque plein, arc non défini
            continue
        out.append((cx, cy, r, rms, len(pts), nb))
    return out


def detect_stems(P, h, lo, size, h_min=0.3, h_max=3.3, slice_h=0.3, cell=0.03, link=0.2, min_slices=3,
                 min_top=1.5, h_eval=0.6, low_max=1.8):
    """P : points (repère physique), h : hauteur au-dessus du sol, lo / size : emprise (m).
    → tableau (x, y, r, inclinaison_deg, n_tranches, haut_m) et la liste des cercles (pour le diagnostic)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    shape = (int(np.ceil(size[0] / cell)), int(np.ceil(size[1] / cell)))
    edges = np.arange(h_min, h_max + 1e-6, slice_h)
    circles = []
    for k in range(len(edges) - 1):
        m = (h >= edges[k]) & (h < edges[k + 1])
        for c in slice_circles(P[m, :2], lo, shape, cell=cell):
            circles.append((*c, k, (edges[k] + edges[k + 1]) / 2))
    C = np.array(circles, float).reshape(-1, 8)               # cx, cy, r, rms, n, arc, tranche, h_milieu
    if not len(C):
        return np.zeros((0, 6)), C
    pairs = cKDTree(C[:, :2]).query_pairs(link, output_type="ndarray")
    pairs = pairs[C[pairs[:, 0], 6] != C[pairs[:, 1], 6]]     # liens entre tranches différentes seulement
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(C), len(C)))
    ng, lab = connected_components(g, directed=False)
    stems = []
    for s in range(ng):
        G = C[lab == s]
        ks = np.unique(G[:, 6])
        if len(ks) < min_slices or edges[int(ks.max())] < min_top:
            continue
        # un cercle par tranche (le plus peuplé), puis droite centre = a + b·h
        best = np.array([G[G[:, 6] == k][np.argmax(G[G[:, 6] == k][:, 4])] for k in ks])
        hh = best[:, 7]
        A = np.stack([np.ones_like(hh), hh], 1)
        wts = np.sqrt(best[:, 4])
        cxy = np.linalg.lstsq(A * wts[:, None], best[:, :2] * wts[:, None], rcond=None)[0]
        x, y = cxy[0] + h_eval * cxy[1]
        tilt = float(np.degrees(np.arctan(np.hypot(*cxy[1]))))
        low = best[hh <= low_max]
        r = float(np.median((low if len(low) else best)[:, 2]))
        stems.append((x, y, r, tilt, len(ks), float(edges[int(ks.max()) + 1])))
    S = np.array(stems, float).reshape(-1, 6)
    # troncs dédoublés (deux groupes pour un même arbre) : on garde le plus étayé
    if len(S) > 1:
        order = np.argsort(-S[:, 4])
        kept, tree = [], None
        for i in order:
            if kept:
                d = np.hypot(*(S[kept, :2] - S[i, :2]).T)
                if np.any(d < np.maximum(S[kept, 2] + S[i, 2], 0.1)):
                    continue
            kept.append(i)
        S = S[np.sort(kept)]
    return S, C


def continuity_stems(P, h, lo, size, h_min=0.3, h_max=3.3, slice_h=0.3, cell=0.04, min_frac=0.8, max_ext=1.2,
                     h_full=1.8, r_cap=0.3):
    """Troncs par CONTINUITÉ VERTICALE : cellules occupées (à une cellule près) dans au moins `min_frac` des tranches
    entre h_min et h_max. Rattrape les troncs que l'ajustement par tranche manque parce que leur anneau est soudé à des
    branches basses (conifères en étoile) ou à un buisson. Une branche est horizontale, un buisson et une personne
    restent sous ~1,9 m : ni l'un ni l'autre n'est continu jusqu'à 3,3 m.
    → tableau (x, y, r, 0, n_cellules, h_max)."""
    import cv2

    shape = (int(np.ceil(size[0] / cell)), int(np.ceil(size[1] / cell)))
    edges = np.arange(h_min, h_max + 1e-6, slice_h)
    V = np.zeros(shape, np.uint8)
    full = np.ones(shape, bool)                       # occupé dans TOUTES les tranches au-dessus de h_full
    for k in range(len(edges) - 1):
        m = (h >= edges[k]) & (h < edges[k + 1])
        ij = np.floor((P[m, :2] - lo) / cell).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
        O = np.zeros(shape, np.uint8)
        O[ij[ok, 0], ij[ok, 1]] = 1
        O = cv2.dilate(O, np.ones((3, 3), np.uint8))
        V += O
        if edges[k] >= h_full - 1e-6:
            full &= O > 0
    # une traînée de personne (≤ 1,9 m) surmontée de branches éparses atteignait 80 % des tranches : on exige
    # en plus la continuité sans trou au-dessus de h_full
    core = ((V >= int(np.ceil(min_frac * (len(edges) - 1)))) & full).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(core, np.ones((3, 3), np.uint8)), connectivity=8)
    out = []
    for c in range(1, n):
        w, hgt = stats[c, cv2.CC_STAT_WIDTH] * cell, stats[c, cv2.CC_STAT_HEIGHT] * cell
        if max(w, hgt) > max_ext:                         # massif de feuillage dense : pas un tronc
            continue
        ii, jj = np.nonzero((lab == c) & (core > 0))
        if len(ii) < 3:
            continue
        xy = np.stack([lo[0] + (ii + 0.5) * cell, lo[1] + (jj + 0.5) * cell], 1)
        # rayon mesuré sur le cœur continu seul (pas sur la composante dilatée, ni sur les branches) ; chaque tranche
        # est dilatée d'une cellule, d'où le retrait
        r_eq = 0.5 * max(np.ptp(ii) + 1, np.ptp(jj) + 1) * cell - cell
        if len(xy) >= 8 and r_eq > 0.08:
            cx, cy, r, rms = fit_circle(xy)
            if not (0.04 <= r <= 0.7 and rms < max(0.04, 0.3 * r)):
                cx, cy, r = xy[:, 0].mean(), xy[:, 1].mean(), r_eq
        else:
            cx, cy, r = xy[:, 0].mean(), xy[:, 1].mean(), max(0.04, r_eq)
        # dans un massif dense, le cœur continu englobe des branches : rayon plafonné (les gros troncs nets sont
        # trouvés par les cercles)
        out.append((cx, cy, float(min(r, r_cap)), 0.0, len(ii), h_max))
    return np.array(out, float).reshape(-1, 6)


def merge_stems(A, B):
    """Union de deux listes de troncs ; un tronc de B recouvrant un tronc de A est écarté (A prioritaire)."""
    if not len(A):
        return B
    if not len(B):
        return A
    from scipy.spatial import cKDTree

    d, j = cKDTree(A[:, :2]).query(B[:, :2])
    keep = d > A[j, 2] + B[:, 2] + 0.15              # même arbre vu par les deux méthodes, centres un peu décalés
    return np.concatenate([A, B[keep]])


def load_splat_points(ply, ply_to_phys):
    """Gaussiennes d'un PLY 3DGS (repère du PLY) → positions physiques, opacité activée, plus grande échelle."""
    with open(ply, "rb") as f:
        props, n = [], 0
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                props.append((line.split()[-1], "<f4"))
            elif line == "end_header":
                break
        G = np.fromfile(f, dtype=np.dtype(props), count=n)
    Q = (np.stack([G["x"], G["y"], G["z"]], 1) @ np.asarray(ply_to_phys, np.float32).T).astype(np.float32)
    op = (1 / (1 + np.exp(-G["opacity"]))).astype(np.float32)
    sc = np.exp(np.stack([G["scale_0"], G["scale_1"], G["scale_2"]], 1)).max(1).astype(np.float32)
    return Q, op, sc


def splat_density(xy, r, Q, hq, op, sc, h_band=(0.3, 2.0), op_min=0.2, sc_max=0.3, tree=None):
    """Gaussiennes opaques et petites par m² dans les disques (xy, r), tranche h_band au-dessus du sol."""
    from scipy.spatial import cKDTree

    if tree is None:
        m = (hq > h_band[0]) & (hq < h_band[1]) & (op > op_min) & (sc < sc_max)
        tree = cKDTree(Q[m, :2])
    r = np.broadcast_to(np.asarray(r, float), (len(xy),))
    return np.array([len(tree.query_ball_point(p, rr)) for p, rr in zip(xy, r)]) / (np.pi * r ** 2)


def low_obstacles(P, h, lo, res, shape, Q, hq, op, sc, stems, free_xy, h_band=(0.3, 1.2), min_points=8,
                  min_slices=5, splat_min=1500.0, free_radius=0.35):
    """Obstacles autres que les troncs (souches, troncs couchés, rochers, buissons denses) : cellules de `res` m vues
    À LA FOIS par le LiDAR (≥ `min_points` points dans h_band, étalés sur ≥ `min_slices` tranches de 10 cm) et par le
    splat (≥ `splat_min` gaussiennes opaques par m² dans la même tranche, voisinage 3×3). Le LiDAR seul gardait les
    traînées des personnes qui marchaient à côté du robot (absentes du splat : murs invisibles). → masque (nx, ny)."""
    import cv2
    from scipy.ndimage import uniform_filter
    from scipy.spatial import cKDTree

    nx, ny = shape
    m = (h > h_band[0]) & (h < h_band[1])
    ij = np.floor((P[m, :2] - lo) / res).astype(np.int64)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    cell = ij[ok, 0] * ny + ij[ok, 1]
    n_pts = np.bincount(cell, minlength=nx * ny).reshape(nx, ny)
    n_sl = int(round((h_band[1] - h_band[0]) / 0.1))
    sl = np.zeros((nx * ny, n_sl), bool)
    sl[cell, np.clip(((h[m][ok] - h_band[0]) / 0.1).astype(int), 0, n_sl - 1)] = True
    lidar = (n_pts >= min_points) & (sl.sum(1).reshape(nx, ny) >= min_slices)
    ms = (hq > h_band[0]) & (hq < h_band[1]) & (op > 0.2) & (sc < 0.3)
    ijq = np.floor((Q[ms, :2] - lo) / res).astype(np.int64)
    okq = (ijq[:, 0] >= 0) & (ijq[:, 0] < nx) & (ijq[:, 1] >= 0) & (ijq[:, 1] < ny)
    nq = np.bincount(ijq[okq, 0] * ny + ijq[okq, 1], minlength=nx * ny).reshape(nx, ny).astype(np.float32)
    splat = uniform_filter(nq, size=3) / (res * res) >= splat_min
    obs = lidar & splat
    xx, yy = np.meshgrid(lo[0] + res * (np.arange(nx) + 0.5), lo[1] + res * (np.arange(ny) + 0.5), indexing="ij")
    centers = np.stack([xx.ravel(), yy.ravel()], 1)
    if free_xy is not None:
        obs &= (cKDTree(free_xy[:, :2]).query(centers)[0] >= free_radius).reshape(nx, ny)
    if len(stems):                                            # déjà couvert par un tronc
        d, j = cKDTree(stems[:, :2]).query(centers)
        obs &= (d > stems[j, 2] + 0.1).reshape(nx, ny)
    return cv2.morphologyEx(obs.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)


def splat_support(stems, Q, hq, op, sc, h_band=(0.3, 2.0), op_min=0.2, sc_max=0.3):
    """Présence du tronc dans le splat : gaussiennes opaques et petites dans le disque r + 8 cm, rapportées à la
    densité d'un anneau voisin (r + 0,3 … r + 0,6 m). → (n_dedans, contraste)."""
    from scipy.spatial import cKDTree

    m = (hq > h_band[0]) & (hq < h_band[1]) & (op > op_min) & (sc < sc_max)
    tree = cKDTree(Q[m, :2])
    n_in, contrast = [], []
    for x, y, r, *_ in stems:
        a_in = np.pi * (r + 0.08) ** 2
        a_out = np.pi * ((r + 0.6) ** 2 - (r + 0.3) ** 2)
        k_in = len(tree.query_ball_point((x, y), r + 0.08))
        k_out = len(tree.query_ball_point((x, y), r + 0.6)) - len(tree.query_ball_point((x, y), r + 0.3))
        n_in.append(k_in)
        contrast.append((k_in / a_in) / (k_out / a_out + 1.0))
    return np.array(n_in), np.array(contrast)


SOURCES = ("tronc (cercles)", "tronc (continuité)", "obstacle bas")


def build_obstacles(P, terrain, splat=None, free_xy=None, path_margin=0.2, invisible_max=300.0, cover_min=100.0,
                    block=0.2, low_free_radius=0.5):
    """Obstacles de collision : troncs (cercles par tranche + continuité verticale) et obstacles bas (LiDAR ET splat).

    P : points LiDAR (repère physique) ; terrain : dict H, lo, res (sol) ; splat : (Q, op, sc) ou None ; free_xy :
    trajet réel (le robot y est passé).
    Un tronc est retiré s'il est impossible (bord à < `path_margin` du trajet réel : il traverserait le corps de
    l'ANYmal, ~0,55 m de large) ou franchement invisible dans une
    zone où le splat existe (< `invisible_max` gaussiennes/m² dans son disque, alors que le sol alentour est
    reconstruit : ≥ `cover_min` gaussiennes/m² dans 1,5 m). Repères HÖB-1 : sol nu près du sentier 585/m² (p95),
    troncs ~9 000/m² (médiane).
    → cylindres (x, y, rayon, z_base), source (indice dans SOURCES), masque des obstacles bas, statistiques."""
    from scipy.spatial import cKDTree
    from terrain import height_at

    H, lo, res = terrain["H"], np.asarray(terrain["lo"], float), float(terrain["res"])
    size = np.array(H.shape) * res
    h = P[:, 2] - height_at(terrain, P[:, :2].astype(np.float64))
    A, C = detect_stems(P, h, lo, size)
    B = continuity_stems(P, h, lo, size)
    S = merge_stems(A, B)
    src = np.r_[np.zeros(len(A)), np.ones(len(S) - len(A))].astype(int)
    stats = {"cercles": int(len(C)), "troncs_cercles": int(len(A)), "troncs_continuite": int(len(S) - len(A))}
    keep = np.ones(len(S), bool)
    if free_xy is not None and len(S):
        dp, _ = cKDTree(free_xy[:, :2]).query(S[:, :2])
        imp = dp - S[:, 2] < path_margin
        stats["retires_sur_le_trajet"] = int(imp.sum())
        keep &= ~imp
    Q = op = sc = hq = None
    if splat is not None:
        Q, op, sc = splat
        hq = Q[:, 2] - height_at(terrain, Q[:, :2].astype(np.float64))
        dens = splat_density(S[:, :2], S[:, 2] + 0.08, Q, hq, op, sc)
        gnd = (hq > -0.3) & (hq < 0.3) & (op > 0.2)
        cov = splat_density(S[:, :2], 1.5, Q, hq, op, sc, tree=cKDTree(Q[gnd, :2]))
        inv = (cov >= cover_min) & (dens < invisible_max)
        stats["retires_invisibles_splat"] = int((inv & keep).sum())
        stats["troncs_vus_splat"] = int((keep & ~inv & (dens >= 1500)).sum())
        keep &= ~inv
    S, src = S[keep], src[keep]
    zb = np.min([height_at(terrain, S[:, :2] + off * S[:, 2:3]) for off in
                 ([0, 0], [1, 0], [-1, 0], [0, 1], [0, -1])], axis=0) if len(S) else np.zeros(0)
    cyl = np.c_[S[:, :3], zb] if len(S) else np.zeros((0, 4))
    low = np.zeros(H.shape, bool)
    if splat is not None:
        # couloir de 0,5 m (centres de cellule) : près du trajet, les « obstacles bas » sont surtout des branchages et
        # de la végétation que les pattes de l'ANYmal traversaient ; à 0,35 m, 49 poses du trajet réel avaient un
        # obstacle bas à moins de 0,38 m (demi-largeur du robot à roues)
        low = low_obstacles(P, h, lo, res, H.shape, Q, hq, op, sc, S, free_xy, free_radius=low_free_radius)
        from lidar_terrain import obstacle_cylinders

        lc = obstacle_cylinders(low, H, lo, res, small_area=0.0, step=block)
        cyl = np.concatenate([cyl, lc]) if len(lc) else cyl
        src = np.r_[src, np.full(len(lc), 2)]
        stats["obstacles_bas_cylindres"] = int(len(lc))
    stats["troncs"] = int((src < 2).sum())
    stats["rayon_median_m"] = round(float(np.median(S[:, 2])), 3) if len(S) else None
    return cyl.astype(np.float32), src.astype(np.int8), low, stats
