"""M4 — splits spatiaux + vignettes d'annotation (tourne sur la desktop, sans Modal).

Pré-enregistré : les rectangles test/val ci-dessous sont figés AVANT toute annotation.
Ajuster les coordonnées uniquement en regardant l'étendue de la carte, jamais les labels.

Étapes :
  1. table des superpoints : centroïdes, taille, groupe M2, incertitude, features DINO
     dispo (restriction : seuls les sp avec cnt_all > 0 sont éligibles, sinon l'ablation D
     serait pénalisée par des features manquantes et non par leur qualité) ;
  2. zones test / val / train d'après les rectangles ; buffer : un sp test à ≤ 12 cm d'un
     sp train candidat est retiré du pool test ;
  3. échantillonnage : test = stratifié par groupe M2 (~120), train = 60 % actif
     (incertitude M2) + 40 % stratifié (~280 au total) ;
  4. écrit out/m4/splits.json, out/m4/labels.csv (pass 1 à remplir + 30 répétitions
     pass 2 en fin de fichier, ordre mélangé) et out/m4/README.txt (consignes) ;
  5. vignettes : sélection de vues par projection des VOXELS du sp (même géométrie que
     M2/M3 : incidence par voxel cos ≥ 0.25, marge 30 px, ≤ 55 m, caméra pointant vers
     le sp) ; cercle rouge sur la médiane des voxels projetés, rayon = étalement réel ;
     en-tête neutre (taille, altitude, nb de vues) — AUCUN rendu M2/DINO (aveugle).

Lancement (desktop) :
  cd /home/ambroise/dino && setsid nohup venv/bin/python splatpipe/stage2/m4_splits_sheets.py \
      > m4.log 2>&1 < /dev/null &
Sorties : out/m4/splits.json, out/m4/labels.csv, out/m4/README.txt, out/m4/sheets/sp_<id>.png
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/ambroise/dino")
from splatpipe.stage1.maptools import pack_ijk

DATA_DIR = os.environ.get("DINO_DATA_DIR", "/home/ambroise/dino/data_local/scene")
OUT_DIR = os.environ.get("DINO_OUT_DIR", "/home/ambroise/dino/out")
M4_DIR = f"{OUT_DIR}/m4"
RAW_DIR = f"{DATA_DIR}/rawimgs"
SP_NPZ = f"{DATA_DIR}/reports/stage2_material/superpoints.npz"
M2_SEM = f"{DATA_DIR}/reports/stage2_material/m2/superpoint_semantics.npz"
FEAT_NPZ = f"{OUT_DIR}/superpoint_features_desktop.npz"
MAP_PLY = f"{DATA_DIR}/map/map_5cm_static.ply"

# --- rectangles pré-enregistrés (à ajuster sur l'étendue de la carte UNIQUEMENT) ---
RECT_TEST_A = (-88.0, -64.0, -56.0, -34.0)   # (x0, x1, y0, y1) bâtiment moderne + racks nord
RECT_TEST_B = (-78.0, -64.0, -8.0, 8.0)      # cour nord / église
RECT_VAL = (-92.0, -78.0, 6.0, 22.0)         # validation (non annotée en passe 1)

BUFFER_M = 0.12
N_TEST_TARGET = 120
N_TRAIN_TARGET = 280
MIN_TEST_PER_GROUP = 6
MIN_TRAIN_PER_GROUP = 4
N_REPEATS = 30
SURFACE_MIN = 20
RANDOM_SEED = 42
MAX_RANGE_M = 55.0
COS_MIN = 0.25          # incidence par voxel (même seuil que M2/M3)
# NB : les images fisheye sont des recadrages 1440x1080 du cercle ~193° — le champ utile
# n'est que ~116°x88° ; on classe donc les candidates par pointing DÉCROISSANT (les vues
# de face, parfois lointaines, sont les seules qui survivent au test par voxels).
N_CAND_PER_CAM = 12     # 1er palier de candidates par caméra ; 40 en repli pour les sps difficiles
PASS_GAP_S = 3.0        # étalement temporel : images de passages distincts de la marche
VOX_FRAC_TIERS = (0.25, 0.08)  # part visible minimale des voxels : palier principal, repli
HALF_CROP = 200
OCCL_R = 0.8            # rayon du corridor d'occluseurs autour du segment caméra -> sp
BIT_SKY, BIT_PERSON = 1, 2

GROUP_NAMES = {0: "sky", 1: "person", 2: "bike", 3: "vehicle", 4: "foliage", 5: "ground-naturel",
               6: "ground-pave", 7: "batiment", 8: "vitrage", 9: "grille-barriere", 10: "poteaux",
               11: "roche-eau", 12: "mobilier-urbain", 13: "autre"}

MAPPING_M2_MATERIAL = {
    "foliage": "foliage", "ground-naturel": "grass", "ground-pave": "concrete",
    "batiment": "concrete", "bike": "metal", "vehicle": "metal", "poteaux": "metal",
    "grille-barriere": "metal", "roche-eau": "rock", "mobilier-urbain": "metal",
    "sky": "other", "person": "other", "vitrage": "glass", "autre": "other",
}

README = """M4 — annotation (aveugle) de superpoints
=========================================

Pour chaque sp_id de labels.csv (colonne material à remplir) :
  1. ouvrir out/m4/sheets/sp_<id>.png — jusqu'à 4 vues ; RANGÉE HAUTE = zoom résolution
     native, le superpoint est entouré en rouge ; RANGÉE BASSE = la même vue entière
     (contexte) avec le badge "x % du sp visible" ;
  2. attribuer le MATÉRIAU dominant du patch entouré (pas l'objet) :
       foliage   — feuillage d'arbre ou d'arbuste (organique, feuilles)
       trunk     — troncs, branches lignifiées (écorce)
       grass     — herbe, terre, mousse (sol naturel)
       metal     — métal peint ou brut : vélos, grilles, poteaux, mobilier, voitures
       concrete  — béton, pierre de taille, pavé, bitume, brique PEINTE non reconnaissable
       brick     — brique visible (terre cuite)
       rock      — roche, pierre brute, murs en pierre apparente
       glass     — vitrage (fenêtres, vérandas)
       other     — rien d'autre ne convient (plastique inconnu, mixte illisible, ciel…)
  3. si le patch est illisible ou mixte sans dominant : other + unsure=1 ;
  4. unsure=1 dès que vous hésitez, notamment si le badge indique une faible part
     visible (< 30 %) ou si le sp est à cheval sur deux matériaux ;
  5. les ~30 dernières lignes (pass=2) sont des répétitions : re-annotez sans regarder
     vos réponses pass 1 (idéalement un autre jour) ;
  6. si sp_<id>.png n'existe pas : laisser la ligne vide (sp sans vue image exploitable).

Le consensus M2 et DINO ont volontairement été cachés : annotez ce que VOUS voyez.
"""


def read_ply_points(path, fields):
    """PLY binaire little-endian (copie locale ; superpoints.py importe Modal)."""
    ply_t = {"float": "f4", "double": "f8", "uchar": "u1", "char": "i1", "ushort": "u2",
             "short": "i2", "uint": "u4", "int": "i4"}
    with open(path, "rb") as f:
        lines = []
        while True:
            ln = f.readline()
            if not ln:
                raise ValueError("en-tete PLY inacheve")
            lines.append(ln)
            if ln.strip() == b"end_header":
                break
        offset = f.tell()
    fmt, n, props = None, 0, []
    for ln in lines:
        t = ln.decode("ascii", "replace").strip().split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element" and t[1] == "vertex":
            n = int(t[2])
        elif t[0] == "property":
            props.append((t[2], ply_t[t[1]]))
    assert fmt == "binary_little_endian", fmt
    arr = np.fromfile(path, dtype=np.dtype(props), count=n, offset=offset)
    return {k: arr[k] for k in fields}


def in_rect(x, y, r):
    return (x >= r[0]) & (x <= r[1]) & (y >= r[2]) & (y <= r[3])


def stratified_sample(cand_idx, groups, n_target, min_per_group, rng):
    chosen = []
    pool = np.array(cand_idx, dtype=np.int64)
    for g in np.unique(groups[pool]):
        gidx = pool[groups[pool] == g]
        take = min(min_per_group, len(gidx))
        chosen.extend(rng.choice(gidx, take, replace=False).tolist())
    chosen = np.unique(np.array(chosen, dtype=np.int64))
    rest = np.setdiff1d(pool, chosen)
    n_rest = max(0, n_target - len(chosen))
    if n_rest and len(rest):
        chosen = np.unique(np.concatenate([chosen, rng.choice(rest, min(n_rest, len(rest)), replace=False)]))
    return chosen


def build_cam_views():
    """Par caméra : poses (positions + rotations), modèle fisheye, chemin jpg par stamp."""
    import yaml

    from splatpipe.common.camera_model import FisheyeCamera

    with open(f"{DATA_DIR}/poses/cameras.json") as f:
        poses = json.load(f)["cameras"]
    with open(f"{DATA_DIR}/sensors.yaml") as f:
        sensors = yaml.safe_load(f)
    views = {}
    for c in sorted(poses):
        stamps = sorted(poses[c])
        pos = np.empty((len(stamps), 3), np.float64)
        rot = np.empty((len(stamps), 3, 3), np.float64)
        for i, k in enumerate(stamps):
            T = np.asarray(poses[c][k]["T_map_cam_start"], np.float64)
            pos[i] = T[:3, 3]
            rot[i] = T[:3, :3].T      # carte→caméra : T_map_cam stocke caméra→carte (cf. pose_convention_check.py)
        stamp2rel = {}
        with open(f"{DATA_DIR}/cameras/{c}/index.csv") as f:
            for row in csv.DictReader(f):
                stamp2rel[str(int(row["stamp_ns"]))] = os.path.basename(row["rel_path"])
        views[c] = {
            "stamps": stamps, "pos": pos, "rot": rot,
            "model": FisheyeCamera.from_dict(sensors["cameras"][c]),
            "jpg": {k: f"{RAW_DIR}/{c}/{stamp2rel[k]}" for k in stamps if k in stamp2rel},
        }
    return views


def build_spatial_grid(xyz_g):
    """Index exact des voxels par cellule 1 m (clé entière inversible) pour extraire
    rapidement les occluseurs proches d'un segment caméra -> sp."""
    assert np.all(np.abs(xyz_g[:, :2]) < 500) and np.all((xyz_g[:, 2] > -500) & (xyz_g[:, 2] < 500))
    cell = np.floor(xyz_g).astype(np.int64)
    key = (((cell[:, 0] + 512) << 20) | ((cell[:, 1] + 512) << 10) | (cell[:, 2] + 512))
    o2 = np.argsort(key, kind="stable")
    ks = key[o2]
    ukeys, starts_g = np.unique(ks, return_index=True)
    return {"o2": o2, "ukeys": ukeys, "starts": starts_g,
            "counts": np.diff(np.r_[starts_g, len(ks)]).astype(np.int64)}


def capsule_voxels(grid, cc, cxy, r=OCCL_R):
    """Indices (lignes de xyz_g) des voxels à ≤ r du segment cc -> cxy (vectorisé).

    Préfiltre grossier (cellules 1 m le long du segment) : le test précis est fait
    ensuite par le cône par voxel, donc le sur-ensemble est inoffensif.
    """
    seg = cxy - cc
    n = max(int(np.linalg.norm(seg)) + 1, 2)
    steps = cc + seg[None, :] * np.linspace(0.0, 1.0, n)[:, None]
    cells = np.floor(steps).astype(np.int64)
    nb = np.array([[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)])
    q = (cells[:, None, :] + nb[None, :, :]).reshape(-1, 3)
    keys = ((q[:, 0] + 512) << 20) | ((q[:, 1] + 512) << 10) | (q[:, 2] + 512)
    uk, st, ct, o2 = grid["ukeys"], grid["starts"], grid["counts"], grid["o2"]
    j = np.searchsorted(uk, keys)
    j = np.clip(j, 0, len(uk) - 1)
    valid = uk[j] == keys
    if not valid.any():
        return np.empty(0, np.int64)
    j = j[valid]
    ct_v, st_v = ct[j], st[j]
    total = int(ct_v.sum())
    offs = np.repeat(np.cumsum(ct_v) - ct_v, ct_v)
    rows = np.repeat(st_v, ct_v) + (np.arange(total) - offs)
    cand = np.unique(o2[rows])
    p = grid["xyz"][cand]
    L2 = max(float(seg @ seg), 1e-9)
    t = np.clip(((p - cc) @ seg) / L2, 0.0, 1.0)
    proj = cc + t[:, None] * seg
    d2 = np.einsum("ij,ij->i", p - proj, p - proj)
    return cand[d2 <= max(r, 0.5) ** 2]


def mask_of(views, c, k, cache):
    """Masque sémantique plein cadre (ciel/personne) pour l'image (c, k), cache FIFO."""
    if (c, k) not in cache:
        if len(cache) > 120:
            cache.pop(next(iter(cache)))
        p = f"{DATA_DIR}/masks/{c}/{k}.png"
        from PIL import Image
        cache[(c, k)] = np.asarray(Image.open(p)) if os.path.exists(p) else None
    return cache[(c, k)]


def pick_views_voxels(views, cxy, vox, nrms, grid, mask_cache, n_want=4, detail=None):
    """Sélection de vues par projection des voxels du sp, avec occlusion par cône exact.

    Pour chaque caméra : images à ≤ 55 m dont l'axe optique pointe vers le sp (classement
    par pointing décroissant), 12 puis 40 images réparties sur des passages distincts.
    Une vue est acceptée si ≥ max(6, 8 %) des voxels sont projetés dans l'image (marge
    30 px) avec incidence cos ≥ 0.25, hors masque ciel/personne, et sans occluseur
    persistant (n_scans ≥ 3) sur son rayon à ±w (w = 0.005 + empreinte pixel 0.0033·t) à
    ≥ 15 cm devant lui — sinon le cercle se pose sur l'occluseur et non sur le sp.
    Les vues sont triées par fraction visible décroissante (le badge de la feuille
    l'affiche) et prises en rotation entre caméras.
    Retourne [(cam, stamp, u_med, v_med, r_px, frac_visible)] ; si `detail` est une
    liste, y ajoute aussi (cam, stamp, u_med, v_med, r_px, frac, distance_m,
    uv_voxels_visibles) pour dessiner l'empreinte voxel (m4_sheets_dots.py).
    """
    nv = len(vox)
    if nv > 800:   # sps géants : test sur sous-échantillon (la fraction est un ratio)
        sub = np.sort(np.random.default_rng(0).choice(nv, 800, replace=False))
        vox, nrms = vox[sub], nrms[sub]
    else:
        sub = None
    nv = len(vox)
    per_cam = {}
    for c, v in views.items():
        d = np.linalg.norm(v["pos"] - cxy, axis=1)
        dinv = 1.0 / np.maximum(d, 1e-9)
        # rot = R carte->caméra (transposée de T_map_cam[:3,:3] à la lecture) : pcf = (p - c) @ rot.T,
        # axe optique en carte = 3e LIGNE de rot = 3e COLONNE de T_map_cam[:3,:3]
        pointing = ((cxy - v["pos"]) * dinv[:, None] * v["rot"][:, 2, :]).sum(1)
        model = v["model"]
        acc, seen = [], set()
        for n_cand in (N_CAND_PER_CAM, 40):
            if len(acc) >= 3:
                break
            ok = np.flatnonzero(d <= MAX_RANGE_M)
            ok = ok[np.lexsort((d[ok], -pointing[ok]))]     # pointing desc, puis distance
            # étalement temporel : première puis images de passages distincts
            picked_idx: list[int] = []
            for j in ok:
                tj = int(v["stamps"][j])
                if all(abs(tj - int(v["stamps"][j2])) >= PASS_GAP_S * 1e9 for j2 in picked_idx):
                    picked_idx.append(j)
                if len(picked_idx) >= n_cand:
                    break
            if len(picked_idx) < 6:
                picked_idx = list(ok[:n_cand])
            for i in picked_idx:
                if v["stamps"][i] in seen:
                    continue
                seen.add(v["stamps"][i])
                R, cc = v["rot"][i], v["pos"][i]
                pcf = (vox - cc) @ R.T
                depth = np.sqrt(np.einsum("ij,ij->i", pcf, pcf))
                view_w = (pcf / depth[:, None]) @ R              # direction cam -> sp (monde)
                cos_i = np.einsum("ij,ij->i", nrms, -view_w)     # incidence
                uv, valid, _ = model.project(pcf, margin=30)
                # --- occlusion : test de cône par voxel (voir docstring) ; pré-filtre
                #     angulaire (cône englobant le sp) + persistance n_scans >= 3 contre
                #     le bruit résiduel de la carte.
                ui = uv[:, 0].round().astype(np.int64)
                vi_ = uv[:, 1].round().astype(np.int64)
                libre = np.ones(nv, bool)
                occ = capsule_voxels(grid, cc, cxy, r=0.8)
                occ = occ[grid["nscans"][occ] >= 3]
                if len(occ):
                    pcf_o = (grid["xyz"][occ] - cc) @ R.T
                    pcf_o = pcf_o[pcf_o[:, 2] > 0.05]
                    pcf_c = (cxy - cc) @ R.T
                    d_cen = float(np.linalg.norm(pcf_c))
                    axe = pcf_c / d_cen
                    sp_rad = float(np.linalg.norm(vox - cxy, axis=1).max())
                    ang = np.arctan2(sp_rad + 0.8, d_cen) + 0.01
                    n2_o = np.einsum("ij,ij->i", pcf_o, pcf_o)
                    pcf_o = pcf_o[(pcf_o @ axe) >= np.cos(ang) * np.sqrt(n2_o)]
                    n2_o = np.einsum("ij,ij->i", pcf_o, pcf_o)
                    for k0 in range(0, nv, 64):
                        sl = slice(k0, min(k0 + 64, nv))
                        uu = pcf[sl] / depth[sl, None]               # (B,3) rayons unitaires
                        t = uu @ pcf_o.T                             # (B,M) abscisses
                        lat2 = n2_o[None, :] - t * t
                        w = 0.005 + 0.0033 * t
                        bloque = (t > 0.3) & (t < depth[sl, None] - 0.15) & (lat2 < w * w)
                        libre[sl] = ~bloque.any(1)
                # --- masques ciel/personne : un sp projeté sur le ciel est vu par la tranche
                msk = mask_of(views, c, v["stamps"][i], mask_cache)
                if msk is not None:
                    nonsky = (msk[vi_.clip(0, msk.shape[0] - 1), ui.clip(0, msk.shape[1] - 1)]
                              & (BIT_SKY | BIT_PERSON)) == 0
                else:
                    nonsky = np.ones(nv, bool)
                good = valid & (cos_i >= COS_MIN) & libre & nonsky
                nok = int(good.sum())
                if nok < max(6, int(0.08 * nv)):
                    continue
                uvv = uv[good]
                u_med, v_med = float(np.median(uvv[:, 0])), float(np.median(uvv[:, 1]))
                spread = max(np.percentile(uvv[:, 0], 90) - np.percentile(uvv[:, 0], 10),
                             np.percentile(uvv[:, 1], 90) - np.percentile(uvv[:, 1], 10))
                r_px = float(np.clip(spread / 2 + 8, 12, 80))
                acc.append((float(d[i]), v["stamps"][i], u_med, v_med, r_px, nok / nv, uvv))
        if acc:
            per_cam[c] = sorted(acc, key=lambda e: (-e[5], e[0]))
    out, di = {}, 0
    while len(out) < n_want and len(out) < sum(len(a) for a in per_cam.values()):
        progressed = False
        for c in sorted(per_cam):
            if di < len(per_cam[c]):
                out[(c, per_cam[c][di][1])] = per_cam[c][di]
                progressed = True
                if len(out) == n_want:
                    break
        if not progressed:
            break
        di += 1
    if detail is None:      # comportement historique : 6-tuples
        return [(c, k, u, v2, r, fr)
                for (c, k), (d, _k, u, v2, r, fr, _uv) in out.items()]
    ext = [(c, k, u, v2, r, fr, float(d), np.asarray(uv))
           for (c, k), (d, _k, u, v2, r, fr, uv) in out.items()]
    detail.extend(ext)
    return ext


def sp_has_view(views, cxy, vox, nrms, grid, mask_cache):
    """True si au moins une vue passe le test complet (sans toucher aux jpg)."""
    return len(pick_views_voxels(views, cxy, vox, nrms, grid, mask_cache, n_want=1)) > 0


def main() -> None:
    from PIL import Image, ImageDraw, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True   # jpg tronqués tolérés
    from scipy.spatial import cKDTree

    os.makedirs(f"{M4_DIR}/sheets", exist_ok=True)

    # --- table superpoints.
    # sp_id (superpoints.npz) est aligné aux LIGNES du PLY (fichier déjà trié par clés
    # pack_ijk) mais non monotone (ids attribués dans un autre parcours) : regrouper
    # par argsort(sp_id) — jamais reduceat sur l'ordre fichier.
    cols = read_ply_points(MAP_PLY, ["x", "y", "z", "nx", "ny", "nz", "n_scans"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    nrm = np.stack([cols["nx"], cols["ny"], cols["nz"]], 1).astype(np.float64)
    nsc = cols["n_scans"].astype(np.int64)
    keys = pack_ijk(np.floor(xyz / 0.05).astype(np.int64))
    o = np.argsort(keys, kind="stable")
    xyz_s, nrm_s, nsc_s = xyz[o], nrm[o], nsc[o]
    del xyz, nrm, cols
    with np.load(SP_NPZ) as z:
        sp_id = z["sp_id"]
    n_sp = int(sp_id.max()) + 1
    cnt = np.bincount(sp_id, minlength=n_sp)
    og = np.argsort(sp_id, kind="stable")
    sp_g = sp_id[og]
    starts = np.flatnonzero(np.r_[True, sp_g[1:] != sp_g[:-1]])
    assert sp_g[0] == 0 and len(starts) == n_sp, "sp_id non aligné aux lignes du PLY ?"
    cen = np.zeros((n_sp, 3))
    cen[sp_g[starts]] = np.add.reduceat(xyz_s[og], starts, axis=0) / np.maximum(cnt, 1)[:, None]
    xyz_g, nrm_g, nsc_g = xyz_s[og], nrm_s[og], nsc_s[og]   # voxels groupés par sp
    b0, b1 = starts, np.r_[starts[1:], len(sp_g)]
    del xyz_s, nrm_s, nsc_s, sp_g, starts, og, o, keys

    with np.load(M2_SEM) as z:
        dom = z["dominant_group"]
        wsum = z["wsum"].astype(np.float64)
    with np.load(FEAT_NPZ) as z:
        has_feat = z["cnt_all"] > 0

    surface = cnt >= SURFACE_MIN
    seen = cnt > 0
    elig = surface & seen & has_feat

    # --- zones + buffer 12 cm contre le pool train
    x, y = cen[:, 0], cen[:, 1]
    is_test = (in_rect(x, y, RECT_TEST_A) | in_rect(x, y, RECT_TEST_B)) & elig
    is_val = in_rect(x, y, RECT_VAL) & elig & ~is_test
    is_train_cand = elig & ~is_test & ~is_val
    tree = cKDTree(cen[is_train_cand])
    d, _ = tree.query(cen[is_test], distance_upper_bound=BUFFER_M + 1e-9)
    conflict = d <= BUFFER_M
    test_rows = np.flatnonzero(is_test)
    is_test[test_rows[conflict]] = False
    print(f"zones : test {int(is_test.sum())} (après buffer), val {int(is_val.sum())}, "
          f"train cand {int(is_train_cand.sum())}", flush=True)

    # --- vignettes : modèles caméra + index spatial d'occlusion avant la sélection
    views = build_cam_views()
    grid = build_spatial_grid(xyz_g)
    grid["xyz"] = xyz_g
    grid["nscans"] = nsc_g       # persistance des voxels (>= 3 scans = structure reelle)          # accès aux coordonnées des occluseurs depuis les helpers
    mask_cache = {}

    # --- échantillonnage itératif : écarter les sps sans aucune vue projetable
    # (un sp invisible des images est hors du domaine évalué, pour toutes les ablations ;
    # il est remplacé dans le même pool, test inclus, puis documenté dans splits.json)
    def sample_selection(excl):
        rng2 = np.random.default_rng(RANDOM_SEED)
        test_idx2 = stratified_sample(np.flatnonzero(is_test & ~excl), dom,
                                      N_TEST_TARGET, MIN_TEST_PER_GROUP, rng2)
        is_test_sel2 = np.zeros(n_sp, bool)
        is_test_sel2[test_idx2] = True
        train_pool2 = np.flatnonzero(is_train_cand & ~excl & ~is_test_sel2)
        share = np.clip(wsum / np.maximum(cnt, 1), 0, 1)
        unc2 = 1.0 - share
        unc2[~seen] = -1.0
        order2 = train_pool2[np.argsort(-unc2[train_pool2])]
        n_active2 = int(0.6 * N_TRAIN_TARGET)
        active_sel2 = order2[:n_active2]
        rest2 = np.setdiff1d(train_pool2, active_sel2)
        divers2 = stratified_sample(rest2, dom, N_TRAIN_TARGET - len(active_sel2),
                                    MIN_TRAIN_PER_GROUP, rng2)
        return test_idx2, np.unique(np.concatenate([active_sel2, divers2])), len(active_sel2)

    rng = np.random.default_rng(RANDOM_SEED)
    exclude = np.zeros(n_sp, bool)
    dropped_invisible = []
    annot_ok = {}          # cache : un sp jugé annotable reste annotable (vues identiques)
    for round_i in range(6):
        test_idx, train_sel, n_active = sample_selection(exclude)
        sel_all = sorted(set(test_idx.tolist()) | set(train_sel.tolist()))
        bad = []
        for spi in sel_all:
            if spi not in annot_ok:
                annot_ok[spi] = sp_has_view(views, cen[spi], xyz_g[b0[spi]:b1[spi]],
                                            nrm_g[b0[spi]:b1[spi]], grid, mask_cache)
            if not annot_ok[spi]:
                bad.append(spi)
        print(f"tour {round_i + 1} : {len(sel_all)} sélectionnés, {len(bad)} sans vue", flush=True)
        if not bad:
            break
        exclude[np.array(bad, dtype=np.int64)] = True
        dropped_invisible.extend(bad)
    sel_all = sorted(set(test_idx.tolist()) | set(train_sel.tolist()))
    is_test_sel = np.zeros(n_sp, bool)
    is_test_sel[test_idx] = True
    is_train_sel = np.zeros(n_sp, bool)
    is_train_sel[train_sel] = True
    print(f"sélection : test {int(is_test_sel.sum())}, train {int(is_train_sel.sum())} "
          f"(actif {n_active}), écartés sans vue : {len(dropped_invisible)}", flush=True)

    # --- répartition par groupe (contrôle)
    per_group = {}
    for zone, mask in (("test", is_test_sel), ("train", is_train_sel), ("val", is_val)):
        u, c_ = np.unique(dom[mask], return_counts=True)
        per_group[zone] = {GROUP_NAMES.get(int(g), str(g)): int(n_) for g, n_ in zip(u, c_)}
    splits = {
        "rectangles": {"test_a": RECT_TEST_A, "test_b": RECT_TEST_B, "val": RECT_VAL},
        "buffer_m": BUFFER_M, "seed": RANDOM_SEED,
        "test_sp": test_idx.tolist(), "train_sp": train_sel.tolist(),
        "n_val_sp": int(is_val.sum()),
        "dropped_invisible_sp": dropped_invisible,
        "per_group": per_group,
        "mapping_m2_to_material": MAPPING_M2_MATERIAL,
        "notes": ("pré-enregistré avant annotation (cf. m4-m5-plan.md) ; éligibilité : "
                  "surface ≥ 20 voxels, cnt_all > 0 (features DINO dispo, sinon l'ablation D "
                  "mesurerait des features manquantes et non leur qualité) ; les sps sans "
                  "aucune vue image projetable sont écartés (remplacés dans le même pool) : "
                  "un sp invisible des images est hors du domaine évalué ici, pour toutes "
                  "les ablations"),
    }
    with open(f"{M4_DIR}/splits.json", "w") as f:
        json.dump(splits, f, indent=1)

    # --- labels.csv : pass 1 trié par sp_id, puis 30 répétitions pass 2 mélangées en fin
    repeat_ids = list(map(int, rng.permutation(
        rng.choice(np.array(sel_all, dtype=np.int64), N_REPEATS, replace=False))))
    with open(f"{M4_DIR}/labels.csv", "w") as f:
        f.write("sp_id,material,unsure,pass,date\n")
        for i in sel_all:
            f.write(f"{i},,,1,\n")
        for i in repeat_ids:
            f.write(f"{i},,,2,\n")
    with open(f"{M4_DIR}/README.txt", "w") as f:
        f.write(README)

    img_cache = {}
    n_sheets = 0
    for spi in sel_all:
        cxy = cen[spi]
        vox = xyz_g[b0[spi]:b1[spi]]
        nrms = nrm_g[b0[spi]:b1[spi]]
        picked = pick_views_voxels(views, cxy, vox, nrms, grid, mask_cache)
        crops = []
        for c, k, u, v2, r, fr in picked:
            if (c, k) not in img_cache:
                jpg = views[c]["jpg"].get(k)
                if not jpg or not os.path.exists(jpg):
                    continue
                img_cache[(c, k)] = Image.open(jpg).convert("RGB")
            img = img_cache[(c, k)]
            W, H = img.size
            u0 = int(np.clip(u - HALF_CROP, 0, W - 2 * HALF_CROP))
            v0 = int(np.clip(v2 - HALF_CROP, 0, H - 2 * HALF_CROP))
            crop = img.crop((u0, v0, u0 + 2 * HALF_CROP, v0 + 2 * HALF_CROP))
            dr = ImageDraw.Draw(crop)
            rr = max(4, int(r))
            dr.ellipse([u - u0 - rr, v2 - v0 - rr, u - u0 + rr, v2 - v0 + rr],
                       outline=(255, 0, 0), width=4)
            full = img.resize((400, 300))          # contexte : la vue entière
            sc = 400.0 / W
            rr2 = max(3, int(r * sc))
            ImageDraw.Draw(full).ellipse([u * sc - rr2, v2 * sc - rr2,
                                          u * sc + rr2, v2 * sc + rr2],
                                         outline=(255, 0, 0), width=2)
            ImageDraw.Draw(full).text((6, 282), f"{c} : {fr:.0%} du sp visible",
                                      fill=(255, 255, 0))
            crops.append((c, crop, full))
        if not crops:
            print(f"[warn] sp {spi} : aucune vue projetable", flush=True)
            continue
        sheet = Image.new("RGB", (404 * len(crops), 752), "white")
        for i2, (c, cr, fullim) in enumerate(crops):
            sheet.paste(cr, (404 * i2 + 2, 22))       # zoom résolution native
            sheet.paste(fullim, (404 * i2 + 2, 430))  # contexte vue entière
            ImageDraw.Draw(sheet).text((404 * i2 + 8, 4), c, fill=(0, 0, 0))
        ImageDraw.Draw(sheet).text((8, 734),
                                   f"sp {spi} | z_carte={cxy[2]:.1f} m | {int(cnt[spi])} vox | "
                                   f"{len(crops)} vues", fill=(0, 0, 0))
        sheet.save(f"{M4_DIR}/sheets/sp_{spi}.png")
        n_sheets += 1
        if n_sheets % 50 == 0:
            print(f"vignettes {n_sheets}/{len(sel_all)}", flush=True)

    splits["n_sheets"] = n_sheets
    splits["n_sans_vue"] = len(sel_all) - n_sheets
    with open(f"{M4_DIR}/splits.json", "w") as f:
        json.dump(splits, f, indent=1)
    print(f"M4 prêt : splits.json, labels.csv, {n_sheets} vignettes -> {M4_DIR}", flush=True)


if __name__ == "__main__":
    main()
