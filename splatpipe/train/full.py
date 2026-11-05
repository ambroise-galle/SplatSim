"""Entraînement pleine scène par fenêtres glissantes le long de la trajectoire (4.1) — v3.

Les paramètres pleine carte (14,4 M gaussiennes) restent sur GPU sans gradient ;
chaque fenêtre (rayon `r_opt` autour d'un centre d'arc, tous les `stride_m` mètres)
en copie ses gaussiennes, les optimise sur les balayages contemporains (pertes B0–B4,
cf. losses.py), puis réécrit le résultat dans les tenseurs pleine carte — les fenêtres
consécutives se recouvrent, ce qui assure la continuité.

v3 (25/09) — corrections de la v2, dont les previews étaient de mauvaise qualité :
  1. paramétrisation 3DGS standard : échelles en log, opacités en logit (v2 optimisait les valeurs
     brutes : 10 % d'opacités négatives = gaussiennes mortes, sans gradient → trous noirs) ;
  2. couleur initiale relevée dans les photos (`color_lift.py`) et convertie en SH (v2 prenait la palette
     ALÉATOIRE de visualisation des superpoints, sans conversion RGB→SH) ; degré SH 1 activé après le
     premier tiers des pas, pas d'apprentissage des coefficients d'ordre 1 = 1/20 du DC ;
  3. B2 (espace libre) restreint aux rayons montants hors du secteur de l'opérateur : les rayons bloqués
     par son corps (retours < 1,5 m supprimés) étaient pris pour du vide et effaçaient le sol du chemin ;
  4. perte image 0,8·L1 + 0,2·(1 − SSIM), masques ciel + personne + coins hors du domaine fisheye de gsplat ;
  5. exposition par image (gain et biais par canal), l'appareil étant en exposition automatique ;
  6. perte image limitée aux pixels dont la première surface visible est dans la fenêtre : le rendu ne
     contient que les gaussiennes de la fenêtre, et la v2 demandait aux gaussiennes proches d'expliquer
     les bâtiments lointains ;
  7. décroissance exponentielle du pas sur les positions à l'intérieur de chaque fenêtre.
v3.1 (25/09 soir) — le run pleine scène v3 dégradait les zones traversées par beaucoup de fenêtres (la cour :
8 fenêtres, 23 → 17 dB) : gaussiennes étirées ou géantes (jusqu'à 37 m, dans le ciel), 39 % d'opacités < 0,05 :
  8. pixels de ciel : opacité rendue poussée vers 0 (`--sky-w`) — le ciel était exclu de la perte, une gaussienne
     pouvait y grandir sans pénalité ;
  9. taille plafonnée (`--scale-max`, 30 cm) ;
 10. pas réduit (`--revisit-damp`, ×0,2) pour les gaussiennes déjà optimisées par une fenêtre précédente : chaque
     fenêtre repartait aux pas maximaux, et les passages répétés faisaient dériver la scène.
v3.2 (26/09) — v3.1 ne gagne que 0,5–1,4 dB dans la cour. La trajectoire n'y repasse jamais (aucune image à
< 12 m hors du passage de 243 s) : ce sont les fenêtres voisines (2, 3, 12, 17, 18), dont les images voient la
cour de loin ou pas du tout, qui la dégradent. Adam normalise la taille des pas : des gradients minuscules et
bruités produisent quand même des déplacements normaux. Et le ×0,2 de la v3.1 freinait la fenêtre de la cour
elle-même, ses gaussiennes ayant déjà été « vues » par les fenêtres 2 et 3.
 11. dans chaque fenêtre, seules les gaussiennes BIEN OBSERVÉES par ses images (visibles au z-buffer, à
     ≤ `--obs-max-dist` m, dans ≥ `--min-obs` images) sont mises à jour ; les autres sont rendues mais figées ;
 12. amortissement des revisites désactivé par défaut (`--revisit-damp 1`).
v3.3 (08/10) — capteurs lus dans `sensors.yaml` (LiDAR, caméras fisheye OU sténopé : `common/train_camera.py`),
pour entraîner aussi DiTer++ (Ouster OS1-64 + RealSense redressée) ; comportement Oxford inchangé par défaut.
 13. `--min-obs-lidar N` : une gaussienne touchée par ≥ N balayages d'entraînement de la fenêtre compte comme bien
     observée (avec une seule caméra frontale à champ étroit, la règle 11 figeait 95 % des gaussiennes).
Sortie `scene_full.npz` : valeurs ACTIVÉES (échelles, opacités dans [0, 1], quaternions normés, SH), lisibles
telles quelles par gsplat ; `exposure.npz` : exposition apprise par image.

Lancement (desktop) :
  cd ~/dino && venv/bin/python -m splatpipe.train.full --root data_local/scene \
      --stride-m 20 --steps-per-window 3000 --out-dir out/train_full_v3
  essai local autour d'une pose : ajouter --focus cam0:1710338243391100552
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

RANGE_MAX = 28.0
MATCH_CM = 10.0
C0 = 0.28209479177387814
LR = {"means": 1e-3, "quats": 1e-3, "log_scales": 5e-3, "logit_opac": 5e-2, "sh0": 2.5e-3,
      "shN": 2.5e-3 / 20, "expo": 1e-3}
EXPO_REG = 1e-3
MASK_CELL = 16          # cellule (px) du masque « première surface dans la fenêtre »
SKY_W, SCALE_MAX, REVISIT_DAMP = 0.1, 0.30, 1.0
MIN_OBS, OBS_MAX_DIST = 5, 25.0


def _arc_centers(origins: np.ndarray, stride_m: float) -> np.ndarray:
    """Centres de fenêtres tous les `stride_m` m d'arc le long de la trajectoire."""
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(origins, axis=0), axis=1))])
    total = d[-1]
    centers = []
    t = 0.0
    j = 0
    while t <= total:
        while j < len(d) - 1 and d[j + 1] < t:
            j += 1
        centers.append(origins[j])
        t += stride_m
    return np.asarray(centers)


def _load_voxel_colors(ply_path: str):
    """RGB par voxel d'un PLY coloré aligné aux lignes de la carte. None si absent.
    ⚠️ `superpoints_colored.ply` est une palette aléatoire de visualisation : ne pas l'utiliser comme couleur."""
    if not os.path.exists(ply_path):
        return None
    ply_t = {"float": "f4", "double": "f8", "uchar": "u1", "char": "i1", "ushort": "u2",
             "short": "i2", "uint": "u4", "int": "i4"}
    with open(ply_path, "rb") as f:
        lines = []
        while True:
            ln = f.readline()
            lines.append(ln)
            if ln.strip() == b"end_header":
                break
        off = f.tell()
    n, props = 0, []
    for ln in lines:
        t = ln.decode("ascii", "replace").strip().split()
        if not t:
            continue
        if t[0] == "element" and t[1] == "vertex":
            n = int(t[2])
        elif t[0] == "property":
            props.append((t[2], ply_t[t[1]]))
    arr = np.fromfile(ply_path, dtype=np.dtype(props), count=n, offset=off)
    return np.stack([arr["red"], arr["green"], arr["blue"]], 1).astype(np.float32) / 255.0


def _ssim_map(x, y):
    """SSIM (fenêtre gaussienne 11, σ 1,5) de deux images (3, H, W) → carte (H, W) moyennée sur les canaux."""
    import torch
    import torch.nn.functional as F

    g = torch.exp(-((torch.arange(11, device=x.device, dtype=x.dtype) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    kx = g.view(1, 1, 1, 11).repeat(3, 1, 1, 1)
    ky = g.view(1, 1, 11, 1).repeat(3, 1, 1, 1)

    def blur(z):
        z = F.conv2d(F.pad(z[None], (5, 5, 0, 0), mode="reflect"), kx, groups=3)
        return F.conv2d(F.pad(z, (0, 0, 5, 5), mode="reflect"), ky, groups=3)[0]

    mx, my = blur(x), blur(y)
    sxx = blur(x * x) - mx * mx
    syy = blur(y * y) - my * my
    sxy = blur(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return s.mean(0)


def _b2_allowed_cells(root: str, data, stamps: list[int], min_el_deg: float = 5.0, n_sample: int = 60):
    """Cellules (rang, colonne) où « pas de retour » veut vraiment dire « vide » : rayons montants
    (élévation ≥ min_el_deg), hors du secteur d'azimut bloqué par l'opérateur. Le secteur est mesuré :
    colonnes dont les rayons descendants (qui touchent normalement le sol) sont vides > 30 % du temps."""
    lm = json.load(open(f"{root}/lidar_model/{data.lidar_id}.json"))
    el = np.degrees(np.asarray(lm["row_elevations_rad"], float))
    n_rows, n_cols = int(lm["n_rows"]), int(lm["n_cols"])
    occ = np.zeros((n_rows, n_cols))
    sample = stamps[:: max(1, len(stamps) // n_sample)][:n_sample]
    for st in sample:
        with np.load(f"{root}/{data.deskew_dir}/{st}.npz") as z:
            f = z["ret"] == 0
            o = np.zeros((n_rows, n_cols), bool)
            o[z["row"][f], z["col"][f]] = True
        occ += o
    empty = 1 - occ / len(sample)
    down = el < -10.0
    blocked = empty[down].mean(0) > 0.3
    for _ in range(4):                                   # dilatation de 4 colonnes (azimut circulaire)
        blocked = blocked | np.roll(blocked, 1) | np.roll(blocked, -1)
    allowed = (el[:, None] >= min_el_deg) & (~blocked[None, :])
    info = {"colonnes_operateur": int(blocked.sum()), "rangs_montants": int((el >= min_el_deg).sum()),
            "cellules_b2": int(allowed.sum())}
    return allowed, info


def run_full(root: str, stride_m: float = 20.0, steps_per_window: int = 3000,
             r_opt: float = 35.0, core_m: float = 20.0, holdout_every: int = 10,
             weights: dict[str, float] | None = None, out_dir: str = "out/train_full_v3",
             eval_every: int = 300, seed: int = 0, sh_degree: int = 1, focus: str = "",
             focus_r: float = 30.0, windows_sel: str = "", opacity_init: float = 0.95,
             ssim_w: float = 0.2, exposure: bool = True, b2_min_el: float = 5.0,
             colors_npz: str = "", sky_w: float = SKY_W, scale_max: float = SCALE_MAX,
             revisit_damp: float = REVISIT_DAMP, min_obs: int = MIN_OBS, min_obs_lidar: int = 0,
             obs_max_dist: float = OBS_MAX_DIST) -> dict:
    import torch
    import yaml
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True   # un jpg tronqué (44 o) dans la séquence

    import gsplat
    from splatpipe.common.train_camera import make_camera
    from splatpipe.stage4.overfit_core import _load_lidar_model_root, _load_map_disks
    from splatpipe.train.data import WindowData, lidar_paths
    from splatpipe.train.losses import b1_lidar, b2_free, b3_flatten

    weights = weights or {"b0": 1.0, "b1": 1.0, "b2": 0.05, "b3": 0.05, "b4": 1.0}
    dev = "cuda"
    rng = np.random.default_rng(seed)
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    lidar_model, _ = _load_lidar_model_root(root, dev)
    means, quats, scales, opac, map_nsc = _load_map_disks(dev, f"{root}/map/map_5cm_static.ply")
    n_map = len(means)
    log_scales = torch.log(scales)
    logit_opac = torch.logit(torch.full_like(opac, opacity_init))
    del scales, opac
    # couleur : relevé photo (color_lift.py) → SH DC ; ordres supérieurs à zéro
    colors_npz = colors_npz or f"{root}/reports/stage4/map_colors_lift.npz"
    if os.path.exists(colors_npz):
        col0 = np.load(colors_npz)["rgb"].astype(np.float32)
        color_src = colors_npz
    else:
        col0 = np.full((n_map, 3), 0.5, np.float32)
        color_src = "gris 0.5 (map_colors_lift.npz absent)"
    assert len(col0) == n_map, (len(col0), n_map)
    sh0 = torch.from_numpy(((col0 - 0.5) / C0)[:, None, :]).to(dev)
    K_sh = (sh_degree + 1) ** 2
    shN = torch.zeros(n_map, K_sh - 1, 3, device=dev)
    del col0

    sensors = yaml.safe_load(open(f"{root}/sensors.yaml"))
    lidar_id, deskew_dir = lidar_paths(root)
    data = WindowData(root, sorted(int(x) for x in
                      yaml.safe_load(open(f"{root}/splits.yaml"))["lidars"][lidar_id]["train"]),
                      holdout_every=0)
    cam_K = {c: make_camera(sensors["cameras"][c], root) for c in data.cam_names}
    gvalid = {c: torch.from_numpy(cam_K[c].valid_mask()).to(dev) for c in cam_K}
    print(f"capteurs : LiDAR {lidar_id} ({deskew_dir}), caméras "
          f"{ {c: cam_K[c].model for c in cam_K} }", flush=True)
    stamps_all = [s for s in data.stamps
                  if os.path.exists(f"{root}/{deskew_dir}/{s}.npz")]
    b2_ok_np, b2_info = _b2_allowed_cells(root, data, stamps_all, min_el_deg=b2_min_el)
    b2_ok = torch.from_numpy(b2_ok_np).to(dev)
    print(f"B2 : {json.dumps(b2_info)}", flush=True)

    mids = np.array([data.sweep_light(s) for s in stamps_all])
    origins = data.origins(mids)
    stamp_of_origin = {s: o for s, o in zip(stamps_all, origins)}
    centers = _arc_centers(origins, stride_m)
    print(f"scène : {n_map} gaussiennes, {len(stamps_all)} balayages, "
          f"{len(centers)} fenêtres (pas {stride_m} m) ; couleurs : {color_src}", flush=True)

    # sous-ensemble de fenêtres (essais) : liste explicite ou voisinage d'une pose caméra
    run_idx = list(range(len(centers)))
    if windows_sel:
        run_idx = [int(x) for x in windows_sel.split(",")]
    elif focus:
        cam_f, st_f = focus.split(":")
        p_f = np.asarray(data.poses[cam_f][st_f]["T_map_cam_start"], float)[:3, 3]
        run_idx = [i for i, c in enumerate(centers) if np.linalg.norm(c - p_f) <= focus_r]
    print(f"fenêtres entraînées : {run_idx}", flush=True)

    veg_full = None
    lab = f"{root}/reports/stage2_material/m4/labels_m2_consensus.npz"
    sp_path = f"{root}/reports/stage2_material/superpoints.npz"
    if os.path.exists(lab) and os.path.exists(sp_path):
        mat = np.load(lab)["material"]
        sp_id = np.load(sp_path)["sp_id"]
        veg_full = torch.from_numpy(mat[sp_id] == "foliage").to(dev)
        print(f"B3 : masque végétation chargé ({float(veg_full.float().mean()):.1%} des gaussiennes, non aplaties)", flush=True)
    else:
        # jusqu'au 25/09, ce fichier n'était pas à cet emplacement sur la desktop (il est dans out/m4/) :
        # v2 et le 1er run v3 ont aplati la végétation sans le signaler
        print(f"⚠️ B3 : {lab} absent — végétation NON protégée de l'aplatissement", flush=True)

    windows = []
    for w_i, c in enumerate(centers):
        if w_i not in run_idx:
            continue
        near = torch.nonzero(((means - torch.from_numpy(c).to(dev)) ** 2).sum(1) < r_opt ** 2).ravel()
        win_stamps = sorted(s for s, o in stamp_of_origin.items() if np.linalg.norm(o - c) <= core_m)
        hold = set(win_stamps[::holdout_every])
        train_stamps = [s for s in win_stamps if s not in hold]
        windows.append(dict(index=w_i, center=c, near=near.cpu().numpy(), train=train_stamps,
                            holdout=sorted(hold)))
        print(f"fenêtre {w_i:2d}: {len(train_stamps)} balayages train, "
              f"{len(hold)} holdout, {len(near)} gaussiennes", flush=True)

    expo_store: dict[str, np.ndarray] = {}
    visits = np.zeros(n_map, np.uint8)                               # nb de fenêtres ayant optimisé chaque gaussienne
    def load_img(fr):
        """Image uint8 + masque valide (ni ciel ni personne), décodés à chaque appel, SANS cache : un cache des
        images décodées d'une fenêtre (~750 × 6 Mo) a saturé les 15 Go de RAM de la desktop le 25/09."""
        p_img = f"{root}/rawimgs/{fr['cam']}/{os.path.basename(fr['rel'])}"
        if not os.path.exists(p_img):
            p_img = f"{root}/{fr['rel']}"
        img = np.array(Image.open(p_img).convert("RGB"))
        mp = f"{root}/masks/{fr['cam']}/{fr['stamp']}.png"
        m = np.asarray(Image.open(mp)) if os.path.exists(mp) else np.zeros(img.shape[:2], np.uint8)
        return img, (m & 3) == 0, (m & 1) > 0              # image, valide, ciel

    def expand_mask(mc, cam):
        K = cam_K[cam]
        return mc.repeat_interleave(MASK_CELL, 0).repeat_interleave(MASK_CELL, 1)[:K.height, :K.width]

    def window_pixel_mask(fr, center, r_in, obs=None):
        """Cellules MASK_CELL² dont toutes les surfaces visibles (z-buffer des voxels, ±10 cm) sont à
        ≤ r_in du centre de fenêtre : les seules que les gaussiennes de la fenêtre peuvent expliquer."""
        K = cam_K[fr["cam"]]
        T = torch.tensor(fr["T_wc"], dtype=torch.float32, device=dev)
        d = means - T[:3, 3]
        idx = torch.nonzero((d * d).sum(1) < 80.0 ** 2).squeeze(1)
        pc = d[idx] @ T[:3, :3]
        dep = pc.norm(dim=1)
        u, v, ok = K.project(pc)
        idx, u, v, dep = idx[ok], u[ok], v[ok], dep[ok]
        Wc, Hc = (K.width + MASK_CELL - 1) // MASK_CELL, (K.height + MASK_CELL - 1) // MASK_CELL
        cell = (v.long() // MASK_CELL) * Wc + u.long() // MASK_CELL
        dmin = torch.full((Hc * Wc,), float("inf"), device=dev)
        dmin.scatter_reduce_(0, cell, dep, reduce="amin")
        vis = dep <= dmin[cell] + 0.10
        if obs is not None:                                   # comptage des observations proches (v3.2)
            near_obs = idx[vis & (dep <= obs_max_dist)]
            obs.index_add_(0, near_obs, torch.ones_like(near_obs, dtype=obs.dtype))
        inside = (((means[idx[vis]] - torch.from_numpy(center).float().to(dev)) ** 2).sum(1) <= r_in ** 2).float()
        allin = torch.full((Hc * Wc,), 2.0, device=dev)            # 2 = aucune surface
        allin.scatter_reduce_(0, cell[vis], inside, reduce="amin")
        return (allin == 1.0).view(Hc, Wc)            # grossier (cellules MASK_CELL²), GPU : ~6 Ko

    # ---------------------------------------------------------------- boucle
    history = []
    from scipy.spatial import cKDTree
    tree = cKDTree(means.detach().cpu().numpy())
    nsc_np = map_nsc
    for w in windows:
        w_i = w["index"]
        tw = time.time()
        near = torch.from_numpy(w["near"]).to(dev)
        w_means = means[near].clone().requires_grad_(True)
        w_quats = quats[near].clone().requires_grad_(True)
        w_logs = log_scales[near].clone().requires_grad_(True)
        w_logit = logit_opac[near].clone().requires_grad_(True)
        w_sh0 = sh0[near].clone().requires_grad_(True)
        w_shN = shN[near].clone().requires_grad_(True)
        veg_w = veg_full[near] if veg_full is not None else None

        sweep_gpu = {}
        obs_lid = np.zeros(n_map, np.int32) if min_obs_lidar > 0 else None
        for st in w["train"] + w["holdout"]:
            sw = data.sweep(st)
            if sw["xyz"] is None and "nn_map" not in sw:            # ne devrait plus arriver ; on recharge
                data._sweeps.pop(st, None)
                sw = data.sweep(st)
            f = sw["first"]
            rng_f = sw["rng"][f]
            row_f, col_f = sw["row"][f], sw["col"][f]
            dyn_f = sw["dyn_pt"][f] if sw["dyn_pt"] is not None else None
            if "nn_map" not in sw:
                # voxel de la carte le plus proche de chaque point premier écho (indice dans la carte ENTIÈRE),
                # calculé une fois : `xyz` est libéré après chaque fenêtre et les fenêtres se recouvrent
                dn, j = tree.query(sw["xyz"][f], workers=-1)
                sw["nn_map"], sw["nn_dist"] = j.astype(np.int32), dn.astype(np.float32)
            j = sw["nn_map"]
            if dyn_f is not None:
                keep = (rng_f <= RANGE_MAX) & (~dyn_f)
            else:
                keep = (rng_f <= RANGE_MAX) & (nsc_np[j] >= 3)
            if obs_lid is not None and st in w["train"]:          # v3.3 : observations LiDAR
                obs_lid[np.unique(j[keep & (sw["nn_dist"] <= 0.10)])] += 1
            fr_, fc_ = sw["free_rows"].astype(np.int64), sw["free_cols"].astype(np.int64)
            ok_free = b2_ok_np[fr_, fc_]
            sweep_gpu[st] = dict(
                row=torch.from_numpy(row_f[keep].astype(np.int64)).to(dev),
                col=torch.from_numpy(col_f[keep].astype(np.int64)).to(dev),
                rng=torch.from_numpy(rng_f[keep].astype(np.float32)).to(dev),
                free_row=torch.from_numpy(fr_[ok_free]).to(dev),
                free_col=torch.from_numpy(fc_[ok_free]).to(dev),
                T0=torch.tensor(np.linalg.inv(sw["T0"]), dtype=torch.float32, device=dev),
                T1=torch.tensor(np.linalg.inv(sw["T1"]), dtype=torch.float32, device=dev),
                rows_f=torch.from_numpy(row_f.astype(np.int64)).to(dev),
                cols_f=torch.from_numpy(col_f.astype(np.int64)).to(dev),
                rng_f=torch.from_numpy(rng_f.astype(np.float32)).to(dev))

        def acts():
            return torch.exp(w_logs), torch.sigmoid(w_logit)

        def render_lidar(st):
            g = sweep_gpu[st]
            sc, op = acts()
            renders, alphas, _ = gsplat.rasterization(
                w_means, w_quats, sc, op, w_sh0, g["T0"][None],
                torch.eye(3, device=dev)[None], lidar_model.n_columns, lidar_model.n_rows,
                camera_model="lidar", lidar_coeffs=lidar_model, with_ut=True, with_eval3d=True,
                packed=False, render_mode="RGB-Ed", global_z_order=False, far_plane=300.0,
                sh_degree=0,
                rolling_shutter=gsplat.RollingShutterType.ROLLING_LEFT_TO_RIGHT,
                viewmats_rs=g["T1"][None])
            return renders[0, ..., 3], alphas[0, ..., 0]

        def render_rgb(fr, deg):
            K = cam_K[fr["cam"]]
            viewmat = torch.tensor(np.linalg.inv(fr["T_wc"]), dtype=torch.float32, device=dev)[None]
            sc, op = acts()
            colors = torch.cat([w_sh0, w_shN], 1) if deg > 0 else w_sh0
            out, alph, _ = gsplat.rasterization(
                w_means, w_quats, sc, op, colors, viewmat, K.K_tensor(dev), K.width, K.height,
                with_ut=True, with_eval3d=True, packed=False, render_mode="RGB", far_plane=300.0, sh_degree=deg,
                **K.gsplat_kwargs(dev))
            return out[0], alph[0, ..., 0]

        def eval_sweep(st):
            g = sweep_gpu[st]
            with torch.no_grad():
                d_r, a_r = render_lidar(st)
            dr = d_r.double().cpu().numpy()
            ar = a_r.double().cpu().numpy()
            rm = g["rng_f"].cpu().numpy()
            rows, cols = g["rows_f"].cpu().numpy(), g["cols_f"].cpu().numpy()
            has = ar[rows, cols] > 0.5
            drc = dr[rows, cols]
            ok = has & np.isfinite(drc) & (drc < 100.0) & np.isfinite(rm)
            err = drc[ok] - rm[ok]
            matched = np.abs(err) <= MATCH_CM / 100
            return {"matched_frac": round(float(matched.mean()), 4) if len(err) else None,
                    "median_abs_cm": round(float(np.median(np.abs(err[matched]))) * 100, 2) if matched.any() else None,
                    "divergent_median_cm": round(float(np.median(err[~matched])) * 100, 2) if (~matched).any() else None}

        def eval_rgb(st):
            """PSNR sur une image holdout, pixels valides ET dans la fenêtre ; brut (exposition neutre) et
            aligné (gain/biais par canal ajustés : l'exposition d'une image non vue est inconnue)."""
            frs = data.frames_for_sweep(st)
            if not frs:
                return None
            fr = frs[0]
            with torch.no_grad():
                rgb = render_rgb(fr, sh_degree)[0].clamp(0, 1)
                img_u8, valid_np, _ = load_img(fr)
                img = torch.from_numpy(img_u8).to(dev).float() / 255.0
                valid = (torch.from_numpy(valid_np).to(dev) & gvalid[fr["cam"]]
                         & expand_mask(window_pixel_mask(fr, w["center"], r_opt - 3.0), fr["cam"]))
                if valid.sum() < 1000:
                    return None
                r, gt = rgb[valid], img[valid]
                raw = float(-10 * torch.log10(((r - gt) ** 2).mean()))
                a = ((r - r.mean(0)) * (gt - gt.mean(0))).mean(0) / r.var(0, unbiased=False).clamp(min=1e-8)
                fit = (a * (r - r.mean(0)) + gt.mean(0)).clamp(0, 1)
                al = float(-10 * torch.log10(((fit - gt) ** 2).mean()))
                return {"psnr_brut": round(raw, 2), "psnr_aligne": round(al, 2), "frac_px": round(float(valid.float().mean()), 3)}

        frames_of = {st: data.frames_for_sweep(st) for st in w["train"]}
        keys = sorted({f"{fr['cam']}:{fr['stamp']}" for frs in frames_of.values() for fr in frs})
        kidx = {k: i for i, k in enumerate(keys)}
        w_expo = torch.tensor(np.stack([expo_store.get(k, np.zeros(6, np.float32)) for k in keys])
                              if keys else np.zeros((0, 6), np.float32), device=dev).requires_grad_(True)
        # masques « première surface dans la fenêtre » (une fois par fenêtre ; CPU, bool)
        r_in = r_opt - 3.0
        win_mask = {}
        obs = torch.zeros(n_map, dtype=torch.int32, device=dev)
        with torch.no_grad():
            for frs in frames_of.values():
                for fr in frs:
                    win_mask[f"{fr['cam']}:{fr['stamp']}"] = window_pixel_mask(fr, w["center"], r_in, obs=obs)
        well = obs[near] >= min_obs                                  # gaussiennes bien observées par la fenêtre
        if obs_lid is not None:                                       # v3.3 : ou touchées par le LiDAR
            well = well | torch.from_numpy(obs_lid[w["near"]] >= min_obs_lidar).to(dev)
        del obs
        anchor = means[near].detach().clone()
        groups = [{"params": [w_means], "lr": LR["means"]}, {"params": [w_quats], "lr": LR["quats"]},
                  {"params": [w_logs], "lr": LR["log_scales"]}, {"params": [w_logit], "lr": LR["logit_opac"]},
                  {"params": [w_sh0], "lr": LR["sh0"]}, {"params": [w_shN], "lr": LR["shN"]}]
        if exposure and len(keys):
            groups.append({"params": [w_expo], "lr": LR["expo"]})
        opt = torch.optim.Adam(groups, eps=1e-15)
        plist = [w_means, w_quats, w_logs, w_logit, w_sh0, w_shN]
        # facteur de pas par gaussienne : 0 = figée (mal observée), revisit_damp si déjà optimisée, 1 sinon
        fac = well.float()
        fac[well & torch.from_numpy(visits[w["near"]] > 0).to(dev)] *= revisit_damp
        old = fac < 1.0                                                # lignes dont la mise à jour est réduite
        n_old = int(old.sum())
        fac_old = fac[old]
        order = list(w["train"])
        rng.shuffle(order)
        ev = {}
        loss_log = []
        for it in range(1, steps_per_window + 1):
            opt.param_groups[0]["lr"] = LR["means"] * (0.1 ** ((it - 1) / steps_per_window))
            deg = 0 if it <= steps_per_window // 3 else sh_degree
            st = order[(it - 1) % len(order)]
            g = sweep_gpu[st]
            opt.zero_grad(set_to_none=True)
            d_r, a_r = render_lidar(st)
            loss = weights.get("b1", 0.0) * b1_lidar(d_r[g["row"], g["col"]], g["rng"], delta=0.02)
            if weights.get("b2", 0.0) and len(g["free_row"]):
                loss = loss + weights["b2"] * b2_free(a_r[g["free_row"], g["free_col"]])
            frs = frames_of[st]
            l_rgb = torch.zeros((), device=dev)
            if frs and weights.get("b0", 0.0):
                fr = frs[int(rng.integers(len(frs)))]
                key = f"{fr['cam']}:{fr['stamp']}"
                img_u8, valid_np, sky_np = load_img(fr)
                valid = (torch.from_numpy(valid_np).to(dev) & gvalid[fr["cam"]]
                         & expand_mask(win_mask[key], fr["cam"]))
                if valid.sum() > 1000:
                    img = torch.from_numpy(img_u8).to(dev).float() / 255.0
                    rgb, alpha_img = render_rgb(fr, deg)
                    if sky_w > 0:
                        skym = torch.from_numpy(sky_np).to(dev) & gvalid[fr["cam"]]
                        if skym.any():
                            loss = loss + sky_w * alpha_img[skym].mean()
                    if exposure:
                        e = w_expo[kidx[key]]
                        rgb = rgb * torch.exp(e[:3]) + e[3:]
                        loss = loss + EXPO_REG * (e ** 2).sum()
                    l1 = (rgb - img).abs()[valid].mean()
                    if ssim_w > 0:
                        s = _ssim_map(rgb.permute(2, 0, 1), img.permute(2, 0, 1))[valid].mean()
                        l_rgb = (1 - ssim_w) * l1 + ssim_w * (1 - s)
                    else:
                        l_rgb = l1
                    loss = loss + weights["b0"] * l_rgb
            if weights.get("b3", 0.0):
                loss = loss + weights["b3"] * b3_flatten(torch.exp(w_logs), veg_w)
            if weights.get("b4", 0.0):
                loss = loss + weights["b4"] * torch.nn.functional.mse_loss(w_means, anchor)
            loss.backward()
            if n_old:
                prev = [p.detach()[old].clone() for p in plist]
            opt.step()
            with torch.no_grad():
                if n_old:
                    for p_, pv in zip(plist, prev):
                        f_ = fac_old.view(-1, *([1] * (p_.dim() - 1)))
                        p_[old] = pv + f_ * (p_[old] - pv)
                    del prev
                w_logs.clamp_(max=float(np.log(scale_max)))
            if it % 100 == 0:
                loss_log.append(round(float(l_rgb), 4))
            if eval_every and it % eval_every == 0 and w["holdout"]:
                ev[f"it{it}"] = eval_sweep(w["holdout"][it // eval_every % len(w["holdout"])])
        for st in w["train"] + w["holdout"]:
            sw_c = data._sweeps.get(st)        # seulement ceux encore en cache (data.sweep rechargerait)
            if sw_c is not None and "nn_map" in sw_c:
                sw_c["xyz"] = None             # libère la RAM (xyz inutile après masques)
        with torch.no_grad():
            means[near] = w_means.detach()
            quats[near] = w_quats.detach()
            log_scales[near] = w_logs.detach()
            logit_opac[near] = w_logit.detach()
            sh0[near] = w_sh0.detach()
            shN[near] = w_shN.detach()
            wn = w["near"][well.cpu().numpy()]                              # « optimisée » = bien observée ici
            visits[wn] = np.minimum(visits[wn].astype(np.int32) + 1, 255).astype(np.uint8)
            ew = w_expo.detach().cpu().numpy()
            for k, i in kidx.items():
                expo_store[k] = ew[i]
        ev_train = eval_sweep(w["train"][len(w["train"]) // 2]) if w["train"] else None
        ev_hold = {s: eval_sweep(s) for s in w["holdout"][:2]}
        ev_rgb = {s: eval_rgb(s) for s in w["holdout"][:3]}
        with torch.no_grad():
            op_w = torch.sigmoid(w_logit)
            stats = {"opac_lt_0.05": round(float((op_w < 0.05).float().mean()), 4),
                     "opac_median": round(float(op_w.median()), 3),
                     "scale_t_p99_cm": round(float(torch.quantile(torch.exp(w_logs[:200000, 0]), 0.99)) * 100, 2),
                     "scale_max_cm": round(float(torch.exp(w_logs.max())) * 100, 1),
                     "frac_bien_observees": round(float(well.float().mean()), 3),
                     "frac_pas_reduit_ou_fige": round(n_old / max(len(near), 1), 3)}
        import resource
        stats["ram_max_go"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
        entry = dict(window=w_i, center=[round(float(x), 2) for x in w["center"]], n_gaussians=int(len(near)), n_sweeps=len(w["train"]), n_images=len(keys),
                     wall_s=round(time.time() - tw, 1), eval_train=ev_train, eval_holdout=ev_hold,
                     eval_rgb=ev_rgb, eval_curve=ev, loss_rgb_curve=loss_log, param_stats=stats)
        history.append(entry)
        print(f"fenêtre {w_i:2d}/{len(centers)-1} : train {json.dumps(ev_train)} | "
              f"holdout {json.dumps(list(ev_hold.values())[0]) if ev_hold else None} | "
              f"rgb {json.dumps(ev_rgb)} | {json.dumps(stats)} | {entry['wall_s']:.0f}s", flush=True)
        del w_means, w_quats, w_logs, w_logit, w_sh0, w_shN, w_expo, opt, win_mask, plist, old, fac, fac_old, well
        torch.cuda.empty_cache()
        if len(history) % 5 == 0 or w is windows[-1]:
            _save_scene(out_dir, means, quats, log_scales, logit_opac, sh0, shN, expo_store)
            with open(f"{out_dir}/history.json", "w") as f:
                json.dump(history, f, indent=1, ensure_ascii=False)

    res = {"n_windows": len(windows), "windows": [w["index"] for w in windows], "weights": weights,
           "stride_m": stride_m, "steps_per_window": steps_per_window, "sh_degree": sh_degree,
           "opacity_init": opacity_init, "ssim_w": ssim_w, "exposure": exposure, "b2": b2_info,
           "color_init": color_src, "lr": LR, "sky_w": sky_w, "scale_max": scale_max, "revisit_damp": revisit_damp,
           "min_obs": min_obs, "min_obs_lidar": min_obs_lidar, "obs_max_dist": obs_max_dist,
           "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
           "wall_s": round(time.time() - t0, 1), "history": history}
    with open(f"{out_dir}/result.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    return res


def _save_scene(out_dir, means, quats, log_scales, logit_opac, sh0, shN, expo_store):
    import torch

    with torch.no_grad():
        q = quats / quats.norm(dim=1, keepdim=True).clamp(min=1e-12)
        np.savez_compressed(f"{out_dir}/scene_full.npz",
                            means=means.cpu().numpy().astype(np.float32),
                            quats=q.cpu().numpy().astype(np.float16),
                            scales=torch.exp(log_scales).cpu().numpy().astype(np.float16),
                            opac=torch.sigmoid(logit_opac).cpu().numpy().astype(np.float16),
                            colors=torch.cat([sh0, shN], 1).cpu().numpy().astype(np.float16),
                            param_version=np.array("v3_activated"))
    keys = sorted(expo_store)
    np.savez_compressed(f"{out_dir}/exposure.npz", keys=np.array(keys),
                        log_gain_bias=np.stack([expo_store[k] for k in keys]) if keys else np.zeros((0, 6)))


if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--root", required=True)
    pa.add_argument("--stride-m", type=float, default=20.0)
    pa.add_argument("--steps-per-window", type=int, default=3000)
    pa.add_argument("--r-opt", type=float, default=35.0)
    pa.add_argument("--core-m", type=float, default=20.0)
    pa.add_argument("--weights", default="b0=1,b1=1,b2=0.05,b3=0.05,b4=1.0")
    pa.add_argument("--eval-every", type=int, default=300)
    pa.add_argument("--out-dir", default="out/train_full_v3")
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--sh-degree", type=int, default=1)
    pa.add_argument("--focus", default="", help="cam:stamp — n'entraîner que les fenêtres proches de cette pose")
    pa.add_argument("--focus-r", type=float, default=30.0)
    pa.add_argument("--windows", default="", help="indices de fenêtres, séparés par des virgules")
    pa.add_argument("--opacity-init", type=float, default=0.95)
    pa.add_argument("--ssim-w", type=float, default=0.2)
    pa.add_argument("--no-exposure", action="store_true")
    pa.add_argument("--b2-min-el", type=float, default=5.0)
    pa.add_argument("--colors-npz", default="")
    pa.add_argument("--sky-w", type=float, default=SKY_W, help="poids opacité → 0 sur les pixels de ciel (0 = v3)")
    pa.add_argument("--scale-max", type=float, default=SCALE_MAX, help="taille max d'une gaussienne (m)")
    pa.add_argument("--revisit-damp", type=float, default=REVISIT_DAMP,
                    help="facteur de pas pour les gaussiennes déjà optimisées (1 = désactivé)")
    pa.add_argument("--min-obs", type=int, default=MIN_OBS,
                    help="nb min d'images de la fenêtre où une gaussienne est visible pour être optimisée (0 = tout)")
    pa.add_argument("--obs-max-dist", type=float, default=OBS_MAX_DIST, help="distance max d'une observation (m)")
    pa.add_argument("--min-obs-lidar", type=int, default=0,
                    help="v3.3 : une gaussienne touchée (≤ 10 cm) par ≥ N balayages d'entraînement de la fenêtre est "
                         "aussi optimisée (0 = v3.2 : images seules ; indispensable avec une seule caméra frontale)")
    a = pa.parse_args()
    w = {}
    for part in a.weights.split(","):
        if part.strip():
            n, _, v = part.partition("=")
            w[n.strip()] = float(v)
    res = run_full(a.root, stride_m=a.stride_m, steps_per_window=a.steps_per_window,
                   r_opt=a.r_opt, core_m=a.core_m, weights=w, out_dir=a.out_dir,
                   eval_every=a.eval_every, seed=a.seed, sh_degree=a.sh_degree, focus=a.focus,
                   focus_r=a.focus_r, windows_sel=a.windows, opacity_init=a.opacity_init,
                   ssim_w=a.ssim_w, exposure=not a.no_exposure, b2_min_el=a.b2_min_el,
                   colors_npz=a.colors_npz, sky_w=a.sky_w, scale_max=a.scale_max,
                   revisit_damp=a.revisit_damp, min_obs=a.min_obs, obs_max_dist=a.obs_max_dist,
                   min_obs_lidar=a.min_obs_lidar)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1, ensure_ascii=False))
