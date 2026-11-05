"""Étape 0.5 — Test T0.4 : superposition LiDAR → images (à l'arrêt et en mouvement) et lacet caméra↔LiDAR.

Principe
--------
Pour chaque triplet d'images choisi (à l'arrêt, en marche, en rotation rapide) :
1. on prend les tours LiDAR qui encadrent l'instant image et, pour chaque azimut, le point le plus proche
   dans le temps (|t_point − t_image| ≤ 50 ms) ;
2. on extrait les **discontinuités de profondeur** dans la grille (ring, colonne alignée en azimut) des tours
   (`lidar/hesai/*.npz`, écho le plus proche) : saut de distance entre deux cellules voisines nettement
   plus grand que les pas voisins (rejette le sol rasant). Le point retenu est le point **avant-plan**,
   placé à mi-chemin angulaire du voisin (position non biaisée du bord), ou au centre de sa cellule si
   celle-ci a un second écho lointain (le faisceau chevauche le bord). Catégories : `jump_col` (voisins
   en azimut → bords plutôt verticaux), `jump_row` (voisins entre rings), `void_col`/`void_row` (voisin
   sans retour au-dessus de l'horizon : ciel) ;
3. chaque point est désanamorphosé à l'instant image avec la GT :
   `p_base(t_img) = inv(T_world_base(t_img)) · T_world_base(t_point) · T_base_lidar · p_lidar` ;
4. projection fisheye (`splatpipe.common.camera_model`), filtrage d'occultation (tampon de profondeur
   grossier de tous les points), puis distance au contour Canny le plus proche (transformée de distance,
   interpolation bilinéaire).

Configurations évaluées : calibration fournie, corrections de rotation `T_base_cam' = C · T_base_cam`
(C = exp(rotvec) dans base, pivot à l'origine de base ou au centre de chaque caméra, commune ou par caméra),
décalage temporel caméra (`t_ref = t_image + dt`), sans compensation du mouvement, calibrations d'autres dates.

Lancement (depuis la racine du projet), dans cet ordre :
  .venv/bin/modal run splatpipe/stage0/overlay_check.py --stage variants   # optionnel : choix des paramètres
  .venv/bin/modal run splatpipe/stage0/overlay_check.py --stage sweeps     # balayages 1D (lacet, tangage, roulis, dt)
  .venv/bin/modal run splatpipe/stage0/overlay_check.py --stage grid       # grille 3D de rotations
  .venv/bin/modal run splatpipe/stage0/overlay_check.py --stage finalize [--apply]

`finalize` évalue les candidats, calcule T0.4, applique la règle de décision `DECISION`, dessine les
superpositions et écrit `<scène>/reports/stage0_overlay.json` (copie : `logs/stage0/overlay.json`, images :
`logs/stage0/overlay/`). `sensors.yaml` n'est modifié que si `--apply` est passé **et** que la règle l'autorise
(`sensors_provided.yaml` = copie exacte de l'original). Scores bruts : `<scène>/reports/stage0_overlay/*_scores.npz`.
Ne touche ni à `/data/raw`, ni à `lidar*/`. CPU uniquement (~1 min par stage avec le fan-out).
"""

from __future__ import annotations

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume

app = modal.App("splatsim-stage0-overlay")
vol = data_volume()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy", "scipy", "pyyaml", "opencv-python-headless")
    .add_local_python_source("splatpipe")
)

DATASET = "oxford_spires"
SEQ = "2024-03-13-observatory-quarter-01"
CAMS = ("cam0", "cam1", "cam2")
LIDAR_ID = "hesai_qt64"
LIDAR_SUBDIR = "lidar/hesai"
LIDAR_MODEL = "lidar_model/hesai_qt64.json"
OUT_SUBDIR = "reports/stage0_overlay"
CALIB_ROOT = "raw/oxford_spires/calibration"  # relatif à DATA_MOUNT
OTHER_CALIBS = ("calibration-sequences-2024-06-29", "calibration-sequences-2024-10-29")

CATS = ("jump_col", "jump_row", "void_col", "void_row")
DEPTH_EDGES = (1.5, 4.0, 8.0, 16.0, 1e9)  # distance caméra (m) → 4 classes
N_GROUPS = len(CATS) * (len(DEPTH_EDGES) - 1)

PRM = {
    # contours image
    "gauss_ksize": 5,
    "gauss_sigma": 1.2,
    "canny_low": 40,
    "canny_high": 110,
    # discontinuités LiDAR
    "time_half_window_s": 0.05,
    "jump_abs_m": 0.30,
    "jump_rel": 0.08,
    "jump_context_ratio": 3.0,
    "min_range_m": 1.5,
    "max_range_m": 60.0,
    "void_max_range_m": 40.0,
    "void_min_world_elev_deg": 2.0,
    # projection / score
    "theta_limit_deg": 85.0,
    "select_margin_px": 20,
    "eval_margin_px": 2,
    "occ_cell_px": 4,
    "occ_rel": 0.03,
    "occ_abs_m": 0.15,
    "trunc_px": 10.0,
    "hist_bin_px": 0.1,
    "hist_n_bins": 200,  # + 1 classe de débordement (≥ 20 px)
    "gt_max_gap_s": 0.2,
}

# Sélection des images
STATIC_MAX_SPEED = 0.10  # m/s
STATIC_MAX_RATE_DEG = 10.0
WALK_MIN_SPEED = 0.5
WALK_MAX_RATE_DEG = 20.0
FAST_MIN_RATE_DEG = 35.0
N_WALK = 80
N_FAST = 40
FAST_MIN_SEP_S = 1.0

# Variantes de paramètres comparées (stage `variants`) : netteté du minimum et stabilité de l'estimation.
VARIANTS = {
    "A_s1.2_40-110_mid": {"mixed_refine": False},
    "B_s1.2_40-110_mix": {},
    "C_s2.0_40-110_mix": {"gauss_sigma": 2.0, "gauss_ksize": 9},
    "D_s1.2_80-200_mix": {"canny_low": 80, "canny_high": 200},
    "E_s2.0_60-160_mix": {"gauss_sigma": 2.0, "gauss_ksize": 9, "canny_low": 60, "canny_high": 160},
}


# =============================================================================
# Utilitaires
# =============================================================================


def _py(x):
    """Convertit récursivement numpy → types Python sérialisables."""
    import numpy as np

    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return _py(x.tolist())
    if isinstance(x, np.floating):
        v = float(x)
        return None if v != v else v
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float) and x != x:
        return None
    return x


def scene(seq: str):
    from splatpipe.common.scene import Scene

    return Scene(DATASET, seq, data_root=DATA_MOUNT)


def load_provided_sensors(sc) -> dict:
    """Calibration **fournie** : `sensors_provided.yaml` s'il existe (copie de l'original), sinon `sensors.yaml`."""
    import os

    import yaml

    p = sc.path("sensors_provided.yaml")
    src = p if os.path.exists(p) else sc.sensors_path
    with open(src) as f:
        s = yaml.safe_load(f)
    if "calibration_corrections" in s:
        raise RuntimeError(f"{src} contient déjà des corrections : ce n'est pas la calibration fournie")
    return s


def rotvec_deg_to_R(rv_deg):
    import numpy as np

    from splatpipe.common import geometry as g

    rv = np.radians(np.asarray(rv_deg, dtype=np.float64))
    a = np.linalg.norm(rv)
    if a < 1e-15:
        return np.eye(3)
    ax = rv / a
    return g.quat_to_rot(np.r_[ax * np.sin(a / 2), np.cos(a / 2)])


def T_base_cam_from_camlidar_yaml(path: str, T_base_lidar) -> dict:
    """`cam-lidar-imu.yaml` (DiffCal) → {cam: T_base_cam} avec la même formule que l'ingestion."""
    import numpy as np
    import yaml

    from splatpipe.common import geometry as g

    with open(path) as f:
        ext = yaml.safe_load(f)
    out = {}
    for cam in CAMS:
        M = np.array(ext[cam]["T_cam_lidar"], dtype=np.float64)
        U, _, Vt = np.linalg.svd(M[:3, :3])
        out[cam] = np.asarray(T_base_lidar) @ g.inv(g.se3_from_rt(U @ Vt, M[:3, 3]))
    return out


def apply_config_extrinsics(T_base_cam, cfg) -> dict:
    """Extrinsèques d'une configuration : override éventuel puis `C · T_base_cam` (pivot base ou caméra)."""
    import numpy as np

    rv_all = cfg.get("rotvec_deg", (0.0, 0.0, 0.0))
    rv_cam = cfg.get("rotvec_deg_by_cam") or {}
    pivot = cfg.get("pivot", "base")
    out = {}
    for cam in CAMS:
        C = rotvec_deg_to_R(rv_cam.get(cam, rv_all))
        T = np.array(cfg["T_base_cam"][cam] if cfg.get("T_base_cam") else T_base_cam[cam], dtype=np.float64)
        T2 = T.copy()
        T2[:3, :3] = C @ T[:3, :3]
        if pivot == "base":
            T2[:3, 3] = C @ T[:3, 3]
        elif pivot != "cam":
            raise ValueError(pivot)
        out[cam] = T2
    return out


# =============================================================================
# Discontinuités LiDAR
# =============================================================================


def scan_discontinuities(scan: dict, model, prm: dict = PRM) -> dict:
    """Discontinuités de profondeur d'un tour (grille ring × colonne, écho le plus proche).

    Retourne des tableaux alignés : `dir` (N,3) direction unitaire du bord (repère lidar), `range` (distance
    avant-plan), `t` (temps du point avant-plan), `cat` (indice dans CATS), `void_ray` (N,3, rayon de la cellule
    vide ou NaN), plus `all_xyz`, `all_t` (tous les points valides, pour l'occultation).
    """
    import numpy as np

    xyz = scan["pt_xyz"].astype(np.float64)
    ring = scan["pt_ring"].astype(np.int64)
    col = scan["pt_col"].astype(np.int64)
    ret = scan["pt_ret"].astype(np.int64)
    t = scan["pt_t"].astype(np.float64)
    H, W = model.n_rows, model.n_cols
    rng = np.linalg.norm(xyz, axis=1)
    row = model.ring_to_row[ring]

    # Grille alignée en azimut : le QT64 a 4 groupes de lasers décalés de ±5,6° en azimut, donc la même
    # colonne de deux rows voisines ne vise pas la même direction. Colonne alignée k = col − round(offset_row / pas),
    # pas = pas angulaire signé des colonnes (sens de rotation).
    step = float(np.median(np.diff(model.column_azimuths_rad)))
    shift = np.rint(model.row_azimuth_offsets_rad / -step).astype(np.int64)  # az(r, c) ≈ az0 + step·(c − shift_r)
    shift -= shift.min()
    Wa = W + int(shift.max())
    kcol = col + (int(shift.max()) - shift[row])  # ≥ 0 ; même k ⇒ même azimuth à ±½ pas près

    ok_all = (ret >= 0) & (rng > 0.3) & np.isfinite(rng)
    sel = np.nonzero((ret == 0) & (rng > 0) & np.isfinite(rng))[0]
    idx = np.full((H, Wa), -1, dtype=np.int64)
    idx[row[sel], kcol[sel]] = sel
    sel1 = np.nonzero((ret == 1) & (rng > 0) & np.isfinite(rng))[0]
    r1 = np.full((H, Wa), np.nan)
    r1[row[sel1], kcol[sel1]] = rng[sel1]  # écho distinct le plus lointain (cellules à double écho)
    # Cellules réellement tirées (pour distinguer « pas de retour » de « hors grille » après décalage).
    fired = np.zeros((H, Wa), dtype=bool)
    rr = np.arange(H)[:, None]
    fired[rr, np.arange(W)[None, :] + (int(shift.max()) - shift)[:, None]] = True
    r = np.where(idx >= 0, rng[np.maximum(idx, 0)], np.nan)
    unit = xyz / np.maximum(rng, 1e-9)[:, None]

    out = {k: [] for k in ("dir", "range", "t", "cat", "void_ray")}
    abs_thr, rel_thr, ratio = prm["jump_abs_m"], prm["jump_rel"], prm["jump_context_ratio"]

    for axis, (cat_jump, cat_void) in ((1, (0, 2)), (0, (1, 3))):
        # On se ramène à des lignes 1D le long de `axis` : tableau (L, K) où K = longueur le long de l'axe.
        R = r if axis == 1 else r.T
        R1 = r1 if axis == 1 else r1.T
        I = idx if axis == 1 else idx.T
        Fz = fired if axis == 1 else fired.T
        L, K = R.shape
        rb, rc = R[:, 0:K - 1], R[:, 1:K]  # paire (k, k+1)
        ib, ic = I[:, 0:K - 1], I[:, 1:K]
        fb, fc = Fz[:, 0:K - 1], Fz[:, 1:K]
        # pas de contexte : (k−1, k) côté b et (k+1, k+2) côté c
        pad = np.full((L, 1), np.nan)
        step_b = np.abs(np.concatenate([pad, R[:, 1:K - 1] - R[:, 0:K - 2]], axis=1))  # |r_k − r_{k−1}|
        step_c = np.abs(np.concatenate([R[:, 2:K] - R[:, 1:K - 1], pad], axis=1))  # |r_{k+2} − r_{k+1}|
        with np.errstate(invalid="ignore"):
            fg = np.fmin(rb, rc)
            thr = np.maximum(abs_thr, rel_thr * fg)
            jump = np.abs(rc - rb)
            both = np.isfinite(rb) & np.isfinite(rc)
            # Un pas de contexte qui est lui-même un saut (objet fin d'une cellule) n'est pas pris en compte.
            sb = np.where(step_b < thr, step_b, np.nan)
            sc_ = np.where(step_c < thr, step_c, np.nan)
            ctx = np.fmax(sb, sc_)
            is_jump = both & (jump > thr) & np.isfinite(ctx) & (jump > ratio * ctx)
            is_jump &= (fg >= prm["min_range_m"]) & (fg <= prm["max_range_m"])
        li, ki = np.nonzero(is_jump)
        b_is_fg = rb[li, ki] < rc[li, ki]
        f_idx = np.where(b_is_fg, ib[li, ki], ic[li, ki])
        g_idx = np.where(b_is_fg, ic[li, ki], ib[li, ki])
        d = unit[f_idx] + unit[g_idx]
        if prm.get("mixed_refine", True):
            # Cellule avant-plan à double écho (écho lointain au-delà du seuil) : le faisceau chevauche le bord,
            # qui est donc dans l'empreinte de cette cellule → on garde sa direction au lieu du mi-chemin.
            r1_f = np.where(b_is_fg, R1[li, ki], R1[li, np.minimum(ki + 1, K - 1)])
            with np.errstate(invalid="ignore"):
                mixed = np.isfinite(r1_f) & (r1_f - rng[f_idx] > thr[li, ki])
            d = np.where(mixed[:, None], 2.0 * unit[f_idx], d)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        out["dir"].append(d)
        out["range"].append(rng[f_idx])
        out["t"].append(t[f_idx])
        out["cat"].append(np.full(len(f_idx), cat_jump))
        out["void_ray"].append(np.full((len(f_idx), 3), np.nan))

        # Bords « vides » : cellule valide (contexte lisse de son côté) et voisine tirée sans aucun retour.
        with np.errstate(invalid="ignore"):
            thr_b = np.maximum(abs_thr, rel_thr * rb)
            thr_c = np.maximum(abs_thr, rel_thr * rc)
            v1 = (np.isfinite(rb) & ~np.isfinite(rc) & fc & (step_b < thr_b) & (rb >= prm["min_range_m"])
                  & (rb <= prm["void_max_range_m"]))
            v2 = (np.isfinite(rc) & ~np.isfinite(rb) & fb & (step_c < thr_c) & (rc >= prm["min_range_m"])
                  & (rc <= prm["void_max_range_m"]))
        for vm, f_is_b in ((v1, True), (v2, False)):
            li, ki = np.nonzero(vm)
            if len(li) == 0:
                continue
            f_idx = (ib if f_is_b else ic)[li, ki]
            e_k = ki + 1 if f_is_b else ki  # indice le long de l'axe de la cellule vide
            e_row, e_kc = (li, e_k) if axis == 1 else (e_k, li)
            e_col = e_kc - (int(shift.max()) - shift[e_row])
            ray = model.element_rays(e_row, e_col)
            d = unit[f_idx] + ray
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            out["dir"].append(d)
            out["range"].append(rng[f_idx])
            out["t"].append(t[f_idx])
            out["cat"].append(np.full(len(f_idx), cat_void))
            out["void_ray"].append(ray)

    res = {k: (np.concatenate(v) if v else None) for k, v in out.items()}
    if res["dir"] is None:
        res = {"dir": np.zeros((0, 3)), "range": np.zeros(0), "t": np.zeros(0), "cat": np.zeros(0, int),
               "void_ray": np.zeros((0, 3))}
    res["all_xyz"] = xyz[ok_all]
    res["all_t"] = t[ok_all]
    return res


# =============================================================================
# Image
# =============================================================================


def image_edges(path: str, prm: dict = PRM):
    """JPEG → (gris uint8, contours Canny uint8, transformée de distance float32 en px)."""
    import cv2
    import numpy as np

    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    k = prm["gauss_ksize"]
    blur = cv2.GaussianBlur(gray, (k, k), prm["gauss_sigma"])
    edges = cv2.Canny(blur, prm["canny_low"], prm["canny_high"], L2gradient=True)
    dt = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return gray, edges, dt


def sample_bilinear(img, u, v):
    import numpy as np

    h, w = img.shape
    u = np.clip(u, 0.0, w - 1.000001)
    v = np.clip(v, 0.0, h - 1.000001)
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    ax = u - x0
    ay = v - y0
    a = img[y0, x0]
    b = img[y0, x0 + 1]
    c = img[y0 + 1, x0]
    d = img[y0 + 1, x0 + 1]
    return (a * (1 - ax) + b * ax) * (1 - ay) + (c * (1 - ax) + d * ax) * ay


# =============================================================================
# Préparation d'un triplet
# =============================================================================


class FrameData:
    """Données d'un triplet : contours image, discontinuités LiDAR en world, points pour l'occultation."""

    def __init__(self, sc, frame: dict, sensors: dict, model, traj, prm: dict = PRM, keep_images: bool = False):
        import numpy as np

        from splatpipe.common import geometry as g
        from splatpipe.common.camera_model import FisheyeCamera

        self.prm = prm
        self.frame = frame
        self.t_img = float(frame["t"])
        self.traj = traj
        self.T_bl = np.array(sensors["lidars"][LIDAR_ID]["T_base_lidar"], dtype=np.float64)
        self.cams = {c: FisheyeCamera.from_dict(sensors["cameras"][c]) for c in CAMS}
        self.T_base_cam = {c: np.array(sensors["cameras"][c]["T_base_cam"], dtype=np.float64) for c in CAMS}

        # ---- contours
        self.dt, self.gray, self.edges, self.edge_density = {}, {}, {}, {}
        for c in CAMS:
            gray, edges, dt = image_edges(sc.abs_path(frame[f"{c}_rel_path"]), prm)
            self.dt[c] = dt
            self.edge_density[c] = float((edges > 0).mean())
            if keep_images:
                self.gray[c], self.edges[c] = gray, edges

        # ---- LiDAR : discontinuités et tous les points, fenêtre |t − t_img| ≤ h
        h = prm["time_half_window_s"]
        dirs, rngs, ts, cats, vrays, axyz, at = [], [], [], [], [], [], []
        self.scans_used = []
        for name in frame["scans"]:
            with np.load(f"{sc.path(LIDAR_SUBDIR)}/{name}") as z:
                scan = {k: z[k] for k in ("pt_xyz", "pt_ring", "pt_col", "pt_ret", "pt_t")}
            dsc = scan_discontinuities(scan, model, prm)
            m = np.abs(dsc["t"] - self.t_img) <= h
            ma = np.abs(dsc["all_t"] - self.t_img) <= h
            self.scans_used.append({"file": name, "n_edges": int(m.sum()), "n_points": int(ma.sum())})
            dirs.append(dsc["dir"][m]); rngs.append(dsc["range"][m]); ts.append(dsc["t"][m])
            cats.append(dsc["cat"][m]); vrays.append(dsc["void_ray"][m])
            axyz.append(dsc["all_xyz"][ma]); at.append(dsc["all_t"][ma])
        e_dir = np.concatenate(dirs); e_rng = np.concatenate(rngs); e_t = np.concatenate(ts)
        e_cat = np.concatenate(cats); e_vray = np.concatenate(vrays)
        a_xyz = np.concatenate(axyz); a_t = np.concatenate(at)

        tg, pg, qg = traj
        e_l = e_dir * e_rng[:, None]
        self.e_base_raw = g.transform_points(self.T_bl, e_l)  # sans compensation du mouvement
        pw, qw, vw = g.interpolate_poses(tg, pg, qg, e_t, max_gap_s=prm["gt_max_gap_s"])
        Rw = g.quat_to_rot(qw)
        self.e_world = np.einsum("nij,nj->ni", Rw, self.e_base_raw) + pw
        # Filtre « ciel » des bords vides : rayon de la cellule vide au-dessus de l'horizon world.
        void = e_cat >= 2
        keep = vw.copy()
        if void.any():
            ray_w = np.einsum("nij,nj->ni", Rw[void], (self.T_bl[:3, :3] @ e_vray[void].T).T)
            keep[np.nonzero(void)[0]] &= ray_w[:, 2] > np.sin(np.radians(prm["void_min_world_elev_deg"]))
        self.e_world, self.e_base_raw, self.e_cat, self.e_t = self.e_world[keep], self.e_base_raw[keep], e_cat[keep], e_t[keep]
        pa, qa, va = g.interpolate_poses(tg, pg, qg, a_t, max_gap_s=prm["gt_max_gap_s"])
        a_base = g.transform_points(self.T_bl, a_xyz)
        self.a_world = (np.einsum("nij,nj->ni", g.quat_to_rot(qa), a_base) + pa)[va]
        self.n_edges_raw = int(len(e_t))
        self.n_edges_gt_ok = int(vw.sum())

        # ---- sélection par caméra à la configuration de référence (fournie, dt = 0, compensée)
        self.sel = {}
        T0 = self.pose(0.0)
        eb = self.world_to_base(self.e_world, T0)
        ab = self.world_to_base(self.a_world, T0)
        tl = np.radians(prm["theta_limit_deg"])
        cell = prm["occ_cell_px"]
        for c in CAMS:
            cam = self.cams[c]
            Tbc = self.T_base_cam[c]
            pc = (eb - Tbc[:3, 3]) @ Tbc[:3, :3]
            uv, ok, _ = cam.project(pc, margin=prm["select_margin_px"], theta_limit=tl)
            dist = np.linalg.norm(pc, axis=1)
            ok &= dist >= DEPTH_EDGES[0]
            # tampon de profondeur grossier
            pa_c = (ab - Tbc[:3, 3]) @ Tbc[:3, :3]
            uva, oka, _ = cam.project(pa_c, margin=0, theta_limit=tl)
            da = np.linalg.norm(pa_c, axis=1)
            Hc, Wc = cam.height // cell + 1, cam.width // cell + 1
            zbuf = np.full((Hc, Wc), np.inf)
            ia = (np.clip(uva[oka, 1], 0, cam.height - 1) // cell).astype(np.int64)
            ja = (np.clip(uva[oka, 0], 0, cam.width - 1) // cell).astype(np.int64)
            np.minimum.at(zbuf, (ia, ja), da[oka])
            zpad = np.pad(zbuf, 1, constant_values=np.inf)
            zmin = zbuf.copy()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    zmin = np.minimum(zmin, zpad[1 + di:1 + di + Hc, 1 + dj:1 + dj + Wc])
            ie = (np.clip(uv[:, 1], 0, cam.height - 1) // cell).astype(np.int64)
            je = (np.clip(uv[:, 0], 0, cam.width - 1) // cell).astype(np.int64)
            occluded = dist > zmin[ie, je] * (1 + prm["occ_rel"]) + prm["occ_abs_m"]
            n_occ = int((ok & occluded).sum())
            ok &= ~occluded
            ii = np.nonzero(ok)[0]
            dbin = np.clip(np.searchsorted(DEPTH_EDGES, dist[ii], side="right") - 1, 0, len(DEPTH_EDGES) - 2)
            self.sel[c] = {"idx": ii, "group": self.e_cat[ii] * (len(DEPTH_EDGES) - 1) + dbin, "n_occluded": n_occ,
                           "dist": dist[ii]}

    # ------------------------------------------------------------------ poses
    def pose(self, dt: float):
        import numpy as np

        from splatpipe.common import geometry as g

        tg, pg, qg = self.traj
        p, q, _ = g.interpolate_poses(tg, pg, qg, np.array([self.t_img + dt]))
        return g.quat_to_rot(q[0]), p[0]

    @staticmethod
    def world_to_base(pw, T):
        R, t = T
        return (pw - t) @ R

    # ------------------------------------------------------------------ évaluation
    def evaluate(self, configs: list, hist_ids=()):
        """→ (S [n_cfg, n_cam, N_GROUPS, 4] = (n, Σ min(d, trunc), n(d < 2 px), Σ min(d, 5 px)),
              H [n_hist, n_cam, N_GROUPS, n_bins + 1] histogrammes des distances)."""
        import numpy as np

        prm = self.prm
        trunc = prm["trunc_px"]
        nb = prm["hist_n_bins"]
        tl = np.radians(prm["theta_limit_deg"])
        hist_ids = list(hist_ids)
        S = np.zeros((len(configs), len(CAMS), N_GROUPS, 4), dtype=np.float64)
        Hh = np.zeros((len(hist_ids), len(CAMS), N_GROUPS, nb + 1), dtype=np.int32)
        base_cache = {}
        for k, cfg in enumerate(configs):
            mc = cfg.get("motion_comp", True)
            dt = float(cfg.get("dt_s", 0.0))
            key = (mc, dt)
            if key not in base_cache:
                base_cache[key] = self.world_to_base(self.e_world, self.pose(dt)) if mc else self.e_base_raw
            eb = base_cache[key]
            ext = apply_config_extrinsics(self.T_base_cam, cfg)
            for ci, c in enumerate(CAMS):
                s = self.sel[c]
                if len(s["idx"]) == 0:
                    continue
                T = ext[c]
                pc = (eb[s["idx"]] - T[:3, 3]) @ T[:3, :3]
                uv, ok, _ = self.cams[c].project(pc, margin=prm["eval_margin_px"], theta_limit=tl)
                d = np.full(len(pc), np.inf)
                if ok.any():
                    d[ok] = sample_bilinear(self.dt[c], uv[ok, 0], uv[ok, 1])
                grp = s["group"]
                S[k, ci, :, 0] = np.bincount(grp, minlength=N_GROUPS)
                S[k, ci, :, 1] = np.bincount(grp, weights=np.minimum(d, trunc), minlength=N_GROUPS)
                S[k, ci, :, 2] = np.bincount(grp, weights=(d < 2.0).astype(np.float64), minlength=N_GROUPS)
                S[k, ci, :, 3] = np.bincount(grp, weights=np.minimum(d, 5.0), minlength=N_GROUPS)
                if k in hist_ids:
                    b = np.minimum(np.floor(np.minimum(d, 1e6) / prm["hist_bin_px"]), nb).astype(np.int64)
                    Hh[hist_ids.index(k), ci] = np.bincount(grp * (nb + 1) + b, minlength=N_GROUPS * (nb + 1)).reshape(
                        N_GROUPS, nb + 1)
        return S, Hh

    def stats(self) -> dict:
        import numpy as np

        return {
            "frame_id": int(self.frame["frame_id"]),
            "scans": self.scans_used,
            "n_edges_in_window": self.n_edges_raw,
            "n_edges_gt_ok": self.n_edges_gt_ok,
            "n_edges_after_void_filter": int(len(self.e_cat)),
            "edge_density": self.edge_density,
            "per_cam": {c: {"n_selected": int(len(self.sel[c]["idx"])), "n_occluded": self.sel[c]["n_occluded"],
                            "per_cat": np.bincount(self.e_cat[self.sel[c]["idx"]], minlength=len(CATS))}
                        for c in CAMS},
        }



    # ------------------------------------------------------------------ dessin
    def draw(self, cam: str, cfg: dict, label: str = ""):
        """Image de superposition (BGR) : gris, contours Canny en cyan, discontinuités projetées
        (rouge = jump_col, bleu = jump_row, orange = void). Nécessite `keep_images=True`."""
        import cv2
        import numpy as np

        prm = self.prm
        img = cv2.cvtColor(self.gray[cam], cv2.COLOR_GRAY2BGR)
        e = self.edges[cam] > 0
        img[e] = (0.35 * img[e] + 0.65 * np.array([255, 255, 0])).astype(np.uint8)
        eb = self.world_to_base(self.e_world, self.pose(float(cfg.get("dt_s", 0.0)))) if cfg.get("motion_comp", True) \
            else self.e_base_raw
        T = apply_config_extrinsics(self.T_base_cam, cfg)[cam]
        s = self.sel[cam]
        pc = (eb[s["idx"]] - T[:3, 3]) @ T[:3, :3]
        uv, ok, _ = self.cams[cam].project(pc, margin=prm["eval_margin_px"], theta_limit=np.radians(prm["theta_limit_deg"]))
        d = np.full(len(pc), np.inf)
        d[ok] = sample_bilinear(self.dt[cam], uv[ok, 0], uv[ok, 1])
        colors = {0: (0, 0, 255), 1: (255, 90, 0), 2: (0, 165, 255), 3: (0, 165, 255)}
        cats = self.e_cat[s["idx"]]
        for (u, v), c, o in zip(uv, cats, ok):
            if o:
                cv2.circle(img, (int(round(u)), int(round(v))), 2, colors[int(c)], -1, lineType=cv2.LINE_AA)
        med = float(np.median(np.minimum(d, 1e3))) if len(d) else float("nan")
        jc = cats == 0
        med_jc = float(np.median(np.minimum(d[jc], 1e3))) if jc.any() else float("nan")
        txt = f"{label}  median {med:.2f} px (jump_col {med_jc:.2f} px), n={len(d)}"
        cv2.putText(img, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        center = np.median(uv[ok & jc], axis=0) if (ok & jc).sum() >= 5 else np.array([self.cams[cam].cx, self.cams[cam].cy])
        return img, med, center


# =============================================================================
# Plan : choix des triplets
# =============================================================================

# Demi-tour sur place (jusqu'à ~80°/s autour de z) puis arrêt de ~3 s : même lieu à l'arrêt et en rotation.
TURN_WINDOW_S = (213.5, 218.5)


def plan_frames(sc) -> dict:
    """Choisit les triplets et les tours LiDAR qui couvrent ±50 ms autour de chacun.

    Ensembles (un triplet peut appartenir à plusieurs) : `static` (v < 0,1 m/s et ω < 10°/s), `walk` (80 triplets
    répartis sur la séquence, v > 0,5 m/s, ω < 20°/s), `fast` (ω > 35°/s, espacés d'au moins 1 s), `turn`
    (tous les triplets de la fenêtre TURN_WINDOW_S).
    """
    import os

    import numpy as np

    from splatpipe.common import geometry as g

    fr = sc.load_frames()
    tg, pg, qg = _load_traj(sc)
    t = fr["t"]
    h = 0.05
    p0, q0, v0 = g.interpolate_poses(tg, pg, qg, t - h)
    p1, q1, v1 = g.interpolate_poses(tg, pg, qg, t + h)
    speed = np.linalg.norm(p1 - p0, axis=1) / (2 * h)
    om = g.rot_to_rotvec(np.swapaxes(g.quat_to_rot(q0), 1, 2) @ g.quat_to_rot(q1)) / (2 * h)  # base, rad/s
    rate = np.degrees(np.linalg.norm(om, axis=1))

    names = sorted(f for f in os.listdir(sc.path(LIDAR_SUBDIR)) if f.endswith(".npz"))
    l_t = g.ns_to_s(np.array([int(f[:-4]) for f in names], dtype=np.int64))

    def scans_for(ti):
        j = np.nonzero((l_t <= ti + h + 0.005) & (l_t >= ti - h - 0.105))[0]
        covered = len(j) > 0 and l_t[j[0]] <= ti - h + 0.002 and l_t[j[-1]] + 0.098 >= ti + h
        return [names[k] for k in j], bool(covered)

    ok = v0 & v1 & (t - h > tg[0]) & (t + h < tg[-1])
    rel = t - t[0]
    tags: dict[int, list] = {}

    def add(ids, tag):
        for i in ids:
            tags.setdefault(int(i), []).append(tag)

    add(np.nonzero(ok & (speed < STATIC_MAX_SPEED) & (rate < STATIC_MAX_RATE_DEG))[0], "static")
    walk_c = np.nonzero(ok & (speed > WALK_MIN_SPEED) & (rate < WALK_MAX_RATE_DEG))[0]
    targets = np.linspace(rel[0] + 3, rel[-1] - 3, N_WALK)
    add(np.unique([walk_c[np.argmin(np.abs(rel[walk_c] - x))] for x in targets]), "walk")
    fast_c = np.nonzero(ok & (rate > FAST_MIN_RATE_DEG))[0]
    fast = []
    for i in fast_c[np.argsort(-rate[fast_c])]:
        if all(abs(rel[i] - rel[j]) >= FAST_MIN_SEP_S for j in fast):
            fast.append(int(i))
        if len(fast) >= N_FAST:
            break
    add(fast, "fast")
    add(np.nonzero(ok & (rel >= TURN_WINDOW_S[0]) & (rel <= TURN_WINDOW_S[1]))[0], "turn")
    frames, n_uncovered = [], 0
    for i in sorted(tags):
        sc_names, covered = scans_for(t[i])
        if not covered:
            n_uncovered += 1
            continue
        f = {"sets": tags[i], "frame_id": int(fr["frame_id"][i]), "stamp_ns": int(fr["stamp_ns"][i]),
             "t": float(t[i]), "t_rel": float(rel[i]), "speed_m_s": float(speed[i]), "rate_deg_s": float(rate[i]),
             "omega_base_deg_s": np.degrees(om[i]).tolist(), "scans": sc_names}
        for c in CAMS:
            f[f"{c}_rel_path"] = str(fr[f"{c}_rel_path"][i])
        frames.append(f)
    return {"frames": frames, "n_uncovered": n_uncovered, "n_walk_candidates": int(len(walk_c)),
            "n_fast_candidates": int(len(fast_c))}


def frame_sets(frames: list) -> dict:
    """Masques booléens des ensembles d'analyse."""
    import numpy as np

    has = lambda tag: np.array([tag in f["sets"] for f in frames])  # noqa: E731
    rate = np.array([f["rate_deg_s"] for f in frames])
    speed = np.array([f["speed_m_s"] for f in frames])
    out = {"static": has("static"), "walk": has("walk"), "fast": has("fast"), "turn": has("turn")}
    out["moving"] = out["walk"] | out["fast"]
    out["turn_static"] = out["turn"] & (rate < STATIC_MAX_RATE_DEG) & (speed < 0.15)
    out["turn_rotating"] = out["turn"] & (rate >= 30.0)
    if all("yaw_world_deg" in f for f in frames):
        # Fin du demi-tour : en rotation (≥ 20°/s) mais cap à ±20° du cap à l'arrêt → même scène qu'à l'arrêt.
        yaw = np.array([f["yaw_world_deg"] for f in frames])
        ts = out["turn_static"]
        if ts.any():
            ref = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(yaw[ts])))))
            dyaw = (yaw - ref + 180.0) % 360.0 - 180.0
            out["turn_matched"] = out["turn"] & (rate >= 20.0) & (np.abs(dyaw) <= 20.0)
    out["all"] = np.ones(len(frames), dtype=bool)
    return out


def add_world_yaw(sc, frames: list) -> None:
    """Ajoute le cap (lacet de l'axe x de base dans world, degrés) à chaque triplet."""
    import numpy as np

    from splatpipe.common import geometry as g

    tg, pg, qg = _load_traj(sc)
    _, q, _ = g.interpolate_poses(tg, pg, qg, np.array([f["t"] for f in frames]))
    R = g.quat_to_rot(q)
    for f, Ri in zip(frames, R):
        f["yaw_world_deg"] = float(np.degrees(np.arctan2(Ri[1, 0], Ri[0, 0])))


# =============================================================================
# Configurations
# =============================================================================

YAW_PRELIM_DEG = 0.15  # estimation préliminaire (stage variants) ; sert seulement à centrer un balayage temporel
HANDEYE_YAW_DEG = (0.87, 0.93)  # corrections de lacet COLMAP (T0.5 f) et gyroscope (T0.5 d) de la sous-étape 0.3
GRID3D = {"roll": (-0.6, 0.6, 0.15), "pitch": (-0.6, 0.6, 0.15), "yaw": (-0.4, 1.2, 0.05)}  # degrés, axes x, y, z de base


def _specials(other_calibs: dict) -> list:
    import numpy as np

    cfgs = [
        {"name": "provided"},
        {"name": "rz+0.9", "rotvec_deg": [0.0, 0.0, 0.9]},
        {"name": "gyro_T05d", "rotvec_deg": [-0.07, -0.21, 0.93]},
        {"name": "colmap_T05f", "rotvec_deg": [-0.01, 0.01, 0.87]},
        {"name": "no_motion_comp", "motion_comp": False},
    ]
    for k, T in other_calibs.items():
        cfgs.append({"name": f"calib_{k}", "T_base_cam": {c: np.asarray(T[c]).tolist() for c in CAMS}})
    return cfgs


def configs_sweeps(other_calibs: dict) -> list:
    import numpy as np

    cfgs = _specials(other_calibs)
    grid = np.round(np.arange(-2.0, 2.0001, 0.05), 3)
    for ax, lab in ((2, "yaw"), (1, "pitch"), (0, "roll")):
        for a in grid:
            rv = [0.0, 0.0, 0.0]
            rv[ax] = float(a)
            cfgs.append({"name": f"sweep_{lab}_{a:+.2f}", "rotvec_deg": rv, "sweep": lab, "value": float(a)})
    for a in grid:
        cfgs.append({"name": f"sweep_yawcam_{a:+.2f}", "rotvec_deg": [0.0, 0.0, float(a)], "pivot": "cam",
                     "sweep": "yaw_pivot_cam", "value": float(a)})
    for base_name, rv in (("R0", [0.0, 0.0, 0.0]), ("Rprelim", [0.0, 0.0, YAW_PRELIM_DEG]), ("Rz0.9", [0.0, 0.0, 0.9])):
        for ms in range(-20, 21):
            cfgs.append({"name": f"dt_{base_name}_{ms:+d}ms", "rotvec_deg": rv, "dt_s": ms / 1000.0,
                         "sweep": f"dt_{base_name}", "value": ms / 1000.0})
    return cfgs


def configs_variants() -> list:
    import numpy as np

    cfgs = [{"name": "provided"}]
    grid = np.round(np.arange(-2.0, 2.0001, 0.1), 3)
    for ax, lab in ((2, "yaw"), (1, "pitch"), (0, "roll")):
        for a in grid:
            rv = [0.0, 0.0, 0.0]
            rv[ax] = float(a)
            cfgs.append({"name": f"sweep_{lab}_{a:+.2f}", "rotvec_deg": rv, "sweep": lab, "value": float(a)})
    for ms in range(-20, 21, 2):
        cfgs.append({"name": f"dt_R0_{ms:+d}ms", "dt_s": ms / 1000.0, "sweep": "dt_R0", "value": ms / 1000.0})
    return cfgs


def grid3d_axes():
    import numpy as np

    return {k: np.round(np.arange(a, b + 1e-9, s), 3) for k, (a, b, s) in GRID3D.items()}


def configs_grid() -> list:
    ax = grid3d_axes()
    cfgs = [{"name": "provided"}]
    for rx in ax["roll"]:
        for ry in ax["pitch"]:
            for rz in ax["yaw"]:
                cfgs.append({"name": f"grid_{rx:+.2f}_{ry:+.2f}_{rz:+.2f}", "rotvec_deg": [float(rx), float(ry), float(rz)],
                             "sweep": "grid3d", "value": [float(rx), float(ry), float(rz)]})
    return cfgs


def configs_final(other_calibs: dict, cands: dict, dt_center: dict) -> list:
    """Configurations spéciales + candidates (rotation commune ou par caméra) + balayage temporel fin au candidat."""
    import numpy as np

    cfgs = _specials(other_calibs)
    for name, c in cands.items():
        cfgs.append({"name": name, **c})
    cfgs.append({"name": f"{dt_center['name']}_pivot_cam", **dt_center["cfg"], "pivot": "cam"})
    for ms in np.round(np.arange(-20.0, 20.01, 1.0), 1):
        cfgs.append({"name": f"dt_final_{ms:+.1f}ms", **dt_center["cfg"], "dt_s": float(ms) / 1000.0,
                     "sweep": "dt_final", "value": float(ms) / 1000.0})
    return cfgs


# =============================================================================
# Analyse (numpy)
# =============================================================================


def block_ids(frames: list, block_s: float = 3.0):
    import numpy as np

    return np.floor(np.array([f["t_rel"] for f in frames]) / block_s).astype(np.int64)


def _group_idx(cats, dbins=None, collapsed=False):
    nd = len(DEPTH_EDGES) - 1
    if collapsed:
        return [CATS.index(c) for c in cats]
    return [CATS.index(c) * nd + d for c in cats for d in (dbins if dbins is not None else range(nd))]


def per_frame_sums(S, fmask, cams, cats, stat=1, dbins=None, collapsed=False):
    """Réduction par triplet : (num [F', C], den [F', C]) = Σ_stat et Σ_n sur les caméras et groupes choisis.
    S : [F, C, n_cam, G, 4] (G = catégories × classes de profondeur, ou catégories si `collapsed`)."""
    import numpy as np

    g = _group_idx(cats, dbins, collapsed)
    sub = S[fmask][:, :, cams][:, :, :, g, :]
    num = sub[..., stat].astype(np.float64).sum(axis=(2, 3))
    den = sub[..., 0].astype(np.float64).sum(axis=(2, 3))
    return num, den


def curve(S, fmask, cams, cats, stat=1, weights=None, dbins=None, collapsed=False):
    """Objectif moyen par configuration (Σ_stat / Σ_n) sur les triplets, caméras et groupes choisis."""
    import numpy as np

    num, den = per_frame_sums(S, fmask, cams, cats, stat, dbins, collapsed)
    w = np.ones(num.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    return (w @ num) / np.maximum(w @ den, 1e-9), w @ den


def parabolic_argmin(vals, o):
    import numpy as np

    k = int(np.argmin(o))
    if 0 < k < len(o) - 1:
        y0, y1, y2 = o[k - 1], o[k], o[k + 1]
        den = y0 - 2 * y1 + y2
        if den > 0:
            return float(vals[k] + 0.5 * (y0 - y2) / den * (vals[k + 1] - vals[k])), False
    return float(vals[k]), bool(k in (0, len(o) - 1))


def _boot_weights(blocks_sel, nboot, seed):
    """Poids de bootstrap par blocs : chaque tirage rééchantillonne les blocs avec remise."""
    import numpy as np

    rng = np.random.default_rng(seed)
    ub, inv = np.unique(blocks_sel, return_inverse=True)
    W = np.zeros((nboot, len(blocks_sel)))
    for b in range(nboot):
        cnt = np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)).astype(np.float64)
        W[b] = cnt[inv]
    return W, len(ub)


def estimate_1d(S, cfgs, sweep, fmask, cams, cats, blocks, nboot=300, stat=1, dbins=None, seed=0):
    """Minimum d'un balayage 1D + incertitude par bootstrap par blocs de triplets (blocs de 3 s)."""
    import numpy as np

    ks = [i for i, c in enumerate(cfgs) if c.get("sweep") == sweep]
    vals = np.array([cfgs[i]["value"] for i in ks], dtype=np.float64)
    if fmask.sum() == 0:
        return None
    num, den = per_frame_sums(S[:, ks], fmask, cams, cats, stat, dbins)
    o = num.sum(0) / np.maximum(den.sum(0), 1e-9)
    best, at_edge = parabolic_argmin(vals, o)
    W, nb = _boot_weights(blocks[fmask], nboot, seed)
    ob = (W @ num) / np.maximum(W @ den, 1e-9)
    boots = np.array([parabolic_argmin(vals, row)[0] for row in ob])
    k0 = int(np.argmin(np.abs(vals)))
    return {"best": best, "at_grid_edge": at_edge, "boot_std": float(boots.std()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_blocks": int(nb), "n_frames": int(fmask.sum()), "n_points_per_frame": float(den.sum(0)[k0] / max(fmask.sum(), 1)),
            "objective_at_0": float(o[k0]), "objective_min": float(o.min()),
            "objective_at_ends": [float(o[0]), float(o[-1])]}


def estimate_3d(SG, cfgs, fmask, cams, cats, blocks, nboot=200, seed=0):
    """Minimum sur la grille 3D (roll, pitch, yaw de base) + raffinement parabolique par axe + bootstrap."""
    import numpy as np

    ks = [i for i, c in enumerate(cfgs) if c.get("sweep") == "grid3d"]
    ax = grid3d_axes()
    shape = (len(ax["roll"]), len(ax["pitch"]), len(ax["yaw"]))

    def solve(o):
        o3 = o.reshape(shape)
        i, j, k = np.unravel_index(int(np.argmin(o3)), shape)
        r = [parabolic_argmin(ax["roll"], o3[:, j, k])[0], parabolic_argmin(ax["pitch"], o3[i, :, k])[0],
             parabolic_argmin(ax["yaw"], o3[i, j, :])[0]]
        edge = i in (0, shape[0] - 1) or j in (0, shape[1] - 1) or k in (0, shape[2] - 1)
        return np.array(r), edge, float(o3.min())

    num, den = per_frame_sums(SG[:, ks], fmask, cams, cats, 1, collapsed=True)
    best, edge, omin = solve(num.sum(0) / np.maximum(den.sum(0), 1e-9))
    W, _ = _boot_weights(blocks[fmask], nboot, seed)
    ob = (W @ num) / np.maximum(W @ den, 1e-9)
    boots = np.array([solve(row)[0] for row in ob])
    kp = [i for i, c in enumerate(cfgs) if c["name"] == "provided"][0]
    op, _ = curve(SG[:, [kp]], fmask, cams, cats, 1, collapsed=True)
    return {"best_rotvec_deg": best, "at_grid_edge": bool(edge), "boot_std_deg": boots.std(0),
            "ci95_deg": [np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)],
            "objective_min": omin, "objective_provided": float(op[0]), "n_frames": int(fmask.sum())}


def hist_stats(H, fmask, hk, cams, cats, dbins=None):
    """Médiane, moyenne des 50 % meilleurs, fraction < 2 px et effectif, depuis les histogrammes (pas 0,1 px)."""
    import numpy as np

    g = _group_idx(cats, dbins)
    h = H[fmask][:, hk][:, cams][:, :, g, :].reshape(-1, H.shape[-1]).sum(0).astype(np.float64)
    n = h.sum()
    if n == 0:
        return {"n": 0, "median_px": None, "mean_best50_px": None, "frac_lt_2px": None}
    bw = PRM["hist_bin_px"]
    cdf = np.cumsum(h) / n
    k = int(np.searchsorted(cdf, 0.5))
    lo = cdf[k - 1] if k > 0 else 0.0
    med = (k + (0.5 - lo) / max(cdf[k] - lo, 1e-12)) * bw  # interpolation linéaire dans la classe
    centers = (np.arange(len(h)) + 0.5) * bw
    cum = np.cumsum(h)
    take = np.minimum(h, np.maximum(n / 2 - (cum - h), 0))
    return {"n": int(n), "median_px": float(min(med, bw * (len(h) - 1))), "mean_best50_px": float((take * centers).sum() / take.sum()),
            "frac_lt_2px": float(h[: int(round(2.0 / bw))].sum() / n),
            "median_is_lower_bound": bool(k >= len(h) - 1)}


# =============================================================================
# Fonctions Modal
# =============================================================================


def _load_traj(sc):
    from splatpipe.common import geometry as g

    tg, pg, qg = sc.load_traj("gt")
    tg, pg, qg, _ = g.clean_trajectory(tg, pg, qg)
    return tg, pg, qg


def _other_calibs(sensors) -> dict:
    import numpy as np

    T_bl = np.array(sensors["lidars"][LIDAR_ID]["T_base_lidar"], dtype=np.float64)
    out = {}
    for d in OTHER_CALIBS:
        p = f"{DATA_MOUNT}/{CALIB_ROOT}/{d}/cam-lidar-imu.yaml"
        out[d.replace("calibration-sequences-", "")] = T_base_cam_from_camlidar_yaml(p, T_bl)
    return out


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=2.0, memory=4096, timeout=1800, retries=1)
def process_frame(frame: dict, configs: list, hist_names: list, seq: str = SEQ, variants: dict | None = None,
                  collapse_depth: bool = False):
    """Évalue les configurations sur un triplet, pour chaque variante de paramètres (`{nom: surcharge de PRM}`)."""
    import time

    import numpy as np

    from splatpipe.common.lidar_model import StructuredSpinningLidarModel

    t0 = time.time()
    sc = scene(seq)
    sensors = load_provided_sensors(sc)
    model = StructuredSpinningLidarModel.load_json(sc.path(LIDAR_MODEL))
    traj = _load_traj(sc)
    names = [c["name"] for c in configs]
    hist_ids = [names.index(n) for n in hist_names]
    Ss, Hs, sts = [], [], []
    nd = len(DEPTH_EDGES) - 1
    for vname, over in (variants or {"default": {}}).items():
        t1 = time.time()
        fd = FrameData(sc, frame, sensors, model, traj, prm={**PRM, **over})
        t2 = time.time()
        S, H = fd.evaluate(configs, hist_ids)
        if collapse_depth:
            S = S.reshape(S.shape[0], S.shape[1], len(CATS), nd, S.shape[-1]).sum(axis=3)
        st = fd.stats()
        st["variant"] = vname
        st["runtime_prep_s"] = round(t2 - t1, 2)
        st["runtime_eval_s"] = round(time.time() - t2, 2)
        Ss.append(S.astype("float32"))
        Hs.append(H)
        sts.append(st)
    return {"S": np.stack(Ss), "H": np.stack(Hs), "stats": sts, "runtime_s": round(time.time() - t0, 2)}


def _run_map(frames, configs, hist_names, seq, variants=None, collapse_depth=False):
    import numpy as np

    results = list(process_frame.map(frames, kwargs={"configs": configs, "hist_names": hist_names, "seq": seq,
                                                     "variants": variants, "collapse_depth": collapse_depth},
                                     return_exceptions=True))
    good = [(f, r) for f, r in zip(frames, results) if isinstance(r, dict)]
    errors = [{"frame_id": f["frame_id"], "error": repr(r)[:500]} for f, r in zip(frames, results) if not isinstance(r, dict)]
    S = np.stack([r["S"] for _, r in good])  # [F, V, C, cams, G, 4]
    H = np.stack([r["H"] for _, r in good])
    return [f for f, _ in good], S, H, [r["stats"] for _, r in good], errors


def _save_npz(sc, stage, S, H, meta):
    import json
    import os

    import numpy as np

    out_dir = sc.path(OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(f"{out_dir}/{stage}_scores.npz", S=S, H=H)
    with open(f"{out_dir}/{stage}_meta.json", "w") as f:
        json.dump(_py(meta), f, ensure_ascii=False)


def _load_npz(sc, stage):
    import json

    import numpy as np

    with np.load(sc.path(OUT_SUBDIR, f"{stage}_scores.npz")) as z:
        S, H = z["S"], z["H"]
    with open(sc.path(OUT_SUBDIR, f"{stage}_meta.json")) as f:
        meta = json.load(f)
    return S, H, meta


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=2.0, memory=16384, timeout=3600)
def run_stage(stage: str, seq: str = SEQ, limit: int = 0):
    """Stages `variants`, `sweeps`, `grid` : fan-out sur les triplets, sauvegarde des scores bruts."""
    import time

    t0 = time.time()
    sc = scene(seq)
    sensors = load_provided_sensors(sc)
    plan = plan_frames(sc)
    frames = plan["frames"]
    if limit:
        frames = frames[:: max(1, len(frames) // limit)][:limit]
    other = _other_calibs(sensors)
    variants, collapse = None, False
    if stage == "sweeps":
        configs = configs_sweeps(other)
    elif stage == "grid":
        configs = configs_grid()
        collapse = True
    elif stage == "variants":
        configs = configs_variants()
        variants = VARIANTS
    else:
        raise ValueError(stage)
    hist_names = [c["name"] for c in configs if "sweep" not in c]
    frames, S, H, stats, errors = _run_map(frames, configs, hist_names, seq, variants, collapse)
    meta = {"stage": stage, "seq": seq, "prm": PRM, "cats": CATS, "depth_edges": DEPTH_EDGES, "cams": CAMS,
            "collapsed_depth": collapse, "configs": configs, "hist_names": hist_names, "frames": frames,
            "variants": variants or {"default": {}}, "frame_stats": stats, "errors": errors,
            "plan_counts": {k: v for k, v in plan.items() if k != "frames"}}
    _save_npz(sc, stage, S, H, meta)
    vol.commit()
    sets = frame_sets(frames)
    return _py({"stage": stage, "n_frames": len(frames), "sets": {k: int(v.sum()) for k, v in sets.items()},
                "n_errors": len(errors), "errors": errors[:5], "runtime_s": round(time.time() - t0, 1),
                "S_shape": list(S.shape)})


# =============================================================================
# Finalisation : évaluation des candidats, T0.4, décision, images, rapport
# =============================================================================

# Règle de décision (documentée dans le rapport et la note stage0-5-overlay.md).
DECISION = {
    "handeye_yaw_tol_deg": 0.2,  # |lacet estimé − 0,9°| ≤ 0,2° pour confirmer l'hypothèse gyro/COLMAP
    "cam_consistency_tol_deg": 0.15,  # écart max entre caméras (lacet, triplets en mouvement)
    "set_consistency_tol_deg": 0.15,  # |lacet(arrêt) − lacet(mouvement)|
    "min_improvement_px": 0.5,  # gain minimal de médiane (toutes caméras, tous triplets)
    "min_significant_yaw_deg": 0.3,  # ≈ 3,6 px au centre ; en dessous, dans la dispersion des calibrations
    "t04_median_px": 2.0,
    "t04_same_px": 1.0,  # |médiane(rotation) − médiane(arrêt)| au même endroit (fenêtre `turn`)
    "t04_max_dt_ms": 2.0,
}
MAIN_CATS = ("jump_col",)  # catégorie de référence pour les estimations (bords verticaux, résolution 0,6°)
REPORT_SETS = ("static", "turn_static", "turn_matched", "turn_rotating", "walk", "fast", "moving", "all")


def _rotvec_between(T_ref, T_other):
    import numpy as np

    from splatpipe.common import geometry as g

    return np.degrees(g.rot_to_rotvec(np.asarray(T_other)[:3, :3] @ np.asarray(T_ref)[:3, :3].T))


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=2.0, memory=8192, timeout=1800, retries=1)
def render_overlays(frame: dict, cfgs: list, seq: str = SEQ, crop_wh=(400, 300), zoom: int = 2):
    """Images de superposition : une image pleine résolution par (caméra, configuration) + une bande de
    comparaison zoomée par caméra. Écrit sous reports/stage0_overlay/ et renvoie les octets JPEG."""
    import os

    import cv2
    import numpy as np

    from splatpipe.common.lidar_model import StructuredSpinningLidarModel

    sc = scene(seq)
    sensors = load_provided_sensors(sc)
    model = StructuredSpinningLidarModel.load_json(sc.path(LIDAR_MODEL))
    fd = FrameData(sc, frame, sensors, model, _load_traj(sc), keep_images=True)
    out_dir = sc.path(OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"f{frame['frame_id']:05d}_{frame['overlay_label']}"
    files = {}
    for cam in CAMS:
        panels, center = [], None
        for cfg in cfgs:
            img, med, ctr = fd.draw(cam, cfg, label=f"{cam} {frame['overlay_label']} t={frame['t_rel']:.2f}s "
                                                   f"w={frame['rate_deg_s']:.0f}deg/s | {cfg['name']}")
            center = ctr if center is None else center  # même recadrage pour toutes les configurations
            name = f"{tag}_{cam}_{cfg['name']}.jpg"
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            with open(f"{out_dir}/{name}", "wb") as f:
                f.write(buf.tobytes())
            files[name] = buf.tobytes()
            w, h = crop_wh
            x0 = int(np.clip(center[0] - w / 2, 0, img.shape[1] - w))
            y0 = int(np.clip(center[1] - h / 2, 0, img.shape[0] - h))
            crop = cv2.resize(img[y0:y0 + h, x0:x0 + w], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
            cv2.putText(crop, cfg["name"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(crop, cfg["name"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(crop, f"median {med:.2f} px", (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(crop, f"median {med:.2f} px", (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(crop)
        strip = np.hstack([np.pad(pn, ((0, 0), (0, 6), (0, 0)), constant_values=255) for pn in panels])
        name = f"{tag}_{cam}_compare_zoom.jpg"
        ok, buf = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open(f"{out_dir}/{name}", "wb") as f:
            f.write(buf.tobytes())
        files[name] = buf.tobytes()
    vol.commit()
    return files


def _yaml_dump(obj) -> str:
    """YAML lisible (même style que l'ingestion) : listes de scalaires en « flow »."""
    import yaml

    class Dumper(yaml.SafeDumper):
        pass

    def repr_list(dumper, data):
        flow = all(isinstance(v, (int, float, bool)) or v is None for v in data)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)

    Dumper.add_representer(list, repr_list)
    return yaml.dump(_py(obj), Dumper=Dumper, sort_keys=False, allow_unicode=True, width=120)


def write_corrected_sensors(sc, rotvec_deg, evidence: dict, date: str) -> dict:
    """Copie exacte `sensors.yaml` → `sensors_provided.yaml` (si absente), puis réécrit `sensors.yaml` avec
    `T_base_cam' = C · T_base_cam` et `T_base_imu' = C · T_base_imu` (bloc rigide caméras + IMU, pivot = origine de base)
    et une section `calibration_corrections`."""
    import os
    import shutil

    import numpy as np
    import yaml

    src = sc.sensors_path
    prov = sc.path("sensors_provided.yaml")
    if not os.path.exists(prov):
        shutil.copyfile(src, prov)
    with open(prov) as f:
        text = f.read()
    s = yaml.safe_load(text)
    C = np.eye(4)
    C[:3, :3] = rotvec_deg_to_R(rotvec_deg)
    for cam in CAMS:
        s["cameras"][cam]["T_base_cam"] = C @ np.array(s["cameras"][cam]["T_base_cam"], dtype=np.float64)
        s["cameras"][cam]["source"]["correction"] = "calibration_corrections.camera_block_rotation"
    s["imu"]["T_base_imu"] = C @ np.array(s["imu"]["T_base_imu"], dtype=np.float64)
    s["imu"]["source"]["correction"] = "calibration_corrections.camera_block_rotation"
    if "derived" in s:
        s["derived"]["note"] = "valeurs de contrôle calculées avec la calibration fournie (avant correction)"
    s["calibration_corrections"] = {
        "camera_block_rotation": {
            "applied_to": list(CAMS) + ["imu"],
            "formula": "T_base_x' = C · T_base_x_provided, C = exp(rotvec) (rotation autour de l'origine de base)",
            "rotvec_base_deg": [float(v) for v in rotvec_deg],
            "date": date,
            "provided_calibration": "sensors_provided.yaml",
            "evidence": evidence,
        }
    }
    head = [ln for ln in text.splitlines(keepends=True) if ln.startswith("#")][:4]
    with open(src, "w") as f:
        f.write("".join(head))
        f.write("# CORRIGÉ par splatpipe.stage0.overlay_check (voir calibration_corrections ; original : sensors_provided.yaml)\n")
        f.write(_yaml_dump(s))
    return {"written": src, "provided_copy": prov}


def _stats_tables(S, H, cfgs, hist_names, frames, sets, cfg_list):
    out = {}
    for name in cfg_list:
        hk = hist_names.index(name)
        out[name] = {}
        for st in REPORT_SETS:
            m = sets[st]
            if m.sum() == 0:
                continue
            row = {}
            for ci, cam in enumerate(CAMS):
                row[cam] = {"all": hist_stats(H, m, hk, [ci], CATS), "jump_col": hist_stats(H, m, hk, [ci], ["jump_col"])}
            row["cams_pooled"] = {"all": hist_stats(H, m, hk, [0, 1, 2], CATS),
                                  "jump_col": hist_stats(H, m, hk, [0, 1, 2], ["jump_col"])}
            row["n_frames"] = int(m.sum())
            out[name][st] = row
    return out


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=4.0, memory=32768, timeout=3600)
def finalize(seq: str = SEQ, apply: bool = False, nboot: int = 300):
    import datetime
    import time

    import numpy as np

    t0 = time.time()
    sc = scene(seq)
    sensors = load_provided_sensors(sc)
    other = _other_calibs(sensors)
    Sw, Hw, mw = _load_npz(sc, "sweeps")
    Sg, _, mg = _load_npz(sc, "grid")
    Sw, Hw, Sg = Sw[:, 0], Hw[:, 0], Sg[:, 0]
    frames = mw["frames"]
    assert [f["frame_id"] for f in frames] == [f["frame_id"] for f in mg["frames"]], "sweeps et grid : triplets différents"
    add_world_yaw(sc, frames)
    sets = frame_sets(frames)
    blocks = block_ids(frames)
    cw = mw["configs"]
    CAMSEL = {"cam0": [0], "cam1": [1], "cam2": [2], "joint": [0, 1, 2]}
    CATSEL = {"jump_col": ["jump_col"], "jumps": ["jump_col", "jump_row"], "all": list(CATS)}
    est_sets = ("static", "walk", "fast", "turn_rotating", "moving", "all")

    # ------------------------------------------------------------ estimations 1D
    one_d = {}
    for sweep in ("yaw", "yaw_pivot_cam", "pitch", "roll"):
        one_d[sweep] = {cn: {st: {cam: estimate_1d(Sw, cw, sweep, sets[st], ci, cats, blocks, nboot=nboot)
                                  for cam, ci in CAMSEL.items()} for st in est_sets} for cn, cats in CATSEL.items()}
    yaw_depth = {f"{DEPTH_EDGES[d]}-{DEPTH_EDGES[d + 1]}m": {st: estimate_1d(Sw, cw, "yaw", sets[st], [0, 1, 2], ["jump_col"],
                                                                         blocks, nboot=nboot, dbins=[d])
                                                           for st in ("moving", "all")} for d in range(len(DEPTH_EDGES) - 1)}
    time_sweeps = {}
    for sweep in ("dt_R0", "dt_Rprelim", "dt_Rz0.9"):
        time_sweeps[sweep] = {cn: {st: {cam: estimate_1d(Sw, cw, sweep, sets[st], ci, cats, blocks, nboot=nboot)
                                        for cam, ci in CAMSEL.items()} for st in ("fast", "turn_rotating", "turn_matched", "moving")}
                              for cn, cats in (("jump_col", ["jump_col"]), ("all", list(CATS)))}

    # ------------------------------------------------------------ estimations 3D
    grid3d = {cn: {st: {cam: estimate_3d(Sg, mg["configs"], sets[st], ci, cats, blocks, nboot=max(50, nboot // 3))
                        for cam, ci in CAMSEL.items()} for st in ("static", "moving", "all")}
              for cn, cats in CATSEL.items()}

    # ------------------------------------------------------------ candidats
    yaw_main = one_d["yaw"]["jump_col"]["all"]["joint"]
    yaw_est = round(yaw_main["best"], 2)
    rv3 = [round(float(v), 2) for v in grid3d["jump_col"]["all"]["joint"]["best_rotvec_deg"]]
    rv_cam = {cam: [round(float(v), 2) for v in grid3d["jump_col"]["all"][cam]["best_rotvec_deg"]] for cam in CAMS}
    cands = {"est_yaw": {"rotvec_deg": [0.0, 0.0, yaw_est]}, "est_rot3d": {"rotvec_deg": rv3},
             "est_rot3d_per_cam": {"rotvec_deg_by_cam": rv_cam}}
    cfg_final = configs_final(other, cands, {"name": "est_yaw", "cfg": cands["est_yaw"]})
    hist_final = [c["name"] for c in cfg_final if "sweep" not in c]
    ff, Sf, Hf, stats_f, err_f = _run_map(frames, cfg_final, hist_final, seq)
    assert [f["frame_id"] for f in ff] == [f["frame_id"] for f in frames], "final : triplets manquants"
    Sf, Hf = Sf[:, 0], Hf[:, 0]
    _save_npz(sc, "final", Sf[:, None], Hf[:, None], {"stage": "final", "configs": cfg_final, "hist_names": hist_final,
                                                      "frames": frames, "errors": err_f, "prm": PRM})
    dt_final = {cn: {st: {cam: estimate_1d(Sf, cfg_final, "dt_final", sets[st], ci, cats, blocks, nboot=nboot)
                          for cam, ci in CAMSEL.items()} for st in ("fast", "turn_rotating", "turn_matched", "moving", "all")}
                for cn, cats in (("jump_col", ["jump_col"]), ("all", list(CATS)))}

    # ------------------------------------------------------------ T0.4
    table_cfgs = ["provided", "est_yaw", "est_rot3d", "est_rot3d_per_cam", "est_yaw_pivot_cam", "rz+0.9", "gyro_T05d",
                  "colmap_T05f", "no_motion_comp"] + [f"calib_{k}" for k in other]
    scores = _stats_tables(Sf, Hf, cfg_final, hist_final, frames, sets, table_cfgs)
    kf = {c["name"]: i for i, c in enumerate(cfg_final)}

    def trunc_mean(name, st, cams=(0, 1, 2), cats=MAIN_CATS):
        o, n = curve(Sf[:, [kf[name]]], sets[st], list(cams), list(cats))
        return float(o[0])

    trunc = {name: {st: {"jump_col": trunc_mean(name, st), "all": trunc_mean(name, st, cats=CATS)}
                    for st in REPORT_SETS if sets[st].sum()} for name in table_cfgs}
    D = DECISION
    prov = scores["provided"]
    strict = {st: {cam: prov[st][cam]["all"]["median_px"] < D["t04_median_px"] for cam in CAMS}
              for st in ("static", "moving")}
    same_place = {cam: {"median_turn_static_px": prov["turn_static"][cam]["all"]["median_px"],
                        "median_turn_matched_px": prov["turn_matched"][cam]["all"]["median_px"]} for cam in CAMS}
    for cam in CAMS:
        v = same_place[cam]
        v["abs_diff_px"] = abs(v["median_turn_matched_px"] - v["median_turn_static_px"])
        v["median_turn_matched_without_motion_compensation_px"] = scores["no_motion_comp"]["turn_matched"][cam]["all"]["median_px"]
    same_place["n_frames"] = {"turn_static": int(sets["turn_static"].sum()), "turn_matched": int(sets["turn_matched"].sum())}
    same_place["rate_deg_s_turn_matched"] = [float(f["rate_deg_s"]) for f, m in zip(frames, sets["turn_matched"]) if m]
    dt_best_ms = dt_final["jump_col"]["fast"]["joint"]["best"] * 1000
    t04 = {
        "criterion": "médiane des distances (px) des discontinuités de profondeur LiDAR projetées aux contours Canny < 2 px, "
                     "identique à l'arrêt et en mouvement",
        "pass_strict_median_lt_2px": strict,
        "same_place_static_vs_rotating": same_place,
        "time_offset_best_ms_fast_frames": dt_best_ms,
        "time_offset_ci95_ms": [v * 1000 for v in dt_final["jump_col"]["fast"]["joint"]["ci95"]],
    }
    t04["pass_strict"] = bool(all(all(v.values()) for v in strict.values()))
    t04["pass_static_vs_moving"] = bool(all(same_place[c]["abs_diff_px"] <= D["t04_same_px"] for c in CAMS)
                                        and abs(dt_best_ms) <= D["t04_max_dt_ms"])
    t04["status"] = "PASS" if (t04["pass_strict"] and t04["pass_static_vs_moving"]) else "FAIL"

    # ------------------------------------------------------------ comparaisons externes
    ingest = sc.load_report("stage0_ingest")
    t05 = ingest.get("T0.5", {})
    handeye = {
        "gyro_T05d_rotvec_base_deg": t05.get("d_gyro_vs_gt", {}).get("correction_rotvec_in_base_deg"),
        "colmap_T05f_rotvec_base_deg": t05.get("f_cam0_rotation_vs_colmap", {}).get("median_correction_rotvec_in_base_deg"),
    }
    T_prov = {c: np.array(sensors["cameras"][c]["T_base_cam"]) for c in CAMS}
    other_rv = {k: {c: _rotvec_between(T_prov[c], T[c]) for c in CAMS} for k, T in other.items()}

    # ------------------------------------------------------------ décision
    ym = {cam: one_d["yaw"]["jump_col"]["moving"][cam]["best"] for cam in CAMS}
    y_static = one_d["yaw"]["jump_col"]["static"]["joint"]["best"]
    y_moving = one_d["yaw"]["jump_col"]["moving"]["joint"]["best"]
    med_prov = scores["provided"]["all"]["cams_pooled"]["all"]["median_px"]
    med_09 = scores["rz+0.9"]["all"]["cams_pooled"]["all"]["median_px"]
    med_est = scores["est_yaw"]["all"]["cams_pooled"]["all"]["median_px"]
    checks = {
        "yaw_estimate_deg": yaw_main["best"],
        "yaw_ci95_deg": yaw_main["ci95"],
        "consistent_with_handeye_0.9": bool(abs(yaw_main["best"] - 0.9) <= D["handeye_yaw_tol_deg"]),
        "cameras_consistent": bool(max(ym.values()) - min(ym.values()) <= D["cam_consistency_tol_deg"]),
        "static_vs_moving_consistent": bool(abs(y_static - y_moving) <= D["set_consistency_tol_deg"]),
        "median_all_px": {"provided": med_prov, "est_yaw": med_est, "rz+0.9": med_09},
        "rz0.9_improves_median": bool(med_prov - med_09 >= D["min_improvement_px"]),
        "est_yaw_improves_median": bool(med_prov - med_est >= D["min_improvement_px"]),
        "est_yaw_significant": bool(abs(yaw_main["best"]) >= D["min_significant_yaw_deg"]
                                    and (yaw_main["ci95"][0] > 0 or yaw_main["ci95"][1] < 0)),
    }
    apply_09 = (checks["consistent_with_handeye_0.9"] and checks["cameras_consistent"]
                and checks["static_vs_moving_consistent"] and checks["rz0.9_improves_median"])
    apply_small = (checks["est_yaw_significant"] and checks["cameras_consistent"]
                   and checks["static_vs_moving_consistent"] and checks["est_yaw_improves_median"])
    decision = {
        "rule": D,
        "checks": checks,
        "correct_sensors_yaml": bool(apply_09 or apply_small),
        "rotvec_to_apply_deg": ([0.0, 0.0, yaw_est] if (apply_09 or apply_small) else None),
        "applied": False,
    }
    date = datetime.date.today().isoformat()
    if decision["correct_sensors_yaml"] and apply:
        evidence = {"report": "reports/stage0_overlay.json", "yaw_estimate_deg": yaw_main, "checks": checks}
        decision["write"] = write_corrected_sensors(sc, decision["rotvec_to_apply_deg"], evidence, date)
        decision["applied"] = True

    # ------------------------------------------------------------ images de superposition
    fr_arr = np.array([f["t_rel"] for f in frames])
    rate = np.array([f["rate_deg_s"] for f in frames])
    picks = []
    ts = np.nonzero(sets["turn_static"])[0]
    if len(ts):
        picks.append((int(ts[np.argmin(np.abs(fr_arr[ts] - 216.4))]), "static"))
    tr = np.nonzero(sets["turn_rotating"])[0]
    if len(tr):
        picks.append((int(tr[np.argmax(rate[tr])]), "rotating"))
    wk = np.nonzero(sets["walk"])[0]
    if len(wk):
        nsel = np.array([sum(stats_f[i][0]["per_cam"][c]["per_cat"][0] for c in CAMS) for i in wk])
        picks.append((int(wk[np.argmax(nsel)]), "walking"))
    ov_cfgs = [{"name": "provided"}, {"name": f"est_yaw_{yaw_est:+.2f}deg", **cands["est_yaw"]},
               {"name": "rz+0.90deg", "rotvec_deg": [0.0, 0.0, 0.9]}]
    ov_frames = [{**frames[i], "overlay_label": lab} for i, lab in picks]
    overlay_files = {}
    for res in render_overlays.map(ov_frames, kwargs={"cfgs": ov_cfgs, "seq": seq}):
        overlay_files.update(res)

    report = {
        "step": "stage0_overlay",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "sequence": seq,
        "code": "splatpipe/stage0/overlay_check.py, splatpipe/common/camera_model.py",
        "method": {
            "edges_lidar": "discontinuités de profondeur de la grille (ring × colonne alignée en azimut) des tours "
                           "lidar/hesai/*.npz, écho le plus proche ; point avant-plan au mi-chemin angulaire du voisin, "
                           "ou au centre de la cellule si elle a un second écho lointain ; catégories " + ", ".join(CATS),
            "time": "points à |t_point − t_image| ≤ 50 ms, désanamorphosés à t_image avec la GT",
            "edges_image": f"Canny (flou gaussien σ={PRM['gauss_sigma']}, seuils {PRM['canny_low']}/{PRM['canny_high']}, L2) "
                           "+ transformée de distance, interpolation bilinéaire",
            "objective": f"moyenne des distances tronquées à {PRM['trunc_px']} px ; minimum par balayage (pas 0,05°) + "
                         "interpolation parabolique ; incertitude = bootstrap par blocs de 3 s",
            "correction_convention": "T_base_cam' = C · T_base_cam, C = exp(rotvec) exprimé dans base (x avant, y gauche, z haut)",
            "time_offset_convention": "t_référence = t_image + dt (dt > 0 : l'image est en réalité prise plus tard)",
            "variants": "stage `variants` : 5 jeux de paramètres (contours, position du bord) donnent le même lacet à ±0,05° ; "
                        "retenu : σ 1,2, seuils 40/110, raffinement double écho (meilleur contraste)",
        },
        "params": PRM,
        "frame_sets": {k: int(v.sum()) for k, v in sets.items()},
        "frames": [{k: f[k] for k in ("frame_id", "t_rel", "sets", "speed_m_s", "rate_deg_s", "yaw_world_deg")} for f in frames],
        "T0.4": t04,
        "scores": scores,
        "truncated_mean_px": trunc,
        "estimates": {"yaw_1d": one_d["yaw"], "yaw_pivot_cam_1d": one_d["yaw_pivot_cam"], "pitch_1d": one_d["pitch"],
                      "roll_1d": one_d["roll"], "yaw_by_depth_jump_col": yaw_depth, "grid3d": grid3d,
                      "candidates": cands},
        "time_offset": {"sweeps": time_sweeps, "final_at_est_yaw": dt_final},
        "comparisons": {"handeye_stage0_3": handeye, "handeye_yaw_range_deg": HANDEYE_YAW_DEG,
                        "other_calibration_dates_rotvec_vs_provided_deg": other_rv},
        "decision": decision,
        "overlays": sorted(f"{OUT_SUBDIR}/{n}" for n in overlay_files),
        "raw_scores": [f"{OUT_SUBDIR}/{s_}_scores.npz" for s_ in ("sweeps", "grid", "final")],
        "errors": err_f,
        "runtime_s": round(time.time() - t0, 1),
    }
    report = _py(report)
    sc.write_report("stage0_overlay", report)
    vol.commit()
    keep = {n: b for n, b in overlay_files.items() if n.endswith("compare_zoom.jpg") or "_cam0_" in n}
    return {"report": report, "files": keep}


@app.local_entrypoint()
def main(stage: str = "sweeps", seq: str = SEQ, limit: int = 0, apply: bool = False, nboot: int = 300):
    import json
    import pathlib

    if stage in ("variants", "sweeps", "grid"):
        rep = run_stage.remote(stage, seq, limit)
        print(json.dumps(rep, indent=1, ensure_ascii=False))
    elif stage == "finalize":
        res = finalize.remote(seq, apply, nboot)
        rep = res["report"]
        pathlib.Path("logs").mkdir(exist_ok=True)
        pathlib.Path("logs/stage0/overlay.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        d = pathlib.Path("logs/stage0/overlay")
        d.mkdir(exist_ok=True)
        for name, b in res["files"].items():
            (d / name).write_bytes(b)
        print("T0.4 :", json.dumps(rep["T0.4"], ensure_ascii=False, indent=1))
        print("décision :", json.dumps(rep["decision"], ensure_ascii=False, indent=1))
        print(f"{len(res['files'])} images dans {d} ; rapport logs/stage0/overlay.json ; {rep['runtime_s']} s")
    else:
        raise SystemExit(f"stage inconnu : {stage}")
