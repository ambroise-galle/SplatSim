"""Outils numpy partagés par les sous-étapes 1.4–1.6 (carte LiDAR) : lecture des tours, correction de
distorsion par point, statistiques de voxels (netteté des plans), sous-échantillonnage façon Open3D, ICP
point→plan, E/S PCD/PLY.

Conventions : `etapes/notes/conventions.md` (`T_a_b`, quaternions xyzw, temps Unix float64).
Tous les calculs de géométrie sont en float64 ; le stockage des points en float32.
"""

from __future__ import annotations

import json
import os

import numpy as np

from splatpipe.common.geometry import inv, quat_to_rot, rot_to_quat, se3
from splatpipe.common.trajectory import (
    Trajectory,
    imu_shaped,
    integrate_gyro,
    quat_conj,
    quat_mul,
    smooth_anchor_corrections,
)

DATASET = "oxford_spires"
SEQ = "2024-03-13-observatory-quarter-01"
SITE = "observatory-quarter"
LIDAR_ID = "hesai"
MIN_RANGE_M = 1.5  # opérateur (note 0.4) ; même seuil que GLIM

SCAN_KEYS = ("pt_xyz", "pt_intensity", "pt_t", "pt_ring", "pt_col", "pt_ret", "t_start", "t_end", "stamp_ns")


# =============================================================================
# Scène, tours, trajectoires
# =============================================================================


def scene_root(data_root: str = "/data", seq: str = SEQ) -> str:
    return f"{data_root}/scenes/{DATASET}/{seq}"


def list_scans(root: str) -> tuple[np.ndarray, list[str]]:
    d = f"{root}/lidar/{LIDAR_ID}"
    names = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
    stamps = np.array([int(f[:-4]) for f in names], dtype=np.int64)
    o = np.argsort(stamps)
    return stamps[o], [f"{d}/{names[i]}" for i in o]


def lidar_splits(root: str) -> dict[str, np.ndarray]:
    import yaml

    with open(f"{root}/splits.yaml") as f:
        s = yaml.safe_load(f)["lidars"]["hesai_qt64"]
    return {k: np.asarray(s.get(k) or [], dtype=np.int64) for k in ("train", "test_in", "test_out")}


def load_scan(path: str, keys=SCAN_KEYS) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in keys if k in z.files}


def sensors(root: str) -> dict:
    import yaml

    with open(f"{root}/sensors.yaml") as f:
        s = yaml.safe_load(f)
    return {
        "T_base_lidar": np.asarray(s["lidars"]["hesai_qt64"]["T_base_lidar"], dtype=np.float64),
        "T_base_imu": np.asarray(s["imu"]["T_base_imu"], dtype=np.float64),
    }


def _tum_rows_to_base(a: np.ndarray, T_x_base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lignes `t x y z qx qy qz qw` de T_frame_x → (t, pos, quat) de T_frame_base = T_frame_x · T_x_base."""
    a = np.atleast_2d(a)
    T = se3(a[:, 1:4], a[:, 4:8]) @ T_x_base
    t = a[:, 0]
    o = np.argsort(t, kind="stable")
    keep = np.r_[True, np.diff(t[o]) > 1e-7]
    o = o[keep]
    return t[o], T[o, :3, 3], rot_to_quat(T[o, :3, :3])


def load_glim_run(root: str, run: str, T_base_lidar: np.ndarray) -> dict:
    """Charge une exécution GLIM : ancres optimisées, odométrie et trajectoire IMU haute fréquence (toutes T_·_base).

    - `anchors` : `dump/traj_lidar.txt` (T_map_lidar, 10 Hz) → T_map_base ;
    - `odom`    : `dump/odom_lidar.txt` (T_odom_lidar) → T_odom_base ;
    - `imu_rate`: `dump/<sous-carte>/imu_rate.txt` (T_odom_imu à ~400 Hz, prédiction IMU de l'odométrie) → T_odom_base,
      doublons entre sous-cartes retirés ; T_lidar_imu = celui de l'exécution (`run_info.json`).
    """
    rdir = f"{root}/traj/glim/{run}"
    dump = f"{rdir}/dump"
    with open(f"{rdir}/run_info.json") as f:
        info = json.load(f)
    T_lidar_imu = np.asarray(info["config"]["extrinsics"]["T_lidar_imu"], dtype=np.float64)
    T_base_imu_glim = T_base_lidar @ T_lidar_imu
    T_lidar_base = inv(T_base_lidar)
    out = {"run": run, "T_lidar_imu": T_lidar_imu, "T_base_imu_glim": T_base_imu_glim}
    out["anchors"] = _tum_rows_to_base(np.loadtxt(f"{dump}/traj_lidar.txt"), T_lidar_base)
    out["odom"] = _tum_rows_to_base(np.loadtxt(f"{dump}/odom_lidar.txt"), T_lidar_base)
    rows = []
    for s in sorted(d for d in os.listdir(dump) if d.isdigit()):
        p = f"{dump}/{s}/imu_rate.txt"
        if os.path.exists(p) and os.path.getsize(p) > 0:
            rows.append(np.atleast_2d(np.loadtxt(p)))
    out["imu_rate"] = _tum_rows_to_base(np.concatenate(rows), inv(T_base_imu_glim))
    return out


def gyro_shape(root: str, T_base_imu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orientation du repère base intégrée du gyroscope (imu/imu.npz), repère fixe arbitraire → (t, quat)."""
    with np.load(f"{root}/imu/imu.npz") as z:
        t, g = z["t"], z["gyro"]
    return t, integrate_gyro(t, g, R_body_imu=T_base_imu[:3, :3])


def build_trajectory(
    anchors: tuple,
    method: str,
    frame: str,
    shape: dict | None = None,
    odom: tuple | None = None,
    sigma_s: float = 0.0,
    meta: dict | None = None,
) -> Trajectory:
    """Construit une trajectoire continue T_frame_base.

    `method` :
    - `slerp`        : interpolation linéaire + SLERP entre ancres ;
    - `gyro`         : rotation mise en forme par le gyroscope (`shape["gyro"] = (t, quat)`), translation linéaire ;
    - `glimimu`      : rotation et translation mises en forme par la trajectoire IMU de GLIM (`shape["glimimu"] = (t, pos, quat)`) ;
    `sigma_s` > 0 : ancres lissées au préalable via la correction carte←odométrie (`odom` requis).
    La variante « rigide » (pose du début de tour pour tout le tour) n'est pas une trajectoire : voir `deskew_scan(rigid=True)`.
    """
    at, ap, aq = anchors
    info = {"method": method, "sigma_s": float(sigma_s)}
    if sigma_s > 0:
        ot, op, oq = odom
        # Odométrie aux instants des ancres (mêmes trames GLIM ; interpolation par sécurité)
        from splatpipe.common.geometry import interpolate_poses

        op_a, oq_a, v = interpolate_poses(ot, op, oq, at)
        if not v.all():
            raise ValueError("odométrie ne couvrant pas les ancres")
        ap, aq, sinfo = smooth_anchor_corrections(at, ap, aq, op_a, oq_a, sigma_s)
        info["smoothing"] = sinfo
    if method == "slerp":
        tr = Trajectory(at, ap, aq, frame=frame)
    elif method == "gyro":
        st, sq = shape["gyro"]
        m = (at >= st[0]) & (at <= st[-1])
        s, p, q = imu_shaped(at[m], ap[m], aq[m], st, sq)
        tr = Trajectory(s, p, q, frame=frame)
    elif method == "glimimu":
        st, sp, sq = shape["glimimu"]
        m = (at >= st[0]) & (at <= st[-1])
        s, p, q = imu_shaped(at[m], ap[m], aq[m], st, sq, shape_pos=sp)
        tr = Trajectory(s, p, q, frame=frame)
    else:
        raise ValueError(method)
    tr.meta = {**(meta or {}), **info, "n_anchors": int(len(at))}
    return tr


# =============================================================================
# Correction de distorsion
# =============================================================================


def deskew_scan(
    scan: dict,
    traj: Trajectory | None,
    T_base_lidar: np.ndarray,
    min_range: float = MIN_RANGE_M,
    rigid_T_frame_base: np.ndarray | None = None,
) -> dict:
    """Points d'un tour dans le repère de la trajectoire.

    Par point : `p = T_frame_base(t_point) · T_base_lidar · p_lidar` ; ou, si `rigid_T_frame_base` est donné,
    la même pose pour tout le tour (« avant correction »). Retire les retours < `min_range` (opérateur).
    Renvoie un dict : `xyz` float64 [M,3], `keep` (masque sur les points du fichier), `n_near`, `n_invalid_time`,
    `origin` (position du capteur au milieu du tour), et les attributs filtrés (`intensity`, `t`, `ring`, `col`, `ret`).
    """
    xyz = scan["pt_xyz"].astype(np.float64)
    rng = np.linalg.norm(xyz, axis=1)
    near = rng < min_range
    t = scan["pt_t"]
    Rbl, tbl = T_base_lidar[:3, :3], T_base_lidar[:3, 3]
    pb = xyz @ Rbl.T + tbl
    if rigid_T_frame_base is not None:
        valid = np.ones(len(t), dtype=bool)
        pw = pb @ rigid_T_frame_base[:3, :3].T + rigid_T_frame_base[:3, 3]
        origin = rigid_T_frame_base[:3, :3] @ tbl + rigid_T_frame_base[:3, 3]
    else:
        pos, q, valid = traj.interp(t)
        R = quat_to_rot(q)
        pw = np.einsum("nij,nj->ni", R, pb) + pos
        tm = 0.5 * (float(scan["t_start"]) + float(scan["t_end"]))
        Rm, pm, _ = traj.rot_trans([tm])
        origin = Rm[0] @ tbl + pm[0]
    keep = valid & ~near
    return {
        "xyz": pw[keep],
        "keep": keep,
        "n_points": int(len(t)),
        "n_near": int(near.sum()),
        "n_invalid_time": int((~valid & ~near).sum()),
        "origin": origin,
        "intensity": scan["pt_intensity"][keep],
        "t": t[keep],
        "ring": scan["pt_ring"][keep],
        "col": scan["pt_col"][keep],
        "ret": scan["pt_ret"][keep],
    }


# =============================================================================
# Voxels
# =============================================================================

KEY_BITS = 21
KEY_OFF = 1 << (KEY_BITS - 1)
KEY_MASK = (1 << KEY_BITS) - 1


def pack_ijk(ijk: np.ndarray) -> np.ndarray:
    ijk = ijk.astype(np.int64) + KEY_OFF
    return (ijk[:, 0] << (2 * KEY_BITS)) | (ijk[:, 1] << KEY_BITS) | ijk[:, 2]


def unpack_keys(keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64)
    return np.stack([(keys >> (2 * KEY_BITS)) & KEY_MASK, (keys >> KEY_BITS) & KEY_MASK, keys & KEY_MASK], 1) - KEY_OFF


def group_sorted(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (ordre de tri, débuts de groupes dans l'ordre trié, clés uniques)."""
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    return order, starts, ks[starts]


def voxel_plane_stats(
    xyz: np.ndarray, voxel: float, t: np.ndarray | None = None, min_pts: int = 20, offset: float = 0.0
) -> dict[str, np.ndarray]:
    """Statistiques par voxel (ACP) : n, centroïde, écarts-types propres σ1 ≥ σ2 ≥ σ3, normale, étendue temporelle.

    σ3 = écart-type des distances point→plan ajusté (épaisseur). Seuls les voxels avec ≥ `min_pts` points sont rendus.
    Grille décalée de `offset` (m) sur les trois axes. `center_dist` = distance, le long de la normale, du centroïde au
    centre du voxel : un plan proche d'une face du voxel est tronqué (σ3 sous-estimé), voir `plane_patches`.
    """
    ijk = np.floor((xyz - offset) / voxel).astype(np.int64)
    keys = pack_ijk(ijk)
    order, starts, ukeys = group_sorted(keys)
    n = np.diff(np.r_[starts, len(keys)])
    sel = n >= min_pts
    # Coordonnées locales (précision) : point − coin du voxel
    q = xyz[order] - (ijk[order] * voxel + offset)
    S1 = np.add.reduceat(q, starts, axis=0)
    prods = np.stack([q[:, 0] * q[:, 0], q[:, 0] * q[:, 1], q[:, 0] * q[:, 2], q[:, 1] * q[:, 1], q[:, 1] * q[:, 2], q[:, 2] * q[:, 2]], 1)
    S2 = np.add.reduceat(prods, starts, axis=0)
    del prods, q
    n_s, S1, S2 = n[sel], S1[sel], S2[sel]
    mu = S1 / n_s[:, None]
    C = np.empty((len(n_s), 3, 3))
    idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    for c, (i, j) in enumerate(idx):
        C[:, i, j] = S2[:, c] / n_s - mu[:, i] * mu[:, j]
        C[:, j, i] = C[:, i, j]
    w, V = np.linalg.eigh(C)  # croissant
    w = np.clip(w, 0, None)
    normal = V[:, :, 0]
    out = {
        "keys": ukeys[sel],
        "n": n_s.astype(np.int32),
        "centroid": (mu + unpack_keys(ukeys[sel]) * voxel + offset).astype(np.float32),
        "sigma": np.sqrt(w[:, ::-1]).astype(np.float32),  # σ1, σ2, σ3
        "normal": normal.astype(np.float32),
        "center_dist": np.abs(np.sum((mu - 0.5 * voxel) * normal, axis=1)).astype(np.float32),
    }
    if t is not None:
        ts = t[order]
        out["tspan"] = (np.maximum.reduceat(ts, starts) - np.minimum.reduceat(ts, starts))[sel].astype(np.float32)
    return out


def plane_patches(
    xyz: np.ndarray, voxel: float, t: np.ndarray | None = None, min_pts: int = 20, sigma2_min: float | None = None,
    sigma3_max: float = 0.10, center_frac: float = 0.25,
) -> dict[str, np.ndarray]:
    """Patchs plans sur deux grilles décalées d'un demi-voxel ; on garde, dans chaque grille, les voxels plans dont le
    plan passe à moins de `center_frac · voxel` du centre (pas de troncature par les faces).

    Clé d'un patch = clé du voxel · 2 + indice de grille (0 ou 1), identique d'une carte à l'autre dans le même repère.
    Critères lâches (σ2 ≥ `sigma2_min`, défaut voxel/5 ; σ3 ≤ `sigma3_max`) pour ne pas favoriser les cartes floues.
    """
    sigma2_min = voxel / 5 if sigma2_min is None else sigma2_min
    parts = []
    for gid, off in enumerate((0.0, 0.5 * voxel)):
        st = voxel_plane_stats(xyz, voxel, t=t, min_pts=min_pts, offset=off)
        m = planar_mask(st, sigma2_min, sigma3_max) & (st["center_dist"] <= center_frac * voxel)
        st = {k: v[m] for k, v in st.items()}
        st["keys"] = st["keys"] * 2 + gid
        parts.append(st)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def two_grid_stats(
    xyz: np.ndarray, voxel: float, t: np.ndarray | None = None, min_pts: int = 30, sigma2_min: float | None = None,
    sigma3_max: float = 0.10, center_frac: float = 0.25,
) -> dict[str, np.ndarray]:
    """Comme `plane_patches` mais sans filtrer : tous les voxels (≥ `min_pts`) des deux grilles, avec le booléen
    `planar` (plan étendu, mince et centré). Sert à choisir des patchs communs à plusieurs cartes du même repère."""
    sigma2_min = voxel / 5 if sigma2_min is None else sigma2_min
    parts = []
    for gid, off in enumerate((0.0, 0.5 * voxel)):
        st = voxel_plane_stats(xyz, voxel, t=t, min_pts=min_pts, offset=off)
        st["planar"] = planar_mask(st, sigma2_min, sigma3_max) & (st["center_dist"] <= center_frac * voxel)
        st["keys"] = st["keys"] * 2 + gid
        parts.append(st)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out.pop("normal", None)
    return out


def common_patch_metrics(stats: dict[str, dict], min_planar_frac: float = 0.5, revisit_s: float = 30.0,
                         require_all: bool = True) -> dict:
    """Épaisseur σ3 sur des patchs communs à plusieurs cartes du même repère.

    Patchs retenus : présents (≥ min_pts) dans toutes les cartes (`require_all`) et plans dans au moins
    `min_planar_frac` des cartes. Sous-ensemble « revisités » : étendue temporelle médiane > `revisit_s`.
    Renvoie {nom: {"all": résumé σ3, "revisit": résumé σ3}, "n_common", "n_revisit", "n_union"} ; résumés en mètres.
    """
    names = list(stats)
    keys = np.unique(np.concatenate([stats[n]["keys"] for n in names]))
    R, K = len(names), len(keys)
    present = np.zeros((K, R), bool)
    planar = np.zeros((K, R), bool)
    s3 = np.full((K, R), np.nan, np.float32)
    ts = np.full((K, R), np.nan, np.float32)
    for r, n in enumerate(names):
        i = np.searchsorted(keys, stats[n]["keys"])
        present[i, r] = True
        planar[i, r] = stats[n]["planar"]
        s3[i, r] = stats[n]["sigma"][:, 2]
        if "tspan" in stats[n]:
            ts[i, r] = stats[n]["tspan"]
    cover = present.all(1) if require_all else present.sum(1) >= 2
    common = cover & (planar.sum(1) >= min_planar_frac * R)
    with np.errstate(all="ignore"):
        tsm = np.nanmedian(ts, axis=1)
    revisit = common & (tsm > revisit_s)
    out = {"n_union": int(K), "n_common": int(common.sum()), "n_revisit": int(revisit.sum()), "per_map": {}}
    for r, n in enumerate(names):
        out["per_map"][n] = {"all": summarize(s3[common, r]), "revisit": summarize(s3[revisit, r])}
    out["_common_keys"] = keys[common]
    out["_s3_common"] = s3[common]  # [n_common, n_cartes], ordre de `names`
    out["_names"] = names
    return out


def paired_median_diff(s3: np.ndarray, names: list[str], ref: str, n_boot: int = 300, seed: int = 0) -> dict:
    """Différence appariée par patch σ3(carte) − σ3(ref) : médiane et IC 95 % bootstrap (rééchantillonnage des patchs ;
    les patchs voisins étant corrélés, l'intervalle est optimiste)."""
    rng = np.random.default_rng(seed)
    r = names.index(ref)
    out = {}
    idx = rng.integers(0, len(s3), size=(n_boot, len(s3)))
    for c, n in enumerate(names):
        if n == ref:
            continue
        d = (s3[:, c] - s3[:, r]).astype(np.float64)
        boots = np.median(d[idx], axis=1)
        out[n] = {"median_diff_m": float(np.median(d)), "ci95_m": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                  "frac_patches_worse": float(np.mean(d > 0))}
    return out


def planar_mask(stats: dict, sigma2_min: float, sigma3_max: float) -> np.ndarray:
    """Voxels « plans » : étendus dans deux directions (σ2 ≥ seuil) et minces (σ3 ≤ seuil lâche)."""
    s = stats["sigma"]
    return (s[:, 1] >= sigma2_min) & (s[:, 2] <= sigma3_max)


def summarize(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
        "rms": float(np.sqrt(np.mean(x**2))),
    }


def o3d_voxel_down_sample(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """Réimplémentation de `open3d.geometry.PointCloud.voxel_down_sample` (centroïde par voxel).

    Open3D : `voxel_min_bound = min_bound − voxel/2`, indice = floor((p − voxel_min_bound)/voxel), moyenne des points.
    L'ordre de sortie diffère (Open3D : table de hachage) ; il n'intervient pas dans les métriques.
    """
    if len(xyz) == 0:
        return xyz
    lo = xyz.min(0) - 0.5 * voxel
    ijk = np.floor((xyz - lo) / voxel).astype(np.int64)
    if ijk.max() >= (1 << KEY_BITS):
        raise ValueError("emprise trop grande pour l'empaquetage des clés")
    keys = pack_ijk(ijk - KEY_OFF)
    order, starts, _ = group_sorted(keys)
    n = np.diff(np.r_[starts, len(keys)])
    return np.add.reduceat(xyz[order], starts, axis=0) / n[:, None]


def crop_aabb(xyz: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Masque des points dans la boîte [lo, hi] (bornes incluses, comme Open3D `crop`)."""
    return np.all((xyz >= lo) & (xyz <= hi), axis=1)


# =============================================================================
# Alignement
# =============================================================================


def umeyama_se3(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    T = np.eye(4)
    T[:3, :3] = U @ S @ Vt
    T[:3, 3] = mu_d - T[:3, :3] @ mu_s
    return T


def trajectory_alignment(est: tuple, gt: tuple, max_gap_s: float = 0.1) -> tuple[np.ndarray, dict]:
    """Alignement SE(3) (Umeyama, sans échelle) des positions estimées sur la GT interpolée (comme T1.1 « interp »)."""
    from splatpipe.common.geometry import interpolate_poses

    te, pe, _ = est
    tg, pg, qg = gt
    p_g, _, v = interpolate_poses(tg, pg, qg, te, max_gap_s=max_gap_s)
    T = umeyama_se3(pe[v], p_g[v])
    e = np.linalg.norm(pe[v] @ T[:3, :3].T + T[:3, 3] - p_g[v], axis=1)
    return T, {"n": int(v.sum()), "ate_rmse_m": float(np.sqrt(np.mean(e**2)))}


def knn_normals(points: np.ndarray, tree, k: int = 12, workers: int = -1, chunk: int = 2_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Normales par ACP des k plus proches voisins → (normales [N,3], σ3/σ1 [N])."""
    normals = np.empty((len(points), 3), dtype=np.float32)
    flat = np.empty(len(points), dtype=np.float32)
    for a in range(0, len(points), chunk):
        p = points[a : a + chunk]
        _, idx = tree.query(p, k=k, workers=workers)
        nb = tree.data[idx]
        nb = nb - nb.mean(1, keepdims=True)
        C = np.einsum("nki,nkj->nij", nb, nb) / k
        w, V = np.linalg.eigh(C)
        normals[a : a + chunk] = V[:, :, 0]
        flat[a : a + chunk] = np.sqrt(np.clip(w[:, 0], 0, None) / np.clip(w[:, 2], 1e-12, None))
    return normals, flat


def icp_point_to_plane(
    src: np.ndarray,
    tgt_tree,
    tgt_normals: np.ndarray,
    T_init: np.ndarray,
    schedule=(1.0, 0.5, 0.25, 0.1),
    iters: int = 15,
    workers: int = -1,
) -> tuple[np.ndarray, list]:
    """ICP point→plan (Gauss-Newton linéarisé, correspondances au plus proche sous un seuil décroissant)."""
    T = np.array(T_init, dtype=np.float64)
    hist = []
    for dmax in schedule:
        for it in range(iters):
            p = src @ T[:3, :3].T + T[:3, 3]
            d, idx = tgt_tree.query(p, k=1, distance_upper_bound=dmax, workers=workers)
            m = np.isfinite(d)
            if m.sum() < 1000:
                break
            ps, qt, nt = p[m], tgt_tree.data[idx[m]], tgt_normals[idx[m]].astype(np.float64)
            r = np.sum((ps - qt) * nt, axis=1)
            J = np.concatenate([np.cross(ps, nt), nt], axis=1)  # d r / d(ω, v) pour p' = p + ω×p + v
            # Poids de Huber (échelle robuste)
            s = max(1.4826 * np.median(np.abs(r)), 1e-3)
            w = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-12))
            H = (J * w[:, None]).T @ J
            g = (J * w[:, None]).T @ r
            x = -np.linalg.solve(H + 1e-9 * np.eye(6), g)
            dT = np.eye(4)
            wv, v = x[:3], x[3:]
            th = np.linalg.norm(wv)
            K = np.array([[0, -wv[2], wv[1]], [wv[2], 0, -wv[0]], [-wv[1], wv[0], 0]])
            dT[:3, :3] = np.eye(3) + (np.sin(th) / th if th > 1e-12 else 1.0) * K + ((1 - np.cos(th)) / th**2 if th > 1e-12 else 0.5) * K @ K
            dT[:3, 3] = v
            T = dT @ T
            hist.append({"dmax": dmax, "it": it, "n": int(m.sum()), "rms_p2plane": float(np.sqrt(np.mean(r**2))),
                         "median_abs": float(np.median(np.abs(r))), "step_rot_deg": float(np.degrees(th)), "step_trans_m": float(np.linalg.norm(v))})
            if th < 1e-6 and np.linalg.norm(v) < 1e-5:
                break
    return T, hist


# =============================================================================
# E/S nuages
# =============================================================================

_PCD_TYPES = {("F", 4): "f4", ("F", 8): "f8", ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4", ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4"}


def read_pcd(path: str, fields=("x", "y", "z")) -> tuple[dict, dict]:
    """Lit un PCD binaire (non compressé) → (colonnes demandées, en-tête)."""
    header = {}
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if not line or line.startswith("#"):
                continue
            k, _, v = line.partition(" ")
            header[k] = v
            if k == "DATA":
                break
        if header["DATA"] != "binary":
            raise ValueError(f"PCD {header['DATA']} non géré : {path}")
        names = header["FIELDS"].split()
        sizes = [int(s) for s in header["SIZE"].split()]
        types = header["TYPE"].split()
        counts = [int(c) for c in header.get("COUNT", " ".join("1" * len(names))).split()]
        dt = []
        for i, (nm, sz, ty, ct) in enumerate(zip(names, sizes, types, counts)):
            nm = nm if nm != "_" else f"_pad{i}"
            dt.append((nm, _PCD_TYPES[(ty, sz)], (ct,)) if ct > 1 else (nm, _PCD_TYPES[(ty, sz)]))
        n = int(header["POINTS"])
        arr = np.fromfile(f, dtype=np.dtype(dt), count=n)
    return {k: arr[k] for k in fields}, header


def pcd_viewpoint(header: dict) -> np.ndarray:
    """VIEWPOINT `tx ty tz qw qx qy qz` → matrice 4×4."""
    v = [float(x) for x in header.get("VIEWPOINT", "0 0 0 1 0 0 0").split()]
    return se3(v[:3], [v[4], v[5], v[6], v[3]])


def write_ply(path: str, cols: dict[str, np.ndarray], comments: list[str] | None = None) -> int:
    """PLY binaire little-endian, un élément `vertex` ; `cols` : nom → tableau 1D (types numpy simples)."""
    ply_t = {"f4": "float", "f8": "double", "u1": "uchar", "i1": "char", "u2": "ushort", "i2": "short", "u4": "uint", "i4": "int"}
    n = len(next(iter(cols.values())))
    dt = np.dtype([(k, v.dtype.str.lstrip("<|=")) for k, v in cols.items()])
    arr = np.empty(n, dtype=dt)
    for k, v in cols.items():
        arr[k] = v
    with open(path, "wb") as f:
        import unicodedata

        clean = [unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode("ascii") for c in (comments or [])]
        head = ["ply", "format binary_little_endian 1.0"] + [f"comment {c}" for c in clean]
        head += [f"element vertex {n}"] + [f"property {ply_t[dt[k].str.lstrip('<|=')]} {k}" for k in cols] + ["end_header"]
        f.write(("\n".join(head) + "\n").encode("ascii"))
        f.write(arr.tobytes())
    return os.path.getsize(path)


def spearman(a, b) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def _self_test() -> None:
    import tempfile

    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    # Plans : un sol z=0 bruité 2 cm → σ3 ≈ 0,02
    p = np.c_[rng.uniform(0, 4, 200000), rng.uniform(0, 4, 200000), rng.normal(0, 0.02, 200000)]
    st = voxel_plane_stats(p, 0.5, t=rng.uniform(0, 100, len(p)))
    assert np.median(st["sigma"][:, 2]) < 0.015  # plan sur une face de voxel : tronqué
    pp = plane_patches(p, 0.5, t=rng.uniform(0, 100, len(p)))
    assert abs(np.median(pp["sigma"][:, 2]) - 0.02) < 0.001, np.median(pp["sigma"][:, 2])
    assert np.all(pp["keys"] % 2 == 1) and np.all(np.abs(np.abs(pp["normal"][:, 2]) - 1) < 0.01)
    tg_a = two_grid_stats(p, 0.5, t=np.zeros(len(p)))
    p2 = p.copy()
    p2[:, 2] += rng.normal(0, 0.03, len(p))  # carte plus floue
    tg_b = two_grid_stats(p2, 0.5, t=np.zeros(len(p)))
    cm = common_patch_metrics({"a": tg_a, "b": tg_b})
    assert cm["n_common"] > 0 and cm["per_map"]["b"]["all"]["median"] > cm["per_map"]["a"]["all"]["median"] + 0.01
    # Sous-échantillonnage : 2 points dans le même voxel → centroïde
    q = np.array([[0.0, 0, 0], [0.004, 0.004, 0.004], [0.5, 0.5, 0.5]])
    d = o3d_voxel_down_sample(q, 0.01)
    assert len(d) == 2 and np.any(np.all(np.isclose(d, [0.002, 0.002, 0.002]), axis=1))
    # ICP : retrouver un petit déplacement sur une scène de 3 plans
    g = np.r_[np.c_[rng.uniform(0, 5, 30000), rng.uniform(0, 5, 30000), np.zeros(30000)],
              np.c_[np.zeros(30000), rng.uniform(0, 5, 30000), rng.uniform(0, 5, 30000)],
              np.c_[rng.uniform(0, 5, 30000), np.zeros(30000), rng.uniform(0, 5, 30000)]]
    tree = cKDTree(g)
    nrm, _ = knn_normals(g, tree, k=10)
    Tt = np.eye(4)
    c, s = np.cos(np.radians(1.0)), np.sin(np.radians(1.0))
    Tt[:2, :2] = [[c, -s], [s, c]]
    Tt[:3, 3] = [0.05, -0.03, 0.04]
    src = (g[::5] - Tt[:3, 3]) @ Tt[:3, :3]  # src = inv(Tt)·g
    T, hist = icp_point_to_plane(src, tree, nrm, np.eye(4), schedule=(0.5, 0.2, 0.05))
    assert np.allclose(T, Tt, atol=2e-3), T
    # PLY / PCD
    with tempfile.TemporaryDirectory() as dd:
        write_ply(f"{dd}/a.ply", {"x": np.ones(3, np.float32), "n": np.arange(3, dtype=np.uint32)})
        with open(f"{dd}/b.pcd", "wb") as f:
            f.write(b"VERSION 0.7\nFIELDS x y z _\nSIZE 4 4 4 1\nTYPE F F F U\nCOUNT 1 1 1 4\nWIDTH 2\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 2\nDATA binary\n")
            a = np.zeros(2, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("p", "u1", (4,))])
            a["x"] = [1, 2]
            f.write(a.tobytes())
        cols, h = read_pcd(f"{dd}/b.pcd")
        assert cols["x"].tolist() == [1, 2] and np.allclose(pcd_viewpoint(h), np.eye(4))
    assert abs(spearman([1, 2, 3, 4], [10, 20, 25, 40]) - 1) < 1e-12
    print("splatpipe.stage1.maptools : auto-test OK")


if __name__ == "__main__":
    _self_test()
