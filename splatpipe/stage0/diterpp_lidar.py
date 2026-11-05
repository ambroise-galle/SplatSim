"""DiTer++ — étape 0 LiDAR (Ouster OS1-64 du Go2) : modèle de faisceaux gsplat et contrôle du deskew.

Entrée : tranche extraite par `rosbag_stream.py` (`lidar/<stamp_ns>.npz`, nuage ORGANISÉ 64 × 1024 du pilote
ouster-ros, repère `os_sensor`, `t_rel` = temps de la colonne de mesure depuis le début du tour, `range` en m) et
trajectoire de référence (`*_gt.tum` : pose du repère des nuages dans la carte TLS, une par tour, au temps d'en-tête).

Faits établis sur les données (cf. etapes/notes/datasets/diterpp-stage0.md) :
- le nuage du pilote est « désdécalé » (destaggered) : une colonne du nuage mélange plusieurs instants. La colonne
  de TIR m (0..1023) se lit dans `t_rel` : 1024 valeurs distinctes par tour, toutes les lignes d'une colonne de
  tir partagent le même temps (l'Ouster tire ses 64 voies simultanément) → col gsplat = m, temps linéaire en m ;
- ligne gsplat = `ring` (0 = faisceau le plus haut) ;
- les rayons partent d'une origine située à `h` au-dessus de `os_sensor` (spécification Ouster : 36,18 mm entre
  repère capteur et repère lidar, plus le décalage radial de 15,8 mm des faisceaux) ; `h` est estimé en rendant
  l'élévation de chaque ligne indépendante de la distance.

  .venv/bin/python -m splatpipe.stage0.diterpp_lidar model  --dir data_local/diterpp/lawn
  .venv/bin/python -m splatpipe.stage0.diterpp_lidar deskew --dir data_local/diterpp/lawn
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

N_ROWS, N_COLS = 64, 1024
NAME = "ouster_os1_64"


def load_sweep(path: str) -> dict:
    """Points valides d'un tour : xyz (os_sensor), ligne, colonne de tir, temps relatif, distance pilote."""
    z = np.load(path)
    t = z["t_rel"].astype(np.float64)
    u, m = np.unique(t, return_inverse=True)
    if len(u) != N_COLS:
        raise ValueError(f"{path} : {len(u)} temps de colonne distincts (1024 attendus)")
    v = z["range"] > 0
    return dict(xyz=z["xyz"][v].astype(np.float64), row=z["ring"][v].astype(np.int64), col=m[v].astype(np.int64),
                t_rel=t[v], t_cols=u, rng=z["range"][v].astype(np.float64),
                intensity=z["intensity"][v].astype(np.float32), reflectivity=z["reflectivity"][v],
                n_file=int(len(t)))


def _circ_median(a):
    """Médiane circulaire approchée : médiane des écarts à la moyenne circulaire."""
    c = np.angle(np.exp(1j * a).mean())
    return c + np.median(np.angle(np.exp(1j * (a - c))))


def _wrap(a):
    return np.angle(np.exp(1j * a))


# ============================================================================= modèle
def estimate_model(files: list[str]):
    from splatpipe.common.lidar_model import CLOCKWISE, COUNTER_CLOCKWISE, StructuredSpinningLidarModel

    S = [load_sweep(f) for f in files]
    P = np.concatenate([s["xyz"] for s in S])
    R = np.concatenate([s["row"] for s in S])
    M = np.concatenate([s["col"] for s in S])
    rep = {"n_sweeps": len(S), "n_points": int(len(P))}

    # --- temps des colonnes : linéarité et période
    tc = np.stack([s["t_cols"] for s in S])                       # [n, 1024], s depuis le début du tour
    ar = np.arange(N_COLS)
    res_us = np.stack([(t - np.polyval(np.polyfit(ar, t, 1), ar)) * 1e6 for t in tc])   # par tour
    rep["colonnes"] = {"pas_median_us": float(np.median(np.diff(tc, axis=1)) * 1e6),
                       "duree_tour_ms": float(np.median(tc[:, -1] - tc[:, 0]) * 1e3),
                       "duree_tour_ms_min_max": [float((tc[:, -1] - tc[:, 0]).min() * 1e3),
                                                 float((tc[:, -1] - tc[:, 0]).max() * 1e3)],
                       "ecart_au_lineaire_us_p99": float(np.percentile(np.abs(res_us), 99)),
                       "t0_us_max": float(np.abs(tc[:, 0]).max() * 1e6)}

    # --- hauteur de l'origine des rayons : élévation par ligne indépendante de la distance
    def near_far(h):
        q = P - [0, 0, h]
        el = np.arctan2(q[:, 2], np.hypot(q[:, 0], q[:, 1]))
        rho = np.linalg.norm(q, axis=1)
        near, far = (rho > 1.5) & (rho < 4), rho > 10
        d = [np.median(el[(R == r) & near]) - np.median(el[(R == r) & far]) for r in range(N_ROWS)
             if ((R == r) & near).sum() > 200 and ((R == r) & far).sum() > 200]
        return float(np.sqrt(np.mean(np.degrees(np.array(d)) ** 2))), len(d)
    hs = np.arange(0.025, 0.045, 0.0005)
    rms = [near_far(h)[0] for h in hs]
    h = float(hs[int(np.argmin(rms))])
    rep["origine_rayons"] = {"h_m": h, "rms_proche_loin_deg": float(min(rms)), "rms_h0_deg": near_far(0.0)[0],
                             "n_lignes_test": near_far(h)[1], "spec_ouster_m": 0.03618}

    q = P - [0, 0, h]
    rho = np.linalg.norm(q, axis=1)
    el = np.arctan2(q[:, 2], np.hypot(q[:, 0], q[:, 1]))
    az = np.arctan2(q[:, 1], q[:, 0])
    # points « loin » par ligne : > 6 m si possible ; les lignes basses ne voient le sol qu'à 1–2 m (capteur à
    # ~0,5 m du sol), on descend alors à 2 m puis 1 m
    use = np.zeros(len(P), bool)
    seuil = np.zeros(N_ROWS)
    for r in range(N_ROWS):
        for th in (6.0, 2.0, 1.0):
            k = (R == r) & (rho > th)
            if k.sum() > 500:
                use |= k
                seuil[r] = th
                break
    row_el = np.array([np.median(el[use & (R == r)]) for r in range(N_ROWS)])
    n_far = np.array([(use & (R == r)).sum() for r in range(N_ROWS)])

    # --- sens de rotation et azimut des colonnes : az = a0 + s·m·Δ + offset[ligne]
    dlt = 2 * np.pi / N_COLS
    best = None
    for s in (-1, 1):
        r0 = _wrap(az[use] - s * M[use] * dlt)
        off = np.array([_circ_median(r0[R[use] == r]) for r in range(N_ROWS)])
        e = _wrap(r0 - off[R[use]])
        sc = float(np.median(np.abs(e)))
        if best is None or sc < best[0]:
            best = (sc, s, off)
    _, s, off = best
    a0 = _circ_median(off)
    off = _wrap(off - a0)
    col_nom = a0 + s * np.arange(N_COLS) * dlt
    # écart de chaque colonne au pas uniforme (codeur)
    r_col = _wrap(az[use] - off[R[use]] - col_nom[M[use]])
    col_dev = np.array([np.median(r_col[M[use] == m]) if (M[use] == m).sum() > 20 else 0.0 for m in range(N_COLS)])
    direction = COUNTER_CLOCKWISE if s > 0 else CLOCKWISE
    rep["azimut"] = {"sens": direction, "a0_deg": float(np.degrees(a0)),
                     "ecart_colonnes_au_pas_uniforme_deg_p99": float(np.degrees(np.percentile(np.abs(col_dev), 99))),
                     "offsets_lignes_deg_minmax": [float(np.degrees(off.min())), float(np.degrees(off.max()))]}

    freq = 1.0 / (rep["colonnes"]["pas_median_us"] * 1e-6 * N_COLS)
    model = StructuredSpinningLidarModel(
        name="Ouster OS1-64 (DiTer++ Go2), estimé sur les nuages",
        row_elevations_rad=row_el, column_azimuths_rad=col_nom, row_azimuth_offsets_rad=off,
        spinning_frequency_hz=10.0, spinning_direction=direction, max_returns=1,
        row_ring_ids=np.arange(N_ROWS), row_time_offsets_s=np.zeros(N_ROWS),
        notes={"origine": f"rayons depuis (0, 0, {h:.4f}) m dans os_sensor ; T_map_lidar = T_map_os_sensor · Tz(h)",
               "colonnes": "col = colonne de TIR (rang du temps t_rel), pas la colonne du nuage désdécalé",
               "frequence_mesuree_hz": freq,
               "elevations": "médiane par ligne des points à > 6 m vus depuis l'origine"})

    # --- contrôles : ré-affectation (ligne, colonne) depuis les angles, résidus angulaires
    el_res = np.degrees(el - row_el[R])
    az_res = np.degrees(_wrap(az - off[R] - col_nom[M]))
    row_pred = np.argmin(np.abs(el[:, None] - row_el[None]), axis=1)
    col_pred = np.round(_wrap(az - off[R] - col_nom[0]) / (s * dlt)).astype(np.int64) % N_COLS
    rep["controle"] = {}
    for lo, hi in ((0.0, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 20.0), (20.0, 1e9)):
        k = (rho >= lo) & (rho < hi)
        if k.sum() == 0:
            continue
        rep["controle"][f"{lo:g}-{hi:g} m"] = {
            "n": int(k.sum()),
            "ligne_ok": float((row_pred[k] == R[k]).mean()), "colonne_ok": float((col_pred[k] == M[k]).mean()),
            "res_elev_deg_p50_p99": [float(np.percentile(np.abs(el_res[k]), 50)), float(np.percentile(np.abs(el_res[k]), 99))],
            "res_azim_deg_p50_p99": [float(np.percentile(np.abs(az_res[k]), 50)), float(np.percentile(np.abs(az_res[k]), 99))]}
    rep["espacement_lignes_deg"] = {"min": float(np.degrees(np.abs(np.diff(row_el)).min())),
                                    "median": float(np.degrees(np.median(np.abs(np.diff(row_el)))))}
    rep["points_loin_par_ligne_min"] = int(n_far.min())
    rep["seuil_distance_par_ligne_m"] = {f"{th:g}": int((seuil == th).sum()) for th in (6.0, 2.0, 1.0)}
    # occultation proche (corps du robot) : fréquence des retours < 0,6 m par cellule (ligne, colonne de tir)
    near = np.zeros((N_ROWS, N_COLS))
    np.add.at(near, (R[rho < 0.6], M[rho < 0.6]), 1.0)
    rep["proche_0.6m"] = {"frac_points": float((rho < 0.6).mean()),
                          "cellules_occultees_50pct": int((near / len(S) > 0.5).sum())}
    return model, rep, near / len(S)


# ============================================================================= trajectoire et deskew
GT_DT_S = 0.0487     # la pose GT datée τ (en-tête = début du tour) est la pose à τ + 48,7 ms (≈ milieu du tour)
IMU_DT_S = 0.0       # (référence : le temps des IMU est celui du LiDAR une fois la GT recalée de GT_DT_S)


def gt_trajectory(path: str, dt: float = 0.0):
    """GT brute (une pose par tour). `dt` est ajouté aux temps (GT_DT_S = convention mesurée)."""
    from splatpipe.common.trajectory import Trajectory

    g = np.loadtxt(path)            # temps en notation scientifique (read_tum les refuse) ; précision ~0,2 µs
    return Trajectory(g[:, 0] + dt, g[:, 1:4], g[:, 4:8], frame="tls", body="os_sensor",
                      meta={"source": path, "dt_s": dt})


def fit_imu(gt, imu_t, gyro, lag: int = 5) -> dict:
    """Rotation os_sensor ← imu et décalage temporel, par moindres carrés sur les rotations relatives de la GT à
    `lag` poses d'écart. Le gyroscope est intégré une seule fois : son biais (~3e-4 rad/s mesuré, soit 0,01° sur
    0,5 s) est négligeable à cette échelle. Le résidu mesure le bruit des orientations GT."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as Rot

    from splatpipe.common.trajectory import _interp_quat_at, integrate_gyro

    sel = (gt.t > imu_t[0] + 1.5) & (gt.t < imu_t[-1] - 1.5)
    tg, Rg = gt.t[sel], Rot.from_quat(gt.quat[sel])
    rel_g = Rg[:-lag].inv() * Rg[lag:]
    q = integrate_gyro(imu_t, gyro)
    # orientation initiale : on teste les 4 retournements de 180° autour des axes (montage)
    flips = [np.diag(d).astype(float) for d in ([1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1])]

    def resid(x, F):
        Rsi = Rot.from_rotvec(x[:3]) * Rot.from_matrix(F)
        qi = Rot.from_quat(_interp_quat_at(imu_t + x[3], q, tg)) * Rsi.inv()
        return ((rel_g.inv() * (qi[:-lag].inv() * qi[lag:])).as_rotvec()).ravel()
    best = None
    for F in flips:
        r = least_squares(resid, np.zeros(4), args=(F,), x_scale=[0.01] * 4, diff_step=1e-3)
        if best is None or r.cost < best[0].cost:
            best = (r, F)
    r, F = best
    Rsi = (Rot.from_rotvec(r.x[:3]) * Rot.from_matrix(F)).as_matrix()
    e = r.fun.reshape(-1, 3)
    return {"R_sensor_imu": Rsi, "dt_s": float(r.x[3]), "bias": np.zeros(3),
            "residu_rot_rel_deg_rms": float(np.degrees(np.sqrt((e ** 2).sum(1).mean()))), "lag_poses": lag}


def build_trajectory(gt, imu_t, gyro, fit: dict, smooth_rot_s: float = 0.0, smooth_pos_s: float = 0.0):
    """Trajectoire continue (400 Hz) : passe par les poses GT (éventuellement lissées) et suit le gyroscope
    entre elles (orientation) ; translation linéaire entre ancres. Lissage : `smooth_anchor_corrections`
    avec l'orientation intégrée comme odométrie (σ en s) ; positions : noyau gaussien σ = smooth_pos_s."""
    from splatpipe.common.trajectory import (Trajectory, _interp_quat_at, imu_shaped, integrate_gyro,
                                             smooth_anchor_corrections)

    q_imu = integrate_gyro(imu_t + fit["dt_s"], gyro - fit["bias"], R_body_imu=fit["R_sensor_imu"])
    t_imu = imu_t + fit["dt_s"]
    inside = (gt.t >= t_imu[0]) & (gt.t <= t_imu[-1])
    at, ap, aq = gt.t[inside], gt.pos[inside].copy(), gt.quat[inside].copy()
    info = {}
    if smooth_rot_s > 0:
        O = _interp_quat_at(t_imu, q_imu, at)
        _, aq, info["rot"] = smooth_anchor_corrections(at, ap, aq, np.zeros_like(ap), O, smooth_rot_s)
    if smooth_pos_s > 0:
        ps = np.empty_like(ap)
        for i in range(len(at)):
            j = np.abs(at - at[i]) < 4 * smooth_pos_s
            w = np.exp(-0.5 * ((at[j] - at[i]) / smooth_pos_s) ** 2)
            ps[i] = (w[:, None] * ap[j]).sum(0) / w.sum()
        info["pos_change_cm_median"] = float(np.median(np.linalg.norm(ps - ap, axis=1)) * 100)
        ap = ps
    out_t = np.arange(at[0], at[-1], 0.0025)
    t, p, q = imu_shaped(at, ap, aq, t_imu, q_imu, out_t=out_t)
    return Trajectory(t, p, q, frame="tls", body="os_sensor",
                      meta={"methode": "imu_shaped", "smooth_rot_s": smooth_rot_s, "smooth_pos_s": smooth_pos_s,
                            "gt_dt_s": float(gt.meta.get("dt_s", 0.0)), **{k: v for k, v in info.items()}})


def deskew_points(sw: dict, stamp_ns: int, traj, dt: float = 0.0, mode: str = "interp"):
    """Points du tour dans le repère de la carte. `mode` : interp (pose au temps de chaque colonne) ou
    rigid (pose unique au temps d'en-tête). `dt` : décalage ajouté au temps des points avant interpolation."""
    t_abs = stamp_ns / 1e9 + (sw["t_rel"] if mode == "interp" else 0.0) + dt
    tq, inv = np.unique(np.atleast_1d(t_abs), return_inverse=True)
    T, ok = traj.matrices(tq)
    Tp = T[np.broadcast_to(inv, (len(sw["xyz"]),))]
    return np.einsum("nij,nj->ni", Tp[:, :3, :3], sw["xyz"]) + Tp[:, :3, 3]


def map_distances(tree, xyz_map, p, k: int = 16, chunk: int = 200_000):
    """Distance au plus proche point TLS, et distance au plan local (PCA des k voisins) là où le voisinage est
    plan (λmin/Σλ < 0,01, rayon < 15 cm) — métrique insensible à la densité de la carte et à l'herbe."""
    d1 = np.empty(len(p))
    dp = np.full(len(p), np.nan)
    nz = np.full(len(p), np.nan)
    for a in range(0, len(p), chunk):
        q = p[a:a + chunk]
        dd, ii = tree.query(q, k=k, workers=-1)
        d1[a:a + chunk] = dd[:, 0]
        nb = xyz_map[ii]
        mu = nb.mean(1)
        c = np.einsum("nki,nkj->nij", nb - mu[:, None], nb - mu[:, None]) / k
        lam, vec = np.linalg.eigh(c)
        planar = (lam[:, 0] / lam.sum(1).clip(1e-12) < 0.01) & (dd[:, -1] < 0.15)
        dist = np.abs(np.einsum("ni,ni->n", q - mu, vec[:, :, 0]))
        dp[a:a + chunk][planar] = dist[planar]
        nz[a:a + chunk] = np.abs(vec[:, 2, 0])
    return d1, dp, nz


def deskew_check(d: str, sl: str, gt_name: str, n: int = 24, radius: float = 40.0) -> dict:
    """Points deskewés selon plusieurs variantes de trajectoire, comparés à la carte TLS."""
    from scipy.spatial import cKDTree

    from splatpipe.common.pcd import read_pcd

    m = read_pcd(f"{d}/Lawn_GT.pcd")
    xyz_map = np.stack([m["x"], m["y"], m["z"]], 1).astype(np.float64)
    gt0 = gt_trajectory(f"{d}/{gt_name}")
    gt1 = gt_trajectory(f"{d}/{gt_name}", GT_DT_S)
    imu = np.load(f"{d}/{sl}/imu_body.npz")
    fit = fit_imu(gt1, imu["t"], imu["gyro"])
    print("imu", json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fit.items()}), flush=True)
    files = sorted(glob.glob(f"{d}/{sl}/lidar/*.npz"))
    files = files[len(files) // (n + 1)::len(files) // (n + 1)][:n]
    centre = gt0.interp(np.array([int(os.path.basename(f)[:-4]) / 1e9 for f in files]))[0].mean(0)
    keep = np.linalg.norm(xyz_map - centre, axis=1) < radius + 60
    xyz_map = xyz_map[keep]
    tree = cKDTree(xyz_map)
    variants = {
        "GT brute, pose d'en-tête pour tout le tour": (gt0, "rigid"),
        "GT brute interpolée, temps d'en-tête": (gt0, "interp"),
        "GT recalée +48,7 ms interpolée": (gt1, "interp"),
        "GT recalée + gyroscope entre poses": (build_trajectory(gt1, imu["t"], imu["gyro"], fit), "interp"),
        "idem + orientations lissées σ 0,5 s": (build_trajectory(gt1, imu["t"], imu["gyro"], fit, 0.5), "interp"),
        "idem + positions lissées σ 0,15 s": (build_trajectory(gt1, imu["t"], imu["gyro"], fit, 0.5, 0.15), "interp"),
    }
    sweeps = []
    for f in files:
        sw = load_sweep(f)
        rho = np.linalg.norm(sw["xyz"], axis=1)
        k = (rho > 1.0) & (rho < radius)
        sweeps.append(({kk: (v[k] if isinstance(v, np.ndarray) and len(v) == len(rho) else v)
                        for kk, v in sw.items()}, int(os.path.basename(f)[:-4])))
    out = {}
    for name, (traj, mode) in variants.items():
        p = np.concatenate([deskew_points(sw, st, traj, 0.0, mode) for sw, st in sweeps])
        d1, dp, nz = map_distances(tree, xyz_map, p)
        ok = np.isfinite(dp)
        out[name] = {"pt_mediane_cm": float(np.median(d1) * 100), "pt_p90_cm": float(np.percentile(d1, 90) * 100),
                     "pt_frac_5cm": float((d1 < 0.05).mean())}
        # murs (normale horizontale) : surfaces stables entre le scan TLS et le passage ; sol : pelouse (l'herbe
        # pousse, le laser y pénètre) — séparés pour isoler l'erreur de pose
        for lab, m in (("murs", ok & (nz < 0.3)), ("sol", ok & (nz > 0.9))):
            out[name][lab] = {"frac_points": float(m.mean()), "mediane_cm": float(np.median(dp[m]) * 100),
                              "p90_cm": float(np.percentile(dp[m], 90) * 100),
                              "frac_2cm": float((dp[m] < 0.02).mean())}
        print(name, json.dumps(out[name]), flush=True)
    fit_j = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fit.items()}
    return {"n_sweeps": len(files), "rayon_m": radius, "gt_dt_s": GT_DT_S, "imu": fit_j, "variantes": out}


def refine_poses(d: str, sl: str, gt_name: str, out_tum: str, every: int = 1, sub: int = 3, iters: int = 8,
                 radius: float = 25.0) -> dict:
    """Recalage de chaque tour sur les surfaces planes de la carte TLS (point → plan, 6 ddl, perte robuste),
    à partir de la trajectoire « GT recalée + gyroscope ». Le mouvement pendant le tour est celui de cette
    trajectoire (relatif à la pose du milieu du tour) ; seule la pose du milieu est corrigée. Les points d'un
    tour sur deux (indices pairs) servent à l'ajustement, les impairs au contrôle. Sortie : TUM des poses du
    milieu des tours (repère os_sensor), à utiliser comme ancres de la trajectoire."""
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation as Rot

    from splatpipe.common.pcd import read_pcd

    gt1 = gt_trajectory(f"{d}/{gt_name}", GT_DT_S)
    imu = np.load(f"{d}/{sl}/imu_body.npz")
    traj = build_trajectory(gt1, imu["t"], imu["gyro"], fit_imu(gt1, imu["t"], imu["gyro"]))
    m = read_pcd(f"{d}/Lawn_GT.pcd")
    tls = np.stack([m["x"], m["y"], m["z"]], 1).astype(np.float64)
    lo, hi = traj.pos.min(0) - radius - 5, traj.pos.max(0) + radius + 5
    tls = tls[np.all((tls >= lo) & (tls <= hi), axis=1)]
    tt = cKDTree(tls)
    files = sorted(glob.glob(f"{d}/{sl}/lidar/*.npz"))[::every]
    rows, stats = [], []
    for i, f in enumerate(files):
        st = int(os.path.basename(f)[:-4])
        sw = load_sweep(f)
        rho = np.linalg.norm(sw["xyz"], axis=1)
        k = np.flatnonzero((rho > 1.0) & (rho < radius))[::sub]
        t_mid = st / 1e9 + 0.5 * (sw["t_cols"][0] + sw["t_cols"][-1])
        if t_mid < traj.t_start + 0.1 or t_mid > traj.t_end - 0.1:
            continue
        Tm = traj.matrices(np.array([t_mid]))[0][0]
        Tp = traj.matrices(st / 1e9 + sw["t_rel"][k])[0]
        Trel = np.einsum("ij,njk->nik", np.linalg.inv(Tm), Tp)          # pose du point relative au milieu
        q = np.einsum("nij,nj->ni", Trel[:, :3, :3], sw["xyz"][k]) + Trel[:, :3, 3]   # points dans le repère milieu
        x = np.zeros(6)
        for _ in range(iters):                                         # ICP : appariements recalculés
            Rc = Rot.from_rotvec(x[:3]).as_matrix()
            pw = (q @ Rc.T + x[3:]) @ Tm[:3, :3].T + Tm[:3, 3]
            dd, ii = tt.query(pw, k=12, workers=-1)
            nb = tls[ii]
            mu = nb.mean(1)
            c = np.einsum("nki,nkj->nij", nb - mu[:, None], nb - mu[:, None]) / 12
            lam, vec = np.linalg.eigh(c)
            ok = (lam[:, 0] / lam.sum(1).clip(1e-12) < 0.01) & (dd[:, -1] < 0.15) & (dd[:, 0] < 0.3)
            fit_idx = ok & (np.arange(len(q)) % 2 == 0)
            if fit_idx.sum() < 100:
                break
            nn, mm_, qq = vec[fit_idx, :, 0], mu[fit_idx], q[fit_idx]

            def res(y):
                Ry = Rot.from_rotvec(y[:3]).as_matrix()
                pwy = (qq @ Ry.T + y[3:]) @ Tm[:3, :3].T + Tm[:3, 3]
                return ((pwy - mm_) * nn).sum(1)
            x = least_squares(res, x, loss="soft_l1", f_scale=0.03, x_scale=[0.005] * 3 + [0.02] * 3).x
        # contrôle sur les points impairs, avant / après
        Rc = Rot.from_rotvec(x[:3]).as_matrix()
        ctrl = ok & (np.arange(len(q)) % 2 == 1)

        def err(Rx, tx):
            pw = (q[ctrl] @ Rx.T + tx) @ Tm[:3, :3].T + Tm[:3, 3]
            return ((pw - mu[ctrl]) * vec[ctrl, :, 0]).sum(1)
        e0, e1 = err(np.eye(3), np.zeros(3)), err(Rc, x[3:])
        Tn = Tm.copy()
        Tn[:3, :3] = Tm[:3, :3] @ Rc
        Tn[:3, 3] = Tm[:3, :3] @ x[3:] + Tm[:3, 3]
        qn = Rot.from_matrix(Tn[:3, :3]).as_quat()
        rows.append([t_mid, *Tn[:3, 3], *qn])
        stats.append([np.degrees(np.linalg.norm(x[:3])), np.linalg.norm(x[3:]) * 100, x[5] * 100,
                      np.median(np.abs(e0)) * 100, np.median(np.abs(e1)) * 100, int(ctrl.sum())])
        if i % 100 == 0:
            print(f"[recalage] {i}/{len(files)} rot {stats[-1][0]:.3f}° trans {stats[-1][1]:.2f} cm "
                  f"contrôle {stats[-1][3]:.2f} → {stats[-1][4]:.2f} cm", flush=True)
    np.savetxt(out_tum, np.array(rows), fmt="%.9f")
    S = np.array(stats)
    rep = {"n_tours": len(rows), "correction_rot_deg_p50_p90": np.percentile(S[:, 0], [50, 90]).round(3).tolist(),
           "correction_trans_cm_p50_p90": np.percentile(S[:, 1], [50, 90]).round(2).tolist(),
           "correction_z_capteur_cm_mediane": float(np.median(S[:, 2])),
           "controle_plan_cm_avant_apres_mediane": [float(np.median(S[:, 3])), float(np.median(S[:, 4]))]}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["model", "deskew", "refine"])
    ap.add_argument("--dir", default="data_local/diterpp/lawn")
    ap.add_argument("--slice", default="upper_day_604_784")
    ap.add_argument("--gt", default="Lawn_upper_day_gt.tum")
    ap.add_argument("--out", default=None, help="dossier de sortie (défaut : <dir>/<slice>/stage0)")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    out = args.out or f"{args.dir}/{args.slice}/stage0"
    os.makedirs(out, exist_ok=True)
    if args.stage == "model":
        files = sorted(glob.glob(f"{args.dir}/{args.slice}/lidar/*.npz"))
        files = files[:: max(1, len(files) // args.n)][:args.n]
        model, rep, near = estimate_model(files)
        model.save_json(f"{out}/{NAME}.json")
        np.savez_compressed(f"{out}/{NAME}_near_range.npz", freq=near.astype(np.float32))
        json.dump(rep, open(f"{out}/{NAME}_estimation.json", "w"), indent=1, ensure_ascii=False)
        print(json.dumps(rep, indent=1, ensure_ascii=False))
    elif args.stage == "refine":
        rep = refine_poses(args.dir, args.slice, args.gt, f"{out}/poses_recalees_tls.tum")
        json.dump(rep, open(f"{out}/refine.json", "w"), indent=1, ensure_ascii=False)
        print(json.dumps(rep, indent=1, ensure_ascii=False))
    else:
        rep = deskew_check(args.dir, args.slice, args.gt)
        json.dump(rep, open(f"{out}/deskew_check.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
