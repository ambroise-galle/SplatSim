"""Étape 1.3 — Test T1.1 : erreurs absolue (ATE) et relative (RPE) de trajectoire contre `traj/gt.tum`.

Trajectoires évaluées (toutes en convention `T_<repère propre>_base`, corps = base) :
- `glim`        : `traj/glim.tum` (GLIM, optimisation globale, variante retenue) ;
- `glim_odom`   : `traj/glim/<variante>/odom_base.tum` (odométrie GLIM seule, sans optimisation globale) ;
- autres variantes GLIM trouvées sous `traj/glim/<variante>/traj_base.tum` ;
- `vilens`, `hba` : baselines fournies (images clés à 1,25 Hz, repère local).

Deux protocoles :
1. **benchmark** (identique à `scripts/localisation_benchmark` d'Oxford Spires) :
   `evo_ape tum gt.tum est.tum --align --t_max_diff 0.01` → association au plus proche (≤ 10 ms),
   alignement SE(3) de Umeyama sans échelle, RMSE de translation. Calculé avec la bibliothèque evo.
2. **interp** (le nôtre) : la GT (non uniforme, 20 Hz + débuts de scans) est interpolée (linéaire + SLERP) à
   chaque instant estimé (intervalle encadrant ≤ 0,1 s), alignement SE(3) sans échelle (numpy), ATE translation
   (RMSE, moyenne, médiane, max) et rotation (angle de R_gtᵀ·R_align·R_est, en degrés) ; RPE sur des segments de
   1 m et 10 m de longueur parcourue (GT), toutes les origines, erreur de translation (m, %) et de rotation (°).
   Pour comparer à fréquence égale, GLIM est aussi interpolé aux instants des images clés VILENS.

Critère T1.1 (plan : « au niveau des baselines du benchmark ») : ATE RMSE (protocole benchmark) de GLIM
≤ max(1,2 × meilleure baseline, meilleure baseline + 1 cm), baselines = VILENS-SLAM et HBA fournis.
GLIM n'étant pas déterministe (threads asynchrones), le verdict porte sur la moyenne des exécutions répétées de la
variante publiée : PASS / FAIL, ou LIMITE si l'écart au seuil est inférieur à l'erreur type de la moyenne.

Lancement :
  .venv/bin/modal run splatpipe/stage1/eval_traj.py
Sorties : `<scène>/reports/stage1_traj.json`, `<scène>/reports/stage1_traj/*.png` (copies locales dans `logs/`).
"""

from __future__ import annotations

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume
from splatpipe.stage1.images import eval_image

app = modal.App("splatsim-stage1-eval-traj")
vol = data_volume()

DATASET = "oxford_spires"
SEQ = "2024-03-13-observatory-quarter-01"
GT_MAX_GAP_S = 0.1
BENCH_T_MAX_DIFF = 0.01
RPE_DISTANCES_M = (1.0, 10.0)
PASS_REL = 1.2
PASS_ABS_M = 0.01

# Couleurs (palette catégorielle de référence, mode clair) ; GT en encre neutre.
COLORS = {"gt": "#52514e", "glim": "#2a78d6", "vilens": "#eb6834", "hba": "#1baf7a", "glim_odom": "#4a3aa7"}
SURFACE = "#fcfcfb"


# =============================================================================
# Métriques (numpy)
# =============================================================================


def umeyama_se3(src, dst):
    """Rotation R et translation t minimisant Σ‖dst − (R·src + t)‖² (sans échelle)."""
    import numpy as np

    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    return R, mu_d - R @ mu_s


def _stats(x) -> dict:
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "rmse": float(np.sqrt(np.mean(x**2))),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def eval_interp(gt, est) -> dict:
    """ATE et RPE avec GT interpolée aux instants estimés. `gt`, `est` = (t, pos, quat_xyzw)."""
    import numpy as np

    from splatpipe.common.geometry import (
        interpolate_poses,
        quat_to_rot,
        rot_to_rotvec,
        rotation_angle,
        se3_from_rt,
    )

    t_gt, p_gt, q_gt = gt
    t_e, p_e, q_e = est
    p_g, q_g, valid = interpolate_poses(t_gt, p_gt, q_gt, t_e, max_gap_s=GT_MAX_GAP_S)
    t, p_est, R_est = t_e[valid], p_e[valid], quat_to_rot(q_e[valid])
    p_ref, R_ref = p_g[valid], quat_to_rot(q_g[valid])
    out = {"n_est": int(len(t_e)), "n_matched": int(valid.sum()), "t_span_s": float(t[-1] - t[0]) if len(t) else 0.0}
    if len(t) < 3:
        return out
    R_a, t_a = umeyama_se3(p_est, p_ref)
    p_al = p_est @ R_a.T + t_a
    R_al = R_a[None] @ R_est
    e_t = np.linalg.norm(p_al - p_ref, axis=1)
    e_r = np.degrees(rotation_angle(np.swapaxes(R_ref, 1, 2) @ R_al))
    out["align"] = {"R": R_a.tolist(), "t": t_a.tolist(), "yaw_deg": float(np.degrees(np.arctan2(R_a[1, 0], R_a[0, 0])))}
    out["ate_trans_m"] = _stats(e_t)
    out["ate_rot_deg"] = _stats(e_r)
    # Biais de rotation constant exprimé dans le repère corps (signature d'une extrinsèque ou d'un repère corps différent).
    rv = np.degrees(rot_to_rotvec(np.swapaxes(R_ref, 1, 2) @ R_al))
    bias = rv.mean(0)
    out["ate_rot_body_bias_deg"] = bias.tolist()
    out["ate_rot_deg_body_debiased"] = _stats(np.linalg.norm(rv - bias, axis=1))
    out["_series"] = {"t": t, "p_al": p_al, "p_ref": p_ref, "e_t": e_t, "e_r": e_r}

    # RPE par distance parcourue (GT), toutes les origines.
    T_ref = se3_from_rt(R_ref, p_ref)
    T_est = se3_from_rt(R_est, p_est)
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p_ref, axis=0), axis=1))])
    out["path_length_m"] = float(dist[-1])
    out["rpe"] = {}
    for d in RPE_DISTANCES_M:
        i = np.arange(len(t))
        j = np.searchsorted(dist, dist + d, side="left")
        ok = j < len(t)
        i, j = i[ok], j[ok]
        if len(i) == 0:
            out["rpe"][f"{d:g}m"] = {"n": 0}
            continue
        seg = dist[j] - dist[i]
        dQ = np.linalg.inv(T_ref[i]) @ T_ref[j]
        dP = np.linalg.inv(T_est[i]) @ T_est[j]
        E = np.linalg.inv(dQ) @ dP
        et = np.linalg.norm(E[:, :3, 3], axis=1)
        er = np.degrees(rotation_angle(E[:, :3, :3]))
        out["rpe"][f"{d:g}m"] = {
            "trans_m": _stats(et),
            "trans_pct": _stats(100.0 * et / seg),
            "rot_deg": _stats(er),
            "segment_length_m": _stats(seg),
            "dt_s": _stats(t[j] - t[i]),
        }
    return out


def time_offset_scan(gt, est) -> dict:
    """Décalage temporel δ (s) tel que la pose estimée datée t corresponde à la GT en t + δ : balayage de l'ATE
    (translation et rotation, alignement SE(3) refait à chaque δ). Diagnostic seulement (T1.1 reste à δ = 0)."""
    import numpy as np

    from splatpipe.common.geometry import interpolate_poses, quat_to_rot, rotation_angle

    t_gt, p_gt, q_gt = gt
    t_e, p_e, q_e = est

    def ate(delta):
        p_g, q_g, ok = interpolate_poses(t_gt, p_gt, q_gt, t_e + delta, max_gap_s=GT_MAX_GAP_S)
        R_a, t_a = umeyama_se3(p_e[ok], p_g[ok])
        e_t = np.linalg.norm(p_e[ok] @ R_a.T + t_a - p_g[ok], axis=1)
        R_al = R_a[None] @ quat_to_rot(q_e[ok])
        e_r = np.degrees(rotation_angle(np.swapaxes(quat_to_rot(q_g[ok]), 1, 2) @ R_al))
        return float(np.sqrt(np.mean(e_t**2))), float(np.sqrt(np.mean(e_r**2)))

    coarse = np.arange(-0.040, 0.0401, 0.002)
    vals = np.array([ate(d) for d in coarse])
    d0 = coarse[np.argmin(vals[:, 1])]
    fine = np.arange(d0 - 0.002, d0 + 0.00201, 0.00025)
    fvals = np.array([ate(d) for d in fine])
    k = int(np.argmin(fvals[:, 1]))
    z = ate(0.0)
    return {
        "best_delta_rot_ms": float(1000 * fine[k]),
        "at_best": {"ate_trans_rmse_m": float(fvals[k, 0]), "ate_rot_rmse_deg": float(fvals[k, 1])},
        "at_zero": {"ate_trans_rmse_m": z[0], "ate_rot_rmse_deg": z[1]},
        "best_delta_trans_ms": float(1000 * coarse[np.argmin(vals[:, 0])]),
        "coarse_grid_ms": (1000 * coarse).round(1).tolist(),
        "coarse_rot_rmse_deg": vals[:, 1].round(4).tolist(),
        "coarse_trans_rmse_m": vals[:, 0].round(5).tolist(),
    }


def eval_benchmark_evo(gt_path: str, est_path: str) -> dict:
    """Protocole du benchmark Oxford Spires avec evo (APE translation, --align, --t_max_diff 0.01)."""
    import copy

    from evo.core import metrics, sync
    from evo.tools import file_interface

    ref = file_interface.read_tum_trajectory_file(gt_path)
    est = file_interface.read_tum_trajectory_file(est_path)
    ref_a, est_a = sync.associate_trajectories(ref, est, max_diff=BENCH_T_MAX_DIFF)
    est_al = copy.deepcopy(est_a)
    est_al.align(ref_a, correct_scale=False)
    out = {"n_est": int(est.num_poses), "n_matched": int(est_a.num_poses)}
    for name, rel in (("ape_trans_m", metrics.PoseRelation.translation_part), ("ape_rot_deg", metrics.PoseRelation.rotation_angle_deg)):
        ape = metrics.APE(rel)
        ape.process_data((ref_a, est_al))
        s = ape.get_all_statistics()
        out[name] = {k: float(v) for k, v in s.items()}
    return out


def eval_rpe_evo(gt_path: str, est_path: str, delta_m: float) -> dict:
    """Contrôle croisé RPE (evo, delta en mètres, toutes les paires) après association ≤ 10 ms."""
    from evo.core import metrics, sync
    from evo.tools import file_interface

    ref = file_interface.read_tum_trajectory_file(gt_path)
    est = file_interface.read_tum_trajectory_file(est_path)
    ref_a, est_a = sync.associate_trajectories(ref, est, max_diff=BENCH_T_MAX_DIFF)
    out = {}
    for name, rel in (("trans_m", metrics.PoseRelation.translation_part), ("rot_deg", metrics.PoseRelation.rotation_angle_deg)):
        rpe = metrics.RPE(rel, delta=delta_m, delta_unit=metrics.Unit.meters, rel_delta_tol=0.1, all_pairs=True)
        try:
            rpe.process_data((ref_a, est_a))
            out[name] = {k: float(v) for k, v in rpe.get_all_statistics().items()}
        except Exception as e:  # pas assez de paires
            out[name] = {"error": str(e)}
    return out


# =============================================================================
# Figures
# =============================================================================


def plot_top_view(path: str, series: dict, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    gt_drawn = False
    for name, s in series.items():
        if not gt_drawn:
            ax.plot(s["p_ref"][:, 0], s["p_ref"][:, 1], color=COLORS["gt"], lw=3.0, alpha=0.35, label="GT (world)")
            gt_drawn = True
        ax.plot(s["p_al"][:, 0], s["p_al"][:, 1], color=COLORS.get(name, "#e34948"), lw=1.2, label=s["label"])
    ref0 = next(iter(series.values()))["p_ref"]
    ax.plot(ref0[0, 0], ref0[0, 1], "o", color=COLORS["gt"], ms=8, label="départ")
    ax.set_aspect("equal")
    ax.set_xlabel("x world (m)", color="#52514e")
    ax.set_ylabel("y world (m)", color="#52514e")
    ax.grid(color="#e6e5e0", lw=0.6)
    for sp in ax.spines.values():
        sp.set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.set_title(title, color="#0b0b0b", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_error_maps(path: str, series: dict) -> None:
    """Petits multiples : trajectoire GT colorée par l'ATE de translation de chaque méthode (échelle commune)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection

    names = list(series)
    vmax = max(float(np.percentile(series[n]["e_t"], 99)) for n in names) * 100.0
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 5.2), dpi=150, facecolor=SURFACE, squeeze=False)
    lc = None
    for ax, n in zip(axes[0], names):
        s = series[n]
        pts = s["p_ref"][:, :2]
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="Blues", norm=plt.Normalize(0.0, vmax), linewidths=2.0)
        lc.set_array(100.0 * 0.5 * (s["e_t"][:-1] + s["e_t"][1:]))
        ax.set_facecolor(SURFACE)
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title(f"{s['label']} — RMSE {100 * np.sqrt(np.mean(s['e_t'] ** 2)):.1f} cm", color="#0b0b0b", fontsize=10)
        ax.grid(color="#e6e5e0", lw=0.6)
        ax.tick_params(colors="#52514e", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#c3c2b7")
    cb = fig.colorbar(lc, ax=axes[0].tolist(), shrink=0.8, pad=0.01)
    cb.set_label("ATE translation (cm)", color="#52514e")
    fig.suptitle("Erreur absolue le long de la trajectoire (x/y world, m)", color="#0b0b0b", fontsize=11)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_errors(path: str, series: dict, t0: float) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), dpi=150, sharex=True, facecolor=SURFACE)
    for ax, key, ylabel in ((axes[0], "e_t", "ATE translation (cm)"), (axes[1], "e_r", "ATE rotation (°)")):
        ax.set_facecolor(SURFACE)
        for name, s in series.items():
            y = s[key] * (100.0 if key == "e_t" else 1.0)
            ax.plot(s["t"] - t0, y, color=COLORS.get(name, "#e34948"), lw=1.2, label=s["label"],
                    marker="o" if len(s["t"]) < 600 else None, ms=2.5)
        ax.set_ylabel(ylabel, color="#52514e")
        ax.grid(color="#e6e5e0", lw=0.6)
        ax.set_ylim(bottom=0)
        for sp in ax.spines.values():
            sp.set_color("#c3c2b7")
        ax.tick_params(colors="#52514e")
    axes[1].set_xlabel("temps depuis le début de la GT (s)", color="#52514e")
    axes[0].legend(frameon=False, fontsize=9, ncol=len(series))
    axes[0].set_title("Erreur absolue après alignement SE(3) (GT interpolée)", color="#0b0b0b", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# =============================================================================
# Fonction Modal
# =============================================================================


def _strip(d):
    """Retire les séries numpy (clé `_series`) pour la sérialisation JSON."""
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if k != "_series"}
    if isinstance(d, list):
        return [_strip(v) for v in d]
    return d


@app.function(image=eval_image(), volumes={DATA_MOUNT: vol}, cpu=4, memory=8192, timeout=1800)
def evaluate(seq: str = SEQ, main_variant: str | None = None) -> dict:
    import datetime
    import glob
    import json
    import os
    import shutil

    import numpy as np

    from splatpipe.common.geometry import read_tum, write_tum
    from splatpipe.common.scene import Scene

    vol.reload()
    sc = Scene(DATASET, seq)
    rep_dir = sc.path("reports", "stage1_traj")
    os.makedirs(rep_dir, exist_ok=True)
    tmp = "/tmp/traj"
    os.makedirs(tmp, exist_ok=True)

    gt_path = sc.traj_path("gt")
    gt = read_tum(gt_path)

    run_info = {}
    ri_path = sc.path("traj", "glim", "run_info.json")
    if os.path.exists(ri_path):
        with open(ri_path) as f:
            run_info = json.load(f)
    if main_variant is None:
        main_variant = run_info.get("selected_variant")
    main_run = run_info.get("selected_run", main_variant)

    cands = {"glim": (sc.traj_path("glim"), f"GLIM {main_run or ''} (global)".replace("  ", " ")), "vilens": (sc.traj_path("vilens"), "VILENS-SLAM"), "hba": (sc.traj_path("hba"), "HBA")}
    for vdir in sorted(glob.glob(sc.path("traj", "glim", "*", ""))):
        v = os.path.basename(os.path.dirname(vdir))
        if os.path.exists(os.path.join(vdir, "traj_base.tum")):
            cands[f"glim_{v}"] = (os.path.join(vdir, "traj_base.tum"), f"GLIM {v} (global)")
        if os.path.exists(os.path.join(vdir, "odom_base.tum")):
            cands[f"glim_{v}_odom"] = (os.path.join(vdir, "odom_base.tum"), f"GLIM {v} (odométrie seule)")
    if main_run and f"glim_{main_run}_odom" in cands:
        cands["glim_odom"] = cands[f"glim_{main_run}_odom"]

    results, series = {}, {}
    for name, (path, label) in cands.items():
        if not os.path.exists(path):
            results[name] = {"missing": path}
            continue
        # evo lit mal les commentaires en tête sur certaines versions : copie sans commentaires.
        clean = os.path.join(tmp, f"{name}.tum")
        with open(path) as fi, open(clean, "w") as fo:
            fo.writelines(l for l in fi if l.strip() and not l.startswith("#"))
        est = read_tum(path)
        r = {"path": path, "label": label, "n_poses": int(len(est[0])),
             "rate_hz": float((len(est[0]) - 1) / (est[0][-1] - est[0][0])) if len(est[0]) > 1 else None}
        r["interp"] = eval_interp(gt, est)
        r["time_offset_scan"] = time_offset_scan(gt, est)
        r["benchmark"] = eval_benchmark_evo(_clean_copy(gt_path, tmp), clean)
        r["rpe_evo_crosscheck"] = {f"{d:g}m": eval_rpe_evo(_clean_copy(gt_path, tmp), clean, d) for d in RPE_DISTANCES_M}
        results[name] = r
        series[name] = dict(r["interp"].get("_series", {}), label=label)

    # GLIM interpolé aux instants des images clés VILENS (comparaison à fréquence égale).
    if "glim" in results and "missing" not in results["glim"] and "vilens" in results and "missing" not in results["vilens"]:
        from splatpipe.common.geometry import interpolate_poses

        g = read_tum(sc.traj_path("glim"))
        tv = read_tum(sc.traj_path("vilens"))[0]
        p, q, ok = interpolate_poses(*g, tv, max_gap_s=0.25)
        sub_path = os.path.join(tmp, "glim_at_vilens.tum")
        write_tum(sub_path, tv[ok], p[ok], q[ok])
        r = {"label": "GLIM interpolé aux instants VILENS", "n_poses": int(ok.sum())}
        r["interp"] = eval_interp(gt, read_tum(sub_path))
        r["benchmark"] = eval_benchmark_evo(_clean_copy(gt_path, tmp), sub_path)
        results["glim_at_vilens_stamps"] = r

    # Répétabilité : GLIM n'est pas déterministe (threads asynchrones) → dispersion entre exécutions identiques.
    import re as _re

    repeat = {}
    for n, r in results.items():
        m = _re.fullmatch(r"glim_(calib|imu_reest)(_r\d+)?(_odom)?", n)
        if m and "benchmark" in r:
            key = m.group(1) + (m.group(3) or "")
            repeat.setdefault(key, {"runs": {}, "ate_bench_rmse_cm": []})
            v = 100 * r["benchmark"]["ape_trans_m"]["rmse"]
            repeat[key]["runs"][n] = {
                "ate_bench_rmse_cm": v,
                "ate_rot_rmse_deg": r["interp"]["ate_rot_deg"]["rmse"],
                "rpe1m_trans_rmse_cm": 100 * r["interp"]["rpe"]["1m"]["trans_m"]["rmse"],
                "rpe10m_trans_rmse_cm": 100 * r["interp"]["rpe"]["10m"]["trans_m"]["rmse"],
            }
            repeat[key]["ate_bench_rmse_cm"].append(v)
    for key, g in repeat.items():
        a = np.array(g.pop("ate_bench_rmse_cm"))
        g["ate_bench_rmse_cm_stats"] = {"n": int(len(a)), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else None,
                                        "min": float(a.min()), "max": float(a.max())}

    # Verdict T1.1
    def ate_bench(n):
        try:
            return results[n]["benchmark"]["ape_trans_m"]["rmse"]
        except (KeyError, TypeError):
            return None

    verdict = {"criterion": f"ATE RMSE benchmark GLIM ≤ max({PASS_REL} × meilleure baseline, meilleure baseline + {PASS_ABS_M * 100:.0f} cm)"}
    a_g, a_v, a_h = ate_bench("glim"), ate_bench("vilens"), ate_bench("hba")
    if a_g is not None and a_v is not None and a_h is not None:
        best = min(a_v, a_h)
        thr = max(PASS_REL * best, best + PASS_ABS_M)
        verdict.update({
            "glim_ate_rmse_cm": 100 * a_g, "vilens_ate_rmse_cm": 100 * a_v, "hba_ate_rmse_cm": 100 * a_h,
            "best_baseline": "vilens" if a_v <= a_h else "hba", "threshold_cm": 100 * thr,
            "glim_vs_best_ratio": a_g / best, "result_single_run": "PASS" if a_g <= thr else "FAIL",
        })
        # Verdict retenu : moyenne des exécutions répétées de la variante publiée (GLIM n'est pas déterministe).
        per_variant = {}
        for key, g in repeat.items():
            if key.endswith("_odom"):
                continue
            st = g["ate_bench_rmse_cm_stats"]
            vals = [x["ate_bench_rmse_cm"] for x in g["runs"].values()]
            sem = st["std"] / np.sqrt(st["n"]) if st.get("std") else 0.0
            margin = st["mean"] - 100 * thr
            res = "LIMITE" if abs(margin) < sem else ("PASS" if margin < 0 else "FAIL")
            per_variant[key] = dict(st, sem_cm=sem, margin_cm=margin, median=float(np.median(vals)),
                                    n_runs_pass=int(sum(v <= 100 * thr for v in vals)),
                                    n_runs_better_than_vilens=int(sum(v <= 100 * a_v for v in vals)),
                                    n_runs_better_than_hba=int(sum(v <= 100 * a_h for v in vals)),
                                    result=res)
        verdict["per_variant_repeats"] = per_variant
        if main_variant in per_variant:
            verdict["result"] = per_variant[main_variant]["result"]
            verdict["result_basis"] = (
                f"moyenne des {per_variant[main_variant]['n']} exécutions de la variante « {main_variant} » comparée au seuil ; "
                "LIMITE si |moyenne − seuil| < erreur type de la moyenne (écart non significatif)")
        else:
            verdict["result"] = verdict["result_single_run"]
            verdict["result_basis"] = "exécution publiée seule"
        # Comparaisons secondaires (information) : RPE 1 m et 10 m, protocole interp.
        sec = {}
        for d in RPE_DISTANCES_M:
            k = f"{d:g}m"
            vals = {}
            for n in ("glim", "glim_at_vilens_stamps", "vilens", "hba"):
                try:
                    vals[n] = {"trans_rmse_cm": 100 * results[n]["interp"]["rpe"][k]["trans_m"]["rmse"],
                               "rot_rmse_deg": results[n]["interp"]["rpe"][k]["rot_deg"]["rmse"]}
                except (KeyError, TypeError):
                    pass
            sec[k] = vals
        verdict["rpe_summary"] = sec
    else:
        verdict["result"] = "INCOMPLET"

    # Figures
    figs = {}
    main = {k: series[k] for k in ("glim", "vilens", "hba") if k in series and "p_al" in series[k]}
    if main:
        p = os.path.join(rep_dir, "top_view.png")
        plot_top_view(p, main, f"T1.1 — {seq} : trajectoires alignées SE(3) sur la GT (vue de dessus)")
        figs["top_view"] = p
        p = os.path.join(rep_dir, "ate_vs_time.png")
        plot_errors(p, main, float(gt[0][0]))
        figs["ate_vs_time"] = p
        p = os.path.join(rep_dir, "ate_maps.png")
        plot_error_maps(p, main)
        figs["ate_maps"] = p
    if "glim" in series and "glim_odom" in series and "p_al" in series.get("glim_odom", {}):
        p = os.path.join(rep_dir, "glim_global_vs_odom.png")
        plot_top_view(p, {"glim": series["glim"], "glim_odom": series["glim_odom"]}, "GLIM : optimisation globale vs odométrie seule")
        figs["glim_global_vs_odom"] = p

    report = {
        "stage": "stage1_traj",
        "test": "T1.1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "dataset": DATASET,
        "sequence": seq,
        "gt": {"path": gt_path, "n_poses": int(len(gt[0])), "t0": float(gt[0][0]), "t1": float(gt[0][-1])},
        "protocols": {
            "benchmark": f"evo APE translation, association au plus proche ≤ {BENCH_T_MAX_DIFF * 1000:.0f} ms, alignement SE(3) sans échelle (= evo_ape tum --align --t_max_diff 0.01, scripts Oxford Spires)",
            "interp": f"GT interpolée (linéaire + SLERP, intervalle ≤ {GT_MAX_GAP_S} s), Umeyama SE(3) sans échelle, ATE translation/rotation ; RPE toutes origines sur {list(RPE_DISTANCES_M)} m de longueur GT, sans alignement",
            "rpe_evo_crosscheck": "evo RPE delta en mètres (rel_delta_tol 0,1, all_pairs) après association ≤ 10 ms",
        },
        "glim_main_variant": main_variant,
        "glim_main_run": main_run,
        "glim_run": run_info,
        "glim_repeatability": repeat,
        "trajectories": _strip(results),
        "verdict": verdict,
        "figures": figs,
    }
    sc.write_report("stage1_traj", report)
    vol.commit()
    png = {os.path.basename(p): open(p, "rb").read() for p in figs.values()}
    return {"report": json.loads(json.dumps(report)), "png": png}


def _clean_copy(path: str, tmp: str) -> str:
    import os

    out = os.path.join(tmp, "gt_clean.tum")
    if not os.path.exists(out):
        with open(path) as fi, open(out, "w") as fo:
            fo.writelines(l for l in fi if l.strip() and not l.startswith("#"))
    return out


@app.local_entrypoint()
def main(seq: str = SEQ, main_variant: str = ""):
    import json
    import os

    res = evaluate.remote(seq, main_variant or None)
    os.makedirs("logs/stage1/traj", exist_ok=True)
    with open("logs/stage1/traj.json", "w") as f:
        json.dump(res["report"], f, indent=2, ensure_ascii=False)
    for name, data in res["png"].items():
        with open(os.path.join("logs/stage1/traj", name), "wb") as f:
            f.write(data)
    v = res["report"]["verdict"]
    print(json.dumps({k: g["ate_bench_rmse_cm_stats"] for k, g in res["report"]["glim_repeatability"].items()}, indent=1))
    print(json.dumps(v, indent=2, ensure_ascii=False))
    for n, r in res["report"]["trajectories"].items():
        if "interp" not in r:
            print(n, r)
            continue
        b = r.get("benchmark", {}).get("ape_trans_m", {})
        i = r["interp"]
        print(
            f"{n:28s} N={r.get('n_poses')} bench ATE rmse={100 * b.get('rmse', float('nan')):.2f} cm (n={r['benchmark'].get('n_matched')}) | "
            f"interp ATE rmse={100 * i['ate_trans_m']['rmse']:.2f} cm med={100 * i['ate_trans_m']['median']:.2f} max={100 * i['ate_trans_m']['max']:.2f} "
            f"rot rmse={i['ate_rot_deg']['rmse']:.3f}° | RPE1m={100 * i['rpe']['1m']['trans_m']['rmse']:.2f} cm "
            f"RPE10m={100 * i['rpe']['10m']['trans_m']['rmse']:.2f} cm"
        )
        if "time_offset_scan" in r:
            ts = r["time_offset_scan"]
            print(f"{'':28s} bias corps (°)={[round(v, 3) for v in i['ate_rot_body_bias_deg']]} rot débiaisée rmse="
                  f"{i['ate_rot_deg_body_debiased']['rmse']:.3f}° | δt* (rot)={ts['best_delta_rot_ms']:+.2f} ms → rot "
                  f"{ts['at_best']['ate_rot_rmse_deg']:.3f}° trans {100 * ts['at_best']['ate_trans_rmse_m']:.2f} cm")
