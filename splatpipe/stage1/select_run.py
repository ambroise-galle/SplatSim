"""Étape 1.4a/1.4b — Choix sans vérité terrain de l'exécution GLIM et de la trajectoire de correction de distorsion.

Critère de netteté (sans GT), fixé avant de regarder les résultats :
- carte rapide : 1 tour d'entraînement sur 5 (`splits.yaml`), retours < 1,5 m retirés, deux échos ;
- voxels de 0,5 m sur deux grilles décalées d'un demi-voxel ; épaisseur d'un patch = σ3 = √(plus petite valeur propre),
  c.-à-d. l'écart-type des distances point→plan ;
- patchs communs : présents (≥ 30 points) dans toutes les cartes comparées et plans (σ2 ≥ 10 cm, σ3 ≤ 10 cm, plan à
  moins de 12,5 cm du centre du voxel) dans au moins la moitié d'entre elles ;
- **score = médiane de σ3 sur les patchs communs** (plus bas = plus net) ; en complément : moyenne, p90, et la même
  médiane restreinte aux patchs revisités (étendue temporelle > 30 s), plus sensible aux fermetures de boucle.

1.4a : les 12 exécutions, trajectoire mise en forme par le gyroscope entre les poses GLIM (méthode `gyro`).
       Pour information seulement : corrélation de Spearman entre le score et l'ATE T1.1 (`reports/stage1_traj.json`).
1.4b : sur l'exécution retenue, variantes `rigid` (pose du début de tour pour tout le tour), `slerp`, `gyro`, `glimimu`,
       avec ou sans lissage des ancres (σ = 1 s, 3 s) → trajectoire finale = variante la plus nette. Écrit
       `traj/final.tum` (10 Hz, instants des tours), `traj/spline.npz` (≈ 400 Hz), `traj/final_to_world.yaml`
       (alignement SE(3) sur la GT façon T1.1, pour l'évaluation uniquement).

Lancement :
  .venv/bin/modal run splatpipe/stage1/select_run.py::runs_main
  .venv/bin/modal run splatpipe/stage1/select_run.py::traj_main [--run <exécution>]
"""

from __future__ import annotations

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume
from splatpipe.stage1.images import map_image

app = modal.App("splatsim-stage1-select-run")
vol = data_volume()

SEQ = "2024-03-13-observatory-quarter-01"
SUBSAMPLE = 5
VOXEL = 0.5
MIN_PTS = 30
REPORT_DIR = "reports/stage1_map"


def _variant_name(run: str, method: str, sigma: float, rigid: bool) -> str:
    if rigid:
        return f"{run}__rigid"
    return f"{run}__{method}" + (f"_s{sigma:g}" if sigma > 0 else "")


def make_glim_trajectory(root: str, run: str, method: str, sigma: float, sens: dict, glim: dict | None = None):
    from splatpipe.stage1.maptools import build_trajectory, gyro_shape, load_glim_run

    g = glim or load_glim_run(root, run, sens["T_base_lidar"])
    shape = {}
    if method == "gyro":
        shape["gyro"] = gyro_shape(root, sens["T_base_imu"])
    elif method == "glimimu":
        shape["glimimu"] = g["imu_rate"]
    tr = build_trajectory(g["anchors"], method, "map", shape=shape, odom=g["odom"], sigma_s=sigma,
                          meta={"glim_run": run, "T_frame": "T_map_base (repère carte GLIM)"})
    return tr, g


def accumulate(root: str, idx: list[int], paths: list[str], stamps, traj, sens: dict, rigid_anchors=None, threads: int = 8,
               T_out=None, keep_attrs: bool = False) -> dict:
    """Charge et corrige les tours `idx` → points concaténés (float64), temps, et comptes.

    `rigid_anchors` = (t, pos, quat) : pose de l'ancre du tour (à ±2 ms du début) appliquée à tout le tour.
    `T_out` (4×4) : transformation appliquée aux points en sortie (p. ex. map → world).
    """
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from splatpipe.common.geometry import se3
    from splatpipe.stage1.maptools import deskew_scan, load_scan

    def work(i):
        sc = load_scan(paths[i])
        rigid_T = None
        if rigid_anchors is not None:
            at, ap, aq = rigid_anchors
            j = int(np.argmin(np.abs(at - float(sc["t_start"]))))
            if abs(at[j] - float(sc["t_start"])) > 0.002:
                return None
            rigid_T = se3(ap[j], aq[j])
        d = deskew_scan(sc, traj, sens["T_base_lidar"], rigid_T_frame_base=rigid_T)
        if d["n_invalid_time"] > 0.02 * max(d["n_points"] - d["n_near"], 1):
            return None
        xyz = d["xyz"]
        if T_out is not None:
            xyz = xyz @ T_out[:3, :3].T + T_out[:3, 3]
        r = {"xyz": xyz, "t": d["t"], "n_near": d["n_near"], "n_points": d["n_points"], "origin": d["origin"]}
        if keep_attrs:
            r.update({k: d[k] for k in ("intensity", "ret")})
        return r

    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(work, idx))
    used = [r for r in res if r is not None]
    out = {
        "xyz": np.concatenate([r["xyz"] for r in used]),
        "t": np.concatenate([r["t"] for r in used]),
        "scan_id": np.concatenate([np.full(len(r["xyz"]), k, np.int32) for k, r in enumerate(used)]),
        "origins": np.array([r["origin"] for r in used]),
        "n_scans_requested": len(idx),
        "n_scans_used": len(used),
        "n_points_raw": int(sum(r["n_points"] for r in used)),
        "n_near_removed": int(sum(r["n_near"] for r in used)),
        "stamps_used": [int(stamps[i]) for i, r in zip(idx, res) if r is not None],
    }
    if T_out is not None:
        out["origins"] = out["origins"] @ T_out[:3, :3].T + T_out[:3, 3]
    if keep_attrs:
        out["intensity"] = np.concatenate([r["intensity"] for r in used])
        out["ret"] = np.concatenate([r["ret"] for r in used])
    return out


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=8, memory=48 * 1024, timeout=3600)
def variant_stats(run: str, method: str = "gyro", sigma: float = 0.0, rigid: bool = False, subsample: int = SUBSAMPLE,
                  seq: str = SEQ, tag: str = "runs") -> dict:
    import os
    import time

    import numpy as np

    from splatpipe.stage1.maptools import lidar_splits, list_scans, scene_root, sensors, two_grid_stats

    t0 = time.time()
    root = scene_root(DATA_MOUNT, seq)
    sens = sensors(root)
    stamps, paths = list_scans(root)
    train = set(lidar_splits(root)["train"].tolist())
    idx = [i for i, s in enumerate(stamps) if int(s) in train][::subsample]
    traj, g = make_glim_trajectory(root, run, "slerp" if rigid else method, sigma, sens)
    t_traj = time.time() - t0
    acc = accumulate(root, idx, paths, stamps, traj, sens, rigid_anchors=g["anchors"] if rigid else None)
    t_acc = time.time() - t0 - t_traj
    st = two_grid_stats(acc["xyz"], VOXEL, t=acc["t"], min_pts=MIN_PTS)
    name = _variant_name(run, method, sigma, rigid)
    out_dir = f"{root}/{REPORT_DIR}/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/{name}.npz", **st)
    vol.commit()
    planar = st["planar"]
    return {
        "name": name, "run": run, "method": "rigid" if rigid else method, "sigma_s": sigma,
        "n_scans_used": acc["n_scans_used"], "n_scans_requested": acc["n_scans_requested"],
        "n_points": int(len(acc["xyz"])), "n_voxels": int(len(planar)), "n_planar": int(planar.sum()),
        "own_planar_sigma3_median_m": float(np.median(st["sigma"][planar, 2])),
        "traj_meta": traj.meta, "traj_s": round(t_traj, 1), "accumulate_s": round(t_acc, 1),
        "total_s": round(time.time() - t0, 1),
    }


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=2, memory=16 * 1024, timeout=1800)
def aggregate(names: list[str], tag: str, seq: str = SEQ) -> dict:
    import numpy as np

    from splatpipe.stage1.maptools import common_patch_metrics, scene_root

    root = scene_root(DATA_MOUNT, seq)
    stats = {}
    for n in names:
        with np.load(f"{root}/{REPORT_DIR}/{tag}/{n}.npz") as z:
            stats[n] = {k: z[k] for k in ("keys", "sigma", "planar", "tspan")}
    cm = common_patch_metrics(stats)
    cm.pop("_common_keys")
    return cm


def _ate_by_run() -> dict:
    import json

    with open("logs/stage1/traj.json") as f:
        rep = json.load(f)
    out = {}
    for variant in ("calib", "imu_reest"):
        for k, v in rep["glim_repeatability"][variant]["runs"].items():
            out[k.removeprefix("glim_")] = v
    return out


@app.local_entrypoint()
def runs_main(seq: str = SEQ):
    import json
    import os

    import numpy as np

    from splatpipe.stage1.maptools import spearman

    runs = [f"{v}{s}" for v in ("calib", "imu_reest") for s in ("", "_r2", "_r3", "_r4", "_r5", "_r6")]
    os.makedirs("logs/stage1/map", exist_ok=True)
    res = list(variant_stats.map(runs, kwargs={"method": "gyro", "seq": seq, "tag": "runs"}))
    names = [r["name"] for r in res]
    cm = aggregate.remote(names, "runs", seq)
    ate = _ate_by_run()
    table = []
    for r in res:
        m = cm["per_map"][r["name"]]
        a = ate[r["run"]]
        table.append({
            "run": r["run"], "score_sigma3_median_cm": 100 * m["all"]["median"], "sigma3_mean_cm": 100 * m["all"]["mean"],
            "sigma3_p90_cm": 100 * m["all"]["p90"], "revisit_sigma3_median_cm": 100 * m["revisit"]["median"],
            "ate_bench_rmse_cm": a["ate_bench_rmse_cm"], "ate_rot_rmse_deg": a["ate_rot_rmse_deg"],
            "rpe10m_trans_rmse_cm": a["rpe10m_trans_rmse_cm"], "n_scans_used": r["n_scans_used"], "n_points": r["n_points"],
        })
    best = min(table, key=lambda x: x["score_sigma3_median_cm"])
    ates = [x["ate_bench_rmse_cm"] for x in table]
    corr = {}
    for key in ("score_sigma3_median_cm", "sigma3_mean_cm", "sigma3_p90_cm", "revisit_sigma3_median_cm"):
        vals = [x[key] for x in table]
        corr[key] = {"spearman_vs_ate": spearman(vals, ates), "spearman_vs_rpe10m": spearman(vals, [x["rpe10m_trans_rmse_cm"] for x in table])}
    ate_rank = sorted(ates).index(best["ate_bench_rmse_cm"]) + 1
    out = {"criterion": "médiane de σ3 sur patchs communs (0,5 m, deux grilles), 1 tour d'entraînement sur 5, méthode gyro",
           "selected_run": best["run"], "selected_ate_rank_among_12": ate_rank, "table": table, "correlations": corr,
           "common_patches": {k: cm[k] for k in ("n_union", "n_common", "n_revisit")}, "jobs": res,
           "ate_mean_cm": float(np.mean(ates)), "ate_best_cm": float(min(ates))}
    with open("logs/stage1/map/select_runs.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("jobs",)}, indent=1))


# =============================================================================
# 1.4b — variantes de trajectoire sur l'exécution retenue, puis écriture de la trajectoire finale
# =============================================================================


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=4, memory=16 * 1024, timeout=1800)
def write_final(run: str, method: str, sigma: float, selection: dict, seq: str = SEQ, suffix: str = "") -> dict:
    """Écrit traj/spline{suffix}.npz, traj/final{suffix}.tum, traj/final_to_world{suffix}.yaml.

    `suffix = ""` : chemins historiques (calib_r4) ; toute autre valeur (p. ex. "_calib", sous-étape 1.7a) écrit des
    fichiers parallèles sans toucher aux sorties existantes.
    """
    import os
    import time

    import numpy as np
    import yaml

    from splatpipe.common.geometry import interpolate_poses, read_tum, rot_to_quat, rotation_angle
    from splatpipe.common.trajectory import quat_conj, quat_log, quat_mul
    from splatpipe.stage1.maptools import list_scans, scene_root, sensors, trajectory_alignment

    root = scene_root(DATA_MOUNT, seq)
    sens = sensors(root)
    traj, g = make_glim_trajectory(root, run, method, sigma, sens)
    traj.meta.update({
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection": selection,
        "description": "T_map_base(t) ; échantillons denses, interpolation linéaire + SLERP (splatpipe.common.trajectory.Trajectory)",
    })
    os.makedirs(f"{root}/traj", exist_ok=True)
    traj.save(f"{root}/traj/spline{suffix}.npz")
    # final.tum : aux débuts de tours (t_start des npz) couverts par la trajectoire
    stamps, paths = list_scans(root)
    t_start = []
    for p in paths:
        with np.load(p) as z:
            t_start.append(float(z["t_start"]))
    t_start = np.array(t_start)
    inside = (t_start >= traj.t_start) & (t_start <= traj.t_end)
    header = (f"GLIM {run} + méthode {method} (σ lissage {sigma:g} s) : T_map_base aux instants t_start des tours,\n"
              f"map = repère carte de GLIM (pas world). Trajectoire continue : traj/spline{suffix}.npz. Étape 1.4b"
              + ("" if not suffix else f" (variante{suffix}, sous-étape 1.7a)."))
    traj.write_tum(f"{root}/traj/final{suffix}.tum", t_query=t_start[inside], header=header)
    # Alignement vers world (T1.1 « interp » : Umeyama SE(3) sur la GT interpolée) — pour l'évaluation uniquement
    gt = read_tum(f"{root}/traj/gt.tum")
    tq = t_start[inside]
    p, q, _ = traj.interp(tq)
    T_world_map, ainfo = trajectory_alignment((tq, p, q), gt)
    # Écarts de la trajectoire finale aux ancres GLIM brutes
    at, ap, aq = g["anchors"]
    pp, qq, v = traj.interp(at)
    dq = quat_mul(quat_conj(aq[v]), qq[v])
    dang = np.degrees(np.linalg.norm(quat_log(dq), axis=1))
    dpos = np.linalg.norm(pp[v] - ap[v], axis=1)
    doc = {
        "description": f"T_world_map : aligne le repère carte GLIM (traj/final{suffix}.tum, traj/spline{suffix}.npz, "
                       "lidar/hesai_deskewed" + ("" if not suffix else suffix) + ") sur world. "
                       "Estimé avec la GT (traj/gt.tum) : POUR L'ÉVALUATION UNIQUEMENT (T1.2 en repère world, T1.3).",
        "method": f"Umeyama SE(3) sans échelle, positions de traj/spline{suffix}.npz aux t_start des tours vs GT interpolée (≤ 0,1 s), comme T1.1 interp",
        "T_world_map": T_world_map.tolist(),
        "quat_xyzw": rot_to_quat(T_world_map[:3, :3]).tolist(),
        "yaw_deg": float(np.degrees(np.arctan2(T_world_map[1, 0], T_world_map[0, 0]))),
        "n_poses": ainfo["n"], "ate_rmse_m_after_alignment": ainfo["ate_rmse_m"],
        "icp_refined": None,
    }
    with open(f"{root}/traj/final_to_world{suffix}.yaml", "w") as f:
        f.write("# Alignement du repère carte GLIM sur world — POUR L'ÉVALUATION UNIQUEMENT (utilise la GT).\n")
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    vol.commit()
    return {
        "suffix": suffix, "run": run, "method": method, "sigma_s": sigma,
        "spline": {"path": f"traj/spline{suffix}.npz", "n": int(len(traj.t)), "t0": traj.t_start, "t1": traj.t_end,
                   "rate_hz": float((len(traj.t) - 1) / (traj.t_end - traj.t_start)),
                   "bytes": os.path.getsize(f"{root}/traj/spline{suffix}.npz")},
        "final_tum": {"path": f"traj/final{suffix}.tum", "n": int(inside.sum()), "n_scans": int(len(t_start)),
                      "bytes": os.path.getsize(f"{root}/traj/final{suffix}.tum")},
        "deviation_from_raw_anchors": {"rot_deg_median": float(np.median(dang)), "rot_deg_p95": float(np.percentile(dang, 95)),
                                       "rot_deg_max": float(dang.max()), "pos_m_median": float(np.median(dpos)),
                                       "pos_m_p95": float(np.percentile(dpos, 95)), "pos_m_max": float(dpos.max())},
        "final_to_world": doc,
        "traj_meta": {k: v for k, v in traj.meta.items() if k != "selection"},
    }


@app.local_entrypoint()
def traj_main(run: str = "", seq: str = SEQ, write: bool = True):
    import json

    if not run:
        with open("logs/stage1/map/select_runs.json") as f:
            run = json.load(f)["selected_run"]
    variants = [("slerp", 0.0, True), ("slerp", 0.0, False), ("gyro", 0.0, False), ("glimimu", 0.0, False),
                ("gyro", 1.0, False), ("gyro", 3.0, False), ("glimimu", 1.0, False), ("glimimu", 3.0, False),
                # ajoutés après le premier passage (optimum au bord de la grille σ ∈ {0, 1, 3} s) : critère toujours sans GT
                ("glimimu", 6.0, False), ("glimimu", 10.0, False)]
    res = list(variant_stats.starmap([(run, m, s, r) for m, s, r in variants], kwargs={"seq": seq, "tag": "traj"}))
    names = [r["name"] for r in res]
    cm = aggregate.remote(names, "traj", seq)
    table = []
    for r in res:
        m = cm["per_map"][r["name"]]
        table.append({"variant": r["name"], "method": r["method"], "sigma_s": r["sigma_s"],
                      "sigma3_median_cm": 100 * m["all"]["median"], "sigma3_mean_cm": 100 * m["all"]["mean"],
                      "sigma3_p90_cm": 100 * m["all"]["p90"], "revisit_sigma3_median_cm": 100 * m["revisit"]["median"],
                      "n_scans_used": r["n_scans_used"]})
    cand = [x for x in table if x["method"] != "rigid"]
    best = min(cand, key=lambda x: x["sigma3_median_cm"])
    sel = {"criterion": "médiane de σ3 sur patchs communs (0,5 m), 1 tour d'entraînement sur 5", "run": run,
           "method": best["method"], "sigma_s": best["sigma_s"], "table": table,
           "common_patches": {k: cm[k] for k in ("n_union", "n_common", "n_revisit")}}
    out = {"selection": sel, "jobs": res}
    if write:
        out["final"] = write_final.remote(run, best["method"], best["sigma_s"], {k: v for k, v in sel.items() if k != "table"}, seq)
    with open("logs/stage1/map/select_traj.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"selection": sel, "final": out.get("final")}, indent=1))


@app.local_entrypoint()
def debug_main(run: str = "calib_r4", method: str = "gyro", sigma: float = 0.0, rigid: bool = False, subsample: int = 25):
    import json

    print(json.dumps(variant_stats.remote(run, method, sigma, rigid, subsample, SEQ, "debug"), indent=1))


@app.local_entrypoint()
def final_main(run: str = "calib_r4", method: str = "glimimu", sigma: float = 3.0, seq: str = SEQ, reason: str = ""):
    """Écrit la trajectoire finale pour une variante donnée (après `traj_main --no-write`)."""
    import json

    with open("logs/stage1/map/select_traj.json") as f:
        prev = json.load(f)
    sel = {k: v for k, v in prev["selection"].items() if k != "table"}
    sel.update({"run": run, "method": method, "sigma_s": sigma, "decision_note": reason})
    prev["final"] = write_final.remote(run, method, sigma, sel, seq)
    prev["selection"]["final_choice"] = {"method": method, "sigma_s": sigma, "decision_note": reason}
    with open("logs/stage1/map/select_traj.json", "w") as f:
        json.dump(prev, f, indent=1)
    print(json.dumps(prev["final"], indent=1))


@app.local_entrypoint()
def final_variant_main(run: str, method: str = "glimimu", sigma: float = 3.0, suffix: str = "_calib",
                       seq: str = SEQ, reason: str = ""):
    """Sous-étape 1.7a — trajectoire d'une autre exécution GLIM dans des fichiers suffixés, sans toucher
    à `select_traj.json` ni aux sorties `calib_r4`. Journal : logs/stage1/map/write_final{suffix}.json."""
    import json
    import os

    sel = {"criterion": "essai 1.7a : exécution au meilleur ATE (et la plus nette en comparaison appariée)",
           "run": run, "method": method, "sigma_s": sigma, "suffix": suffix, "decision_note": reason}
    out = write_final.remote(run, method, sigma, sel, seq, suffix)
    os.makedirs("logs/stage1/map", exist_ok=True)
    with open(f"logs/stage1/map/write_final{suffix}.json", "w") as f:
        json.dump({"selection": sel, "final": out}, f, indent=1)
    print(json.dumps(out, indent=1, ensure_ascii=False))
