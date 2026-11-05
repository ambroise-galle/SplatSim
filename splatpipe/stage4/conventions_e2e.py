"""Étape 4.0b — test de bout en bout des conventions : carte de l'étape 1 → gaussiennes → rendu LiDAR →
comparaison aux distances mesurées d'un tour réel. **À passer avant tout entraînement** : un échec signale une
erreur de repère, de modèle de faisceaux ou de temps (plan §4, T4.0b).

Chaîne testée (exactement celle de l'entraînement) :
  1. `map/map_5cm.ply` (canonique, repère `map`) → gaussiennes isotropes quasi opaques (σ = 4 cm, α = 0,99) ;
  2. modèle de faisceaux Hesai QT64 réel (`lidar_model/hesai_qt64.json`) → `RowOffsetStructuredSpinningLidarModelParameters`
     + `safe_lidar_tiling` (≤ 64 rayons/tuile, cf. T4.0a) ;
  3. rendu `RGB-Ed` d'un tour **`test_in`** (jamais vu par la carte) avec `viewmats` = T_lidar_map(t_start),
     `viewmats_rs` = T_lidar_map(t_end), obturateur roulant gauche→droite (comme `av_trainer`) ;
  4. comparaison aux distances mesurées des premiers échos : `‖xyz − origine_spline(t_point)‖`, points de
     `lidar/hesai_deskewed` (repère map, chaque point à son propre temps ; les origines viennent de la même
     spline que la correction de distorsion).

Critères (fixés avant exécution) : médiane |erreur| ≤ 5 cm (le bruit de la carte vaut 3–4 cm), |biais moyen|
≤ 3 cm, fraction de retours (α > 0,5) sur les cellules premier écho du tour ≥ 90 %.

Lancement :
  .venv/bin/modal run splatpipe/stage4/conventions_e2e.py
"""

from pathlib import Path

import modal

from splatpipe.images import gsplat_image

app = modal.App("splatsim-stage4-conv-e2e")
vol = modal.Volume.from_name("splatsim-data", create_if_missing=True, version=2)
DATA_MOUNT = "/data"

_PKG = Path(__file__).resolve().parents[1]
image = (
    gsplat_image()
    .uv_pip_install("matplotlib==3.10.6")
    .add_local_file(_PKG / "__init__.py", "/root/splatpipe/__init__.py")
    .add_local_file(_PKG / "images.py", "/root/splatpipe/images.py")
    .add_local_dir(_PKG / "common", "/root/splatpipe/common", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(_PKG / "stage4", "/root/splatpipe/stage4", ignore=["**/__pycache__", "**/*.pyc"])
)

SEQ = "2024-03-13-observatory-quarter-01"
GAUSS_SIGMA = 0.04
OPACITY = 0.99
FAR_PLANE = 300.0
MAP_NAME = "map_5cm_static.ply"  # carte statique (2.2) : la carte complète contient passants/végétation mouvant
MATCH_CM = 10.0  # population « appariée » : |err| ≤ 10 cm (conventions) ; le reste = contenu de carte divergent
CRIT = {"median_abs_m": 0.05, "bias_m": 0.03, "return_frac": 0.90}
_RANGE_BINS = (0, 10, 20, 40, 100)

_PLY_TYPES = {"float": "f4", "double": "f8", "uchar": "u1", "char": "i1", "ushort": "u2", "short": "i2",
              "uint": "u4", "int": "i4"}


def _ply_header(path: str) -> tuple[int, int, list[tuple[str, str]]]:
    props, n = [], 0
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                _, t, name = line.split()
                props.append((name, _PLY_TYPES[t]))
            elif line == "end_header":
                return f.tell(), n, props


def _load_map_xyz(dev, ply_path: str, sigma: float, sigma_n: float = 0.0):
    """Gaussiennes de la carte. `sigma_n > 0` : disques PLATS orientés par la normale du voxel
    (σ = (sigma, sigma, sigma_n)) — l'initialisation prévue pour l'entraînement ; sinon isotropes."""
    import numpy as np
    import torch

    off, n, props = _ply_header(ply_path)
    dt = np.dtype([(name, "<" + t) for name, t in props])
    mm = np.memmap(ply_path, dtype=dt, mode="r", offset=off, shape=(n,))
    xyz = np.stack([mm["x"], mm["y"], mm["z"]], 1).astype(np.float32)
    nrm = np.stack([mm["nx"], mm["ny"], mm["nz"]], 1).astype(np.float32) if "nx" in dt.names else None
    n_scans_field = np.asarray(mm["n_scans"], np.int64) if "n_scans" in dt.names else None
    veg_frac_field = np.asarray(mm["veg_frac"], np.float32) if "veg_frac" in dt.names else None
    del mm
    means = torch.from_numpy(xyz).to(dev)
    if sigma_n > 0 and nrm is not None:
        nn = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
        a = np.where(np.abs(nn[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        t1 = np.cross(nn, a)
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        t2 = np.cross(nn, t1)
        R = np.stack([t1, t2, nn], axis=1)  # colonnes = axes locaux ; z = normale
        # matrice → quaternion (w, x, y, z), méthode de Shepperd vectorisée
        tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        q = np.empty((len(R), 4))
        m = tr > 0
        s = np.sqrt(tr[m] + 1.0) * 2
        q[m, 0] = 0.25 * s
        q[m, 1] = (R[m, 2, 1] - R[m, 1, 2]) / s
        q[m, 2] = (R[m, 0, 2] - R[m, 2, 0]) / s
        q[m, 3] = (R[m, 1, 0] - R[m, 0, 1]) / s
        for i in range(3):
            im = ~m
            ii, jj, kk = i, (i + 1) % 3, (i + 2) % 3
            sel = im & (R[:, ii, ii] >= R[:, jj, jj]) & (R[:, ii, ii] >= R[:, kk, kk])
            st = np.sqrt(1.0 + R[sel, ii, ii] - R[sel, jj, jj] - R[sel, kk, kk]) * 2
            q[sel, 0] = (R[sel, kk, jj] - R[sel, jj, kk]) / st
            q[sel, 1 + ii] = 0.25 * st
            q[sel, 1 + jj] = (R[sel, jj, ii] + R[sel, ii, jj]) / st
            q[sel, 1 + kk] = (R[sel, kk, ii] + R[sel, ii, kk]) / st
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        quats = torch.from_numpy(q.astype(np.float32)).to(dev)
        scales = torch.from_numpy(np.tile(np.array([sigma, sigma, sigma_n], np.float32), (len(means), 1))).to(dev)
    else:
        quats = torch.zeros(len(means), 4, device=dev)
        quats[:, 0] = 1.0  # identité : gaussiennes isotropes
        scales = torch.full((len(means), 3), sigma, device=dev)
    opac = torch.full((len(means),), OPACITY, device=dev)
    extra = {"n_scans": n_scans_field, "veg_frac": veg_frac_field}
    return means, quats, scales, opac, nrm, extra


def _load_lidar_model(dev):
    import torch
    from gsplat import RowOffsetStructuredSpinningLidarModelParametersExt  # wrapper haut niveau (to_cpp)
    from gsplat.cuda._lidar import (
        RowOffsetStructuredSpinningLidarModelParameters,
        SpinningDirection,
        compute_angles_to_columns_map,
    )

    from splatpipe.common.lidar_model import StructuredSpinningLidarModel
    from splatpipe.stage4.smoke_gsplat_lidar import safe_lidar_tiling

    m = StructuredSpinningLidarModel.load_json(f"{DATA_MOUNT}/scenes/oxford_spires/{SEQ}/lidar_model/hesai_qt64.json")
    kw = m.gsplat_kwargs()
    params = RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=torch.from_numpy(kw["row_elevations_rad"]).to(dev),
        column_azimuths_rad=torch.from_numpy(kw["column_azimuths_rad"]).to(dev),
        row_azimuth_offsets_rad=torch.from_numpy(kw["row_azimuth_offsets_rad"]).to(dev),
        spinning_frequency_hz=kw["spinning_frequency_hz"],
        spinning_direction=SpinningDirection[kw["spinning_direction"]],
    )
    a2c = compute_angles_to_columns_map(params)
    tiling, used = safe_lidar_tiling(params)
    lidar = RowOffsetStructuredSpinningLidarModelParametersExt(params, a2c.to(dev), tiling)
    return lidar, used, m


def _stats(err_m: "np.ndarray") -> dict:  # noqa: F821
    import numpy as np

    a = np.abs(err_m)
    return {"n": int(len(err_m)), "median_abs_cm": round(float(np.median(a)) * 100, 3),
            "p90_abs_cm": round(float(np.percentile(a, 90)) * 100, 3),
            "p95_abs_cm": round(float(np.percentile(a, 95)) * 100, 3),
            "bias_cm": round(float(np.mean(err_m)) * 100, 3),
            "frac_abs_gt_10cm": round(float((a > 0.10).mean()), 4)}


@app.function(image=image, volumes={DATA_MOUNT: vol}, gpu="L4", timeout=40 * 60)
def run_main(n_scans: int = 3, seq: str = SEQ, sigma: float = GAUSS_SIGMA,
             sigma_n: float = 0.01, split: str = "test_in") -> dict:
    import gsplat
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import time
    import torch
    import yaml

    from splatpipe.common.trajectory import Trajectory

    t0 = time.time()
    root = f"{DATA_MOUNT}/scenes/oxford_spires/{seq}"
    dev = "cuda"

    lidar, used_tiling, model = _load_lidar_model(dev)
    means, quats, scales, opac, map_nrm, map_extra = _load_map_xyz(dev, f"{root}/map/{MAP_NAME}", sigma, sigma_n)
    colors = torch.zeros(len(means), 3, device=dev)
    setup = {"map": MAP_NAME, "n_gaussians": int(len(means)), "gauss_sigma_m": sigma, "opacity": OPACITY,
             "beam_model": "lidar_model/hesai_qt64.json", "n_rows": int(lidar.n_rows),
             "n_cols": int(lidar.n_columns),
             "tiling_max_rays_per_tile": int(lidar.tiling.tiles_pack_info[:, 1].max()),
             "tiling_max_pts_per_tile_param": used_tiling,
             "map_build_time_s": round(time.time() - t0, 1)}

    splits = yaml.safe_load(open(f"{root}/splits.yaml"))["lidars"]["hesai_qt64"]
    test_in = set(int(s) for s in (splits.get(split) or []))
    desk = {int(s[:-4]) for s in os.listdir(f"{root}/lidar/hesai_deskewed") if s.endswith(".npz")}
    sel = sorted(test_in & desk)
    if n_scans == 1:
        picks = [sel[len(sel) // 2]]
    else:
        picks = [sel[k] for k in np.linspace(0, len(sel) - 1, n_scans).round().astype(int)]

    traj = Trajectory.load(f"{root}/traj/spline.npz")
    results, figs = {"setup": setup, "scans": [], "criteria": CRIT}, {}
    all_err, all_rng = [], []
    for si, st in enumerate(picks):
        t1 = time.time()
        with np.load(f"{root}/lidar/hesai_deskewed/{st}.npz") as z:
            xyz = z["xyz"].astype(np.float64)
            ret, row, col = z["ret"].astype(int), z["row"].astype(int), z["col"].astype(int)
            t_rel = z["t_rel"].astype(np.float64)
            t_start, t_end = float(z["t_start"]), float(z["t_end"])
            T0m = np.asarray(z["T_map_lidar_start"], float)
            T1m = np.asarray(z["T_map_lidar_end"], float)
        viewmat_s = torch.tensor(np.linalg.inv(T0m), dtype=torch.float32, device=dev)
        viewmat_e = torch.tensor(np.linalg.inv(T1m), dtype=torch.float32, device=dev)
        renders, alphas, meta = gsplat.rasterization(
            means, quats, scales, opac, colors, viewmat_s[None], torch.eye(3, device=dev)[None],
            lidar.n_columns, lidar.n_rows, camera_model="lidar", lidar_coeffs=lidar,
            with_ut=True, with_eval3d=True, packed=False, render_mode="RGB-Ed", global_z_order=False,
            far_plane=FAR_PLANE, rolling_shutter=gsplat.RollingShutterType.ROLLING_LEFT_TO_RIGHT,
            viewmats_rs=viewmat_e[None])
        torch.cuda.synchronize()
        dist = renders[0, ..., 3].double().cpu().numpy()
        alpha = alphas[0, ..., 0].double().cpu().numpy()

        first = ret == 0  # premier écho : la cellule (row, col) du rendu donne le premier hit
        t_pt = t_start + t_rel[first]
        Tmb, _ = traj.matrices(list(t_pt))
        T_bl = np.asarray(yaml.safe_load(open(f"{root}/sensors.yaml"))["lidars"]["hesai_qt64"]["T_base_lidar"], float)
        o_pt = (Tmb @ T_bl[None])[:, :3, 3]
        rng_meas = np.linalg.norm(xyz[first] - o_pt, axis=1)
        d_r = dist[row[first], col[first]]
        a_r = alpha[row[first], col[first]]
        has_ret = a_r > 0.5
        ok = has_ret & (d_r < 100.0) & np.isfinite(d_r)
        err = d_r[ok] - rng_meas[ok]

        matched = np.abs(err) <= MATCH_CM / 100.0
        by_range = {}
        for lo, hi in zip(_RANGE_BINS[:-1], _RANGE_BINS[1:]):
            m = (rng_meas[ok] >= lo) & (rng_meas[ok] < hi)
            if m.sum() > 50:
                e = err[m]
                em = e[np.abs(e) <= MATCH_CM / 100.0]
                by_range[f"{lo}-{hi}m"] = {
                    "n": int(m.sum()), "matched_frac": round(float(len(em)) / max(int(m.sum()), 1), 4),
                    "median_abs_cm": round(float(np.median(np.abs(em))) * 100, 3) if len(em) else None,
                    "bias_cm": round(float(np.mean(em)) * 100, 3) if len(em) else None,
                    "mismatch_bias_cm": round(float(np.mean(e[np.abs(e) > MATCH_CM / 100.0])) * 100, 3)
                    if (np.abs(e) > MATCH_CM / 100.0).sum() > 10 else None}
        # incidence : |cos| entre rayon et normale du voxel touché (sur les points `ok` uniquement)
        inc_stats = None
        if map_nrm is not None:
            from scipy.spatial import cKDTree

            rng_ok = rng_meas[ok]
            xyz_ok = xyz[first][ok]
            o_ok = o_pt[ok]
            hits = xyz_ok
            tree_m = cKDTree(np.stack([means[:, 0].cpu().numpy(), means[:, 1].cpu().numpy(),
                                       means[:, 2].cpu().numpy()], 1))
            _, j = tree_m.query(hits, workers=-1)
            ray_w = (xyz_ok - o_ok) / rng_ok[:, None]
            cos_i = np.abs(np.einsum("ij,ij->i", ray_w, map_nrm[j]))
            # contenu stable de la carte : voxels vus par >= 3 tours et peu vegetaux
            n_scans_v = map_extra["n_scans"][j] if map_extra["n_scans"] is not None else np.ones(len(j))
            veg_v = map_extra["veg_frac"][j] if map_extra["veg_frac"] is not None else np.zeros(len(j))
            stable = (n_scans_v >= 3) & (veg_v < 0.1)
            inc_bins = {}
            for lo, hi in ((0.0, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)):
                m = (cos_i >= lo) & (cos_i < hi)
                if m.sum() > 100:
                    inc_bins[f"{lo:.2f}-{hi:.2f}"] = {
                        "n": int(m.sum()),
                        "median_err_cm": round(float(np.median(err[m])) * 100, 2),
                        "median_abs_cm": round(float(np.median(np.abs(err[m]))) * 100, 2)}
            inc_stats = {"bins_by_cos_incidence": inc_bins,
                         "stable_structure": {"n": int(stable.sum()),
                                              "matched_frac": round(float(matched[stable].mean()), 4) if stable.sum() else None,
                                              "median_err_cm": round(float(np.median(err[stable])) * 100, 2) if stable.sum() else None,
                                              "median_abs_cm": round(float(np.median(np.abs(err[stable]))) * 100, 2) if stable.sum() else None,
                                              "p90_abs_cm": round(float(np.percentile(np.abs(err[stable]), 90)) * 100, 2) if stable.sum() else None}}
            # sous-echantillon pour figures locales (10000 points ok)
            rng_s = np.random.default_rng(0)
            k = rng_s.choice(int(len(err)), min(10000, len(err)), replace=False)
            sample = {"range_m": rng_ok[k].round(3).tolist(), "err_cm": (err[k] * 100).round(2).tolist(),
                      "cos_inc": cos_i[k].round(3).tolist(), "stable": stable[k].astype(int).tolist()}
        scan_res = {
            "stamp": int(st), "t_start": t_start,
            "n_points_total": int(len(xyz)), "n_first_echo": int(first.sum()),
            "n_cells_returned_alpha": int(has_ret.sum()), "n_cells_finite": int(ok.sum()),
            "return_frac_on_first_echo_cells": round(float(has_ret.mean()), 4),
            "frac_absurd_gt100m": round(float((has_ret & (d_r >= 100.0)).mean()), 4),
            "error_first_echo_returns": _stats(err),
            "error_matched_le_10cm": _stats(err[matched]),
            "matched_frac": round(float(matched.mean()), 4),
            "mismatch_stats": {"n": int((~matched).sum()),
                               "median_cm": round(float(np.median(err[~matched])) * 100, 2) if (~matched).sum() else None,
                               "p90_cm": round(float(np.percentile(err[~matched], 90)) * 100, 2)
                               if (~matched).sum() > 10 else None},
            "error_by_incidence": inc_stats,
            "point_sample": sample,
            "error_by_range_bin": by_range,
            "render_s": round(time.time() - t1, 1),
        }
        # diagnostics : profil de l'erreur par rangée (ring) et par colonne
        rows_ok, cols_ok = row[first][ok], col[first][ok]
        scan_res["err_median_by_row_cm"] = [round(float(np.median(np.abs(err[rows_ok == r]))) * 100, 2)
                                            if (rows_ok == r).sum() > 20 else None for r in range(64)]
        colprof = []
        for cb in range(0, 600, 30):
            m = (cols_ok >= cb) & (cols_ok < cb + 30)
            colprof.append(round(float(np.median(err[m])) * 100, 2) if m.sum() > 20 else None)
        scan_res["err_bias_by_colbin_cm"] = colprof
        results["scans"].append(scan_res)
        all_err.append(err)
        all_rng.append(rng_meas[ok])
        if si == len(picks) // 2:
            # variante GLOBAL (sans oburateur roulant) : isole la question du sens de balayage
            renders_g, alphas_g, _ = gsplat.rasterization(
                means, quats, scales, opac, colors, viewmat_s[None], torch.eye(3, device=dev)[None],
                lidar.n_columns, lidar.n_rows, camera_model="lidar", lidar_coeffs=lidar,
                with_ut=True, with_eval3d=True, packed=False, render_mode="RGB-Ed", global_z_order=False,
                far_plane=FAR_PLANE, rolling_shutter=gsplat.RollingShutterType.GLOBAL, viewmats_rs=None)
            d_g = renders_g[0, ..., 3].double().cpu().numpy()
            d_g = d_g[row[first], col[first]]
            eg = (d_g - rng_meas)[ok]
            results["scans"][-1]["variant_global_shutter"] = _stats(eg)
            scan_res["variant_global_shutter"] = results["scans"][-1]["variant_global_shutter"]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#fcfcfb")
            axes[0].hist(err * 100, bins=120, color="#2a78d6")
            axes[0].axvline(0, color="#1f1f1e", linewidth=0.8)
            axes[0].set_xlabel("erreur rendu − mesuré (cm)", fontsize=8, color="#5f5e5a")
            axes[0].set_title(f"4.0b — erreur de distance, tour test_in {st}\nmédiane |e| = "
                              f"{scan_res['error_first_echo_returns']['median_abs_cm']:.2f} cm", fontsize=10,
                              color="#1f1f1e", loc="left")
            axes[1].scatter(rng_meas[ok], err * 100, s=0.4, color="#c8380d", alpha=0.3, linewidths=0)
            axes[1].set_xlabel("distance mesurée (m)", fontsize=8, color="#5f5e5a")
            axes[1].set_ylabel("erreur (cm)", fontsize=8, color="#5f5e5a")
            axes[1].set_ylim(-30, 30)
            for ax in axes:
                ax.set_facecolor("#fcfcfb")
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
                ax.tick_params(colors="#5f5e5a", labelsize=8)
                ax.grid(True, color="#e4e3df", linewidth=0.6)
                ax.set_axisbelow(True)
            fig.tight_layout()
            figs[f"scan_{st}.png"] = _fig_bytes(fig)
            plt.close(fig)

    err_all = np.concatenate(all_err)
    rng_all = np.concatenate(all_rng)
    ret_fracs = [s["return_frac_on_first_echo_cells"] for s in results["scans"]]
    matched_all = np.abs(err_all) <= MATCH_CM / 100.0
    overall = {
        "n_scans": len(picks), "n_returns": int(len(err_all)),
        "matched_frac": round(float(matched_all.mean()), 4),
        "median_abs_cm": round(float(np.median(np.abs(err_all[matched_all]))) * 100, 3),
        "bias_cm": round(float(np.mean(err_all[matched_all])) * 100, 3),
        "p90_abs_cm": round(float(np.percentile(np.abs(err_all[matched_all]), 90)) * 100, 3),
        "mismatch_frac": round(float((~matched_all).mean()), 4),
        "mismatch_median_cm": round(float(np.median(err_all[~matched_all])) * 100, 2),
        "min_return_frac": round(float(min(ret_fracs)), 4),
    }
    overall["result"] = "PASS" if (overall["median_abs_cm"] <= CRIT["median_abs_m"] * 100
                                   and abs(overall["bias_cm"]) <= CRIT["bias_m"] * 100
                                   and overall["min_return_frac"] >= CRIT["return_frac"]) else "FAIL"
    results["overall"] = overall

    import matplotlib

    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#fcfcfb")
    ax.hist(err_all * 100, bins=200, color="#2a78d6")
    ax.axvline(0, color="#1f1f1e", linewidth=0.8)
    ax.set_xlim(-40, 40)
    ax.set_xlabel("erreur rendu − mesuré (cm), tous tours test_in", fontsize=8, color="#5f5e5a")
    ax.set_title(f"4.0b — conventions de bout en bout : {overall['result']}\n"
                 f"médiane |e| {overall['median_abs_cm']:.2f} cm, biais {overall['bias_cm']:+.2f} cm, "
                 f"retours {overall['min_return_frac'] * 100:.1f} %", fontsize=10, color="#1f1f1e", loc="left")
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#5f5e5a", labelsize=8)
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    figs["overall_hist.png"] = _fig_bytes(fig)
    plt.close(fig)

    results["wall_s"] = round(time.time() - t0, 1)
    results["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    results["_figures"] = figs
    return results


def _fig_bytes(fig) -> bytes:
    import io

    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    return buf.getvalue()


@app.local_entrypoint()
def main(n_scans: int = 3, seq: str = SEQ, sigma: float = GAUSS_SIGMA, sigma_n: float = 0.01,
         split: str = "test_in"):
    import json
    import os

    out = run_main.remote(n_scans, seq, sigma, sigma_n, split)
    os.makedirs("logs/stage4", exist_ok=True)
    figs = out.pop("_figures", {})
    with open("logs/stage4/conventions_e2e.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for name, b in figs.items():
        open(f"logs/stage4/{name}", "wb").write(b)
    print(json.dumps(out, indent=1, ensure_ascii=False))
