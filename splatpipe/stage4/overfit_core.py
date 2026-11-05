"""Étape 4.0c — cœur du surapprentissage d'un balayage LiDAR réel + d'une image réelle.

Module pur Python (sans Modal) : appelé par l'enveloppe Modal `overfit_tour_image.py`
ou directement sur la desktop (`venv/bin/python splatpipe/stage4/overfit_core.py --root ...`).

Conventions validées en 4.0b : disques plats orientés (σ_t 4 cm, σ_n 1 cm, α 0,99),
modèle de faisceaux QT64 + `safe_lidar_tiling`, rendu `RGB-Ed` rolling shutter.
Nouveau en 4.0c : chemin image fisheye équidistant OpenCV (`radial_coeffs` = k1..k4,
obturateur global → une seule viewmat) et optimisation (Adam) des deux pertes :

  - LiDAR : Huber(d_rendu − d_mesuré), cellules premier écho ≤ RANGE_MAX, hors
    rayons dynamiques (sidecar `semantics_lidar/` si présent, sinon `n_scans >= 3`) ;
  - RGB : L1 sur les pixels non-ciel de l'image la plus proche du milieu du balayage.

Contrôles : métriques appariées / divergentes séparées (leçon 4.0b), sur le balayage
optimisé ET un balayage de contrôle (dérive).
"""

from __future__ import annotations

import json
import os
import time

MAP_NAME = "map_5cm_static.ply"
SIGMA_T, SIGMA_N, OPACITY = 0.04, 0.01, 0.99
RANGE_MAX = 28.0        # cellules de perte LiDAR : distance mesurée ≤ (marge dans le crop)
HUBER_DELTA_M = 0.02
LR = {"means": 1e-3, "quats": 1e-3, "scales": 1e-3, "opac": 5e-3, "colors": 5e-3}
MATCH_CM = 10.0
_PLY_TYPES = {"float": "f4", "double": "f8", "uchar": "u1", "char": "i1", "ushort": "u2",
              "short": "i2", "uint": "u4", "int": "i4"}


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


def _load_map_disks(dev, ply_path: str, sigma_t: float = SIGMA_T, sigma_n: float = SIGMA_N):
    """Gaussiennes disques PLATS orientés par la normale du voxel (initialisation 4.0b)."""
    import numpy as np
    import torch

    off, n, props = _ply_header(ply_path)
    dt = np.dtype([(name, "<" + t) for name, t in props])
    mm = np.memmap(ply_path, dtype=dt, mode="r", offset=off, shape=(n,))
    xyz = np.stack([mm["x"], mm["y"], mm["z"]], 1).astype(np.float32)
    nrm = np.stack([mm["nx"], mm["ny"], mm["nz"]], 1).astype(np.float32)
    n_scans = np.asarray(mm["n_scans"], np.int64)
    del mm
    means = torch.from_numpy(xyz).to(dev)
    nn = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    a = np.where(np.abs(nn[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = np.cross(nn, a)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(nn, t1)
    R = np.stack([t1, t2, nn], axis=1)      # colonnes = axes locaux ; z = normale
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
    scales = torch.from_numpy(np.tile(np.array([sigma_t, sigma_t, sigma_n], np.float32),
                                      (len(means), 1))).to(dev)
    opac = torch.full((len(means),), OPACITY, device=dev)
    return means, quats, scales, opac, n_scans


def safe_lidar_tiling(params, tile_size=8, start=64, step=4):
    """`compute_tiling` dont AUCUNE tuile ne dépasse tile_size² rayons — copié de
    smoke_gsplat_lidar.py (T4.0a) pour ne pas importer modal sur la desktop."""
    from gsplat.cuda._lidar import compute_tiling

    m = start
    while True:
        tiling = compute_tiling(params, max_pts_per_tile=m)
        if int(tiling.tiles_pack_info[:, 1].max()) <= tile_size * tile_size:
            return tiling, m
        m -= step


def _load_lidar_model_root(root: str, dev):
    """Modèle de faisceaux (QT64 à Oxford, OS1-64 sur DiTer++ : `lidar_model/<id>.json`, id = premier LiDAR de
    `sensors.yaml`) + tiling sûr (cf. T4.0a ; wrapper haut niveau obligatoire)."""
    import torch
    from gsplat import RowOffsetStructuredSpinningLidarModelParametersExt
    from gsplat.cuda._lidar import (
        RowOffsetStructuredSpinningLidarModelParameters,
        SpinningDirection,
        compute_angles_to_columns_map,
    )

    from splatpipe.common.lidar_model import StructuredSpinningLidarModel

    from splatpipe.train.data import lidar_paths

    m = StructuredSpinningLidarModel.load_json(f"{root}/lidar_model/{lidar_paths(root)[0]}.json")
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
    return lidar, used


def _nearest_idx(means_t, xyz_np):
    """indice du gaussian le plus proche pour chaque point (numpy, KD-tree)."""
    import numpy as np
    from scipy.spatial import cKDTree

    tree = cKDTree(means_t.detach().cpu().numpy())
    _, j = tree.query(xyz_np, workers=-1)
    return j


def run_overfit(root: str, dev: str = "cuda", n_iters: int = 400, r_opt: float = 35.0,
                render_scale: float = 1.0, cam_name: str = "cam0") -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    import yaml
    from PIL import Image

    from splatpipe.common.camera_model import FisheyeCamera
    from splatpipe.common.trajectory import Trajectory

    t0 = time.time()
    import gsplat

    lidar, _ = _load_lidar_model_root(root, dev)
    means, quats, scales, opac, map_nsc = _load_map_disks(dev, f"{root}/map/{MAP_NAME}")
    n_map = len(means)

    # -------------------------------------------------- balayages train + contrôle
    splits = yaml.safe_load(open(f"{root}/splits.yaml"))["lidars"]["hesai_qt64"]
    desk = {int(s[:-4]) for s in os.listdir(f"{root}/lidar/hesai_deskewed") if s.endswith(".npz")}
    train = sorted((set(int(s) for s in (splits.get("train") or []))) & desk)
    if len(train) < 2:
        raise RuntimeError(f"pas assez de balayages train : {len(train)}")
    i_mid = len(train) // 2
    sweeps = {"train": train[i_mid],
              "holdout": train[i_mid + 10] if len(train) > i_mid + 10 else train[i_mid - 1]}

    traj = Trajectory.load(f"{root}/traj/spline.npz")
    T_bl = np.asarray(yaml.safe_load(open(f"{root}/sensors.yaml"))["lidars"]["hesai_qt64"]["T_base_lidar"], float)

    sweep_data = {}
    for tag, st in sweeps.items():
        with np.load(f"{root}/lidar/hesai_deskewed/{st}.npz") as z:
            d = {k: z[k] for k in ("xyz", "ret", "row", "col", "t_rel", "t_start", "t_end",
                                   "T_map_lidar_start", "T_map_lidar_end")}
        d["xyz"] = d["xyz"].astype(np.float64)
        first = d["ret"].astype(int) == 0
        Tmb, _ = traj.matrices([0.5 * (float(d["t_start"]) + float(d["t_end"]))])
        o_mid = (np.asarray(Tmb[0]) @ T_bl)[:3, 3]
        sweep_data[tag] = dict(d, first=first, o_mid=o_mid, stamp=int(st))

    o_ref = sweep_data["train"]["o_mid"]
    near = torch.nonzero(((means - torch.from_numpy(o_ref).to(dev)) ** 2).sum(1) < r_opt ** 2).ravel()
    means, quats, scales, opac = (p[near].clone() for p in (means, quats, scales, opac))
    del near
    torch.cuda.empty_cache()
    colors = torch.full((len(means), 3), 0.5, device=dev)
    for p in (means, quats, scales, opac, colors):
        p.requires_grad_(True)
    n_opt = len(means)
    print(f"zone optimisée : {n_opt} / {n_map} gaussiennes (rayon {r_opt} m)", flush=True)

    # -------------------------------------------------- cellules de perte (train)
    d_tr = sweep_data["train"]
    first = d_tr["first"]
    xyz_f, row_f, col_f = d_tr["xyz"][first], d_tr["row"][first], d_tr["col"][first]
    t_pt = float(d_tr["t_start"]) + d_tr["t_rel"][first].astype(np.float64)
    Tmb, _ = traj.matrices(list(t_pt))
    o_pt = (np.asarray(Tmb) @ T_bl[None])[:, :3, 3]
    rng_meas = np.linalg.norm(xyz_f - o_pt, axis=1)
    viewmat_s = torch.tensor(np.linalg.inv(np.asarray(d_tr["T_map_lidar_start"], float)),
                             dtype=torch.float32, device=dev)
    viewmat_e = torch.tensor(np.linalg.inv(np.asarray(d_tr["T_map_lidar_end"], float)),
                             dtype=torch.float32, device=dev)

    dyn_mode = "n_scans>=3"
    keep = (rng_meas <= RANGE_MAX) & (map_nsc[_nearest_idx(means, xyz_f)] >= 3)
    sc_dir = f"{root}/semantics_lidar"
    if os.path.isdir(sc_dir):
        cands = [f for f in os.listdir(sc_dir) if str(d_tr["stamp"]) in f]
        if cands:
            try:
                z = np.load(f"{sc_dir}/{cands[0]}")
                key = next(k for k in z.files if "dyn" in k.lower())
                dyn_pt = z[key].astype(bool)[: len(d_tr["xyz"])]
                keep = (rng_meas <= RANGE_MAX) & (~dyn_pt[first])
                dyn_mode = f"semantics_lidar/{cands[0]}:{key}"
            except Exception as e:  # noqa: BLE001 — repli documenté
                dyn_mode += f" (sidecar illisible : {e})"
    print(f"masque dyn : {dyn_mode} ; cellules de perte : {int(keep.sum())}/{len(rng_meas)}", flush=True)
    row_t = torch.from_numpy(row_f[keep].astype(np.int64)).to(dev)
    col_t = torch.from_numpy(col_f[keep].astype(np.int64)).to(dev)
    rng_t = torch.from_numpy(rng_meas[keep].astype(np.float32)).to(dev)

    def lidar_render():
        renders, alphas, _ = gsplat.rasterization(
            means, quats, scales, opac, colors, viewmat_s[None], torch.eye(3, device=dev)[None],
            lidar.n_columns, lidar.n_rows, camera_model="lidar", lidar_coeffs=lidar,
            with_ut=True, with_eval3d=True, packed=False, render_mode="RGB-Ed",
            global_z_order=False, far_plane=300.0,
            rolling_shutter=gsplat.RollingShutterType.ROLLING_LEFT_TO_RIGHT,
            viewmats_rs=viewmat_e[None])
        return renders[0, ..., 3], alphas[0, ..., 0]

    # -------------------------------------------------- image fisheye
    import csv as _csv

    cams_cfg = yaml.safe_load(open(f"{root}/sensors.yaml"))["cameras"]
    poses = json.load(open(f"{root}/poses/cameras.json"))["cameras"]
    cam = cam_name
    stamps = sorted(int(s) for s in poses[cam].keys())
    t_mid_ns = int(0.5 * (float(d_tr["t_start"]) + float(d_tr["t_end"])) * 1e9)
    st_img = min(stamps, key=lambda s: abs(s - t_mid_ns))
    entry = poses[cam][str(st_img)]
    T_wc = np.asarray(entry["T_map_cam_start"], float)   # obturateur global : début = fin
    K = FisheyeCamera.from_dict(cams_cfg[cam])
    viewmat_c = torch.tensor(np.linalg.inv(T_wc), dtype=torch.float32, device=dev)
    s = render_scale
    W_r, H_r = int(K.width * s) // 2 * 2, int(K.height * s) // 2 * 2
    Ks = torch.zeros(1, 3, 3, device=dev)
    Ks[0, 0, 0], Ks[0, 1, 1] = K.fx * s, K.fy * s
    Ks[0, 0, 2], Ks[0, 1, 2] = K.cx * s, K.cy * s
    Ks[0, 2, 2] = 1.0
    radial = torch.tensor([[K.k1, K.k2, K.k3, K.k4]], dtype=torch.float32, device=dev)

    rel = None
    with open(f"{root}/cameras/{cam}/index.csv") as f:
        for r in _csv.DictReader(f):
            if int(r["stamp_ns"]) == st_img:
                rel = r["rel_path"]
                break
    if rel is None:
        raise RuntimeError(f"image {st_img} absente de l'index {cam}")
    # rel_path = chemin du dataset brut ; les jpg recadrés sont dans rawimgs/<cam>/<basename>
    base = os.path.basename(rel)
    img_path = f"{root}/rawimgs/{cam}/{base}"
    if not os.path.exists(img_path):
        img_path = f"{root}/{rel}"                     # repli : chemin direct (volume Modal)
    img_u8 = np.asarray(Image.open(img_path).convert("RGB"))
    if s != 1.0:
        img_u8 = np.asarray(Image.fromarray(img_u8).resize((W_r, H_r), Image.BILINEAR))
    img_t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).to(dev)
    msk_path = f"{root}/masks/{cam}/{st_img}.png"
    sky = None
    if os.path.exists(msk_path):
        msk = np.asarray(Image.open(msk_path))
        if s != 1.0:
            msk = np.asarray(Image.fromarray(msk).resize((W_r, H_r), Image.NEAREST))
        sky = (msk & 1) > 0                                     # bit 1 = ciel
    sky_t = torch.from_numpy(sky).to(dev) if sky is not None else None

    def rgb_render():
        out, alphas, _ = gsplat.rasterization(
            means, quats, scales, opac, colors, viewmat_c[None], Ks, W_r, H_r,
            camera_model="fisheye", radial_coeffs=radial, with_ut=True, with_eval3d=True,
            packed=False, render_mode="RGB", far_plane=300.0)
        return out[0], alphas[0, ..., 0]

    # -------------------------------------------------- évaluation LiDAR (plein balayage)
    def eval_lidar(tag):
        d = sweep_data[tag]
        f = d["first"]
        xyz_f = d["xyz"][f]
        t_pt = float(d["t_start"]) + d["t_rel"][f].astype(np.float64)
        Tmb, _ = traj.matrices(list(t_pt))
        o_pt = (np.asarray(Tmb) @ T_bl[None])[:, :3, 3]
        rm = np.linalg.norm(xyz_f - o_pt, axis=1)
        vs = torch.tensor(np.linalg.inv(np.asarray(d["T_map_lidar_start"], float)),
                          dtype=torch.float32, device=dev)
        ve = torch.tensor(np.linalg.inv(np.asarray(d["T_map_lidar_end"], float)),
                          dtype=torch.float32, device=dev)
        with torch.no_grad():
            renders, alphas, _ = gsplat.rasterization(
                means, quats, scales, opac, colors, vs[None], torch.eye(3, device=dev)[None],
                lidar.n_columns, lidar.n_rows, camera_model="lidar", lidar_coeffs=lidar,
                with_ut=True, with_eval3d=True, packed=False, render_mode="RGB-Ed",
                global_z_order=False, far_plane=300.0,
                rolling_shutter=gsplat.RollingShutterType.ROLLING_LEFT_TO_RIGHT,
                viewmats_rs=ve[None])
        dr = renders[0, ..., 3].double().cpu().numpy()
        ar = alphas[0, ..., 0].double().cpu().numpy()
        has = ar[d["row"][f], d["col"][f]] > 0.5
        dr_c = dr[d["row"][f], d["col"][f]]
        okk = has & np.isfinite(dr_c) & (dr_c < 100.0)
        err = dr_c[okk] - rm[okk]
        matched = np.abs(err) <= MATCH_CM / 100
        return {"n_ok": int(okk.sum()),
                "return_frac": round(float(has.mean()), 4),
                "matched_frac": round(float(matched.mean()), 4),
                "median_abs_cm": round(float(np.median(np.abs(err[matched]))) * 100, 2) if matched.any() else None,
                "bias_cm": round(float(np.mean(err[matched])) * 100, 2) if matched.any() else None,
                "divergent_median_cm": round(float(np.median(err[~matched])) * 100, 2) if (~matched).any() else None}

    # -------------------------------------------------- boucle d'optimisation
    opt = torch.optim.Adam([
        {"params": means, "lr": LR["means"]}, {"params": quats, "lr": LR["quats"]},
        {"params": scales, "lr": LR["scales"]}, {"params": opac, "lr": LR["opac"]},
        {"params": colors, "lr": LR["colors"]}])
    huber = torch.nn.HuberLoss(delta=HUBER_DELTA_M)
    hist, grads0 = [], {}
    t_r0 = time.time()
    with torch.no_grad():
        d0, _ = lidar_render()
        l0 = huber(d0[row_t, col_t], rng_t)
    print(f"premier rendu lidar : {time.time() - t_r0:.2f}s ; perte init {float(l0):.4f}", flush=True)
    lidar_train_before = eval_lidar("train")
    lidar_holdout_before = eval_lidar("holdout")
    print(f"before train   : {lidar_train_before}", flush=True)
    print(f"before holdout : {lidar_holdout_before}", flush=True)

    for it in range(n_iters + 1):
        opt.zero_grad(set_to_none=True)
        d_r, _ = lidar_render()
        l_lidar = huber(d_r[row_t, col_t], rng_t)
        rgb, _ = rgb_render()
        l_rgb = F.l1_loss(rgb, img_t) if sky_t is None else \
            F.l1_loss(rgb[~sky_t], img_t[~sky_t])
        loss = l_lidar + l_rgb
        if it == 0:
            loss.backward()
            grads0 = {n: float(p.grad.norm()) if p.grad is not None else 0.0
                      for n, p in (("means", means), ("quats", quats), ("scales", scales),
                                   ("opac", opac), ("colors", colors))}
            opt.zero_grad(set_to_none=True)
            print(f"grad norm it0 : {json.dumps({k: f'{v:.2e}' for k, v in grads0.items()})}", flush=True)
        else:
            loss.backward()
            opt.step()
        if it % 40 == 0:
            hist.append({"iter": it, "l_lidar": round(float(l_lidar), 5), "l_rgb": round(float(l_rgb), 5)})
            print(f"it {it:4d}  l_lidar {float(l_lidar):.5f}  l_rgb {float(l_rgb):.5f}", flush=True)

    # -------------------------------------------------- évaluations finales
    res = {"setup": {"root": root, "n_gaussians_map": int(n_map), "n_gaussians_opt": int(n_opt),
                     "radius_m": r_opt, "iters": n_iters, "render_scale": render_scale,
                     "sweeps": {k: sweeps[k] for k in sweeps},
                     "image": {"cam": cam, "stamp": st_img, "path": rel, "sky_mask": sky is not None},
                     "dyn_mask": dyn_mode, "huber_delta_m": HUBER_DELTA_M, "lr": LR,
                     "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1)},
           "grad_norms_iter0": grads0, "hist": hist,
           "lidar_train_before": lidar_train_before,
           "lidar_holdout_before": lidar_holdout_before}
    res["lidar_train_after"] = eval_lidar("train")
    res["lidar_holdout_after"] = eval_lidar("holdout")
    with torch.no_grad():
        rgb, _ = rgb_render()
        res["rgb_after"] = {"l1": round(float((rgb - img_t).abs().mean()), 5),
                            "psnr": round(float(-10 * torch.log10(((rgb - img_t) ** 2).mean())), 2)}
    res["wall_s"] = round(time.time() - t0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="#fcfcfb")
    axes[0].imshow(img_u8)
    axes[0].set_title(f"photo {cam} ({st_img})", fontsize=9, color="#1f1f1e", loc="left")
    axes[1].imshow(rgb.clamp(0, 1).cpu().numpy())
    axes[1].set_title(f"rendu après {n_iters} it. — PSNR {res['rgb_after']['psnr']} dB",
                      fontsize=9, color="#1f1f1e", loc="left")
    for a in axes:
        a.set_axis_off()
    fig.tight_layout()
    import io

    buf = io.BytesIO()
    fig.savefig(buf, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    res["_figure_rgb"] = buf.getvalue()

    res["verdict"] = {
        "gradients_non_nuls": all(v > 0 for v in grads0.values()),
        "l_lidar_x": round(hist[0]["l_lidar"] / max(hist[-1]["l_lidar"], 1e-9), 1),
        "l_rgb_x": round(hist[0]["l_rgb"] / max(hist[-1]["l_rgb"], 1e-9), 1),
        "critere_plan": "pertes -> ~0, gradients non nuls (SYNTHESE §7 / 4.0c)",
    }
    return res


if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--root", required=True, help="répertoire scène (contient map/, lidar/, poses/…)")
    pa.add_argument("--n-iters", type=int, default=400)
    pa.add_argument("--r-opt", type=float, default=35.0)
    pa.add_argument("--render-scale", type=float, default=1.0)
    pa.add_argument("--out", default="logs/stage4/overfit_tour_image.json")
    a = pa.parse_args()
    out = run_overfit(a.root, n_iters=a.n_iters, r_opt=a.r_opt, render_scale=a.render_scale)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig = out.pop("_figure_rgb", None)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    if fig:
        open(a.out.replace(".json", "_rgb.png"), "wb").write(fig)
    print(json.dumps({k: v for k, v in out.items() if k != "hist"}, indent=1, ensure_ascii=False))
