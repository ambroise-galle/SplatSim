"""Décomposition du coût de rendu gsplat (caméra et LiDAR) : nombre de gaussiennes (élagage par importance),
degré SH, mode de rendu, résolution LiDAR, obturateur roulant. Même scène que bench_sim.py.

  PYTHONPATH=. venv/bin/python simrobot/tools/bench_render.py --tiles data_local/simrobot/grandtour/HOB-1/* \
      --lidar-model data_local/diterpp/lawn/scene_upper_day_604_784_tls/lidar_model/ouster_os1_64.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gaussgym_export import load_tile_mission, tile_cameras  # noqa: E402


def main():
    import torch
    import gsplat
    from gsplat import rasterization
    from scipy.spatial import cKDTree

    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--lidar-model", required=True)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    dev = "cuda"

    tiles = [t.rstrip("/") for t in a.tiles if os.path.exists(f"{t}/splatfacto/splat.ply")]
    cams = [tile_cameras(t) for t in tiles]
    owner = np.concatenate([np.full(len(c), i) for i, c in enumerate(cams)])
    tree = cKDTree(np.concatenate(cams))
    parts = []
    for i, t in enumerate(tiles):
        G = load_tile_mission(t)
        _, j = tree.query(G["means"])
        k = (owner[j] == i) & (G["opac"] >= 0.05)
        parts.append({kk: v[k] for kk, v in G.items()})
    G = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    # importance ~ opacité × surface projetée (produit des 2 plus grandes échelles)
    s = np.sort(G["scales"], 1)
    imp = G["opac"] * s[:, 1] * s[:, 2]
    order = np.argsort(-imp)

    fr = []
    for t in tiles:
        for f in json.load(open(f"{t}/transforms.json"))["frames"]:
            if "hdr_front" in f["file_path"]:
                s_, ns = str(f["timestamp"]).split("_")
                fr.append((int(s_) + int(ns) * 1e-9, f))
    fr.sort(key=lambda x: x[0])
    poses = []
    for _, f in fr[:: max(1, len(fr) // a.n)][: a.n]:
        T = np.array(f["transform_matrix"], float)
        T[:3, 1:3] *= -1
        poses.append(T)
    f0 = fr[0][1]
    VM = torch.tensor(np.linalg.inv(np.stack(poses)), device=dev, dtype=torch.float32)

    def timeit(fn, n=a.n):
        for i in range(3):
            fn(i)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(n):
            fn(i)
        torch.cuda.synchronize()
        return round((time.perf_counter() - t0) / n * 1e3, 2)

    def scene(n_keep, deg):
        idx = torch.tensor(order[:n_keep], device=dev)
        S = {k: torch.tensor(v, device=dev, dtype=torch.float32)[idx] for k, v in G.items()}
        S["quats"] = S["quats"] / S["quats"].norm(dim=1, keepdim=True)
        if deg == 0:
            S["sh"] = S["sh"][:, :1]
        return S

    rep = {"n_total": int(len(order))}
    W, H = 640, 400
    sc = W / float(f0["w"])
    K = torch.tensor([[float(f0["fl_x"]) * sc, 0, float(f0["cx"]) * sc], [0, float(f0["fl_y"]) * sc, float(f0["cy"]) * sc],
                      [0, 0, 1]], device=dev)[None]
    for n_keep in [len(order), 2_500_000, 1_000_000, 500_000]:
        for deg in (3, 0):
            S = scene(n_keep, deg)
            for mode in ("RGB", "RGB+ED"):
                ms = timeit(lambda i: rasterization(S["means"], S["quats"], S["scales"], S["opac"], S["sh"],
                                                    VM[i % len(VM)][None], K, W, H, sh_degree=deg, render_mode=mode,
                                                    far_plane=200.0))
                rep[f"cam640_n{n_keep // 1000}k_sh{deg}_{mode}"] = ms
            del S
            torch.cuda.empty_cache()
        print(json.dumps({k: v for k, v in rep.items() if f"n{n_keep // 1000}k" in k}), flush=True)
    # résolution caméra à 1 M
    S = scene(1_000_000, 3)
    for W2, H2 in [(320, 200), (1280, 800), (1920, 1280)]:
        sc2 = W2 / float(f0["w"])
        K2 = torch.tensor([[float(f0["fl_x"]) * sc2, 0, float(f0["cx"]) * sc2], [0, float(f0["fl_y"]) * sc2, float(f0["cy"]) * sc2],
                           [0, 0, 1]], device=dev)[None]
        rep[f"cam{W2}_n1000k_sh3_RGB+ED"] = timeit(lambda i: rasterization(
            S["means"], S["quats"], S["scales"], S["opac"], S["sh"], VM[i % len(VM)][None], K2, W2, H2, sh_degree=3,
            render_mode="RGB+ED", far_plane=200.0))
    print(json.dumps({k: v for k, v in rep.items() if k.startswith("cam") and "n1000k_sh3_RGB+ED" in k}), flush=True)

    # ------------------------------------------------ LiDAR
    from gsplat import RowOffsetStructuredSpinningLidarModelParametersExt
    from gsplat.cuda._lidar import (RowOffsetStructuredSpinningLidarModelParameters, SpinningDirection,
                                    compute_angles_to_columns_map)
    from splatpipe.common.lidar_model import StructuredSpinningLidarModel
    from splatpipe.stage4.overfit_core import safe_lidar_tiling

    m = StructuredSpinningLidarModel.load_json(a.lidar_model)
    kw = m.gsplat_kwargs()

    def make_lidar(rows):
        sel = np.linspace(0, len(kw["row_elevations_rad"]) - 1, rows).round().astype(int)
        params = RowOffsetStructuredSpinningLidarModelParameters(
            row_elevations_rad=torch.from_numpy(np.ascontiguousarray(kw["row_elevations_rad"][sel])).to(dev),
            column_azimuths_rad=torch.from_numpy(kw["column_azimuths_rad"]).to(dev),
            row_azimuth_offsets_rad=torch.from_numpy(np.ascontiguousarray(kw["row_azimuth_offsets_rad"][sel])).to(dev),
            spinning_frequency_hz=kw["spinning_frequency_hz"], spinning_direction=SpinningDirection[kw["spinning_direction"]])
        tiling, _ = safe_lidar_tiling(params)
        return RowOffsetStructuredSpinningLidarModelParametersExt(params, compute_angles_to_columns_map(params).to(dev), tiling)

    LV = []
    for T in poses:
        fwd = T[:3, 2].copy()
        fwd[2] = 0
        fwd /= np.linalg.norm(fwd)
        R = np.stack([fwd, np.cross([0, 0, 1], fwd), [0, 0, 1]], 1)
        L = np.eye(4)
        L[:3, :3], L[:3, 3] = R, T[:3, 3] + [0, 0, 0.4]
        LV.append(np.linalg.inv(L))
    LV = torch.tensor(np.stack(LV), device=dev, dtype=torch.float32)
    for n_keep in [len(order), 1_000_000]:
        S = scene(n_keep, 0)
        gray = torch.full((len(S["means"]), 3), 0.5, device=dev)
        for rows in (64, 32):
            lid = make_lidar(rows)
            for rs in (True, False):
                def run(i):
                    kw2 = dict(rolling_shutter=gsplat.RollingShutterType.ROLLING_LEFT_TO_RIGHT,
                               viewmats_rs=LV[(i + 1) % len(LV)][None]) if rs else {}
                    return gsplat.rasterization(S["means"], S["quats"], S["scales"], S["opac"], gray,
                                                LV[i % len(LV)][None], torch.eye(3, device=dev)[None], lid.n_columns,
                                                lid.n_rows, camera_model="lidar", lidar_coeffs=lid, with_ut=True,
                                                with_eval3d=True, packed=False, render_mode="RGB-Ed",
                                                global_z_order=False, far_plane=120.0, **kw2)
                rep[f"lidar{rows}x1024_n{n_keep // 1000}k_rs{int(rs)}"] = timeit(run, 20)
        print(json.dumps({k: v for k, v in rep.items() if k.startswith("lidar") and f"n{n_keep // 1000}k" in k}), flush=True)
        del S, gray
        torch.cuda.empty_cache()
    json.dump(rep, open("out/simrobot/bench/bench_render.json", "w"), indent=1)


if __name__ == "__main__":
    main()
