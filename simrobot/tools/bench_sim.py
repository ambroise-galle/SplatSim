"""Banc de performance du simulateur sur la desktop : (1) rendu caméra gsplat, (2) rendu LiDAR à balayage gsplat
(modèle à faisceaux structuré, obturateur roulant), (3) physique MuJoCo d'un robot à roues sur un champ de hauteur
tiré des maillages GaussGym. Scène : tuiles GaussGym assemblées (repère mission, mètres).

  venv/bin/python simrobot/tools/bench_sim.py --tiles data_local/simrobot/grandtour/HOB-1/* \
      --lidar-model data_local/diterpp/lawn/scene_upper_day_604_784_tls/lidar_model/ouster_os1_64.json \
      --out out/simrobot/bench
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


def read_mesh_ply(path):
    """Maillage Open3D binaire (x, y, z en double [+ normales double] + couleurs uchar ; faces uchar + 3 uint)."""
    with open(path, "rb") as f:
        props, nv, nf = [], 0, 0
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                nv = int(line.split()[-1])
            elif line.startswith("element face"):
                nf = int(line.split()[-1])
            elif line.startswith("property") and "list" not in line:
                props.append((line.split()[-1], {"double": "<f8", "float": "<f4", "uchar": "u1"}[line.split()[1]]))
            elif line == "end_header":
                break
        V = np.fromfile(f, dtype=np.dtype(props), count=nv)
        F = np.fromfile(f, dtype=np.dtype([("n", "u1"), ("v", "<u4", 3)]), count=nf)
    return np.stack([V["x"], V["y"], V["z"]], 1), F["v"].astype(np.int64)


def heightfield(meshes, res=0.1, pad=2.0):
    """Champ de hauteur (grille res) : hauteur max des triangles échantillonnés par cellule, trous bouchés au plus proche."""
    from scipy.ndimage import distance_transform_edt

    pts = []
    for V, F in meshes:
        tri = V[F]                                            # (nf, 3, 3)
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        k = np.clip(np.ceil(area / (res * res / 4)), 1, 50).astype(int)
        idx = np.repeat(np.arange(len(F)), k)
        u, v = np.random.default_rng(0).random((2, len(idx)))
        m = u + v > 1
        u[m], v[m] = 1 - u[m], 1 - v[m]
        t = tri[idx]
        pts.append(t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0]))
    P = np.concatenate(pts)
    lo = P[:, :2].min(0) - pad
    ij = np.floor((P[:, :2] - lo) / res).astype(int)
    n = ij.max(0) + 1
    H = np.full(n, -np.inf)
    np.maximum.at(H, (ij[:, 0], ij[:, 1]), P[:, 2])
    hole = ~np.isfinite(H)
    _, (ii, jj) = distance_transform_edt(hole, return_indices=True)
    H = H[ii, jj]
    return H, lo, res, float(hole.mean())


def robot_xml(H, lo, res, start, yaw):
    """Robot à 4 roues motrices (skid-steer, type Husky réduit) sur un hfield MuJoCo."""
    nx, ny = H.shape
    zmin, zmax = float(H.min()), float(H.max())
    sx, sy = nx * res / 2, ny * res / 2
    cx, cy = lo[0] + sx, lo[1] + sy
    qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    wheels = ""
    for name, x, y in [("fl", 0.25, 0.24), ("fr", 0.25, -0.24), ("rl", -0.25, 0.24), ("rr", -0.25, -0.24)]:
        wheels += f"""
      <body name="w_{name}" pos="{x} {y} -0.05" euler="90 0 0">
        <joint name="j_{name}" type="hinge" axis="0 0 -1" damping="0.05"/>
        <geom type="cylinder" size="0.13 0.05" mass="2" friction="1.0 0.02 0.002" rgba=".1 .1 .1 1"/>
      </body>"""
    acts = "".join(f'<velocity name="v_{n}" joint="j_{n}" kv="20" ctrlrange="-15 15"/>' for n in ["fl", "fr", "rl", "rr"])
    return f"""<mujoco model="robot_roues">
  <option timestep="0.002" integrator="implicitfast"/>
  <asset><hfield name="terrain" nrow="{ny}" ncol="{nx}" size="{sx} {sy} {zmax - zmin + 1e-3} 0.5"/></asset>
  <worldbody>
    <geom type="hfield" hfield="terrain" pos="{cx} {cy} {zmin}" friction="1.0 0.02 0.002"/>
    <body name="chassis" pos="{start[0]} {start[1]} {start[2] + 0.45}" quat="{qz[0]} {qz[1]} {qz[2]} {qz[3]}">
      <freejoint/>
      <geom type="box" size="0.35 0.22 0.1" mass="30" rgba=".8 .5 .1 1"/>
      <site name="capteurs" pos="0.3 0 0.25"/>{wheels}
    </body>
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>""", (zmin, zmax)


def main():
    import cv2
    import mujoco
    import torch
    import gsplat
    from gsplat import rasterization
    from scipy.spatial import cKDTree

    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--lidar-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    rep = {}

    # ------------------------------------------------ scène assemblée (Voronoï sur caméras)
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
    S = {k: torch.tensor(v, device=dev, dtype=torch.float32) for k, v in G.items()}
    S["quats"] = S["quats"] / S["quats"].norm(dim=1, keepdim=True)
    rep["n_gaussiennes"] = int(len(G["means"]))
    rep["vram_scene_Go"] = round(torch.cuda.memory_allocated() / 1e9, 2)

    # trajectoire : caméras avant, triées dans le temps
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

    def timeit(fn, n):
        fn(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(n):
            fn(i)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    # ------------------------------------------------ (1) caméra
    for W, H in [(640, 400), (1280, 800), (1920, 1280)]:
        sc = W / float(f0["w"])
        K = torch.tensor([[float(f0["fl_x"]) * sc, 0, float(f0["cx"]) * sc], [0, float(f0["fl_y"]) * sc, float(f0["cy"]) * sc],
                          [0, 0, 1]], device=dev)
        VM = torch.tensor(np.linalg.inv(np.stack(poses)), device=dev, dtype=torch.float32)

        def cam(i, B=1):
            vm = VM[i % len(VM): i % len(VM) + B] if B > 1 else VM[i % len(VM)][None]
            im, al, _ = rasterization(S["means"], S["quats"], S["scales"], S["opac"], S["sh"], vm, K[None].expand(len(vm), 3, 3),
                                      W, H, sh_degree=3, render_mode="RGB+ED", far_plane=200.0)
            return im

        ms = timeit(cam, a.n)
        rep[f"camera_{W}x{H}_ms"] = round(ms, 2)
        if W == 640:
            for B in (4, 2):
                try:
                    msb = timeit(lambda i: cam(i % (len(VM) - B), B), 10)
                    rep[f"camera_{W}x{H}_lot{B}_ms_par_image"] = round(msb / B, 2)
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    rep[f"camera_{W}x{H}_lot{B}"] = "mémoire GPU insuffisante"
            im = cam(len(VM) // 2)[0, ..., :3].clamp(0, 1).cpu().numpy()
            cv2.imwrite(f"{a.out}/camera.jpg", (im * 255).astype(np.uint8)[..., ::-1])

    c_mid = poses[len(poses) // 2][:3, 3]
    near = np.linalg.norm(G["means"][:, :2] - c_mid[:2], axis=1) < 40
    rep["n_gaussiennes_rayon40m"] = int(near.sum())

    # ------------------------------------------------ (2) LiDAR à balayage (OS1-64, 64 × 1024, 10 Hz)
    from gsplat import RowOffsetStructuredSpinningLidarModelParametersExt
    from gsplat.cuda._lidar import (RowOffsetStructuredSpinningLidarModelParameters, SpinningDirection,
                                    compute_angles_to_columns_map)
    from splatpipe.common.lidar_model import StructuredSpinningLidarModel
    from splatpipe.stage4.overfit_core import safe_lidar_tiling

    m = StructuredSpinningLidarModel.load_json(a.lidar_model)
    kw = m.gsplat_kwargs()
    params = RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=torch.from_numpy(kw["row_elevations_rad"]).to(dev),
        column_azimuths_rad=torch.from_numpy(kw["column_azimuths_rad"]).to(dev),
        row_azimuth_offsets_rad=torch.from_numpy(kw["row_azimuth_offsets_rad"]).to(dev),
        spinning_frequency_hz=kw["spinning_frequency_hz"], spinning_direction=SpinningDirection[kw["spinning_direction"]])
    tiling, _ = safe_lidar_tiling(params)
    lid = RowOffsetStructuredSpinningLidarModelParametersExt(params, compute_angles_to_columns_map(params).to(dev), tiling)
    gray = torch.full((len(G["means"]), 3), 0.5, device=dev)
    LV = []
    for T in poses:                                           # LiDAR 0,4 m au-dessus de la caméra, z haut, cap de la caméra
        fwd = T[:3, 2].copy()
        fwd[2] = 0
        fwd /= np.linalg.norm(fwd)
        R = np.stack([fwd, np.cross([0, 0, 1], fwd), [0, 0, 1]], 1)
        L = np.eye(4)
        L[:3, :3], L[:3, 3] = R, T[:3, 3] + [0, 0, 0.4]
        LV.append(np.linalg.inv(L))
    LV = torch.tensor(np.stack(LV), device=dev, dtype=torch.float32)

    def lidar(i):
        r, al, _ = gsplat.rasterization(
            S["means"], S["quats"], S["scales"], S["opac"], gray, LV[i % len(LV)][None], torch.eye(3, device=dev)[None],
            lid.n_columns, lid.n_rows, camera_model="lidar", lidar_coeffs=lid, with_ut=True, with_eval3d=True,
            packed=False, render_mode="RGB-Ed", global_z_order=False, far_plane=120.0,
            rolling_shutter=gsplat.RollingShutterType.ROLLING_LEFT_TO_RIGHT, viewmats_rs=LV[(i + 1) % len(LV)][None])
        return r, al

    rep["lidar_os1_64x1024_ms"] = round(timeit(lidar, a.n), 2)
    r, al = lidar(len(LV) // 2)
    d = r[0, ..., 3].cpu().numpy()
    alpha = al[0, ..., 0].cpu().numpy()
    rep["lidar_frac_retours_alpha>0.5"] = round(float((alpha > 0.5).mean()), 3)
    vis = np.clip(255 * (1 - np.log1p(d) / np.log1p(60)), 0, 255).astype(np.uint8) * (alpha > 0.5)
    cv2.imwrite(f"{a.out}/lidar_profondeur.png", cv2.resize(cv2.applyColorMap(vis, cv2.COLORMAP_TURBO),
                                                            (vis.shape[1], vis.shape[0] * 4), interpolation=cv2.INTER_NEAREST))
    rep["vram_pic_Go"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    # ------------------------------------------------ (3) physique MuJoCo, robot à roues sur hfield
    meshes = [read_mesh_ply(f"{t}/mesh.ply") for t in tiles]
    Hf, lo, res, holes = heightfield(meshes)
    rep["hfield"] = {"cellules": list(Hf.shape), "res_m": res, "frac_trous_bouches": round(holes, 3)}
    start = poses[0][:3, 3].copy()
    i0, j0 = ((start[:2] - lo) / res).astype(int)
    start[2] = Hf[i0, j0]
    fwd = poses[0][:3, 2]
    xml, (zmin, zmax) = robot_xml(Hf, lo, res, start, np.arctan2(fwd[1], fwd[0]))
    model = mujoco.MjModel.from_xml_string(xml)
    model.hfield_data[:] = ((Hf - zmin) / max(zmax - zmin, 1e-3)).T.ravel()   # MuJoCo : ligne = y
    data = mujoco.MjData(model)
    path = np.array([p[:3, 3] for p in poses])
    nsteps = 20000                                           # 40 s simulées
    t0 = time.perf_counter()
    for k in range(nsteps):
        if k % 10 == 0:                                      # suivi de chemin simple (poursuite)
            p = data.qpos[:3]
            q = data.qpos[3:7]
            yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            dd = np.linalg.norm(path[:, :2] - p[:2], axis=1)
            tgt = path[min(int(np.argmin(dd)) + 2, len(path) - 1)]
            err = np.arctan2(tgt[1] - p[1], tgt[0] - p[0]) - yaw
            err = (err + np.pi) % (2 * np.pi) - np.pi
            v, w = 1.0 / 0.13, 3.0 * err
            data.ctrl[:] = [v - w, v + w, v - w, v + w]
        mujoco.mj_step(model, data)
    wall = time.perf_counter() - t0
    rep["physique_pas_par_s"] = int(nsteps / wall)
    rep["physique_facteur_temps_reel"] = round(nsteps * model.opt.timestep / wall, 1)
    rep["physique_distance_parcourue_m"] = round(float(np.linalg.norm(data.qpos[:2] - start[:2])), 1)
    rep["physique_hauteur_chassis_sur_sol_m"] = round(float(data.qpos[2] - Hf[tuple(((data.qpos[:2] - lo) / res).astype(int))]), 2)

    # ------------------------------------------------ boucle complète : physique 500 Hz + caméra 640x400 20 Hz + LiDAR 10 Hz
    budget = rep["camera_640x400_ms"] * 20 + rep["lidar_os1_64x1024_ms"] * 10 + 1000 * 500 / rep["physique_pas_par_s"]
    rep["boucle_ms_par_seconde_simulee"] = round(budget, 1)
    rep["boucle_facteur_temps_reel_estime"] = round(1000 / budget, 1)
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(f"{a.out}/bench.json", "w"), indent=1)


if __name__ == "__main__":
    main()
