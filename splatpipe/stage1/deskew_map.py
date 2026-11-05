"""Étapes 1.4c et 1.5 — Nouvelle correction de distorsion de tous les tours, cartes, préparation de T1.2/T1.3.

1.4c `deskew_main` : chaque tour `lidar/hesai/<stamp>.npz` → `lidar/hesai_deskewed/<stamp>.npz` (repère `map` de GLIM) :
     p_map = T_map_base(t_point) · T_base_lidar · p_lidar avec `traj/spline.npz` ; retours < 1,5 m retirés ; deux échos.
     Tours dont > 2 % des points tombent hors de la trajectoire : non écrits (liste dans `index.json`).
1.5  `maps_main` : une carte par source, points d'entraînement uniquement (`splits.yaml`) :
     - `glim_final`  : tours corrigés ci-dessus ; `map/map_5cm.ply` (repère map) ; alignement map→world affiné par ICP
                       sur le TLS 5 cm (écrit dans `traj/final_to_world.yaml`, évaluation uniquement) ;
     - `glim_rigid`  : poses GLIM brutes (ancres calib_r4 au début du tour), une pose par tour (« avant ») ;
     - `glim_slerp`  : ancres GLIM brutes, interpolation par point SLERP (sans IMU) ;
     - `gt_oracle`   : GT (poses aux débuts de tours) mise en forme par la trajectoire IMU de GLIM, repère world ;
                       `map/map_5cm_gt_oracle.ply` ;
     - `vilens`      : nuages VILENS fournis (365 images clés, repère base, VIEWPOINT), alignés Umeyama + ICP ;
     - `vilens_noicp`: idem, alignement par trajectoire seul (écart de protocole).
     Pour chacune : statistiques de patchs 0,5 m en repère world (T1.2) et nuage 1 cm « protocole benchmark »
     (sous-échantillonnage 1 cm → rognage boîte GT → 1 cm → retrait de l'espace inconnu de l'octree TLS) sauvegardé dans
     `map/eval/<source>_1cm_filtered.npy` pour `eval_map.py::evaluate_main`.

Lancement :
  .venv/bin/modal run splatpipe/stage1/deskew_map.py::deskew_main
  .venv/bin/modal run splatpipe/stage1/deskew_map.py::maps_main
"""

from __future__ import annotations

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume
from splatpipe.stage1.images import map_image

app = modal.App("splatsim-stage1-deskew-map")
vol = data_volume()

SEQ = "2024-03-13-observatory-quarter-01"
RUN = "calib_r4"
OUT_SUB = "lidar/hesai_deskewed"
MAX_INVALID_FRAC = 0.02
VOXEL_MAP = 0.05
VOXEL_T12 = 0.5
RAW = f"{DATA_MOUNT}/raw/oxford_spires"
GT_5CM = f"{RAW}/ground_truth_map/observatory-quarter/merged-clouds-5cm.pcd"


# =============================================================================
# 1.4c — correction de distorsion
# =============================================================================


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=4, memory=16 * 1024, timeout=3600)
def deskew_chunk(stamps: list[int], seq: str = SEQ, spline_rel: str = "traj/spline.npz", out_sub: str = OUT_SUB) -> dict:
    import os
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from splatpipe.common.trajectory import Trajectory
    from splatpipe.stage1.maptools import MIN_RANGE_M, deskew_scan, load_scan, scene_root, sensors

    root = scene_root(DATA_MOUNT, seq)
    sens = sensors(root)
    T_bl = sens["T_base_lidar"]
    traj = Trajectory.load(f"{root}/{spline_rel}")
    os.makedirs(f"{root}/{out_sub}", exist_ok=True)

    def pose(t):
        T, v = traj.matrices([min(max(t, traj.t_start), traj.t_end)])
        return T[0] @ T_bl

    def work(stamp):
        sc = load_scan(f"{root}/lidar/hesai/{stamp}.npz")
        d = deskew_scan(sc, traj, T_bl)
        n_valid_range = d["n_points"] - d["n_near"]
        frac = d["n_invalid_time"] / max(n_valid_range, 1)
        rec = {"stamp": int(stamp), "n_points": d["n_points"], "n_near": d["n_near"], "n_invalid_time": d["n_invalid_time"]}
        if frac > MAX_INVALID_FRAC:
            rec["written"] = False
            return rec
        t0 = float(sc["t_start"])
        path = f"{root}/{out_sub}/{stamp}.npz"
        np.savez_compressed(
            path, xyz=d["xyz"].astype(np.float32), intensity=d["intensity"].astype(np.float32), ret=d["ret"].astype(np.int8),
            row=(63 - d["ring"].astype(np.int16)).astype(np.uint8), col=d["col"].astype(np.int16),
            t_rel=(d["t"] - t0).astype(np.float32), t_start=np.float64(t0), t_end=np.float64(sc["t_end"]),
            stamp_ns=np.int64(stamp), T_map_lidar_start=pose(t0), T_map_lidar_end=pose(float(sc["t_end"])),
            frame=np.array("map"), min_range_m=np.float32(MIN_RANGE_M), n_points_file=np.int64(d["n_points"]),
            n_near_removed=np.int64(d["n_near"]), n_invalid_time_removed=np.int64(d["n_invalid_time"]),
        )
        rec.update({"written": True, "n_out": int(len(d["xyz"])), "bytes": os.path.getsize(path)})
        return rec

    with ThreadPoolExecutor(4) as ex:
        recs = list(ex.map(work, stamps))
    vol.commit()
    return {"records": recs}


@app.local_entrypoint()
def deskew_main(seq: str = SEQ, n_chunks: int = 24, spline_rel: str = "traj/spline.npz", out_sub: str = OUT_SUB,
                tag: str = "", desc: str = ""):
    import json
    import time

    import numpy as np

    t0 = time.time()
    stamps = sorted(int(s) for s in list_stamps.remote(seq))
    chunks = [c.tolist() for c in np.array_split(np.array(stamps, dtype=np.int64), n_chunks)]
    recs = []
    for r in deskew_chunk.map(chunks, kwargs={"seq": seq, "spline_rel": spline_rel, "out_sub": out_sub}):
        recs += r["records"]
    written = [r for r in recs if r["written"]]
    summary = {
        "n_scans": len(recs), "n_written": len(written), "skipped": [r["stamp"] for r in recs if not r["written"]],
        "n_points_file": int(sum(r["n_points"] for r in recs)), "n_near_removed": int(sum(r["n_near"] for r in recs)),
        "n_invalid_time_removed_written": int(sum(r["n_invalid_time"] for r in written)),
        "n_points_out": int(sum(r["n_out"] for r in written)), "bytes": int(sum(r["bytes"] for r in written)),
        "wall_s": round(time.time() - t0, 1), "n_chunks": n_chunks, "spline": spline_rel, "out_sub": out_sub,
    }
    write_index.remote(summary, seq, out_sub, desc)
    with open(f"logs/stage1/map/deskew{tag}.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "skipped"} | {"n_skipped": len(summary["skipped"])}, indent=1))


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=1, memory=2048, timeout=600)
def list_stamps(seq: str = SEQ) -> list[int]:
    from splatpipe.stage1.maptools import list_scans, scene_root

    stamps, _ = list_scans(scene_root(DATA_MOUNT, seq))
    return [int(s) for s in stamps]


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=1, memory=2048, timeout=600)
def write_index(summary: dict, seq: str = SEQ, out_sub: str = OUT_SUB, desc: str = ""):
    import json

    from splatpipe.stage1.maptools import scene_root

    root = scene_root(DATA_MOUNT, seq)
    doc = dict(summary)
    doc["description"] = desc or (
        "Tours corrigés de la distorsion avec traj/spline.npz (GLIM calib_r4, forme IMU GLIM, ancres lissées σ=3 s). "
        "xyz float32 repère map (GLIM) ; intensity ; ret (0 premier écho, 1 dernier écho distinct) ; row = 63 − ring ; "
        "col ; t_rel = pt_t − t_start ; T_map_lidar_start/end ; retours < 1,5 m retirés.")
    with open(f"{root}/{out_sub}/index.json", "w") as f:
        json.dump(doc, f, indent=1)
    vol.commit()


# =============================================================================
# 1.5 — cartes
# =============================================================================


def map_5cm(xyz, intensity, ret, scan_id, origins, voxel=VOXEL_MAP, k_normals=16):
    """Carte voxelisée : centroïde, normale (ACP des k plus proches centroïdes, orientée vers les capteurs),
    nb de points, nb de tours distincts, intensité moyenne, part de derniers échos, planéité σ3/σ1."""
    import numpy as np
    from scipy.spatial import cKDTree

    from splatpipe.stage1.maptools import knn_normals, pack_ijk

    ijk = np.floor(xyz / voxel).astype(np.int64)
    keys = pack_ijk(ijk)
    del ijk
    order = np.lexsort((scan_id, keys))
    ks, ss = keys[order], scan_id[order]
    new = np.r_[True, ks[1:] != ks[:-1]]
    starts = np.flatnonzero(new)
    n = np.diff(np.r_[starts, len(ks)])
    new_scan = new | np.r_[True, ss[1:] != ss[:-1]]
    n_scans = np.add.reduceat(new_scan.astype(np.int64), starts)
    del ks, ss, new, new_scan
    p = xyz[order]
    cen = np.add.reduceat(p, starts, axis=0) / n[:, None]
    view = origins[scan_id[order]] - p
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    view = np.add.reduceat(view, starts, axis=0)
    del p
    inten = np.add.reduceat(intensity[order].astype(np.float64), starts) / n if intensity is not None else np.zeros(len(n))
    ret1 = np.add.reduceat((ret[order] == 1).astype(np.float64), starts) / n if ret is not None else np.zeros(len(n))
    tree = cKDTree(cen, balanced_tree=False, compact_nodes=False)
    nrm, flat = knn_normals(cen, tree, k=k_normals)
    flip = np.sum(nrm * view, axis=1) < 0
    nrm[flip] *= -1
    return {"x": cen[:, 0].astype(np.float32), "y": cen[:, 1].astype(np.float32), "z": cen[:, 2].astype(np.float32),
            "nx": nrm[:, 0], "ny": nrm[:, 1], "nz": nrm[:, 2], "n_points": n.astype(np.uint32), "n_scans": n_scans.astype(np.uint32),
            "intensity": inten.astype(np.float32), "last_echo_frac": ret1.astype(np.float32), "flatness": flat}


def _gt_anchor_trajectory(root, sens, stamps_all):
    """GT aux débuts de tours (±2 ms) mise en forme par la trajectoire IMU de GLIM → Trajectory (world)."""
    import numpy as np

    from splatpipe.common.geometry import read_tum
    from splatpipe.stage1.maptools import build_trajectory, load_glim_run

    t, p, q = read_tum(f"{root}/traj/gt.tum")
    o = np.argsort(t)
    t, p, q = t[o], p[o], q[o]
    ts = np.asarray(stamps_all, dtype=np.int64) / 1e9
    j = np.clip(np.searchsorted(t, ts), 1, len(t) - 1)
    j = np.where(np.abs(t[j - 1] - ts) < np.abs(t[j] - ts), j - 1, j)
    ok = np.abs(t[j] - ts) < 0.002
    info = {"n_gt": int(len(t)), "n_scans": int(len(ts)), "n_scan_start_matched": int(ok.sum())}
    if ok.mean() > 0.9:
        sel = np.unique(j[ok])
        info["anchors"] = "poses GT aux débuts de tours"
    else:
        sel = np.arange(len(t))
        info["anchors"] = "toutes les poses GT"
    g = load_glim_run(root, RUN, sens["T_base_lidar"])
    tr = build_trajectory((t[sel], p[sel], q[sel]), "glimimu", "world", shape={"glimimu": g["imu_rate"]},
                          meta={"source": "traj/gt.tum", **info})
    return tr, info


def _query_octree(points_world, T_e57_world, tls_dir: str, workers: int = 8) -> tuple:
    """Masque « connu » : OU sur les 19 octrees de station (points passés dans le repère des E57)."""
    import glob
    import os
    import subprocess
    import time

    import numpy as np

    t0 = time.time()
    p = points_world @ T_e57_world[:3, :3].T + T_e57_world[:3, 3]
    p.astype(np.float32).tofile("/tmp/q.bin")
    trees = sorted(glob.glob(f"{tls_dir}/octree/*.bt"))
    mask = np.zeros(len(points_world), dtype=bool)
    env = dict(os.environ, OMP_NUM_THREADS="2")
    for a in range(0, len(trees), workers):
        procs = [(tb, subprocess.Popen(["octomap_tools", "query", tb, "/tmp/q.bin", f"/tmp/m_{i}.bin"], env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE))
                 for i, tb in enumerate(trees[a:a + workers], start=a)]
        for i, (tb, pr) in enumerate(procs, start=a):
            out, err = pr.communicate()
            if pr.returncode != 0:
                raise RuntimeError(f"octree query {tb}: {err.decode()[-500:]}")
            mask |= np.fromfile(f"/tmp/m_{i}.bin", dtype=np.uint8).astype(bool)
            os.remove(f"/tmp/m_{i}.bin")
    os.remove("/tmp/q.bin")
    return mask, {"n_trees": len(trees), "query_s": round(time.time() - t0, 1)}


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=16, memory=128 * 1024, timeout=3 * 3600)
def build_map(source: str, T_provided_e57: list, seq: str = SEQ, suffix: str = "") -> dict:
    """`suffix = ""` : chemins historiques ; sinon (p. ex. "_calib", 1.7a) sorties parallèles :
    reports/stage1_map/t12/{source}{suffix}.npz, map/map_5cm{suffix}.ply, map/eval/{source}{suffix}_1cm_filtered.npy,
    traj/spline{suffix}.npz, traj/final_to_world{suffix}.yaml (ICP affiné écrit dedans)."""
    import glob
    import json
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import yaml
    from scipy.spatial import cKDTree

    from splatpipe.common.geometry import inv, read_tum, rot_to_rotvec
    from splatpipe.stage1.maptools import (
        crop_aabb, icp_point_to_plane, knn_normals, lidar_splits, list_scans, load_glim_run, o3d_voxel_down_sample,
        pcd_viewpoint, read_pcd, scene_root, sensors, summarize, trajectory_alignment, two_grid_stats, write_ply,
    )
    from splatpipe.stage1.select_run import accumulate, make_glim_trajectory

    T0 = time.time()
    timing = {}
    out_name = source + suffix
    root = scene_root(DATA_MOUNT, seq)
    sens = sensors(root)
    stamps, paths = list_scans(root)
    train = set(lidar_splits(root)["train"].tolist())
    idx = [i for i, s in enumerate(stamps) if int(s) in train]
    info = {"source": source}
    if source.endswith("_kf"):
        # Même densité que la carte VILENS : le tour le plus proche de chaque image clé VILENS (tous splits confondus)
        tv = np.array(sorted(int(a) + int(b) * 1e-9 for a, b in (os.path.basename(f)[6:-4].split("_") for f in
                      glob.glob(f"{RAW}/sequences/{seq}/processed/vilens-slam/undist-clouds/*.pcd"))))
        ts = stamps / 1e9
        j = np.clip(np.searchsorted(ts, tv), 1, len(ts) - 1)
        j = np.where(np.abs(ts[j - 1] - tv) < np.abs(ts[j] - tv), j - 1, j)
        idx = sorted(set(j.tolist()))
        info["kf_match_dt_s"] = {"median": float(np.median(np.abs(ts[j] - tv))), "max": float(np.max(np.abs(ts[j] - tv)))}
        source = source[:-3]
        info["subset"] = "365 images clés VILENS"
    yaml_path = f"{root}/traj/final_to_world{suffix}.yaml"
    with open(yaml_path) as f:
        f2w = yaml.safe_load(f)
    tls_info = json.load(open(f"{root}/map/tls/tls_info.json"))
    gt_lo = np.array(tls_info["gt_check"]["gt_1cm_bbox"][0])
    gt_hi = np.array(tls_info["gt_check"]["gt_1cm_bbox"][1])

    def tls_target():
        c5, _ = read_pcd(GT_5CM)
        g5 = np.stack([c5["x"], c5["y"], c5["z"]], 1).astype(np.float64)
        tree = cKDTree(g5, balanced_tree=False, compact_nodes=False)
        nrm, _ = knn_normals(g5, tree, k=12)
        return tree, nrm

    def icp_refine(pts_frame, T_init):
        """ICP du nuage (sous-échantillonné 10 cm, dans la boîte GT) sur le TLS 5 cm."""
        sub = o3d_voxel_down_sample(pts_frame, 0.10)
        w = sub @ T_init[:3, :3].T + T_init[:3, 3]
        sub = sub[crop_aabb(w, gt_lo, gt_hi)]
        rng = np.random.default_rng(0)
        if len(sub) > 1_500_000:
            sub = sub[rng.choice(len(sub), 1_500_000, replace=False)]
        tree, nrm = tls_target()
        T, hist = icp_point_to_plane(sub, tree, nrm, T_init, schedule=(0.5, 0.25, 0.1, 0.05), iters=20)
        dT = T @ inv(T_init)
        return T, {"delta_rot_deg": float(np.degrees(np.linalg.norm(rot_to_rotvec(dT[:3, :3])))),
                   "delta_trans_m": dT[:3, 3].tolist(), "final": hist[-1], "n_src": int(len(sub)), "n_iters": len(hist)}

    frame_pts = None  # points dans le repère de la source (pour la carte 5 cm)
    intensity = ret = None
    if source == "glim_final":
        # Recalcul à la volée depuis les tours bruts avec traj/spline{suffix}.npz (identique à lidar/hesai_deskewed{suffix},
        # contrôlé sur quelques tours) : la relecture des fichiers fraîchement écrits sur le Volume était trop lente (~3 Mo/s).
        from splatpipe.common.trajectory import Trajectory

        traj = Trajectory.load(f"{root}/traj/spline{suffix}.npz")
        acc = accumulate(root, idx, paths, stamps, traj, sens, threads=16, keep_attrs=True)
        frame_pts, tpts, scan_id, origins = acc["xyz"], acc["t"], acc["scan_id"], acc["origins"]
        intensity, ret = acc["intensity"], acc["ret"]
        res = acc["stamps_used"]
        chk = []
        for st_ in res[:: max(1, len(res) // 3)][:3]:
            k = res.index(st_)
            with np.load(f"{root}/{OUT_SUB}{suffix}/{st_}.npz") as z:
                ref = z["xyz"]
            chk.append(float(np.abs(frame_pts[scan_id == k].astype(np.float32) - ref).max()))
        info["check_vs_hesai_deskewed_max_abs_m"] = chk
        info["n_scans"] = len(res)
        timing["load_s"] = round(time.time() - T0, 1)
        T_init = np.array(f2w["T_world_map"])
        T_w, icp = icp_refine(frame_pts, T_init)
        if source == "glim_final":
            info["icp"] = icp
            f2w["icp_refined"] = {"T_world_map": T_w.tolist(), **icp,
                                  "method": "ICP point→plan (Huber) de la carte 10 cm (boîte GT) sur merged-clouds-5cm.pcd, "
                                            "seuils 0,5/0,25/0,1/0,05 m, départ = T_world_map (trajectoire)"}
            with open(yaml_path, "w") as f:
                f.write("# Alignement du repère carte GLIM sur world — POUR L'ÉVALUATION UNIQUEMENT (utilise la GT).\n")
                yaml.safe_dump(f2w, f, sort_keys=False, allow_unicode=True)
            vol.commit()
            timing["icp_s"] = round(time.time() - T0 - timing["load_s"], 1)
    elif source in ("glim_rigid", "glim_slerp"):
        if not f2w.get("icp_refined"):
            raise RuntimeError("lancer glim_final d'abord (ICP)")
        T_w = np.array(f2w["icp_refined"]["T_world_map"])
        traj, g = make_glim_trajectory(root, RUN, "slerp", 0.0, sens)
        acc = accumulate(root, idx, paths, stamps, traj, sens, rigid_anchors=g["anchors"] if source == "glim_rigid" else None,
                         threads=16, keep_attrs=True)
        frame_pts, tpts, scan_id, origins = acc["xyz"], acc["t"], acc["scan_id"], acc["origins"]
        intensity, ret = acc["intensity"], acc["ret"]
        info["n_scans"] = acc["n_scans_used"]
    elif source == "gt_oracle":
        traj, ginfo = _gt_anchor_trajectory(root, sens, stamps)
        info["gt_traj"] = ginfo
        acc = accumulate(root, idx, paths, stamps, traj, sens, threads=16, keep_attrs=True)
        frame_pts, tpts, scan_id, origins = acc["xyz"], acc["t"], acc["scan_id"], acc["origins"]
        intensity, ret = acc["intensity"], acc["ret"]
        info["n_scans"] = acc["n_scans_used"]
        T_w = np.eye(4)
        # Pour information : correction ICP résiduelle de la carte GT vs TLS (non appliquée)
        _, icp = icp_refine(frame_pts, np.eye(4))
        info["icp_residual_not_applied"] = icp
    elif source in ("vilens", "vilens_noicp"):
        files = sorted(glob.glob(f"{RAW}/sequences/{seq}/processed/vilens-slam/undist-clouds/*.pcd"))

        def load_v(fp):
            cols, h = read_pcd(fp)
            VP = pcd_viewpoint(h)
            p = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
            sec, nsec = os.path.basename(fp)[6:-4].split("_")
            return p @ VP[:3, :3].T + VP[:3, 3], VP, int(sec) + int(nsec) * 1e-9

        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(load_v, files))
        frame_pts = np.concatenate([r[0] for r in res])
        scan_id = np.concatenate([np.full(len(r[0]), k, np.int32) for k, r in enumerate(res)])
        tv = np.array([r[2] for r in res])
        tpts = tv[scan_id]
        VPs = np.array([r[1] for r in res])
        origins = VPs[:, :3, 3]
        info["n_scans"] = len(res)
        rng_v = np.linalg.norm(frame_pts - origins[scan_id], axis=1)
        info["vilens_range_m"] = summarize(rng_v)
        info["vilens_frac_lt_1p5m"] = float((rng_v < 1.5).mean())
        from splatpipe.common.geometry import rot_to_quat

        gt = read_tum(f"{root}/traj/gt.tum")
        T_init, ainfo = trajectory_alignment((tv, VPs[:, :3, 3], rot_to_quat(VPs[:, :3, :3])), gt)
        info["traj_alignment"] = ainfo
        if source == "vilens":
            T_w, icp = icp_refine(frame_pts, T_init)
            info["icp"] = icp
        else:
            T_w = T_init
        info["T_world_vilens"] = T_w.tolist()
    else:
        raise ValueError(source)
    timing["points_s"] = round(time.time() - T0, 1)
    info["n_points"] = int(len(frame_pts))
    world = frame_pts @ T_w[:3, :3].T + T_w[:3, 3]

    # ---- T1.2 : patchs 0,5 m en repère world
    st = two_grid_stats(world, VOXEL_T12, t=tpts, min_pts=30)
    os.makedirs(f"{root}/reports/stage1_map/t12", exist_ok=True)
    np.savez(f"{root}/reports/stage1_map/t12/{out_name}.npz", **st)
    info["t12_own_planar_sigma3"] = summarize(st["sigma"][st["planar"], 2])
    timing["t12_s"] = round(time.time() - T0 - timing["points_s"], 1)

    # ---- carte 5 cm (PLY) pour la carte finale (repère map) et l'oracle GT (world)
    if source in ("glim_final", "gt_oracle"):
        t1 = time.time()
        m = map_5cm(frame_pts, intensity, ret, scan_id, origins)
        os.makedirs(f"{root}/map", exist_ok=True)
        ply = f"{root}/map/map_5cm{suffix}.ply" if source == "glim_final" else f"{root}/map/map_5cm_gt_oracle{suffix}.ply"
        frame_name = (f"map (repère carte GLIM ; world via traj/final_to_world{suffix}.yaml)" if source == "glim_final" else "world")
        comments = [f"SplatSim etape 1.5 source={out_name} frame={frame_name}", "voxel 0.05 m, tours train uniquement, retours < 1.5 m retires",
                    "n_points = points bruts, n_scans = tours distincts, normales ACP k=16 orientees vers le capteur",
                    "flatness = sqrt(l_min/l_max) des 16 voisins, last_echo_frac = part des derniers echos (ret=1)"]
        info["ply"] = {"path": ply, "bytes": write_ply(ply, m, comments=comments), "n_voxels": int(len(m["x"])),
                       "n_scans_median": float(np.median(m["n_scans"])), "frac_single_scan": float(np.mean(m["n_scans"] == 1))}
        vol.commit()
        timing["map5cm_s"] = round(time.time() - t1, 1)

    # ---- T1.3 : nuage 1 cm selon le protocole
    t1 = time.time()
    d1 = o3d_voxel_down_sample(world, 0.01)
    n_merged = len(d1)
    d1 = d1[crop_aabb(d1, gt_lo, gt_hi)]
    n_crop = len(d1)
    d2 = o3d_voxel_down_sample(d1, 0.01)
    del d1
    T_e57_world = inv(np.array(T_provided_e57))
    known, qinfo = _query_octree(d2, T_e57_world, f"{root}/map/tls")
    os.makedirs(f"{root}/map/eval", exist_ok=True)
    np.save(f"{root}/map/eval/{out_name}_1cm_filtered.npy", d2[known].astype(np.float32))
    if source.startswith("vilens"):
        np.save(f"{root}/map/eval/{source}_1cm_nofilter.npy", d2.astype(np.float32))
    vol.commit()
    info["t13_cloud"] = {"n_merged_1cm": int(n_merged), "n_cropped": int(n_crop), "n_downsampled": int(len(d2)),
                         "n_known": int(known.sum()), "frac_known": float(known.mean()), **qinfo}
    timing["t13_prep_s"] = round(time.time() - t1, 1)
    timing["total_s"] = round(time.time() - T0, 1)
    info["timing"] = timing
    info["T_world_frame"] = T_w.tolist()
    return info


@app.local_entrypoint()
def maps_main(seq: str = SEQ, sources: str = "glim_final,glim_rigid,glim_slerp,gt_oracle,vilens,vilens_noicp",
              suffix: str = ""):
    import json
    import os

    T = json.load(open("logs/stage1/map/tls_frame_check.json"))["T_provided_e57"]
    srcs = sources.split(",")
    path = "logs/stage1/map/maps.json" if not suffix else f"logs/stage1/map/maps{suffix}.json"
    out = json.load(open(path)) if os.path.exists(path) else {}
    # Phase 1 en parallèle ; glim_rigid/glim_slerp ensuite (ils réutilisent l'ICP de glim_final)
    phases = [[s for s in srcs if s not in ("glim_rigid", "glim_slerp")], [s for s in srcs if s in ("glim_rigid", "glim_slerp")]]
    for ph in phases:
        if not ph:
            continue
        for s, r in zip(ph, build_map.map(ph, kwargs={"T_provided_e57": T, "seq": seq, "suffix": suffix}, return_exceptions=True)):
            out[s] = r if isinstance(r, dict) else {"error": str(r)}
            print(s, "ERREUR " + out[s]["error"][-800:] if "error" in out[s] else out[s]["timing"], flush=True)
            json.dump(out, open(path, "w"), indent=1)
