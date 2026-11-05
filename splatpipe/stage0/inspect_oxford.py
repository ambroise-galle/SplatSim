"""Étape 0.2 — Inspection des formats réels d'une séquence Oxford Spires téléchargée.

Lancement :
  .venv/bin/modal run splatpipe/stage0/inspect_oxford.py --seq 2024-03-13-observatory-quarter-01
"""

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume

app = modal.App("splatsim-stage0-inspect")
vol = data_volume()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy", "pypcd4", "pyyaml")
    .add_local_python_source("splatpipe")
)

ROOT = f"{DATA_MOUNT}/raw/oxford_spires"


def _stamp(name: str) -> float:
    base = name.rsplit(".", 1)[0]
    if base.startswith("cloud_"):
        s, ns = base[len("cloud_"):].split("_")
        return int(s) + int(ns) * 1e-9
    return float(base)


def _series(stamps):
    import numpy as np

    t = np.sort(np.asarray(stamps, dtype=np.float64))
    if len(t) < 2:
        return {"n": int(len(t))}
    dt = np.diff(t)
    return {
        "n": int(len(t)),
        "t0": float(t[0]),
        "t1": float(t[-1]),
        "duration_s": round(float(t[-1] - t[0]), 3),
        "rate_hz": round(float(1.0 / np.median(dt)), 3),
        "dt_max_s": round(float(dt.max()), 4),
        "gaps_gt_2periods": int((dt > 2 * np.median(dt)).sum()),
    }


def _pcd_summary(path: str) -> dict:
    import numpy as np
    from pypcd4 import PointCloud

    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line.startswith("DATA"):
                break
    pc = PointCloud.from_path(path)
    arr = pc.numpy()
    fields = list(pc.fields)
    out = {"header": header, "fields": fields, "n_points": int(arr.shape[0])}
    stats = {}
    for i, name in enumerate(fields):
        col = arr[:, i].astype(np.float64)
        finite = np.isfinite(col)
        stats[name] = {
            "min": float(col[finite].min()) if finite.any() else None,
            "max": float(col[finite].max()) if finite.any() else None,
            "n_unique": int(len(np.unique(col[finite][:200000]))) if finite.any() else 0,
            "nan": int((~finite).sum()),
        }
    out["stats"] = stats
    return out


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=4, memory=16384, timeout=3600)
def inspect(seq: str) -> dict:
    import os

    import numpy as np

    seq_dir = f"{ROOT}/sequences/{seq}"
    report = {"seq": seq, "tree": {}}

    for dirpath, dirnames, files in os.walk(seq_dir):
        rel = dirpath.replace(seq_dir, "") or "/"
        exts = {}
        for f in files:
            e = f.rsplit(".", 1)[-1] if "." in f else ""
            exts[e] = exts.get(e, 0) + 1
        report["tree"][rel] = exts

    raw = f"{seq_dir}/raw"
    cams = sorted(d for d in os.listdir(raw) if d.startswith("cam") and os.path.isdir(f"{raw}/{d}"))
    report["cameras"] = {c: _series([_stamp(f) for f in os.listdir(f"{raw}/{c}") if f.endswith(".jpg")]) for c in cams}
    if len(cams) == 3:
        sets = [set(round(_stamp(f), 4) for f in os.listdir(f"{raw}/{c}") if f.endswith(".jpg")) for c in cams]
        report["cameras_common_stamps"] = len(sets[0] & sets[1] & sets[2])

    lidar_dir = f"{raw}/lidar-clouds"
    lidar_files = sorted(f for f in os.listdir(lidar_dir) if f.endswith(".pcd")) if os.path.isdir(lidar_dir) else []
    report["lidar_raw"] = _series([_stamp(f) for f in lidar_files])
    if lidar_files:
        mid = lidar_files[len(lidar_files) // 2]
        report["lidar_raw"]["example_file"] = mid
        report["lidar_raw"]["example"] = _pcd_summary(f"{lidar_dir}/{mid}")

    undist_dir = f"{seq_dir}/processed/vilens-slam/undist-clouds"
    if os.path.isdir(undist_dir):
        uf = sorted(f for f in os.listdir(undist_dir) if f.endswith(".pcd"))
        report["lidar_vilens_undist"] = _series([_stamp(f) for f in uf])
        if uf:
            report["lidar_vilens_undist"]["example"] = _pcd_summary(f"{undist_dir}/{uf[len(uf) // 2]}")

    imu_path = f"{raw}/imu.csv"
    with open(imu_path) as f:
        head = [next(f).strip() for _ in range(3)]
    imu = np.genfromtxt(imu_path, delimiter=",", skip_header=1)
    report["imu"] = {"head": head, "shape": list(imu.shape)}
    report["imu"]["series_col0"] = _series(imu[:, 0] if imu[0, 0] < 1e12 else imu[:, 0] * 1e-9)

    trajs = {}
    tdir = f"{seq_dir}/processed/trajectory"
    for f in sorted(os.listdir(tdir)):
        a = np.loadtxt(f"{tdir}/{f}", comments="#")
        path_len = float(np.linalg.norm(np.diff(a[:, 1:4], axis=0), axis=1).sum())
        trajs[f] = {**_series(a[:, 0]), "path_length_m": round(path_len, 1),
                    "xyz_min": a[:, 1:4].min(0).round(2).tolist(), "xyz_max": a[:, 1:4].max(0).round(2).tolist()}
    report["trajectories"] = trajs
    report["vilens_slam_poses_head"] = open(f"{seq_dir}/processed/vilens-slam/slam-poses.csv").read().splitlines()[:3]

    gt_dir = [d for d in os.listdir(f"{ROOT}/ground_truth_map")]
    report["gt_map"] = {}
    for d in gt_dir:
        for f in os.listdir(f"{ROOT}/ground_truth_map/{d}"):
            if f.endswith(".pcd"):
                s = _pcd_summary(f"{ROOT}/ground_truth_map/{d}/{f}")
                report["gt_map"][f"{d}/{f}"] = {"header": s["header"], "n_points": s["n_points"],
                                                "bbox": {k: [v["min"], v["max"]] for k, v in s["stats"].items() if k in "xyz"}}
    report["calibration_files"] = sorted(os.listdir(f"{ROOT}/calibration"))
    return report


@app.local_entrypoint()
def main(seq: str = "2024-03-13-observatory-quarter-01"):
    import json
    import pathlib

    rep = inspect.remote(seq)
    out = pathlib.Path("logs/stage0/inspect.json")
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2)[:20000])
