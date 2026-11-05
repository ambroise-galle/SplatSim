"""Étapes 1.1–1.2 — GLIM (odométrie LiDAR-inertielle + optimisation globale) sans affichage sur Modal.

Commandes (depuis la racine du projet) :
  # contrôles : GPU, installation GLIM, lecture du bag par rosbag2 (Jazzy) et par `rosbags`, champs du nuage
  .venv/bin/modal run splatpipe/stage1/glim_run.py::probe_main
  # exécution complète (variant = calib | imu_reest), écrit traj/glim/<variant>/ et traj/glim.tum
  .venv/bin/modal run splatpipe/stage1/glim_run.py::main --variant calib

Bag : `/data/raw/oxford_spires/sequences/<seq>/raw/ros2bag/<bag>/` (téléchargé par `download_bag.py`), copié sur
le disque local du conteneur avant lecture (évite les accès aléatoires SQLite sur le Volume).
"""

from __future__ import annotations

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume
from splatpipe.stage1.images import GLIM_SRC_CONFIG, ROS_ENV, glim_image

app = modal.App("splatsim-stage1-glim")
vol = data_volume()
image = glim_image()

DATASET = "oxford_spires"
SEQ = "2024-03-13-observatory-quarter-01"
BAG_NAME = "1710338090_2024-03-13-13-54-51"
IMU_TOPIC = "/alphasense_driver_ros/imu"
POINTS_TOPIC = "/hesai/pandar"


def _bag_dir(seq: str, bag: str) -> str:
    return f"{DATA_MOUNT}/raw/oxford_spires/sequences/{seq}/raw/ros2bag/{bag}"


def _sh(cmd: str, timeout: float | None = None, check: bool = False) -> dict:
    import subprocess

    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    out = {"cmd": cmd, "rc": r.returncode, "stdout": r.stdout[-6000:], "stderr": r.stderr[-6000:]}
    if check and r.returncode:
        raise RuntimeError(f"échec ({r.returncode}) : {cmd}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return out


@app.function(image=image, volumes={DATA_MOUNT: vol}, gpu="L4", cpu=4, memory=16384, timeout=1800)
def probe(seq: str = SEQ, bag: str = BAG_NAME) -> dict:
    """Contrôles préalables (aucune écriture sur le Volume)."""
    import os

    import numpy as np

    res = {}
    res["nvidia_smi"] = _sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
    res["glim_bins"] = _sh(f"{ROS_ENV} && ls /root/ros2_ws/install/glim_ros/lib/glim_ros/ && ls {GLIM_SRC_CONFIG}")
    res["ldd_gpu"] = _sh(
        "ls /root/ros2_ws/install/glim/lib/ | head -50; ldd /root/ros2_ws/install/glim/lib/libodometry_estimation_gpu.so "
        "| grep -i 'cuda\\|not found' | head"
    )
    res["git"] = _sh("cd /root/ros2_ws/src/glim && git log -1 --format='%H %cd' ; cd /root/ros2_ws/src/glim_ros2 && "
                     "git log -1 --format='%H %cd'; cd /root/gtsam_points && git log -1 --format='%H %cd'")
    d = _bag_dir(seq, bag)
    res["bag_ls"] = _sh(f"ls -la {d}")
    res["ros2_bag_info"] = _sh(f"{ROS_ENV} && ros2 bag info {d}", timeout=600)

    # Lecture directe (rosbags) : champs et temps des premiers nuages, cadence IMU.
    from rosbags.highlevel import AnyReader
    from pathlib import Path

    info = {"clouds": [], "imu": []}
    with AnyReader([Path(d)]) as reader:
        conns = [c for c in reader.connections if c.topic == POINTS_TOPIC]
        for k, (conn, t_rec, raw) in enumerate(reader.messages(connections=conns)):
            msg = reader.deserialize(raw, conn.msgtype)
            fields = [(f.name, int(f.offset), int(f.datatype), int(f.count)) for f in msg.fields]
            dt = np.dtype(
                {
                    "names": [f.name for f in msg.fields],
                    "formats": [{7: "<f4", 8: "<f8", 4: "<u2", 2: "<u1", 6: "<u4", 5: "<i4"}[f.datatype] for f in msg.fields],
                    "offsets": [f.offset for f in msg.fields],
                    "itemsize": msg.point_step,
                }
            )
            arr = np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)
            ts = arr["timestamp"]
            xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1)
            fin = np.isfinite(xyz).all(1)
            rng = np.linalg.norm(np.where(fin[:, None], xyz, 0), axis=1)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            info["clouds"].append(
                {
                    "recv_t": t_rec / 1e9,
                    "header_stamp": stamp,
                    "frame_id": msg.header.frame_id,
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "point_step": int(msg.point_step),
                    "fields": fields,
                    "ts_min": float(ts.min()),
                    "ts_max": float(ts.max()),
                    "ts_zero": int((ts == 0).sum()),
                    "ts_min_minus_stamp": float(ts.min() - stamp),
                    "ts_max_minus_stamp": float(ts.max() - stamp),
                    "ts_sorted": bool(np.all(np.diff(ts) >= 0)),
                    "n_nonfinite": int((~fin).sum()),
                    "n_zero_range": int((rng[fin] < 0.01).sum()),
                    "range_p50": float(np.median(rng[fin])),
                }
            )
            if k >= 4:
                break
        conns = [c for c in reader.connections if c.topic == IMU_TOPIC]
        for k, (conn, t_rec, raw) in enumerate(reader.messages(connections=conns)):
            msg = reader.deserialize(raw, conn.msgtype)
            a, w = msg.linear_acceleration, msg.angular_velocity
            info["imu"].append(
                {
                    "recv_t": t_rec / 1e9,
                    "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                    "frame_id": msg.header.frame_id,
                    "acc": [a.x, a.y, a.z],
                    "gyro": [w.x, w.y, w.z],
                }
            )
            if k >= 4:
                break
    res["bag_content"] = info
    res["disk"] = _sh("df -h /tmp / | tail -3; nproc; free -g | head -2")
    return res


@app.local_entrypoint()
def probe_main():
    import json

    res = probe.remote()
    with open("logs/stage1/glim_probe.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str)[:20000])


# =============================================================================
# Configuration GLIM
# =============================================================================

VARIANTS = ("calib", "imu_reest")
BAG_LOCAL_ROOT = "/tmp/bag"
CONFIG_DIR = "/tmp/glim_config"
DUMP_DIR = "/tmp/glim_dump"
GLIM_BIN = "/root/ros2_ws/install/glim_ros/lib/glim_ros/glim_rosbag"

# Réglages propres à la séquence (écarts aux valeurs par défaut de GLIM, justifiés dans la note stage1-glim.md).
PREPROCESS_OVERRIDES = {
    # Opérateur dans le champ (étape 0.4) : 10,6 % des retours < 1 m, secteur arrière fixe.
    "distance_near_thresh": 1.5,
    # Hesai QT64 : portée utile ~60 m.
    "distance_far_thresh": 80.0,
    "num_threads": 4,
}
ODOMETRY_OVERRIDES = {"num_threads": 4}


def strip_json_comments(text: str) -> str:
    """Retire les commentaires `//` et `/* */` d'un JSON de GLIM (hors chaînes de caractères)."""
    out, i, n = [], 0, len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
        elif c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def load_glim_config(name: str) -> dict:
    import json
    import os

    with open(os.path.join(GLIM_SRC_CONFIG, name)) as f:
        return json.loads(strip_json_comments(f.read()))


def tum_list(T) -> list[float]:
    """4×4 → [x, y, z, qx, qy, qz, qw] (format TUM de GLIM)."""
    from splatpipe.common.geometry import se3_to_tq

    t, q = se3_to_tq(T)
    return [float(v) for v in list(t) + list(q)]


def extrinsics_for_variant(variant: str, scene_root: str) -> dict:
    """`T_lidar_imu = inv(T_base_lidar) · T_base_imu` pour la variante, avec la provenance des valeurs."""
    import hashlib
    import json

    import numpy as np
    import yaml

    from splatpipe.common.geometry import compose, inv, is_rotation, rotation_angle_between

    sensors_path = f"{scene_root}/sensors.yaml"
    raw = open(sensors_path, "rb").read()
    sensors = yaml.safe_load(raw)
    T_base_lidar = np.asarray(sensors["lidars"]["hesai_qt64"]["T_base_lidar"], dtype=np.float64)
    T_base_imu_cal = np.asarray(sensors["imu"]["T_base_imu"], dtype=np.float64)
    info = {
        "sensors_yaml": sensors_path,
        "sensors_yaml_sha256": hashlib.sha256(raw).hexdigest(),
        "sensors_yaml_has_calibration_corrections": "calibration_corrections" in sensors,
        "calibration_corrections": sensors.get("calibration_corrections"),
        "T_base_lidar": T_base_lidar.tolist(),
        "T_base_imu_sensors_yaml": T_base_imu_cal.tolist(),
    }
    if variant == "calib":
        T_base_imu = T_base_imu_cal
        info["T_base_imu_source"] = "sensors.yaml:imu.T_base_imu (chaîne DiffCal + Kalibr fournie)"
    elif variant == "imu_reest":
        with open(f"{scene_root}/reports/stage0_ingest.json") as f:
            rep = json.load(f)
        R_est = np.asarray(rep["T0.5"]["d_gyro_vs_gt"]["R_base_imu_estimated"], dtype=np.float64)
        U, _, Vt = np.linalg.svd(R_est)
        R_est = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        T_base_imu = T_base_imu_cal.copy()
        T_base_imu[:3, :3] = R_est
        info["T_base_imu_source"] = (
            "rotation : reports/stage0_ingest.json:T0.5.d_gyro_vs_gt.R_base_imu_estimated (gyro vs GT, Kabsch) ; "
            "translation : sensors.yaml"
        )
        info["R_base_imu_change_deg"] = float(np.degrees(rotation_angle_between(T_base_imu_cal[:3, :3], R_est)))
    else:
        raise ValueError(f"variante inconnue : {variant}")
    assert is_rotation(T_base_imu[:3, :3], 1e-6) and is_rotation(T_base_lidar[:3, :3], 1e-9)
    T_lidar_imu = compose(inv(T_base_lidar), T_base_imu)
    info["T_base_imu_used"] = T_base_imu.tolist()
    info["T_lidar_imu"] = T_lidar_imu.tolist()
    info["T_lidar_imu_tum"] = tum_list(T_lidar_imu)
    return info


def write_glim_config(variant: str, scene_root: str, imu_noise: str = "glim_default") -> dict:
    """Écrit la configuration GLIM complète dans CONFIG_DIR à partir des fichiers par défaut de l'image."""
    import json
    import os
    import shutil

    shutil.rmtree(CONFIG_DIR, ignore_errors=True)
    os.makedirs(CONFIG_DIR)
    ext = extrinsics_for_variant(variant, scene_root)

    cfg = {}
    for name in sorted(os.listdir(GLIM_SRC_CONFIG)):
        if name.endswith(".json"):
            cfg[name] = load_glim_config(name)

    cfg["config.json"]["global"].update(
        {
            "config_odometry": "config_odometry_gpu.json",
            "config_sub_mapping": "config_sub_mapping_gpu.json",
            "config_global_mapping": "config_global_mapping_gpu.json",
        }
    )
    ros = cfg["config_ros.json"]["glim_ros"]
    ros.update(
        {
            "enable_local_mapping": True,
            "enable_global_mapping": True,
            "keep_raw_points": False,
            "imu_time_offset": 0.0,  # sensors.yaml:time_offsets (synchro matérielle ; +1,25 ms mesuré en T0.5, négligé)
            "points_time_offset": 0.0,
            "acc_scale": 1.0,  # m/s² (T0.5)
            "ang_scale": 1.0,  # rad/s
            "extension_modules": [],  # pas de visualiseur (Iridescence/OpenGL) ni de moniteur : exécution sans affichage
            "imu_topic": IMU_TOPIC,
            "points_topic": POINTS_TOPIC,
            "image_topic": "/unused_image_topic",
        }
    )
    sens = cfg["config_sensors.json"]["sensors"]
    sens.update(
        {
            "global_shutter_lidar": False,
            "T_lidar_imu": ext["T_lidar_imu_tum"],
            "intensity_field": "intensity",
            "ring_field": "ring",
            # Champ `timestamp` float64 = temps Unix absolu par point (secondes) : réglage explicite, pas d'autoconf.
            "autoconf_perpoint_times": False,
            "autoconf_prefer_frame_time": False,
            "perpoint_relative_time": False,
            "perpoint_time_scale": 1.0,
        }
    )
    if imu_noise == "allan_x10":
        import yaml

        s = yaml.safe_load(open(f"{scene_root}/sensors.yaml"))["imu"]
        sens.update(
            {
                "imu_acc_noise": 10 * s["accelerometer_noise_density"],
                "imu_gyro_noise": 10 * s["gyroscope_noise_density"],
                "imu_bias_noise_acc": 10 * s["accelerometer_random_walk"],
                "imu_bias_noise_gyro": 10 * s["gyroscope_random_walk"],
            }
        )
    cfg["config_preprocess.json"]["preprocess"].update(PREPROCESS_OVERRIDES)
    cfg["config_odometry_gpu.json"]["odometry_estimation"].update(ODOMETRY_OVERRIDES)
    cfg["config_logging.json"]["logging"]["log_dir"] = "/tmp/glim_logs"

    for name, c in cfg.items():
        with open(os.path.join(CONFIG_DIR, name), "w") as f:
            json.dump(c, f, indent=2)
    os.makedirs("/tmp/glim_logs", exist_ok=True)

    key = {
        "config.json": cfg["config.json"]["global"],
        "config_ros.json": ros,
        "config_sensors.json": {k: v for k, v in sens.items() if not k.startswith(("T_lidar_camera", "intrinsics", "distortion", "image_size", "global_shutter_camera"))},
        "config_preprocess.json": cfg["config_preprocess.json"]["preprocess"],
        "config_odometry_gpu.json": cfg["config_odometry_gpu.json"]["odometry_estimation"],
        "config_sub_mapping_gpu.json": cfg["config_sub_mapping_gpu.json"]["sub_mapping"],
        "config_global_mapping_gpu.json": cfg["config_global_mapping_gpu.json"]["global_mapping"],
    }
    return {"extrinsics": ext, "imu_noise": imu_noise, "key_config": key}


# =============================================================================
# Exécution
# =============================================================================


def _parse_graph(dump: str) -> dict:
    """Facteurs de recalage entre sous-cartes (graph.txt) et fermetures de boucle implicites."""
    import os
    import re

    import numpy as np

    out = {}
    gpath = os.path.join(dump, "graph.txt")
    if not os.path.exists(gpath):
        return {"error": "graph.txt absent"}
    pairs, header = [], {}
    for line in open(gpath):
        parts = line.split()
        if parts and parts[0] == "matching_cost":
            pairs.append((parts[1], int(parts[2]), int(parts[3])))
        elif len(parts) == 2 and parts[0].endswith(":"):
            header[parts[0][:-1]] = int(parts[1])
    out.update(header)
    # Positions des origines de sous-cartes.
    origins = {}
    for d in sorted(os.listdir(dump)):
        p = os.path.join(dump, d, "data.txt")
        if re.fullmatch(r"\d{6}", d) and os.path.exists(p):
            lines = [l.strip() for l in open(p).read().splitlines()]
            k = lines.index("T_world_origin:")
            M = np.array([[float(v) for v in lines[k + 1 + r].split()] for r in range(4)])
            origins[int(d)] = M[:3, 3]
    if origins:
        ids = sorted(origins)
        P = np.array([origins[i] for i in ids])
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        idx = {i: k for k, i in enumerate(ids)}
        gaps, loops = [], []
        for typ, a, b in pairs:
            if a in idx and b in idx:
                path = abs(cum[idx[b]] - cum[idx[a]])
                eucl = float(np.linalg.norm(origins[a] - origins[b]))
                gaps.append(abs(b - a))
                if path > 30.0:
                    loops.append({"submaps": [a, b], "path_m": round(float(path), 1), "dist_m": round(eucl, 2)})
        gaps = np.array(gaps) if gaps else np.zeros(0, dtype=int)
        out["n_submaps_dirs"] = len(ids)
        out["submap_path_length_m"] = float(cum[-1])
        out["n_factors"] = int(len(pairs))
        out["n_factors_consecutive"] = int((gaps == 1).sum())
        out["n_factors_non_consecutive"] = int((gaps > 1).sum())
        out["n_loop_factors_path_gt_30m"] = len(loops)
        out["loop_factors_path_gt_30m"] = loops[:200]
    return out


def _convert_traj(src: str, dst: str, T_base_lidar, header: str) -> dict:
    """TUM `T_map_lidar` (GLIM) → TUM `T_map_base = T_map_lidar · inv(T_base_lidar)`."""
    import numpy as np

    from splatpipe.common.geometry import clean_trajectory, compose, inv, read_tum_ns, se3, se3_to_tq, write_tum

    ns, pos, q = read_tum_ns(src)
    ns, pos, q, info = clean_trajectory(ns, pos, q)
    T = compose(se3(pos, q), inv(np.asarray(T_base_lidar))[None])
    p2, q2 = se3_to_tq(T)
    write_tum(dst, ns, p2, q2, header=header)
    info["t0"] = int(ns[0])
    info["t1"] = int(ns[-1])
    return info


@app.function(image=image, volumes={DATA_MOUNT: vol}, gpu="L4", cpu=8, memory=32768, timeout=2 * 3600)
def run_glim(variant: str = "calib", seq: str = SEQ, bag: str = BAG_NAME, imu_noise: str = "glim_default",
             tag: str = "") -> dict:
    import datetime
    import json
    import os
    import re
    import shutil
    import subprocess
    import time

    import numpy as np

    from splatpipe.common.scene import Scene

    t_start = time.time()
    vol.reload()
    sc = Scene(DATASET, seq)
    name = variant + (f"_{tag}" if tag else "")
    out_dir = sc.path("traj", "glim", name)
    info = {
        "variant": variant,
        "name": name,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "image": __import__("splatpipe.stage1.images", fromlist=["GLIM_IMAGE_REF"]).GLIM_IMAGE_REF,
        "gpu": _sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")["stdout"].strip(),
        "glim_commits": _sh("cd /root/ros2_ws/src/glim && git log -1 --format='%H %cd'; cd /root/ros2_ws/src/glim_ros2 && git log -1 --format='%H %cd'; cd /root/gtsam_points && git log -1 --format='%H %cd'")["stdout"].strip().splitlines(),
    }

    # 1) Configuration (sensors.yaml lu maintenant ; sa version est tracée par son SHA-256).
    info["config"] = write_glim_config(variant, sc.root, imu_noise)

    # 2) Bag : copie locale.
    src = _bag_dir(seq, bag)
    local = os.path.join(BAG_LOCAL_ROOT, bag)
    t0 = time.time()
    if not os.path.exists(os.path.join(local, "metadata.yaml")):
        os.makedirs(local, exist_ok=True)
        for f in os.listdir(src):
            shutil.copyfile(os.path.join(src, f), os.path.join(local, f))
    info["bag"] = {"source": src, "local": local, "copy_s": round(time.time() - t0, 1),
                   "size_bytes": sum(os.path.getsize(os.path.join(local, f)) for f in os.listdir(local))}

    # 3) GLIM
    shutil.rmtree(DUMP_DIR, ignore_errors=True)
    log_path = "/tmp/glim_stdout.log"
    cmd = (
        f"{ROS_ENV} && export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST && exec {GLIM_BIN} {local} --ros-args "
        f"-p config_path:={CONFIG_DIR} -p auto_quit:=true -p dump_path:={DUMP_DIR}"
    )
    info["command"] = cmd
    gpu_samples = []
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(["bash", "-c", cmd], stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        rusage, status = None, None
        last = time.time()
        while True:
            pid, st, ru = os.wait4(proc.pid, os.WNOHANG)
            if pid != 0:
                rusage, status = ru, st
                break
            now = time.time()
            if now - t0 > 100 * 60:
                proc.kill()
                pid, st, ru = os.wait4(proc.pid, 0)
                rusage, status = ru, st
                info["killed_timeout"] = True
                break
            r = _sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits")
            try:
                u, m = [float(v) for v in r["stdout"].strip().split(",")]
                gpu_samples.append((now - t0, u, m))
            except ValueError:
                pass
            if now - last > 30:  # suivi visible dans `modal app logs`
                last = now
                tail = _sh(f"tail -n 2 {log_path}")["stdout"].strip().replace("\n", " | ")
                print(f"[{now - t0:6.0f} s] GPU {gpu_samples[-1][1] if gpu_samples else -1:.0f} % | {tail[-300:]}", flush=True)
            time.sleep(max(0.0, 2.0 - (time.time() - now)))
    wall = time.time() - t0
    rc = os.waitstatus_to_exitcode(status)
    g = np.array(gpu_samples) if gpu_samples else np.zeros((0, 3))
    gpu_busy_s = float(np.sum(g[:, 1] / 100.0 * np.diff(np.concatenate([[0.0], g[:, 0]])))) if len(g) else None
    info["run"] = {
        "returncode": rc,
        "wall_s": round(wall, 1),
        "cpu_user_s": round(rusage.ru_utime, 1),
        "cpu_sys_s": round(rusage.ru_stime, 1),
        "max_rss_gib": round(rusage.ru_maxrss / 1024**2, 2),
        "gpu_util_mean_pct": float(g[:, 1].mean()) if len(g) else None,
        "gpu_util_weighted_s": gpu_busy_s,
        "gpu_mem_max_mib": float(g[:, 2].max()) if len(g) else None,
        "bag_duration_s": 288.99,
        "realtime_factor": round(288.99 / wall, 2),
    }
    log = open(log_path, errors="replace").read()
    pats = {
        "warn": r"\[warning\]|\[warn\]", "error": r"\[error\]", "critical": r"\[critical\]",
        "skip_points": r"skip an invalid point cloud", "skip_imu": r"skip an invalid IMU",
        "large_gap": r"large time gap", "rewind": r"rewind", "large_time_diff_points_imu": r"large time difference",
    }
    info["log_counts"] = {k: len(re.findall(p, log, flags=re.I)) for k, p in pats.items()}
    info["log_head"] = log[:4000]
    info["log_tail"] = log[-6000:]
    speeds = [float(x) for x in re.findall(r"playback speed: ([0-9.]+)x", log)]
    info["run"]["playback_speed_median"] = float(np.median(speeds)) if speeds else None

    # 4) Sorties : dump complet + trajectoires converties en T_map_base.
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(log_path, os.path.join(out_dir, "glim_stdout.log"))
    with open(os.path.join(out_dir, "glim_config_written.json"), "w") as f:
        json.dump({n: json.load(open(os.path.join(CONFIG_DIR, n))) for n in sorted(os.listdir(CONFIG_DIR))}, f, indent=2)
    if os.path.isdir(DUMP_DIR):
        dst_dump = os.path.join(out_dir, "dump")
        shutil.rmtree(dst_dump, ignore_errors=True)
        shutil.copytree(DUMP_DIR, dst_dump)
        if os.path.isdir("/tmp/glim_logs"):
            shutil.copytree("/tmp/glim_logs", os.path.join(out_dir, "glim_logs"), dirs_exist_ok=True)
        info["dump_size_mib"] = round(
            sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(DUMP_DIR) for f in fs) / 2**20, 1)
        info["dump_files"] = sorted(os.listdir(DUMP_DIR))[:30]
        vol.commit()
        try:
            info["graph"] = _parse_graph(DUMP_DIR)
        except Exception as e:  # l'analyse du graphe ne doit pas faire perdre le dump
            info["graph"] = {"error": repr(e)}
        T_base_lidar = info["config"]["extrinsics"]["T_base_lidar"]
        hdr = ("GLIM ({}) : T_map_base, map = repère carte de GLIM (gravité alignée, origine au 1er scan).\n"
               "Convertie depuis {} : T_map_base = T_map_lidar · inv(T_base_lidar). Temps = premier point du scan.")
        conv = {}
        for srcname, dstname in (("traj_lidar.txt", "traj_base.tum"), ("odom_lidar.txt", "odom_base.tum")):
            p = os.path.join(DUMP_DIR, srcname)
            if os.path.exists(p) and os.path.getsize(p) > 0:
                try:
                    conv[dstname] = _convert_traj(p, os.path.join(out_dir, dstname), T_base_lidar, hdr.format(name, srcname))
                except Exception as e:
                    conv[dstname] = {"error": repr(e)}
        info["trajectories"] = conv
    info["total_function_s"] = round(time.time() - t_start, 1)
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)
    vol.commit()
    return json.loads(json.dumps(info, default=str))


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=2, memory=4096, timeout=600)
def select_variant(name: str, seq: str = SEQ) -> dict:
    """Publie `traj/glim/<name>/traj_base.tum` comme `traj/glim.tum` et écrit `traj/glim/run_info.json`."""
    import json
    import os
    import shutil

    from splatpipe.common.scene import Scene

    vol.reload()
    sc = Scene(DATASET, seq)
    src = sc.path("traj", "glim", name, "traj_base.tum")
    shutil.copyfile(src, sc.traj_path("glim"))
    runs = {}
    for d in sorted(os.listdir(sc.path("traj", "glim"))):
        p = sc.path("traj", "glim", d, "run_info.json")
        if os.path.exists(p):
            r = json.load(open(p))
            runs[d] = {k: r.get(k) for k in ("variant", "created_utc", "image", "gpu", "glim_commits", "run", "log_counts", "graph", "trajectories", "dump_size_mib")}
            runs[d]["T_lidar_imu_tum"] = r["config"]["extrinsics"]["T_lidar_imu_tum"]
            runs[d]["sensors_yaml_sha256"] = r["config"]["extrinsics"]["sensors_yaml_sha256"]
            runs[d]["sensors_yaml_has_calibration_corrections"] = r["config"]["extrinsics"]["sensors_yaml_has_calibration_corrections"]
            if runs[d].get("graph"):
                runs[d]["graph"] = {k: v for k, v in runs[d]["graph"].items() if k != "loop_factors_path_gt_30m"}
    import re

    summary = {"selected_run": name, "selected_variant": re.sub(r"_r\d+$", "", name), "glim_tum": sc.traj_path("glim"),
               "source": src, "runs": runs}
    with open(sc.path("traj", "glim", "run_info.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    vol.commit()
    return summary


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=4, memory=16384, timeout=1800)
def export_map(name: str = "calib", seq: str = SEQ) -> dict:
    """Carte GLIM : fusion des sous-cartes du dump (`T_world_origin · points`, comme `GlobalMapping::export_points`)
    → `traj/glim/<name>/glim_map.ply` (float32 x y z [intensity], repère carte de GLIM). Sous-produit de contrôle ;
    la carte de référence 5 cm sera reconstruite en 1.4–1.5 à partir des scans bruts."""
    import os
    import re

    import numpy as np

    from splatpipe.common.scene import Scene

    vol.reload()
    sc = Scene(DATASET, seq)
    dump = sc.path("traj", "glim", name, "dump")
    pts, inten, n_sub = [], [], 0
    for d in sorted(os.listdir(dump)):
        sub = os.path.join(dump, d)
        if not (re.fullmatch(r"\d{6}", d) and os.path.exists(os.path.join(sub, "points_compact.bin"))):
            continue
        lines = [l.strip() for l in open(os.path.join(sub, "data.txt")).read().splitlines()]
        k = lines.index("T_world_origin:")
        M = np.array([[float(v) for v in lines[k + 1 + r].split()] for r in range(4)])
        p = np.fromfile(os.path.join(sub, "points_compact.bin"), dtype="<f4").reshape(-1, 3).astype(np.float64)
        pts.append(p @ M[:3, :3].T + M[:3, 3])
        ip = os.path.join(sub, "intensities_compact.bin")
        inten.append(np.fromfile(ip, dtype="<f4") if os.path.exists(ip) else np.zeros(len(p), np.float32))
        n_sub += 1
    P = np.concatenate(pts).astype(np.float32)
    I = np.concatenate(inten).astype(np.float32)
    out = sc.path("traj", "glim", name, "glim_map.ply")
    rec = np.empty(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
    rec["x"], rec["y"], rec["z"], rec["intensity"] = P[:, 0], P[:, 1], P[:, 2], I
    with open(out, "wb") as f:
        f.write((
            "ply\nformat binary_little_endian 1.0\ncomment GLIM submaps merged, frame = GLIM map\n"
            f"element vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property float intensity\nend_header\n"
        ).encode())
        rec.tofile(f)
    vol.commit()
    return {"ply": out, "n_submaps": n_sub, "n_points": int(len(P)), "bbox_min": P.min(0).tolist(),
            "bbox_max": P.max(0).tolist(), "size_mib": round(os.path.getsize(out) / 2**20, 1)}


@app.local_entrypoint()
def main(variant: str = "calib", imu_noise: str = "glim_default", tag: str = "", select: bool = False):
    import json
    import os

    info = run_glim.remote(variant, SEQ, BAG_NAME, imu_noise, tag)
    name = info["name"]
    os.makedirs("logs/stage1/traj", exist_ok=True)
    with open(f"logs/stage1/traj/glim_run_{name}.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    short = {k: info.get(k) for k in ("run", "log_counts", "graph", "trajectories", "dump_size_mib", "total_function_s")}
    if short.get("graph"):
        short["graph"] = {k: v for k, v in short["graph"].items() if k != "loop_factors_path_gt_30m"}
    print(json.dumps(short, indent=2, ensure_ascii=False))
    print(info.get("log_tail", "")[-3000:])
    if select:
        print(json.dumps(select_variant.remote(name), indent=2, ensure_ascii=False)[:3000])
    print(json.dumps(export_map.remote(name), indent=2))


@app.local_entrypoint()
def select_main(name: str = "calib"):
    import json

    print(json.dumps(select_variant.remote(name), indent=2, ensure_ascii=False)[:5000])
