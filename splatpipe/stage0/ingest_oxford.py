"""Étape 0.3 — Ingestion d'une séquence Oxford Spires au format canonique + tests T0.1 et T0.5.

Écrit sous `/data/scenes/oxford_spires/<seq>/` : meta.yaml, sensors.yaml, cameras/<cam>/index.csv,
cameras/frames.csv, imu/imu.npz, traj/{gt,vilens,hba,colmap}.tum, splits.yaml, reports/stage0_ingest.json.
Ne touche ni à `/data/raw` (lecture seule) ni à `lidar/` et `lidar_model/` (ingestion LiDAR, autre module).
Idempotent : une relance réécrit les mêmes fichiers.

Lancement (depuis la racine du projet) :
  .venv/bin/modal run splatpipe/stage0/ingest_oxford.py --seq 2024-03-13-observatory-quarter-01
"""

from __future__ import annotations

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume

app = modal.App("splatsim-stage0-ingest")
vol = data_volume()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy", "scipy", "pyyaml")
    .add_local_python_source("splatpipe")
)

DATASET = "oxford_spires"
RAW_ROOT = "raw/oxford_spires"  # relatif à DATA_MOUNT
CAMS = ("cam0", "cam1", "cam2")
CAM_LABEL = {"cam0": "front", "cam1": "left", "cam2": "right"}
CAM_EXPECTED_AXIS_BASE = {"cam0": [1.0, 0.0, 0.0], "cam1": [0.0, 1.0, 0.0], "cam2": [0.0, -1.0, 0.0]}
LIDAR_ID = "hesai_qt64"
NOMINAL_HZ = {"camera": 20.0, "imu": 400.0, "lidar": 10.0}
T_BASE_LIDAR_TQ = ([0.0, 0.0, 0.124], [0.0, 0.0, 1.0, 0.0])  # README calibration du dataset
IMU_HEADER = "secs,nsecs,acc_x,acc_y,acc_z,ang_vel_x,ang_vel_y,ang_vel_z"
G_REF = 9.81  # m/s² (Oxford : ~9,812)

# Tolérances des tests (choisies ici, documentées dans la note stage0-3).
TRIPLET_TOL_NS = 5_000_000  # 5 ms ≪ période caméra 50 ms
RATE_REL_TOL = 0.01  # fréquence médiane à ±1 % du nominal
ACC_NORM_TOL = 0.2  # |‖g‖ − 9,81| < 0,2 m/s² (2 %)
GRAVITY_ANGLE_TOL_DEG = 2.0
CAM_AXIS_TOL_DEG = 5.0
BASELINE_RANGE_M = (0.02, 0.25)
GYRO_EXTRINSIC_CONVENTION_TOL_DEG = 5.0  # erreur de convention (axes permutés, 180°…) si dépassé
GYRO_EXTRINSIC_PRECISION_TOL_DEG = 1.0  # précision attendue d'une calibration (sinon WARN)
CAM_HANDEYE_CONVENTION_TOL_DEG = 5.0
CAM_HANDEYE_PRECISION_TOL_DEG = 0.3  # ≈ 3,6 px au centre de l'image (f ≈ 698 px)
LIDAR_CLOSURE_ROT_TOL_DEG = 0.5
LIDAR_CLOSURE_TRANS_TOL_M = 0.10
GYRO_TIME_OFFSET_TOL_S = 0.005
STATIC_SPEED_MAX = 0.10  # m/s
STATIC_ANGRATE_MAX_DEG = 10.0  # °/s
STATIC_MIN_DURATION_S = 0.5


# =============================================================================
# Utilitaires
# =============================================================================


def _py(x):
    """Convertit récursivement numpy → types Python sérialisables (JSON/YAML)."""
    import numpy as np

    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return _py(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and x != x:
        return None
    return x


def _yaml_dump(obj) -> str:
    """YAML lisible : listes de scalaires en style « flow », matrices ligne par ligne."""
    import yaml

    class Dumper(yaml.SafeDumper):
        pass

    def repr_list(dumper, data):
        flow = all(isinstance(v, (int, float, bool)) or v is None for v in data)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)

    Dumper.add_representer(list, repr_list)
    return yaml.dump(_py(obj), Dumper=Dumper, sort_keys=False, allow_unicode=True, width=120)


def _stamp_ns_from_name(name: str) -> int:
    """`1710338243.457672000.pcd` ou `cloud_<s>_<ns>.pcd` → nanosecondes (exact)."""
    from splatpipe.common.geometry import parse_stamp_ns

    base = name.rsplit(".", 1)[0]
    if base.startswith("cloud_"):
        s, ns = base[len("cloud_"):].split("_")
        return int(s) * 1_000_000_000 + int(ns)
    return parse_stamp_ns(base)


def _list_stamped(dirpath: str, ext: str):
    """Liste les fichiers `<stamp>.<ext>` → (stamp_ns trié int64, noms triés, nb de doublons)."""
    import os

    import numpy as np

    names = [f for f in os.listdir(dirpath) if f.endswith("." + ext)]
    ns = np.array([_stamp_ns_from_name(f) for f in names], dtype=np.int64)
    order = np.argsort(ns, kind="stable")
    ns, names = ns[order], [names[i] for i in order]
    n_dup = int(np.sum(np.diff(ns) == 0)) if len(ns) > 1 else 0
    return ns, names, n_dup


def _nearest(ref_sorted, q):
    """Pour chaque q, indice et |écart| du plus proche élément de `ref_sorted` (int64 ns)."""
    import numpy as np

    ref_sorted = np.asarray(ref_sorted)
    q = np.asarray(q)
    n = len(ref_sorted)
    i = np.searchsorted(ref_sorted, q)
    i0 = np.clip(i - 1, 0, n - 1)
    i1 = np.clip(i, 0, n - 1)
    d0 = np.abs(q - ref_sorted[i0])
    d1 = np.abs(ref_sorted[i1] - q)
    j = np.where(d1 < d0, i1, i0)
    return j, np.minimum(d0, d1)


def _pct(x, ps=(0, 50, 90, 95, 99, 100)):
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {}
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


def _series_check(ns, nominal_hz: float, n_dup: int = 0, raw_unsorted_steps: int | None = None) -> dict:
    """T0.1 pour une série d'horodatages (triés) : monotonie, fréquence, trous, régularité de la grille.

    Seuils relatifs à la période médiane P (robuste à la dérive d'horloge et au bruit flottant) :
    - « perte » : Δt > 1,5 P (au moins un échantillon manquant) → listée dans `gaps` ;
    - « trou > 2 périodes » (critère du plan) : Δt > 2,5 P, c.-à-d. au moins 2 échantillons consécutifs manquants.
    Un Δt de exactement 2 P (une seule image perdue) n'est donc PAS un trou > 2 périodes.
    """
    import numpy as np

    from splatpipe.common.geometry import ns_to_s

    ns = np.asarray(ns, dtype=np.int64)
    t = ns_to_s(ns)
    d = np.diff(ns).astype(np.float64) * 1e-9
    med = float(np.median(d))
    rate = 1.0 / med
    drop_idx = np.nonzero(d > 1.5 * med)[0]
    gaps = [
        {
            "t_before": float(t[i]),
            "t_after": float(t[i + 1]),
            "stamp_ns_before": int(ns[i]),
            "stamp_ns_after": int(ns[i + 1]),
            "t_rel_s": round(float(t[i] - t[0]), 3),
            "dt_s": round(float(d[i]), 6),
            "missing_est": int(round(d[i] / med)) - 1,
            "gt_2_periods": bool(d[i] > 2.5 * med),
        }
        for i in drop_idx
    ]
    n_gt2 = int(sum(gp["gt_2_periods"] for gp in gaps))
    # Régularité : écart à une grille t = a + b·k (k = rang théorique), révèle un déclenchement matériel.
    # k cumulé pas à pas (robuste à la dérive entre période médiane et période moyenne).
    k = np.concatenate([[0.0], np.cumsum(np.maximum(np.round(d / med), 1.0))])
    A = np.stack([np.ones_like(k), k], axis=1)
    y = (ns - ns[0]).astype(np.float64) * 1e-9
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    strictly_increasing = bool(np.all(np.diff(ns) > 0))
    rate_ok = abs(rate / nominal_hz - 1.0) < RATE_REL_TOL
    out = {
        "n": int(len(ns)),
        "t0": float(t[0]),
        "t1": float(t[-1]),
        "duration_s": round(float(t[-1] - t[0]), 3),
        "raw_order_unsorted_steps": raw_unsorted_steps,
        "n_duplicates": int(n_dup),
        "strictly_increasing_after_sort": strictly_increasing,
        "nominal_hz": nominal_hz,
        "median_dt_s": med,
        "median_rate_hz": round(rate, 4),
        "mean_rate_hz": round(float((len(ns) - 1) / (t[-1] - t[0])), 4),
        "rate_rel_err": round(rate / nominal_hz - 1.0, 5),
        "min_dt_s": round(float(d.min()), 6),
        "max_gap_s": round(float(d.max()), 6),
        "n_drops": int(len(gaps)),
        "n_drops_single": int(sum(gp["missing_est"] == 1 for gp in gaps)),
        "n_missing_est": int(sum(gp["missing_est"] for gp in gaps)),
        "n_gaps_gt_2_periods": n_gt2,
        "gaps": gaps,
        "grid_fit_period_s": float(coef[1]),
        "grid_residual_std_ms": round(float(resid.std() * 1e3), 4),
        "grid_residual_maxabs_ms": round(float(np.abs(resid).max() * 1e3), 4),
    }
    out["pass_monotonic"] = strictly_increasing and n_dup == 0
    out["pass_rate"] = bool(rate_ok)
    out["pass_no_gap"] = n_gt2 == 0
    out["status"] = "PASS" if (out["pass_monotonic"] and rate_ok and n_gt2 == 0) else "FAIL"
    return out


# =============================================================================
# Calibration → sensors.yaml
# =============================================================================


def _build_sensors(calib_dir: str, imu_measured_rate: float) -> tuple[dict, dict]:
    """Lit la calibration Kalibr/DiffCal et construit sensors.yaml + contrôles de cohérence."""
    import numpy as np
    import yaml

    from splatpipe.common import geometry as g

    with open(f"{calib_dir}/cam-lidar-imu.yaml") as f:
        ext = yaml.safe_load(f)
    with open(f"{calib_dir}/imu.yaml") as f:
        imu_cal = yaml.safe_load(f)

    T_base_lidar = g.se3(*T_BASE_LIDAR_TQ)
    checks = {"T_cam_lidar_matrix_vs_quaternion": {}}
    cameras, T_base_cam = {}, {}
    for cam in CAMS:
        with open(f"{calib_dir}/{cam}.yaml") as f:
            ci = yaml.safe_load(f)
        K = np.array(ci["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        D = [float(v) for v in ci["distortion_coefficients"]["data"]]
        assert ci["distortion_model"] == "equidistant", ci["distortion_model"]
        assert len(D) == 4
        P = np.array(ci["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)
        M = np.array(ext[cam]["T_cam_lidar"], dtype=np.float64)
        Tq = g.se3(ext[cam]["C_r_CL"], ext[cam]["C_q_CL"])
        checks["T_cam_lidar_matrix_vs_quaternion"][cam] = {
            "rot_diff_deg": float(np.degrees(g.rotation_angle_between(M[:3, :3], Tq[:3, :3]))),
            "trans_diff_m": float(np.abs(M[:3, 3] - Tq[:3, 3]).max()),
            "orthonormality_err": float(np.abs(M[:3, :3].T @ M[:3, :3] - np.eye(3)).max()),
            "det": float(np.linalg.det(M[:3, :3])),
            "last_row_ok": bool(np.allclose(M[3], [0, 0, 0, 1])),
            "projection_matrix_equals_K": bool(np.allclose(P[:, :3], K)),
            "quat_norm": float(np.linalg.norm(ext[cam]["C_q_CL"])),
        }
        # Matrice orthonormalisée (SVD) pour éviter de propager ~1e-15 d'erreur ; écart rapporté ci-dessus.
        U, _, Vt = np.linalg.svd(M[:3, :3])
        M_clean = g.se3_from_rt(U @ Vt, M[:3, 3])
        T_base_cam[cam] = T_base_lidar @ g.inv(M_clean)
        cameras[cam] = {
            "label": CAM_LABEL[cam],
            "model": "OPENCV_FISHEYE",
            "width": int(ci["image_width"]),
            "height": int(ci["image_height"]),
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "k1": D[0],
            "k2": D[1],
            "k3": D[2],
            "k4": D[3],
            "rate_hz": NOMINAL_HZ["camera"],
            "shutter": "global",
            "T_base_cam": T_base_cam[cam],
            "index": f"cameras/{cam}/index.csv",
            "source": {
                "intrinsics": f"{RAW_ROOT}/calibration/{cam}.yaml (Kalibr equidistant, 2024-03-19)",
                "extrinsics": f"{RAW_ROOT}/calibration/cam-lidar-imu.yaml:{cam}.T_cam_lidar (DiffCal, 2024-03-19)",
                "formula": "T_base_cam = T_base_lidar · inv(T_cam_lidar)",
            },
        }

    q_ci = ext["cam0"]["C_q_CI"]
    T_cam0_imu = g.se3(ext["cam0"]["C_r_CI"], q_ci)
    T_base_imu = T_base_cam["cam0"] @ T_cam0_imu
    checks["C_q_CI_norm"] = float(np.linalg.norm(q_ci))

    lidars = {
        LIDAR_ID: {
            "model": "Hesai QT64",
            "rate_hz": NOMINAL_HZ["lidar"],
            "T_base_lidar": T_base_lidar,
            "beam_model": f"lidar_model/{LIDAR_ID}.json",
            "raw_clouds": "raw/lidar-clouds/<s>.<ns>.pcd (nom = temps du premier point)",
            "source": "calibration/README.md : T_base_lidar t_xyz (0, 0, 0.124), q_xyzw (0, 0, 1, 0)",
        }
    }
    imu = {
        "model": "Alphasense Core (Bosch BMI085)",
        "frame": "imu",
        "T_base_imu": T_base_imu,
        "rate_hz": float(imu_cal.get("update_rate", NOMINAL_HZ["imu"])),
        "measured_rate_hz": round(float(imu_measured_rate), 4),
        "units": {"acc": "m/s^2", "gyro": "rad/s"},
        "accelerometer_noise_density": float(imu_cal["accelerometer_noise_density"]),
        "accelerometer_random_walk": float(imu_cal["accelerometer_random_walk"]),
        "gyroscope_noise_density": float(imu_cal["gyroscope_noise_density"]),
        "gyroscope_random_walk": float(imu_cal["gyroscope_random_walk"]),
        "noise_units": {
            "accelerometer_noise_density": "m/s^2/sqrt(Hz)",
            "accelerometer_random_walk": "m/s^3/sqrt(Hz)",
            "gyroscope_noise_density": "rad/s/sqrt(Hz)",
            "gyroscope_random_walk": "rad/s^2/sqrt(Hz)",
        },
        "data": "imu/imu.npz",
        "source": {
            "noise": f"{RAW_ROOT}/calibration/imu.yaml (Allan variance, 2024-06-27/29)",
            "extrinsics": f"{RAW_ROOT}/calibration/cam-lidar-imu.yaml:cam0.C_r_CI/C_q_CI = T_cam0_imu (Kalibr, 2024-06-29)",
            "formula": "T_base_imu = T_base_cam0 · T_cam0_imu",
        },
    }
    time_offsets = {c: 0.0 for c in CAMS}
    time_offsets[LIDAR_ID] = 0.0
    time_offsets["imu"] = 0.0

    # ---- valeurs dérivées de contrôle
    pos = {c: T_base_cam[c][:3, 3] for c in CAMS}
    axis = {c: T_base_cam[c][:3, 2] for c in CAMS}
    baselines, rel_angles = {}, {}
    for a, b in [("cam0", "cam1"), ("cam0", "cam2"), ("cam1", "cam2")]:
        baselines[f"{a}_{b}"] = float(np.linalg.norm(pos[a] - pos[b]))
        rel_angles[f"{a}_{b}"] = float(np.degrees(g.rotation_angle_between(T_base_cam[a][:3, :3], T_base_cam[b][:3, :3])))
    derived = {
        "camera_position_in_base_m": {c: pos[c] for c in CAMS},
        "camera_optical_axis_in_base": {c: axis[c] for c in CAMS},
        "camera_axis_angle_to_expected_deg": {
            c: float(np.degrees(g.angle_between_vectors(axis[c], CAM_EXPECTED_AXIS_BASE[c]))) for c in CAMS
        },
        "camera_baselines_m": baselines,
        "camera_relative_rotation_deg": rel_angles,
        "imu_position_in_base_m": T_base_imu[:3, 3],
        "imu_z_angle_to_base_z_deg": float(np.degrees(g.angle_between_vectors(T_base_imu[:3, 2], [0, 0, 1]))),
        "imu_axes_in_base": {"x": T_base_imu[:3, 0], "y": T_base_imu[:3, 1], "z": T_base_imu[:3, 2]},
        "lidar_position_in_base_m": T_base_lidar[:3, 3],
        "T_lidar_imu": g.inv(T_base_lidar) @ T_base_imu,
        "T_cam0_imu": T_cam0_imu,
    }
    sensors = {
        "contract_version": "0.1",
        "base_frame": "base",
        "cameras": cameras,
        "lidars": lidars,
        "imu": imu,
        "time_offsets": time_offsets,
        "derived": derived,
    }
    return _py(sensors), _py(checks)


def _write_sensors_yaml(path: str, sensors: dict) -> None:
    head = (
        "# sensors.yaml — contrat de données v0.1 (splatpipe.stage0.ingest_oxford)\n"
        "# T_a_b : p_a = T_a_b · p_b (4×4, lignes). Quaternions xyzw. Distances en m, angles en rad sauf *_deg.\n"
        "# Caméras : repère optique OpenCV (x droite, y bas, z avant), modèle OPENCV_FISHEYE (équidistant Kalibr).\n"
        "# base : bas de l'appareil, x avant, y gauche, z haut.\n"
    )
    parts = [head]
    for key in ("contract_version", "base_frame", "cameras", "lidars", "imu"):
        parts.append(_yaml_dump({key: sensors[key]}))
    parts.append(
        "# time_offsets : décalage (s) à AJOUTER à l'horodatage du capteur pour obtenir le temps de référence.\n"
        "# Tous nuls : caméras et IMU Alphasense synchronisées matériellement (FPGA), horloge Alphasense\n"
        "# synchronisée au Hesai par PTP (précision annoncée sub-milliseconde). Vérifié par T0.5 (gyro vs GT)\n"
        "# pour l'IMU ; le décalage caméra↔LiDAR sera vérifié par T0.4 (superposition à l'arrêt / en mouvement).\n"
    )
    parts.append(_yaml_dump({"time_offsets": sensors["time_offsets"]}))
    parts.append("# derived : valeurs de contrôle calculées (non utilisées comme source de vérité).\n")
    parts.append(_yaml_dump({"derived": sensors["derived"]}))
    with open(path, "w") as f:
        f.write("".join(parts))


# =============================================================================
# Trajectoires
# =============================================================================


def _gt_kinematics(tg, pg, qg, tq, h=0.1):
    """Vitesse (monde, m/s) et vitesse angulaire (repère base, rad/s) par différences centrées ±h."""
    import numpy as np

    from splatpipe.common import geometry as g

    p1, q1, v1 = g.interpolate_poses(tg, pg, qg, tq + h, max_gap_s=0.25)
    p0, q0, v0 = g.interpolate_poses(tg, pg, qg, tq - h, max_gap_s=0.25)
    vel = (p1 - p0) / (2 * h)
    R0, R1 = g.quat_to_rot(q0), g.quat_to_rot(q1)
    omega_base = g.rot_to_rotvec(np.swapaxes(R0, -1, -2) @ R1) / (2 * h)
    return vel, omega_base, v0 & v1


def _identify_body_frame(t_x, R_x, tg, pg, qg, candidates: dict) -> dict:
    """Identifie le repère « corps » d'une trajectoire locale en comparant ses rotations relatives à celles de la GT.

    Si la trajectoire donne T_L_B avec B = base·X (X = T_base_B), alors R_rel_B = Xᵀ · R_rel_base · X.
    L'angle des rotations relatives est invariant ; c'est l'axe qui discrimine les hypothèses.
    """
    import numpy as np

    from splatpipe.common import geometry as g

    _, qa, va = g.interpolate_poses(tg, pg, qg, t_x[:-1], max_gap_s=0.25)
    _, qb, vb = g.interpolate_poses(tg, pg, qg, t_x[1:], max_gap_s=0.25)
    Rg = np.swapaxes(g.quat_to_rot(qa), -1, -2) @ g.quat_to_rot(qb)
    ok = va & vb & (np.degrees(g.rotation_angle(Rg)) > 5.0)
    out = {"n_pairs_used": int(ok.sum()), "hypotheses": {}}
    # Deux sens de stockage : T_L_B (monde←corps, TUM usuel) ou T_B_L (corps←monde, style COLMAP).
    for storage, R_LB in (("T_local_body", R_x), ("T_body_local", np.swapaxes(R_x, -1, -2))):
        Rx = np.swapaxes(R_LB[:-1], -1, -2) @ R_LB[1:]
        for name, X in candidates.items():
            pred = np.swapaxes(X, -1, -2) @ Rg[ok] @ X
            err = np.degrees(g.rotation_angle_between(pred, Rx[ok]))
            out["hypotheses"][f"{storage}:{name}"] = {
                "median_err_deg": float(np.median(err)),
                "p90_err_deg": float(np.percentile(err, 90)),
            }
    hyp = out["hypotheses"]
    out["best"] = min(hyp, key=lambda n: hyp[n]["median_err_deg"])
    return out


def _handeye_rotation(t_x, R_x, tg, pg, qg, lag: int = 1, min_angle_deg: float = 2.0, sel=None):
    """Rotation X = R_base_B par « hand-eye » rotationnel : rotvec(R_rel_B) = Xᵀ · rotvec(R_rel_base).

    Invariant à l'alignement global (repère local) de la trajectoire X. Renvoie (X, n_paires, valeurs singulières).
    """
    import numpy as np

    from splatpipe.common import geometry as g

    ta, tb = t_x[:-lag], t_x[lag:]
    _, qa, va = g.interpolate_poses(tg, pg, qg, ta, max_gap_s=0.25)
    _, qb, vb = g.interpolate_poses(tg, pg, qg, tb, max_gap_s=0.25)
    Rg = np.swapaxes(g.quat_to_rot(qa), -1, -2) @ g.quat_to_rot(qb)
    Rx = np.swapaxes(R_x[:-lag], -1, -2) @ R_x[lag:]
    ok = va & vb & (np.degrees(g.rotation_angle(Rg)) > min_angle_deg) & (tb - ta < 3.0)
    if sel is not None:
        ok &= sel[: len(ok)]
    a = g.rot_to_rotvec(Rg[ok])
    b = g.rot_to_rotvec(Rx[ok])
    U, S, Vt = np.linalg.svd(a.T @ b)
    M = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T  # b ≈ M·a, M = Xᵀ
    return M.T, int(ok.sum()), S


# =============================================================================
# Fermeture LiDAR → world (scan fixe)
# =============================================================================


def _read_pcd(path: str):
    """Lecteur PCD binaire minimal (numpy) → tableau structuré (champs `_` renommés `_padN`)."""
    import numpy as np

    with open(path, "rb") as f:
        hdr = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: en-tête PCD incomplet")
            line = line.decode("ascii", "replace").strip()
            if not line or line.startswith("#"):
                continue
            k, _, v = line.partition(" ")
            hdr[k] = v.split()
            if k == "DATA":
                break
        if hdr["DATA"][0] != "binary":
            raise ValueError(f"{path}: seul DATA binary est supporté")
        names, sizes, types = hdr["FIELDS"], [int(x) for x in hdr["SIZE"]], hdr["TYPE"]
        counts = [int(x) for x in hdr.get("COUNT", ["1"] * len(names))]
        kind = {"F": "f", "U": "u", "I": "i"}
        dt = []
        for i, (n, sz, ty, c) in enumerate(zip(names, sizes, types, counts)):
            name = n if n != "_" else f"_pad{i}"
            dt.append((name, f"<{kind[ty]}{sz}", (c,)) if c > 1 else (name, f"<{kind[ty]}{sz}"))
        npts = int(hdr["POINTS"][0])
        data = f.read()
    return np.frombuffer(data, dtype=np.dtype(dt), count=npts)


def _kabsch_T(P, Q):
    """Transformation rigide T minimisant Σ|Q − (R·P + t)|²."""
    import numpy as np

    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mq - R @ mp
    return T


def _icp(src, tree, dst, schedule=(1.0, 0.5, 0.25, 0.12), iters=20):
    """ICP point-à-point (distances max décroissantes). Renvoie (T_corr, stats) avec dst ≈ T_corr · src."""
    import numpy as np

    def stats(pts):
        d, _ = tree.query(pts, distance_upper_bound=0.5, workers=-1)
        fin = np.isfinite(d)
        return {
            "median_nn_m_within_0.5": float(np.median(d[fin])) if fin.any() else None,
            "frac_within_0.10m": float(np.mean(d < 0.10)),
        }

    before = stats(src)
    T = np.eye(4)
    cur = src.copy()
    for max_d in schedule:
        for _ in range(iters):
            d, j = tree.query(cur, distance_upper_bound=max_d, workers=-1)
            m = np.isfinite(d)
            if m.sum() < 500:
                break
            dT = _kabsch_T(cur[m], dst[j[m]])
            cur = cur @ dT[:3, :3].T + dT[:3, 3]
            T = dT @ T
            if np.abs(dT[:3, :3] - np.eye(3)).max() < 1e-6 and np.abs(dT[:3, 3]).max() < 1e-5:
                break
    return T, {"before": before, "after": stats(cur)}


def _lidar_world_closure(seq_dir, site, lidar_ns, lidar_names, tg, pg, qg, T_base_lidar, R_base_imu, R_base_imu_est):
    """Test de fermeture : p_world = T_world_base(t_point) · T_base_lidar · p_lidar doit coller au scan fixe.

    Pour quelques scans (lents et en marche normale), désanamorphose chaque point avec la GT interpolée à son
    horodatage, puis recale par ICP sur la carte de référence. La correction exprimée dans le repère base
    mesure l'incohérence entre GT, T_base_lidar et world (≈ 0 attendu). Sert aussi d'arbitre si le gyroscope
    et la calibration IMU divergent.
    """
    import os

    import numpy as np
    from scipy.spatial import cKDTree

    from splatpipe.common import geometry as g

    map_path = f"{DATA_MOUNT}/{RAW_ROOT}/ground_truth_map/{site}/merged-clouds-5cm.pcd"
    if not os.path.exists(map_path):
        return {"status": "WARN", "reason": f"carte absente : {map_path}"}
    mp = _read_pcd(map_path)
    map_xyz = np.stack([mp["x"], mp["y"], mp["z"]], axis=1).astype(np.float64)
    del mp

    t_scan = g.ns_to_s(lidar_ns)
    t_mid = t_scan + 0.05
    vel, om, ok = _gt_kinematics(tg, pg, qg, t_mid)
    speed = np.linalg.norm(vel, axis=1)
    angrate = np.degrees(np.linalg.norm(om, axis=1))
    ok &= (t_scan > tg[0] + 0.3) & (t_scan + 0.1 < tg[-1] - 0.3)
    score = np.where(ok, speed + angrate / 10.0, np.inf)
    chosen = []
    for i in np.argsort(score):
        if not np.isfinite(score[i]) or len(chosen) >= 3:
            break
        if all(abs(t_scan[i] - t_scan[j]) > 2.0 for j in chosen):
            chosen.append(int(i))
    valid_idx = np.nonzero(ok)[0]
    for qtl in (0.2, 0.5, 0.8):
        chosen.append(int(valid_idx[int(qtl * (len(valid_idx) - 1))]))

    per_scan, body_rv, body_t = [], [], []
    for i in chosen:
        a = _read_pcd(f"{seq_dir}/raw/lidar-clouds/{lidar_names[i]}")
        xyz = np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)
        ts = a["timestamp"].astype(np.float64)
        r = np.linalg.norm(xyz, axis=1)
        keep = np.isfinite(r) & (r > 3.0) & (r < 40.0)
        idx = np.nonzero(keep)[0][::3]
        pw, qw, vv = g.interpolate_poses(tg, pg, qg, ts[idx], max_gap_s=0.25)
        idx, pw, qw = idx[vv], pw[vv], qw[vv]
        p_base = g.transform_points(T_base_lidar, xyz[idx])
        p_world = (g.quat_to_rot(qw) @ p_base[:, :, None])[:, :, 0] + pw
        # Version sans désanamorphose (pose unique au milieu du tour), pour mesurer l'effet du mouvement.
        pm, qm, _ = g.interpolate_poses(tg, pg, qg, np.array([t_mid[i]]))
        T_wb = g.se3(pm[0], qm[0])
        p_world_rigid = g.transform_points(T_wb @ T_base_lidar, xyz[idx])
        c = pm[0]
        crop = np.all(np.abs(map_xyz[:, :2] - c[:2]) < 45.0, axis=1)
        dst = map_xyz[crop]
        tree = cKDTree(dst)
        T_corr, st = _icp(p_world, tree, dst)
        _, st_rigid = _icp(p_world_rigid, tree, dst, schedule=(0.5,), iters=1)
        C_b = g.inv(T_wb) @ T_corr @ T_wb  # correction dans le repère base : T_world_base_vrai ≈ T_wb · C_b
        rv_b = np.degrees(g.rot_to_rotvec(C_b[:3, :3]))
        body_rv.append(rv_b)
        body_t.append(C_b[:3, 3])
        per_scan.append(
            {
                "file": lidar_names[i],
                "t_rel_s": round(float(t_scan[i] - tg[0]), 2),
                "gt_speed_ms": float(speed[i]),
                "gt_angrate_deg_s": float(angrate[i]),
                "n_points_used": int(len(idx)),
                "n_map_points_crop": int(len(dst)),
                "icp_before": st["before"],
                "icp_after": st["after"],
                "no_deskew_before": st_rigid["before"],
                "correction_world_rotvec_deg": np.degrees(g.rot_to_rotvec(T_corr[:3, :3])),
                "correction_world_translation_m": T_corr[:3, 3],
                "correction_base_rotvec_deg": rv_b,
                "correction_base_translation_m": C_b[:3, 3],
                "correction_angle_deg": float(np.linalg.norm(rv_b)),
            }
        )
    body_rv = np.array(body_rv)
    body_t = np.array(body_t)
    med_angle = float(np.median(np.linalg.norm(body_rv, axis=1)))
    med_trans = float(np.median(np.linalg.norm(body_t, axis=1)))
    gyro_corr = np.degrees(g.rot_to_rotvec(R_base_imu_est @ R_base_imu.T))
    ok_ = med_angle < LIDAR_CLOSURE_ROT_TOL_DEG and med_trans < LIDAR_CLOSURE_TRANS_TOL_M
    return {
        "method": "désanamorphose par point avec GT·T_base_lidar puis ICP point-à-point sur "
        f"ground_truth_map/{site}/merged-clouds-5cm.pcd (crop ±45 m)",
        "scans": per_scan,
        "median_correction_base_rotvec_deg": np.median(body_rv, axis=0),
        "median_correction_angle_deg": med_angle,
        "median_correction_translation_m": med_trans,
        "gyro_correction_of_R_base_imu_rotvec_deg": gyro_corr,
        "pass": bool(ok_),
        "status": "PASS" if ok_ else "FAIL",
    }


# =============================================================================
# Fonction Modal
# =============================================================================


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=4, memory=16384, timeout=1800)
def ingest(seq: str) -> dict:
    import csv
    import datetime as dt
    import os
    import time

    import numpy as np

    from splatpipe.common import geometry as g
    from splatpipe.common.scene import Scene

    t_start = time.time()
    vol.reload()
    seq_rel = f"{RAW_ROOT}/sequences/{seq}"
    seq_dir = f"{DATA_MOUNT}/{seq_rel}"
    calib_dir = f"{DATA_MOUNT}/{RAW_ROOT}/calibration"
    site = seq.split("-", 3)[3].rsplit("-", 1)[0]  # 2024-03-13-observatory-quarter-01 → observatory-quarter
    sc = Scene(DATASET, seq, data_root=DATA_MOUNT)
    sc.makedirs()
    for cam in CAMS:
        os.makedirs(sc.path("cameras", cam), exist_ok=True)
    report: dict = {
        "stage": "stage0_ingest",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset": DATASET,
        "sequence": seq,
        "scene_root": sc.root,
        "tolerances": {
            "triplet_tol_ms": TRIPLET_TOL_NS / 1e6,
            "rate_rel_tol": RATE_REL_TOL,
            "drop_threshold": "dt > 1,5 × période médiane (au moins un échantillon manquant)",
            "gap_gt_2_periods_threshold": "dt > 2,5 × période médiane (au moins 2 échantillons consécutifs manquants)",
            "acc_norm_tol_ms2": ACC_NORM_TOL,
            "gravity_angle_tol_deg": GRAVITY_ANGLE_TOL_DEG,
            "cam_axis_tol_deg": CAM_AXIS_TOL_DEG,
            "baseline_range_m": list(BASELINE_RANGE_M),
            "gyro_extrinsic_convention_tol_deg": GYRO_EXTRINSIC_CONVENTION_TOL_DEG,
            "gyro_extrinsic_precision_tol_deg": GYRO_EXTRINSIC_PRECISION_TOL_DEG,
            "cam_handeye_convention_tol_deg": CAM_HANDEYE_CONVENTION_TOL_DEG,
            "cam_handeye_precision_tol_deg": CAM_HANDEYE_PRECISION_TOL_DEG,
            "lidar_closure_rot_tol_deg": LIDAR_CLOSURE_ROT_TOL_DEG,
            "lidar_closure_trans_tol_m": LIDAR_CLOSURE_TRANS_TOL_M,
            "gyro_time_offset_tol_s": GYRO_TIME_OFFSET_TOL_S,
            "static": {"speed_max_ms": STATIC_SPEED_MAX, "angrate_max_deg_s": STATIC_ANGRATE_MAX_DEG,
                       "min_duration_s": STATIC_MIN_DURATION_S},
        },
    }

    # ------------------------------------------------------------------ caméras
    cam_ns, cam_names, t01 = {}, {}, {}
    for cam in CAMS:
        ns, names, n_dup = _list_stamped(f"{seq_dir}/raw/{cam}", "jpg")
        cam_ns[cam], cam_names[cam] = ns, names
        t01[cam] = _series_check(ns, NOMINAL_HZ["camera"], n_dup=n_dup)
        with open(sc.camera_index_path(cam), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stamp_ns", "t", "rel_path"])
            for s, name in zip(ns, names):
                w.writerow([int(s), g.format_stamp_ns(s), f"{seq_rel}/raw/{cam}/{name}"])
    # Les trous d'une caméra coïncident-ils avec ceux des autres ?
    for cam in CAMS:
        for gap in t01[cam]["gaps"]:
            lo = gap["stamp_ns_before"] + 1_000_000
            hi = gap["stamp_ns_after"] - 1_000_000
            gap["frames_in_gap_other_cams"] = {
                o: int(np.sum((cam_ns[o] > lo) & (cam_ns[o] < hi))) for o in CAMS if o != cam
            }

    # Triplets synchronisés (référence cam0).
    ref = cam_ns["cam0"]
    idx = {"cam0": np.arange(len(ref))}
    ok = np.ones(len(ref), dtype=bool)
    for cam in ("cam1", "cam2"):
        j, d = _nearest(cam_ns[cam], ref)
        idx[cam] = j
        ok &= d <= TRIPLET_TOL_NS
    trip = {c: idx[c][ok] for c in CAMS}
    for cam in ("cam1", "cam2"):
        assert len(np.unique(trip[cam])) == len(trip[cam]), f"appariement non unique pour {cam}"
    trip_ns = ref[ok]
    spread_ns = np.max(np.stack([cam_ns[c][trip[c]] for c in CAMS]), axis=0) - np.min(
        np.stack([cam_ns[c][trip[c]] for c in CAMS]), axis=0
    )
    with open(sc.frames_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "stamp_ns", "t"] + [f"{c}_idx" for c in CAMS] + [f"{c}_rel_path" for c in CAMS])
        for k in range(len(trip_ns)):
            w.writerow(
                [k, int(trip_ns[k]), g.format_stamp_ns(trip_ns[k])]
                + [int(trip[c][k]) for c in CAMS]
                + [f"{seq_rel}/raw/{c}/{cam_names[c][trip[c][k]]}" for c in CAMS]
            )
    union_ns = np.unique(np.concatenate([cam_ns[c] for c in CAMS]))
    triplets = {
        "n_triplets": int(len(trip_ns)),
        "tolerance_ms": TRIPLET_TOL_NS / 1e6,
        "unmatched_images": {c: int(len(cam_ns[c]) - len(trip_ns)) for c in CAMS},
        "n_unique_stamps_all_cams": int(len(union_ns)),
        "max_intra_triplet_spread_ns": int(spread_ns.max()) if len(spread_ns) else None,
        "n_triplets_exact_same_stamp": int(np.sum(spread_ns == 0)),
        "triplet_series": _series_check(trip_ns, NOMINAL_HZ["camera"]),
    }
    triplets["triplet_series"].pop("gaps")  # déjà détaillés par caméra

    # ------------------------------------------------------------------ IMU
    imu_csv = f"{seq_dir}/raw/imu.csv"
    with open(imu_csv) as f:
        header = f.readline().strip()
    if header != IMU_HEADER:
        raise RuntimeError(f"en-tête IMU inattendu : {header!r}")
    raw = np.loadtxt(imu_csv, delimiter=",", skiprows=1, dtype=np.float64)
    secs = raw[:, 0].astype(np.int64)
    nsecs = raw[:, 1].astype(np.int64)
    assert np.all((nsecs >= 0) & (nsecs < 1_000_000_000))
    imu_ns_raw = secs * 1_000_000_000 + nsecs
    unsorted_steps = int(np.sum(np.diff(imu_ns_raw) < 0))
    order = np.argsort(imu_ns_raw, kind="stable")
    imu_ns = imu_ns_raw[order]
    keep = np.ones(len(imu_ns), dtype=bool)
    keep[1:] = np.diff(imu_ns) > 0
    n_dup_imu = int((~keep).sum())
    imu_ns = imu_ns[keep]
    acc = raw[order][keep, 2:5]
    gyro = raw[order][keep, 5:8]
    t_imu = g.ns_to_s(imu_ns)
    t01["imu"] = _series_check(imu_ns, NOMINAL_HZ["imu"], n_dup=n_dup_imu, raw_unsorted_steps=unsorted_steps)
    np.savez(sc.imu_path, t=t_imu, stamp_ns=imu_ns, acc=acc, gyro=gyro)

    # ------------------------------------------------------------------ LiDAR (horodatages des fichiers seulement)
    lidar_ns, lidar_names, n_dup_l = _list_stamped(f"{seq_dir}/raw/lidar-clouds", "pcd")
    t01[LIDAR_ID] = _series_check(lidar_ns, NOMINAL_HZ["lidar"], n_dup=n_dup_l)

    # ------------------------------------------------------------------ calibration
    sensors, calib_checks = _build_sensors(calib_dir, t01["imu"]["median_rate_hz"])
    _write_sensors_yaml(sc.sensors_path, sensors)
    T_base_imu = np.array(sensors["imu"]["T_base_imu"])
    T_base_cam = {c: np.array(sensors["cameras"][c]["T_base_cam"]) for c in CAMS}
    T_base_lidar = np.array(sensors["lidars"][LIDAR_ID]["T_base_lidar"])
    R_base_imu = T_base_imu[:3, :3]

    # ------------------------------------------------------------------ trajectoires
    traj_src = {
        "gt": "gt-tum.txt",
        "vilens": "vilens-slam-tum.txt",
        "hba": "hba-tum.txt",
        "colmap": "colmap-tum.txt",
    }
    # Repère « corps » candidat des trajectoires locales (identifié par rotations relatives vs GT).
    cands = {
        "base": np.eye(3),
        "lidar": T_base_lidar[:3, :3],
        "cam0": T_base_cam["cam0"][:3, :3],
        "cam0_opengl": T_base_cam["cam0"][:3, :3] @ np.diag([1.0, -1.0, -1.0]),
        "imu": R_base_imu,
    }
    traj_info, trajs = {}, {}
    for name, fname in traj_src.items():  # gt en premier (référence de l'identification)
        src = f"{seq_dir}/processed/trajectory/{fname}"
        ns, pos, quat = g.read_tum_ns(src)
        ns, pos, quat, info = g.clean_trajectory(ns, pos, quat)
        if name == "gt":
            frame = "T_world_base, world = repère du scan laser fixe (Leica RTC360)"
        else:
            tg_ns, pg, qg = trajs["gt"]
            ident = _identify_body_frame(g.ns_to_s(ns), g.quat_to_rot(quat), g.ns_to_s(tg_ns), pg, qg, cands)
            info["body_frame_identification"] = ident
            storage, body = ident["best"].split(":")
            err = ident["hypotheses"][ident["best"]]["median_err_deg"]
            if storage == "T_local_body" and err < 1.0:
                frame = f"T_local_{body} (corps identifié, erreur médiane {err:.2f}°) ; repère local propre à la méthode, PAS world"
            else:
                frame = "repère local propre à la méthode, PAS world ; corps non identifié"
        g.write_tum(
            sc.traj_path(name),
            ns,
            pos,
            quat,
            header=f"t tx ty tz qx qy qz qw\n{name}: {frame}\nsource: {seq_rel}/processed/trajectory/{fname}",
        )
        trajs[name] = (ns, pos, quat)
        ser = _series_check(ns, 20.0 if name == "gt" else 1.25)
        info.update(
            {
                "source": f"{seq_rel}/processed/trajectory/{fname}",
                "file": f"traj/{name}.tum",
                "frame": frame,
                "t0": ser["t0"],
                "t1": ser["t1"],
                "median_rate_hz": ser["median_rate_hz"],
                "max_gap_s": ser["max_gap_s"],
                "path_length_m": float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()),
            }
        )
        traj_info[name] = info
    tg_ns, pg, qg = trajs["gt"]
    tg = g.ns_to_s(tg_ns)

    # ------------------------------------------------------------------ GT vs horodatages caméra / LiDAR
    _, d_trip = _nearest(trip_ns, tg_ns)
    _, d_union = _nearest(union_ns, tg_ns)
    _, d_lidar = _nearest(lidar_ns, tg_ns)
    on_cam = d_union <= 500_000
    on_lidar = (d_lidar <= 500_000) & ~on_cam
    _, d_trip_to_gt = _nearest(tg_ns, trip_ns)
    _, d_lidar_to_gt = _nearest(tg_ns, lidar_ns)
    gt_align = {
        "abs_dt_gt_to_nearest_triplet_ms": _pct(d_trip * 1e-6),
        "abs_dt_gt_to_nearest_any_camera_stamp_ms": _pct(d_union * 1e-6),
        "n_gt": int(len(tg_ns)),
        "n_gt_on_camera_stamp_le_0.5ms": int(on_cam.sum()),
        "n_gt_exactly_on_camera_stamp": int(np.sum(d_union == 0)),
        "n_gt_on_lidar_file_stamp_le_0.5ms": int(on_lidar.sum()),
        "n_gt_other": int(np.sum(~on_cam & ~on_lidar)),
        "frac_triplets_with_gt_within_0.5ms": float(np.mean(d_trip_to_gt <= 500_000)),
        "frac_triplets_with_gt_exact": float(np.mean(d_trip_to_gt == 0)),
        "frac_images_with_gt_exact": {c: float(np.mean(_nearest(tg_ns, cam_ns[c])[1] == 0)) for c in CAMS},
        "frac_lidar_scans_with_gt_within_0.5ms": float(np.mean(d_lidar_to_gt <= 500_000)),
        "gt_t0_minus_first_image_s": float((tg_ns[0] - union_ns[0]) * 1e-9),
        "gt_t1_minus_last_image_s": float((tg_ns[-1] - union_ns[-1]) * 1e-9),
    }
    other = np.nonzero(~on_cam & ~on_lidar)[0]
    gt_align["gt_other_examples"] = [
        {
            "t_rel_to_first_image_s": round(float((tg_ns[i] - union_ns[0]) * 1e-9), 4),
            "dt_nearest_camera_ms": round(float(d_union[i] * 1e-6), 3),
            "dt_nearest_lidar_ms": round(float(d_lidar[i] * 1e-6), 3),
        }
        for i in other[:40]
    ]
    # Cohérence des poses GT aux temps LiDAR avec l'interpolation des poses aux temps caméra.
    if on_cam.sum() > 2 and on_lidar.sum() > 0:
        pc, qc, vc = g.interpolate_poses(tg[on_cam], pg[on_cam], qg[on_cam], tg[on_lidar], max_gap_s=0.12)
        dpos = np.linalg.norm(pc - pg[on_lidar], axis=1)[vc]
        drot = np.degrees(g.rotation_angle_between(g.quat_to_rot(qc), g.quat_to_rot(qg[on_lidar])))[vc]
        gt_align["lidar_stamp_poses_vs_camera_stamp_interp"] = {
            "n": int(vc.sum()),
            "dpos_m": _pct(dpos, (50, 90, 99, 100)),
            "drot_deg": _pct(drot, (50, 90, 99, 100)),
        }

    # ================================================================== T0.5
    t05: dict = {}
    # (c) extrinsèques
    der = sensors["derived"]
    axis_ok = all(v < CAM_AXIS_TOL_DEG for v in der["camera_axis_angle_to_expected_deg"].values())
    base_ok = all(BASELINE_RANGE_M[0] <= v <= BASELINE_RANGE_M[1] for v in der["camera_baselines_m"].values())
    mq = calib_checks["T_cam_lidar_matrix_vs_quaternion"]
    mq_ok = all(v["rot_diff_deg"] < 0.01 and v["trans_diff_m"] < 1e-4 and v["orthonormality_err"] < 1e-6 for v in mq.values())
    imu_up_ok = der["imu_z_angle_to_base_z_deg"] < CAM_AXIS_TOL_DEG
    t05["c_extrinsics"] = {
        "camera_axis_angle_to_expected_deg": der["camera_axis_angle_to_expected_deg"],
        "camera_baselines_m": der["camera_baselines_m"],
        "camera_relative_rotation_deg": der["camera_relative_rotation_deg"],
        "imu_z_angle_to_base_z_deg": der["imu_z_angle_to_base_z_deg"],
        "calibration_matrix_vs_quaternion": mq,
        "C_q_CI_norm": calib_checks["C_q_CI_norm"],
        "pass_axes": axis_ok,
        "pass_baselines": base_ok,
        "pass_matrix_quaternion": mq_ok,
        "pass_imu_up": imu_up_ok,
        "status": "PASS" if (axis_ok and base_ok and mq_ok and imu_up_ok) else "FAIL",
    }

    # IMU dans le monde via la GT.
    _, q_imu, v_imu = g.interpolate_poses(tg, pg, qg, t_imu, max_gap_s=0.25)
    R_world_imu = g.quat_to_rot(q_imu) @ R_base_imu
    acc_world = np.einsum("nij,nj->ni", R_world_imu, acc)
    acc_norm = np.linalg.norm(acc, axis=1)

    # Périodes quasi statiques (GT) sur une grille de 0,05 s.
    grid = np.arange(max(tg[0], t_imu[0]) + 0.1, min(tg[-1], t_imu[-1]) - 0.1, 0.05)
    vel, om, vk = _gt_kinematics(tg, pg, qg, grid)
    speed = np.linalg.norm(vel, axis=1)
    angrate = np.degrees(np.linalg.norm(om, axis=1))
    static = vk & (speed < STATIC_SPEED_MAX) & (angrate < STATIC_ANGRATE_MAX_DEG)
    segs = []
    if static.any():
        sidx = np.nonzero(static)[0]
        for s in np.split(sidx, np.nonzero(np.diff(sidx) > 1)[0] + 1):
            a_, b_ = grid[s[0]] - 0.025, grid[s[-1]] + 0.025
            if b_ - a_ < STATIC_MIN_DURATION_S:
                continue
            m = v_imu & (t_imu >= a_) & (t_imu <= b_)
            if m.sum() < 50:
                continue
            mw = acc_world[m].mean(0)
            segs.append(
                {
                    "t_start": float(a_),
                    "t_rel_s": round(float(a_ - t_imu[0]), 2),
                    "duration_s": round(float(b_ - a_), 2),
                    "n_imu": int(m.sum()),
                    "max_gt_speed_ms": float(speed[s].max()),
                    "max_gt_angrate_deg_s": float(angrate[s].max()),
                    "acc_norm_mean": float(acc_norm[m].mean()),
                    "acc_norm_std": float(acc_norm[m].std()),
                    "norm_of_mean_acc": float(np.linalg.norm(acc[m].mean(0))),
                    "gyro_norm_mean": float(np.linalg.norm(gyro[m], axis=1).mean()),
                    "mean_acc_imu": acc[m].mean(0),
                    "mean_acc_world": mw,
                    "gravity_angle_err_deg": float(np.degrees(g.angle_between_vectors(mw, [0, 0, 1]))),
                    "_mask": m,
                }
            )
    static_mask = np.zeros(len(t_imu), dtype=bool)
    for s in segs:
        static_mask |= s.pop("_mask")
    total_static = float(sum(s["duration_s"] for s in segs))

    # Estimateur robuste sur toute la séquence : moyenne de R_world_imu·f = g + Δv/T (Δv ≈ 0).
    whole = acc_world[v_imu].mean(0)
    # Fenêtres de 5 s compensées du mouvement : moyenne(R·f) − Δv_GT/W = g.
    W = 5.0
    win = []
    for s0 in np.arange(max(tg[0], t_imu[0]) + 0.3, min(tg[-1], t_imu[-1]) - W - 0.3, W):
        m = v_imu & (t_imu >= s0) & (t_imu < s0 + W)
        vv, _, vok = _gt_kinematics(tg, pg, qg, np.array([s0, s0 + W]))
        if m.sum() < 0.9 * W * NOMINAL_HZ["imu"] or not vok.all():
            continue
        ge = acc_world[m].mean(0) - (vv[1] - vv[0]) / W
        raw_mean = acc_world[m].mean(0)
        win.append(
            (
                np.degrees(g.angle_between_vectors(ge, [0, 0, 1])),
                np.linalg.norm(ge),
                np.degrees(g.angle_between_vectors(raw_mean, [0, 0, 1])),
            )
        )
    win = np.array(win)
    g_hat = float(np.linalg.norm(whole))
    # Résidu moyen dans le repère IMU (≈ biais accéléromètre + désalignement roulis/tangage).
    R_imu_world = np.swapaxes(R_world_imu[v_imu], -1, -2)
    resid_imu = (acc[v_imu] - R_imu_world @ np.array([0.0, 0.0, g_hat])).mean(0)

    if static_mask.any():
        a_norm_static = {
            "n_segments": len(segs),
            "total_duration_s": total_static,
            "acc_norm_mean": float(acc_norm[static_mask].mean()),
            "acc_norm_std": float(acc_norm[static_mask].std()),
            "norm_of_mean_acc_world": float(np.linalg.norm(acc_world[static_mask].mean(0))),
        }
    else:
        a_norm_static = {"n_segments": 0, "total_duration_s": 0.0}
    a_ref = a_norm_static.get("norm_of_mean_acc_world", g_hat)
    t05["a_accelerometer_norm"] = {
        "reference_g_ms2": G_REF,
        "units_inferred": "m/s^2" if 5.0 < g_hat < 15.0 else ("g" if 0.5 < g_hat < 1.5 else "inconnu"),
        "quasi_static": a_norm_static,
        "static_segments": segs,
        "whole_sequence_norm_of_mean_world_ms2": g_hat,
        "compensated_5s_windows_norm_mean_ms2": float(win[:, 1].mean()) if len(win) else None,
        "compensated_5s_windows_norm_std_ms2": float(win[:, 1].std()) if len(win) else None,
        "all_samples_acc_norm_pct": _pct(acc_norm, (1, 50, 99)),
        "deviation_from_ref_ms2": float(a_ref - G_REF),
        "deviation_from_ref_rel": float(a_ref / G_REF - 1.0),
        "pass": bool(abs(a_ref - G_REF) < ACC_NORM_TOL and abs(g_hat - G_REF) < ACC_NORM_TOL),
    }
    t05["a_accelerometer_norm"]["status"] = "PASS" if t05["a_accelerometer_norm"]["pass"] else "FAIL"
    if abs(a_ref - G_REF) > 0.1:
        t05["a_accelerometer_norm"]["note"] = (
            f"norme mesurée {a_ref:.3f} m/s² ({100 * (a_ref / G_REF - 1):+.2f} %) : biais/facteur d'échelle "
            "accéléromètre à estimer si l'IMU est utilisée (offset BMI085 typique ±20 mg ≈ ±0,2 m/s²)."
        )

    static_angle = (
        float(np.degrees(g.angle_between_vectors(acc_world[static_mask].mean(0), [0, 0, 1])))
        if static_mask.any()
        else None
    )
    whole_angle = float(np.degrees(g.angle_between_vectors(whole, [0, 0, 1])))
    b_pass = (
        (static_angle is None or static_angle < GRAVITY_ANGLE_TOL_DEG)
        and whole_angle < GRAVITY_ANGLE_TOL_DEG
        and (len(win) == 0 or np.percentile(win[:, 0], 95) < GRAVITY_ANGLE_TOL_DEG)
    )
    t05["b_gravity_direction"] = {
        "chain": "a_world = R_world_base(t) · R_base_imu · f_imu, attendu (0, 0, +g)",
        "quasi_static_mean_acc_world": acc_world[static_mask].mean(0) if static_mask.any() else None,
        "quasi_static_angle_err_deg": static_angle,
        "whole_sequence_mean_acc_world": whole,
        "whole_sequence_angle_err_deg": whole_angle,
        "compensated_5s_windows": {
            "n": int(len(win)),
            "angle_err_deg": _pct(win[:, 0], (50, 95, 100)) if len(win) else {},
            "uncompensated_angle_err_deg": _pct(win[:, 2], (50, 95, 100)) if len(win) else {},
        },
        "mean_residual_in_imu_frame_ms2": resid_imu,
        "equivalent_tilt_of_residual_deg": float(np.degrees(np.arctan2(np.linalg.norm(resid_imu[:2]), g_hat))),
        "pass": bool(b_pass),
    }
    t05["b_gravity_direction"]["status"] = "PASS" if b_pass else "FAIL"

    # (d) gyroscope vs vitesse angulaire GT : rotation R_imu_base (Wahba/Kabsch) et décalage temporel.
    h = 0.05
    tt = np.arange(max(tg[0], t_imu[0]) + 0.3, min(tg[-1], t_imu[-1]) - 0.3, 0.05)
    csum = np.concatenate([np.zeros((1, 3)), np.cumsum(gyro, axis=0)])
    i0 = np.searchsorted(t_imu, tt - h)
    i1 = np.searchsorted(t_imu, tt + h)
    gyro_avg = (csum[i1] - csum[i0]) / np.maximum(i1 - i0, 1)[:, None]

    def gt_omega_imu(delta, R_ib=R_base_imu.T):
        _, om_b, okk = _gt_kinematics(tg, pg, qg, tt + delta, h=h)
        return om_b @ R_ib.T, om_b, okk

    offs = np.arange(-0.05, 0.0501, 0.001)
    rms = []
    for d_ in offs:
        om_i, _, okk = gt_omega_imu(d_)
        r = gyro_avg[okk] - om_i[okk]
        r = r - r.mean(0)
        rms.append(np.sqrt((r**2).sum(1).mean()))
    rms = np.array(rms)
    kbest = int(np.argmin(rms))
    best = float(offs[kbest])
    if 0 < kbest < len(offs) - 1:  # raffinement parabolique
        y0, y1, y2 = rms[kbest - 1], rms[kbest], rms[kbest + 1]
        den = y0 - 2 * y1 + y2
        if den > 0:
            best += float(0.5 * (y0 - y2) / den * 0.001)
    def kabsch_R_imu_base(delta):
        om_i, om_b, okk = gt_omega_imu(delta)
        A_ = gyro_avg[okk] - gyro_avg[okk].mean(0)
        B_ = om_b[okk] - om_b[okk].mean(0)
        U, _, Vt = np.linalg.svd(B_.T @ A_)
        dsign = np.sign(np.linalg.det(Vt.T @ U.T))
        return Vt.T @ np.diag([1.0, 1.0, dsign]) @ U.T, om_i, okk

    R_ib_0, om_i0, ok0 = kabsch_R_imu_base(0.0)
    half = tt < np.median(tt)
    halves = {}
    for hname, hm in (("first_half", half), ("second_half", ~half)):
        _, om_bh, okh = _gt_kinematics(tg, pg, qg, tt, h=h)
        okh &= hm
        A_ = gyro_avg[okh] - gyro_avg[okh].mean(0)
        B_ = om_bh[okh] - om_bh[okh].mean(0)
        U, _, Vt = np.linalg.svd(B_.T @ A_)
        R_h = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        halves[hname] = np.degrees(g.rot_to_rotvec(R_h.T @ R_base_imu.T))
    R_ib_best, _, _ = kabsch_R_imu_base(best)
    gyro_bias = (gyro_avg[ok0] - om_i0[ok0]).mean(0)
    corr = [float(np.corrcoef(gyro_avg[ok0][:, k], om_i0[ok0][:, k])[0, 1]) for k in range(3)]
    R_base_imu_est = R_ib_best.T
    ext_err = float(np.degrees(g.rotation_angle_between(R_base_imu_est, R_base_imu)))
    ext_err_0 = float(np.degrees(g.rotation_angle_between(R_ib_0.T, R_base_imu)))
    # Correction à appliquer (repère base) : R_base_imu_est = C · R_base_imu.
    corr_rotvec_base_deg = np.degrees(g.rot_to_rotvec(R_base_imu_est @ R_base_imu.T))
    # Hypothèse « GT = T_world_lidar » (erreur de repère) pour montrer le pouvoir discriminant du test.
    ext_err_if_lidar = float(
        np.degrees(g.rotation_angle_between(R_base_imu_est, T_base_lidar[:3, :3].T @ R_base_imu))
    )
    conv_ok = ext_err < GYRO_EXTRINSIC_CONVENTION_TOL_DEG and abs(best) < GYRO_TIME_OFFSET_TOL_S and min(corr) > 0.95
    # Critère de précision évalué sur la configuration stockée (time_offsets = 0).
    prec_ok = ext_err_0 < GYRO_EXTRINSIC_PRECISION_TOL_DEG
    t05["d_gyro_vs_gt"] = {
        "method": "vitesse angulaire GT (différences ±50 ms) ramenée dans le repère imu vs gyro moyenné sur la même fenêtre ; "
        "rotation R_imu_base estimée par Kabsch/Wahba (biais retiré par centrage)",
        "n_samples": int(ok0.sum()),
        "corr_per_axis": corr,
        "rms_residual_rad_s_at_0": float(rms[np.argmin(np.abs(offs))]),
        "rms_residual_rad_s_best": float(rms[kbest]),
        "best_time_offset_s": best,
        "time_offset_convention": "gyro(t) ≈ ω_GT(t + offset) : offset à ajouter aux temps IMU pour les caler sur la GT",
        "gyro_bias_estimate_rad_s": gyro_bias,
        "R_base_imu_estimated_vs_calibrated_deg": ext_err,
        "R_base_imu_estimated_vs_calibrated_deg_at_offset_0": ext_err_0,
        "correction_rotvec_in_base_deg": corr_rotvec_base_deg,
        "correction_rotvec_in_base_deg_by_half_at_offset_0": halves,
        "imu_x_axis_yaw_in_base_deg": {
            "calibrated": float(np.degrees(np.arctan2(R_base_imu[1, 0], R_base_imu[0, 0]))),
            "estimated": float(np.degrees(np.arctan2(R_base_imu_est[1, 0], R_base_imu_est[0, 0]))),
        },
        "R_base_imu_estimated": R_base_imu_est,
        "R_base_imu_estimated_vs_hypothesis_gt_is_lidar_frame_deg": ext_err_if_lidar,
        "gyro_norm_mean_rad_s": float(np.linalg.norm(gyro, axis=1).mean()),
        "pass_convention": bool(conv_ok),
        "pass_precision_1deg": bool(prec_ok),
    }
    t05["d_gyro_vs_gt"]["status"] = "PASS" if (conv_ok and prec_ok) else ("WARN" if conv_ok else "FAIL")

    # (e) Fermeture LiDAR → world : scans bruts désanamorphosés avec GT · T_base_lidar, recalés (ICP) sur le scan fixe.
    t05["e_lidar_world_closure"] = _lidar_world_closure(
        seq_dir, site, lidar_ns, lidar_names, tg, pg, qg, T_base_lidar, R_base_imu, R_base_imu_est
    )

    # (f) Rotation cam0 indépendante : trajectoire COLMAP (repère cam0 OpenGL) vs GT, par hand-eye rotationnel.
    nc_, _, qc_ = trajs["colmap"]
    tc_ = g.ns_to_s(nc_)
    Rc_ = g.quat_to_rot(qc_) @ np.diag([1.0, -1.0, -1.0])  # OpenGL → OpenCV : T_local_cam0
    R_bc0 = T_base_cam["cam0"][:3, :3]
    by_lag, by_half = {}, {}
    for lag in (1, 2, 3):
        X, n_, S_ = _handeye_rotation(tc_, Rc_, tg, pg, qg, lag=lag)
        by_lag[f"lag{lag}"] = {
            "n_pairs": n_,
            "singular_values": S_,
            "correction_rotvec_in_base_deg": np.degrees(g.rot_to_rotvec(X @ R_bc0.T)),
            "angle_deg": float(np.degrees(g.rotation_angle_between(X, R_bc0))),
        }
    half_c = tc_ < np.median(tc_)
    for hname, hm in (("first_half", half_c), ("second_half", ~half_c)):
        X, n_, _ = _handeye_rotation(tc_, Rc_, tg, pg, qg, lag=1, sel=hm)
        by_half[hname] = {"n_pairs": n_, "correction_rotvec_in_base_deg": np.degrees(g.rot_to_rotvec(X @ R_bc0.T))}
    cam_angle = float(np.median([v["angle_deg"] for v in by_lag.values()]))
    cam_rv = np.median(np.array([v["correction_rotvec_in_base_deg"] for v in by_lag.values()]), axis=0)
    fx0 = sensors["cameras"]["cam0"]["fx"]
    t05["f_cam0_rotation_vs_colmap"] = {
        "method": "hand-eye rotationnel entre rotations relatives COLMAP (cam0, convention OpenGL convertie en OpenCV) "
        "et GT (base) ; invariant à l'alignement global de COLMAP. Indépendant de DiffCal si COLMAP a été lancé sans a priori d'extrinsèques.",
        "by_lag": by_lag,
        "by_half_lag1": by_half,
        "median_angle_deg": cam_angle,
        "median_correction_rotvec_in_base_deg": cam_rv,
        "equivalent_shift_at_image_center_px": float(fx0 * np.radians(cam_angle)),
        "gyro_correction_of_R_base_imu_rotvec_deg": corr_rotvec_base_deg,
        "pass_convention": bool(cam_angle < CAM_HANDEYE_CONVENTION_TOL_DEG),
        "pass_precision": bool(cam_angle < CAM_HANDEYE_PRECISION_TOL_DEG),
    }
    t05["f_cam0_rotation_vs_colmap"]["status"] = (
        "PASS" if cam_angle < CAM_HANDEYE_PRECISION_TOL_DEG else ("WARN" if cam_angle < CAM_HANDEYE_CONVENTION_TOL_DEG else "FAIL")
    )

    statuses = [t05[k]["status"] for k in ("a_accelerometer_norm", "b_gravity_direction", "c_extrinsics",
                                            "d_gyro_vs_gt", "e_lidar_world_closure", "f_cam0_rotation_vs_colmap")]
    t05["status"] = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")

    # ================================================================== T0.1 synthèse
    cams_gaps = {c: t01[c]["n_gaps_gt_2_periods"] for c in CAMS}
    strict_pass = all(t01[k]["status"] == "PASS" for k in list(CAMS) + ["imu", LIDAR_ID])
    only_cam_gaps = all(
        t01[k]["pass_monotonic"] and t01[k]["pass_rate"] for k in list(CAMS) + ["imu", LIDAR_ID]
    ) and t01["imu"]["pass_no_gap"] and t01[LIDAR_ID]["pass_no_gap"]
    max_cam_gap = max(t01[c]["max_gap_s"] for c in CAMS)
    report["T0.1"] = {
        "series": t01,
        "triplets": triplets,
        "gt_time_alignment": gt_align,
        "status_strict": "PASS" if strict_pass else "FAIL",
        "summary": {
            "camera_gaps_gt_2_periods": cams_gaps,
            "camera_drops_any": {c: t01[c]["n_drops"] for c in CAMS},
            "camera_drops_single_frame": {c: t01[c]["n_drops_single"] for c in CAMS},
            "camera_missing_frames_est": {c: t01[c]["n_missing_est"] for c in CAMS},
            "max_camera_gap_s": max_cam_gap,
            "imu_gaps_gt_2_periods": t01["imu"]["n_gaps_gt_2_periods"],
            "lidar_gaps_gt_2_periods": t01[LIDAR_ID]["n_gaps_gt_2_periods"],
        },
        "recommendation": (
            "Aucune action."
            if strict_pass
            else (
                "ACCEPTER : seules des images caméra isolées manquent (trous ≤ "
                f"{max_cam_gap:.3f} s, soit ≤ {int(round(max_cam_gap / 0.05)) - 1} images consécutives) ; "
                "horodatages sur grille matérielle régulière, IMU et LiDAR sans trou. On travaille sur les "
                "triplets synchronisés (cameras/frames.csv) et on interpole les poses ; aucun impact attendu."
                if only_cam_gaps and max_cam_gap <= 0.2
                else "BLOQUANT : trous hors caméras ou trop longs, à investiguer."
            )
        ),
    }
    report["T0.5"] = t05
    report["trajectories"] = traj_info

    # ================================================================== splits
    fid = np.arange(len(trip_ns))
    cam_test = fid[fid % 10 == 0]
    cam_train = fid[fid % 10 != 0]
    lid = np.arange(len(lidar_ns))
    lid_test = lidar_ns[lid % 10 == 0]
    lid_train = lidar_ns[lid % 10 != 0]
    splits = {
        "contract_version": "0.1",
        "cameras": {
            "unit": "frame_id (ligne de cameras/frames.csv, triplet cam0/cam1/cam2 synchronisé)",
            "rule": "test_in = frame_id % 10 == 0 ; train = le reste",
            "n_train": len(cam_train),
            "n_test_in": len(cam_test),
            "n_test_out": 0,
            "train": cam_train,
            "test_in": cam_test,
            "test_out": [],
        },
        "lidars": {
            LIDAR_ID: {
                "unit": "stamp_ns du fichier raw/lidar-clouds/<s>.<ns>.pcd (temps du premier point)",
                "rule": "scans triés par temps ; test_in = rang % 10 == 0 ; train = le reste",
                "n_train": len(lid_train),
                "n_test_in": len(lid_test),
                "n_test_out": 0,
                "train": lid_train,
                "test_in": lid_test,
                "test_out": [],
            }
        },
    }
    with open(sc.splits_path, "w") as f:
        f.write(
            "# splits.yaml — contrat v0.1 (splatpipe.stage0.ingest_oxford)\n"
            "# test_in : vues interpolées le long de la trajectoire d'entraînement (1 sur 10).\n"
            "# test_out : vide pour l'instant. Futur : une autre séquence du même site (observatory-quarter),\n"
            "#            recalée dans world, pour mesurer la dégradation loin de la trajectoire d'entraînement.\n"
        )
        f.write(_yaml_dump(splits))
    report["splits"] = {
        "cameras": {k: splits["cameras"][k] for k in ("rule", "n_train", "n_test_in", "n_test_out")},
        "lidar": {k: splits["lidars"][LIDAR_ID][k] for k in ("rule", "n_train", "n_test_in", "n_test_out")},
    }

    # ================================================================== meta.yaml
    meta = {
        "dataset": DATASET,
        "sequence": seq,
        "site": site,
        "contract_version": "0.1",
        "created_utc": report["created_utc"],
        "generator": "splatpipe.stage0.ingest_oxford",
        "units": {
            "length": "m",
            "angle": "rad (degrés seulement dans les rapports, suffixe _deg)",
            "time": "s Unix float64 (t) ; ns int64 (stamp_ns) exacts depuis les noms de fichiers",
            "acc": "m/s^2",
            "gyro": "rad/s",
        },
        "conventions": {
            "transform": "T_a_b 4×4 : p_a = T_a_b · p_b",
            "quaternion": "xyzw (Hamilton)",
            "tum": "t tx ty tz qx qy qz qw = T_world_base(t) (pour gt.tum)",
            "frames": {
                "base": "bas de l'appareil, x avant, y gauche, z haut",
                "lidar": "Hesai QT64 ; T_base_lidar = t(0, 0, 0.124), q_xyzw(0, 0, 1, 0)",
                "cam0": "optique OpenCV, avant",
                "cam1": "optique OpenCV, gauche",
                "cam2": "optique OpenCV, droite",
                "imu": "IMU Alphasense (x gauche, y arrière, z haut par rapport à base)",
                "world": "scan laser fixe de référence (gt.tum, ground_truth_map)",
            },
        },
        "sources": {
            "data_root": DATA_MOUNT,
            "sequence_dir": seq_rel,
            "cameras": {c: f"{seq_rel}/raw/{c}" for c in CAMS},
            "imu": f"{seq_rel}/raw/imu.csv",
            "lidar_clouds": f"{seq_rel}/raw/lidar-clouds",
            "trajectories": {k: v["source"] for k, v in traj_info.items()},
            "calibration": f"{RAW_ROOT}/calibration/{{cam0,cam1,cam2,cam-lidar-imu,imu}}.yaml",
            "ground_truth_map": f"{RAW_ROOT}/ground_truth_map/{site}/",
        },
        "trajectories": {
            k: {
                "file": v["file"],
                "frame": v["frame"],
                "n": v["n_out"],
                **(
                    {"body_frame_best_match": v["body_frame_identification"]["best"]}
                    if "body_frame_identification" in v
                    else {}
                ),
            }
            for k, v in traj_info.items()
        },
        "notes": [
            "Les images ne sont pas copiées : cameras/<cam>/index.csv pointe vers les JPEG bruts (rel_path relatif à /data).",
            "vilens.tum, hba.tum et colmap.tum sont dans leurs propres repères locaux, PAS dans world ; "
            "ne pas les comparer à gt.tum sans alignement (evo -a).",
            "gt.tum mélange des poses aux temps caméra et aux temps de début de scan LiDAR (voir reports/stage0_ingest.json).",
            "Repère corps identifié par comparaison des rotations relatives à la GT (reports/stage0_ingest.json, "
            "trajectories.*.body_frame_identification) : cam0_opengl = cam0 avec y et z inversés (x droite, y haut, z arrière).",
            "Durée d'exposition des images inconnue (non fournie par le dataset) ; horodatage supposé au milieu d'exposition.",
            "lidar/ et lidar_model/ sont produits par l'ingestion LiDAR (autre module).",
        ],
    }
    with open(sc.meta_path, "w") as f:
        f.write("# meta.yaml — contrat de données v0.1\n")
        f.write(_yaml_dump(meta))

    report["outputs"] = {
        "meta": "meta.yaml",
        "sensors": "sensors.yaml",
        "camera_indices": {c: {"file": f"cameras/{c}/index.csv", "n": int(len(cam_ns[c]))} for c in CAMS},
        "frames": {"file": "cameras/frames.csv", "n": int(len(trip_ns))},
        "imu": {"file": "imu/imu.npz", "n": int(len(imu_ns)), "keys": ["t", "stamp_ns", "acc", "gyro"]},
        "trajectories": {k: v["file"] for k, v in traj_info.items()},
        "splits": "splits.yaml",
    }
    report["calibration_checks"] = calib_checks
    report["runtime_s"] = round(time.time() - t_start, 1)
    report = _py(report)
    sc.write_report("stage0_ingest", report)
    vol.commit()
    return report


@app.local_entrypoint()
def main(seq: str = "2024-03-13-observatory-quarter-01"):
    import json
    import pathlib

    rep = ingest.remote(seq)
    out = pathlib.Path("logs/stage0/ingest.json")
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    t01, t05 = rep["T0.1"], rep["T0.5"]
    print("T0.1 strict :", t01["status_strict"], "|", json.dumps(t01["summary"]))
    for k, v in t01["series"].items():
        print(
            f"  {k}: n={v['n']} rate={v['median_rate_hz']} Hz drops={v['n_drops']} gaps>2P={v['n_gaps_gt_2_periods']} "
            f"max_gap={v['max_gap_s']} s grid_resid_std={v['grid_residual_std_ms']} ms"
        )
    print("  triplets:", {k: v for k, v in t01["triplets"].items() if k != "triplet_series"})
    print("  GT align:", json.dumps(t01["gt_time_alignment"], ensure_ascii=False)[:2500])
    print("T0.5 :", t05["status"])
    skip = ("static_segments", "calibration_matrix_vs_quaternion", "scans")
    for k in ("a_accelerometer_norm", "b_gravity_direction", "c_extrinsics", "d_gyro_vs_gt", "e_lidar_world_closure",
              "f_cam0_rotation_vs_colmap"):
        v = {kk: vv for kk, vv in t05[k].items() if kk not in skip}
        print(f"  {k}:", json.dumps(v, ensure_ascii=False)[:2500])
    for sc_ in t05["e_lidar_world_closure"].get("scans", []):
        print("    scan", json.dumps(sc_)[:800])
    for name, info in rep["trajectories"].items():
        bf = info.get("body_frame_identification", {})
        print(f"  traj {name}: n={info['n_out']} dups={info['n_duplicates_removed']} best_body={bf.get('best')}",
              {h: round(x["median_err_deg"], 3) for h, x in bf.get("hypotheses", {}).items()})
    print(f"rapport complet : {out} ; runtime {rep['runtime_s']} s")
