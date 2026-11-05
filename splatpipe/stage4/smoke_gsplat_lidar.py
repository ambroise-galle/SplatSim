"""Étape 4 — préparation : image gsplat compilée + test T4.0a (rendu LiDAR synthétique).

Contenu (voir etapes/notes/stage4-gsplat-env.md) :
  a. import de gsplat, extension CUDA précompilée chargée sans JIT, temps d'import et de premier appel ;
  b. T4.0a : LiDAR tournant structuré synthétique (32 lignes, 1024 colonnes, 10 Hz) dans une pièce
     faite de gaussiennes plates quasi opaques ; distance rendue (`RGB-Ed`) vs distance analytique
     rayon–plan ;
  c. contrôle sténopé (pinhole) sur la même scène : profondeur `ED` (z) et distance `Ed` vs analytique ;
  d. gradient : un mur décalé de 30 cm converge vers les distances LiDAR analytiques
     (`gsplat.losses.lidar_distance_loss`) + vérification du gradient par différences finies ;
  e. (option `--pytest`) tests pytest de gsplat liés au LiDAR, exécutés dans le conteneur.

Lancement (depuis la racine du projet) :
  .venv/bin/modal run splatpipe/stage4/smoke_gsplat_lidar.py
  .venv/bin/modal run splatpipe/stage4/smoke_gsplat_lidar.py --pytest
  .venv/bin/modal run splatpipe/stage4/smoke_gsplat_lidar.py --only-pytest

Conventions (vérifiées ici) :
  - `viewmats` = monde → capteur (4×4). Repère LiDAR : x avant (azimut 0), y gauche, z haut ;
    rayon = (cos az·cos él, sin az·cos él, sin él). Caméra sténopé : convention OpenCV (x droite,
    y bas, z avant), centres de pixels en (u+0,5, v+0,5).
  - Lignes LiDAR triées par élévation décroissante ; colonnes triées dans le sens de rotation, sans
    boucler sur 2π. Sortie de `rasterization` : [1, n_lignes, n_colonnes, 4] ; canal 3 = distance.
  - Unités : mètres, radians.
  - Pavage LiDAR : au plus 64 rayons par tuile (tile_size = 8) → `safe_lidar_tiling()`. Le défaut de
    `compute_tiling` (256) et même `max_pts_per_tile=64` (66–68 réels) perdent des rayons ou rendent
    des distances aberrantes.
  - Obturateur GLOBAL : `viewmats_rs=None` obligatoire.
"""

from pathlib import Path

import modal

from splatpipe.images import GSPLAT_BUILD_INFO, GSPLAT_DIR, TORCH_EXTENSIONS_DIR, gsplat_image

app = modal.App("splatsim-stage4-smoke-gsplat")
# Seuls les fichiers utiles sont montés (pas tout `splatpipe/`, modifié en parallèle par d'autres étapes :
# Modal refuse un fichier monté qui change pendant la construction).
_PKG = Path(__file__).resolve().parents[1]
image = (
    gsplat_image()
    .add_local_file(_PKG / "__init__.py", "/root/splatpipe/__init__.py")
    .add_local_file(_PKG / "images.py", "/root/splatpipe/images.py")
    .add_local_dir(_PKG / "stage4", "/root/splatpipe/stage4", ignore=["**/__pycache__", "**/*.pyc"])
)


# --------------------------------------------------------------------------------------------------
# Scène et capteurs synthétiques
# --------------------------------------------------------------------------------------------------

SENSOR_POS = (2.0, -1.0, 1.8)  # m, repère monde (sol en z = 0)
SENSOR_YPR_DEG = (25.0, 2.0, -1.0)  # lacet, tangage, roulis : R = Rz(lacet)·Ry(tangage)·Rx(roulis)

# Pièce sans plafond : x ∈ [-4, 10], y ∈ [-8, 4], murs de 3,3 m. Mur « avant » à x = 10
# (8 m devant le capteur selon x monde).
ROOM = {
    "front_x10": ((10.0, -8.0, 0.0), (0.0, 12.0, 0.0), (0.0, 0.0, 3.3)),
    "back_x-4": ((-4.0, 4.0, 0.0), (0.0, -12.0, 0.0), (0.0, 0.0, 3.3)),
    "left_y4": ((10.0, 4.0, 0.0), (-14.0, 0.0, 0.0), (0.0, 0.0, 3.3)),
    "right_y-8": ((-4.0, -8.0, 0.0), (14.0, 0.0, 0.0), (0.0, 0.0, 3.3)),
    "floor_z0": ((-4.0, -8.0, 0.0), (14.0, 0.0, 0.0), (0.0, 12.0, 0.0)),
}
SPACING = 0.04  # m, pas de la grille de gaussiennes
SIGMA_T = 0.04  # m, écart-type tangent
SIGMA_N = 0.001  # m, écart-type normal (épaisseur)
OPACITY = 0.99
EDGE_MARGIN = 0.25  # m, rayons exclus s'ils touchent à moins de cette distance d'un bord de surface


def _rot_ypr(yaw_deg, pitch_deg, roll_deg):
    import numpy as np

    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    return rz @ ry @ rx


def _mat_to_quat_wxyz(m):
    import numpy as np

    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1.0) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def make_surface_gaussians(surfaces, spacing, sigma_t, sigma_n, opacity, device):
    """Gaussiennes plates sur une grille régulière couvrant chaque rectangle (coin, arête a, arête b)."""
    import numpy as np
    import torch

    means, quats, scales, labels = [], [], [], []
    for k, (corner, a, b) in enumerate(surfaces.values()):
        corner, a, b = np.array(corner), np.array(a), np.array(b)
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        na, nb = int(round(la / spacing)), int(round(lb / spacing))
        ua, ub = a / la, b / lb
        n = np.cross(ua, ub)
        rot = np.stack([ua, ub, n], axis=1)  # colonnes : axes locaux x, y, z (z = normale)
        ia, ib = np.meshgrid((np.arange(na) + 0.5) / na, (np.arange(nb) + 0.5) / nb, indexing="ij")
        pts = corner + ia.reshape(-1, 1) * a + ib.reshape(-1, 1) * b
        means.append(pts)
        quats.append(np.tile(_mat_to_quat_wxyz(rot), (len(pts), 1)))
        scales.append(np.tile([sigma_t, sigma_t, sigma_n], (len(pts), 1)))
        labels.append(np.full(len(pts), k))
    f = lambda x: torch.tensor(np.concatenate(x), dtype=torch.float32, device=device)  # noqa: E731
    means_t, quats_t, scales_t = f(means), f(quats), f(scales)
    opac = torch.full((len(means_t),), opacity, dtype=torch.float32, device=device)
    return means_t, quats_t, scales_t, opac, torch.tensor(np.concatenate(labels), device=device)


def analytic_hits(origin, dirs, surfaces, margin):
    """Intersection rayons–rectangles (float64). Renvoie distance t, indice de surface, masque intérieur."""
    import torch

    dev = dirs.device
    o = torch.as_tensor(origin, dtype=torch.float64, device=dev)
    d = dirs.to(torch.float64)
    best_t = torch.full(d.shape[:-1], float("inf"), dtype=torch.float64, device=dev)
    best_k = torch.full(d.shape[:-1], -1, dtype=torch.long, device=dev)
    best_interior = torch.zeros(d.shape[:-1], dtype=torch.bool, device=dev)
    for k, (corner, a, b) in enumerate(surfaces.values()):
        c = torch.tensor(corner, dtype=torch.float64, device=dev)
        a = torch.tensor(a, dtype=torch.float64, device=dev)
        b = torch.tensor(b, dtype=torch.float64, device=dev)
        n = torch.linalg.cross(a, b)
        n = n / n.norm()
        denom = d @ n
        t = ((c - o) @ n) / denom
        p = o + t[..., None] * d
        la, lb = a.norm(), b.norm()
        sa = (p - c) @ (a / la)
        sb = (p - c) @ (b / lb)
        inside = (t > 0) & (sa >= 0) & (sa <= la) & (sb >= 0) & (sb <= lb) & (denom.abs() > 1e-9)
        edge = torch.minimum(torch.minimum(sa, la - sa), torch.minimum(sb, lb - sb))
        better = inside & (t < best_t)
        best_t = torch.where(better, t, best_t)
        best_k = torch.where(better, torch.full_like(best_k, k), best_k)
        best_interior = torch.where(better, edge >= margin, best_interior)
    return best_t, best_k, best_interior


def qt64_like_elevations_deg(n_rows=64, top=52.1, bottom=-52.5):
    """Élévations non uniformes type Hesai QT64 : pas ~1,46° au centre, ~2,3° aux extrémités (décroissantes)."""
    import numpy as np

    u = np.linspace(-1, 1, n_rows - 1)
    steps = 1.46 + (2.3 - 1.46) * u**2
    cum = np.concatenate([[0.0], np.cumsum(steps)])
    return list(top - cum * (top - bottom) / cum[-1])


_TILING_USED = {}


def safe_lidar_tiling(params, tile_size=8, start=64, step=4):
    """`compute_tiling` dont AUCUNE tuile ne dépasse tile_size² rayons.

    Le noyau 3DGUT LiDAR (tile_size=8 pour n_rows < 1080) ne traite que les 64 premiers rayons d'une tuile ;
    `max_pts_per_tile` n'est qu'une cible approximative (on observe 66–68 pour 64). On le diminue donc
    jusqu'à ce que le maximum réel respecte la limite.
    """
    from gsplat.cuda._lidar import compute_tiling

    m = start
    while True:
        tiling = compute_tiling(params, max_pts_per_tile=m)
        if int(tiling.tiles_pack_info[:, 1].max()) <= tile_size * tile_size:
            return tiling, m
        m -= step


def make_lidar(
    device,
    n_rows=32,
    n_cols=1024,
    el_max_deg=15.0,
    clockwise=True,
    row_offsets_rad=0.0,
    max_pts_per_tile=64,
    seed=0,
    elevations_deg=None,
    az_start_deg=None,
):
    """LiDAR tournant structuré : élévations régulières dans [-el_max, +el_max] (ou `elevations_deg`,
    décroissantes), 360° en n_cols colonnes."""
    import math

    import torch
    from gsplat import RowOffsetStructuredSpinningLidarModelParametersExt
    from gsplat.cuda._lidar import (
        RowOffsetStructuredSpinningLidarModelParameters,
        SpinningDirection,
        compute_angles_to_columns_map,
        compute_tiling,
    )

    if elevations_deg is not None:
        elev = torch.deg2rad(torch.tensor(elevations_deg, dtype=torch.float64)).to(torch.float32)
        n_rows = len(elevations_deg)
    else:
        elev = torch.linspace(math.radians(el_max_deg), -math.radians(el_max_deg), n_rows, dtype=torch.float32)
    step = 2 * math.pi / n_cols
    idx = torch.arange(n_cols, dtype=torch.float64)
    if az_start_deg is not None:  # ex. QT64 : colonne 0 à 89,2835°, puis pas de 360°/n_cols
        a0 = math.radians(az_start_deg)
        az = a0 - idx * step if clockwise else a0 + idx * step
    elif clockwise:  # azimuts décroissants (sens horaire vu du dessus), sans bouclage 2π
        az = math.pi - (idx + 0.5) * step
    else:
        az = -math.pi + (idx + 0.5) * step
    az = az.to(torch.float32)
    if row_offsets_rad:
        g = torch.Generator().manual_seed(seed)
        offs = (torch.rand(n_rows, generator=g) - 0.5) * 2 * row_offsets_rad
    else:
        offs = torch.zeros(n_rows)
    params = RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=elev.to(device),
        column_azimuths_rad=az.to(device),
        row_azimuth_offsets_rad=offs.to(torch.float32).to(device),
        spinning_frequency_hz=10.0,
        spinning_direction=SpinningDirection.CLOCKWISE if clockwise else SpinningDirection.COUNTER_CLOCKWISE,
    )
    a2c = compute_angles_to_columns_map(params)
    if max_pts_per_tile == "safe":
        tiling, used = safe_lidar_tiling(params)
    else:
        tiling, used = compute_tiling(params, max_pts_per_tile=max_pts_per_tile), max_pts_per_tile
    _TILING_USED["max_pts_per_tile"] = used
    lidar = RowOffsetStructuredSpinningLidarModelParametersExt(params, a2c.to(device), tiling)
    return lidar


def lidar_world_dirs(lidar, r_ws):
    """Directions monde (float64) de chaque élément [n_lignes, n_colonnes, 3]."""
    import torch

    el = lidar.row_elevations_rad.double()[:, None]
    az = lidar.column_azimuths_rad.double()[None, :] + lidar.row_azimuth_offsets_rad.double()[:, None]
    d = torch.stack([torch.cos(az) * torch.cos(el), torch.sin(az) * torch.cos(el), torch.sin(el).expand_as(az)], -1)
    return d @ torch.as_tensor(r_ws, dtype=torch.float64, device=d.device).T


def _stats(err):
    """Statistiques d'erreur (mètres → millimètres)."""
    import torch

    e = err.double()
    if e.numel() == 0:
        return {"n": 0}
    a = e.abs()
    return {
        "n": int(e.numel()),
        "median_abs_mm": round(float(a.median()) * 1e3, 3),
        "p95_abs_mm": round(float(torch.quantile(a, 0.95)) * 1e3, 3),
        "max_abs_mm": round(float(a.max()) * 1e3, 3),
        "mean_signed_mm": round(float(e.mean()) * 1e3, 3),
    }


def _time_cuda(fn, n=10):
    import time

    import torch

    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {"median_ms": round(ts[len(ts) // 2] * 1e3, 2), "min_ms": round(ts[0] * 1e3, 2)}


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------


def _test_import():
    """a. Import de gsplat, extension précompilée, pas de JIT."""
    import json
    import os
    import subprocess
    import sys
    import time

    t0 = time.perf_counter()
    import torch
    import torch.utils.cpp_extension as cpp_ext

    t_torch = time.perf_counter() - t0

    jit_calls = []
    orig_load, orig_jit = cpp_ext.load, cpp_ext._jit_compile

    def _spy_load(*a, **k):
        jit_calls.append(("load", k.get("name", a[0] if a else None)))
        return orig_load(*a, **k)

    def _spy_jit(*a, **k):
        jit_calls.append(("_jit_compile", a[0] if a else k.get("name")))
        return orig_jit(*a, **k)

    cpp_ext.load, cpp_ext._jit_compile = _spy_load, _spy_jit

    ext_dir_before = {
        p: os.path.getmtime(os.path.join(TORCH_EXTENSIONS_DIR, p))
        for p in (os.listdir(TORCH_EXTENSIONS_DIR) if os.path.isdir(TORCH_EXTENSIONS_DIR) else [])
    }

    t1 = time.perf_counter()
    import gsplat
    from gsplat.cuda._backend import _C

    t_import = time.perf_counter() - t1
    csrc = sys.modules.get("gsplat.csrc")

    # Premier appel CUDA (petit rendu sténopé)
    dev = "cuda"
    t2 = time.perf_counter()
    N = 1000
    means = torch.randn(N, 3, device=dev) + torch.tensor([0, 0, 5.0], device=dev)
    quats = torch.nn.functional.normalize(torch.randn(N, 4, device=dev), dim=-1)
    scales = torch.rand(N, 3, device=dev) * 0.05
    opac = torch.rand(N, device=dev)
    colors = torch.rand(N, 3, device=dev)
    K = torch.tensor([[[100.0, 0, 64], [0, 100.0, 48], [0, 0, 1]]], device=dev)
    img, alpha, _ = gsplat.rasterization(means, quats, scales, opac, colors, torch.eye(4, device=dev)[None], K, 128, 96)
    torch.cuda.synchronize()
    t_first = time.perf_counter() - t2

    # Extension « sensors » (JIT-only, précompilée dans l'image) : chargement sans recompilation
    so_before = {}
    sensors_dir = os.path.join(TORCH_EXTENSIONS_DIR, "gsplat_sensors_cuda")
    if os.path.isdir(sensors_dir):
        so_before = {f: os.path.getmtime(os.path.join(sensors_dir, f)) for f in os.listdir(sensors_dir) if f.endswith(".so")}
    t3 = time.perf_counter()
    sensors_ok, sensors_err = True, ""
    try:
        from gsplat.sensors.kernels._backend import _get_backend

        _get_backend()
    except Exception as e:  # noqa: BLE001
        sensors_ok, sensors_err = False, repr(e)[:500]
    t_sensors = time.perf_counter() - t3
    so_after = {}
    if os.path.isdir(sensors_dir):
        so_after = {f: os.path.getmtime(os.path.join(sensors_dir, f)) for f in os.listdir(sensors_dir) if f.endswith(".so")}

    cpp_ext.load, cpp_ext._jit_compile = orig_load, orig_jit

    with open(GSPLAT_BUILD_INFO) as f:
        build_info = json.load(f)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=GSPLAT_DIR, capture_output=True, text=True).stdout.strip()
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "gpu": smi,
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "arch_list_compiled": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "gsplat_version": str(gsplat.__version__),
        "gsplat_commit": commit,
        "gsplat_file": gsplat.__file__,
        "csrc_file": getattr(csrc, "__file__", None),
        "csrc_is_prebuilt": bool(csrc is not None and _C is csrc),
        "has_3dgs": bool(gsplat.has_3dgs()),
        "has_3dgut": bool(gsplat.has_3dgut()),
        "has_camera_wrappers": bool(gsplat.has_camera_wrappers()),
        "import_torch_s": round(t_torch, 2),
        "import_gsplat_s": round(t_import, 2),
        "first_rasterization_call_s": round(t_first, 2),
        "first_render_finite": bool(torch.isfinite(img).all().item()),
        "jit_calls_during_core_import": [c for c in jit_calls if c[1] != "gsplat_sensors_cuda"],
        "sensors_ext": {
            "ok": sensors_ok,
            "error": sensors_err,
            "load_s": round(t_sensors, 2),
            "so_unchanged": bool(so_before == so_after and len(so_before) > 0),
        },
        "torch_extensions_dir_entries": sorted(ext_dir_before),
        "build_info": build_info,
    }


def _render_lidar(scene, lidar, viewmat, rolling="left_to_right"):
    import gsplat
    import torch

    means, quats, scales, opac = scene
    rs = gsplat.RollingShutterType.ROLLING_LEFT_TO_RIGHT if rolling == "left_to_right" else gsplat.RollingShutterType.GLOBAL
    colors = torch.zeros(len(means), 3, device=means.device)
    renders, alphas, meta = gsplat.rasterization(
        means,
        quats,
        scales,
        opac,
        colors,
        viewmat[None],
        torch.eye(3, device=means.device)[None],
        lidar.n_columns,
        lidar.n_rows,
        camera_model="lidar",
        lidar_coeffs=lidar,
        with_ut=True,
        with_eval3d=True,
        packed=False,
        render_mode="RGB-Ed",
        global_z_order=False,
        far_plane=200.0,
        rolling_shutter=rs,
        viewmats_rs=viewmat[None] if rolling == "left_to_right" else None,  # GLOBAL exige None
    )
    return renders[0, ..., 3], alphas[0, ..., 0], meta


def _test_lidar():
    """b. T4.0a — rendu LiDAR vs distances analytiques."""
    import time

    import numpy as np
    import torch

    dev = "cuda"
    r_ws = _rot_ypr(*SENSOR_YPR_DEG)
    t_ws = np.eye(4)
    t_ws[:3, :3], t_ws[:3, 3] = r_ws, SENSOR_POS
    viewmat = torch.tensor(np.linalg.inv(t_ws), dtype=torch.float32, device=dev)

    t0 = time.perf_counter()
    means, quats, scales, opac, labels = make_surface_gaussians(ROOM, SPACING, SIGMA_T, SIGMA_N, OPACITY, dev)
    scene = (means, quats, scales, opac)
    names = list(ROOM)
    out = {
        "scene": {
            "n_gaussians": int(len(means)),
            "per_surface": {n: int((labels == k).sum()) for k, n in enumerate(names)},
            "spacing_m": SPACING,
            "sigma_tangent_m": SIGMA_T,
            "sigma_normal_m": SIGMA_N,
            "opacity": OPACITY,
            "sensor_pos": SENSOR_POS,
            "sensor_yaw_pitch_roll_deg": SENSOR_YPR_DEG,
            "edge_margin_m": EDGE_MARGIN,
            "build_s": round(time.perf_counter() - t0, 2),
        },
        "variants": {},
    }

    variants = {
        "cw_offsets0_tile64_rs": dict(clockwise=True, row_offsets_rad=0.0, max_pts_per_tile=64, rolling="left_to_right"),
        "cw_offsets0_tile64_global": dict(clockwise=True, row_offsets_rad=0.0, max_pts_per_tile=64, rolling="global"),
        "cw_rowoffsets0.1rad_tile64_rs": dict(clockwise=True, row_offsets_rad=0.1, max_pts_per_tile=64, rolling="left_to_right"),
        "ccw_offsets0_tile64_rs": dict(clockwise=False, row_offsets_rad=0.0, max_pts_per_tile=64, rolling="left_to_right"),
        "cw_offsets0_tile256_rs(av_trainer_default)": dict(
            clockwise=True, row_offsets_rad=0.0, max_pts_per_tile=256, rolling="left_to_right"
        ),
        "qt64_like_64rows_600cols_tile64_rs": dict(
            clockwise=True,
            row_offsets_rad=0.0,
            max_pts_per_tile=64,
            rolling="left_to_right",
            n_cols=600,
            elevations_deg=qt64_like_elevations_deg(),
            az_start_deg=89.2835,
        ),
        "qt64_like_64rows_600cols_tile64_rs_seam180": dict(
            clockwise=True,
            row_offsets_rad=0.0,
            max_pts_per_tile=64,
            rolling="left_to_right",
            n_cols=600,
            elevations_deg=qt64_like_elevations_deg(),
        ),
        "cw_offsets0_SAFE_rs": dict(clockwise=True, row_offsets_rad=0.0, max_pts_per_tile="safe", rolling="left_to_right"),
        "ccw_offsets0_SAFE_rs": dict(clockwise=False, row_offsets_rad=0.0, max_pts_per_tile="safe", rolling="left_to_right"),
        "qt64_like_64rows_600cols_SAFE_rs": dict(
            clockwise=True,
            row_offsets_rad=0.0,
            max_pts_per_tile="safe",
            rolling="left_to_right",
            n_cols=600,
            elevations_deg=qt64_like_elevations_deg(),
            az_start_deg=89.2835,
        ),
        "qt64_like_64rows_600cols_tile256_rs(default_compute_tiling)": dict(
            clockwise=True,
            row_offsets_rad=0.0,
            max_pts_per_tile=256,
            rolling="left_to_right",
            n_cols=600,
            elevations_deg=qt64_like_elevations_deg(),
            az_start_deg=89.2835,
        ),
    }
    dist_ref = None
    for vname, v in variants.items():
        try:
            t1 = time.perf_counter()
            lidar = make_lidar(
                dev,
                n_cols=v.get("n_cols", 1024),
                clockwise=v["clockwise"],
                row_offsets_rad=v["row_offsets_rad"],
                max_pts_per_tile=v["max_pts_per_tile"],
                elevations_deg=v.get("elevations_deg"),
                az_start_deg=v.get("az_start_deg"),
            )
            t_model = time.perf_counter() - t1
            dist, alpha, meta = _render_lidar(scene, lidar, viewmat, v["rolling"])
            torch.cuda.synchronize()
            dirs = lidar_world_dirs(lidar, r_ws)
            t_gt, k_gt, interior = analytic_hits(SENSOR_POS, dirs, ROOM, EDGE_MARGIN)
            has_hit = k_gt >= 0
            ret = alpha > 0.5
            err = dist.double() - t_gt
            res = {
                "lidar_model_build_s": round(t_model, 2),
                "n_rays": int(dist.numel()),
                "tiling": {
                    "n_bins_azimuth": int(lidar.tiling.n_bins_azimuth),
                    "n_bins_elevation": int(lidar.tiling.n_bins_elevation),
                    "max_rays_per_tile": int(lidar.tiling.tiles_pack_info[:, 1].max()),
                "max_pts_per_tile_param": _TILING_USED.get("max_pts_per_tile"),
                "n_tiles_over_64": int((lidar.tiling.tiles_pack_info[:, 1] > 64).sum()),
                "n_rays_beyond_64_in_tile": int((lidar.tiling.tiles_pack_info[:, 1] - 64).clamp(min=0).sum()),
                    "tile_size_used": int(meta["tile_size"]),
                },
                "frac_rays_with_analytic_hit": round(float(has_hit.float().mean()), 4),
                "return_fraction_on_hits(alpha>0.5)": round(float(ret[has_hit].float().mean()), 4),
                "return_fraction_on_interior_hits": round(float(ret[has_hit & interior].float().mean()), 4),
                "false_returns_without_hit": int((ret & ~has_hit).sum()),
                "n_rays_without_hit": int((~has_hit).sum()),
                "alpha_interior_hits": {
                    "p1": round(float(torch.quantile(alpha[has_hit & interior].double(), 0.01)), 4),
                    "median": round(float(alpha[has_hit & interior].median()), 4),
                },
                "alpha_max_without_hit_interior_region": round(float(alpha[~has_hit].max()), 4) if (~has_hit).any() else None,
                "error_interior_returns": _stats(err[has_hit & interior & ret]),
                "error_all_returns_on_hits(incl_edges)": _stats(err[has_hit & ret]),
                "error_by_surface_interior": {
                    n: _stats(err[has_hit & interior & ret & (k_gt == k)]) for k, n in enumerate(names)
                },
            }
            bad = has_hit & interior & ret & (err.abs() > 0.01)
            miss = has_hit & interior & ~ret
            rows_b, cols_b = torch.nonzero(bad, as_tuple=True)
            rows_m, cols_m = torch.nonzero(miss, as_tuple=True)
            az_col = lidar.column_azimuths_rad.double()
            res["outliers_interior_gt_10mm"] = {
                "n": int(bad.sum()),
                "cols": sorted(set(cols_b.tolist()))[:60],
                "rows": sorted(set(rows_b.tolist()))[:60],
                "col_azimuth_deg": sorted({round(float(torch.rad2deg(az_col[c])), 2) for c in cols_b.tolist()})[:30],
                "err_mm": [round(float(e) * 1e3, 1) for e in err[bad][:20]],
            }
            res["missing_returns_interior"] = {
                "n": int(miss.sum()),
                "cols": sorted(set(cols_m.tolist()))[:60],
                "rows": sorted(set(rows_m.tolist()))[:60],
            }
            res["n_cols"] = int(lidar.n_columns)
            res["col0_azimuth_deg"] = round(float(torch.rad2deg(az_col[0])), 4)
            if vname.startswith("cw_offsets0_tile64_rs"):
                dist_ref = (dist.clone(), alpha.clone())
                res["timing_forward"] = _time_cuda(lambda: _render_lidar(scene, lidar, viewmat, "left_to_right"), n=20)
                m = means.clone().requires_grad_(True)

                def fwd_bwd():
                    d, a, _ = _render_lidar((m, quats, scales, opac), lidar, viewmat, "left_to_right")
                    (d * a).sum().backward()

                res["timing_forward_backward"] = _time_cuda(fwd_bwd, n=10)
                res["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            if vname.startswith("cw_offsets0_tile64_global") and dist_ref is not None:
                both = (alpha > 0.5) & (dist_ref[1] > 0.5)
                res["max_abs_diff_vs_rolling_shutter_render_mm"] = round(
                    float((dist - dist_ref[0])[both].abs().max()) * 1e3, 6
                )
                res["max_abs_alpha_diff_vs_rolling_shutter"] = round(float((alpha - dist_ref[1]).abs().max()), 6)
        except Exception:  # noqa: BLE001
            import traceback

            res = {"error": traceback.format_exc()[-3000:]}
        out["variants"][vname] = res
    return out


def _test_pinhole():
    """c. Contrôle sténopé : profondeur z (ED) et distance le long du rayon (Ed) vs analytique."""
    import gsplat
    import numpy as np
    import torch

    dev = "cuda"
    r_ws = _rot_ypr(*SENSOR_YPR_DEG)
    r_sc = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)  # caméra OpenCV regardant selon x capteur
    r_wc = r_ws @ r_sc
    t_wc = np.eye(4)
    t_wc[:3, :3], t_wc[:3, 3] = r_wc, SENSOR_POS
    viewmat = torch.tensor(np.linalg.inv(t_wc), dtype=torch.float32, device=dev)
    W, H, f = 640, 480, 400.0
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32, device=dev)

    means, quats, scales, opac, _ = make_surface_gaussians(ROOM, SPACING, SIGMA_T, SIGMA_N, OPACITY, dev)
    colors = torch.zeros(len(means), 3, device=dev)

    u = torch.arange(W, device=dev, dtype=torch.float64) + 0.5
    v = torch.arange(H, device=dev, dtype=torch.float64) + 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    d_cam = torch.stack([(uu - W / 2) / f, (vv - H / 2) / f, torch.ones_like(uu)], -1)  # z = 1
    d_w = d_cam @ torch.as_tensor(r_wc, dtype=torch.float64, device=dev).T
    norm = d_w.norm(dim=-1)
    t_unit, k_gt, interior = analytic_hits(SENSOR_POS, d_w / norm[..., None], ROOM, EDGE_MARGIN)
    has_hit = k_gt >= 0
    z_gt = t_unit / norm  # d_cam a z = 1 → profondeur = t / ||d||
    names = list(ROOM)

    def run(mode, with_ut, gz, gt):
        img, alpha, _ = gsplat.rasterization(
            means, quats, scales, opac, colors, viewmat[None], K[None], W, H,
            render_mode=mode, with_ut=with_ut, with_eval3d=with_ut, packed=False, global_z_order=gz,
        )  # fmt: skip
        d = img[0, ..., 3].double()
        a = alpha[0, ..., 0]
        ret = a > 0.5
        err = d - gt
        m = has_hit & interior & ret
        return {
            "return_fraction_on_interior_hits": round(float(ret[has_hit & interior].float().mean()), 4),
            "error_interior": _stats(err[m]),
            "error_by_surface_interior": {n: _stats(err[m & (k_gt == k)]) for k, n in enumerate(names)},
        }

    return {
        "image": {"W": W, "H": H, "f": f},
        "frac_pixels_with_analytic_hit": round(float(has_hit.float().mean()), 4),
        "classic_3dgs_ED_zdepth": run("RGB+ED", False, True, z_gt),
        "3dgut_eval3d_ED_zdepth": run("RGB+ED", True, True, z_gt),
        "3dgut_eval3d_Ed_ray_distance": run("RGB-Ed", True, True, t_unit),
    }


def _test_gradient():
    """d. Mur décalé de +30 cm optimisé vers les distances LiDAR analytiques."""
    import gsplat
    import numpy as np
    import torch

    dev = "cuda"
    r_ws = _rot_ypr(*SENSOR_YPR_DEG)
    t_ws = np.eye(4)
    t_ws[:3, :3], t_ws[:3, 3] = r_ws, SENSOR_POS
    viewmat = torch.tensor(np.linalg.inv(t_ws), dtype=torch.float32, device=dev)

    wall_true = {"front_x10": ((10.0, -6.0, 0.4), (0.0, 10.0, 0.0), (0.0, 0.0, 2.9))}
    X_TRUE, OFFSET = 10.0, 0.30
    means0, quats, scales, opac, _ = make_surface_gaussians(wall_true, SPACING, SIGMA_T, SIGMA_N, OPACITY, dev)
    g = torch.Generator(device=dev).manual_seed(0)
    means_init = means0.clone()
    means_init[:, 0] += OFFSET + 0.02 * torch.randn(len(means0), device=dev, generator=g)

    lidar = make_lidar(dev, max_pts_per_tile="safe")
    dirs = lidar_world_dirs(lidar, r_ws)
    t_gt, k_gt, interior = analytic_hits(SENSOR_POS, dirs, wall_true, 0.3)
    mask = ((k_gt >= 0) & interior)[..., None]
    gt = torch.where(mask[..., 0], t_gt, torch.zeros_like(t_gt)).float()[..., None]

    def loss_of(m):
        d, a, _ = _render_lidar((m, quats, scales, opac), lidar, viewmat)
        return gsplat.losses.lidar_distance_loss(d[..., None], gt, valid_mask=mask), d, a

    # Vérification du gradient par différences finies : translation globale selon x
    m = means_init.clone().requires_grad_(True)
    L, _, _ = loss_of(m)
    L.backward()
    grad_shift_autograd = float(m.grad[:, 0].sum())
    eps = 1e-3
    with torch.no_grad():
        shift = torch.tensor([eps, 0, 0], device=dev)
        Lp = float(loss_of(means_init + shift)[0])
        Lm = float(loss_of(means_init - shift)[0])
    grad_shift_fd = (Lp - Lm) / (2 * eps)

    # Optimisation
    param = means_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([param], lr=5e-3)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)  # 5e-3 → 2,5e-4 en 300 pas
    steps = 300
    seen = torch.zeros(len(param), dtype=torch.bool, device=dev)
    history = []
    init_stats = None
    for it in range(steps + 1):
        opt.zero_grad()
        L, d, a = loss_of(param)
        if it == 0:
            init_stats = _stats((d.double() - t_gt)[mask[..., 0] & (a > 0.5)])
        if it % 25 == 0 or it == steps:
            history.append((it, round(float(L) * 1e3, 3)))
        if it == steps:
            break
        L.backward()
        seen |= param.grad.abs().sum(-1) > 0
        opt.step()
        sched.step()
    final_stats = _stats((d.double() - t_gt)[mask[..., 0] & (a > 0.5)])
    with torch.no_grad():
        ex0 = means_init[seen, 0] - X_TRUE
        ex1 = param[seen, 0] - X_TRUE
        tang = (param[seen, 1:] - means_init[seen, 1:]).norm(dim=-1)
        # Points d'impact analytiques des rayons supervisés sur le plan x = 10 (coordonnées y, z)
        o = torch.tensor(SENSOR_POS, dtype=torch.float64, device=dev)
        hits = (o + t_gt[..., None] * dirs)[mask[..., 0]][:, 1:].float()
        dmin_final = torch.cdist(param[:, 1:].detach(), hits).min(dim=1).values
        obs = dmin_final <= SIGMA_T
        ex_obs = param[obs, 0] - X_TRUE
        dmin_init = torch.cdist(means_init[:, 1:], hits).min(dim=1).values
        obs0 = dmin_init <= SIGMA_T
        ex_obs0_final = param[obs0, 0] - X_TRUE
    return {
        "n_gaussians": int(len(param)),
        "n_gaussians_with_gradient": int(seen.sum()),
        "n_supervised_rays": int(mask.sum()),
        "init_offset_m": OFFSET,
        "init_noise_std_m": 0.02,
        "optimizer": "Adam(means only, lr=5e-3, ExponentialLR gamma=0.99)",
        "return_fraction_final_on_supervised": round(float((a > 0.5)[mask[..., 0]].float().mean()), 4),
        "steps": steps,
        "fd_check_global_x_shift": {
            "autograd": round(grad_shift_autograd, 5),
            "finite_diff": round(grad_shift_fd, 5),
            "rel_err": round(abs(grad_shift_autograd - grad_shift_fd) / max(abs(grad_shift_fd), 1e-12), 4),
        },
        "loss_mm_history(step,L1_mm)": history,
        "rendered_distance_error_init": init_stats,
        "rendered_distance_error_final": final_stats,
        "gaussian_x_error_init_mm": {
            "mean": round(float(ex0.mean()) * 1e3, 2),
            "mean_abs": round(float(ex0.abs().mean()) * 1e3, 2),
        },
        "gaussian_x_error_final_mm": {
            "mean": round(float(ex1.mean()) * 1e3, 2),
            "mean_abs": round(float(ex1.abs().mean()) * 1e3, 2),
            "median_abs": round(float(ex1.abs().median()) * 1e3, 2),
        },
        "tangential_drift_mm_median": round(float(tang.median()) * 1e3, 2),
        "observed_final_within_1sigma_of_a_hit": {
            "n": int(obs.sum()),
            "frac_abs_x_err_lt_2mm": round(float((ex_obs.abs() < 0.002).float().mean()), 4),
            "frac_abs_x_err_lt_10mm": round(float((ex_obs.abs() < 0.010).float().mean()), 4),
            "frac_abs_x_err_gt_100mm": round(float((ex_obs.abs() > 0.100).float().mean()), 4),
            "x_error_mean_mm": round(float(ex_obs.mean()) * 1e3, 2),
            "x_error_median_abs_mm": round(float(ex_obs.abs().median()) * 1e3, 2),
            "x_error_p95_abs_mm": round(float(torch.quantile(ex_obs.abs().double(), 0.95)) * 1e3, 2),
        },
        "observed_init_within_1sigma_of_a_hit(final_x_error)": {
            "n": int(obs0.sum()),
            "x_error_mean_mm": round(float(ex_obs0_final.mean()) * 1e3, 2),
            "x_error_median_abs_mm": round(float(ex_obs0_final.abs().median()) * 1e3, 2),
            "x_error_p95_abs_mm": round(float(torch.quantile(ex_obs0_final.abs().double(), 0.95)) * 1e3, 2),
        },
    }


@app.function(image=image, timeout=10 * 60)
def build_info() -> dict:
    """Sans GPU : déclenche/valide la construction de l'image et renvoie les infos de compilation."""
    import json
    import os

    with open(GSPLAT_BUILD_INFO) as f:
        info = json.load(f)
    info["so_files"] = sorted(
        os.path.join(r, n) for r, _, fs in os.walk(GSPLAT_DIR + "/gsplat") for n in fs if n.endswith(".so")
    )
    print(json.dumps(info, indent=2))
    return info


@app.function(image=image, gpu="L4", timeout=40 * 60)
def smoke() -> dict:
    import json
    import time
    import traceback

    results = {}
    for name, fn in [("a_import", _test_import), ("b_T4.0a_lidar", _test_lidar), ("c_pinhole", _test_pinhole), ("d_gradient", _test_gradient)]:
        t0 = time.perf_counter()
        try:
            results[name] = fn()
            results[name]["_elapsed_s"] = round(time.perf_counter() - t0, 1)
        except Exception:  # noqa: BLE001
            results[name] = {"error": traceback.format_exc()[-4000:]}
        print(f"[{name}] terminé en {time.perf_counter() - t0:.1f}s", flush=True)
    print("SMOKE_RESULTS_JSON " + json.dumps(results), flush=True)  # copie dans les logs Modal
    return results


PYTEST_GROUPS = {
    "test_rasterization_lidar": ["tests/test_rasterization.py", "-k", "lidar"],
    "test_basic_lidar": ["tests/test_basic.py", "-k", "lidar"],
    "test_losses_lidar": ["tests/test_losses.py", "-k", "lidar"],
    "sensors_lidars": [
        "tests/sensors/kernels/lidars",
        "tests/sensors/models/lidars",
        "tests/sensors/functional/test_lidars.py",
        "tests/sensors/kernels/test_lidar_dispatch.py",
    ],
}


@app.function(image=image, gpu="L4", timeout=45 * 60)
def gsplat_pytests(group_timeout_s: int = 900) -> dict:
    import re
    import subprocess
    import time

    out = {}
    for name, args in PYTEST_GROUPS.items():
        cmd = ["python", "-m", "pytest", *args, "-q", "-rfE", "-p", "no:cacheprovider", "--color=no"]
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=GSPLAT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            stdout, _ = p.communicate(timeout=group_timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            p.kill()
            stdout, _ = p.communicate()
            timed_out = True
        dt = time.time() - t0
        tail = stdout.strip().splitlines()[-40:]
        summary = next((line for line in reversed(tail) if re.search(r"(passed|failed|error|skipped|no tests ran)", line)), "")
        counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed|deselected)", summary)}
        out[name] = {
            "cmd": " ".join(cmd),
            "returncode": p.returncode,
            "timed_out": timed_out,
            "seconds": round(dt, 1),
            "summary": summary,
            "counts": counts,
            "failures": [line for line in tail if line.startswith(("FAILED", "ERROR"))][:20],
            "tail": tail[-15:] if (p.returncode not in (0, 5) or timed_out) else [],
        }
        print(f"[pytest {name}] {summary} ({dt:.0f}s)", flush=True)
    import json

    print("PYTEST_RESULTS_JSON " + json.dumps(out), flush=True)
    return out


@app.local_entrypoint()
def main(pytest: bool = False, only_pytest: bool = False):
    import json

    log_dir = Path(__file__).resolve().parents[2] / "logs"
    (log_dir / "stage4").mkdir(parents=True, exist_ok=True)
    if not only_pytest:
        res = smoke.remote()
        print(json.dumps(res, indent=2, ensure_ascii=False))
        (log_dir / "stage4" / "smoke_gsplat_lidar.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    if pytest or only_pytest:
        res = gsplat_pytests.remote()
        print(json.dumps(res, indent=2, ensure_ascii=False))
        (log_dir / "stage4" / "gsplat_pytest.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
