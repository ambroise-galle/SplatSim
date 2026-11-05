"""Étape 1.6 — T1.3 : carte LiDAR contre le scan laser fixe (TLS), protocole du benchmark de reconstruction
Oxford Spires (`scripts/reconstruction_benchmark/main.py`, `oxspires_tools/eval.py`, commit b456e1e).

Protocole du benchmark (lu dans le code) et notre reproduction :
1. **Vérité terrain.** Scans TLS individuels (E57) → nuage fusionné sous-échantillonné à 1 cm (`merge_downsample_clouds`,
   Open3D `voxel_down_sample(0.01)`) + octree octomap à 0,1 m construit en lançant les rayons depuis chaque station
   (`processPCDFolder` : `insertPointCloud(cloud, origin, -1, lazy_eval=true, discretize=true)`).
   Ici : nuage 1 cm **fourni** (`merged-clouds-1cm-no-colour.pcd`, contrôlé contre les E57) ; octree reconstruit depuis
   les 19 E57 (`individual_cloud_e57.zip`, décompressé sur le disque éphémère du conteneur, jamais dans `/data/raw`), un
   arbre par station (union des voxels connus = arbre unique du benchmark), avec liboctomap et `octomap_tools.cpp`.
2. **Carte évaluée.** Nuages individuels dans le repère GT, fusionnés et sous-échantillonnés à 1 cm → rognés à la boîte
   englobante du nuage GT → re-sous-échantillonnés à 1 cm → points dans l'espace inconnu de l'octree GT retirés.
3. **Métriques** (`get_recon_metrics_multi_thresholds`, seuils 3/5/10/20 cm) : précision (accuracy) = moyenne des
   distances carte→GT ; complétude = moyenne des distances GT→carte (nuage GT complet, non rogné) ; precision/recall =
   part des distances < seuil ; F = moyenne harmonique.

Lancement (depuis la racine du projet) :
  .venv/bin/modal run splatpipe/stage1/eval_map.py::octomap_selftest_main
  .venv/bin/modal run splatpipe/stage1/eval_map.py::tls_main           # téléchargements + stations + octrees
  .venv/bin/modal run splatpipe/stage1/eval_map.py::evaluate_main      # T1.2 (agrégation) + T1.3 + figures + rapport
"""

from __future__ import annotations

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume
from splatpipe.stage1.images import map_image

app = modal.App("splatsim-stage1-eval-map")
vol = data_volume()

SEQ = "2024-03-13-observatory-quarter-01"
SITE = "observatory-quarter"
HF_REPO = "ori-drs/oxford_spires_dataset"
RAW = f"{DATA_MOUNT}/raw/oxford_spires"
GT_1CM_REL = f"ground_truth_map/{SITE}/merged-clouds-1cm-no-colour.pcd"
GT_5CM_REL = f"ground_truth_map/{SITE}/merged-clouds-5cm.pcd"
E57_ZIP_REL = f"ground_truth_map/{SITE}/individual_cloud_e57.zip"
OCTOMAP_RES = 0.1
DOWNSAMPLE = 0.01
THRESHOLDS = (0.03, 0.05, 0.1, 0.2)
PAPER_VILENS = {  # Oxford Spires, tableau 4, « Radcliffe Obs. Quarter 01 », carte VILENS-SLAM
    "accuracy": 0.047, "completeness": 0.233,
    "precision@0.05": 0.708, "recall@0.05": 0.536, "f1@0.05": 0.610,
    "precision@0.1": 0.909, "recall@0.1": 0.806, "f1@0.1": 0.854,
}


def tls_dir(seq: str = SEQ) -> str:
    from splatpipe.stage1.maptools import scene_root

    return f"{scene_root(DATA_MOUNT, seq)}/map/tls"


def _run(cmd: list[str]) -> dict:
    import json
    import subprocess

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} {cmd[1]} rc={r.returncode}: {r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


# =============================================================================
# Auto-test de l'outil octomap
# =============================================================================


@app.function(image=map_image(), cpu=2, memory=4096, timeout=600)
def octomap_selftest() -> dict:
    import os
    import tempfile

    import numpy as np

    # Un mur plan à x = 5 m vu depuis l'origine : l'espace entre l'origine et le mur est libre (connu), derrière inconnu.
    yy, zz = np.meshgrid(np.arange(-2, 2, 0.02), np.arange(-1, 1, 0.02))
    wall = np.c_[np.full(yy.size, 5.0), yy.ravel(), zz.ravel()].astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        wall.tofile(f"{d}/w.bin")
        b = _run(["octomap_tools", "build", f"{d}/w.bin", "0", "0", "0", "0.1", f"{d}/t.bt"])
        q = np.array([[2.5, 0, 0], [5.02, 0.05, 0.05], [7.0, 0, 0], [2.5, 5, 0], [-1, 0, 0]], dtype=np.float32)
        q.tofile(f"{d}/q.bin")
        r = _run(["octomap_tools", "query", f"{d}/t.bt", f"{d}/q.bin", f"{d}/m.bin"])
        mask = np.fromfile(f"{d}/m.bin", dtype=np.uint8).tolist()
        size = os.path.getsize(f"{d}/t.bt")
    ok = mask == [1, 1, 0, 0, 0]
    return {"build": b, "query": r, "mask": mask, "expected": [1, 1, 0, 0, 0], "ok": ok, "bt_bytes": size,
            "octomap_version": open("/opt/splatsim/octomap_version.txt").read().strip()}


@app.local_entrypoint()
def octomap_selftest_main():
    import json

    print(json.dumps(octomap_selftest.remote(), indent=1))


# =============================================================================
# Préparation du TLS : téléchargements, stations, octrees
# =============================================================================


def _process_station(args) -> dict:
    """Une station E57 : points (repère world) → centres uniques des voxels 0,1 m (entrée octomap discrétisée) et
    clés uniques 1 cm (contrôle du nuage fusionné fourni)."""
    import os
    import time
    import zipfile

    import numpy as np
    import pye57

    zip_path, member, out_dir, tmp_dir = args
    name = os.path.basename(member)[:-4]
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        e57_path = zf.extract(member, tmp_dir)
    e = pye57.E57(e57_path)
    h = e.get_header(0)
    data = e.read_scan(0, ignore_missing_fields=True)  # transform=True par défaut → repère world ; points invalides retirés
    xyz = np.stack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]], 1)
    del data
    getattr(e, "close", lambda: None)()
    os.remove(e57_path)
    t_read = time.time() - t0
    origin = np.asarray(h.translation, dtype=np.float64)
    quat_wxyz = np.asarray(h.rotation, dtype=np.float64)
    # Le benchmark passe par float32 (PCD) puis Affine3f : on reproduit la conversion float32.
    x32 = xyz.astype(np.float32)
    # octomap : key = floor(resolution_factor · coord) + 32768 ; discretize → centres des voxels uniques
    k = np.floor(x32.astype(np.float64) * (1.0 / 0.1)).astype(np.int64)
    kk = np.unique((k[:, 0] + 32768) << 32 | (k[:, 1] + 32768) << 16 | (k[:, 2] + 32768))
    kx, ky, kz = (kk >> 32) - 32768, ((kk >> 16) & 0xFFFF) - 32768, (kk & 0xFFFF) - 32768
    centers = ((np.stack([kx, ky, kz], 1).astype(np.float64) + 0.5) * 0.1).astype(np.float32)
    np.savez(f"{out_dir}/{name}.npz", origin=origin, quat_wxyz=quat_wxyz, endpoints_0p1=centers,
             n_points=np.int64(len(xyz)), bbox_min=xyz.min(0), bbox_max=xyz.max(0))
    # Clés 1 cm (grille ancrée en −1000 m) pour contrôler le nuage fusionné fourni
    k1 = np.unique(pack_1cm(xyz))
    np.save(f"{tmp_dir}/{name}_1cm.npy", k1)
    return {"name": name, "n_points": int(len(xyz)), "n_endpoint_voxels_0p1": int(len(centers)),
            "n_voxels_1cm": int(len(k1)), "origin": origin.tolist(), "quat_wxyz": quat_wxyz.tolist(),
            "bbox_min": xyz.min(0).tolist(), "bbox_max": xyz.max(0).tolist(), "read_s": round(t_read, 1),
            "total_s": round(time.time() - t0, 1)}


def pack_1cm(xyz):
    import numpy as np

    k = np.floor((np.asarray(xyz, dtype=np.float64) + 1000.0) / 0.01).astype(np.int64)
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=16, memory=96 * 1024,
              timeout=3 * 3600)
def tls_prepare(seq: str = SEQ, workers: int = 8) -> dict:
    import json
    import os
    import time
    import zipfile
    from multiprocessing import Pool

    import numpy as np
    from huggingface_hub import hf_hub_download

    from splatpipe.stage1.maptools import read_pcd

    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tdir = tls_dir(seq)
    os.makedirs(f"{tdir}/stations", exist_ok=True)
    # 1. Nuage GT 1 cm → /data/raw (seul ajout autorisé)
    t0 = time.time()
    gt1 = f"{RAW}/{GT_1CM_REL}"
    if not (os.path.exists(gt1) and os.path.getsize(gt1) == 1953549152):
        hf_hub_download(HF_REPO, GT_1CM_REL, repo_type="dataset", local_dir=RAW)
        vol.commit()
    out["gt_1cm"] = {"path": gt1, "bytes": os.path.getsize(gt1), "download_s": round(time.time() - t0, 1)}
    # 2. E57 → disque éphémère
    t0 = time.time()
    tmp = "/tmp/tls"
    os.makedirs(tmp, exist_ok=True)
    zpath = hf_hub_download(HF_REPO, E57_ZIP_REL, repo_type="dataset", local_dir=tmp)
    out["e57_zip"] = {"bytes": os.path.getsize(zpath), "download_s": round(time.time() - t0, 1)}
    with zipfile.ZipFile(zpath) as zf:
        members = sorted(m for m in zf.namelist() if m.endswith(".e57"))
    t0 = time.time()
    with Pool(workers) as pool:
        stations = pool.map(_process_station, [(zpath, m, f"{tdir}/stations", tmp) for m in members], chunksize=1)
    out["stations_s"] = round(time.time() - t0, 1)
    out["stations"] = stations
    vol.commit()
    # 3. Contrôle du nuage 1 cm fourni contre les E57
    keys = np.unique(np.concatenate([np.load(f"{tmp}/{s['name']}_1cm.npy") for s in stations]))
    cols, hdr = read_pcd(gt1)
    g = np.stack([cols["x"], cols["y"], cols["z"]], 1)
    rng = np.random.default_rng(0)
    samp = g[rng.choice(len(g), 2_000_000, replace=False)]
    member = np.isin(pack_1cm(samp), keys)
    # tolérance d'un voxel (grilles différentes) : voisins ±1 cm
    near = member.copy()
    for dx in (-1, 1):
        for ax in range(3):
            off = np.zeros(3)
            off[ax] = dx * 0.01
            near |= np.isin(pack_1cm(samp + off), keys)
    s5, h5 = read_pcd(f"{RAW}/{GT_5CM_REL}")
    g5 = np.stack([s5["x"], s5["y"], s5["z"]], 1)
    bb_st_min = np.min([s["bbox_min"] for s in stations], 0)
    bb_st_max = np.max([s["bbox_max"] for s in stations], 0)
    out["gt_check"] = {
        "gt_1cm_header": {k: hdr[k] for k in ("FIELDS", "POINTS", "VIEWPOINT")},
        "gt_1cm_n": int(len(g)), "gt_1cm_bbox": [g.min(0).tolist(), g.max(0).tolist()],
        "gt_5cm_n": int(len(g5)), "gt_5cm_bbox": [g5.min(0).tolist(), g5.max(0).tolist()], "gt_5cm_viewpoint": h5["VIEWPOINT"],
        "e57_total_points": int(sum(s["n_points"] for s in stations)),
        "e57_bbox": [bb_st_min.tolist(), bb_st_max.tolist()],
        "e57_unique_voxels_1cm_grid_minus1000": int(len(keys)),
        "gt_1cm_sample_in_e57_voxels": float(member.mean()),
        "gt_1cm_sample_within_1voxel_of_e57": float(near.mean()),
    }
    with open(f"{tdir}/tls_info.json", "w") as f:
        json.dump(out, f, indent=1)
    vol.commit()
    return out


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=2, memory=64 * 1024, timeout=2 * 3600)
def tls_octree(name: str, seq: str = SEQ) -> dict:
    import os
    import time

    import numpy as np

    tdir = tls_dir(seq)
    os.makedirs(f"{tdir}/octree", exist_ok=True)
    z = np.load(f"{tdir}/stations/{name}.npz")
    z["endpoints_0p1"].astype(np.float32).tofile("/tmp/pts.bin")
    o = z["origin"].astype(np.float32)  # point3d octomap = float
    t0 = time.time()
    out_bt = f"{tdir}/octree/{name}.bt"
    r = _run(["octomap_tools", "build", "/tmp/pts.bin", repr(float(o[0])), repr(float(o[1])), repr(float(o[2])),
              str(OCTOMAP_RES), "/tmp/tree.bt"])
    import shutil

    shutil.copyfile("/tmp/tree.bt", out_bt)
    vol.commit()
    r.update({"name": name, "bt_bytes": os.path.getsize(out_bt), "wall_s": round(time.time() - t0, 1)})
    return r


@app.local_entrypoint()
def tls_main(seq: str = SEQ, skip_prepare: bool = False):
    import json
    import os

    os.makedirs("logs/stage1/map", exist_ok=True)
    if not skip_prepare:
        prep = tls_prepare.remote(seq)
        with open("logs/stage1/map/tls_prepare.json", "w") as f:
            json.dump(prep, f, indent=1)
        names = [s["name"] for s in prep["stations"]]
        print(json.dumps({k: v for k, v in prep.items() if k != "stations"}, indent=1))
    else:
        names = [f"ObservatoryQuarter-{i:03d}" for i in range(1, 22) if i not in (14, 17)]
    res = list(tls_octree.map(names, kwargs={"seq": seq}, return_exceptions=True))
    ok = [r for r in res if isinstance(r, dict)]
    bad = [str(r) for r in res if not isinstance(r, dict)]
    with open("logs/stage1/map/tls_octree.json", "w") as f:
        json.dump({"ok": ok, "errors": bad}, f, indent=1)
    print(f"octrees : {len(ok)} OK, {len(bad)} erreurs")
    for r in ok:
        print(r)


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=16, memory=64 * 1024, timeout=3600)
def tls_frame_check(seq: str = SEQ) -> dict:
    """Écart de repère entre les stations E57 (poses d'en-tête) et le nuage fusionné fourni (repère world) :
    ICP point→plan des centres de voxels 0,1 m des stations sur le nuage fourni à 5 cm."""
    import glob

    import numpy as np
    from scipy.spatial import cKDTree

    from splatpipe.common.geometry import rot_to_rotvec
    from splatpipe.stage1.maptools import icp_point_to_plane, knn_normals, read_pcd

    c5, _ = read_pcd(f"{RAW}/{GT_5CM_REL}")
    g5 = np.stack([c5["x"], c5["y"], c5["z"]], 1).astype(np.float64)
    tree = cKDTree(g5)
    nrm, flat = knn_normals(g5, tree, k=12)
    src = np.concatenate([np.load(f)["endpoints_0p1"] for f in sorted(glob.glob(f"{tls_dir(seq)}/stations/*.npz"))]).astype(np.float64)
    rng = np.random.default_rng(0)
    src = src[rng.choice(len(src), min(1_000_000, len(src)), replace=False)]
    d0, _ = tree.query(src, workers=-1)
    T, hist = icp_point_to_plane(src, tree, nrm, np.eye(4), schedule=(0.3, 0.15, 0.08), iters=20)
    p = src @ T[:3, :3].T + T[:3, 3]
    d1, _ = tree.query(p, workers=-1)
    return {"T_provided_e57": T.tolist(), "rot_deg": float(np.degrees(np.linalg.norm(rot_to_rotvec(T[:3, :3])))),
            "trans_m": T[:3, 3].tolist(), "nn_before_median": float(np.median(d0)), "nn_after_median": float(np.median(d1)),
            "icp_last": hist[-1] if hist else None, "n_src": int(len(src))}


@app.local_entrypoint()
def tls_frame_main(seq: str = SEQ):
    import json

    r = tls_frame_check.remote(seq)
    with open("logs/stage1/map/tls_frame_check.json", "w") as f:
        json.dump(r, f, indent=1)
    print(json.dumps(r, indent=1))


# =============================================================================
# T1.2 (agrégation) + T1.3 (métriques) + figures + rapport
# =============================================================================

SOURCES = ["glim_final", "glim_rigid", "glim_slerp", "gt_oracle", "vilens", "vilens_noicp", "glim_final_kf", "gt_oracle_kf"]
LABELS = {
    "glim_final": "GLIM calib_r4, trajectoire finale (par point, forme IMU, σ=3 s)",
    "glim_rigid": "GLIM calib_r4, une pose par tour (avant correction)",
    "glim_slerp": "GLIM calib_r4, par point SLERP 10 Hz (sans IMU)",
    "gt_oracle": "Oracle GT (poses GT + forme IMU)",
    "vilens": "VILENS-SLAM fourni (notre protocole)",
    "vilens_noicp": "VILENS-SLAM fourni, sans ICP",
    "vilens_nofilter": "VILENS-SLAM fourni, sans filtre octree",
    "glim_final_kf": "GLIM final, 365 tours aux images clés VILENS",
    "gt_oracle_kf": "Oracle GT, 365 tours aux images clés VILENS",
}
SHORT = {"glim_final": "GLIM final", "glim_rigid": "GLIM rigide", "glim_slerp": "GLIM SLERP", "gt_oracle": "Oracle GT",
         "vilens": "VILENS", "vilens_noicp": "VILENS sans ICP", "vilens_nofilter": "VILENS sans octree",
         "glim_final_kf": "GLIM final (365 tours)", "gt_oracle_kf": "Oracle GT (365 tours)",
         "glim_final_calib": "GLIM final `calib` (1.7a)"}
COLORS = {"glim_final": "#2a78d6", "vilens": "#eb6834", "gt_oracle": "#52514e", "glim_rigid": "#4a3aa7",
          "glim_slerp": "#1baf7a", "vilens_noicp": "#eda100", "vilens_nofilter": "#e87ba4", "glim_final_kf": "#2a78d6",
          "gt_oracle_kf": "#52514e", "glim_final_calib": "#c8380d"}
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def recon_metrics(d_acc, d_cmp, thresholds=THRESHOLDS) -> dict:
    import numpy as np

    out = {"accuracy": float(np.mean(d_acc)), "completeness": float(np.mean(d_cmp)),
           "accuracy_median": float(np.median(d_acc)), "completeness_median": float(np.median(d_cmp)),
           "n_input": int(len(d_acc)), "n_gt": int(len(d_cmp))}
    for t in thresholds:
        p, r = float(np.mean(d_acc < t)), float(np.mean(d_cmp < t))
        out[f"precision@{t:g}"], out[f"recall@{t:g}"] = p, r
        out[f"f1@{t:g}"] = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return out


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _figures(fig_dir: str, topview: dict, hists: dict, t12: dict, runs: dict | None, traj: dict | None) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    files = []
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    # 1. Vue de dessus colorée par l'erreur carte→TLS
    names = [n for n in ("glim_final", "vilens", "gt_oracle") if n in topview]
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 5.4), facecolor=SURFACE, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        x, y, d = topview[n]
        o = np.argsort(d)
        sc = ax.scatter(x[o], y[o], c=np.clip(d[o], 0, 0.2), s=0.05, cmap=cmap, vmin=0, vmax=0.2, rasterized=True, linewidths=0)
        _style(ax)
        ax.set_aspect("equal")
        ax.set_title(SHORT[n], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("x world (m)", fontsize=8, color=INK2)
    axes[0].set_ylabel("y world (m)", fontsize=8, color=INK2)
    cb = fig.colorbar(sc, ax=axes.tolist(), shrink=0.8, pad=0.01)
    cb.set_label("distance au TLS (m, bornée à 0,2)", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    fig.suptitle("T1.3 — erreur carte → scan laser fixe (points filtrés, 1 cm)", fontsize=11, color=INK, x=0.01, ha="left")
    f = f"{fig_dir}/t13_topview_error.png"
    fig.savefig(f, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    files.append(f)
    # 2. Histogrammes des distances
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for ax, key, title in ((axes[0], "acc", "Précision : carte → TLS"), (axes[1], "cmp", "Complétude : TLS → carte")):
        for n in ("glim_final", "glim_rigid", "vilens", "gt_oracle"):
            if n not in hists:
                continue
            edges, h = hists[n][key]
            c = 0.5 * (edges[1:] + edges[:-1])
            ax.plot(c * 100, h / h.sum() / (edges[1] - edges[0]) / 100, color=COLORS[n], linewidth=1.6, label=SHORT[n])
        for t in (5, 10):
            ax.axvline(t, color=INK2, linewidth=0.8, linestyle=":")
        _style(ax)
        ax.set_xlim(0, 40 if key == "acc" else 100)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("distance (cm)", fontsize=8, color=INK2)
        ax.set_ylabel("densité (1/cm)", fontsize=8, color=INK2)
    axes[0].legend(fontsize=8, frameon=False)
    f = f"{fig_dir}/t13_distance_hist.png"
    fig.savefig(f, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    files.append(f)
    # 3. T1.2 : σ3 sur les patchs communs
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    s3, nm = t12["s3"], t12["names"]
    bins = np.linspace(0, 0.10, 81)
    for c, n in enumerate(nm):
        h, e = np.histogram(s3[:, c], bins=bins)
        ax.plot(100 * 0.5 * (e[1:] + e[:-1]), h, color=COLORS[n], linewidth=1.6, label=f"{SHORT[n]} (méd. {100 * np.median(s3[:, c]):.2f} cm)")
    ax.axvline(6, color=INK2, linewidth=0.8, linestyle=":")
    ax.text(6.1, ax.get_ylim()[1] * 0.95, "seuil 6 cm", fontsize=7, color=INK2, va="top")
    _style(ax)
    ax.set_xlabel("épaisseur σ3 d'un patch plan de 0,5 m (cm)", fontsize=8, color=INK2)
    ax.set_ylabel("patchs", fontsize=8, color=INK2)
    ax.set_title(f"T1.2 — épaisseur des plans, {len(s3)} patchs communs (repère world)", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    f = f"{fig_dir}/t12_sigma3_hist.png"
    fig.savefig(f, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    files.append(f)
    # 4. Choix de l'exécution : score sans GT vs ATE
    if runs:
        fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=SURFACE)
        tb = runs["table"]
        xs = [r["ate_bench_rmse_cm"] for r in tb]
        ys = [r["score_sigma3_median_cm"] for r in tb]
        ax.scatter(xs, ys, s=40, color=COLORS["glim_final"], edgecolors=SURFACE, linewidths=1.5, zorder=3)
        for r in tb:
            sel = r["run"] == runs["selected_run"]
            ax.annotate(r["run"] + (" (retenue)" if sel else ""), (r["ate_bench_rmse_cm"], r["score_sigma3_median_cm"]),
                        xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK if sel else INK2,
                        fontweight="bold" if sel else "normal")
        _style(ax)
        rho = runs["correlations"]["score_sigma3_median_cm"]["spearman_vs_ate"]
        ax.set_xlabel("ATE T1.1 (cm, protocole benchmark) — information seulement", fontsize=8, color=INK2)
        ax.set_ylabel("score sans GT : médiane σ3 (cm)", fontsize=8, color=INK2)
        ax.set_title(f"1.4a — netteté de la carte vs ATE, 12 exécutions (Spearman ρ = {rho:.2f})", fontsize=10, color=INK, loc="left")
        f = f"{fig_dir}/select_runs.png"
        fig.savefig(f, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        files.append(f)
    # 5. Variantes de trajectoire
    if traj:
        tb = traj["selection"]["table"]
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
        labels = [r["variant"].split("__")[1] for r in tb]
        vals = [r["sigma3_median_cm"] for r in tb]
        yy = np.arange(len(tb))[::-1]
        ax.barh(yy, vals, height=0.6, color=COLORS["glim_final"])
        for y, v in zip(yy, vals):
            ax.text(v + 0.05, y, f"{v:.3f}", va="center", fontsize=7, color=INK2)
        ax.set_yticks(yy, labels, fontsize=8)
        _style(ax)
        ax.grid(False, axis="y")
        ax.set_xlabel("médiane σ3 sur patchs communs (cm), 1 tour d'entraînement sur 5", fontsize=8, color=INK2)
        ax.set_title("1.4b — variantes de correction de distorsion (calib_r4)", fontsize=10, color=INK, loc="left")
        f = f"{fig_dir}/traj_variants.png"
        fig.savefig(f, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        files.append(f)
    return files


def _dir_bytes(path: str) -> int:
    import os

    if os.path.isfile(path):
        return os.path.getsize(path)
    tot = 0
    for dp, _, fs in os.walk(path):
        tot += sum(os.path.getsize(os.path.join(dp, f)) for f in fs)
    return tot


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=32, memory=192 * 1024, timeout=3 * 3600)
def evaluate(context: dict, seq: str = SEQ) -> dict:
    import glob
    import os
    import time

    import numpy as np
    from scipy.spatial import cKDTree

    from splatpipe.stage1.maptools import common_patch_metrics, paired_median_diff, read_pcd, scene_root, summarize

    T0 = time.time()
    root = scene_root(DATA_MOUNT, seq)
    fig_dir = f"{root}/reports/stage1_map"
    os.makedirs(fig_dir, exist_ok=True)
    rep = {"stage": "1.4-1.6", "tests": ["T1.2", "T1.3"], "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "dataset": "oxford_spires", "sequence": seq}
    # ------------------------------------------------------------------ T1.3
    cols, _ = read_pcd(f"{RAW}/{GT_1CM_REL}")
    gt = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    del cols
    tree_gt = cKDTree(gt, leafsize=32, balanced_tree=False, compact_nodes=False)
    t_tree = time.time() - T0
    rng = np.random.default_rng(0)
    t13, hists, topview = {}, {}, {}
    edges_acc = np.linspace(0, 1.0, 401)
    edges_cmp = np.linspace(0, 3.0, 601)
    clouds = [(s, f"{root}/map/eval/{s}_1cm_filtered.npy") for s in SOURCES]
    clouds.append(("vilens_nofilter", f"{root}/map/eval/vilens_1cm_nofilter.npy"))
    for s, path in clouds:
        if not os.path.exists(path):
            continue
        t1 = time.time()
        F = np.load(path).astype(np.float64)
        d_acc, _ = tree_gt.query(F, workers=-1)
        tf = cKDTree(F, leafsize=32, balanced_tree=False, compact_nodes=False)
        d_cmp, _ = tf.query(gt, workers=-1)
        del tf
        m = recon_metrics(d_acc, d_cmp)
        m["eval_s"] = round(time.time() - t1, 1)
        t13[s] = m
        hists[s] = {"acc": (edges_acc, np.histogram(np.clip(d_acc, 0, 1.0), edges_acc)[0].astype(np.float64)),
                    "cmp": (edges_cmp, np.histogram(np.clip(d_cmp, 0, 3.0), edges_cmp)[0].astype(np.float64))}
        k = rng.choice(len(F), min(1_500_000, len(F)), replace=False)
        topview[s] = (F[k, 0], F[k, 1], d_acc[k])
        del F, d_acc, d_cmp
    paper = PAPER_VILENS

    def cmp_baseline(base: dict, name: str = "glim_final") -> dict:
        g = t13[name]
        return {"accuracy_ok": g["accuracy"] <= base["accuracy"], "f1@0.05_ok": g["f1@0.05"] >= base["f1@0.05"],
                "f1@0.1_ok": g["f1@0.1"] >= base["f1@0.1"], "completeness_ok": g["completeness"] <= base["completeness"]}

    verdict13 = {}
    if "glim_final" in t13 and "vilens" in t13:
        a, b = cmp_baseline(t13["vilens"]), cmp_baseline(paper)
        verdict13 = {
            "criterion": "plan : ≥ baseline VILENS-SLAM → précision (accuracy) ≤ et F@10 cm ≥ ; verdict principal contre VILENS évalué avec NOTRE protocole (même filtrage), contrôle contre les valeurs de l'article",
            "vs_vilens_our_protocol": a, "vs_vilens_paper": b,
            "result": "PASS" if (a["accuracy_ok"] and a["f1@0.1_ok"]) else "FAIL",
            "result_vs_paper": "PASS" if (b["accuracy_ok"] and b["f1@0.1_ok"]) else "FAIL",
        }
        if "glim_final_kf" in t13:
            c = cmp_baseline(t13["vilens"], "glim_final_kf")
            verdict13["diagnostic_same_keyframes_vs_vilens_our_protocol"] = c
            verdict13["result_same_keyframes"] = "PASS" if (c["accuracy_ok"] and c["f1@0.1_ok"]) else "FAIL"
    rep["T1.3"] = {"protocol": {
        "gt_cloud": f"{RAW}/{GT_1CM_REL} (fourni ; 122 096 566 points)",
        "octree": "octomap 0,1 m par station E57 (19), union des voxels connus ; requêtes après passage world→repère E57 (tls_frame_check)",
        "input": "points des tours d'entraînement (VILENS : 365 nuages fournis) en world → voxel 1 cm (centroïde, ancrage Open3D) → boîte GT → voxel 1 cm → points connus",
        "thresholds_m": list(THRESHOLDS), "distances": "plus proche voisin (cKDTree), précision sur la carte filtrée, complétude sur le nuage GT complet"},
        "paper_vilens": paper, "results": t13, "verdict": verdict13,
        "context_clouds": context.get("maps_t13", {}), "gt_tree_build_s": round(t_tree, 1)}
    # ------------------------------------------------------------------ T1.2
    def load_t12(names):
        st = {}
        for n in names:
            p = f"{root}/reports/stage1_map/t12/{n}.npz"
            if os.path.exists(p):
                with np.load(p) as z:
                    st[n] = {k: z[k] for k in ("keys", "sigma", "planar", "tspan")}
        return st

    all5 = load_t12(["glim_final", "glim_rigid", "glim_slerp", "gt_oracle", "vilens"])
    t12 = {}
    cm5 = common_patch_metrics(all5)
    t12["common_all_maps"] = {k: cm5[k] for k in ("n_union", "n_common", "n_revisit", "per_map")}
    t12["common_all_maps"]["paired_vs_glim_final"] = paired_median_diff(cm5["_s3_common"], cm5["_names"], "glim_final")
    fig_t12 = {"s3": cm5["_s3_common"], "names": cm5["_names"]}
    g4 = {k: v for k, v in all5.items() if k != "vilens"}
    cm4 = common_patch_metrics(g4)
    t12["common_glim_gt"] = {k: cm4[k] for k in ("n_union", "n_common", "n_revisit", "per_map")}
    t12["common_glim_gt"]["paired_vs_glim_final"] = paired_median_diff(cm4["_s3_common"], cm4["_names"], "glim_final")
    own = {n: summarize(v["sigma"][v["planar"], 2]) for n, v in all5.items()}
    t12["own_planar_patches"] = own
    pm = t12["common_glim_gt"]["per_map"]
    fin, rig = pm["glim_final"]["all"]["median"], pm["glim_rigid"]["all"]["median"]
    pr = t12["common_glim_gt"]["paired_vs_glim_final"]["glim_rigid"]
    improve = (rig - fin) / rig
    t12["verdict"] = {
        "criterion": "baisse nette avant→après (médiane σ3 : −10 % au moins et IC 95 % apparié excluant 0) ET épaisseur médiane < 6 cm (2 × ±3 cm QT64)",
        "glim_final_sigma3_median_cm": 100 * fin, "glim_rigid_sigma3_median_cm": 100 * rig, "relative_drop": improve,
        "paired_rigid_minus_final_ci95_cm": [100 * x for x in pr["ci95_m"]],
        "gt_oracle_sigma3_median_cm": 100 * pm["gt_oracle"]["all"]["median"],
        "result": "PASS" if (improve >= 0.10 and pr["ci95_m"][0] > 0 and fin < 0.06) else "FAIL",
    }
    rep["T1.2"] = t12
    # ------------------------------------------------------------------ significativité des choix 1.4
    sig = {}
    for tag, ref in (("runs", context.get("selected_run_name")), ("traj", context.get("selected_traj_name"))):
        files = sorted(glob.glob(f"{root}/reports/stage1_map/{tag}/*.npz"))
        if not files or not ref:
            continue
        st = {}
        for p in files:
            with np.load(p) as z:
                st[os.path.basename(p)[:-4]] = {k: z[k] for k in ("keys", "sigma", "planar", "tspan")}
        cm = common_patch_metrics(st)
        sig[tag] = {"ref": ref, "n_common": cm["n_common"], "paired": paired_median_diff(cm["_s3_common"], cm["_names"], ref)}
    rep["selection_significance"] = sig
    rep["selection"] = context.get("selection", {})
    # ------------------------------------------------------------------ figures, tailles
    files = _figures(fig_dir, topview, hists, fig_t12, context.get("runs"), context.get("traj"))
    rep["figures"] = [os.path.relpath(f, root) for f in files]
    sizes = {}
    for rel in ("lidar/hesai_deskewed", "map/map_5cm.ply", "map/map_5cm_gt_oracle.ply", "map/eval", "map/tls", "traj/spline.npz",
                "traj/final.tum", "reports/stage1_map", f"{RAW}/{GT_1CM_REL}"):
        p = rel if rel.startswith("/") else f"{root}/{rel}"
        if os.path.exists(p):
            sizes[rel] = _dir_bytes(p)
    rep["storage_bytes"] = sizes
    rep["context"] = {k: v for k, v in context.items() if k not in ("runs", "traj", "selection")}
    rep["evaluate_s"] = round(time.time() - T0, 1)
    import json

    with open(f"{root}/reports/stage1_map.json", "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False, default=float)
    vol.commit()
    return {"report": json.loads(json.dumps(rep, default=float)), "figures": {os.path.basename(f): open(f, "rb").read() for f in files}}


@app.local_entrypoint()
def evaluate_main(seq: str = SEQ):
    import json
    import os

    L = "logs/stage1/map"
    ld = lambda n: json.load(open(f"{L}/{n}.json")) if os.path.exists(f"{L}/{n}.json") else None  # noqa: E731
    runs, traj, maps = ld("select_runs"), ld("select_traj"), ld("maps") or {}
    fc = traj["selection"].get("final_choice", {})
    context = {
        "runs": {k: runs[k] for k in ("table", "selected_run", "correlations", "common_patches", "selected_ate_rank_among_12", "criterion")},
        "traj": {"selection": traj["selection"]},
        "selection": {"run": runs["selected_run"], "runs_criterion": runs["criterion"], "correlations": runs["correlations"],
                      "traj_method": fc.get("method"), "traj_sigma_s": fc.get("sigma_s"), "traj_decision_note": fc.get("decision_note"),
                      "final_files": traj.get("final")},
        "selected_run_name": f"{runs['selected_run']}__gyro",
        "selected_traj_name": f"{runs['selected_run']}__{fc.get('method')}_s{fc.get('sigma_s'):g}",
        "maps_t13": {k: v.get("t13_cloud") for k, v in maps.items() if isinstance(v, dict)},
        "maps_info": {k: {kk: vv for kk, vv in v.items() if kk not in ("t13_cloud",)} for k, v in maps.items() if isinstance(v, dict)},
        "deskew": ld("deskew"), "tls_frame_check": ld("tls_frame_check"),
        "tls_prepare": {k: v for k, v in (ld("tls_prepare") or {}).items() if k != "stations"},
        "tls_octree": [{k: r[k] for k in ("name", "n_points", "leafs_before_prune", "leafs_after_prune", "insert_s", "bt_bytes")}
                       for r in (ld("tls_octree") or {}).get("ok", [])],
    }
    out = evaluate.remote(context, seq)
    os.makedirs(L, exist_ok=True)
    with open("logs/stage1/map.json", "w") as f:
        json.dump(out["report"], f, indent=1, ensure_ascii=False)
    for name, b in out["figures"].items():
        open(f"{L}/{name}", "wb").write(b)
    r = out["report"]
    print(json.dumps({"T1.2": r["T1.2"]["verdict"], "T1.3": r["T1.3"]["verdict"]}, indent=1, ensure_ascii=False))
    for s, m in r["T1.3"]["results"].items():
        print(f"{s:16s} acc {m['accuracy']:.3f} comp {m['completeness']:.3f} P/R/F@5 {m['precision@0.05']:.3f}/{m['recall@0.05']:.3f}/{m['f1@0.05']:.3f} "
              f"@10 {m['precision@0.1']:.3f}/{m['recall@0.1']:.3f}/{m['f1@0.1']:.3f} n {m['n_input']}")


@app.function(image=map_image(), cpu=2, memory=4096, timeout=600)
def figures_selftest(runs: dict, traj: dict) -> dict:
    import os

    import numpy as np

    rng = np.random.default_rng(0)
    d = "/tmp/figtest"
    os.makedirs(d, exist_ok=True)
    tv = {n: (rng.uniform(-100, 50, 20000), rng.uniform(-100, 100, 20000), np.abs(rng.normal(0, 0.05, 20000))) for n in ("glim_final", "vilens", "gt_oracle")}
    ea, ec = np.linspace(0, 1, 401), np.linspace(0, 3, 601)
    h = {n: {"acc": (ea, np.histogram(np.abs(rng.normal(0, .05, 10000)), ea)[0].astype(float)),
             "cmp": (ec, np.histogram(np.abs(rng.normal(0, .3, 10000)), ec)[0].astype(float))} for n in ("glim_final", "glim_rigid", "vilens", "gt_oracle")}
    t12 = {"s3": np.abs(rng.normal(0.035, 0.01, (1000, 5))), "names": ["glim_final", "glim_rigid", "glim_slerp", "gt_oracle", "vilens"]}
    files = _figures(d, tv, h, t12, runs, traj)
    return {os.path.basename(f): open(f, "rb").read() for f in files}


@app.local_entrypoint()
def figures_selftest_main():
    import json

    out = figures_selftest.remote(json.load(open("logs/stage1/map/select_runs.json")), json.load(open("logs/stage1/map/select_traj.json")))
    d = "logs/stage1/map/figtest"
    import os

    os.makedirs(d, exist_ok=True)
    for k, v in out.items():
        open(f"{d}/{k}", "wb").write(v)
    print(sorted(out))


# =============================================================================
# Sous-étape 1.7a — évaluation de la variante {source}{suffix} contre la carte existante
# =============================================================================

VARIANT_COLORS = {"glim_final": "#2a78d6", "glim_final_calib": "#c8380d", "gt_oracle": "#52514e"}
VARIANT_SHORT = {"glim_final": "GLIM final calib_r4", "glim_final_calib": "GLIM final calib", "gt_oracle": "Oracle GT"}


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=32, memory=192 * 1024, timeout=3 * 3600)
def evaluate_variant(suffix: str, reference: dict, seq: str = SEQ) -> dict:
    """Métriques T1.3 recalculées pour la carte `glim_final{suffix}` (protocole identique à `evaluate`), T1.2 en
    patchs communs contre `glim_final` et `gt_oracle`, figures de comparaison et PLY colorés par erreur.
    `reference` = section T1.3 du rapport précédent (metrics VILENS, glim_final…), pour le verdict sans recalcul.
    Écrit reports/stage1_map_variant{suffix}.json + reports/stage1_map_variant{suffix}/*.ply sur le Volume."""
    import json
    import os
    import time

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.spatial import cKDTree

    from splatpipe.common.geometry import read_tum, rot_to_quat
    from splatpipe.stage1.maptools import common_patch_metrics, paired_median_diff, read_pcd, scene_root, summarize, write_ply

    T0 = time.time()
    root = scene_root(DATA_MOUNT, seq)
    variant = f"glim_final{suffix}"
    fig_dir = f"{root}/reports/stage1_map_variant{suffix}"
    os.makedirs(fig_dir, exist_ok=True)
    rep = {"stage": "1.7a", "variant": variant, "reference_map": "glim_final",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dataset": "oxford_spires", "sequence": seq}
    files, ply_files = [], []
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)

    # ---- T1.3 : distances au TLS pour la variante, la carte de référence et l'oracle (pour figures appariées)
    cols, _ = read_pcd(f"{RAW}/{GT_1CM_REL}")
    gt = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    del cols
    tree_gt = cKDTree(gt, leafsize=32, balanced_tree=False, compact_nodes=False)
    t_tree = time.time() - T0
    rng = np.random.default_rng(0)
    edges_acc = np.linspace(0, 1.0, 401)
    edges_cmp = np.linspace(0, 3.0, 601)
    t13, hists, topview = {}, {}, {}
    for s in (variant, "glim_final", "gt_oracle"):
        path = f"{root}/map/eval/{s}_1cm_filtered.npy"
        if not os.path.exists(path):
            continue
        t1 = time.time()
        F = np.load(path).astype(np.float64)
        d_acc, _ = tree_gt.query(F, workers=-1)
        tf = cKDTree(F, leafsize=32, balanced_tree=False, compact_nodes=False)
        d_cmp, _ = tf.query(gt, workers=-1)
        del tf
        m = recon_metrics(d_acc, d_cmp)
        m["eval_s"] = round(time.time() - t1, 1)
        t13[s] = m
        hists[s] = {"acc": (edges_acc, np.histogram(np.clip(d_acc, 0, 1.0), edges_acc)[0].astype(np.float64)),
                    "cmp": (edges_cmp, np.histogram(np.clip(d_cmp, 0, 3.0), edges_cmp)[0].astype(np.float64))}
        k = rng.choice(len(F), min(1_500_000, len(F)), replace=False)
        topview[s] = (F[k, 0], F[k, 1], d_acc[k])
        del F, d_acc, d_cmp
    rep["T1.3"] = {"results_computed": t13, "reference_report": reference.get("results", {}),
                   "paper_vilens": PAPER_VILENS}

    # ---- verdict T1.3 de la variante (critère inchangé : précision ≤ VILENS et F@10 ≥ VILENS, protocole identique)
    verdict = {}
    if variant in t13 and "vilens" in reference.get("results", {}):
        v, base = t13[variant], reference["results"]["vilens"]
        verdict = {"criterion": "précision (accuracy) ≤ VILENS ET F@10 cm ≥ VILENS, tous deux évalués avec NOTRE protocole",
                   "accuracy_cm_variant": 100 * v["accuracy"], "accuracy_cm_vilens": 100 * base["accuracy"],
                   "f1@0.1_variant": v["f1@0.1"], "f1@0.1_vilens": base["f1@0.1"],
                   "accuracy_ok": v["accuracy"] <= base["accuracy"], "f1@0.1_ok": v["f1@0.1"] >= base["f1@0.1"],
                   "f1@0.05_ok": v["f1@0.05"] >= base["f1@0.05"], "completeness_ok": v["completeness"] <= base["completeness"]}
        verdict["result"] = "PASS" if (verdict["accuracy_ok"] and verdict["f1@0.1_ok"]) else "FAIL"
    rep["T1.3"]["verdict"] = verdict

    # ---- figures T1.3 : vue de dessus + histogrammes
    names = [n for n in (variant, "glim_final", "gt_oracle") if n in topview]
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 5.4), facecolor=SURFACE, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        x, y, d = topview[n]
        o = np.argsort(d)
        sc = ax.scatter(x[o], y[o], c=np.clip(d[o], 0, 0.2), s=0.05, cmap=cmap, vmin=0, vmax=0.2, rasterized=True, linewidths=0)
        _style(ax)
        ax.set_aspect("equal")
        ax.set_title(VARIANT_SHORT[n] + f" — précision {100 * t13[n]['accuracy']:.2f} cm", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("x world (m)", fontsize=8, color=INK2)
    axes[0].set_ylabel("y world (m)", fontsize=8, color=INK2)
    cb = fig.colorbar(sc, ax=axes.tolist(), shrink=0.8, pad=0.01)
    cb.set_label("distance au TLS (m, bornée à 0,2)", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    fig.suptitle(f"T1.3 (1.7a) — erreur carte → scan laser fixe, {variant}", fontsize=11, color=INK, x=0.01, ha="left")
    f = f"{fig_dir}/t13_topview_error{suffix}.png"
    fig.savefig(f, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    files.append(f)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    for n in names:
        edges, h = hists[n]["acc"]
        c = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(c * 100, h / h.sum() / (edges[1] - edges[0]) / 100, color=VARIANT_COLORS.get(n, COLORS.get(n)),
                linewidth=1.6, label=f"{VARIANT_SHORT[n]} (méd. {100 * t13[n]['accuracy_median']:.2f} cm)")
    for t in (5, 10):
        ax.axvline(t, color=INK2, linewidth=0.8, linestyle=":")
    _style(ax)
    ax.set_xlim(0, 40)
    ax.set_xlabel("précision : distance carte → TLS (cm)", fontsize=8, color=INK2)
    ax.set_ylabel("densité (1/cm)", fontsize=8, color=INK2)
    ax.set_title(f"T1.3 (1.7a) — distribution des distances, {variant}", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    f = f"{fig_dir}/t13_accuracy_hist{suffix}.png"
    fig.savefig(f, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    files.append(f)

    # ---- T1.2 : patchs communs variante / glim_final / gt_oracle + nuage de points apparié
    t12 = {}
    st = {}
    for n in (variant, "glim_final", "gt_oracle"):
        p = f"{root}/reports/stage1_map/t12/{n}.npz"
        if os.path.exists(p):
            with np.load(p) as z:
                st[n] = {k: z[k] for k in ("keys", "sigma", "planar", "tspan")}
    if variant in st and "glim_final" in st:
        cm = common_patch_metrics(st)
        t12["common"] = {k: cm[k] for k in ("n_union", "n_common", "n_revisit", "per_map")}
        t12["paired_vs_glim_final"] = paired_median_diff(cm["_s3_common"], cm["_names"], "glim_final")
        t12["paired_vs_variant"] = paired_median_diff(cm["_s3_common"], cm["_names"], variant)
        s3 = cm["_s3_common"]
        iv, ir = cm["_names"].index(variant), cm["_names"].index("glim_final")
        fig, ax = plt.subplots(figsize=(5.6, 5.4), facecolor=SURFACE)
        ax.plot([0, 12], [0, 12], color=INK2, linewidth=0.8, linestyle=":")
        ax.scatter(100 * s3[:, ir], 100 * s3[:, iv], s=1.5, color=VARIANT_COLORS[variant], alpha=0.25, linewidths=0)
        med_r, med_v = 100 * np.median(s3[:, ir]), 100 * np.median(s3[:, iv])
        _style(ax)
        ax.set_xlim(0, 12), ax.set_ylim(0, 12)
        ax.set_aspect("equal")
        ax.set_xlabel(f"épaisseur σ3 GLIM final calib_r4 (cm), méd. {med_r:.2f}", fontsize=8, color=INK2)
        ax.set_ylabel(f"épaisseur σ3 {VARIANT_SHORT[variant]} (cm), méd. {med_v:.2f}", fontsize=8, color=INK2)
        above = float((s3[:, iv] > s3[:, ir]).mean())
        ax.set_title(f"T1.2 (1.7a) — {len(s3)} patchs communs, {variant}\n(au-dessus de la diagonale : {100 * above:.0f} %)",
                     fontsize=10, color=INK, loc="left")
        f = f"{fig_dir}/t12_paired_scatter{suffix}.png"
        fig.savefig(f, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        files.append(f)
        rep["T1.2"] = t12

    # ---- trajectoires en world (ICP affiné de chaque carte) + GT
    tr_info = {}
    fig, ax = plt.subplots(figsize=(7, 6.4), facecolor=SURFACE)
    t_gt, p_gt, _ = read_tum(f"{root}/traj/gt.tum")
    ax.plot(p_gt[:, 0], p_gt[:, 1], color="#b5b4b0", linewidth=1.0, label="GT (world)")
    for n, sfx in (("glim_final", ""), (variant, suffix)):
        with open(f"{root}/traj/final_to_world{sfx}.yaml") as f:
            T_w = np.array(yaml.safe_load(f)["icp_refined"]["T_world_map"])
        t, p, q = read_tum(f"{root}/traj/final{sfx}.tum")
        pw = p @ T_w[:3, :3].T + T_w[:3, 3]
        ax.plot(pw[:, 0], pw[:, 1], color=VARIANT_COLORS[n], linewidth=1.1, label=VARIANT_SHORT[n])
        tr_info[n] = {"n_poses": int(len(t)), "T_world_map_yaw_deg": float(np.degrees(np.arctan2(T_w[1, 0], T_w[0, 0])))}
    _style(ax)
    ax.set_aspect("equal")
    ax.set_xlabel("x world (m)", fontsize=8, color=INK2)
    ax.set_ylabel("y world (m)", fontsize=8, color=INK2)
    ax.set_title(f"1.7a — trajectoires finales en world (alignées par ICP), {variant}", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    f = f"{fig_dir}/trajectories_world{suffix}.png"
    fig.savefig(f, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    files.append(f)
    rep["trajectories"] = tr_info

    # ---- PLY colorés par erreur (world, sous-échantillonnés) pour CloudCompare
    for n in (variant, "glim_final"):
        if n not in t13:
            continue
        path = f"{root}/map/eval/{n}_1cm_filtered.npy"
        F = np.load(path).astype(np.float64)
        d, _ = tree_gt.query(F, workers=-1)
        k = rng.choice(len(F), min(2_000_000, len(F)), replace=False)
        cols2 = np.clip(d[k] / 0.2, 0, 1)
        rgba = (cmap(cols2)[:, :3] * 255).astype(np.uint8)
        ply = f"{fig_dir}/{n}_errcol{suffix}.ply"
        nb = write_ply(ply, {"x": F[k, 0].astype(np.float32), "y": F[k, 1].astype(np.float32), "z": F[k, 2].astype(np.float32),
                             "red": rgba[:, 0], "green": rgba[:, 1], "blue": rgba[:, 2], "dist_m": d[k].astype(np.float32)},
                       comments=[f"SplatSim 1.7a {n} : nuage 1cm filtre (world), couleur = distance NN au TLS (0-0.2 m)",
                                 "bleu = proche du TLS, rouge = loin ; sous-echantillon 2M points"])
        ply_files.append(ply)
        del F, d
    rep["ply_files"] = [os.path.basename(p) for p in ply_files]

    rep["storage_bytes"] = {os.path.basename(p): os.path.getsize(p) for p in ply_files}
    rep["gt_tree_build_s"] = round(t_tree, 1)
    rep["evaluate_s"] = round(time.time() - T0, 1)
    with open(f"{root}/reports/stage1_map_variant{suffix}.json", "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False, default=float)
    vol.commit()
    return {"report": json.loads(json.dumps(rep, default=float)),
            "figures": {os.path.basename(f): open(f, "rb").read() for f in files}}


@app.local_entrypoint()
def evaluate_variant_main(suffix: str = "_calib", seq: str = SEQ):
    import json
    import os

    prev = json.load(open("logs/stage1/map.json"))  # rapport 1.4–1.6 (metrics de référence, VILENS inclus)
    out = evaluate_variant.remote(suffix, {"results": prev["T1.3"]["results"]}, seq)
    L = f"logs/stage1/map{suffix}"
    os.makedirs(L, exist_ok=True)
    with open(f"{L}/stage1_map_variant{suffix}.json", "w") as f:
        json.dump(out["report"], f, indent=1, ensure_ascii=False)
    for name, b in out["figures"].items():
        open(f"{L}/{name}", "wb").write(b)
    r = out["report"]
    print(json.dumps({"verdict_T1.3": r["T1.3"]["verdict"], "T1.2": r.get("T1.2", {}).get("common", {}).get("per_map")},
                     indent=1, ensure_ascii=False))
    for s, m in r["T1.3"]["results_computed"].items():
        print(f"{s:20s} acc {m['accuracy']:.3f} comp {m['completeness']:.3f} P/R/F@5 {m['precision@0.05']:.3f}/{m['recall@0.05']:.3f}/{m['f1@0.05']:.3f} "
              f"@10 {m['precision@0.1']:.3f}/{m['recall@0.1']:.3f}/{m['f1@0.1']:.3f} n {m['n_input']}")
