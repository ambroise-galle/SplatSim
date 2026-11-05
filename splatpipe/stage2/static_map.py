"""Sous-étape 2.2 (suite) — carte statique 5 cm à partir des étiquettes DUFOMap, et contrôle T1.3.

`build_static_map` : tours d'entraînement (`lidar/hesai_deskewed`) moins les points étiquetés dynamiques
(`lidar/dyn_labels_dufo`) → `map/map_5cm_static.ply` (repère map, même format/protocole que `map/map_5cm.ply` :
voxels 5 cm, normales ACP k=16, n_scans, intensité, last_echo_frac, flatness). Écrit aussi, pour le contrôle,
le nuage 1 cm « protocole benchmark » des points statiques ET des points retirés (world, via le `icp_refined`
de `traj/final_to_world.yaml` inchangé) dans `reports/stage2_dufo/`.

`eval_static` : métriques T1.3 (protocole inchangé : voxel 1 cm → boîte GT → voxel 1 cm → filtre octree TLS)
sur la carte statique, et distribution des distances au TLS des points retirés — un point retiré proche du TLS
est de la géométrie réelle déplacée (végétation au vent), un point retiré loin du TLS est un objet transitoire
(passant, véhicule en mouvement) absent du scan de référence.

Lancement :
  .venv/bin/modal run splatpipe/stage2/static_map.py::static_map_main
  .venv/bin/modal run splatpipe/stage2/static_map.py::eval_static_main
"""

from __future__ import annotations

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume
from splatpipe.stage1.images import map_image

app = modal.App("splatsim-stage2-static-map")
vol = data_volume()

SEQ = "2024-03-13-observatory-quarter-01"
DESK_SUB = "lidar/hesai_deskewed"
DYN_SUB = "lidar/dyn_labels_dufo"
OUT_DIR = "reports/stage2_dufo"
RAW = f"{DATA_MOUNT}/raw/oxford_spires"
GT_1CM = f"{RAW}/ground_truth_map/observatory-quarter/merged-clouds-1cm-no-colour.pcd"


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=16, memory=128 * 1024, timeout=3 * 3600)
def build_static_map(seq: str = SEQ) -> dict:
    import json
    import os
    import time

    import numpy as np
    import yaml

    from splatpipe.common.geometry import read_tum
    from splatpipe.stage1.deskew_map import map_5cm
    from splatpipe.stage1.maptools import (
        crop_aabb, lidar_splits, list_scans, o3d_voxel_down_sample, scene_root, write_ply,
    )

    T0 = time.time()
    root = scene_root(DATA_MOUNT, seq)
    stamps, paths = list_scans(root)
    train = set(lidar_splits(root)["train"].tolist())
    # 11 tours d'entraînement sont hors trajectoire (absents de hesai_deskewed) : on les exclut comme en 1.5.
    import os

    desk = {int(s[:-4]) for s in os.listdir(f"{root}/{DESK_SUB}") if s.endswith(".npz")}
    idx = [i for i, s in enumerate(stamps) if int(s) in train and int(s) in desk]
    with open(f"{root}/traj/final_to_world.yaml") as f:
        T_w = np.array(yaml.safe_load(f)["icp_refined"]["T_world_map"])

    xyz_l, inten_l, ret_l, sid_l, org_l = [], [], [], [], []
    removed_l, removed_org_l, removed_sid_l = [], [], []
    per_scan = []
    for k, i in enumerate(idx):
        st = int(stamps[i])
        with np.load(f"{root}/{DESK_SUB}/{st}.npz") as z:
            xyz = z["xyz"]
            inten, ret = z["intensity"], z["ret"]
            T0_, T1_ = z["T_map_lidar_start"], z["T_map_lidar_end"]
        with np.load(f"{root}/{DYN_SUB}/{st}.npz") as z:
            dyn = z["dyn"].astype(bool)
        org = 0.5 * (T0_[:3, 3] + T1_[:3, 3])
        keep = ~dyn
        xyz_l.append(xyz[keep])
        inten_l.append(inten[keep])
        ret_l.append(ret[keep])
        sid_l.append(np.full(int(keep.sum()), k, np.int32))
        org_l.append(org)
        removed_l.append(xyz[dyn])
        removed_sid_l.append(np.full(int(dyn.sum()), k, np.int32))
        removed_org_l.append(org)
        per_scan.append({"stamp": st, "n_points": int(len(xyz)), "n_dyn": int(dyn.sum())})
    xyz = np.concatenate(xyz_l)
    inten = np.concatenate(inten_l)
    ret = np.concatenate(ret_l)
    sid = np.concatenate(sid_l)
    origins = np.array(org_l)
    n_total = len(xyz) + sum(len(r) for r in removed_l)
    info = {
        "n_scans_train": len(idx), "n_points_total": int(n_total), "n_points_static": int(len(xyz)),
        "n_points_removed": int(n_total - len(xyz)), "frac_removed": float((n_total - len(xyz)) / n_total),
        "dyn_frac_from_labels": float(np.mean([p["n_dyn"] / max(p["n_points"], 1) for p in per_scan])),
    }
    m = map_5cm(xyz.astype(np.float64), inten, ret, sid, origins)
    os.makedirs(f"{root}/map", exist_ok=True)
    ply = f"{root}/map/map_5cm_static.ply"
    frame_name = "map (repere carte GLIM ; world via traj/final_to_world.yaml)"
    comments = [f"SplatSim etape 2.2 carte statique DUFOMap frame={frame_name}",
                "voxel 0.05 m, tours train, points dyn_labels_dufo==0 (DUFOMap 0.1 m, d_s 0.2, d_p 2) retires",
                "meme format que map_5cm.ply ; retours < 1.5 m retires ; deux echos gardes"]
    info["ply"] = {"path": "map/map_5cm_static.ply", "bytes": write_ply(ply, m, comments=comments),
                   "n_voxels": int(len(m["x"]))}

    # Nuages 1 cm « protocole » (world) : statique et retirés, pour eval_static
    os.makedirs(f"{root}/{OUT_DIR}", exist_ok=True)
    for name, pts in (("static", xyz.astype(np.float64)), ("removed", np.concatenate(removed_l).astype(np.float64))):
        w = pts @ T_w[:3, :3].T + T_w[:3, 3]
        d1 = o3d_voxel_down_sample(w, 0.01)
        d1 = d1[crop_aabb(d1, *np.array(json.load(open(f"{root}/map/tls/tls_info.json"))["gt_check"]["gt_1cm_bbox"]))]
        d2 = o3d_voxel_down_sample(d1, 0.01)
        np.save(f"{root}/{OUT_DIR}/{name}_1cm.npy", d2.astype(np.float32))
        info[f"n_{name}_1cm"] = int(len(d2))
    vol.commit()
    info["total_s"] = round(time.time() - T0, 1)
    json.dump({"info": info, "per_scan": per_scan}, open(f"{root}/{OUT_DIR}/build_info.json", "w"), indent=1)
    vol.commit()
    return info


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=32, memory=192 * 1024, timeout=3 * 3600)
def eval_static(T_provided_e57: list, reference: dict, seq: str = SEQ) -> dict:
    """Métriques T1.3 de la carte statique ; analyse des points retirés ; figures + PLY."""
    import json
    import os
    import time

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial import cKDTree
    from matplotlib.colors import LinearSegmentedColormap

    from splatpipe.common.geometry import inv
    from splatpipe.stage1.deskew_map import _query_octree
    from splatpipe.stage1.eval_map import BLUE_RAMP, INK, INK2, GRID, SURFACE, _style, recon_metrics
    from splatpipe.stage1.maptools import read_pcd, scene_root

    T0 = time.time()
    root = scene_root(DATA_MOUNT, seq)
    out_dir = f"{root}/{OUT_DIR}"
    rep = {"stage": "2.2", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "reference_full_map": reference}
    with open(f"{root}/map/tls/tls_info.json") as f:
        tls_info = json.load(f)
    gt_lo, gt_hi = (np.array(tls_info["gt_check"]["gt_1cm_bbox"][k]) for k in (0, 1))
    cols, _ = read_pcd(GT_1CM)
    gt = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    del cols
    tree_gt = cKDTree(gt, leafsize=32, balanced_tree=False, compact_nodes=False)
    del gt
    T_e57_world = inv(np.array(T_provided_e57))

    files = []
    # --- carte statique : protocole complet (filtre octree) → métriques
    S = np.load(f"{out_dir}/static_1cm.npy").astype(np.float64)
    known, qinfo = _query_octree(S, T_e57_world, f"{root}/map/tls")
    Sf = S[known]
    d_acc, _ = tree_gt.query(Sf, workers=-1)
    tf = cKDTree(Sf, leafsize=32, balanced_tree=False, compact_nodes=False)
    d_cmp, _ = tf.query(tree_gt.data, workers=-1)
    m = recon_metrics(d_acc, d_cmp)
    rep["T1.3_static_map"] = m | {"octree_query": qinfo, "n_known": int(known.sum())}

    # --- points retirés : distances au TLS (sans filtre octree : on veut tout ce qui a été enlevé)
    R = np.load(f"{out_dir}/removed_1cm.npy").astype(np.float64)
    d_r, _ = tree_gt.query(R, workers=-1)
    rep["removed"] = {"n_1cm": int(len(R)), "dist_to_tls": {
        "median": float(np.median(d_r)), "p90": float(np.percentile(d_r, 90)),
        "frac_lt_5cm": float((d_r < 0.05).mean()), "frac_lt_10cm": float((d_r < 0.10).mean()),
        "frac_gt_30cm": float((d_r > 0.30).mean())}}

    rng = np.random.default_rng(0)
    cmap = LinearSegmentedColormap.from_list("near_far", ["#2a78d6", "#7fb2e5", "#e8c8b0", "#c8380d"])
    # Figure 1 : points retirés en vue de dessus, colorés par distance au TLS (bleu = proche = géométrie réelle
    # déplacée — végétation au vent ; rouge = loin = transitoires absents du TLS — passants, véhicules mobiles)
    k = rng.choice(len(R), min(1_200_000, len(R)), replace=False)
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURFACE)
    o = np.argsort(d_r[k])
    sc = ax.scatter(R[k, 0][o], R[k, 1][o], c=np.clip(d_r[k][o], 0, 0.5), s=0.08,
                    cmap=cmap, vmin=0, vmax=0.5, rasterized=True, linewidths=0)
    _style(ax)
    ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label("distance au TLS (m, bornée à 0,5) : bleu = proche (géométrie déplacée), rouge = loin (transitoire)",
                 fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    ax.set_title("2.2 — points retirés par DUFOMap (world)\ntrajectoires rouges = passants ; nappes bleues = végétation/objets déplacés",
                 fontsize=10, color=INK, loc="left")
    ax.set_xlabel("x world (m)", fontsize=8, color=INK2)
    ax.set_ylabel("y world (m)", fontsize=8, color=INK2)
    f = f"{out_dir}/removed_topview.png"
    fig.savefig(f, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    files.append(f)

    # Figure 2 : histogrammes des distances retirés / gardés
    k2 = rng.choice(len(S), min(2_000_000, len(S)), replace=False)
    d_s2, _ = tree_gt.query(S[k2], workers=-1)
    rep["kept_dist_to_tls"] = {"median": float(np.median(d_s2)), "frac_gt_30cm": float((d_s2 > 0.30).mean())}
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    bins = np.linspace(0, 1.0, 201)
    for arr, lbl, c in ((d_r, "retirés (DUFOMap dyn)", "#c8380d"), (d_s2, "gardés (statiques)", "#2a78d6")):
        h, e = np.histogram(np.clip(arr, 0, 1.0), bins=bins)
        ax.plot(0.5 * (e[1:] + e[:-1]) * 100, h / h.sum() / (e[1] - e[0]) / 100, color=c, linewidth=1.6, label=lbl)
    _style(ax)
    ax.set_xlim(0, 60)
    ax.set_xlabel("distance au TLS (cm)", fontsize=8, color=INK2)
    ax.set_ylabel("densité (1/cm)", fontsize=8, color=INK2)
    ax.set_title("2.2 — distance au TLS des points retirés contre gardés", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    f = f"{out_dir}/removed_dist_hist.png"
    fig.savefig(f, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    files.append(f)

    # PLY : nuage retiré coloré par distance (contrôle visuel CloudCompare)
    from splatpipe.stage1.maptools import write_ply

    rgba = (cmap(np.clip(d_r[k] / 0.5, 0, 1))[:, :3] * 255).astype(np.uint8)
    ply = f"{out_dir}/removed_errcol.ply"
    write_ply(ply, {"x": R[k, 0].astype(np.float32), "y": R[k, 1].astype(np.float32), "z": R[k, 2].astype(np.float32),
                    "red": rgba[:, 0], "green": rgba[:, 1], "blue": rgba[:, 2], "dist_m": d_r[k].astype(np.float32)},
              comments=["SplatSim 2.2 points retires DUFOMap (world 1cm, 2M max), couleur = distance NN au TLS (0-0.5 m)"])
    rep["ply"] = os.path.basename(ply)

    rep["timing"] = {"total_s": round(time.time() - T0, 1)}
    with open(f"{out_dir}/eval_static.json", "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False, default=float)
    vol.commit()
    return {"report": json.loads(json.dumps(rep, default=float)),
            "figures": {os.path.basename(f): open(f, "rb").read() for f in files}}


@app.local_entrypoint()
def static_map_main(seq: str = SEQ):
    import json

    out = build_static_map.remote(seq)
    with open("logs/stage2/static_map.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


@app.local_entrypoint()
def eval_static_main(seq: str = SEQ):
    import json
    import os

    T = json.load(open("logs/stage1/map/tls_frame_check.json"))["T_provided_e57"]
    prev = json.load(open("logs/stage1/map_calib/stage1_map_variant_calib.json"))
    ref = prev["T1.3"]["results_computed"]["glim_final_calib"]
    out = eval_static.remote(T, {"accuracy": ref["accuracy"], "completeness": ref["completeness"],
                                 "f1@0.05": ref["f1@0.05"], "f1@0.1": ref["f1@0.1"]}, seq)
    os.makedirs("logs/stage2/dufo", exist_ok=True)
    with open("logs/stage2/static_eval.json", "w") as f:
        json.dump(out["report"], f, indent=1, ensure_ascii=False)
    for name, b in out["figures"].items():
        open(f"logs/stage2/dufo/{name}", "wb").write(b)
    r = out["report"]
    print(json.dumps({"static_map_T1.3": r["T1.3_static_map"], "removed": r["removed"],
                      "reference_full_map": r["reference_full_map"]}, indent=1))
