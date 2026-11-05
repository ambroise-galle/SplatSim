"""Sous-étape 2.2 (verdict formel T2.1) — rappel des points dynamiques et retrait de la végétation.

Protocole (adaptation D1 : Oxford Spires n'a ni annotation manuelle ni étiquettes fournies — voir
`etapes/etape2-nettoyage.md`) : les masques image de la 2.1 (bits : 0 ciel, 1 personne, 2 véhicules,
3 végétation) sont **projetés dans les tours corrigés** avec les poses de la 3.1 (`poses/cameras.json`) :

  point (xyz, repère map, temps propre t_pt) → caméra c → FisheyeCamera.project(T_map_cam⁻¹ · p) → bit lu.

Filtres de protocole (fixés avant mesure) :
- points à ±30 ms du milieu d'exposition de l'image (mouvement du porteur ≈ 4 cm) ;
- masques « personne » et « végétation » **érosion 2 px** (pixels mixtes de silhouette) ;
- marge image 2 px ; portée ≤ 55 m.
Chaque point projeté reçoit le bit lu et l'étiquette DUFOMap du sidecar `lidar/dyn_labels_dufo`.

Critères T2.1 (plan) : **rappel dynamique ≥ 90 %** sur les points « personne » ; **statique retiré
≤ 1 %** sur les points « végétation ». Bit « véhicules » (vélos garés, statiques en réalité) : information.
Rend aussi quelques images de contrôle : points « personne » projetés, verts = retirés par DUFOMap (bons),
rouges = retenus (ratés).

Lancement :
  .venv/bin/modal run splatpipe/stage2/t21_check.py::t21_main --stride 5
"""

from __future__ import annotations

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume
from splatpipe.stage1.images import map_image

app = modal.App("splatsim-stage2-t21")
vol = data_volume()

SEQ = "2024-03-13-observatory-quarter-01"
TIME_WINDOW_S = 0.03
ERODE_PX = 2
MARGIN_PX = 2
MAX_RANGE_M = 55.0
CNT = ("n_proj", "n_person", "person_dyn", "n_veg", "veg_dyn", "n_veh", "veh_dyn", "n_other", "other_dyn")


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=2, memory=32 * 1024, timeout=3600)
def check_indices(indices: list[int], seq: str = SEQ, n_render: int = 1) -> dict:
    import json
    import os
    import time

    import numpy as np
    from PIL import Image
    from scipy.ndimage import binary_erosion

    from splatpipe.common.camera_model import FisheyeCamera
    from splatpipe.common.scene import Scene

    t0 = time.time()
    sc = Scene("oxford_spires", seq, data_root=DATA_MOUNT)
    cams = sc.camera_ids()
    with open(sc.path("poses", "cameras.json")) as f:
        poses = json.load(f)["cameras"]
    idx = {c: sc.load_camera_index(c) for c in cams}
    frames = sc.load_frames()
    cam_models = {c: FisheyeCamera.from_dict(sc.camera(c)) for c in cams}

    dd = sc.path("lidar", "hesai_deskewed")
    dl = sc.path("lidar", "dyn_labels_dufo")
    scan_stamps = np.array(sorted(int(s[:-4]) for s in os.listdir(dd) if s.endswith(".npz")))
    scan_mid = scan_stamps / 1e9 + 0.05
    renders = []
    tot = {k: 0 for k in CNT}
    per_frame = []
    n_scans_used: set[int] = set()
    for i in indices:
        row = {k: 0 for k in CNT} | {"frame_id": int(frames["frame_id"][i]), "cams": 0}
        t_trip = float(frames["t"][i])
        j = int(np.argmin(np.abs(scan_mid - t_trip)))
        if abs(scan_mid[j] - t_trip) > 0.06:
            per_frame.append(row)
            continue
        st = int(scan_stamps[j])
        with np.load(f"{dd}/{st}.npz") as z:
            xyz, t_rel = z["xyz"], z["t_rel"]
            t_start = float(z["t_start"])
        with np.load(f"{dl}/{st}.npz") as z:
            dyn = z["dyn"].astype(bool)
        t_pt = t_start + t_rel.astype(np.float64)
        n_scans_used.add(st)
        for c in cams:
            key = str(int(idx[c]["stamp_ns"][frames[f"{c}_idx"][i]]))
            if key not in poses[c]:
                continue
            mpath = sc.path("masks", c, f"{key}.png")
            if not os.path.exists(mpath):
                continue
            t_cam = poses[c][key]["t_start"]
            T_mc = np.array(poses[c][key]["T_map_cam_start"])
            mask = np.asarray(Image.open(mpath))
            # encodage 2.1 : champ de bits (1<<bit) — ciel 1, personne 2, véhicule 4, végétation 8
            person = binary_erosion((mask & (1 << 1)) > 0, iterations=ERODE_PX)
            veg = binary_erosion((mask & (1 << 3)) > 0, iterations=ERODE_PX)
            veh = (mask & (1 << 2)) > 0
            keep = (np.abs(t_pt - t_cam) <= TIME_WINDOW_S) & \
                   (np.linalg.norm(xyz.astype(np.float64) - T_mc[:3, 3], axis=1) <= MAX_RANGE_M)
            if not keep.any():
                continue
            pts = xyz[keep].astype(np.float64)
            uv, valid, _ = cam_models[c].project((pts - T_mc[:3, 3]) @ T_mc[:3, :3], margin=MARGIN_PX)
            ui = uv[valid].round().astype(int)
            H, W = mask.shape
            inb = (ui[:, 0] >= 0) & (ui[:, 0] < W) & (ui[:, 1] >= 0) & (ui[:, 1] < H)
            if not inb.any():
                continue
            b = ui[inb]
            bit = np.zeros(len(b), np.uint8)
            bit[veg[b[:, 1], b[:, 0]]] = 3
            bit[veh[b[:, 1], b[:, 0]]] = 2
            bit[person[b[:, 1], b[:, 0]]] = 1
            d = dyn[keep][valid][inb]
            row["n_proj"] += len(bit)
            for B, nk, nd in ((1, "n_person", "person_dyn"), (3, "n_veg", "veg_dyn"), (2, "n_veh", "veh_dyn")):
                m = bit == B
                row[nk] += int(m.sum())
                row[nd] += int(d[m].sum())
            row["n_other"] += int((bit == 0).sum())
            row["other_dyn"] += int(d[bit == 0].sum())
            row["cams"] += 1
            if (bit == 1).sum() >= 50 and len(renders) < n_render:
                renders.append((c, key, int(frames["stamp_ns"][i]), pts[valid][inb][bit == 1],
                                d[bit == 1], T_mc, cam_models[c]))
        per_frame.append(row)
        for k in CNT:
            tot[k] += row[k]

    os.makedirs(f"{sc.root}/reports/stage2_t21", exist_ok=True)
    rendered = []
    for c, key, stamp_triplet, p_pts, p_dyn, T_mc, cam in renders:
        ipos = np.flatnonzero(idx[c]["stamp_ns"] == int(key))
        img = Image.open(sc.abs_path(idx[c]["rel_path"][int(ipos[0])]))
        a = np.asarray(img).copy()
        uv, valid, _ = cam.project((p_pts.astype(np.float64) - T_mc[:3, 3]) @ T_mc[:3, :3], margin=3)
        ui = uv[valid].round().astype(int)
        H, W = a.shape[:2]
        inb = (ui[:, 0] >= 1) & (ui[:, 0] < W - 1) & (ui[:, 1] >= 1) & (ui[:, 1] < H - 1)
        col = np.where(p_dyn[valid][inb, None], np.array((40, 200, 80), np.uint8), np.array((255, 40, 40), np.uint8))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                a[ui[inb, 1] + dy, ui[inb, 0] + dx] = col
        fp = f"{sc.root}/reports/stage2_t21/person_{c}_{stamp_triplet}.png"
        Image.fromarray(a).save(fp)
        rendered.append({"cam": c, "stamp_ns": key, "n_person_pts": int(len(p_pts)),
                         "frac_dyn": float(p_dyn.mean()), "png": os.path.basename(fp)})
    vol.commit()
    return {"counts": tot, "n_frames": len(per_frame), "n_scans_used": len(n_scans_used),
            "per_frame": per_frame, "rendered": rendered, "chunk_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def t21_main(stride: int = 5, n_chunks: int = 6, seq: str = SEQ):
    import json
    import os

    n = _n_frames.remote(seq)
    idx_all = list(range(0, n, stride))
    groups = [idx_all[k::n_chunks] for k in range(n_chunks) if idx_all[k::n_chunks]]
    outs = list(check_indices.map(groups, kwargs={"seq": seq}))
    agg = {k: sum(o["counts"][k] for o in outs) for k in CNT}
    pf = [r for o in outs for r in o["per_frame"]]
    rendered = [r for o in outs for r in o["rendered"]]
    summary = {
        "protocol": {"stride": stride, "n_frames": len(idx_all), "n_chunks": len(groups),
                     "time_window_s": TIME_WINDOW_S, "erode_px": ERODE_PX, "margin_px": MARGIN_PX,
                     "max_range_m": MAX_RANGE_M, "source_masks": "masks/<cam>/<stamp>.png (2.1)",
                     "dyn": "lidar/dyn_labels_dufo (2.2)", "poses": "poses/cameras.json (3.1)"},
        "counts": agg,
        "person_recall": round(agg["person_dyn"] / max(agg["n_person"], 1), 4),
        "veg_removed": round(agg["veg_dyn"] / max(agg["n_veg"], 1), 4),
        "veh_removed": round(agg["veh_dyn"] / max(agg["n_veh"], 1), 4),
        "other_removed": round(agg["other_dyn"] / max(agg["n_other"], 1), 4),
        "rendered": rendered,
    }
    summary["verdict"] = {
        "criterion": "T2.1 : rappel dynamique (personne) >= 0.90 ; statique retire (vegetation) <= 0.01",
        "recall_ok": summary["person_recall"] >= 0.90,
        "veg_static_kept_ok": summary["veg_removed"] <= 0.01,
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/stage2/t21.json", "w") as f:
        json.dump(summary | {"per_frame": pf}, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_frame"}, indent=1, ensure_ascii=False))


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=1, memory=2048, timeout=600)
def _n_frames(seq: str = SEQ) -> int:
    from splatpipe.common.scene import Scene

    return len(Scene("oxford_spires", seq, data_root=DATA_MOUNT).load_frames()["frame_id"])
