"""Sous-étape 2.1 — Contrôle qualité des masques sémantiques (T2.2 adapté, voir `etapes/etape2-nettoyage.md`).

Pas d'annotation manuelle ni d'étiquettes fournies pour Oxford Spires : les masques sont contrôlés contre le
LiDAR (retours réels et rayons sans retour) et par cohérence temporelle. **Usage de la trajectoire GT
uniquement pour l'évaluation** (motion-compensation), jamais pour produire les masques eux-mêmes.

La projection et la compensation de mouvement reprennent exactement la formule validée en T0.4
(`splatpipe/stage0/overlay_check.py`, lue mais non modifiée) :
    p_base(t_img) = inv(T_world_base(t_img)) · T_world_base(t_point) · T_base_lidar · p_lidar
en s'appuyant sur les mêmes primitives communes (`splatpipe.common.geometry`, `splatpipe.common.camera_model`).

Tests :
- **Précision ciel** : retours LiDAR (deux échos) à moins de 55 m projetés dans les images ; fraction de ces
  pixels étiquetés ciel (bit 0). Cible < 1 %. Répartition par caméra et par région (haut/bas de l'image) ;
  pires cas relevés pour inspection visuelle.
- **Proxy de rappel ciel** : rayons des faisceaux sans retour, élévation > +10°, placés à 200 m (modèle de
  faisceaux `lidar_model/hesai_qt64.json`) ; fraction étiquetée ciel. Pas de cible stricte (vitres, façades
  lointaines donnent aussi des non-retours).
- **Cohérence temporelle** : IoU des masques ciel et végétation entre triplets consécutifs (0,05 s), par caméra.
- Histogramme de classes par caméra et fraction de pixels « personne », lus depuis
  `reports/stage2_segment_run.json` (calculés exactement sur les 16 896 images pendant la segmentation).
- ~12 images de superposition (image + masques colorés + points LiDAR, rouges si étiquetés ciel).

Lancement :
  .venv/bin/modal run splatpipe/stage2/check_masks.py --n-sample 300

Ne touche ni `/data/raw`, ni `lidar/`, `lidar_model/`, `traj/`, `sensors.yaml`, `cameras/`, `splits.yaml`,
`map/`, `splits.yaml` (lecture seule). N'écrit que sous `reports/stage2_seg.json` et `reports/stage2_seg/`.
"""

from __future__ import annotations

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume
from splatpipe.stage2.images import segmentation_image

app = modal.App("splatsim-stage2-check")
vol = data_volume()
image = segmentation_image()

DATASET = "oxford_spires"
SEQ = "2024-03-13-observatory-quarter-01"
CAMS = ("cam0", "cam1", "cam2")
LIDAR_ID = "hesai_qt64"
LIDAR_SUBDIR = "lidar/hesai"
LIDAR_MODEL = "lidar_model/hesai_qt64.json"
OUT_SUBDIR = "reports/stage2_seg"

# Bits des masques : doivent rester identiques à `splatpipe/stage2/segment_images.py`.
BIT_SKY, BIT_PERSON, BIT_VEHICLE, BIT_VEGETATION = 0, 1, 2, 3

QA = {
    "time_half_window_s": 0.05,  # points retenus si |t_point - t_image| <= cette fenêtre (comme T0.4)
    "gt_max_gap_s": 0.2,  # interpolation GT refusée au-delà (comme T0.4)
    "max_range_precision_m": 55.0,  # cible T2.2 adaptée
    "min_range_m": 0.3,
    "recall_min_elev_deg": 10.0,
    "recall_far_range_m": 200.0,
    "margin_px": 2,
    "theta_limit_deg": 85.0,
    "sky_precision_target": 0.01,  # < 1 %
    "temporal_iou_warn_below": 0.85,  # seuil informatif choisi ici (pas de cible du plan) ; voir les notes
    "min_points_for_worst_case": 20,
}


def scene(seq: str):
    from splatpipe.common.scene import Scene

    return Scene(DATASET, seq, data_root=DATA_MOUNT)


def _load_traj(sc):
    from splatpipe.common import geometry as g

    tg, pg, qg = sc.load_traj("gt")
    tg, pg, qg, _ = g.clean_trajectory(tg, pg, qg)
    return tg, pg, qg


def _py(x):
    import numpy as np

    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return _py(x.tolist())
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if v != v else v
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float) and x != x:
        return None
    return x


# =============================================================================
# Plan : triplets échantillonnés, répartis sur la séquence
# =============================================================================


def plan_qa_frames(sc, n_sample: int) -> list[dict]:
    frames = sc.load_frames()
    n = len(frames["frame_id"])
    n_sample = min(n_sample, n)
    idx = sorted(set(round(k * (n - 1) / max(n_sample - 1, 1)) for k in range(n_sample)))
    out = []
    for i in idx:
        f = {"frame_id": int(frames["frame_id"][i]), "stamp_ns": int(frames["stamp_ns"][i]),
             "t": float(frames["t"][i])}
        nxt = None
        if i + 1 < n:
            dt = float(frames["t"][i + 1] - frames["t"][i])
            if abs(dt - 0.05) < 0.02:
                nxt = {"stamp_ns": int(frames["stamp_ns"][i + 1]), "dt": dt}
        f["next"] = nxt
        for cam in CAMS:
            f[f"{cam}_rel_path"] = str(frames[f"{cam}_rel_path"][i])
        out.append(f)
    return out


# =============================================================================
# QA par triplet (LiDAR : précision ciel, proxy de rappel ; cohérence temporelle)
# =============================================================================


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=2, memory=4096, timeout=600, retries=1)
def qa_frame(frame: dict, seq: str = SEQ, params: dict = QA) -> dict:
    import os

    import numpy as np
    from PIL import Image

    from splatpipe.common import geometry as g
    from splatpipe.common.camera_model import FisheyeCamera
    from splatpipe.common.lidar_model import StructuredSpinningLidarModel

    sc = scene(seq)
    sensors = sc.load_sensors()
    model = StructuredSpinningLidarModel.load_json(sc.path(LIDAR_MODEL))
    tg, pg, qg = _load_traj(sc)
    T_bl = np.asarray(sensors["lidars"][LIDAR_ID]["T_base_lidar"], dtype=np.float64)
    cams = {c: FisheyeCamera.from_dict(sensors["cameras"][c]) for c in CAMS}
    T_bc = {c: np.asarray(sensors["cameras"][c]["T_base_cam"], dtype=np.float64) for c in CAMS}
    tlim = np.radians(params["theta_limit_deg"])

    t_img = frame["t"]
    half = params["time_half_window_s"]
    stamp_ns = frame["stamp_ns"]

    masks = {}
    for c in CAMS:
        mpath = sc.path("masks", c, f"{stamp_ns}.png")
        if not os.path.exists(mpath):
            return {"frame_id": frame["frame_id"], "error": f"masque manquant : {mpath}"}
        masks[c] = np.asarray(Image.open(mpath))

    lidar_dir = sc.path(LIDAR_SUBDIR)
    names = sorted(f for f in os.listdir(lidar_dir) if f.endswith(".npz"))
    cand = [fn for fn in names if
            int(fn[:-4]) / 1e9 <= t_img + half + 0.02 and int(fn[:-4]) / 1e9 + 0.12 >= t_img - half - 0.02]
    if not cand:
        return {"frame_id": frame["frame_id"], "t": t_img, "error": "aucun tour LiDAR dans la fenêtre"}

    scans = []
    for fn in cand:
        with np.load(f"{lidar_dir}/{fn}") as z:
            scans.append({"file": fn, "pt_xyz": z["pt_xyz"], "pt_ret": z["pt_ret"], "pt_t": z["pt_t"],
                         "t_start": float(z["t_start"]), "t_end": float(z["t_end"]), "range0": z["range"][0]})

    # ---- points réels (deux échos) dans la fenêtre temporelle -> précision ciel
    xs, ts, rs = [], [], []
    for s in scans:
        rng = np.linalg.norm(s["pt_xyz"], axis=1)
        m = ((s["pt_ret"] >= 0) & (rng > params["min_range_m"]) & (rng < params["max_range_precision_m"])
             & (np.abs(s["pt_t"] - t_img) <= half))
        if m.any():
            xs.append(s["pt_xyz"][m]); ts.append(s["pt_t"][m]); rs.append(rng[m])
    p_lidar = np.concatenate(xs) if xs else np.zeros((0, 3))
    t_pt = np.concatenate(ts) if ts else np.zeros((0,))

    p0, q0, v0 = g.interpolate_poses(tg, pg, qg, np.array([t_img]), max_gap_s=params["gt_max_gap_s"])
    if not v0[0]:
        return {"frame_id": frame["frame_id"], "t": t_img, "error": "pose GT invalide à t_img"}
    R0, t0 = g.quat_to_rot(q0[0]), p0[0]

    def compensate(p_lidar_pts, t_pts):
        p_base_raw = g.transform_points(T_bl, p_lidar_pts)
        pw, qw, vw = g.interpolate_poses(tg, pg, qg, t_pts, max_gap_s=params["gt_max_gap_s"])
        Rw = g.quat_to_rot(qw)
        p_world = np.einsum("nij,nj->ni", Rw, p_base_raw) + pw
        p_base_img = (p_world - t0) @ R0
        return p_base_img, vw

    p_base_img, valid_pt = compensate(p_lidar, t_pt)
    p_base_img, rng_pt = p_base_img[valid_pt], np.concatenate(rs)[valid_pt] if rs else np.zeros((0,))

    precision = {}
    for c in CAMS:
        Tbc, cam = T_bc[c], cams[c]
        pc = (p_base_img - Tbc[:3, 3]) @ Tbc[:3, :3]
        uv, ok, _ = cam.project(pc, margin=params["margin_px"], theta_limit=tlim)
        H, W = cam.height, cam.width
        n_pts = int(ok.sum())
        if n_pts:
            ui = np.clip(np.round(uv[ok, 0]).astype(int), 0, W - 1)
            vi = np.clip(np.round(uv[ok, 1]).astype(int), 0, H - 1)
            bits = masks[c][vi, ui]
            is_sky = (bits & (1 << BIT_SKY)) > 0
            top = vi < H // 2
            precision[c] = {"n_points": n_pts, "n_sky": int(is_sky.sum()),
                            "n_top": int(top.sum()), "n_sky_top": int((is_sky & top).sum()),
                            "n_bottom": int((~top).sum()), "n_sky_bottom": int((is_sky & ~top).sum())}
        else:
            precision[c] = {"n_points": 0, "n_sky": 0, "n_top": 0, "n_sky_top": 0, "n_bottom": 0, "n_sky_bottom": 0}

    # ---- proxy de rappel : rayons sans retour, élévation > seuil, tour le plus proche de t_img
    s0 = min(scans, key=lambda s: abs(0.5 * (s["t_start"] + s["t_end"]) - t_img))
    el_thr = np.radians(params["recall_min_elev_deg"])
    rows_above = np.nonzero(model.row_elevations_rad > el_thr)[0]
    sub = s0["range0"][rows_above, :]
    rr, cc = np.nonzero(sub == 0)
    recall = {c: {"n_valid": 0, "n_sky": 0} for c in CAMS}
    n_empty_cells = int(len(rr))
    if n_empty_cells:
        rows_sel, cols_sel = rows_above[rr], cc
        ray = model.element_rays(rows_sel, cols_sel)
        p_far = ray * params["recall_far_range_m"]
        t_elem = model.element_times(cols_sel, s0["t_start"], s0["t_end"])
        p_base_img2, valid2 = compensate(p_far, t_elem)
        p_base_img2 = p_base_img2[valid2]
        for c in CAMS:
            Tbc, cam = T_bc[c], cams[c]
            pc = (p_base_img2 - Tbc[:3, 3]) @ Tbc[:3, :3]
            uv, ok, _ = cam.project(pc, margin=params["margin_px"], theta_limit=tlim)
            n_valid = int(ok.sum())
            if n_valid:
                H, W = cam.height, cam.width
                ui = np.clip(np.round(uv[ok, 0]).astype(int), 0, W - 1)
                vi = np.clip(np.round(uv[ok, 1]).astype(int), 0, H - 1)
                bits = masks[c][vi, ui]
                recall[c] = {"n_valid": n_valid, "n_sky": int(((bits & (1 << BIT_SKY)) > 0).sum())}

    # ---- cohérence temporelle (triplet suivant, 0,05 s)
    temporal = {}
    nxt = frame.get("next")
    if nxt:
        for c in CAMS:
            p1 = sc.path("masks", c, f"{nxt['stamp_ns']}.png")
            if os.path.exists(p1):
                m1 = np.asarray(Image.open(p1))
                a_sky, b_sky = (masks[c] & (1 << BIT_SKY)) > 0, (m1 & (1 << BIT_SKY)) > 0
                a_veg, b_veg = (masks[c] & (1 << BIT_VEGETATION)) > 0, (m1 & (1 << BIT_VEGETATION)) > 0

                def iou(a, b):
                    u = int((a | b).sum())
                    return float((a & b).sum() / u) if u > 0 else None

                temporal[c] = {"dt_s": nxt["dt"], "sky_iou": iou(a_sky, b_sky), "veg_iou": iou(a_veg, b_veg)}

    return {"frame_id": frame["frame_id"], "t": t_img, "precision": precision, "recall_proxy": recall,
            "n_empty_cells_above_horizon": n_empty_cells, "temporal": temporal, "n_scans_used": len(scans)}


# =============================================================================
# Superpositions visuelles
# =============================================================================

MASK_COLORS_BGR = {BIT_SKY: (255, 210, 0), BIT_PERSON: (0, 0, 255), BIT_VEHICLE: (0, 255, 255),
                   BIT_VEGETATION: (0, 200, 0)}


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=2, memory=4096, timeout=600, retries=1)
def render_overlay(frame: dict, seq: str = SEQ, params: dict = QA) -> dict:
    """`frame` doit porter la clé `qa_label` (utilisée dans le nom de fichier et le texte incrusté)."""
    import os

    label = frame.get("qa_label", "")

    import cv2
    import numpy as np
    from PIL import Image

    from splatpipe.common import geometry as g
    from splatpipe.common.camera_model import FisheyeCamera

    sc = scene(seq)
    sensors = sc.load_sensors()
    tg, pg, qg = _load_traj(sc)
    T_bl = np.asarray(sensors["lidars"][LIDAR_ID]["T_base_lidar"], dtype=np.float64)
    cams = {c: FisheyeCamera.from_dict(sensors["cameras"][c]) for c in CAMS}
    T_bc = {c: np.asarray(sensors["cameras"][c]["T_base_cam"], dtype=np.float64) for c in CAMS}
    tlim = np.radians(params["theta_limit_deg"])
    t_img, stamp_ns, half = frame["t"], frame["stamp_ns"], params["time_half_window_s"]

    lidar_dir = sc.path(LIDAR_SUBDIR)
    names = sorted(f for f in os.listdir(lidar_dir) if f.endswith(".npz"))
    cand = [fn for fn in names if
            int(fn[:-4]) / 1e9 <= t_img + half + 0.02 and int(fn[:-4]) / 1e9 + 0.12 >= t_img - half - 0.02]
    xs, ts = [], []
    for fn in cand:
        with np.load(f"{lidar_dir}/{fn}") as z:
            rng = np.linalg.norm(z["pt_xyz"], axis=1)
            m = ((z["pt_ret"] >= 0) & (rng > params["min_range_m"]) & (rng < params["max_range_precision_m"])
                 & (np.abs(z["pt_t"] - t_img) <= half))
            if m.any():
                xs.append(z["pt_xyz"][m]); ts.append(z["pt_t"][m])
    p_lidar = np.concatenate(xs) if xs else np.zeros((0, 3))
    t_pt = np.concatenate(ts) if ts else np.zeros((0,))
    p0, q0, v0 = g.interpolate_poses(tg, pg, qg, np.array([t_img]), max_gap_s=params["gt_max_gap_s"])
    R0, t0 = g.quat_to_rot(q0[0]), p0[0]
    p_base_raw = g.transform_points(T_bl, p_lidar)
    pw, qw, vw = g.interpolate_poses(tg, pg, qg, t_pt, max_gap_s=params["gt_max_gap_s"])
    Rw = g.quat_to_rot(qw)
    p_world = np.einsum("nij,nj->ni", Rw, p_base_raw) + pw
    p_base_img = ((p_world - t0) @ R0)[vw]

    out_dir = sc.path(OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for c in CAMS:
        img = cv2.imread(sc.abs_path(frame[f"{c}_rel_path"]))
        mpath = sc.path("masks", c, f"{stamp_ns}.png")
        mask = np.asarray(Image.open(mpath))
        overlay = img.copy()
        for bit, color in MASK_COLORS_BGR.items():
            sel = (mask & (1 << bit)) > 0
            overlay[sel] = color
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

        Tbc, cam = T_bc[c], cams[c]
        pc = (p_base_img - Tbc[:3, 3]) @ Tbc[:3, :3]
        uv, ok, _ = cam.project(pc, margin=0, theta_limit=tlim)
        if ok.any():
            H, W = cam.height, cam.width
            ui = np.clip(np.round(uv[ok, 0]).astype(int), 0, W - 1)
            vi = np.clip(np.round(uv[ok, 1]).astype(int), 0, H - 1)
            is_sky = (mask[vi, ui] & (1 << BIT_SKY)) > 0
            for u, v, sky in zip(ui, vi, is_sky):
                cv2.circle(img, (int(u), int(v)), 3, (0, 0, 255) if sky else (60, 220, 60), -1, cv2.LINE_AA)
        txt = f"{label} {c} frame {frame['frame_id']} t={t_img:.2f}s  n_lidar={int(ok.sum())}"
        cv2.putText(img, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        name = f"f{frame['frame_id']:05d}_{label}_{c}.jpg"
        ok_enc, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open(f"{out_dir}/{name}", "wb") as f:
            f.write(buf.tobytes())
        files[name] = buf.tobytes()
    vol.commit()
    return files


# =============================================================================
# Agrégation et rapport final
# =============================================================================


def _percentiles(vals: list[float]) -> dict:
    import numpy as np

    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {"n": int(len(a)), "mean": float(a.mean()), "min": float(a.min()), "p25": float(np.percentile(a, 25)),
            "median": float(np.percentile(a, 50)), "p75": float(np.percentile(a, 75)), "max": float(a.max())}


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=4, memory=8192, timeout=3600)
def finalize(seq: str = SEQ, n_sample: int = 300, n_overlay_frames: int = 3) -> dict:
    import datetime
    import time

    import numpy as np

    t0 = time.time()
    sc = scene(seq)
    plan = plan_qa_frames(sc, n_sample)
    results = list(qa_frame.map(plan, kwargs={"seq": seq, "params": QA}, return_exceptions=True))
    good = [r for r in results if isinstance(r, dict) and "error" not in r]
    bad = [r if isinstance(r, dict) else repr(r)[:300] for r in results
           if not (isinstance(r, dict) and "error" not in r)]

    # ---- précision ciel : agrégée globalement, par caméra, par région
    prec_glob = {"n_points": 0, "n_sky": 0, "n_top": 0, "n_sky_top": 0, "n_bottom": 0, "n_sky_bottom": 0}
    prec_by_cam = {c: dict(prec_glob) for c in CAMS}
    worst = []  # (frac, frame_id, cam, n_points)
    for r in good:
        for c in CAMS:
            p = r["precision"][c]
            for k in prec_glob:
                prec_by_cam[c][k] += p[k]
                prec_glob[k] += p[k]
            if p["n_points"] >= QA["min_points_for_worst_case"]:
                worst.append((p["n_sky"] / p["n_points"], r["frame_id"], c, p["n_points"]))
    worst.sort(reverse=True)

    def frac(d, num_key, den_key):
        return (d[num_key] / d[den_key]) if d[den_key] else None

    precision_report = {
        "overall": {"frac_sky": frac(prec_glob, "n_sky", "n_points"), **prec_glob,
                    "frac_sky_top": frac(prec_glob, "n_sky_top", "n_top"),
                    "frac_sky_bottom": frac(prec_glob, "n_sky_bottom", "n_bottom")},
        "by_camera": {c: {"frac_sky": frac(prec_by_cam[c], "n_sky", "n_points"), **prec_by_cam[c],
                          "frac_sky_top": frac(prec_by_cam[c], "n_sky_top", "n_top"),
                          "frac_sky_bottom": frac(prec_by_cam[c], "n_sky_bottom", "n_bottom")} for c in CAMS},
        "target": QA["sky_precision_target"],
        "worst_cases": [{"frac_sky": w[0], "frame_id": w[1], "cam": w[2], "n_points": w[3]} for w in worst[:10]],
    }
    precision_report["pass"] = bool(precision_report["overall"]["frac_sky"] is not None
                                    and precision_report["overall"]["frac_sky"] < QA["sky_precision_target"])
    precision_report["pass_by_camera"] = {
        c: bool(precision_report["by_camera"][c]["frac_sky"] is not None
               and precision_report["by_camera"][c]["frac_sky"] < QA["sky_precision_target"]) for c in CAMS}

    # ---- proxy de rappel
    rec_glob = {"n_valid": 0, "n_sky": 0}
    rec_by_cam = {c: dict(rec_glob) for c in CAMS}
    for r in good:
        for c in CAMS:
            rr = r["recall_proxy"][c]
            for k in rec_glob:
                rec_by_cam[c][k] += rr[k]
                rec_glob[k] += rr[k]
    recall_report = {
        "overall": {"frac_sky": frac(rec_glob, "n_sky", "n_valid"), **rec_glob},
        "by_camera": {c: {"frac_sky": frac(rec_by_cam[c], "n_sky", "n_valid"), **rec_by_cam[c]} for c in CAMS},
        "note": "pas de cible stricte : vitres et façades lointaines donnent aussi des non-retours",
    }

    # ---- cohérence temporelle
    temporal_by_cam = {c: {"sky": [], "veg": []} for c in CAMS}
    for r in good:
        for c in CAMS:
            t = r.get("temporal", {}).get(c)
            if t and t["sky_iou"] is not None:
                temporal_by_cam[c]["sky"].append(t["sky_iou"])
            if t and t["veg_iou"] is not None:
                temporal_by_cam[c]["veg"].append(t["veg_iou"])
    temporal_report = {c: {"sky_iou": _percentiles(temporal_by_cam[c]["sky"]),
                           "veg_iou": _percentiles(temporal_by_cam[c]["veg"])} for c in CAMS}
    temporal_report["warn_threshold_median"] = QA["temporal_iou_warn_below"]
    temporal_report["status_by_camera"] = {
        c: ("PASS" if (temporal_report[c]["sky_iou"].get("median") or 0) >= QA["temporal_iou_warn_below"]
           else "WARN") for c in CAMS}

    # ---- histogramme de classes (depuis le rapport de la segmentation, calculé sur les 16 896 images)
    try:
        seg_run = sc.load_report("stage2_segment_run")
        classes_meta = None
        import json as _json
        import os as _os
        cpath = sc.path("semantics", "classes.json")
        if _os.path.exists(cpath):
            with open(cpath) as f:
                classes_meta = _json.load(f)
        hist = {c: np.asarray(v, dtype=np.int64) for c, v in seg_run["class_hist_by_cam"].items()}
        class_stats = {}
        for c in CAMS:
            h = hist[c]
            total = int(h.sum())
            bits = np.zeros(len(h), dtype=np.uint8)
            if classes_meta:
                for idx, d in classes_meta["bit_detail"].items():
                    b = 0
                    if "sky" in d["matched_bits"]:
                        b |= 1 << BIT_SKY
                    if "person" in d["matched_bits"]:
                        b |= 1 << BIT_PERSON
                    if "vehicle" in d["matched_bits"]:
                        b |= 1 << BIT_VEHICLE
                    if "vegetation" in d["matched_bits"]:
                        b |= 1 << BIT_VEGETATION
                    bits[int(idx)] = b
            class_stats[c] = {
                "n_pixels_total": total,
                "frac_sky": float((h[(bits & (1 << BIT_SKY)) > 0]).sum() / total) if total else None,
                "frac_person": float((h[(bits & (1 << BIT_PERSON)) > 0]).sum() / total) if total else None,
                "frac_vehicle": float((h[(bits & (1 << BIT_VEHICLE)) > 0]).sum() / total) if total else None,
                "frac_vegetation": float((h[(bits & (1 << BIT_VEGETATION)) > 0]).sum() / total) if total else None,
                "top10_classes_by_pixels": [
                    {"class_id": int(i), "label": classes_meta["id2label"].get(str(i), str(i)) if classes_meta else str(i),
                     "frac": float(h[i] / total)}
                    for i in np.argsort(-h)[:10] if total],
            }
        histogram_report = {"per_camera": class_stats, "source": "reports/stage2_segment_run.json"}
    except Exception as e:  # noqa: BLE001
        histogram_report = {"error": repr(e)[:500]}

    # ---- superpositions : quelques triplets répartis + le pire cas de précision ciel
    pick_i = np.linspace(0, len(plan) - 1, n_overlay_frames).round().astype(int)
    ov_frames = [{**plan[i], "qa_label": f"sample{k}"} for k, i in enumerate(pick_i)]
    if worst:
        wf = next((f for f in plan if f["frame_id"] == worst[0][1]), None)
        if wf is not None:
            ov_frames.append({**wf, "qa_label": "worst_sky_precision"})
    overlay_files = {}
    for res in render_overlay.map(ov_frames, kwargs={"seq": seq}):
        overlay_files.update(res)

    report = {
        "step": "stage2_seg", "seq": seq,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "code": "splatpipe/stage2/check_masks.py (projection/compensation reprises de "
                "splatpipe/stage0/overlay_check.py, T0.4)",
        "method": {
            "motion_compensation": "p_base(t_img) = inv(T_world_base(t_img)) · T_world_base(t_point) · "
                                    "T_base_lidar · p_lidar ; GT utilisée uniquement pour l'évaluation",
            "time_window_s": QA["time_half_window_s"], "max_range_precision_m": QA["max_range_precision_m"],
            "recall_min_elev_deg": QA["recall_min_elev_deg"], "recall_far_range_m": QA["recall_far_range_m"],
        },
        "n_sample_frames": len(plan), "n_frame_errors": len(bad), "frame_errors_sample": bad[:10],
        "sky_precision_vs_lidar": precision_report,
        "sky_recall_proxy": recall_report,
        "temporal_consistency": temporal_report,
        "class_histogram": histogram_report,
        "overlays": sorted(f"{OUT_SUBDIR}/{n}" for n in overlay_files),
        "runtime_s": round(time.time() - t0, 1),
    }
    report = _py(report)
    sc.write_report("stage2_seg", report)
    vol.commit()
    return {"report": report, "files": overlay_files}


@app.local_entrypoint()
def main(seq: str = SEQ, n_sample: int = 300, n_overlay_frames: int = 3):
    import json
    import pathlib

    res = finalize.remote(seq, n_sample, n_overlay_frames)
    rep = res["report"]
    pathlib.Path("logs").mkdir(exist_ok=True)
    pathlib.Path("logs/stage2/seg.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    d = pathlib.Path("logs/stage2/seg")
    d.mkdir(exist_ok=True)
    for name, b in res["files"].items():
        (d / name).write_bytes(b)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("class_histogram",)}, indent=2, ensure_ascii=False))
    print(f"{len(res['files'])} images -> {d} ; rapport logs/stage2/seg.json ; {rep['runtime_s']} s")
