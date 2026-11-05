"""Sous-étape 3.1 — poses caméra par interpolation de la trajectoire finale (`traj/spline.npz`).

Méthode (plan, étape 3.1) : obturateur global sur D1 → début = fin d'exposition ; la pose caméra est

    T_map_cam(t) = T_map_base(t) · T_base_cam          avec t = t_exposition(cam) + time_offset(cam)

`t_exposition` = milieu d'exposition de la colonne `t` de `cameras/<cam>/index.csv`, prise aux indices
`<cam>_idx` de `cameras/frames.csv` (les 5 632 triplets synchronisés). Le repère « monde » des étapes 2–4 est
le repère carte GLIM (`map`) — le passage à `world` ne sert qu'à l'évaluation (`traj/final_to_world.yaml`).

Sortie `poses/cameras.json` : pour chaque caméra et chaque image, `t_start`, `t_end` et les deux poses 4×4
(identiques ici, obturateur global). `validate_main` ajoute trois images de contrôle : points LiDAR corrigés
projetés sur l'image avec ces poses (`<cam>_<stamp>_overlay.png`) — l'œil juge l'alignement avant COLMAP (T3.1).

Lancement :
  .venv/bin/modal run splatpipe/stage3/poses_from_traj.py::poses_main
  .venv/bin/modal run splatpipe/stage3/poses_from_traj.py::validate_main --n-frames 3
"""

from __future__ import annotations

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume
from splatpipe.stage1.images import map_image

app = modal.App("splatsim-stage3-poses")
vol = data_volume()

SEQ = "2024-03-13-observatory-quarter-01"


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=2, memory=32 * 1024, timeout=1800)
def write_poses(seq: str = SEQ) -> dict:
    import json
    import os
    import time

    import numpy as np

    from splatpipe.common.trajectory import Trajectory
    from splatpipe.common.scene import Scene

    t0 = time.time()
    sc = Scene("oxford_spires", seq, data_root=DATA_MOUNT)
    traj = Trajectory.load(sc.path("traj", "spline.npz"))
    cams = sc.camera_ids()
    idx = {c: sc.load_camera_index(c) for c in cams}
    frames = sc.load_frames()
    T_bc = {c: sc.T_base(c) for c in cams}
    off = {c: sc.time_offset(c) for c in cams}

    out = {"description": "T_map_cam(t) = T_map_base(t)·T_base_cam, spline.npz (GLIM calib, glimimu, σ=3 s) ; "
                          "repère map (carte GLIM) ; obturateur global : t_start = t_end = milieu d'exposition + offset",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "spline": {"t0": traj.t_start, "t1": traj.t_end}, "global_shutter": True, "cameras": {}}
    stats = {"n_frames": int(len(frames["frame_id"])), "cams": {}}
    for c in cams:
        sel = frames[f"{c}_idx"]
        t_expo = idx[c]["t"][sel].astype(np.float64) + off[c]
        inside = (t_expo >= traj.t_start) & (t_expo <= traj.t_end)
        Ts = traj.matrices(list(t_expo))[0]
        cams_out = {}
        for k, i in enumerate(sel):
            if not inside[k]:
                continue
            stamp = int(idx[c]["stamp_ns"][i])
            T = (Ts[k] @ T_bc[c]).tolist()
            cams_out[str(stamp)] = {"t_start": float(t_expo[k]), "t_end": float(t_expo[k]),
                                    "T_map_cam_start": T, "T_map_cam_end": T}
        out["cameras"][c] = cams_out
        stats["cams"][c] = {"n_written": len(cams_out), "n_frames": int(len(sel)),
                            "n_outside_spline": int((~inside).sum()),
                            "t_min": float(t_expo.min()), "t_max": float(t_expo.max())}
    os.makedirs(sc.path("poses"), exist_ok=True)
    p = sc.path("poses", "cameras.json")
    with open(p, "w") as f:
        json.dump(out, f)
    vol.commit()
    stats["bytes"] = os.path.getsize(p)
    stats["write_s"] = round(time.time() - t0, 1)
    return stats


@app.local_entrypoint()
def poses_main(seq: str = SEQ):
    import json

    print(json.dumps(write_poses.remote(seq), indent=1))


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=2, memory=32 * 1024, timeout=1800)
def validate_frame(frame_id: int, seq: str = SEQ) -> dict:
    """Superposition LiDAR→image avec les poses de `poses/cameras.json` : contrôle visuel de la 3.1."""
    import json
    import os

    import numpy as np
    from PIL import Image

    from splatpipe.common.camera_model import FisheyeCamera
    from splatpipe.common.scene import Scene

    sc = Scene("oxford_spires", seq, data_root=DATA_MOUNT)
    with open(sc.path("poses", "cameras.json")) as f:
        poses = json.load(f)["cameras"]
    frames = sc.load_frames()
    j = int(np.flatnonzero(frames["frame_id"] == frame_id)[0])
    stamp = int(frames["stamp_ns"][j])
    out_dir = f"{sc.root}/reports/stage3_poses"
    os.makedirs(out_dir, exist_ok=True)
    res = {"frame_id": frame_id, "stamp_ns": stamp}
    for c in sc.camera_ids():
        key = str(stamp)
        if key not in poses[c]:
            # le triplet est aligné sur cam0 : chercher l'image la plus proche de cette caméra
            cand = sorted(poses[c], key=lambda s: abs(int(s) - stamp))
            key = cand[0]
            res[f"{c}_dt_ms"] = float((int(key) - stamp) / 1e6)
        T_mc = np.array(poses[c][key]["T_map_cam_start"])
        cam = FisheyeCamera.from_dict(sc.camera(c))
        img = Image.open(sc.abs_path(sc.load_camera_index(c)["rel_path"][
            int(np.flatnonzero(sc.load_camera_index(c)["stamp_ns"] == int(key))[0])]))
        # tour corrigé le plus proche de l'image
        dd = f"{sc.path('lidar', 'hesai_deskewed')}"
        names = sorted(f for f in os.listdir(dd) if f.endswith(".npz"))
        fn = min(names, key=lambda s: abs(int(s[:-4]) / 1e9 - float(poses[c][key]["t_start"])))
        with np.load(f"{dd}/{fn}") as z:
            xyz, t_rel, t_start = z["xyz"], z["t_rel"], float(z["t_start"])
        t_mid = t_start + float(np.median(t_rel))
        res[f"{c}_scan_dt_ms"] = float((t_mid - float(poses[c][key]["t_start"])) * 1000)
        p_cam = (xyz.astype(np.float64) - T_mc[:3, 3]) @ T_mc[:3, :3]
        uv, valid, _ = cam.project(p_cam, margin=2)
        inside = (np.linalg.norm(xyz.astype(np.float64) - T_mc[:3, 3], axis=1) < 55) & valid
        a = np.asarray(img).copy()
        uvv = uv[inside].round().astype(int)
        ok = (uvv[:, 0] >= 1) & (uvv[:, 0] < a.shape[1] - 1) & (uvv[:, 1] >= 1) & (uvv[:, 1] < a.shape[0] - 1)
        far = np.linalg.norm(xyz[inside], axis=1) > 30
        # disques 3×3 : visibles après réduction d'échelle ; rouge = < 30 m, bleu = loin
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                a[uvv[ok, 1] + dy, uvv[ok, 0] + dx] = np.where(far[ok, None], np.array((40, 160, 255), np.uint8),
                                                               np.array((255, 40, 40), np.uint8))
        f_png = f"{out_dir}/{c}_{stamp}_overlay.png"
        Image.fromarray(a).save(f_png)
        crop = a[270:810, 360:1080]
        Image.fromarray(crop).resize((1440, 1080), Image.LANCZOS).save(f"{out_dir}/{c}_{stamp}_overlay_zoom.png")
        res[f"{c}_projected"] = int(ok.sum())
        res[f"{c}_in_range"] = int(inside.sum())
    vol.commit()
    return res


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=1, memory=4096, timeout=600)
def pick_frames(n_frames: int, seq: str = SEQ) -> list[int]:
    import numpy as np

    from splatpipe.common.scene import Scene

    sc = Scene("oxford_spires", seq, data_root=DATA_MOUNT)
    fr = sc.load_frames()["frame_id"]
    return [int(fr[k]) for k in np.linspace(0, len(fr) - 1, n_frames).round().astype(int)]


@app.local_entrypoint()
def validate_remote_main(n_frames: int = 3, seq: str = SEQ):
    import json
    import os

    fids = pick_frames.remote(n_frames, seq)
    os.makedirs("logs/stage3/poses", exist_ok=True)
    for fid in fids:
        r = validate_frame.remote(fid, seq)
        print(json.dumps(r, indent=1))
