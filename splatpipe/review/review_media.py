"""Paquet de revue — supports visuels à partir des sorties des étapes 1 à 3.

Livrables (écrits sur le Volume dans `reports/review/`, copies locales dans `logs/review/`) :
  1. `labeled_accumulation_sub6.ply` : accumulation LiDAR **étiquetée** par point (1 point sur 6),
     champs x/y/z, rgb (groupe sémantique ; magenta = dynamique DUFOMap), ade, bits, dyn, n_votes, scan_idx
     (index dans `labeled_accumulation_index.json`) ;
  2. `seq_<cam>_rgb_top_sem_bottom.mp4` : séquence images — RVB dessus, couche sémantique dessous ;
  3. `seq_<cam>_projected.mp4` : RVB avec le nuage étiqueté du tour le plus proche projeté par les poses 3.1.

Couleurs partagées : végétation vert (40,170,60) · personne rouge (230,40,40) · véhicule jaune (235,200,40)
· ciel bleu (60,140,235) · structure gris (120,120,124) · dynamique DUFOMap magenta (200,60,200).

Lancement :
  .venv/bin/modal run splatpipe/review/review_media.py::labeled_ply_main
  .venv/bin/modal run splatpipe/review/review_media.py::seq_video_main
  .venv/bin/modal run splatpipe/review/review_media.py::proj_video_main
"""

from __future__ import annotations

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume

app = modal.App("splatsim-review")
vol = data_volume()

# Image légère : ffmpeg + PIL + numpy suffisent (lecture scènes, projection fisheye, écriture PLY).
# `add_local_python_source` en dernier (Modal interdit les étapes de build après add_local_*).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_pip_install("numpy==2.3.3", "pillow==11.3.0", "pyyaml==6.0.2")
    .add_local_python_source("splatpipe")
)

SEQ = "2024-03-13-observatory-quarter-01"
OUT_DIR = "reports/review"
WIDTH = 960
FPS = 20
GROUP_COLORS = {"vegetation": (40, 170, 60), "person": (230, 40, 40), "vehicle": (235, 200, 40),
                "sky": (60, 140, 235), "structure": (120, 120, 124), "dufo_dyn": (200, 60, 200)}
BIT_SKY, BIT_PERSON, BIT_VEHICLE, BIT_VEGETATION = 1, 2, 4, 8


def _lut_bits() -> "np.ndarray":
    import numpy as np

    lut = np.zeros((256, 3), np.uint8)
    lut[0] = GROUP_COLORS["structure"]
    lut[BIT_SKY] = GROUP_COLORS["sky"]
    lut[BIT_PERSON] = GROUP_COLORS["person"]
    lut[BIT_VEHICLE] = GROUP_COLORS["vehicle"]
    lut[BIT_VEGETATION] = GROUP_COLORS["vegetation"]
    return lut


def _start_ffmpeg(w: int, h: int, fps: int, out_path: str):
    import subprocess

    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", out_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _group_rgb(bits: "np.ndarray", dyn_dufo: "np.ndarray") -> "np.ndarray":
    import numpy as np

    lut = _lut_bits()
    rgb = lut[bits].copy()
    rgb[dyn_dufo] = GROUP_COLORS["dufo_dyn"]
    return rgb


# --------------------------------------------------------------------------------------------------
# 1. accumulation étiquetée (PLY)
# --------------------------------------------------------------------------------------------------


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=8, memory=96 * 1024, timeout=2 * 3600)
def labeled_ply(stride: int = 6, seq: str = SEQ) -> dict:
    import json
    import os
    import time

    import numpy as np

    from splatpipe.stage1.maptools import scene_root, write_ply

    t0 = time.time()
    root = scene_root(DATA_MOUNT, seq)
    dd = f"{root}/lidar/hesai_deskewed"
    dl = f"{root}/lidar/dyn_labels_dufo"
    sl = f"{root}/semantics_lidar"
    stamps = sorted(int(s[:-4]) for s in os.listdir(dd) if s.endswith(".npz") and int(s[:-4]) in
                    {int(f[:-4]) for f in os.listdir(sl) if f.endswith(".npz")})
    xyz_l, rgb_l, ade_l, bits_l, dyn_l, nv_l, sid_l = [], [], [], [], [], [], []
    index = {"stamps": [], "scan_idx": []}
    for k, st in enumerate(stamps):
        with np.load(f"{dd}/{st}.npz") as z:
            xyz = z["xyz"][::stride]
        with np.load(f"{dl}/{st}.npz") as z:
            dyn_dufo = z["dyn"][::stride].astype(bool)
        with np.load(f"{sl}/{st}.npz") as z:
            ade, bits, dyn, nv = z["ade"][::stride], z["bits"][::stride], z["dyn"][::stride], z["n_votes"][::stride]
        xyz_l.append(xyz)
        rgb_l.append(_group_rgb(bits, dyn_dufo))
        ade_l.append(ade)
        bits_l.append(bits)
        dyn_l.append(dyn)
        nv_l.append(nv)
        sid_l.append(np.full(len(xyz), k, np.uint32))
        index["stamps"].append(int(st))
        index["scan_idx"].append(k)
    xyz = np.concatenate(xyz_l)
    rgb = np.concatenate(rgb_l)
    n = len(xyz)
    cols = {"x": xyz[:, 0].astype(np.float32), "y": xyz[:, 1].astype(np.float32), "z": xyz[:, 2].astype(np.float32),
            "red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2],
            "ade": np.concatenate(ade_l), "bits": np.concatenate(bits_l), "dyn": np.concatenate(dyn_l),
            "n_votes": np.concatenate(nv_l), "scan_idx": np.concatenate(sid_l)}
    os.makedirs(f"{root}/{OUT_DIR}", exist_ok=True)
    ply = f"{root}/{OUT_DIR}/labeled_accumulation_sub{stride}.ply"
    nb = write_ply(ply, cols, comments=[
        f"SplatSim revue : accumulation LiDAR etiquetee (1 point sur {stride})",
        "rgb = groupe semantique (vert vegetation, rouge personne, jaune vehicule, bleu ciel, gris structure) ;",
        "magenta = dynamique DUFOMap (dyn_labels_dufo) ; ade = classe ADE20K majoritaire (255 = sans vote) ;",
        "dyn = union DUFOMap OU bit personne (2.3) ; scan_idx -> labeled_accumulation_index.json"])
    index.update({"stride": stride, "n_points": int(n), "n_scans": len(stamps),
                  "colors": GROUP_COLORS, "ply": os.path.basename(ply), "ply_bytes": nb})
    with open(f"{root}/{OUT_DIR}/labeled_accumulation_index.json", "w") as f:
        json.dump(index, f, indent=1)
    vol.commit()
    return {"n_points": int(n), "n_scans": len(stamps), "ply_bytes": nb, "total_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def labeled_ply_main(stride: int = 6, seq: str = SEQ):
    import json

    print(json.dumps(labeled_ply.remote(stride, seq), indent=1))


# --------------------------------------------------------------------------------------------------
# 2. vidéo séquence : RVB dessus, sémantique dessous
# --------------------------------------------------------------------------------------------------


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=8, memory=32 * 1024, timeout=2 * 3600)
def seq_video(cam: str = "cam0", width: int = WIDTH, fps: int = FPS, seq: str = SEQ) -> dict:
    import os
    import time

    import numpy as np
    from PIL import Image

    from splatpipe.common.scene import Scene

    t0 = time.time()
    sc = Scene("oxford_spires", seq, data_root=DATA_MOUNT)
    idx = sc.load_camera_index(cam)
    frames = sc.load_frames()
    sel = frames[f"{cam}_idx"]
    lut = _lut_bits()
    H = int(round(1080 * width / 1440))
    W = width
    out = f"{sc.root}/{OUT_DIR}/seq_{cam}_rgb_top_sem_bottom.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    proc = _start_ffmpeg(W, 2 * H, fps, out)
    n_missing = 0
    prev = None
    for i in range(len(sel)):
        j = int(sel[i])
        rel = str(idx["rel_path"][j])
        rgb = sem = None
        if os.path.exists(sc.abs_path(rel)):
            im = Image.open(sc.abs_path(rel)).resize((W, H), Image.BILINEAR)
            rgb = np.asarray(im)
        mpath = sc.path("masks", cam, f"{int(idx['stamp_ns'][j])}.png")
        if os.path.exists(mpath):
            m = np.asarray(Image.open(mpath))
            sem = lut[m]
            sem = np.asarray(Image.fromarray(sem).resize((W, H), Image.NEAREST))
        if rgb is None or sem is None:
            if prev is None:
                n_missing += 1
                continue
            rgb, sem = prev
        else:
            prev = (rgb, sem)
        proc.stdin.write(np.vstack([rgb, sem]).tobytes())
    proc.stdin.close()
    proc.wait()
    vol.commit()
    return {"video": os.path.basename(out), "bytes": os.path.getsize(out), "n_frames": int(len(sel)),
            "n_missing_start": n_missing, "size": f"{W}x{2 * H}", "fps": fps,
            "total_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def seq_video_main(cam: str = "cam0", width: int = WIDTH, fps: int = FPS, seq: str = SEQ):
    import json

    print(json.dumps(seq_video.remote(cam, width, fps, seq), indent=1))


# --------------------------------------------------------------------------------------------------
# 3. vidéo séquence : nuage étiqueté projeté dans les images
# --------------------------------------------------------------------------------------------------


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=8, memory=48 * 1024, timeout=2 * 3600)
def proj_video(cam: str = "cam0", width: int = WIDTH, fps: int = FPS, seq: str = SEQ,
               time_window_s: float = 0.03, max_range_m: float = 45.0) -> dict:
    import json
    import os
    import time

    import numpy as np
    from PIL import Image

    from splatpipe.common.camera_model import FisheyeCamera
    from splatpipe.common.scene import Scene

    t0 = time.time()
    sc = Scene("oxford_spires", seq, data_root=DATA_MOUNT)
    idx = sc.load_camera_index(cam)
    frames = sc.load_frames()
    sel = frames[f"{cam}_idx"]
    with open(sc.path("poses", "cameras.json")) as f:
        poses = json.load(f)["cameras"][cam]
    cam_model = FisheyeCamera.from_dict(sc.camera(cam))
    H = int(round(1080 * width / 1440))
    W = width
    dd = sc.path("lidar", "hesai_deskewed")
    sl = sc.path("semantics_lidar")
    scan_stamps = np.array(sorted(int(s[:-4]) for s in os.listdir(dd) if s.endswith(".npz")))
    scan_mid = scan_stamps / 1e9 + 0.05

    out = f"{sc.root}/{OUT_DIR}/seq_{cam}_projected.mp4"
    proc = _start_ffmpeg(W, H, fps, out)
    lut = _lut_bits()
    cache = {}
    n_held, n_frames_drawn, n_pts_drawn = 0, 0, 0
    prev = None
    for i in range(len(sel)):
        j = int(sel[i])
        key = str(int(idx["stamp_ns"][j]))
        rel = str(idx["rel_path"][j])
        if not (os.path.exists(sc.abs_path(rel)) and key in poses):
            if prev is not None:
                proc.stdin.write(prev.tobytes())
            else:
                n_held += 1
            continue
        t_cam = poses[key]["t_start"]
        T_mc = np.asarray(poses[key]["T_map_cam_start"], float)
        im = Image.open(sc.abs_path(rel)).resize((W, H), Image.BILINEAR)
        a = np.asarray(im).copy()
        jj = int(np.argmin(np.abs(scan_mid - t_cam)))
        if abs(scan_mid[jj] - t_cam) <= 0.06:
            st = int(scan_stamps[jj])
            if st not in cache:
                with np.load(f"{dd}/{st}.npz") as z:
                    with np.load(f"{sl}/{st}.npz") as z2:
                        cache[st] = (z["xyz"], z["t_rel"].astype(np.float64), float(z["t_start"]),
                                     z2["bits"], z2["dyn"])
                if len(cache) > 3:
                    cache.pop(next(iter(cache)))
            xyz, t_rel, t_start, bits, dyn = cache[st]
            t_pt = t_start + t_rel
            o = T_mc[:3, 3]
            rng3 = np.linalg.norm(xyz.astype(np.float64) - o, axis=1)
            keep = (np.abs(t_pt - t_cam) <= time_window_s) & (rng3 <= max_range_m)
            if keep.any():
                p_cam = (xyz[keep].astype(np.float64) - o) @ T_mc[:3, :3]
                uv, valid, _ = cam_model.project(p_cam, margin=1)
                ui = uv[valid].round().astype(int)
                inb = (ui[:, 0] >= 0) & (ui[:, 0] < W) & (ui[:, 1] >= 0) & (ui[:, 1] < H)
                b = ui[inb]
                col = lut[bits[keep][valid][inb]].copy()
                col[dyn[keep][valid][inb]] = GROUP_COLORS["dufo_dyn"]
                sx, sy = W / 1440, H / 1080
                xs, ys = (b[:, 0] * sx).astype(int), (b[:, 1] * sy).astype(int)
                a[ys, xs] = col
                a[np.minimum(ys + 1, H - 1), xs] = col
                n_pts_drawn += int(inb.sum())
            n_frames_drawn += 1
        else:
            n_held += 1
        frame = a
        proc.stdin.write(frame.tobytes())
        prev = frame
    proc.stdin.close()
    proc.wait()
    vol.commit()
    return {"video": os.path.basename(out), "bytes": os.path.getsize(out), "n_frames": int(len(sel)),
            "n_frames_drawn": n_frames_drawn, "n_pts_drawn": n_pts_drawn, "n_held": n_held,
            "size": f"{W}x{H}", "fps": fps, "total_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def proj_video_main(cam: str = "cam0", width: int = WIDTH, fps: int = FPS, seq: str = SEQ):
    import json

    print(json.dumps(proj_video.remote(cam, width, fps, seq), indent=1))
