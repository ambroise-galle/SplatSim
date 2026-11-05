"""Tâches 2.5.4–2.5.5 (jalon M3 du plan 2.5) — features DINOv2 par superpoint.

**Écart au plan assumé (coût)** : au lieu de stocker ~35 Go de patch tokens puis une seconde passe de
pooling, l'extraction GPU (DINOv2 ViT-B/14, entrée 518×518, fp16) et le pooling par superpoint sont **fusés
dans une seule passe** : pour chaque image (les 3 caméras de chaque triplet), les voxels visibles (même
z-buffer/rejets que M2) sont projetés, leur pixel donne un patch (grille 37×37 sur l'image 518×518), et la
feature du patch est accumulée par superpoint (somme + comptage, non pondérée) globalement et **par caméra**
(pour la cohérence intra-superpoint inter-vues). Le consensus M2 (classe dominante) est joint aux sorties
comme prior pour les ablations M5.

Accumulateurs par conteneur : `feat_sum` (n_sp × 768 fp32 ≈ 7 Go), `feat_cam` (n_sp × 3 × 768 fp32 ≈ 21 Go).
Checkpoint toutes les 500 images (préemptions, reprise exacte). Agrégation : moyennes, **cohérence
intra-superpoint** (cosinus entre moyennes par caméra, sp avec ≥ 8 votes/caméra, vs paires aléatoires —
cible ≥ 0,7 et ≥ 10 points au-dessus), couverture, rendu PCA-RGB de la carte, `superpoint_features.npz`.

Lancement (pattern spawn+detach robuste, cf. M2) :
  .venv/bin/modal run --detach splatpipe/stage2/dino_features.py::dino_pilot_main     # 60 images, contrôle
  .venv/bin/modal run --detach splatpipe/stage2/dino_features.py::dino_spawn_main --part p --n-parts 4  (×4)
  .venv/bin/modal run splatpipe/stage2/dino_features.py::dino_aggregate_main --n-parts 4
  .venv/bin/modal run splatpipe/stage2/dino_features.py::dino_fetch_main
"""

from __future__ import annotations

import numpy as np

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume
from splatpipe.stage1.images import map_image

app = modal.App("splatsim-stage2-dino")
vol = data_volume()

SEQ = "2024-03-13-observatory-quarter-01"
SP_NPZ = "reports/stage2_material/superpoints.npz"
M2_SEM = "reports/stage2_material/m2/superpoint_semantics.npz"
MAP_PLY = "map/map_5cm_static.ply"
OUT_BASE = "reports/stage2_material"
DINO_MODEL = "facebook/dinov2-base"
IMG_SIZE = 518
PATCH = 14
GRID = IMG_SIZE // PATCH  # 37
MAX_RANGE_M = 55.0
COS_MIN = 0.25
ZCELL = 4
DEPTH_TOL = 0.10
MARGIN_PX = 1
BIT_SKY, BIT_PERSON = 1, 2
CKPT_EVERY = 500
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def dino_image() -> modal.Image:
    """L4 : torch cu128 + transformers (DINOv2) + la pile numpy/PIL de projection."""
    return (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
        .uv_pip_install(
            "transformers==4.46.3", "numpy==2.3.3", "scipy==1.16.2", "pillow==11.3.0",
            "pyyaml==6.0.2", "matplotlib==3.10.6",
        )
        .env({"HF_HOME": "/opt/hf_cache", "MPLBACKEND": "Agg"})
        .add_local_python_source("splatpipe")
    )


# =============================================================================
# Worker GPU : extraction + pooling par superpoint
# =============================================================================

@app.function(image=dino_image(), gpu="L4", volumes={DATA_MOUNT: vol}, cpu=8, memory=56 * 1024,
              timeout=6 * 3600)
def dino_worker(part: int, n_parts: int, tag: str = "m3", limit_images: int = 0, seq: str = SEQ) -> dict:
    import json
    import os
    import time

    import torch
    from PIL import Image
    from transformers import AutoModel

    from splatpipe.common.camera_model import FisheyeCamera
    from splatpipe.common.scene import Scene
    from splatpipe.stage1.maptools import pack_ijk, scene_root
    from splatpipe.stage2.dino_features import (
        BIT_PERSON, BIT_SKY, CKPT_EVERY, COS_MIN, DEPTH_TOL, GRID, IMG_SIZE, MARGIN_PX, MAP_PLY,
        MAX_RANGE_M, OUT_BASE, PATCH, SP_NPZ, STD, ZCELL,
    )
    from splatpipe.stage2.superpoints import read_ply_points

    t0 = time.time()
    root = scene_root(DATA_MOUNT, seq)
    out_dir = f"{root}/{OUT_BASE}/{tag}"
    os.makedirs(out_dir, exist_ok=True)

    cols = read_ply_points(f"{root}/{MAP_PLY}", ["x", "y", "z", "nx", "ny", "nz"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float32)
    nrm = np.stack([cols["nx"], cols["ny"], cols["nz"]], 1).astype(np.float32)
    keys = pack_ijk(np.floor(xyz.astype(np.float64) / 0.05).astype(np.int64))
    o = np.argsort(keys, kind="stable")
    xyz_s, nrm_s = xyz[o], nrm[o]
    del cols, xyz, nrm, keys, o
    with np.load(f"{root}/{SP_NPZ}") as z:
        sp_id = z["sp_id"]
    n_sp = int(sp_id.max()) + 1

    sc = Scene("oxford_spires", seq, data_root=DATA_MOUNT)
    cam_models = {c: FisheyeCamera.from_dict(sc.camera(c)) for c in sc.camera_ids()}
    with open(f"{root}/poses/cameras.json") as f:
        poses = json.load(f)["cameras"]
    cams = sorted(poses)
    frames = sc.load_frames()
    idx = {c: sc.load_camera_index(c) for c in cams}

    # images = (cam, stamp, rel_path) des triplets à poses valides ; tranche équilibrée par part
    images = []
    for t in range(len(frames["frame_id"])):
        rows = {c: int(frames[f"{c}_idx"][t]) for c in cams}
        st = {c: int(idx[c]["stamp_ns"][rows[c]]) for c in cams}
        if not all(str(st[c]) in poses[c] for c in cams):
            continue
        for c in cams:
            images.append((c, st[c], str(idx[c]["rel_path"][rows[c]])))
    images = images[part::n_parts]
    if limit_images:
        images = images[:limit_images]

    model = AutoModel.from_pretrained(DINO_MODEL, torch_dtype=torch.float16).cuda().eval()

    feat_sum = np.zeros((n_sp, 768), np.float32)
    feat_cam = np.zeros((n_sp, 3, 768), np.float32)
    cnt_cam = np.zeros((n_sp, 3), np.uint32)
    n_vues = np.zeros(n_sp, np.uint32)
    cam_bits = np.zeros(n_sp, np.uint8)
    ckpt_path = f"{out_dir}/dino_part_{part}_ckpt.npz"
    done0 = 0
    if os.path.exists(ckpt_path):
        with np.load(ckpt_path) as z:
            feat_sum = z["feat_sum"].astype(np.float32)
            feat_cam = z["feat_cam"].astype(np.float32)
            cnt_cam, n_vues, cam_bits = z["cnt_cam"], z["n_vues"], z["cam_bits"]
            done0 = int(z["done"])
        print(f"[reprise] part {part} : {done0} images déjà faites", flush=True)
    n_done = 0
    n_vis_tot = 0

    def save_ckpt():
        np.savez_compressed(ckpt_path, feat_sum=feat_sum.astype(np.float16),
                            feat_cam=feat_cam.astype(np.float16), cnt_cam=cnt_cam, n_vues=n_vues,
                            cam_bits=cam_bits, done=n_done)
        vol.commit()

    B = 12
    for b0 in range(0, len(images), B):
        batch = images[b0:b0 + B]
        tnp = np.empty((len(batch), IMG_SIZE, IMG_SIZE, 3), np.float32)
        for bi, (c, st, rel) in enumerate(batch):
            img = Image.open(f"{DATA_MOUNT}/raw/{rel.removeprefix('raw/')}").convert("RGB")
            tnp[bi] = np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), np.float32)
        tnp = (tnp / 255.0 - MEAN) / STD
        x = torch.from_numpy(tnp.transpose(0, 3, 1, 2)).cuda().half()
        with torch.no_grad():
            hid = model(pixel_values=x).last_hidden_state[:, 1:, :]  # (B, 1369, 768)
        tokens = hid.float().cpu().numpy().reshape(len(batch), GRID, GRID, 768)

        for bi, (c, st, rel) in enumerate(batch):
            n_done += 1
            if n_done <= done0:
                continue
            k = cams.index(c)
            tok = tokens[bi]
            T_mc = np.asarray(poses[c][str(st)]["T_map_cam_start"], np.float64)
            R, cc = T_mc[:3, :3].T, T_mc[:3, 3]  # R = rotation carte→caméra (T_map_cam est caméra→carte)
            msk = np.asarray(Image.open(f"{root}/masks/{c}/{st}.png"))
            model_c = cam_models[c]
            cw = (model_c.width + ZCELL - 1) // ZCELL
            # préfiltrage fragmenté + projection
            cand_chunks = []
            for a0 in range(0, len(xyz_s), 2_000_000):
                dd = xyz_s[a0:a0 + 2_000_000] - cc.astype(np.float32)
                m = np.einsum("ij,ij->i", dd, dd) <= MAX_RANGE_M**2
                cand_chunks.append(np.flatnonzero(m) + a0)
            cand = np.concatenate(cand_chunks)
            del cand_chunks
            pcf = (xyz_s[cand].astype(np.float64) - cc) @ R.T
            uv, valid, _ = model_c.project(pcf, margin=MARGIN_PX)
            jj = np.flatnonzero(valid)
            cand, pcf, uv = cand[jj], pcf[jj], uv[jj]
            depth = np.sqrt(np.einsum("ij,ij->i", pcf, pcf))
            view_w = (pcf / depth[:, None]) @ R
            cos_i = np.einsum("ij,ij->i", nrm_s[cand].astype(np.float64), -view_w)
            keep = cos_i >= COS_MIN
            cand, depth, uv = cand[keep], depth[keep], uv[keep]
            ui = uv[:, 0].round().astype(np.int64)
            vi = uv[:, 1].round().astype(np.int64)
            bad = (msk[vi, ui] & (BIT_SKY | BIT_PERSON)) != 0
            cand, depth, ui, vi = cand[~bad], depth[~bad], ui[~bad], vi[~bad]
            cell = (vi // ZCELL) * cw + (ui // ZCELL)
            dmin = np.full(270 * cw, np.inf)
            np.minimum.at(dmin, cell, depth)
            vis = depth <= dmin[cell] + DEPTH_TOL
            spv = sp_id[cand[vis]]
            pu = np.clip(ui[vis] * IMG_SIZE // model_c.width, 0, GRID - 1)
            pv = np.clip(vi[vis] * IMG_SIZE // model_c.height, 0, GRID - 1)
            feats = tok[pv, pu].astype(np.float32)
            feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-6)  # unitaires
            np.add.at(feat_sum, spv, feats)
            np.add.at(feat_cam[:, k, :], spv, feats)
            np.add.at(cnt_cam[:, k], spv, 1)
            cn = np.bincount(spv, minlength=n_sp)
            n_vues += cn.astype(np.uint32)
            cam_bits[np.flatnonzero(cn)] |= 1 << k
            n_vis_tot += int(vis.sum())
        if n_done % CKPT_EVERY < B:  # checkpoint à chaque multiple de CKPT_EVERY franchi dans le batch
            save_ckpt()
            print(f"[part {part}] {n_done}/{len(images)} images ({time.time()-t0:.0f} s)", flush=True)

    if n_done % CKPT_EVERY != 0:
        save_ckpt()
    np.savez_compressed(f"{out_dir}/dino_part_{part}.npz", feat_sum=feat_sum.astype(np.float16),
                        feat_cam=feat_cam.astype(np.float16), cnt_cam=cnt_cam, n_vues=n_vues,
                        cam_bits=cam_bits)
    os.remove(ckpt_path)
    vol.commit()
    return {"part": part, "n_images": int(n_done), "n_vis_hits": n_vis_tot,
            "s": round(time.time() - t0, 1)}


# =============================================================================
# Agrégation : moyennes, cohérence intra-sp, PCA-RGB
# =============================================================================

@app.function(image=dino_image(), volumes={DATA_MOUNT: vol}, cpu=16, memory=80 * 1024, timeout=4 * 3600)
def dino_aggregate(n_parts: int, tag: str = "m3", seq: str = SEQ) -> dict:
    import json
    import os
    import time

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from splatpipe.stage1.maptools import pack_ijk, scene_root
    from splatpipe.stage2.dino_features import OUT_BASE, SP_NPZ
    from splatpipe.stage2.superpoints import read_ply_points

    t0 = time.time()
    rngm = np.random.default_rng(0)
    root = scene_root(DATA_MOUNT, seq)
    out_dir = f"{root}/{OUT_BASE}/{tag}"
    with np.load(f"{root}/{SP_NPZ}") as z:
        sp_id = z["sp_id"]
    n_sp = int(sp_id.max()) + 1
    with np.load(f"{root}/{OUT_BASE}/m2/superpoint_semantics.npz") as z:
        dom_g = z["dominant_group"]
        wsum_m2 = z["wsum"]

    feat_sum = np.zeros((n_sp, 768), np.float32)
    feat_cam = np.zeros((n_sp, 3, 768), np.float32)
    cnt_cam = np.zeros((n_sp, 3), np.uint32)
    n_vues = np.zeros(n_sp, np.uint32)
    cam_bits = np.zeros(n_sp, np.uint8)
    for p in range(n_parts):
        f = f"{out_dir}/dino_part_{p}.npz"
        with np.load(f) as z:
            feat_sum += z["feat_sum"].astype(np.float32)
            feat_cam += z["feat_cam"].astype(np.float32)
            cnt_cam += z["cnt_cam"]
            n_vues += z["n_vues"]
            cam_bits |= z["cam_bits"]
        os.remove(f)
    cnt = cnt_cam.sum(1)
    seen = cnt > 0
    feat_mean = np.zeros((n_sp, 768), np.float16)
    feat_mean[seen] = (feat_sum[seen] / cnt[seen, None]).astype(np.float16)
    cam_mean = np.zeros((n_sp, 3, 768), np.float16)
    for k in range(3):
        ccm = cnt_cam[:, k] > 0
        cam_mean[ccm, k, :] = (feat_cam[ccm, k, :] / cnt_cam[ccm, k, None]).astype(np.float16)

    # cohérence intra-superpoint : cosinus entre moyennes par caméra (≥ 8 votes/caméra)
    fn = cam_mean.astype(np.float32)
    fn /= np.maximum(np.linalg.norm(fn, axis=3, keepdims=True), 1e-6)
    cos_all, cos_rand = [], []
    rngl = np.random.default_rng(1)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        m = (cnt_cam[:, i] >= 8) & (cnt_cam[:, j] >= 8)
        cs = (fn[m, i] * fn[m, j]).sum(1)
        cos_all.append(cs)
        cos_rand.append(cs[rngl.permutation(len(cs))])
    cos_all = np.concatenate(cos_all)
    cos_rand = np.concatenate(cos_rand)

    rep = {"stage": "2.5.4/2.5.5", "tag": tag, "n_superpoints": int(n_sp),
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "coverage_superpoint": float(seen.mean()),
           "n_comparable_sp": int(len(cos_all)),
           "cos_intra_sp_mean": round(float(cos_all.mean()), 4),
           "cos_intra_sp_p10": round(float(np.percentile(cos_all, 10)), 4),
           "cos_random_mean": round(float(cos_rand.mean()), 4),
           "cos_gap": round(float(cos_all.mean() - cos_rand.mean()), 4),
           "timing_s": round(time.time() - t0, 1)}

    np.savez(f"{out_dir}/superpoint_features.npz", feat=feat_mean, feat_cam=cam_mean,
             cnt=cnt.astype(np.uint32), cnt_cam=cnt_cam, n_vues=n_vues, cam_bits=cam_bits,
             m2_dominant_group=dom_g.astype(np.int16), m2_wsum=wsum_m2.astype(np.float32))

    # figures : PCA-RGB de la carte + histogramme des cosinus
    figures: dict[str, bytes] = {}
    cols = read_ply_points(f"{root}/map/map_5cm_static.ply", ["x", "y", "z"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    keys = pack_ijk(np.floor(xyz / 0.05).astype(np.int64))
    o = np.argsort(keys, kind="stable")
    xyz_s = xyz[o].astype(np.float32)
    feat_s = feat_mean[sp_id].astype(np.float32)
    seen_s = seen[sp_id]
    sizes = np.bincount(sp_id, minlength=n_sp)
    surf = seen_s & (sizes[sp_id] >= 20)
    F = feat_s[surf]
    F = F - F.mean(0, keepdims=True)
    sub = rngm.choice(len(F), min(60_000, len(F)), replace=False)
    _, _, Vt = np.linalg.svd(F[sub], full_matrices=False)
    P = F @ Vt[:3].T
    lo, hi = np.percentile(P, [1, 99], axis=0)
    Pn = np.clip((P - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    rgbp = np.full((n, 3), 0.85, np.float32)
    rgbp[surf] = Pn
    fig, ax = plt.subplots(figsize=(10, 9), facecolor="white")
    idxr = rngm.choice(n, 1_800_000, replace=False)
    ax.scatter(xyz_s[idxr, 0], xyz_s[idxr, 1], c=rgbp[idxr], s=0.08, marker=".", rasterized=True,
               linewidths=0)
    ax.set_aspect("equal")
    ax.set_title(f"M3 [{tag}] — PCA(RGB) des features DINOv2 par superpoint (gris = sans feature)", fontsize=9,
                 loc="left")
    ax.set_xlabel("x map (m)", fontsize=8)
    ax.set_ylabel("y map (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    fp = f"{out_dir}/dino_pca_top.png"
    fig.savefig(fp, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figures[os.path.basename(fp)] = open(fp, "rb").read()

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    bins = np.linspace(-1, 1, 80)
    ax.hist(cos_rand, bins=bins, alpha=0.6, label="paires aléatoires", color="#bbbbbb")
    ax.hist(cos_all, bins=bins, alpha=0.7, label="paires intra-superpoint (caméras)", color="#2a78d6")
    ax.axvline(0.7, color="#c8380d", linestyle="--", linewidth=1, label="cible 0,7")
    ax.set_xlabel("cosinus entre moyennes par caméra", fontsize=8)
    ax.set_ylabel("nb de superpoints", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title(f"cohérence intra-superpoint : moyenne {rep['cos_intra_sp_mean']}, "
                 f"aléatoire {rep['cos_random_mean']}, écart {rep['cos_gap']}", fontsize=9, loc="left")
    ax.tick_params(labelsize=7)
    fc = f"{out_dir}/dino_cos_hist.png"
    fig.savefig(fc, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figures[os.path.basename(fc)] = open(fc, "rb").read()

    with open(f"{out_dir}/report.json", "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    vol.commit()
    return {"report": rep, "figures": figures}


# =============================================================================
# Entrées locales (pattern spawn+detach robuste, cf. M2)
# =============================================================================

@app.local_entrypoint()
def dino_pilot_main(limit_images: int = 60):
    import json
    import os

    out = dino_worker.remote(0, 1, tag="m3_pilot", limit_images=limit_images)
    agg = dino_aggregate.remote(1, tag="m3_pilot")
    os.makedirs("logs/stage2/material", exist_ok=True)
    for name, b in agg["figures"].items():
        open(f"logs/stage2/material/{name}", "wb").write(b)
    with open("logs/stage2/material/m3_pilot_report.json", "w") as f:
        json.dump(agg["report"], f, indent=1, ensure_ascii=False)
    print(json.dumps({"worker": out, "report": agg["report"]}, indent=1, ensure_ascii=False))


@app.local_entrypoint()
def dino_spawn_main(part: int, n_parts: int = 4, tag: str = "m3"):
    h = dino_worker.spawn(part, n_parts, tag=tag)
    print(f"spawned dino part {part}: {h.object_id}")


@app.function(image=dino_image(), volumes={DATA_MOUNT: vol}, cpu=2, memory=8 * 1024, timeout=1800)
def dino_fetch(tag: str = "m3") -> dict:
    import os

    out_dir = f"{DATA_MOUNT}/scenes/oxford_spires/{SEQ}/{OUT_BASE}/{tag}"
    files = {}
    rp = f"{out_dir}/report.json"
    if os.path.exists(rp):
        files["report.json"] = open(rp, "rb").read()
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".png"):
            files[f] = open(f"{out_dir}/{f}", "rb").read()
    return files


@app.local_entrypoint()
def dino_fetch_main(tag: str = "m3"):
    import json
    import os

    files = dino_fetch.remote(tag)
    os.makedirs("logs/stage2/material", exist_ok=True)
    for name, b in files.items():
        open(f"logs/stage2/material/{name if name != 'report.json' else f'{tag}_report.json'}", "wb").write(b)
    print(json.dumps(json.loads(files["report.json"]), indent=1, ensure_ascii=False))


@app.local_entrypoint()
def dino_aggregate_main(n_parts: int = 4, tag: str = "m3"):
    import json
    import os

    agg = dino_aggregate.remote(n_parts, tag=tag)
    os.makedirs("logs/stage2/material", exist_ok=True)
    for name, b in agg["figures"].items():
        open(f"logs/stage2/material/{name}", "wb").write(b)
    with open(f"logs/stage2/material/{tag}_report.json", "w") as f:
        json.dump(agg["report"], f, indent=1, ensure_ascii=False)
    print(json.dumps(agg["report"], indent=1, ensure_ascii=False))
