"""M3 runner LOCAL pour le desktop (CPU, sans Modal) — pool DINOv2 par superpoint.

Même algorithme de pooling que le worker Modal (`dino_features.py`) : DINOv2 ViT-B/14 518² (CPU, threads),
voxels visibles par image (z-buffer 4×4 ±10 cm, incidence ≥ 0,25, rejets ciel/personne), accumulation par
superpoint (globale) et par caméra (superpoints de surface ≥ 20 voxels uniquement — mémoire).

Précalcul : sous-ensemble de voxels à portée (≤ 55 m) d'au moins une pose de la marche (cKDTree, une fois,
mis en cache). Checkpoints : out/ckpt.pt toutes les 300 images, reprise exacte.

Lancement :  cd /home/ambroise/dino && nohup venv/bin/python dino_desktop.py --stride 3 > run.log 2>&1 &
Sortie :     out/superpoint_features_desktop.npz
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

DATA_DIR = "/home/ambroise/dino/data_local/scene"
OUT_DIR = "/home/ambroise/dino/out"
SP_NPZ = f"{DATA_DIR}/reports/stage2_material/superpoints.npz"
M2_SEM = f"{DATA_DIR}/reports/stage2_material/m2/superpoint_semantics.npz"
MAP_PLY = f"{DATA_DIR}/map/map_5cm_static.ply"
DINO_MODEL = "facebook/dinov2-base"
IMG_SIZE = 518
GRID = IMG_SIZE // 14
MAX_RANGE_M = 55.0
COS_MIN = 0.25
ZCELL = 4
DEPTH_TOL = 0.10
MARGIN_PX = 1
BIT_SKY, BIT_PERSON = 1, 2
CKPT_EVERY = 300
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

S: dict = {}


def load_shared(data_dir: str) -> None:
    import sys

    sys.path.insert(0, "/home/ambroise/dino")
    from splatpipe.stage1.maptools import pack_ijk

    def read_ply_points(path, fields):
        """PLY binaire little-endian, un élément vertex -> colonnes demandées (copie locale,
        le module `superpoints` importe Modal)."""
        ply_t = {"float": "f4", "double": "f8", "uchar": "u1", "char": "i1", "ushort": "u2",
                 "short": "i2", "uint": "u4", "int": "i4"}
        with open(path, "rb") as f:
            lines = []
            while True:
                ln = f.readline()
                if not ln:
                    raise ValueError(f"en-tête PLY inachevé : {path}")
                lines.append(ln)
                if ln.strip() == b"end_header":
                    break
            offset = f.tell()
        fmt, n, props = None, 0, []
        for ln in lines:
            t = ln.decode("ascii", "replace").strip().split()
            if not t:
                continue
            if t[0] == "format":
                fmt = t[1]
            elif t[0] == "element" and t[1] == "vertex":
                n = int(t[2])
            elif t[0] == "property":
                props.append((t[2], ply_t[t[1]]))
        assert fmt == "binary_little_endian", fmt
        arr = np.fromfile(path, dtype=np.dtype(props), count=n, offset=offset)
        return {k: arr[k] for k in fields}

    cols = read_ply_points(MAP_PLY, ["x", "y", "z", "nx", "ny", "nz"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float32)
    nrm = np.stack([cols["nx"], cols["ny"], cols["nz"]], 1).astype(np.float32)
    keys = pack_ijk(np.floor(xyz.astype(np.float64) / 0.05).astype(np.int64))
    o = np.argsort(keys, kind="stable")
    S["xyz"] = xyz[o]
    S["nrm"] = nrm[o]
    S["xyz64"] = xyz[o].astype(np.float64)
    with np.load(SP_NPZ) as z:
        S["sp_id"] = z["sp_id"]
    S["sizes"] = np.bincount(S["sp_id"], minlength=int(S["sp_id"].max()) + 1)
    with open(f"{data_dir}/poses/cameras.json") as f:
        S["poses"] = json.load(f)["cameras"]
    import yaml

    with open(f"{data_dir}/sensors.yaml") as f:
        sensors = yaml.safe_load(f)
    from splatpipe.common.camera_model import FisheyeCamera

    S["cams"] = sorted(S["poses"])
    S["models"] = {c: FisheyeCamera.from_dict(sensors["cameras"][c]) for c in S["cams"]}
    cache = f"{OUT_DIR}/reachable.npz"
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(cache):
        with np.load(cache) as z:
            S["reach"] = z["reach"]
    else:
        from scipy.spatial import cKDTree

        pos = [np.asarray(k["T_map_cam_start"], np.float64)[:3, 3]
               for c in S["cams"] for k in S["poses"][c].values()]
        tree = cKDTree(np.stack(pos))
        d, _ = tree.query(S["xyz"], distance_upper_bound=MAX_RANGE_M, workers=-1)
        S["reach"] = d <= MAX_RANGE_M
        np.savez_compressed(cache, reach=S["reach"])
    print(f"voxels atteignables : {int(S['reach'].sum()):,} / {len(S['reach']):,}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import torch

    torch.set_num_threads(10)
    load_shared(DATA_DIR)
    from PIL import Image
    from transformers import AutoModel

    cams = S["cams"]
    models = S["models"]
    poses = S["poses"]
    xyz_s, nrm_s, sp_id = S["xyz"], S["nrm"], S["sp_id"]
    n_sp = int(sp_id.max()) + 1
    surf = S["sizes"] >= 20
    surf_idx = np.flatnonzero(surf)
    surf_lut = np.full(n_sp, -1, np.int64)
    surf_lut[surf_idx] = np.arange(len(surf_idx))
    cam_pos = {c: cams.index(c) for c in cams}

    dino = AutoModel.from_pretrained(DINO_MODEL, torch_dtype=torch.float32).eval()
    print("DINOv2 chargé", flush=True)

    import csv

    idx = {}
    for c in cams:
        with open(f"{DATA_DIR}/cameras/{c}/index.csv") as f:
            rd = csv.DictReader(f)
            stamps_l, rel_l = [], []
            for row in rd:
                stamps_l.append(int(row["stamp_ns"]))
                rel_l.append(row["rel_path"])
        idx[c] = {"stamp_ns": np.array(stamps_l, np.int64), "rel_path": np.array(rel_l)}
    with open(f"{DATA_DIR}/cameras/frames.csv") as f:
        header = f.readline().strip().split(",")
        rows = [ln.strip().split(",") for ln in f if ln.strip()]
    col = {n2: i for i, n2 in enumerate(header)}
    # stride sur les TRIPLETS (pas sur les images : un stride multiple de 3 ne prendrait
    # qu'une caméra !) — chaque triplet retenu apporte ses 3 images
    triplets = []
    for t in range(len(rows)):
        entries = []
        okk = True
        for c in cams:
            r = int(rows[t][col[f"{c}_idx"]])
            sst = int(idx[c]["stamp_ns"][r])
            rel = str(idx[c]["rel_path"][r])
            if str(sst) not in poses[c]:
                okk = False
                break
            entries.append((c, sst, rel, r))
        if okk:
            triplets.append(entries)
    images = [e for trip in triplets[:: args.stride] for e in trip]
    if args.limit:
        images = images[: args.limit]
    print(f"images à traiter : {len(images)}", flush=True)

    # feat_sum : memmap disque fp16 + clip (fp32 7 Go en RAM -> OOM sur 15 Go) ; survit aux crashes
    fs_path = f"{OUT_DIR}/feat_sum.dat"
    feat_sum = np.memmap(fs_path, dtype=np.float16, mode="w+", shape=(n_sp, 768))
    feat_cam = np.zeros((len(surf_idx), 3, 768), np.float16)
    cnt_cam_surf = np.zeros((len(surf_idx), 3), np.uint32)
    cnt_all = np.zeros(n_sp, np.uint32)
    n_vues = np.zeros(n_sp, np.uint32)
    cam_bits = np.zeros(n_sp, np.uint8)
    ck = f"{OUT_DIR}/ckpt.pt"
    done0 = 0
    if os.path.exists(ck):
        try:
            ckd = torch.load(ck, weights_only=False)
        except Exception as e:
            print(f"[checkpoint corrompu, ignoré] {e}", flush=True)
            ckd = None
        if ckd is not None:
            feat_cam = ckd["feat_cam"]
            cnt_cam_surf, cnt_all = ckd["cnt_cam_surf"], ckd["cnt_all"]
            n_vues, cam_bits, done0 = ckd["n_vues"], ckd["cam_bits"], int(ckd["done"])
            print(f"[reprise] {done0} images déjà faites", flush=True)

    t0 = time.time()
    n_vis_tot = 0
    n_done = 0
    B = 12
    ck_next = ((done0 // CKPT_EVERY) + 1) * CKPT_EVERY

    def save_ckpt(done):
        feat_sum.flush()
        torch.save({"feat_cam": feat_cam, "cnt_cam_surf": cnt_cam_surf,
                    "cnt_all": cnt_all, "n_vues": n_vues, "cam_bits": cam_bits, "done": done,
                    "surf_idx": surf_idx}, ck)

    for b0 in range(0, len(images), B):
        batch = images[b0:b0 + B]
        # --- forward DINOv2 du batch (un appel pour 12 images)
        tnp = np.empty((len(batch), IMG_SIZE, IMG_SIZE, 3), np.float32)
        for bi, (c, sst, rel, _r) in enumerate(batch):
            rgb = Image.open(f"{DATA_DIR}/rawimgs/{c}/{os.path.basename(rel)}").convert("RGB")
            tnp[bi] = np.asarray(rgb.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), np.float32)
        tnp = (tnp / 255.0 - MEAN) / STD
        with torch.no_grad():
            hid = dino(pixel_values=torch.from_numpy(tnp.transpose(0, 3, 1, 2))).last_hidden_state[:, 1:, :]
        tokens = hid.numpy().reshape(len(batch), GRID, GRID, 768)

        # --- pooling par image
        for bi, (c, sst, rel, _r) in enumerate(batch):
            i_img = b0 + bi
            if i_img < done0:
                continue
            k = cam_pos[c]
            tok = tokens[bi]
            T_mc = np.asarray(poses[c][str(sst)]["T_map_cam_start"], np.float64)
            R, cc = T_mc[:3, :3].T, T_mc[:3, 3]  # R = rotation carte→caméra (T_map_cam est caméra→carte)
            msk = np.asarray(Image.open(f"{DATA_DIR}/masks/{c}/{sst}.png"))
            model_c = models[c]
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
            keepm = cos_i >= COS_MIN
            cand, depth, uv = cand[keepm], depth[keepm], uv[keepm]
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
            feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-6)
            n_vis_tot += int(vis.sum())

            # accumulation par unique-sp : tri + reduceat
            o2 = np.argsort(spv, kind="stable")
            spv_s, feats_s = spv[o2], feats[o2]
            starts = np.flatnonzero(np.r_[True, spv_s[1:] != spv_s[:-1]])
            usp = spv_s[starts]
            feat_sum[usp] += np.clip(np.add.reduceat(feats_s, starts, axis=0), -60000, 60000).astype(np.float16)
            cnt_all[usp] += np.diff(np.r_[starts, len(spv_s)]).astype(np.uint32)
            msk_surf = surf[spv]
            if msk_surf.any():
                sps_s, feats_c = spv[msk_surf], feats[msk_surf]
                o3 = np.argsort(sps_s, kind="stable")
                sps_s, feats_c = sps_s[o3], feats_c[o3]
                starts3 = np.flatnonzero(np.r_[True, sps_s[1:] != sps_s[:-1]])
                usp2 = sps_s[starts3]
                srow = surf_lut[usp2]
                feat_cam[srow, k] += np.add.reduceat(feats_c, starts3, axis=0).astype(np.float16)
                cnt_cam_surf[srow, k] += np.diff(np.r_[starts3, len(sps_s)]).astype(np.uint32)
            cn = np.bincount(spv, minlength=n_sp)
            n_vues += cn.astype(np.uint32)
            cam_bits[np.flatnonzero(cn)] |= 1 << k
            n_done += 1
        if n_done >= ck_next:
            save_ckpt(n_done)
            ck_next += CKPT_EVERY
            print(f"{n_done}/{len(images)} images ({time.time()-t0:.0f} s)", flush=True)
    save_ckpt(n_done)
    with np.load(M2_SEM) as z:
        m2_dom = z["dominant_group"].astype(np.int16)
        m2_wsum = z["wsum"].astype(np.float32)
    np.savez_compressed(f"{OUT_DIR}/superpoint_features_desktop.npz", feat_sum=feat_sum,
                        feat_cam=feat_cam, cnt_cam_surf=cnt_cam_surf, cnt_all=cnt_all,
                        n_vues=n_vues, cam_bits=cam_bits, surf_idx=surf_idx,
                        m2_dominant_group=m2_dom, m2_wsum=m2_wsum)
    print(json.dumps({"done": n_done, "n_vis_hits": n_vis_tot,
                      "total_min": round((time.time() - t0) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
