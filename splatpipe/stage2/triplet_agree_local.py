"""Accord intra-triplet en **local** (sans Modal) — même algorithme que `triplet_agree` dans
`lift_semantics.py`, transposé pour le Mac (numpy multiprocessing, fork + mémoire partagée).

Prérequis : les données ont été téléchargées du Volume vers `data_local/scene/` (voir
`scripts/download_data_local.sh`) : `map/map_5cm_static.ply`, `reports/stage2_material/superpoints.npz`,
`poses/cameras.json`, `semantics/classes.json`, `semantics/`, `masks/`, `sensors.yaml`,
`cameras/index.csv`, `cameras/frames.csv`.

Lancement :  PYTHONPATH=. .venv/bin/python splatpipe/stage2/triplet_agree_local.py --jobs 6
Reprise :    le compteur de chaque worker est checkpointé dans /tmp (toutes les 200 images retraitées) ;
             relancer la même commande reprend là où c'était.

Sortie : `logs/stage2/material/triplet_local.json` (accords par paire, superpoint et voxel, intra-triplet).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

DATA_DIR = "data_local/scene"
OUT_JSON = "logs/stage2/material/triplet_local.json"
CKPT_DIR = "/tmp/triplet_local_ckpt"
BIT_SKY, BIT_PERSON = 1, 2
MAX_RANGE_M = 55.0
COS_MIN = 0.25
ZCELL = 4
DEPTH_TOL = 0.10
MARGIN_PX = 1
INC = None  # non utilisé ici

# groupes ADE20K (copie conforme à lift_semantics.GROUPS — fichier lu sur place)
GROUPS = [
    ("sky", ["sky"]), ("person", ["person", "apparel"]), ("bike", ["bicycle", "minibike"]),
    ("vehicle", ["car", "bus", "truck", "van", "boat", "airplane", "ship"]),
    ("foliage", ["tree", "palm", "plant", "flower"]),
    ("ground-naturel", ["grass", "field", "earth", "sand", "dirt track", "land", "hill"]),
    ("ground-pave", ["road", "sidewalk", "path", "runway", "floor"]),
    ("batiment", ["wall", "building", "house", "skyscraper", "tower", "bridge", "column", "stairs",
                  "step", "stairway"]),
    ("vitrage", ["windowpane", "glass", "screen door", "mirror"]),
    ("grille-barriere", ["fence", "railing", "bannister", "awning", "canopy"]),
    ("poteaux", ["pole", "streetlight", "traffic light"]),
    ("roche-eau", ["rock", "mountain", "water", "river", "lake", "sea", "waterfall", "fountain",
                   "swimming pool"]),
    ("mobilier-urbain", ["bench", "ashcan", "signboard", "booth", "barrel", "box", "sculpture"]),
    ("autre", []),
]


def _group_map(id2label: dict) -> np.ndarray:
    name_to_g = {n: i for i, (_, names) in enumerate(GROUPS) for n in names}
    out = np.full(150, len(GROUPS) - 1, np.int64)
    for k, v in id2label.items():
        out[int(k)] = name_to_g.get(v.strip().lower(), len(GROUPS) - 1)
    return out


# état partagé chargé avant le fork (copy-on-write : lu, jamais écrit)
S: dict = {}


def _load_shared(data_dir: str) -> None:
    from splatpipe.stage2.superpoints import read_ply_points

    cols = read_ply_points(f"{data_dir}/map/map_5cm_static.ply", ["x", "y", "z", "nx", "ny", "nz"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float32)
    nrm = np.stack([cols["nx"], cols["ny"], cols["nz"]], 1).astype(np.float32)
    keys = pack_keys(xyz.astype(np.float64))
    o = np.argsort(keys, kind="stable")
    S["xyz"] = xyz[o]
    S["nrm"] = nrm[o]
    del xyz, nrm
    with np.load(f"{data_dir}/reports/stage2_material/superpoints.npz") as z:
        S["sp_id"] = z["sp_id"]
    with open(f"{data_dir}/semantics/classes.json") as f:
        S["gmap"] = _group_map(json.load(f)["id2label"])
    with open(f"{data_dir}/poses/cameras.json") as f:
        S["poses"] = json.load(f)["cameras"]
    import yaml

    with open(f"{data_dir}/sensors.yaml") as f:
        sensors = yaml.safe_load(f)
    from splatpipe.common.camera_model import FisheyeCamera

    S["cams"] = sorted(S["poses"])
    S["models"] = {c: FisheyeCamera.from_dict(sensors["cameras"][c]) for c in S["cams"]}


def pack_keys(xyz: np.ndarray) -> np.ndarray:
    from splatpipe.stage1.maptools import pack_ijk

    return pack_ijk(np.floor(xyz / 0.05).astype(np.int64))


def _worker(arg: tuple) -> dict:
    part, n_parts = arg
    from PIL import Image

    xyz_s, nrm_s, sp_id, gmap = S["xyz"], S["nrm"], S["sp_id"], S["gmap"]
    cams, models, poses = S["cams"], S["models"], S["poses"]
    n_sp = int(sp_id.max()) + 1
    ng = len(GROUPS)

    trip = []
    with open(f"{S['data_dir']}/cameras/frames.csv") as f:
        header = f.readline().strip().split(",")
        rows = [ln.strip().split(",") for ln in f if ln.strip()]
    col = {n: i for i, n in enumerate(header)}
    idx = {c: np.loadtxt(f"{S['data_dir']}/cameras/{c}/index.csv", delimiter=",", skiprows=1,
                         usecols=(0,), dtype=np.int64) for c in cams}
    for t, row in enumerate(rows):
        st = {c: int(idx[c][int(row[col[f"{c}_idx"]])]) for c in cams}
        if all(str(st[c]) in poses[c] for c in cams):
            trip.append(st)
    trip = trip[part::n_parts]

    ck = f"{CKPT_DIR}/part{part}.json"
    done0 = 0
    tot = [0, 0, 0]
    ok = [0, 0, 0]
    vtot = [0, 0, 0]
    vok = [0, 0, 0]
    n_trip = 0
    if os.path.exists(ck):
        with open(ck) as f:
            c0 = json.load(f)
        done0, tot, ok, vtot, vok, n_trip = (c0["done"], c0["tot"], c0["ok"], c0["vtot"], c0["vok"],
                                             c0["n_trip"])
        print(f"[reprise] part {part} : {done0} triplets déjà faits", flush=True)

    w = [np.zeros(n_sp, np.float32) for _ in cams]
    gh = [np.zeros((n_sp, ng), np.float16) for _ in cams]  # float16 : 3×128 Mo -> 3×64 Mo
    done = 0
    t0 = time.time()
    for st in trip:
        done += 1
        if done <= done0:
            continue
        seen = {}
        for k, c in enumerate(cams):
            T_mc = np.asarray(poses[c][str(st[c])]["T_map_cam_start"], np.float64)
            R, cc = T_mc[:3, :3].T, T_mc[:3, 3]  # R = rotation carte→caméra (T_map_cam est caméra→carte)
            sem = np.asarray(Image.open(f"{S['data_dir']}/semantics/{c}/{st[c]}.png"))
            msk = np.asarray(Image.open(f"{S['data_dir']}/masks/{c}/{st[c]}.png"))
            model = models[c]
            cw = (model.width + ZCELL - 1) // ZCELL
            # préfiltrage fragmenté (pas de tableau 14,4 M × 3 par worker — RAM limitée)
            cR = cc @ R.T  # (3,)
            cand_chunks = []
            for a0 in range(0, len(xyz_s), 2_000_000):
                d3 = xyz_s[a0:a0 + 2_000_000] - cc.astype(np.float32)
                m = np.einsum("ij,ij->i", d3, d3) <= MAX_RANGE_M**2
                cand_chunks.append(np.flatnonzero(m) + a0)
            cand = np.concatenate(cand_chunks) if cand_chunks else np.zeros(0, np.int64)
            del cand_chunks
            pc = xyz_s[cand].astype(np.float64) @ R.T - cR  # (p - c) @ R.T
            uv, valid, _ = model.project(pc, margin=MARGIN_PX)
            j = np.flatnonzero(valid)
            cand, pc, uv = cand[j], pc[j], uv[j]
            depth = np.sqrt(np.einsum("ij,ij->i", pc, pc))
            view_w = (pc / depth[:, None]) @ R
            cos_i = np.einsum("ij,ij->i", nrm_s[cand], -view_w)
            keep = cos_i >= COS_MIN
            cand, depth, cos_i, pc, uv = (a[keep] for a in (cand, depth, cos_i, pc, uv))
            ui = uv[:, 0].round().astype(np.int64)
            vi = uv[:, 1].round().astype(np.int64)
            cls = sem[vi, ui].astype(np.int64)
            bad = ((msk[vi, ui] & (BIT_SKY | BIT_PERSON)) != 0) | (cls >= 150)
            okm = ~bad
            cand, depth, cos_i, ui, vi, cls = (a[okm] for a in (cand, depth, cos_i, ui, vi, cls))
            cell = (vi // ZCELL) * cw + (ui // ZCELL)
            dmin = np.full(270 * cw, np.inf)
            np.minimum.at(dmin, cell, depth)
            vis = depth <= dmin[cell] + DEPTH_TOL
            spv, wv, gv = sp_id[cand[vis]], cos_i[vis].astype(np.float32), gmap[cls[vis]]
            seen[c] = (spv, wv, gv)
            w[k].fill(0)
            gh[k].fill(0)
            np.add.at(w[k], spv, wv)
            np.add.at(gh[k], (spv, gv), 1.0)
        argm = [gh[k].argmax(1) for k in range(3)]
        for a, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            m = (w[i] >= 0.5) & (w[j] >= 0.5)
            tot[a] += int(m.sum())
            ok[a] += int((argm[i][m] == argm[j][m]).sum())
        ws = {}
        for c in cams:
            spv, wv, gv = seen[c]
            u, inv = np.unique(spv, return_inverse=True)
            wsum = np.zeros(len(u), np.float32)
            np.add.at(wsum, inv, wv)
            ghu = np.zeros((len(u), ng), np.float16)
            np.add.at(ghu, (inv, gv), 1.0)
            ws[c] = (u, wsum, ghu.argmax(1))
        for a, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            ui_, wi_, di = ws[cams[i]]
            uj_, wj_, dj = ws[cams[j]]
            _, xi, xj = np.intersect1d(ui_, uj_, assume_unique=True, return_indices=True)
            m = (wi_[xi] >= 0.5) & (wj_[xj] >= 0.5)
            vtot[a] += int(m.sum())
            vok[a] += int((di[xi[m]] == dj[xj[m]]).sum())
        n_trip += 1
        if n_trip % 200 == 0:
            with open(ck, "w") as f:
                json.dump({"done": done, "tot": tot, "ok": ok, "vtot": vtot, "vok": vok,
                           "n_trip": n_trip}, f)
            print(f"[part {part}] {n_trip}/{len(trip)} triplets ({time.time()-t0:.0f} s)", flush=True)
    with open(ck, "w") as f:
        json.dump({"done": done, "tot": tot, "ok": ok, "vtot": vtot, "vok": vok, "n_trip": n_trip}, f)
    return {"part": part, "n_triplets": n_trip, "tot": tot, "ok": ok, "vtot": vtot, "vok": vok,
            "s": round(time.time() - t0, 1)}


def _init(data_dir: str) -> None:
    S["data_dir"] = data_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(CKPT_DIR, exist_ok=True)
    S["data_dir"] = args.data_dir
    t0 = time.time()
    _load_shared(args.data_dir)
    print(f"données chargées en {time.time()-t0:.0f} s", flush=True)
    with mp.get_context("fork").Pool(args.jobs, initializer=_init, initargs=(args.data_dir,)) as pool:
        outs = pool.map(_worker, [(i, args.jobs) for i in range(args.jobs)])
    tot = [sum(o["tot"][a] for o in outs) for a in range(3)]
    ok = [sum(o["ok"][a] for o in outs) for a in range(3)]
    vtot = [sum(o["vtot"][a] for o in outs) for a in range(3)]
    vok = [sum(o["vok"][a] for o in outs) for a in range(3)]
    res = {"n_triplets": sum(o["n_triplets"] for o in outs),
           "sp_agree_per_pair": [round(o / t, 4) if t else None for o, t in zip(ok, tot)],
           "sp_agree_all": round(sum(ok) / sum(tot), 4) if sum(tot) else None,
           "vox_agree_per_pair": [round(o / t, 4) if t else None for o, t in zip(vok, vtot)],
           "vox_agree_all": round(sum(vok) / sum(vtot), 4) if sum(vtot) else None,
           "n_sp_pairs": sum(tot), "n_vox_pairs": sum(vtot)}
    os.makedirs("logs/stage2/material", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
