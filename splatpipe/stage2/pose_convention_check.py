"""Contrôle de la convention des poses caméra (`poses/cameras.json`) contre la carte accumulée.

`cameras.json` stocke `T_map_cam` = caméra → carte (p_map = R · p_cam + t), comme l'écrit
`stage3/poses_from_traj.py`. La projection correcte est donc

    p_cam = (p_map − t) @ R          # = Rᵀ (p_map − t)
    axe optique dans la carte = R[:, 2]   (3ᵉ COLONNE)

Ce script compare cette lecture à celle utilisée par `m4_splits_sheets.py` / `m4_dense_check.py`
(`(p_map − t) @ R.T`, axe optique = 3ᵉ ligne) de deux façons :
  1. sans image : axe optique de cam0 contre la direction de marche, axe « bas » de l'image contre la gravité ;
  2. avec images : rendu de la carte (z-buffer) depuis la pose, score d'alignement bords de profondeur ↔
     gradients de la photo, et planche photo | rendu correct | rendu M4 pour chaque caméra.

Local (Mac ou desktop), sans Modal :
  .venv/bin/python splatpipe/stage2/pose_convention_check.py --scene data_local/scene --raw data_local/rawimgs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from splatpipe.common.camera_model import FisheyeCamera  # noqa: E402

HALF = 2          # rendu à mi-résolution (720×540) : moins de trous entre voxels de 5 cm
MAX_RANGE = 40.0


def read_ply_xyzi(path):
    ply_t = {"float": "f4", "double": "f8", "uchar": "u1", "char": "i1", "ushort": "u2",
             "short": "i2", "uint": "u4", "int": "i4"}
    props, n = [], 0
    with open(path, "rb") as f:
        while True:
            ln = f.readline().decode("ascii", "replace").split()
            if ln and ln[0] == "element" and ln[1] == "vertex":
                n = int(ln[2])
            elif ln and ln[0] == "property":
                props.append((ln[2], ply_t[ln[1]]))
            elif ln and ln[0] == "end_header":
                break
        offset = f.tell()
    arr = np.fromfile(path, dtype=np.dtype(props), count=n, offset=offset)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1)
    return xyz, arr["intensity"].astype(np.float32)


def check_without_images(poses):
    out = {}
    for c in sorted(poses):
        ks = sorted(poses[c], key=int)
        T = np.array([np.asarray(poses[c][k]["T_map_cam_start"], np.float64) for k in ks])
        vel = np.gradient(T[:, :3, 3], axis=0)
        sp = np.linalg.norm(vel, axis=1)
        m = sp > 0.03
        vdir = vel[m] / sp[m, None]
        out[c] = {
            "cos_marche_axe_colonne": float(np.median((vdir * T[m, :3, 2]).sum(1))),
            "cos_marche_axe_ligne": float(np.median((vdir * T[m, 2, :3]).sum(1))),
            "bas_image_vs_gravite_colonne": float(np.median(-T[:, 2, 1])),
            "bas_image_vs_gravite_ligne": float(np.median(-T[:, 1, 2])),
        }
    return out


def render(xyz, inten, R_cam_from_map, t, cam):
    """Rendu z-buffer à mi-résolution : (profondeur, intensité), inf/nan hors points."""
    d = xyz - t
    near = np.einsum("ij,ij->i", d, d) < MAX_RANGE ** 2
    p = d[near] @ R_cam_from_map.T
    uv, valid, _ = cam.project(p, margin=0)
    ok = valid & (p[:, 2] > 0.2)
    W, H = cam.width // HALF, cam.height // HALF
    u = np.clip((uv[ok, 0] / HALF).astype(np.int64), 0, W - 1)
    v = np.clip((uv[ok, 1] / HALF).astype(np.int64), 0, H - 1)
    depth = np.linalg.norm(p[ok], axis=1)
    cell = v * W + u
    o = np.lexsort((depth, cell))
    first = np.r_[True, cell[o][1:] != cell[o][:-1]]
    sel = o[first]
    D = np.full(H * W, np.inf)
    I = np.full(H * W, np.nan)
    D[cell[sel]] = depth[sel]
    I[cell[sel]] = inten[near][ok][sel]
    return D.reshape(H, W), I.reshape(H, W)


def grad_mag(g):
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1] = g[2:] - g[:-2]
    return np.hypot(gx, gy)


def depth_edges(D):
    e = np.zeros(D.shape, bool)
    for a, b, sl_a, sl_b in (((slice(None), slice(1, None)), (slice(None), slice(None, -1)), None, None),
                             ((slice(1, None), slice(None)), (slice(None, -1), slice(None)), None, None)):
        da, db = D[a], D[b]
        fin = np.isfinite(da) & np.isfinite(db)
        jump = np.zeros(da.shape, bool)
        jump[fin] = np.abs(da[fin] - db[fin]) > 0.15 * np.minimum(da[fin], db[fin])
        e[a] |= jump & (da < db)      # le bord appartient au côté proche
        e[b] |= jump & (db < da)
    return e


def edge_score(gray, D):
    G = grad_mag(gray)
    e = depth_edges(D)
    if e.sum() < 50:
        return float("nan"), int(e.sum())
    return float(np.median(G[e]) / max(np.median(G), 1e-6)), int(e.sum())


def to_rgb_render(I):
    fin = np.isfinite(I)
    out = np.full(I.shape + (3,), 235, np.uint8)
    if fin.any():
        lo, hi = np.percentile(I[fin], [2, 98])
        g = np.clip((I[fin] - lo) / max(hi - lo, 1e-6), 0, 1)
        out[fin] = (40 + 200 * g)[:, None].astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data_local/scene")
    ap.add_argument("--raw", default="data_local/rawimgs")
    ap.add_argument("--out", default="logs/stage2/material/pose_convention_check")
    ap.add_argument("--frames", default="0.25,0.5,0.75", help="positions relatives dans la séquence")
    args = ap.parse_args()

    import yaml

    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.scene}/poses/cameras.json") as f:
        poses = json.load(f)["cameras"]
    with open(f"{args.scene}/sensors.yaml") as f:
        sensors = yaml.safe_load(f)
    report = {"sans_image": check_without_images(poses), "images": []}
    print(json.dumps(report["sans_image"], indent=1))

    xyz, inten = read_ply_xyzi(f"{args.scene}/map/map_5cm_static.ply")
    xyz = xyz.astype(np.float64)
    cams = sorted(poses)
    rel = {}
    for c in cams:
        with open(f"{args.scene}/cameras/{c}/index.csv") as f:
            rel[c] = {str(int(r["stamp_ns"])): os.path.basename(r["rel_path"]) for r in csv.DictReader(f)}

    for fr in [float(x) for x in args.frames.split(",")]:
        tiles = []
        for c in cams:
            ks = sorted(poses[c], key=int)
            k = ks[int(fr * (len(ks) - 1))]
            T = np.asarray(poses[c][k]["T_map_cam_start"], np.float64)
            R, t = T[:3, :3], T[:3, 3]
            cam = FisheyeCamera.from_dict(sensors["cameras"][c])
            img = Image.open(f"{args.raw}/{c}/{rel[c][k]}").convert("L")
            img = img.resize((cam.width // HALF, cam.height // HALF), Image.BILINEAR)
            gray = np.asarray(img, np.float32)
            row = {"cam": c, "stamp_ns": k}
            rgbs = [np.repeat(gray.astype(np.uint8)[..., None], 3, 2)]
            # correct : R_cam_map = Rᵀ ; M4 : R_cam_map = R
            for name, Rcm in (("correct_@R", R.T), ("m4_@R.T", R)):
                D, I = render(xyz, inten, Rcm, t, cam)
                s, ne = edge_score(gray, D)
                row[name] = {"score_bords": s, "n_bords": ne, "couverture": float(np.isfinite(D).mean())}
                rgbs.append(to_rgb_render(I))
            report["images"].append(row)
            print(json.dumps(row))
            tiles.append(rgbs)
        h, w = tiles[0][0].shape[:2]
        sheet = Image.new("RGB", (3 * w, len(tiles) * h + 30), "white")
        dr = ImageDraw.Draw(sheet)
        for j, lab in enumerate(("photo", "rendu carte, (p-t) @ R  [correct]", "rendu carte, (p-t) @ R.T  [code M4]")):
            dr.text((j * w + 10, 8), lab, fill=(0, 0, 0))
        for i, rgbs in enumerate(tiles):
            for j, a in enumerate(rgbs):
                sheet.paste(Image.fromarray(a), (j * w, 30 + i * h))
        p = f"{args.out}/sheet_{int(fr * 100):02d}.jpg"
        sheet.save(p, quality=85)
        print("->", p)
    with open(f"{args.out}/report.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
