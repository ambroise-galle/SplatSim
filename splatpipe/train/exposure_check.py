"""L'exposition des caméras est-elle constante ? Mesure directe sur les images (CPU, Mac ou desktop).

Pas d'EXIF dans les JPEG du dataset. On compare donc la luminosité des MÊMES voxels de la carte vus dans deux
images de la même caméra : paires consécutives (50 ms, point de vue quasi identique) et paires à 1 s. Pour
chaque paire : projection fisheye des voxels à ≤ 25 m, z-buffer (cellules 4 px, ±10 cm), luminance lue dans les
deux images, voxels vus dans les deux et non saturés (15–240) → médiane du log-rapport de luminance.
Exposition constante et éclairage stable ⇒ rapports ≈ 1 (|log| < 0,01–0,02). Exposition automatique ⇒ sauts.
Aussi : comparaison entre caméras au même instant (zones de recouvrement) → gains relatifs des 3 caméras.

  .venv/bin/python -m splatpipe.train.exposure_check --root data_local/scene --raw data_local/rawimgs
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from PIL import Image

MAX_RANGE = 25.0
CELL = 4


def load_map(path):
    ply_t = {"float": "f4", "double": "f8", "uchar": "u1", "char": "i1", "ushort": "u2", "short": "i2",
             "uint": "u4", "int": "i4"}
    props, n = [], 0
    with open(path, "rb") as f:
        while True:
            t = f.readline().decode("ascii", "replace").split()
            if t and t[0] == "element" and t[1] == "vertex":
                n = int(t[2])
            elif t and t[0] == "property":
                props.append((t[2], ply_t[t[1]]))
            elif t and t[0] == "end_header":
                break
        off = f.tell()
    a = np.fromfile(path, dtype=np.dtype(props), count=n, offset=off)
    return np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)


def visible_luma(xyz, T, K, img_path, mask_path):
    """{indice voxel: luminance} des voxels visibles (z-buffer) et hors ciel/personne."""
    R, t = T[:3, :3], T[:3, 3]
    d = xyz - t
    near = np.flatnonzero(np.einsum("ij,ij->i", d, d) < MAX_RANGE ** 2)
    pc = d[near] @ R
    r = np.hypot(pc[:, 0], pc[:, 1]).clip(1e-9)
    th = np.arctan2(r, pc[:, 2])
    t2 = th * th
    thd = th * (1 + K["k1"] * t2 + K["k2"] * t2 ** 2 + K["k3"] * t2 ** 3 + K["k4"] * t2 ** 4)
    u = K["fx"] * thd * pc[:, 0] / r + K["cx"]
    v = K["fy"] * thd * pc[:, 1] / r + K["cy"]
    W, H = K["width"], K["height"]
    ok = (th < np.radians(70)) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx, u, v, dep = near[ok], u[ok].astype(int), v[ok].astype(int), np.linalg.norm(pc[ok], axis=1)
    cell = (v // CELL) * ((W + CELL - 1) // CELL) + u // CELL
    dmin = np.full(cell.max() + 1, np.inf)
    np.minimum.at(dmin, cell, dep)
    vis = dep <= dmin[cell] + 0.10
    idx, u, v = idx[vis], u[vis], v[vis]
    img = np.asarray(Image.open(img_path).convert("RGB"), np.float32)
    luma = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    val = luma[v, u]
    keep = (val > 15) & (val < 240)
    if os.path.exists(mask_path):
        m = np.asarray(Image.open(mask_path))
        keep &= (m[v, u] & 3) == 0
    return dict(zip(idx[keep].tolist(), val[keep].tolist()))


def pair_ratio(a, b):
    common = a.keys() & b.keys()
    if len(common) < 300:
        return None, len(common)
    lr = np.log(np.array([b[k] for k in common]) / np.array([a[k] for k in common]))
    return float(np.median(lr)), len(common)


def main():
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_local/scene")
    ap.add_argument("--raw", default="data_local/rawimgs")
    ap.add_argument("--every", type=int, default=200, help="un point de mesure tous les N triplets")
    ap.add_argument("--out", default="logs/stage4/exposure_check.json")
    args = ap.parse_args()

    xyz = load_map(f"{args.root}/map/map_5cm_static.ply")
    sensors = yaml.safe_load(open(f"{args.root}/sensors.yaml"))["cameras"]
    poses = json.load(open(f"{args.root}/poses/cameras.json"))["cameras"]
    rel = {}
    for c in ("cam0", "cam1", "cam2"):
        with open(f"{args.root}/cameras/{c}/index.csv") as f:
            rel[c] = {str(int(r["stamp_ns"])): os.path.basename(r["rel_path"]) for r in csv.DictReader(f)}
    stamps = {c: sorted(poses[c], key=int) for c in poses}
    n = min(len(v) for v in stamps.values())
    cache = {}

    def vl(c, i):
        st = stamps[c][i]
        if (c, st) not in cache:
            cache[(c, st)] = visible_luma(xyz, np.asarray(poses[c][st]["T_map_cam_start"], float), sensors[c],
                                          f"{args.raw}/{c}/{rel[c][st]}", f"{args.root}/masks/{c}/{st}.png")
        return cache[(c, st)]

    res = {"consecutif_50ms": {}, "ecart_1s": {}, "entre_cameras": []}
    for i in range(0, n - 21, args.every):
        for c in ("cam0", "cam1", "cam2"):
            r1, n1 = pair_ratio(vl(c, i), vl(c, i + 1))
            r20, n20 = pair_ratio(vl(c, i), vl(c, i + 20))
            res["consecutif_50ms"].setdefault(c, []).append(r1)
            res["ecart_1s"].setdefault(c, []).append(r20)
        for a, b in (("cam0", "cam1"), ("cam0", "cam2")):
            r, nc = pair_ratio(vl(a, i), vl(b, i))
            if r is not None:
                res["entre_cameras"].append({"paire": f"{b}/{a}", "i": i, "log_rapport": round(r, 4), "n": nc})
        cache.clear()
        print(f"triplet {i}/{n}", flush=True)

    summary = {}
    for k in ("consecutif_50ms", "ecart_1s"):
        for c, vals in res[k].items():
            v = np.array([x for x in vals if x is not None])
            summary[f"{k}_{c}"] = {"n": int(len(v)), "abs_log_p50": round(float(np.median(np.abs(v))), 4),
                                   "abs_log_p90": round(float(np.percentile(np.abs(v), 90)), 4),
                                   "max_abs_log": round(float(np.abs(v).max()), 4),
                                   "frac_ecart_gt_3pct": round(float((np.abs(v) > np.log(1.03)).mean()), 3)}
    for p in ("cam1/cam0", "cam2/cam0"):
        v = np.array([e["log_rapport"] for e in res["entre_cameras"] if e["paire"] == p])
        if len(v):
            summary[f"entre_{p}"] = {"n": int(len(v)), "gain_median": round(float(np.exp(np.median(v))), 3),
                                     "gain_p10_p90": [round(float(np.exp(np.percentile(v, 10))), 3),
                                                      round(float(np.exp(np.percentile(v, 90))), 3)]}
    print(json.dumps(summary, indent=1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"resume": summary, "detail": res}, f, indent=1)


if __name__ == "__main__":
    main()
