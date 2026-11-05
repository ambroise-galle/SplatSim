"""Couleur initiale de chaque voxel de la carte, relevée dans les photos (GPU).

Remplace l'initialisation v2 par `superpoints_colored.ply`, qui est une palette ALÉATOIRE de visualisation des
superpoints (`superpoints.py:palette`), pas une couleur d'image.

Pour un sous-échantillon de triplets (`--stride`, sur les triplets comme M3) et toutes les caméras de `sensors.yaml` :
  1. voxels à ≤ `--max-range` m, projetés (`common/train_camera.py` : fisheye équidistant OpenCV k1–k4 à Oxford,
     sténopé redressé sur DiTer++ ; poses `T_map_cam` caméra → carte : p_map = R·p_cam + t) ;
  2. z-buffer du nuage lui-même en cellules de 4×4 px, tolérance 10 cm (même règle que M2) ;
  3. rejets : incidence cos < 0,25, pixels ciel/personne, angle hors du domaine gsplat ;
  4. accumulation de la couleur du pixel, pondérée par le cos d'incidence.
Sortie `reports/stage4/map_colors_lift.npz` : `rgb` (N, 3) float16 dans l'ordre des lignes du PLY de la carte,
`weight` (N,) float32 ; voxels jamais vus → gris 0,5 (weight = 0).

Desktop, depuis ~/dino :
  venv/bin/python -m splatpipe.train.color_lift --root data_local/scene --stride 6
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

ZCELL = 4
DEPTH_TOL = 0.10
COS_MIN = 0.25


def gsplat_max_angle(cam: dict) -> float:
    """Angle maximal accepté par la projection fisheye de gsplat (Cameras.cuh) : min(racine de dθ_d/dθ,
    rayon pixel max / focale). Au-delà, gsplat ne rend rien (coins noirs)."""
    k = [cam["k1"], cam["k2"], cam["k3"], cam["k4"]]
    th = np.linspace(0, 1.57, 20000)
    d = 1 + 3 * k[0] * th ** 2 + 5 * k[1] * th ** 4 + 7 * k[2] * th ** 6 + 9 * k[3] * th ** 8
    root = th[np.argmax(d <= 0)] if (d <= 0).any() else np.inf
    mx = max(cam["width"] - cam["cx"], cam["cx"])
    my = max(cam["height"] - cam["cy"], cam["cy"])
    rmax = np.hypot(mx, my)
    return float(min(root, max(rmax / cam["fx"], rmax / cam["fy"])))


def gsplat_valid_mask(cam: dict) -> np.ndarray:
    """Pixels (H, W) dont l'angle est dans le domaine gsplat (inverse de la distorsion par table)."""
    k = [cam["k1"], cam["k2"], cam["k3"], cam["k4"]]
    th = np.linspace(0, 1.6, 20000)
    thd = th * (1 + k[0] * th ** 2 + k[1] * th ** 4 + k[2] * th ** 6 + k[3] * th ** 8)
    H, W = cam["height"], cam["width"]
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rd = np.hypot((u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"])
    theta = np.interp(rd, thd, th)
    return theta < gsplat_max_angle(cam) * 0.995


def main():
    import torch
    import yaml
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from splatpipe.stage4.overfit_core import _ply_header

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--max-range", type=float, default=40.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    out = args.out or f"{args.root}/reports/stage4/map_colors_lift.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    off, n, props = _ply_header(f"{args.root}/map/map_5cm_static.ply")
    dt = np.dtype([(name, "<" + t) for name, t in props])
    mm = np.memmap(f"{args.root}/map/map_5cm_static.ply", dtype=dt, mode="r", offset=off, shape=(n,))
    xyz = torch.from_numpy(np.stack([mm["x"], mm["y"], mm["z"]], 1).astype(np.float32)).to(dev)
    nrm = torch.from_numpy(np.stack([mm["nx"], mm["ny"], mm["nz"]], 1).astype(np.float32)).to(dev)
    nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp(min=1e-9)
    del mm
    acc = torch.zeros(n, 3, device=dev)
    wsum = torch.zeros(n, device=dev)

    sensors = yaml.safe_load(open(f"{args.root}/sensors.yaml"))["cameras"]
    poses = json.load(open(f"{args.root}/poses/cameras.json"))["cameras"]
    cams = sorted(poses)
    rel = {}
    for c in cams:
        with open(f"{args.root}/cameras/{c}/index.csv") as f:
            rel[c] = {str(int(r["stamp_ns"])): os.path.basename(r["rel_path"]) for r in csv.DictReader(f)}
    stamps = {c: sorted(poses[c], key=int) for c in cams}
    n_trip = min(len(stamps[c]) for c in cams)
    sel = range(0, n_trip, args.stride)
    from splatpipe.common.train_camera import make_camera

    camobj = {c: make_camera(sensors[c], args.root) for c in cams}
    n_img = 0
    for i in sel:
        for c in cams:
            st = stamps[c][i]
            p_img = f"{args.root}/rawimgs/{c}/{rel[c].get(st, '')}"
            if st not in rel[c] or not os.path.exists(p_img):
                continue
            K = sensors[c]
            W, H = K["width"], K["height"]
            T = torch.tensor(np.asarray(poses[c][st]["T_map_cam_start"], np.float64), dtype=torch.float32, device=dev)
            R, t = T[:3, :3], T[:3, 3]
            d = xyz - t
            near = (d * d).sum(1) < args.max_range ** 2
            idx = torch.nonzero(near).squeeze(1)
            pc = d[idx] @ R                                     # carte -> caméra (R = caméra -> carte)
            depth = pc.norm(dim=1)
            u, v, ok = camobj[c].project(pc)                    # fisheye borné au domaine gsplat, ou sténopé
            cosi = (nrm[idx] * (-d[idx] / depth[:, None])).sum(1).abs()
            ok &= cosi >= COS_MIN
            idx, u, v, depth, cosi = idx[ok], u[ok], v[ok], depth[ok], cosi[ok]
            ui, vi = u.long(), v.long()
            Wc = (W + ZCELL - 1) // ZCELL
            cell = (vi // ZCELL) * Wc + ui // ZCELL
            dmin = torch.full((((H + ZCELL - 1) // ZCELL) * Wc,), float("inf"), device=dev)
            dmin.scatter_reduce_(0, cell, depth, reduce="amin")
            vis = depth <= dmin[cell] + DEPTH_TOL
            idx, ui, vi, cosi = idx[vis], ui[vis], vi[vis], cosi[vis]
            img = torch.from_numpy(np.asarray(Image.open(p_img).convert("RGB"))).to(dev)
            mp = f"{args.root}/masks/{c}/{st}.png"
            if os.path.exists(mp):
                m = torch.from_numpy(np.asarray(Image.open(mp))).to(dev)
                keep = (m[vi, ui] & 3) == 0
                idx, ui, vi, cosi = idx[keep], ui[keep], vi[keep], cosi[keep]
            col = img[vi, ui].float() / 255.0
            acc.index_add_(0, idx, col * cosi[:, None])
            wsum.index_add_(0, idx, cosi)
            n_img += 1
        if (i // args.stride) % 100 == 0:
            print(f"triplet {i}/{n_trip} : {n_img} images, voxels vus {(wsum > 0).float().mean().item():.3f}, "
                  f"{time.time() - t0:.0f} s", flush=True)
    rgb = torch.where(wsum[:, None] > 0, acc / wsum.clamp(min=1e-9)[:, None], torch.full_like(acc, 0.5))
    np.savez_compressed(out, rgb=rgb.cpu().numpy().astype(np.float16), weight=wsum.cpu().numpy())
    rep = {"n_images": n_img, "stride": args.stride, "frac_voxels_vus": round(float((wsum > 0).float().mean()), 4),
           "cameras": {c: camobj[c].model for c in cams},
           "s": round(time.time() - t0, 1), "out": out}
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
