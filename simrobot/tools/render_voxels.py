"""Rendu des cartes de voxels du LiDAR simulé (desktop, gsplat) : chaque voxel = petite gaussienne isotrope dont
l'OPACITÉ est sa probabilité de retour, couleur = hauteur au-dessus du sol. Planche : splat | ancienne carte
(`lidar_voxels.npz`, probabilité devinée) | carte d'occupation (`lidar_occupation.npz`, mesurée en rejouant les
balayages), depuis des poses de la caméra avant GaussGym.

  PYTHONPATH=. venv/bin/python simrobot/tools/render_voxels.py --scene out/simrobot/scenes/HOB-1_full \
      --out out/simrobot/rendu_voxels.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))


def turbo(h):
    h = np.clip(h, 0, 1)[:, None]
    return np.clip(np.concatenate([1.5 - np.abs(4 * h - 3), 1.5 - np.abs(4 * h - 2), 1.5 - np.abs(4 * h - 1)], 1), 0, 1)


def static_voxels(path, T):
    from terrain import height_at

    V = np.load(path)
    lo, res, ny, z0 = V["lo"], float(V["res"]), int(V["ny"]), float(V["z0"])
    cs = V["col_start"]
    col = np.repeat(np.arange(len(cs) - 1), np.diff(cs))
    c = np.stack([lo[0] + (col // ny + 0.5) * res, lo[1] + (col % ny + 0.5) * res,
                  z0 + (V["vz"].astype(float) + 0.5) * res], 1)
    c += V["cen"].astype(float) / 127 * res / 2
    h = c[:, 2] - height_at(T, c[:, :2])
    return c.astype(np.float32), V["phit"] / 255.0, h, res


def occupancy_voxels(path, prior=None):
    """Voxels de la carte d'occupation ; opacité = 1 − exp(−μ·res) : probabilité d'arrêt d'un rayon qui traverse le
    voxel de part en part."""
    V = np.load(path)
    lo, res, ny = V["lo"], float(V["res"]), int(V["ny"])
    cs, zb = V["col_start"], V["zb"].ravel()
    col = np.repeat(np.arange(len(cs) - 1), np.diff(cs))
    k = V["vz"].astype(float)
    c = np.stack([lo[0] + (col // ny + 0.5) * res, lo[1] + (col % ny + 0.5) * res, zb[col] + (k + 0.5) * res], 1)
    below = float(V["below"]) if "below" in V else 1.0
    h = (k + 0.5) * res - below
    mu = V["hits"] / (V["length"] + prior) if (prior is not None and "hits" in V) else V["mu"].astype(np.float32)
    return c.astype(np.float32), 1 - np.exp(-mu * res), h, res


def c2w_from(p, f, u):
    f, u = np.asarray(f, float), np.asarray(u, float)
    r = np.cross(f, u)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = r, -u, f, p                 # OpenCV : x droite, y bas, z avant
    return M


def render(means, opac, colors, scale, c2w, K, W, H, dev="cuda"):
    from gsplat import rasterization

    view = np.linalg.inv(c2w)
    pc = means @ view[:3, :3].T + view[:3, 3]
    keep = (pc[:, 2] > 1.0) & (pc[:, 2] < 45) & (opac > 0.01)          # rien à moins de 1 m : pas de grosses taches
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)   # noqa: E731
    n = int(keep.sum())
    sc = np.full((n, 3), scale, np.float32) if np.ndim(scale) == 0 else scale[keep]
    q = np.zeros((n, 4), np.float32)
    q[:, 0] = 1
    img, alpha, _ = rasterization(t(means[keep]), t(q), t(sc), t(opac[keep]), t(colors[keep]), t(view[None]),
                                  t(K[None]), W, H, near_plane=0.1, far_plane=60.0)
    bg = torch.tensor([0.08, 0.08, 0.1], device=dev)
    out = img[0] + (1 - alpha[0]) * bg
    return (out.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def splat_render(ply, c2w_phys, ply_to_phys, K, W, H, dev="cuda"):
    """Splat de la scène (couleur de base seulement), caméra exprimée dans le repère du PLY."""
    from gsplat import rasterization

    with open(ply, "rb") as f:
        props, n = [], 0
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                props.append((line.split()[-1], "<f4"))
            elif line == "end_header":
                break
        G = np.fromfile(f, dtype=np.dtype(props), count=n)
    M = np.eye(4)
    M[:3, :3] = np.asarray(ply_to_phys).T                                 # physique → PLY
    view = np.linalg.inv(M @ c2w_phys)
    xyz = np.stack([G["x"], G["y"], G["z"]], 1)
    pc = xyz @ view[:3, :3].T + view[:3, 3]
    keep = (pc[:, 2] > 0.1) & (pc[:, 2] < 60)
    G = G[keep]
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)   # noqa: E731
    col = 0.5 + 0.28209479 * np.stack([G["f_dc_0"], G["f_dc_1"], G["f_dc_2"]], 1)
    img, alpha, _ = rasterization(t(xyz[keep]), t(np.stack([G[f"rot_{i}"] for i in range(4)], 1)),
                                  t(np.exp(np.stack([G[f"scale_{i}"] for i in range(3)], 1))),
                                  t(1 / (1 + np.exp(-G["opacity"]))), t(np.clip(col, 0, 1)), t(view[None]),
                                  t(K[None]), W, H, near_plane=0.1, far_plane=80.0)
    out = img[0] + (1 - alpha[0]) * torch.tensor([0.66, 0.73, 0.8], device=dev)
    return (out.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", default="2.6,8.2;35.6,43.5;-36,12", help="positions (physique) des vues")
    ap.add_argument("--prior", type=float, default=None, help="longueur a priori pour recalculer μ")
    ap.add_argument("--raise-m", type=float, default=0.0, help="vue surélevée : caméra montée et reculée de N m")
    ap.add_argument("--W", type=int, default=768)
    ap.add_argument("--H", type=int, default=512)
    a = ap.parse_args()
    from PIL import Image, ImageDraw

    meta = json.load(open(f"{a.scene}/scene.json"))
    T = dict(np.load(f"{a.scene}/terrain.npz"))
    cam = json.load(open(f"{a.scene}/trajets.json"))["cameras"]["hdr_front"]
    P, F, U = np.array(cam["p"]), np.array(cam["f"]), np.array(cam["u"])
    fy = a.H / 2 / np.tan(np.radians(cam["fov_y_deg"]) / 2) * 0.75       # un peu plus serré que la vraie caméra
    K = np.array([[fy, 0, a.W / 2], [0, fy, a.H / 2], [0, 0, 1]])
    maps = []
    if os.path.exists(f"{a.scene}/lidar_voxels.npz"):
        maps.append(("ancienne carte 10 cm (probabilité devinée)", static_voxels(f"{a.scene}/lidar_voxels.npz", T)))
    if os.path.exists(f"{a.scene}/lidar_occupation.npz"):
        maps.append(("occupation 5 cm (opacité = arrêt mesuré)",
                     occupancy_voxels(f"{a.scene}/lidar_occupation.npz", a.prior)))
    rows = []
    for xy in a.points.split(";"):
        x, y = map(float, xy.split(","))
        i = int(np.argmin(np.hypot(P[:, 0] - x, P[:, 1] - y)))
        f = F[i].copy()
        pos = P[i].copy()
        if a.raise_m:                                     # reculer, monter, regarder vers le bas de ~25°
            fh = np.array([f[0], f[1], 0.0]) / max(np.hypot(f[0], f[1]), 1e-6)
            pos = pos - a.raise_m * fh + np.array([0, 0, a.raise_m])
            f = np.cos(np.radians(25)) * fh + np.array([0, 0, -np.sin(np.radians(25))])
        u = np.cross(np.cross(f, [0, 0, 1.0]), f)
        c2w = c2w_from(pos, f, u / np.linalg.norm(u))
        tiles = [("splat", splat_render(f"{a.scene}/merged.ply", c2w, meta["ply_to_physics"], K, a.W, a.H))]
        for name, (c, p, h, res) in maps:
            tiles.append((name, render(c, p.astype(np.float32), turbo((h + 0.3) / 6).astype(np.float32),
                                       res * 0.3, c2w, K, a.W, a.H)))
        rows.append(tiles)
        torch.cuda.empty_cache()
    gap = 6
    out = Image.new("RGB", (len(rows[0]) * (a.W + gap), len(rows) * (a.H + gap) + 30), (20, 20, 24))
    d = ImageDraw.Draw(out)
    from PIL import ImageFont

    font = None
    for fp in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, 18)                       # accents (police par défaut de PIL : sans)
            break
    for r, tiles in enumerate(rows):
        for cidx, (name, im) in enumerate(tiles):
            out.paste(Image.fromarray(im), (cidx * (a.W + gap), 30 + r * (a.H + gap)))
            if r == 0:
                d.text((cidx * (a.W + gap) + 8, 5), name, fill=(240, 240, 240), font=font)
    out.save(a.out, quality=88)
    print(a.out, out.size)


if __name__ == "__main__":
    main()
