"""Terrain de collision tiré DU SPLAT (desktop, gsplat) — même géométrie que ce qui est affiché.

1. Surface vue : profondeur du splat rendue depuis les poses RÉELLES des caméras GaussGym (les seules d'où sa
   géométrie est fiable), pixels opaques (opacité accumulée ≥ `--alpha-min`) à moins de `--max-range` m, projetés en
   3D. (Échec documenté : un rendu orthographique vu d'en dessous trouve une « surface » 4–10 m SOUS le sol — le
   splat contient des gaussiennes parasites sous le sol, invisibles des caméras mais opaques pour un rayon montant.)
2. Sol : par cellule de 10 cm, quantile bas (`--q-sol`) des hauteurs des points vus.
3. Obstacles : cellule où une part importante des points vus est entre `--obs-zmin` et `--obs-zmax` au-dessus du
   sol (tronc, souche, rocher) ; l'herbe et le lierre restent sous `--obs-zmin`.
3. Trous (hors splat) : bouchés par diffusion (inpainting), pas au plus proche, pour ne pas créer de plateaux.

Sortie `terrain_splat.npz` (même format que `terrain.npz` : H, lo, res, couvert + obstacle, sol_brut) et une planche
de comparaison avec le terrain NKSR.

  PYTHONPATH=. venv/bin/python simrobot/sim/splat_terrain.py --scene out/simrobot/scenes/HOB-1_5 \
      --tiles data_local/simrobot/grandtour/HOB-1/*
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from gaussgym_check import read_splat_ply  # noqa: E402
from gaussgym_export import FLIP, rotate_all  # noqa: E402

R_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
DEPTH_BINS = [0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.5, 10.0, 12.0]


def main():
    import cv2
    import torch
    from gsplat import rasterization

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="dossier contenant scene.ply (y haut, recentré) et terrain.npz")
    ap.add_argument("--res", type=float, default=0.1)
    ap.add_argument("--tiles", nargs="+", required=True, help="tuiles GaussGym (poses des caméras de capture)")
    ap.add_argument("--every", type=int, default=3, help="une image sur N")
    ap.add_argument("--depth", choices=["median", "expected"], default="median")
    ap.add_argument("--tag", default="", help="suffixe des fichiers de sortie")
    ap.add_argument("--down", type=int, default=4, help="sous-échantillonnage des images rendues")
    ap.add_argument("--alpha-min", type=float, default=0.9)
    ap.add_argument("--max-range", type=float, default=12.0)
    ap.add_argument("--min-points", type=int, default=3)
    ap.add_argument("--q-sol", type=float, default=0.15)
    ap.add_argument("--obs-zmin", type=float, default=0.25)
    ap.add_argument("--obs-zmax", type=float, default=1.25)
    ap.add_argument("--obs-frac", type=float, default=0.5)
    ap.add_argument("--obs-height", type=float, default=1.0, help="hauteur des piliers d'obstacle dans le hfield")
    a = ap.parse_args()
    dev = "cuda"

    meta = json.load(open(f"{a.scene}/scene.json"))
    G = read_splat_ply(f"{a.scene}/scene.ply")
    rotate_all(G, np.array(meta["ply_to_physics"]), np.zeros(3))   # PLY → repère physique (monde recentré, z haut)
    old = dict(np.load(f"{a.scene}/terrain.npz"))
    lo = old["lo"].astype(float)
    res = a.res
    nx = int(round(old["H"].shape[0] * float(old["res"]) / res))
    ny = int(round(old["H"].shape[1] * float(old["res"]) / res))

    # ---------------- 1. surface vue depuis les caméras de capture
    centre = np.array(meta["centre_monde_m"])
    F4 = np.eye(4)
    F4[:3, :3] = FLIP
    tiles = [t.rstrip("/") for t in a.tiles if os.path.exists(f"{t.rstrip('/')}/transforms.json")]
    frames = [f for t in tiles for f in json.load(open(f"{t}/transforms.json"))["frames"]]
    frames = frames[:: a.every]
    S = {k: torch.tensor(v, device=dev, dtype=torch.float32) for k, v in G.items() if k != "sh"}
    S["quats"] = S["quats"] / S["quats"].norm(dim=1, keepdim=True)
    gray = torch.full((len(S["means"]), 3), 0.5, device=dev)
    pts = []
    for f in frames:
        W, Hh = int(f["w"]) // a.down, int(f["h"]) // a.down
        K = torch.tensor([[float(f["fl_x"]) / a.down, 0, float(f["cx"]) / a.down],
                          [0, float(f["fl_y"]) / a.down, float(f["cy"]) / a.down], [0, 0, 1]], device=dev)[None]
        T = F4 @ np.array(f["transform_matrix"], float)        # mission (z bas) → monde
        T[:3, 1:3] *= -1                                       # OpenGL → OpenCV
        T[:3, 3] -= centre                                     # monde recentré
        vm = torch.tensor(np.linalg.inv(T), device=dev, dtype=torch.float32)[None]
        img, alpha, _ = rasterization(S["means"], S["quats"], S["scales"], S["opac"], gray, vm, K, W, Hh,
                                      render_mode="RGB+ED", near_plane=0.05, far_plane=100.0)
        d = img[0, ..., 3]
        if a.depth == "median":
            # profondeur MÉDIANE (première traversée de 50 % d'opacité accumulée) par « pelage » : opacité rendue avec
            # des plans lointains croissants (gaussiennes coupées sur leur centre). La profondeur ATTENDUE (moyenne
            # pondérée) traverse un sol semi-transparent vu en rasant et ramasse ce qu'il y a derrière.
            prev_a = torch.zeros_like(d)
            prev_d = 0.05
            med = torch.full_like(d, float("inf"))
            for dk in DEPTH_BINS:
                _, al, _ = rasterization(S["means"], S["quats"], S["scales"], S["opac"], gray, vm, K, W, Hh,
                                         render_mode="RGB", near_plane=0.05, far_plane=dk)
                ak = al[0, ..., 0]
                cross = torch.isinf(med) & (ak >= 0.5)
                t = ((0.5 - prev_a) / (ak - prev_a).clamp(min=1e-6)).clamp(0, 1)
                med = torch.where(cross, prev_d + t * (dk - prev_d), med)
                prev_a, prev_d = ak, dk
            d = med
        ok = (alpha[0, ..., 0] >= a.alpha_min) & (d < a.max_range) & (d > 0.1)
        v, u = torch.nonzero(ok, as_tuple=True)
        dd = d[v, u]
        Kc = K[0]
        xc = torch.stack([(u + 0.5 - Kc[0, 2]) / Kc[0, 0] * dd, (v + 0.5 - Kc[1, 2]) / Kc[1, 1] * dd, dd], 1)
        Tt = torch.tensor(T, device=dev, dtype=torch.float32)
        pts.append((xc @ Tt[:3, :3].T + Tt[:3, 3]).cpu().numpy())
    P = np.concatenate(pts)
    ij = np.floor((P[:, :2] - lo) / res).astype(int)
    inb = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
    P, ij = P[inb], ij[inb]
    cell = ij[:, 0] * ny + ij[:, 1]

    # ---------------- 2. sol : quantile bas par cellule
    order = np.lexsort((P[:, 2], cell))
    cs, zs = cell[order], P[order, 2]
    cnt = np.bincount(cs, minlength=nx * ny)
    first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    has = cnt >= a.min_points
    sol_brut = np.full(nx * ny, np.nan, np.float32)
    sol_brut[has] = zs[first[has] + np.floor(a.q_sol * (cnt[has] - 1)).astype(int)]
    sol_brut = sol_brut.reshape(nx, ny)
    valide = has.reshape(nx, ny)

    # ---------------- 3. obstacles : part des points vus dans la bande [obs_zmin, obs_zmax] au-dessus du sol
    hgt = P[:, 2] - np.nan_to_num(sol_brut).ravel()[cell]
    band = (hgt > a.obs_zmin) & (hgt < a.obs_zmax)
    n_band = np.bincount(cell[band], minlength=nx * ny).reshape(nx, ny)
    frac_band = n_band / np.maximum(cnt.reshape(nx, ny), 1)
    obstacle = valide & (n_band >= 2 * a.min_points) & (frac_band >= a.obs_frac)
    obstacle = cv2.morphologyEx(obstacle.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    acc = cnt.reshape(nx, ny).astype(np.float32)

    # ---------------- 3. hauteur finale : sol lissé (médiane 3×3), trous bouchés par diffusion, piliers
    sol = sol_brut.copy()
    sol[~valide] = np.nan
    med = cv2.medianBlur(np.nan_to_num(sol, nan=0).astype(np.float32), 3)
    sol = np.where(valide, med, np.nan)
    zmin, zmax = np.nanmin(sol), np.nanmax(sol)
    norm = np.where(valide, (sol - zmin) / (zmax - zmin), 0).astype(np.float32)
    filled = cv2.inpaint((norm * 65535).astype(np.uint16).astype(np.float32), (~valide).astype(np.uint8), 15,
                         cv2.INPAINT_TELEA) / 65535 * (zmax - zmin) + zmin
    H = np.where(valide, sol, filled).astype(np.float32)
    H_obs = np.where(obstacle, H + a.obs_height, H).astype(np.float32)

    np.savez_compressed(f"{a.scene}/terrain_splat{a.tag}.npz", H=H_obs, lo=lo, res=res, couvert=valide, obstacle=obstacle,
                        sol=H, sol_brut=sol_brut, n_points=acc)

    # ---------------- comparaison avec le terrain NKSR (même grille si même résolution)
    rep = {"grille": [nx, ny], "res_m": res, "frac_sol_splat_valide": round(float(valide.mean()), 3),
           "frac_obstacles": round(float(obstacle.mean()), 4)}
    if old["H"].shape == H.shape:
        d = old["H"] - H
        cov_nksr = old["couvert"].astype(bool)
        for name, m in [("nksr_couvert", cov_nksr & valide), ("nksr_bouche", ~cov_nksr & valide)]:
            if m.any():
                rep[f"ecart_nksr_moins_splat_{name}_cm"] = {
                    "mediane": round(float(np.median(d[m])) * 100, 1),
                    "p10": round(float(np.percentile(d[m], 10)) * 100, 1),
                    "p90": round(float(np.percentile(d[m], 90)) * 100, 1),
                    "frac_au_dessus_de_10cm": round(float((d[m] > 0.10).mean()), 3),
                    "frac_au_dessous_de_10cm": round(float((d[m] < -0.10).mean()), 3)}
        # planche : écart NKSR − splat (rouge = NKSR au-dessus = « flotte »), sol splat, obstacles
        lim = 0.5
        e = np.clip(d / lim, -1, 1)
        rgb = np.zeros(d.shape + (3,), np.uint8)
        rgb[..., 2] = (255 * np.clip(e, 0, 1)).astype(np.uint8)
        rgb[..., 0] = (255 * np.clip(-e, 0, 1)).astype(np.uint8)
        rgb[..., 1] = (60 * cov_nksr).astype(np.uint8)
        rgb[~valide] = 40
        hs = cv2.applyColorMap((255 * (H - zmin) / (zmax - zmin)).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        hs[obstacle] = (255, 255, 255)
        up = lambda im: cv2.resize(np.flipud(np.transpose(im, (1, 0, 2))), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        a1, a2 = up(rgb), up(hs)
        cv2.putText(a1, "NKSR - splat (rouge = flotte > 0, bleu = enfoui ; +-50 cm)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(a2, "sol du splat (blanc = obstacles)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(f"{a.scene}/terrain_comparaison{a.tag}.jpg", np.concatenate([a1, a2], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(f"{a.scene}/terrain_splat{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
