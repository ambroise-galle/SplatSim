"""Comparaison B0 (splat « images seules », gsplat simple_trainer MCMC) contre nos splats (LiDAR + images).

Mêmes vues de test pour tous : le split `val` du parseur COLMAP de gsplat (une image sur 8). ATTENTION : pour un
modèle OPENCV_FISHEYE, ce parseur ne redresse PAS en sténopé ; il retire le polynôme k1–k4 et produit un fisheye
ÉQUIDISTANT idéal (même K). B0 doit donc être entraîné avec `--camera-model fisheye`, et tous les modèles sont rendus
ici en fisheye équidistant sans distorsion. Le parseur d'origine recentre en plus l'image sur (W/2, H/2) au lieu du
point principal (cx, cy) : ~20 px de décalage ici ; corrigé dans la copie locale (`datasets/colmap.py`, lignes mapx/mapy).
- Image : PSNR brut, PSNR aligné (gain + biais par canal ajustés, neutralise l'exposition), SSIM, LPIPS (alex),
  sur les pixels valides (ROI du redressement, hors ciel et hors personne, masques remappés).
- Géométrie : balayages LiDAR holdout de la fenêtre centrée sur la zone (jamais vus par nos splats ni par B0),
  distance rendue (RGB-Ed, modèle de faisceaux QT64) contre mesurée : fraction à ≤ 10 cm, médiane.

Instance, depuis ~/gsplat/examples (le parseur COLMAP de gsplat est importé d'ici) :
  ~/venv/bin/python ~/dino/splatpipe/train/compare_b0.py --colmap ~/b0_cour --root ~/dino/data_local/scene \
      --b0 ~/dino/out/b0_cour_mcmc/ckpts/ckpt_29999_rank0.pt --ours v3=~/dino/out/train_full_v3/scene_full.npz \
      --window 10 --out ~/dino/out/compare_b0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/dino"))
for _ex in ("~/dino/gsplat-src/examples", "~/gsplat/examples"):          # desktop, puis instance Thunder
    if os.path.isdir(os.path.expanduser(_ex)):
        sys.path.insert(0, os.path.expanduser(_ex))
        break


def load_b0(path, dev):
    import torch

    sp = torch.load(path, map_location=dev, weights_only=False)["splats"]
    colors = torch.cat([sp["sh0"], sp["shN"]], 1)
    return dict(means=sp["means"], quats=sp["quats"], scales=torch.exp(sp["scales"]),
                opac=torch.sigmoid(sp["opacities"]), colors=colors,
                sh_degree=int(round(np.sqrt(colors.shape[1]))) - 1, ut=False)


def load_ours(path, dev):
    import torch

    z = np.load(path)
    t = {k: torch.from_numpy(z[k].astype(np.float32)).to(dev) for k in ("means", "quats", "scales", "opac", "colors")}
    return dict(means=t["means"], quats=t["quats"], scales=t["scales"], opac=t["opac"], colors=t["colors"],
                sh_degree=int(round(np.sqrt(t["colors"].shape[1]))) - 1, ut=True)


def render_pinhole(S, viewmat, K, W, H, camera_model="fisheye"):
    """Rendu dans le modèle des images du parseur gsplat : fisheye équidistant idéal (Oxford, nom historique) ou
    sténopé (images déjà redressées : DiTer++, BotanicGarden)."""
    import gsplat

    out, alpha, _ = gsplat.rasterization(
        S["means"], S["quats"], S["scales"], S["opac"], S["colors"], viewmat[None], K[None], W, H,
        sh_degree=S["sh_degree"], packed=False, with_ut=S["ut"], with_eval3d=S["ut"], far_plane=300.0,
        camera_model=camera_model)
    return out[0].clamp(0, 1), alpha[0, ..., 0]


def main():
    import cv2
    import torch
    import yaml
    from PIL import Image

    import gsplat
    from datasets.colmap import Dataset, Parser
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    from splatpipe.stage4.overfit_core import _load_lidar_model_root
    from splatpipe.train.data import WindowData
    from splatpipe.train.full import _ssim_map

    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--b0", nargs="+", required=True, help="chemin du ckpt gsplat, ou nom=chemin (plusieurs)")
    ap.add_argument("--camera-model", default="fisheye", choices=["fisheye", "pinhole"])
    ap.add_argument("--no-lidar", action="store_true", help="pas d'évaluation LiDAR (hors Oxford)")
    ap.add_argument("--ours", nargs="*", default=[], help="nom=chemin scene_full.npz")
    ap.add_argument("--window", type=int, default=10, help="fenêtre dont on prend les balayages holdout")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-every", type=int, default=20)
    args = ap.parse_args()
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)

    # chargement paresseux, une scène à la fois (GPU 8 Go de la desktop)
    specs = {}
    for item in args.b0:
        name, p = item.split("=", 1) if "=" in item else ("B0_rgb_seul", item)
        specs[name] = ("b0", os.path.expanduser(p))
    for item in args.ours:
        name, p = item.split("=", 1)
        specs[name] = ("ours", os.path.expanduser(p))
    loaded = {}

    def get(mname):
        if mname not in loaded:
            loaded.clear()
            torch.cuda.empty_cache()
            kind, path = specs[mname]
            loaded[mname] = load_b0(path, dev) if kind == "b0" else load_ours(path, dev)
        return loaded[mname]
    models = specs

    # ------------------------------------------------------------------ image
    parser = Parser(data_dir=os.path.expanduser(args.colmap), factor=1, normalize=False, test_every=8)
    val = Dataset(parser, split="val")
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(dev)
    img_res = {m: [] for m in models}
    for mname in models:
      S = get(mname)
      for j in range(len(val)):
        d = val[j]
        name = parser.image_names[val.indices[j]]
        cam, st = name[:-4].rsplit("_", 1)                  # « cam0_<stamp> » ou « rgb_left_<stamp> »
        cid = parser.camera_ids[val.indices[j]]
        img = d["image"].to(dev).float() / 255.0
        H, W = img.shape[:2]
        K = d["K"].to(dev).float()
        viewmat = torch.linalg.inv(d["camtoworld"].to(dev).float())
        valid = d["mask"].to(dev) if "mask" in d else torch.ones(H, W, dtype=torch.bool, device=dev)
        vm = f"{args.root}/cameras/{cam}/valid_mask.png"       # images redressées : bords hors image d'origine
        if os.path.exists(vm):
            valid = valid & torch.from_numpy(np.asarray(Image.open(vm)) > 0).to(dev)
        mp = f"{args.root}/masks/{cam}/{st}.png"
        if os.path.exists(mp):                 # masque ciel/personne remappé comme l'image
            m = np.asarray(Image.open(mp))
            if cid in parser.mapx_dict:
                m = cv2.remap(m, parser.mapx_dict[cid], parser.mapy_dict[cid], cv2.INTER_NEAREST)
                x, y, w, h = parser.roi_undist_dict[cid]
                m = m[y:y + h, x:x + w]
            valid = valid & torch.from_numpy((m & 3) == 0).to(dev)
        if True:
            with torch.no_grad():
                rgb, _ = render_pinhole(S, viewmat, K, W, H, args.camera_model)
            r, g = rgb[valid], img[valid]
            raw = float(-10 * torch.log10(((r - g) ** 2).mean()))
            a = ((r - r.mean(0)) * (g - g.mean(0))).mean(0) / r.var(0, unbiased=False).clamp(min=1e-8)
            fit_full = (a * (rgb - r.mean(0)) + g.mean(0)).clamp(0, 1)
            al = float(-10 * torch.log10(((fit_full[valid] - g) ** 2).mean()))
            ssim = float(_ssim_map(fit_full.permute(2, 0, 1), img.permute(2, 0, 1))[valid].mean())
            comp = torch.where(valid[..., None], fit_full, img)      # pixels invalides = vérité (neutres)
            lp = float(lpips(comp.permute(2, 0, 1)[None], img.permute(2, 0, 1)[None]))
            img_res[mname].append(dict(name=name, psnr_brut=raw, psnr_aligne=al, ssim=ssim, lpips=lp))
            if j % args.save_every == 0:
                Image.fromarray((fit_full.cpu().numpy() * 255).astype(np.uint8)).save(f"{args.out}/{mname}_{name}")
        if j % args.save_every == 0 and mname == list(models)[0]:
            Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).save(f"{args.out}/photo_{name}")

    # ------------------------------------------------------------------ LiDAR
    lid_res = {m: [] for m in models}
    lidar_model = None if args.no_lidar else _load_lidar_model_root(args.root, dev)[0]
    if not args.no_lidar:
        data = WindowData(args.root, sorted(int(x) for x in
                          yaml.safe_load(open(f"{args.root}/splits.yaml"))["lidars"]["hesai_qt64"]["train"]), holdout_every=0)
        stamps = [s for s in data.stamps if os.path.exists(f"{args.root}/lidar/hesai_deskewed/{s}.npz")]
        centers = np.load(os.path.expanduser("~/dino/out/window_centers.npy"))
        orig = data.origins(np.array([data.sweep_light(s) for s in stamps]))
        win = sorted(s for s, o in zip(stamps, orig) if np.linalg.norm(o - centers[args.window]) <= 20.0)
        hold = win[::10]                                   # même règle que full.py : jamais optimisés
        for mname in models:
         S = get(mname)
         for st in hold:
            sw = data.sweep(st)
            f = sw["first"]
            rows, cols, rng_m = sw["row"][f], sw["col"][f], sw["rng"][f]
            keep = rng_m <= 28.0
            if sw["dyn_pt"] is not None:
                keep &= ~sw["dyn_pt"][f]
            T0 = torch.tensor(np.linalg.inv(sw["T0"]), dtype=torch.float32, device=dev)
            T1 = torch.tensor(np.linalg.inv(sw["T1"]), dtype=torch.float32, device=dev)
            if True:
                with torch.no_grad():
                    rend, alph, _ = gsplat.rasterization(
                        S["means"], S["quats"], S["scales"], S["opac"], S["colors"][:, :1], T0[None],
                        torch.eye(3, device=dev)[None], lidar_model.n_columns, lidar_model.n_rows,
                        camera_model="lidar", lidar_coeffs=lidar_model, with_ut=True, with_eval3d=True, packed=False,
                        render_mode="RGB-Ed", global_z_order=False, far_plane=300.0, sh_degree=0,
                        rolling_shutter=gsplat.RollingShutterType.ROLLING_LEFT_TO_RIGHT, viewmats_rs=T1[None])
                dr = rend[0, ..., 3].double().cpu().numpy()[rows[keep], cols[keep]]
                ar = alph[0, ..., 0].double().cpu().numpy()[rows[keep], cols[keep]]
                ok = (ar > 0.5) & np.isfinite(dr) & (dr < 100)
                err = dr[ok] - rng_m[keep][ok]
                lid_res[mname].append(dict(stamp=int(st), n=int(keep.sum()), frac_retour=float(ok.mean()),
                                           apparies=float((np.abs(err) <= 0.10).sum() / max(keep.sum(), 1)),
                                           mediane_cm=float(np.median(np.abs(err[np.abs(err) <= 0.10])) * 100)
                                           if (np.abs(err) <= 0.10).any() else None))

    summary = {}
    for m in models:
        ir, lr = img_res[m], lid_res[m]
        summary[m] = {
            "n_images_test": len(ir),
            "psnr_brut": round(float(np.mean([x["psnr_brut"] for x in ir])), 2),
            "psnr_aligne": round(float(np.mean([x["psnr_aligne"] for x in ir])), 2),
            "ssim": round(float(np.mean([x["ssim"] for x in ir])), 4),
            "lpips": round(float(np.mean([x["lpips"] for x in ir])), 4),
            "n_gaussiennes": int(np.load(specs[m][1])["means"].shape[0]) if specs[m][0] == "ours" else None,
            "lidar_n_balayages": len(lr),
            "lidar_apparies_10cm": round(float(np.mean([x["apparies"] for x in lr])), 4) if lr else None,
            "lidar_mediane_appariee_cm": round(float(np.nanmean([x["mediane_cm"] for x in lr if x["mediane_cm"]])), 2) if lr else None,
        }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    with open(f"{args.out}/compare.json", "w") as f:
        json.dump({"resume": summary, "images": img_res, "lidar": lid_res}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
