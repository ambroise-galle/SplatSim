"""Previews photo | rendu | opacité cumulée pour une scène entraînée (ou la carte initiale).

Les valeurs du npz sont passées telles quelles à gsplat (échelles et opacités « activées ») : le même script
lit les scènes v2 (paramètres bruts, bogués) et v3 (activés à l'écriture), ce qui permet la comparaison.

PSNR (pixels hors ciel et hors personne) : « brut » = rendu tel quel ; « aligné » = après ajustement
gain/biais par canal au sens des moindres carrés, ce qui neutralise l'exposition automatique de la
caméra (la scène ne connaît pas l'exposition d'une image non vue).

Desktop, depuis ~/dino :
  venv/bin/python -m splatpipe.train.preview --root data_local/scene --scene out/train_full_v2/scene_full.npz \
      --out out/preview_v2
  venv/bin/python -m splatpipe.train.preview --root data_local/scene --init-map --out out/preview_init
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

C0 = 0.28209479177387814
# poses de contrôle de GLM (cam0) : début de marche, cour intérieure, et une 3e plus loin
DEFAULT_STAMPS = "cam0:1710338128692856234,cam0:1710338243391100552,cam0:1710338315390039163"
SKY_RGB = (0.80, 0.83, 0.88)


def load_scene(args, dev):
    import torch

    if args.init_map:
        from splatpipe.stage4.overfit_core import _load_map_disks
        from splatpipe.train.full import _load_voxel_colors

        means, quats, scales, opac, _ = _load_map_disks(dev, f"{args.root}/map/map_5cm_static.ply")
        col = _load_voxel_colors(f"{args.root}/reports/stage2_material/superpoints_colored.ply")
        if col is None or len(col) != len(means):
            col = np.full((len(means), 3), 0.5, np.float32)
        colors = torch.from_numpy(((col - 0.5) / C0)[:, None, :].astype(np.float32)).to(dev)
        return means, quats, scales, opac, colors, 0
    z = np.load(args.scene)
    t = {k: torch.from_numpy(z[k].astype(np.float32)).to(dev) for k in ("means", "quats", "scales", "opac", "colors")}
    colors = t["colors"]
    if colors.dim() == 2:
        colors = colors[:, None, :]
    sh_degree = int(round(np.sqrt(colors.shape[1]))) - 1
    return t["means"], t["quats"], t["scales"], t["opac"], colors, sh_degree


def aligned_psnr(rgb, img, valid):
    """PSNR après gain/biais par canal ajustés aux moindres carrés sur les pixels valides."""
    out = []
    r, g = rgb[valid], img[valid]
    for ch in range(3):
        A = np.stack([r[:, ch], np.ones(len(r))], 1)
        coef, *_ = np.linalg.lstsq(A, g[:, ch], rcond=None)
        out.append(A @ coef)
    fit = np.clip(np.stack(out, 1), 0, 1)
    mse = ((fit - g) ** 2).mean()
    return float(-10 * np.log10(max(mse, 1e-12)))


def main():
    import torch
    import yaml
    from PIL import Image, ImageDraw

    import gsplat
    from splatpipe.common.train_camera import make_camera

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scene", default="")
    ap.add_argument("--init-map", action="store_true")
    ap.add_argument("--stamps", default=DEFAULT_STAMPS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)

    means, quats, scales, opac, colors, sh_degree = load_scene(args, dev)
    sensors = yaml.safe_load(open(f"{args.root}/sensors.yaml"))
    poses = json.load(open(f"{args.root}/poses/cameras.json"))["cameras"]
    rel = {}
    report = []
    for item in args.stamps.split(","):
        cam, st = item.split(":")
        if cam not in rel:
            import csv
            with open(f"{args.root}/cameras/{cam}/index.csv") as f:
                rel[cam] = {str(int(r["stamp_ns"])): os.path.basename(r["rel_path"]) for r in csv.DictReader(f)}
        K = make_camera(sensors["cameras"][cam], args.root)          # fisheye (Oxford) ou sténopé (DiTer++)
        T_wc = np.asarray(poses[cam][st]["T_map_cam_start"], float)
        viewmat = torch.tensor(np.linalg.inv(T_wc), dtype=torch.float32, device=dev)[None]
        with torch.no_grad():
            out, alpha, _ = gsplat.rasterization(
                means, quats, scales, opac, colors, viewmat, K.K_tensor(dev), K.width, K.height,
                with_ut=True, with_eval3d=True, packed=False, render_mode="RGB", far_plane=300.0,
                sh_degree=sh_degree, **K.gsplat_kwargs(dev))
        rgb = out[0].clamp(0, 1).cpu().numpy()
        a = alpha[0, ..., 0].cpu().numpy()
        img = np.asarray(Image.open(f"{args.root}/rawimgs/{cam}/{rel[cam][st]}").convert("RGB"), np.float32) / 255
        msk_p = f"{args.root}/masks/{cam}/{st}.png"
        m = np.asarray(Image.open(msk_p)) if os.path.exists(msk_p) else np.zeros(a.shape, np.uint8)
        valid = ((m & 3) == 0) & K.valid_mask()
        mse = ((rgb[valid] - img[valid]) ** 2).mean()
        row = {"cam": cam, "stamp": st, "psnr_brut": round(float(-10 * np.log10(mse)), 2),
               "psnr_aligne": round(aligned_psnr(rgb, img, valid), 2),
               "alpha_moy_hors_ciel": round(float(a[valid].mean()), 3),
               "frac_alpha_lt_0.5_hors_ciel": round(float((a[valid] < 0.5).mean()), 3)}
        report.append(row)
        print(json.dumps(row), flush=True)
        comp = rgb * a[..., None] + np.array(SKY_RGB) * (1 - a[..., None])   # ciel neutre pour la lecture
        tiles = [img, comp, np.repeat(a[..., None], 3, 2)]
        W, H = 640, 480 if K.height * 4 == K.width * 3 else int(640 * K.height / K.width)
        sheet = Image.new("RGB", (3 * W, H + 24), "white")
        for i, t in enumerate(tiles):
            sheet.paste(Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8)).resize((W, H)), (i * W, 24))
        ImageDraw.Draw(sheet).text(
            (6, 5), f"{args.label} {cam} {st} | photo | rendu (PSNR brut {row['psnr_brut']} / aligne "
                    f"{row['psnr_aligne']} dB) | opacite cumulee (moy. {row['alpha_moy_hors_ciel']})", fill=(0, 0, 0))
        sheet.save(f"{args.out}/{cam}_{st}.jpg", quality=88)
    with open(f"{args.out}/preview.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
