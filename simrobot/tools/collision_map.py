"""Carte vue de dessus du terrain de collision, avant / après les étiquettes du splat, sur les zones du trajet où le
plus de bosses invisibles ont été aplanies. Fond : relief du sol physique (écart au sol lissé sur 1,5 m) ; gris
foncé = pas de gaussienne visible ; cercles rouges = obstacles rigides ; vert = végétation qui freine ; jaune =
troncs couchés ; magenta = bosses aplanies (après) ; ligne blanche = trajet de l'ANYmal.

  .venv/bin/python simrobot/tools/collision_map.py --scene data_local/simrobot/scenes/HOB-1_full --out logs/simrobot/labels/carte_collisions.png
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3, help="nombre de zones")
    ap.add_argument("--size", type=float, default=16.0, help="côté d'une zone (m)")
    ap.add_argument("--px", type=int, default=6, help="pixels par cellule de 10 cm")
    a = ap.parse_args()
    from PIL import Image, ImageDraw, ImageFont
    from scipy import ndimage
    from scipy.spatial import cKDTree

    T = dict(np.load(f"{a.scene}/terrain.npz"))
    E = np.load(f"{a.scene}/terrain_etiquettes.npz")
    Gm = np.load(f"{a.scene}/geometrie_etiquetee.npz")
    H0, lo, res = T["H"], T["lo"], float(T["res"])
    nx, ny = H0.shape
    H1 = H0 + E["relief_dh"] + E["dH"]
    Hs = ndimage.median_filter(H0, 15)
    path = np.array(json.load(open(f"{a.scene}/trajets.json"))["dlio"]["p"])[:, :2]
    pres, flat = E["splat_present"], E["bosses_aplanies"]
    tr, src, typ = T["troncs"], T["troncs_source"], E["obs_type"].copy()
    # zones : cellules aplanies à moins de 6 m du trajet, regroupées par fenêtre
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cells = np.stack([lo[0] + (gx.ravel() + 0.5) * res, lo[1] + (gy.ravel() + 0.5) * res], 1)
    dpath = cKDTree(path).query(cells)[0].reshape(nx, ny)
    w = int(a.size / res)
    score = ndimage.uniform_filter((flat & (dpath < 6)).astype(float), w)
    centres = []
    for _ in range(a.n):
        k = np.unravel_index(np.argmax(score), score.shape)
        centres.append(k)
        score[max(k[0] - w, 0):k[0] + w, max(k[1] - w, 0):k[1] + w] = 0
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
    except OSError:
        font = None
    tiles = []
    for ci, cj in centres:
        i0, j0 = max(ci - w // 2, 0), max(cj - w // 2, 0)
        i1, j1 = min(i0 + w, nx), min(j0 + w, ny)
        row = []
        for after in (False, True):
            Hp = (H1 if after else H0 + 0 * H1)[i0:i1, j0:j1]
            rel = np.clip((Hp - Hs[i0:i1, j0:j1]) / 0.6, -1, 1)          # ±60 cm
            rgb = np.zeros(rel.shape + (3,), np.float32)
            rgb[..., 0] = 0.5 + 0.5 * np.clip(rel, 0, 1)
            rgb[..., 1] = 0.5 - 0.3 * np.abs(rel)
            rgb[..., 2] = 0.5 + 0.5 * np.clip(-rel, 0, 1)
            rgb[~pres[i0:i1, j0:j1]] *= 0.35                              # pas de gaussienne visible
            if after:
                vg = E["veg_s"][i0:i1, j0:j1] > 0.05
                rgb[vg] = 0.6 * rgb[vg] + 0.4 * np.array([0.1, 0.7, 0.2])
                fl = flat[i0:i1, j0:j1]
                rgb[fl] = 0.5 * rgb[fl] + 0.5 * np.array([0.9, 0.2, 0.9])
            img = Image.fromarray((np.transpose(rgb, (1, 0, 2))[::-1] * 255).astype(np.uint8))   # x → droite, y → haut
            img = img.resize((img.width * a.px, img.height * a.px), Image.NEAREST)
            d = ImageDraw.Draw(img)

            def to_px(x, y):
                return ((x - lo[0]) / res - i0) * a.px, (j1 - j0 - ((y - lo[1]) / res - j0)) * a.px

            pp = [to_px(x, y) for x, y in path if i0 <= (x - lo[0]) / res < i1 and j0 <= (y - lo[1]) / res < j1]
            if len(pp) > 1:
                d.line(pp, fill=(255, 255, 255), width=2)
            for k, (x, y, r, z) in enumerate(tr):
                px, py = to_px(x, y)
                if not (0 <= px < img.width and 0 <= py < img.height):
                    continue
                t = typ[k] if after else 0
                if t == 1:
                    col = (40, 200, 60)
                elif t in (2, 4):
                    continue
                else:
                    col = (255, 40, 40) if src[k] == 2 else (255, 150, 0)
                rp = max(r / res * a.px, 2)
                d.ellipse((px - rp, py - rp, px + rp, py + rp), outline=col, width=2)
            if after:
                for c in Gm["couches"]:
                    d.line([to_px(*c[:2]), to_px(*c[3:5])], fill=(255, 230, 0), width=max(int(2 * c[6] / res * a.px), 3))
            d.text((6, 4), "après (étiquettes)" if after else "avant", fill=(255, 255, 255), font=font)
            row.append(img)
        tiles.append(row)
    W, Hh = tiles[0][0].size
    S = Image.new("RGB", (2 * W + 8, len(tiles) * (Hh + 8) + 46), (20, 20, 24))
    for r_, (a_, b_) in enumerate(tiles):
        S.paste(a_, (0, 46 + r_ * (Hh + 8)))
        S.paste(b_, (W + 8, 46 + r_ * (Hh + 8)))
    d = ImageDraw.Draw(S)
    d.text((8, 4), "Sol physique vu de dessus (rouge : au-dessus du sol lissé, bleu : en dessous, ±60 cm ; sombre : pas de "
                   "gaussienne visible)", fill=(235, 235, 235), font=font)
    d.text((8, 24), "orange : troncs · rouge : obstacles bas rigides · vert : végétation qui freine · jaune : troncs couchés · "
                    f"magenta : bosses aplanies · blanc : trajet · zones de {a.size:.0f} m", fill=(235, 235, 235), font=font)
    S.save(a.out)
    print(a.out, S.size)


if __name__ == "__main__":
    main()
