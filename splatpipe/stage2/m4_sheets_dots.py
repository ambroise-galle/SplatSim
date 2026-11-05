"""Régénère les vignettes M4 avec l'empreinte voxel visible (sortie : out/m4/sheets2/).

Mêmes superpoints, mêmes vues que splits.json (pick_views_voxels est déterministe et
inchangé — seuls le dessin et le badge changent). labels.csv et splits.json ne sont
pas réécrits : les annotations déjà saisies restent valables.

Changement de représentation : le cercle rouge seul ne montre pas quels pixels
appartiennent au superpoint (sur les façades à brise-soleil, les matériaux alternent
tous les 20-50 cm pour des patchs de ~40 cm) — chaque voxel visible est maintenant
marqué d'un point magenta ; le cercle rouge ne sert qu'à trouver la zone.

Lancement (desktop) :
  cd /home/ambroise/dino && setsid nohup venv/bin/python splatpipe/stage2/m4_sheets_dots.py \
      > m4_dots.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/ambroise/dino")
from splatpipe.stage1.maptools import pack_ijk
from splatpipe.stage2 import m4_splits_sheets as M

DOTS = (255, 0, 255)       # magenta : absent des matériaux de la scène
CIRCLE = (255, 0, 0)


def main() -> None:
    from PIL import Image, ImageDraw, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    os.makedirs(f"{M.M4_DIR}/sheets2", exist_ok=True)

    # --- même chargement que M4 (sp_id aligné aux lignes du PLY, regroupement argsort)
    cols = M.read_ply_points(M.MAP_PLY, ["x", "y", "z", "nx", "ny", "nz", "n_scans"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    nrm = np.stack([cols["nx"], cols["ny"], cols["nz"]], 1).astype(np.float64)
    nsc = cols["n_scans"].astype(np.int64)
    keys = pack_ijk(np.floor(xyz / 0.05).astype(np.int64))
    o = np.argsort(keys, kind="stable")
    xyz_s, nrm_s, nsc_s = xyz[o], nrm[o], nsc[o]
    del xyz, nrm, cols
    with np.load(M.SP_NPZ) as z:
        sp_id = z["sp_id"]
    n_sp = int(sp_id.max()) + 1
    cnt = np.bincount(sp_id, minlength=n_sp)
    og = np.argsort(sp_id, kind="stable")
    sp_g = sp_id[og]
    starts = np.flatnonzero(np.r_[True, sp_g[1:] != sp_g[:-1]])
    assert sp_g[0] == 0 and len(starts) == n_sp, "sp_id non aligné aux lignes du PLY ?"
    cen = np.zeros((n_sp, 3))
    cen[sp_g[starts]] = np.add.reduceat(xyz_s[og], starts, axis=0) / np.maximum(cnt, 1)[:, None]
    xyz_g, nrm_g, nsc_g = xyz_s[og], nrm_s[og], nsc_s[og]
    b0, b1 = starts, np.r_[starts[1:], len(sp_g)]
    del xyz_s, nrm_s, nsc_s, sp_g, starts, og, o, keys

    views = M.build_cam_views()
    grid = M.build_spatial_grid(xyz_g)
    grid["xyz"] = xyz_g
    grid["nscans"] = nsc_g
    mask_cache = {}

    spl = json.load(open(f"{M.M4_DIR}/splits.json"))
    sel_all = sorted(set(spl["test_sp"]) | set(spl["train_sp"]))
    print(f"{len(sel_all)} superpoints à dessiner -> {M.M4_DIR}/sheets2/", flush=True)

    img_cache = {}
    n_sheets = 0
    for n_i, spi in enumerate(sel_all):
        cxy = cen[spi]
        vox = xyz_g[b0[spi]:b1[spi]]
        nrms = nrm_g[b0[spi]:b1[spi]]
        detail: list = []
        M.pick_views_voxels(views, cxy, vox, nrms, grid, mask_cache, detail=detail)
        crops = []
        for c, k, u, v2, r, fr, dist, uvv in detail:
            if (c, k) not in img_cache:
                jpg = views[c]["jpg"].get(k)
                if not jpg or not os.path.exists(jpg):
                    continue
                img_cache[(c, k)] = Image.open(jpg).convert("RGB")
            img = img_cache[(c, k)]
            W, H = img.size
            u0 = int(np.clip(u - M.HALF_CROP, 0, W - 2 * M.HALF_CROP))
            v0 = int(np.clip(v2 - M.HALF_CROP, 0, H - 2 * M.HALF_CROP))
            crop = img.crop((u0, v0, u0 + 2 * M.HALF_CROP, v0 + 2 * M.HALF_CROP))
            dr = ImageDraw.Draw(crop)
            rr = max(4, int(r))
            dr.ellipse([u - u0 - rr, v2 - v0 - rr, u - u0 + rr, v2 - v0 + rr],
                       outline=CIRCLE, width=4)
            for pu, pv in uvv:
                x_, y_ = pu - u0, pv - v0
                if 0 <= x_ < 2 * M.HALF_CROP and 0 <= y_ < 2 * M.HALF_CROP:
                    dr.ellipse([x_ - 3, y_ - 3, x_ + 3, y_ + 3], fill=DOTS)
            full = img.resize((400, 300))          # contexte : la vue entière
            scx, scy = 400.0 / W, 300.0 / H
            drf = ImageDraw.Draw(full)
            rr2 = max(3, int(r * scx))
            drf.ellipse([u * scx - rr2, v2 * scy - rr2, u * scx + rr2, v2 * scy + rr2],
                        outline=CIRCLE, width=2)
            for pu, pv in uvv:
                x_, y_ = pu * scx, pv * scy
                if 0 <= x_ < 400 and 0 <= y_ < 300:
                    drf.ellipse([x_ - 2, y_ - 2, x_ + 2, y_ + 2], fill=DOTS)
            drf.text((6, 282), f"{c} · {dist:.0f} m · {fr:.0%} du sp visible",
                     fill=(255, 255, 0))
            crops.append((c, crop, full))
        if not crops:
            print(f"[warn] sp {spi} : aucune vue projetable", flush=True)
            continue
        sheet = Image.new("RGB", (404 * len(crops), 752), "white")
        for i2, (c, cr, fullim) in enumerate(crops):
            sheet.paste(cr, (404 * i2 + 2, 22))       # zoom résolution native
            sheet.paste(fullim, (404 * i2 + 2, 430))  # contexte vue entière
            ImageDraw.Draw(sheet).text((404 * i2 + 8, 4), c, fill=(0, 0, 0))
        ImageDraw.Draw(sheet).text((8, 734),
                                   f"sp {spi} | z_carte={cxy[2]:.1f} m | {int(cnt[spi])} vox | "
                                   f"{len(crops)} vues | point magenta = 1 voxel visible",
                                   fill=(0, 0, 0))
        sheet.save(f"{M.M4_DIR}/sheets2/sp_{spi}.png")
        n_sheets += 1
        if n_sheets % 25 == 0:
            print(f"vignettes {n_sheets}/{len(sel_all)}", flush=True)

    print(f"terminé : {n_sheets}/{len(sel_all)} vignettes -> {M.M4_DIR}/sheets2/", flush=True)


if __name__ == "__main__":
    main()
