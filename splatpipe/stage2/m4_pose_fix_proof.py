"""Preuve avant/après de la correction de convention des poses sur les vignettes M4.

Pour quelques superpoints de `splits.json`, construit la vignette M4 (même code que
`m4_splits_sheets.py` : sélection des vues, occlusion, recadrage) deux fois :
  - AVANT : rotation lue telle quelle dans `cameras.json` (bug : T_map_cam pris pour carte→caméra) ;
  - APRÈS : rotation transposée à la lecture (`build_cam_views`, corrigé).
Si les poses sont justes, les recadrages d'un même superpoint montrent le même objet d'une vue à l'autre.

Local (Mac), depuis la racine du dépôt :
  DINO_DATA_DIR=data_local/scene PYTHONPATH=. .venv/bin/python splatpipe/stage2/m4_pose_fix_proof.py
(`data_local/scene/rawimgs` doit pointer vers les photos brutes.)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m4_splits_sheets as m  # noqa: E402

OUT = os.environ.get("PROOF_OUT", "logs/stage2/material/pose_convention_check")
SPLITS = os.environ.get("SPLITS_JSON", "logs/stage2/material/m4/splits.json")
N_SP = int(os.environ.get("N_SP", "6"))


def sp_table():
    cols = m.read_ply_points(m.MAP_PLY, ["x", "y", "z", "nx", "ny", "nz", "n_scans"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    nrm = np.stack([cols["nx"], cols["ny"], cols["nz"]], 1).astype(np.float64)
    nsc = cols["n_scans"].astype(np.int64)
    keys = m.pack_ijk(np.floor(xyz / 0.05).astype(np.int64))
    o = np.argsort(keys, kind="stable")
    xyz_s, nrm_s, nsc_s = xyz[o], nrm[o], nsc[o]
    with np.load(m.SP_NPZ) as z:
        sp_id = z["sp_id"]
    n_sp = int(sp_id.max()) + 1
    cnt = np.bincount(sp_id, minlength=n_sp)
    og = np.argsort(sp_id, kind="stable")
    sp_g = sp_id[og]
    starts = np.flatnonzero(np.r_[True, sp_g[1:] != sp_g[:-1]])
    cen = np.zeros((n_sp, 3))
    cen[sp_g[starts]] = np.add.reduceat(xyz_s[og], starts, axis=0) / np.maximum(cnt, 1)[:, None]
    b0, b1 = starts, np.r_[starts[1:], len(sp_g)]
    return cen, xyz_s[og], nrm_s[og], nsc_s[og], b0, b1


def sheet_for(views, spi, cen, xyz_g, nrm_g, b0, b1, grid, mask_cache, title):
    vox, nrms = xyz_g[b0[spi]:b1[spi]], nrm_g[b0[spi]:b1[spi]]
    picked = m.pick_views_voxels(views, cen[spi], vox, nrms, grid, mask_cache)
    tiles = []
    for c, k, u, v2, r, fr in picked:
        jpg = views[c]["jpg"].get(k)
        if not jpg or not os.path.exists(jpg):
            continue
        img = Image.open(jpg).convert("RGB")
        W, H = img.size
        h = m.HALF_CROP
        u0 = int(np.clip(u - h, 0, W - 2 * h))
        v0 = int(np.clip(v2 - h, 0, H - 2 * h))
        crop = img.crop((u0, v0, u0 + 2 * h, v0 + 2 * h)).resize((300, 300))
        s = 300 / (2 * h)
        rr = max(4, int(r * s))
        ImageDraw.Draw(crop).ellipse([(u - u0) * s - rr, (v2 - v0) * s - rr, (u - u0) * s + rr, (v2 - v0) * s + rr],
                                     outline=(255, 0, 0), width=3)
        ImageDraw.Draw(crop).text((6, 284), f"{c} {fr:.0%}", fill=(255, 255, 0))
        tiles.append(crop)
    row = Image.new("RGB", (20 + 304 * max(len(tiles), 1), 322), "white")
    ImageDraw.Draw(row).text((4, 4), f"sp {spi} - {title} - {len(tiles)} vue(s)", fill=(0, 0, 0))
    for i, t in enumerate(tiles):
        row.paste(t, (20 + 304 * i, 20))
    return row, len(tiles)


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(SPLITS) as f:
        sp = json.load(f)
    rng = np.random.default_rng(0)
    cand = list(sp["test_sp"])[:]
    chosen = [int(x) for x in rng.choice(cand, size=min(N_SP, len(cand)), replace=False)]
    dropped = [int(x) for x in sp.get("dropped_invisible_sp", [])][:2]
    chosen += dropped

    cen, xyz_g, nrm_g, nsc_g, b0, b1 = sp_table()
    grid = m.build_spatial_grid(xyz_g)
    grid["xyz"] = xyz_g
    grid["nscans"] = nsc_g
    mask_cache = {}
    after = m.build_cam_views()                      # corrigé : rot = Rᵀ
    before = {c: dict(v, rot=v["rot"].transpose(0, 2, 1).copy()) for c, v in after.items()}  # ancien bug

    summary = []
    for spi in chosen:
        r_b, n_b = sheet_for(before, spi, cen, xyz_g, nrm_g, b0, b1, grid, mask_cache, "AVANT (bug)")
        r_a, n_a = sheet_for(after, spi, cen, xyz_g, nrm_g, b0, b1, grid, mask_cache, "APRES (corrige)")
        w = max(r_b.width, r_a.width)
        both = Image.new("RGB", (w, r_b.height + r_a.height), "white")
        both.paste(r_b, (0, 0))
        both.paste(r_a, (0, r_b.height))
        both.save(f"{OUT}/sp_{spi}_avant_apres.jpg", quality=85)
        tag = "ecarte_sans_vue" if spi in dropped else "test"
        summary.append({"sp": spi, "origine": tag, "vues_avant": n_b, "vues_apres": n_a})
        print(summary[-1], flush=True)
    with open(f"{OUT}/m4_proof.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
