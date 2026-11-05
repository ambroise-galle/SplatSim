"""Carte matériau : nuage colorisé par le consensus M2 (mapping -> matériau),
pour contrôle visuel humain du « résultat labelisé » actuel (aucune feature DINO).

Sorties (out/m4_material/) :
  - vue_<n>.jpg  : photo | rendu nuage colorisé par matériau (occlusion peintre,
    ombrage lambert) pour 3 poses réelles de la marche ;
  - topview.png  : ortho top-down x-y des superpoints surface, coloriés par matériau.

Lancement (desktop) :  cd ~/dino && venv/bin/python splatpipe/stage2/m4_material_map.py
"""

import os
import sys

sys.path.insert(0, "/home/ambroise/dino")
sys.path.insert(0, "/home/ambroise/dino/splatpipe/stage2")
import numpy as np
from PIL import Image, ImageDraw, ImageFile

import m4_splits_sheets as m

ImageFile.LOAD_TRUNCATED_IMAGES = True

MATERIALS = ["foliage", "trunk", "grass", "metal", "concrete", "brick", "rock", "glass", "other"]
PALETTE = {
    "foliage": (50, 150, 60), "trunk": (110, 70, 35), "grass": (140, 200, 90),
    "metal": (60, 90, 200), "concrete": (192, 192, 186), "brick": (170, 70, 50),
    "rock": (120, 115, 110), "glass": (90, 200, 220), "other": (205, 60, 205),
}
POSE_SPS = [135769, 41231, 29920]     # jardinière / auvent vélos / tour lointaine


def main() -> None:
    out_dir = "/home/ambroise/dino/out/m4_material"
    os.makedirs(out_dir, exist_ok=True)

    cols = m.read_ply_points(m.MAP_PLY, ["x", "y", "z", "nx", "ny", "nz", "n_scans"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    nrm = np.stack([cols["nx"], cols["ny"], cols["nz"]], 1).astype(np.float64)
    nsc = cols["n_scans"].astype(np.int64)
    keys = m.pack_ijk(np.floor(xyz / 0.05).astype(np.int64))
    o = np.argsort(keys, kind="stable")
    xyz_s, nrm_s, nsc_s = xyz[o], nrm[o], nsc[o]
    del xyz, nrm, nsc, cols
    sp_id = np.load(m.SP_NPZ)["sp_id"]
    n_sp = int(sp_id.max()) + 1
    og = np.argsort(sp_id, kind="stable")
    sp_g = sp_id[og]
    xyz_g, nrm_g, nsc_g = xyz_s[og], nrm_s[og], nsc_s[og]
    del xyz_s, nrm_s, nsc_s, og, o, keys

    m2 = np.load("/home/ambroise/dino/data_local/scene/reports/stage2_material/m2/"
                 "superpoint_semantics.npz")
    mat_of_sp = np.array([m.MAPPING_M2_MATERIAL.get(m.GROUP_NAMES.get(int(g), "autre"), "other")
                          for g in m2["dominant_group"]])
    voxel_mat = mat_of_sp[sp_g]                      # par voxel (ordre groupé)
    rgb = np.array([PALETTE[name] for name in MATERIALS], np.uint8)
    vox_rgb = rgb[[MATERIALS.index(name) for name in voxel_mat]]

    views = m.build_cam_views()
    grid = m.build_spatial_grid(xyz_g)
    grid["xyz"] = xyz_g
    grid["nscans"] = nsc_g
    mask_cache = {}

    # centres par sp (utile pour pick)
    cnt = np.bincount(sp_id, minlength=n_sp)
    st = np.flatnonzero(np.r_[True, sp_g[1:] != sp_g[:-1]])
    cen = np.zeros((n_sp, 3))
    cen[sp_g[st]] = np.add.reduceat(xyz_g, st, axis=0) / np.maximum(cnt, 1)[:, None]

    for n_i, spi in enumerate(POSE_SPS):
        b0v = int(st[spi])
        vox = xyz_g[b0v:b0v + int(cnt[spi])]
        nrms = nrm_g[b0v:b0v + int(cnt[spi])]
        picked = m.pick_views_voxels(views, cen[spi], vox, nrms, grid, mask_cache, n_want=1)
        if not picked:
            print(f"[warn] sp {spi} sans vue", flush=True)
            continue
        c, k, u, v2, r, fr = picked[0]
        v = views[c]
        i = v["stamps"].index(k)
        R, cc = v["rot"][i], v["pos"][i]
        model = v["model"]
        W, H = model.width, model.height
        dif = xyz_g - cc
        d2 = np.einsum("ij,ij->i", dif, dif)
        cand = np.flatnonzero(d2 < 2500.0)
        pcf = dif[cand] @ R.T
        depth = np.sqrt(np.einsum("ij,ij->i", pcf, pcf))
        uv, valid, _ = model.project(pcf, margin=0)
        jj = np.flatnonzero(valid & (pcf[:, 2] > 0.1))
        jj = jj[np.argsort(-depth[jj])]              # peintre : loin -> près
        ui = uv[jj, 0].round().astype(np.int64).clip(0, W - 1)
        vi_ = uv[jj, 1].round().astype(np.int64).clip(0, H - 1)
        # lambert : lumière = caméra
        rays = pcf[jj] / depth[jj, None]
        lam = np.abs(np.einsum("ij,ij->i", nrm_g[cand][jj], rays)).clip(0.25, 1.0)
        render = np.full((H, W, 3), 235, np.uint8)
        render[vi_, ui] = (vox_rgb[cand][jj] * lam[:, None]).astype(np.uint8)
        img = Image.open(v["jpg"][k]).convert("RGB")
        u0 = int(np.clip(u - 400, 0, W - 800))
        v0 = int(np.clip(v2 - 400, 0, H - 800))
        pair = Image.new("RGB", (800, 1600), "white")
        pair.paste(img.crop((u0, v0, u0 + 800, v0 + 800)), (0, 0))
        pair.paste(Image.fromarray(render).crop((u0, v0, u0 + 800, v0 + 800)), (0, 802))
        d = ImageDraw.Draw(pair)
        d.text((8, 782), f"sp {spi} · {c} · {fr:.0%} du sp visible (haut : photo, bas : matériaux M2)",
               fill=(255, 255, 0))
        pair.save(f"{out_dir}/vue_{n_i}_{c}_sp{spi}.jpg", quality=90)
        print(f"vue_{n_i} ok ({c}, {fr:.0%})", flush=True)

    # --- top-down colorisé + légende
    x, y, z = xyz_g[:, 0], xyz_g[:, 1], xyz_g[:, 2]
    keep = (z > -2) & (np.abs(y) < 1e9)
    sc = 8                       # 8 px / m
    x0, y0 = x[keep].min(), y[keep].min()
    Wm = int((x[keep].max() - x0) * sc) + 1
    Hm = int((y[keep].max() - y0) * sc) + 1
    top = np.full((Hm, Wm, 3), 250, np.uint8)
    ix = ((y[keep] - y0) * sc).round().astype(np.int64).clip(0, Hm - 1)
    iy = ((x[keep] - x0) * sc).round().astype(np.int64).clip(0, Wm - 1)
    top[ix, iy] = vox_rgb[keep]
    im = Image.fromarray(top[::-1])              # y carte vers le haut
    d = ImageDraw.Draw(im)
    for n_i, name in enumerate(MATERIALS):
        d.rectangle([10, 10 + 18 * n_i, 26, 26 + 18 * n_i], fill=PALETTE[name], outline=(0, 0, 0))
        d.text((32, 12 + 18 * n_i), name, fill=(0, 0, 0))
    im.save(f"{out_dir}/topview.png")
    print(f"topview {im.size} ok -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
