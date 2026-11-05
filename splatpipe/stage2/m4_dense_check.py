"""Rendu dense final : nuage accumule (occlusion exacte 1px) vs photo, empiles."""
import sys, os
sys.path.insert(0, "/home/ambroise/dino")
sys.path.insert(0, "/home/ambroise/dino/splatpipe/stage2")
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm
import m4_splits_sheets as m

cols = m.read_ply_points(m.MAP_PLY, ["x","y","z","nx","ny","nz","n_scans"])
xyz = np.stack([cols["x"],cols["y"],cols["z"]],1).astype(np.float64)
nsc = cols["n_scans"].astype(np.int64)
keys = m.pack_ijk(np.floor(xyz/0.05).astype(np.int64))
o = np.argsort(keys, kind="stable"); xyz_s = xyz[o]; nsc_s = nsc[o]
with np.load(m.SP_NPZ) as z: sp_id = z["sp_id"]
og = np.argsort(sp_id, kind="stable")
xyz_g = xyz_s[og]; nsc_g = nsc_s[og]
views = m.build_cam_views()

with np.load(m.SP_NPZ) as z: sp_id2 = z["sp_id"]
og2 = np.argsort(sp_id2, kind="stable"); sp_g2 = sp_id2[og2]
starts2 = np.flatnonzero(np.r_[True, sp_g2[1:] != sp_g2[:-1]])
cnt2 = np.bincount(sp_id2, minlength=int(sp_id2.max())+1)
cen2 = np.zeros((int(sp_id2.max())+1, 3))
cen2[sp_g2[starts2]] = np.add.reduceat(xyz_s[og2], starts2, axis=0)/np.maximum(cnt2,1)[:,None]
SPI = 80660
b0v = int(starts2[SPI]); cxy = cen2[SPI]
vox = xyz_g[b0v : b0v + int(cnt2[SPI])]
grid = m.build_spatial_grid(xyz_g); grid["xyz"] = xyz_g; grid["nscans"] = nsc_g
mask_cache = {}
nrm_all = np.stack([cols["nx"],cols["ny"],cols["nz"]],1).astype(np.float64)[o][og]
nrms = nrm_all[b0v : b0v + int(cnt2[SPI])]
picked = m.pick_views_voxels(views, cxy, vox, nrms, grid, mask_cache)

zmin, zmax = -3.0, 12.0
fz = np.clip((xyz_g[:,2]-zmin)/(zmax-zmin), 0, 1)
cmap = matplotlib.colormaps["turbo"]
cols_rgb = (cmap(fz)[:,:3]*255).astype(np.uint8)
shade = (0.55 + 0.45*np.clip(fz*3%1, 0, 1))[:,None]
cols_rgb = (cols_rgb * shade).astype(np.uint8)

os.makedirs("/home/ambroise/dino/out/m4_dense", exist_ok=True)
for vi,(c,k,u,v2,r,fr) in enumerate(picked[:2]):
    v = views[c]
    i = v["stamps"].index(k)
    R, cc = v["rot"][i], v["pos"][i]
    model = v["model"]; W, H = model.width, model.height
    dif = xyz_g - cc
    d2 = np.einsum("ij,ij->i", dif, dif)
    cand = np.flatnonzero(d2 < 2500.0)
    pcf = dif[cand].astype(np.float64) @ R.T
    depth = np.sqrt(np.einsum("ij,ij->i", pcf, pcf))
    uv, valid, _ = model.project(pcf, margin=0)
    front = pcf[:,2] > 0.1
    jj = np.flatnonzero(valid & front)
    ui = uv[jj,0].round().astype(np.int64).clip(0,W-1)
    vi_ = uv[jj,1].round().astype(np.int64).clip(0,H-1)
    t = depth[jj]
    cell = vi_ * W + ui
    dmin = np.full(H*W, np.inf)
    np.minimum.at(dmin, cell, t)
    vis = t <= dmin[cell] + 0.05
    ui, vi_, t, idx = ui[vis], vi_[vis], t[vis], cand[jj][vis]
    render = np.full((H, W, 3), 240, np.uint8)
    render[vi_, ui] = cols_rgb[idx]
    uv_s, val_s, _ = model.project((vox-cc)@R.T, margin=0)
    for x, y in uv_s[val_s].round().astype(int).clip(min=0):
        render[np.clip(y-1,0,H-1):np.clip(y+2,0,H-1), np.clip(x-1,0,W-1):np.clip(x+2,0,W-1)] = (255,0,0)
    photo = Image.open(v["jpg"][k]).convert("RGB")
    W2, H2 = photo.size
    u0 = int(np.clip(u-400, 0, W2-800)); v0 = int(np.clip(v2-400, 0, H2-800))
    pcrop = photo.crop((u0, v0, u0+800, v0+800))
    rcrop = Image.fromarray(render).crop((u0, v0, u0+800, v0+800))
    comp = Image.new("RGB", (804, 1608), "white")
    comp.paste(pcrop, (2, 2))
    comp.paste(rcrop, (2, 806))
    dd = ImageDraw.Draw(comp)
    dd.text((10, 810), f"RENDU nuage accumule - pose {c} (sp 80660 en rouge)", fill=(0,0,0))
    comp.save(f"/home/ambroise/dino/out/m4_dense/dense_{SPI}_{vi}_{c}.png")
    print(f"vue {vi} {c} ok")
print("ok")
