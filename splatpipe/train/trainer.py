"""Trainer 4.1 — fenêtre le long de la trajectoire (fork du squelette av_trainer,
données Oxford Spires ; chargeur CPU/uint8, rendus LiDAR sûrs + fisheye).

Une fenêtre = un rayon autour d'un centre de trajectoire : les gaussiennes de la carte
(disques plats orientés) y sont optimisées sur les balayages/frames contemporains avec
les pertes B0–B4 (cf. losses.py). Le squelette (params dict, Adam, checkpoint, éval)
suit av_trainer ; la donnée et les pertes sont propres à SplatSim.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

RANGE_MAX = 28.0
MATCH_CM = 10.0
HUBER_DELTA = 0.02
LR = {"means": 1e-3, "quats": 1e-3, "scales": 1e-3, "opac": 5e-3, "colors": 5e-3}


class WindowTrainer:
    def __init__(self, root: str, stamps: list[int], r_opt: float = 35.0,
                 holdout_every: int = 10, dev: str = "cuda", seed: int = 0):
        import torch
        from scipy.spatial import cKDTree

        from splatpipe.stage4.overfit_core import _load_lidar_model_root, _load_map_disks
        from splatpipe.train.data import WindowData

        self.torch = torch
        self.dev = dev
        self.rng = np.random.default_rng(seed)
        self.root = root
        self.r_opt = r_opt
        t0 = time.time()

        self.lidar_model, _ = _load_lidar_model_root(root, dev)
        means, quats, scales, opac, map_nsc = _load_map_disks(dev, f"{root}/map/map_5cm_static.ply")
        self.n_map = len(means)

        self.data = WindowData(root, stamps, holdout_every=holdout_every)
        tr = self.data.sweep(self.data.train_stamps[len(self.data.train_stamps) // 2])
        center = tr["o_mid"]
        near = torch.nonzero(((means - torch.from_numpy(center).to(dev)) ** 2).sum(1) < r_opt ** 2).ravel()
        self.orig_idx = near.cpu().numpy()
        self.means, self.quats, self.scales, self.opac = (p[near].clone() for p in (means, quats, scales, opac))
        del near
        torch.cuda.empty_cache()
        for p in (self.means, self.quats, self.scales, self.opac):
            p.requires_grad_(True)
        self.colors = torch.full((len(self.means), 3), 0.5, device=dev, requires_grad=True)
        self.means_init = self.means.detach().clone()
        self.n_opt = len(self.means)

        # matériau par gaussienne (B3 : végétation exclue de l'aplatissement)
        self.veg_mask = None
        lab = f"{root}/reports/m2_material/labels_m2_consensus.npz"
        lab2 = f"{root}/reports/stage2_material/m4/labels_m2_consensus.npz"
        sp_path = f"{root}/reports/stage2_material/superpoints.npz"
        for p in (lab, lab2):
            if os.path.exists(p) and os.path.exists(sp_path):
                mat = np.load(p)["material"]
                sp_id = np.load(sp_path)["sp_id"]
                self.veg_mask = torch.from_numpy(
                    (mat[sp_id[self.orig_idx]] == "foliage")).to(dev)
                break

        # cellules de perte par balayage (tenseurs GPU préchargés) + masque dyn
        self.sweep_gpu: dict[int, dict] = {}
        tree = cKDTree(self.means.detach().cpu().numpy())
        nsc_near = map_nsc[self.orig_idx]
        for st in self.data.stamps:
            sw = self.data.sweep(st)
            f = sw["first"]
            rng_f = sw["rng"][f]
            row_f, col_f = sw["row"][f], sw["col"][f]
            dyn_f = sw["dyn_pt"][f] if sw["dyn_pt"] is not None else None
            if dyn_f is not None:
                keep = (rng_f <= RANGE_MAX) & (~dyn_f)
                dyn_mode = "sidecar"
            else:
                _, j = tree.query(sw["xyz"][f], workers=-1)
                keep = (rng_f <= RANGE_MAX) & (nsc_near[j] >= 3)
                dyn_mode = "n_scans>=3"
            g = dict(
                row=torch.from_numpy(row_f[keep].astype(np.int64)).to(dev),
                col=torch.from_numpy(col_f[keep].astype(np.int64)).to(dev),
                rng=torch.from_numpy(rng_f[keep].astype(np.float32)).to(dev),
                free_row=torch.from_numpy(sw["free_rows"].astype(np.int64)).to(dev),
                free_col=torch.from_numpy(sw["free_cols"].astype(np.int64)).to(dev),
                dyn_mode=dyn_mode,
                T0=torch.tensor(np.linalg.inv(sw["T0"]), dtype=torch.float32, device=dev),
                T1=torch.tensor(np.linalg.inv(sw["T1"]), dtype=torch.float32, device=dev),
            )
            self.sweep_gpu[st] = g
        self._sky_cache: dict = {}
        print(f"fenêtre : {self.n_opt}/{self.n_map} gaussiennes, {len(self.data.train_stamps)} "
              f"balayages train / {len(self.data.holdout)} holdout — préparation "
              f"{time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ rendus
    def render_lidar(self, st: int):
        import gsplat

        g = self.sweep_gpu[st]
        renders, alphas, _ = gsplat.rasterization(
            self.means, self.quats, self.scales, self.opac, self.colors, g["T0"][None],
            self.torch.eye(3, device=self.dev)[None], self.lidar_model.n_columns,
            self.lidar_model.n_rows, camera_model="lidar", lidar_coeffs=self.lidar_model,
            with_ut=True, with_eval3d=True, packed=False, render_mode="RGB-Ed",
            global_z_order=False, far_plane=300.0,
            rolling_shutter=gsplat.RollingShutterType.ROLLING_LEFT_TO_RIGHT,
            viewmats_rs=g["T1"][None])
        return renders[0, ..., 3], alphas[0, ..., 0]

    def render_frame(self, fr: dict, scale: float = 1.0):
        import torch
        import yaml
        from PIL import Image

        from splatpipe.common.camera_model import FisheyeCamera

        K = FisheyeCamera.from_dict(yaml.safe_load(open(f"{self.root}/sensors.yaml"))["cameras"][fr["cam"]])
        viewmat = torch.tensor(np.linalg.inv(fr["T_wc"]), dtype=torch.float32, device=self.dev)[None]
        W_r, H_r = int(K.width * scale) // 2 * 2, int(K.height * scale) // 2 * 2
        Ks = torch.zeros(1, 3, 3, device=self.dev)
        Ks[0, 0, 0], Ks[0, 1, 1] = K.fx * scale, K.fy * scale
        Ks[0, 0, 2], Ks[0, 1, 2] = K.cx * scale, K.cy * scale
        Ks[0, 2, 2] = 1.0
        radial = torch.tensor([[K.k1, K.k2, K.k3, K.k4]], dtype=torch.float32, device=self.dev)

        key = (fr["cam"], fr["stamp"])
        if key not in self._sky_cache:
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            base = os.path.basename(fr["rel"])
            p = f"{self.root}/rawimgs/{fr['cam']}/{base}"
            if not os.path.exists(p):
                p = f"{self.root}/{fr['rel']}"
            img_u8 = np.asarray(Image.open(p).convert("RGB"))
            sky = None
            if fr["sky_path"]:
                m = np.asarray(Image.open(fr["sky_path"]))
                sky = (m & 1) > 0
            self._sky_cache[key] = (img_u8, sky)
        img_u8, sky = self._sky_cache[key]
        if scale != 1.0:
            img_u8 = np.asarray(Image.fromarray(img_u8).resize((W_r, H_r), Image.BILINEAR))
            if sky is not None:
                sky = np.asarray(Image.fromarray(sky.astype(np.uint8) * 255).resize((W_r, H_r), Image.NEAREST)) > 127
        import torch.nn.functional as F
        img = F.interpolate(
            torch.from_numpy(img_u8.astype(np.float32) / 255.0).permute(2, 0, 1)[None],
            size=(H_r, W_r), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).to(self.dev)
        sky_t = None
        if sky is not None:
            sky_t = torch.from_numpy(sky).to(self.dev)
        import gsplat
        out, _, _ = gsplat.rasterization(
            self.means, self.quats, self.scales, self.opac, self.colors, viewmat, Ks,
            W_r, H_r, camera_model="fisheye", radial_coeffs=radial, with_ut=True,
            with_eval3d=True, packed=False, render_mode="RGB", far_plane=300.0)
        return out[0], img, sky_t

    # ------------------------------------------------------------------ éval
    def eval_sweep(self, st: int) -> dict:
        d = self.data.sweep(st)
        f = d["first"]
        rm = d["rng"][f]
        with self.torch.no_grad():
            d_r, a_r = self.render_lidar(st)
        dr = d_r.double().cpu().numpy()
        ar = a_r.double().cpu().numpy()
        rows, cols = d["row"][f], d["col"][f]
        has = ar[rows, cols] > 0.5
        drc = dr[rows, cols]
        ok = has & np.isfinite(drc) & (drc < 100.0) & np.isfinite(rm)
        err = drc[ok] - rm[ok]
        matched = np.abs(err) <= MATCH_CM / 100
        return {"n_ok": int(ok.sum()), "return_frac": round(float(has.mean()), 4),
                "matched_frac": round(float(matched.mean()), 4) if len(err) else None,
                "median_abs_cm": round(float(np.median(np.abs(err[matched]))) * 100, 2) if matched.any() else None,
                "divergent_median_cm": round(float(np.median(err[~matched])) * 100, 2) if (~matched).any() else None}

    # ------------------------------------------------------------------ boucle
    def train(self, steps: int, weights: dict[str, float], eval_every: int = 250,
              ckpt_dir: str | None = None, log_every: int = 50) -> dict:
        import torch

        opt = torch.optim.Adam([
            {"params": self.means, "lr": LR["means"]}, {"params": self.quats, "lr": LR["quats"]},
            {"params": self.scales, "lr": LR["scales"]}, {"params": self.opac, "lr": LR["opac"]},
            {"params": self.colors, "lr": LR["colors"]}])
        hist, t0 = [], time.time()
        order = list(self.data.train_stamps)
        self.rng.shuffle(order)
        holdout_stamps = sorted(self.data.holdout)
        frames_of = {st: self.data.frames_for_sweep(st) for st in order}
        for it in range(1, steps + 1):
            st = order[(it - 1) % len(order)]
            g = self.sweep_gpu[st]
            opt.zero_grad(set_to_none=True)
            d_r, a_r = self.render_lidar(st)
            loss = weights.get("b1", 0.0) * self._b1(d_r, g)
            if weights.get("b2", 0.0):
                loss = loss + weights["b2"] * self._b2(a_r, g)
            frs = frames_of[st]
            if frs and weights.get("b0", 0.0):
                fr = frs[int(self.rng.integers(len(frs)))]
                rgb, img, sky = self.render_frame(fr)
                from splatpipe.train.losses import b0_rgb
                loss = loss + weights["b0"] * b0_rgb(rgb, img, sky)
            if weights.get("b3", 0.0):
                from splatpipe.train.losses import b3_flatten
                loss = loss + weights["b3"] * b3_flatten(self.scales, self.veg_mask)
            if weights.get("b4", 0.0):
                from splatpipe.train.losses import b4_anchor
                loss = loss + weights["b4"] * b4_anchor(self.means, self.means_init)
            loss.backward()
            opt.step()
            if it % log_every == 0:
                hist.append({"iter": it, "loss": round(float(loss), 5), "stamp": int(st)})
                print(f"it {it:5d} loss {float(loss):.5f} (sweep {st})", flush=True)
            if eval_every and it % eval_every == 0 and holdout_stamps:
                hs = holdout_stamps[(it // eval_every) % len(holdout_stamps)]
                m = self.eval_sweep(hs)
                hist[-1]["eval"] = m
                print(f"it {it:5d} eval {hs}: {json.dumps(m)}", flush=True)
            if ckpt_dir and (it % 1000 == 0 or it == steps):
                self.save_checkpoint(ckpt_dir, it)
        hist.append({"wall_s": round(time.time() - t0, 1)})
        return {"history": hist,
                "train_eval_final": self.eval_sweep(sorted(self.data.train_stamps)[len(self.data.train_stamps) // 2]),
                "holdout_eval_final": {str(s): self.eval_sweep(s) for s in holdout_stamps},
                "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1)}

    def _b1(self, d_r, g):
        from splatpipe.train.losses import b1_lidar
        return b1_lidar(d_r[g["row"], g["col"]], g["rng"], delta=HUBER_DELTA)

    def _b2(self, a_r, g):
        from splatpipe.train.losses import b2_free
        return b2_free(a_r[g["free_row"], g["free_col"]])

    def save_checkpoint(self, ckpt_dir: str, it: int) -> str:
        os.makedirs(ckpt_dir, exist_ok=True)
        p = f"{ckpt_dir}/ckpt_{it:06d}.npz"
        np.savez_compressed(
            p, means=self.means.detach().cpu().numpy(), quats=self.quats.detach().cpu().numpy(),
            scales=self.scales.detach().cpu().numpy(), opac=self.opac.detach().cpu().numpy(),
            colors=self.colors.detach().cpu().numpy(), orig_idx=self.orig_idx,
            stamp_it=it)
        return p
