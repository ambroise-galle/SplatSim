"""Caméras du trainer : projection torch, arguments gsplat et pixels valides, selon le modèle de `sensors.yaml`.

- `OPENCV_FISHEYE` (Oxford, équidistant k1–k4) : rendu gsplat `fisheye` + `radial_coeffs` ; angle borné au
  domaine gsplat (rayon max / focale, distorsion ignorée par gsplat) ; coins hors domaine exclus.
- `PINHOLE` (DiTer++ : images redressées une fois à l'ingestion, `ingest_diterpp.py --undistort`) : rendu gsplat
  `pinhole` sans distorsion ; pixels valides = zone couverte par l'image d'origine (`valid_mask`, PNG relatif à la
  racine de la scène).

Repère caméra OpenCV (x droite, y bas, z avant) ; `pc` = points dans ce repère.
"""

from __future__ import annotations

import os

import numpy as np


def make_camera(cfg: dict, root: str = ""):
    model = cfg.get("model", "OPENCV_FISHEYE")
    if model == "OPENCV_FISHEYE":
        return FisheyeTrainCamera(cfg)
    if model == "PINHOLE":
        return PinholeTrainCamera(cfg, root)
    raise ValueError(f"modèle de caméra non géré par le trainer : {model}")


class _Base:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.width, self.height = int(cfg["width"]), int(cfg["height"])
        self.fx, self.fy, self.cx, self.cy = (float(cfg[k]) for k in ("fx", "fy", "cx", "cy"))

    def K_tensor(self, dev):
        import torch

        return torch.tensor([[[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]], dtype=torch.float32,
                            device=dev)


class FisheyeTrainCamera(_Base):
    model = "fisheye"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        from splatpipe.train.color_lift import gsplat_max_angle

        self.k = [float(cfg[k]) for k in ("k1", "k2", "k3", "k4")]
        self.max_angle = gsplat_max_angle(cfg)

    def project(self, pc):
        """→ (u, v, ok) ; ok = devant, dans le domaine angulaire gsplat et dans l'image."""
        import torch

        r = pc[:, :2].norm(dim=1).clamp(min=1e-9)
        th = torch.atan2(r, pc[:, 2])
        t2 = th * th
        k1, k2, k3, k4 = self.k
        thd = th * (1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4)
        u = self.fx * thd * pc[:, 0] / r + self.cx
        v = self.fy * thd * pc[:, 1] / r + self.cy
        ok = (th < self.max_angle) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return u, v, ok

    def gsplat_kwargs(self, dev) -> dict:
        import torch

        return {"camera_model": "fisheye", "radial_coeffs": torch.tensor([self.k], dtype=torch.float32, device=dev)}

    def valid_mask(self) -> np.ndarray:
        from splatpipe.train.color_lift import gsplat_valid_mask

        return gsplat_valid_mask(self.cfg)


class PinholeTrainCamera(_Base):
    model = "pinhole"

    def __init__(self, cfg: dict, root: str = ""):
        super().__init__(cfg)
        self.mask_path = os.path.join(root, cfg["valid_mask"]) if cfg.get("valid_mask") else ""

    def project(self, pc):
        z = pc[:, 2]
        zc = z.clamp(min=1e-6)
        u = self.fx * pc[:, 0] / zc + self.cx
        v = self.fy * pc[:, 1] / zc + self.cy
        ok = (z > 0.05) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return u, v, ok

    def gsplat_kwargs(self, dev) -> dict:
        return {"camera_model": "pinhole"}

    def valid_mask(self) -> np.ndarray:
        if self.mask_path and os.path.exists(self.mask_path):
            from PIL import Image

            return np.asarray(Image.open(self.mask_path)) > 0
        return np.ones((self.height, self.width), bool)
