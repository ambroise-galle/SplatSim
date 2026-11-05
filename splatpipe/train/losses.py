"""Pertes B0–B4 du trainer 4.1 (une fonction par brique, activables indépendamment
pour les ablations 4.2).

B0 RGB          L1 sur pixels non-ciel.
B1 LiDAR        Huber(d_rendu − d_mesuré) sur cellules premier écho valides.
B2 espace libre alpha rendu → 0 sur les cellules du grille sans retour (poids léger :
                une cellule sans retour n'interdit pas tout obstacle, ex. verre).
B3 aplatissement σ_n des disques maintenu ≤ σ_n_max, DÉSACTIVÉ sur la végétation
                (feuillage = gaussiennes volumiques légitimes, bit matériau M2).
B4 ancrage      L2(means − means_init) : la carte reste la référence géométrique ;
                (pose_opt et exposition : hors v1, poses validées T0.4/3.1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def b0_rgb(rgb: torch.Tensor, img: torch.Tensor, sky: torch.Tensor | None) -> torch.Tensor:
    if sky is None:
        return F.l1_loss(rgb, img)
    return F.l1_loss(rgb[~sky], img[~sky])


def b1_lidar(d_r: torch.Tensor, rng_t: torch.Tensor, delta: float = 0.02) -> torch.Tensor:
    return F.huber_loss(d_r, rng_t, delta=delta)


def b2_free(alpha_free: torch.Tensor, floor: float = 0.1) -> torch.Tensor:
    """alpha des cellules sans retour : pénalité au-delà d'un petit plancher
    (le plancher évite de dissoudre les structures fines / bord de cellule)."""
    a = alpha_free.clamp(min=0.0, max=1.0)
    return (a.clamp(min=floor) - floor).pow(2).mean()


def b3_flatten(scales: torch.Tensor, veg_mask: torch.Tensor | None,
               sigma_n_max: float = 0.02) -> torch.Tensor:
    """σ_n (3ᵉ axe, la normale) maintenu fin sur les surfaces non-végétales."""
    sn = scales[:, 2]
    over = (sn - sigma_n_max).clamp(min=0).pow(2)
    if veg_mask is None:
        return over.mean()
    if (~veg_mask).sum() == 0:
        return over.new_zeros(())
    return over[~veg_mask].mean()


def b4_anchor(means: torch.Tensor, means_init: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(means, means_init)
