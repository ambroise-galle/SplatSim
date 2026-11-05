"""Fige la labelisation matériau adoptée : consensus M2 -> matériau, pour tous les
superpoints surface (décision du 22/09, faute de campagne d'annotation humaine).

Sorties : out/m4/labels_m2_consensus.npz
  sp_id, material (codes <U9), dominant_group, dominant_frac, n_vues, surface
+ statistiques de classes (console). Aucun rôle d'entraînement : c'est LE labeling
de scène adopté ; labels.csv (90 lignes humaines) reste un jeu d'évaluation.

Lancement (desktop) :  cd ~/dino && venv/bin/python splatpipe/stage2/m4_material_labels.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, "/home/ambroise/dino")
from splatpipe.stage2 import m4_splits_sheets as M

SURFACE_MIN = 20


def main() -> None:
    sp_id = np.load(M.SP_NPZ)["sp_id"]
    n_sp = int(sp_id.max()) + 1
    cnt = np.bincount(sp_id, minlength=n_sp)
    surface = cnt >= SURFACE_MIN

    m2 = np.load("/home/ambroise/dino/data_local/scene/reports/stage2_material/m2/"
                 "superpoint_semantics.npz")
    dg = m2["dominant_group"]
    frac = m2["dominant_frac"]
    n_vues = m2["n_vues"]

    names = np.array([M.GROUP_NAMES.get(int(g), "autre") for g in dg])
    mat = np.array([M.MAPPING_M2_MATERIAL.get(n, "other") for n in names])

    np.savez_compressed(
        f"{M.M4_DIR}/labels_m2_consensus.npz",
        sp_id=np.arange(n_sp, dtype=np.int64),
        material=mat, dominant_group=dg, dominant_frac=frac, n_vues=n_vues,
        surface=surface, source="consensus M2 (Mask2Former lift), mapping m4_splits_sheets",
    )

    print(f"npz -> {M.M4_DIR}/labels_m2_consensus.npz")
    print(f"\nLabelisation adoptée — {int(surface.sum())} superpoints surface (>= {SURFACE_MIN} voxels) :")
    u, c = np.unique(mat[surface], return_counts=True)
    for name, n in sorted(zip(u, c), key=lambda e: -e[1]):
        print(f"  {name:9s} {int(n):6d}  ({n / surface.sum():5.1%})")
    print("\nGroupes M2 sans aucun sp surface (classes inatteignables via ce mapping) :")
    present = set(dg[surface].tolist())
    for g, name in M.GROUP_NAMES.items():
        if g not in present:
            print(f"  groupe {g} ({name})")
    low = surface & (frac < 0.6)
    print(f"\nconsensus faible (dominant_frac < 0.6) : {int(low.sum())} sp "
          f"({low.sum() / surface.sum():.1%})")


if __name__ == "__main__":
    main()
