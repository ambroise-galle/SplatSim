"""Export d'une scène entraînée (`scene_full.npz`) au format PLY standard 3DGS (INRIA), pour les visualiseurs
web (SuperSplat, etc.). CPU seulement (numpy) : marche sur le Mac comme sur la desktop.

- Découpe une zone (sphère de rayon `--radius` autour d'une pose caméra ou d'un point), sinon la scène entière.
- Retire les gaussiennes presque transparentes (`--min-opacity`, 0,05 par défaut).
- Recentre la zone sur l'origine et passe de z-haut (repère carte) à y-haut (convention des visualiseurs web) :
  (x, y, z) → (x, z, −y), rotation appliquée aussi aux quaternions.
- Couleur : coefficient SH d'ordre 0 seulement (la dépendance à l'angle de vue n'est pas tournée ni exportée).
Format : x y z nx ny nz f_dc_0..2 opacity(logit) scale_0..2(log) rot_0..3 (wxyz), float32 little-endian.

  .venv/bin/python -m splatpipe.train.export_ply --scene <scene_full.npz> --poses data_local/scene/poses/cameras.json \
      --center-pose cam0:1710338243391100552 --radius 40 --out logs/stage4/splat_cour.ply
"""

from __future__ import annotations

import argparse
import json

import numpy as np

Q_ZUP_TO_YUP = np.array([np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0])   # rotation de −90° autour de x (wxyz)
M_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)


def quat_mul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--poses", default="", help="cameras.json (pour --center-pose)")
    ap.add_argument("--center-pose", default="", help="cam:stamp")
    ap.add_argument("--center", default="", help="x,y,z dans le repère carte")
    ap.add_argument("--radius", type=float, default=0.0, help="0 = scène entière")
    ap.add_argument("--min-opacity", type=float, default=0.05)
    ap.add_argument("--no-yup", action="store_true", help="garder le repère carte (z haut)")
    args = ap.parse_args()

    z = np.load(args.scene)
    means = z["means"].astype(np.float64)
    opac = z["opac"].astype(np.float32).ravel()
    scales = z["scales"].astype(np.float32)
    quats = z["quats"].astype(np.float64)
    colors = z["colors"].astype(np.float32)
    dc = colors[:, 0, :] if colors.ndim == 3 else colors
    if "param_version" not in z.files:
        # v2 : paramètres bruts (opacités hors [0, 1], échelles signées) — ramenés dans le domaine valide
        opac = np.clip(opac, 0.0, 1.0)
        scales = np.abs(scales)

    center = np.zeros(3)
    if args.center_pose:
        cam, st = args.center_pose.split(":")
        T = np.asarray(json.load(open(args.poses))["cameras"][cam][st]["T_map_cam_start"], float)
        center = T[:3, 3]
    elif args.center:
        center = np.array([float(v) for v in args.center.split(",")])
    keep = opac >= args.min_opacity
    if args.radius > 0:
        keep &= ((means - center) ** 2).sum(1) <= args.radius ** 2
    n0 = len(means)
    means, opac, scales, quats, dc = means[keep] - center, opac[keep], scales[keep], quats[keep], dc[keep]
    quats /= np.linalg.norm(quats, axis=1, keepdims=True).clip(1e-12)
    if not args.no_yup:
        means = means @ M_ZUP_TO_YUP.T
        quats = quat_mul(np.broadcast_to(Q_ZUP_TO_YUP, quats.shape), quats)

    n = len(means)
    names = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    arr = np.zeros((n, len(names)), np.float32)
    arr[:, 0:3] = means
    arr[:, 6:9] = dc
    o = np.clip(opac, 1e-4, 1 - 1e-4)
    arr[:, 9] = np.log(o / (1 - o))
    arr[:, 10:13] = np.log(np.clip(scales, 1e-7, None))
    arr[:, 13:17] = quats
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" + \
        "".join(f"property float {p}\n" for p in names) + "end_header\n"
    with open(args.out, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())
    print(json.dumps({"gaussiennes_scene": n0, "exportees": n, "centre_carte": [round(float(c), 3) for c in center],
                      "rayon_m": args.radius, "min_opacite": args.min_opacity, "y_haut": not args.no_yup,
                      "taille_mo": round(arr.nbytes / 1e6, 1), "out": args.out}))


if __name__ == "__main__":
    main()
