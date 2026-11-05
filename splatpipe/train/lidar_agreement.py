"""Accord LiDAR rendu / mesuré d'une scène (carte initiale ou splat entraîné), par bande d'élévation et par type de
surface TLS — contrôle de la carte avant entraînement (DiTer++, 08/10).

Pour un échantillon de balayages d'entraînement : rendu gsplat (modèle de faisceaux, obturateur roulant) depuis la
pose du balayage, distance rendue vs mesurée par cellule (premier écho, ≤ 28 m). Fraction à ≤ 10 cm et médiane signée
(négative = rendu trop court : quelque chose devant la surface mesurée). Si `--tls` est donné, chaque point mesuré est
classé par le plan TLS local : mur (normale horizontale), sol (normale verticale), autre (végétation, objets).

  venv/bin/python -m splatpipe.train.lidar_agreement --root <scène> [--scene out/.../scene_full.npz] \
      [--tls data_local/diterpp/lawn/Lawn_GT.pcd] --out <rapport.json>
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    import torch
    import yaml

    import gsplat
    from splatpipe.stage4.overfit_core import _load_lidar_model_root, _load_map_disks
    from splatpipe.train.data import WindowData, lidar_paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scene", default="", help="scene_full.npz (valeurs activées) ; défaut : carte initiale")
    ap.add_argument("--tls", default="")
    ap.add_argument("--every", type=int, default=200)
    ap.add_argument("--split", default="train", choices=["train", "test_in"],
                    help="balayages d'entraînement ou de test (jamais optimisés)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"
    lm, _ = _load_lidar_model_root(args.root, dev)
    if args.scene:
        z = np.load(args.scene)
        means, quats, scales, opac = (torch.from_numpy(z[k].astype(np.float32)).to(dev)
                                      for k in ("means", "quats", "scales", "opac"))
    else:
        means, quats, scales, opac, _ = _load_map_disks(dev, f"{args.root}/map/map_5cm_static.ply")
    col = torch.zeros(len(means), 1, 3, device=dev)
    lid, _ = lidar_paths(args.root)
    mj = json.load(open(f"{args.root}/lidar_model/{lid}.json"))
    el_row = np.degrees(np.asarray(mj["row_elevations_rad"]))
    stamps = sorted(int(x) for x in yaml.safe_load(open(f"{args.root}/splits.yaml"))["lidars"][lid][args.split])
    data = WindowData(args.root, stamps)
    tt = tls = None
    if args.tls:
        from scipy.spatial import cKDTree

        from splatpipe.common.pcd import read_pcd

        t = read_pcd(args.tls)
        tls = np.stack([t["x"], t["y"], t["z"]], 1).astype(np.float64)
        M = means.cpu().numpy()
        lo, hi = M.min(0) - 5, M.max(0) + 5
        tls = tls[np.all((tls >= lo) & (tls <= hi), axis=1)]
        tt = cKDTree(tls)
    E, EL, CL = [], [], []
    for s in stamps[args.every // 2::args.every]:
        sw = data.sweep(s)
        f = sw["first"]
        rows, cols, rm = sw["row"][f], sw["col"][f], sw["rng"][f]
        T0 = torch.tensor(np.linalg.inv(sw["T0"]), dtype=torch.float32, device=dev)
        T1 = torch.tensor(np.linalg.inv(sw["T1"]), dtype=torch.float32, device=dev)
        with torch.no_grad():
            r, a, _ = gsplat.rasterization(
                means, quats, scales, opac, col, T0[None], torch.eye(3, device=dev)[None], lm.n_columns, lm.n_rows,
                camera_model="lidar", lidar_coeffs=lm, with_ut=True, with_eval3d=True, packed=False,
                render_mode="RGB-Ed", global_z_order=False, far_plane=300.0, sh_degree=0,
                rolling_shutter=gsplat.RollingShutterType.ROLLING_LEFT_TO_RIGHT, viewmats_rs=T1[None])
        d = r[0, ..., 3].cpu().numpy()[rows, cols]
        al = a[0, ..., 0].cpu().numpy()[rows, cols]
        ok = (al > 0.5) & (rm <= 28.0)
        cl = np.full(len(rm), 2, np.int8)                         # 0 mur, 1 sol, 2 autre
        if tt is not None:
            p = sw["xyz"][f].astype(np.float64)
            dd, ii = tt.query(p, k=16, workers=-1)
            nb = tls[ii]
            mu = nb.mean(1)
            c = np.einsum("nki,nkj->nij", nb - mu[:, None], nb - mu[:, None]) / 16
            lam, vec = np.linalg.eigh(c)
            planar = (lam[:, 0] / lam.sum(1).clip(1e-12) < 0.01) & (dd[:, -1] < 0.15)
            nz = np.abs(vec[:, 2, 0])
            cl[planar & (nz < 0.3)] = 0
            cl[planar & (nz > 0.9)] = 1
        E.append((d - rm)[ok])
        EL.append(el_row[rows][ok])
        CL.append(cl[ok])
    e, el, cl = np.concatenate(E), np.concatenate(EL), np.concatenate(CL)

    def summ(m):
        return {"n": int(m.sum()), "apparies_10cm": round(float(np.mean(np.abs(e[m]) <= 0.10)), 4),
                "mediane_signee_m": round(float(np.median(e[m])), 3)} if m.any() else None
    rep = {"scene": args.scene or "carte initiale", "split": args.split, "n_balayages": len(E), "tout": summ(np.ones(len(e), bool)),
           "elevation": {f"{lo}..{hi}°": summ((el >= lo) & (el < hi)) for lo, hi in ((-25, -10), (-10, 0), (0, 25))}}
    if tt is not None:
        rep["surface_tls"] = {"murs": summ(cl == 0), "sol": summ(cl == 1), "autre": summ(cl == 2)}
    json.dump(rep, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
