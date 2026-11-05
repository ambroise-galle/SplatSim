"""T4.5 sur DiTer++ : erreur géométrique du splat contre le scan TLS, en fonction de la distance à la trajectoire
d'entraînement.

Capteurs virtuels : pose (milieu du tour) des balayages de TEST (`splits.yaml`, jamais vus à l'entraînement), décalée
latéralement de ±d m (axe y du capteur, même hauteur : `--offsets`) ou verticalement de +h m (`--heights`) — à
hauteur constante, l'incidence sur le sol reste rasante ; plus haut, elle se redresse. Une pose virtuelle est gardée si aucun point TLS
n'est à moins de 0,3 m de son origine (capteur à ~0,5 m du sol). Depuis chaque pose, même grille de faisceaux (modèle Ouster, sans obturateur
roulant) :
- splat : distance rendue (gsplat, RGB-Ed), cellules d'opacité cumulée > 0,5 ;
- TLS : z-buffer des points TLS projetés dans la grille (ligne = élévation la plus proche à ± ½ pas, colonne = azimut
  ramené à la colonne).
Erreur par cellule = splat − TLS, sur les cellules valides des deux côtés à ≤ `--max-range` m : fraction à ≤ 10 cm,
médiane |erreur|, médiane signée, couverture (part des cellules TLS où le splat rend quelque chose), et **erreur selon
la normale** = |erreur| × |cos incidence| (normale TLS locale) : la distance du point rendu à la surface, peu sensible à
l'incidence rasante qui amplifie l'erreur en distance.

  venv/bin/python -m splatpipe.train.eval_tls --root <scène> --scene out/.../scene_full.npz \
      --tls data_local/diterpp/lawn/Lawn_GT.pcd --out out/.../eval_tls.json
  (sans --scene : carte initiale, pour référence)
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def tls_grid(tls_tree, tls, T_world_sensor, model: dict, max_range: float):
    """Z-buffer des points TLS dans la grille (lignes, colonnes) du modèle, depuis la pose donnée → (distance,
    indice du point TLS retenu, -1 si vide)."""
    el_rows = np.asarray(model["row_elevations_rad"])
    col_az = np.asarray(model["column_azimuths_rad"])
    off = np.asarray(model["row_azimuth_offsets_rad"])
    n_r, n_c = len(el_rows), len(col_az)
    o = T_world_sensor[:3, 3]
    idx = tls_tree.query_ball_point(o, max_range)
    p = (tls[idx] - o) @ T_world_sensor[:3, :3]                # repère capteur
    rng = np.linalg.norm(p, axis=1)
    el = np.arcsin(np.clip(p[:, 2] / np.maximum(rng, 1e-9), -1, 1))
    az = np.arctan2(p[:, 1], p[:, 0])
    # ligne : élévation la plus proche, à moins d'un demi-pas
    order = np.argsort(el_rows)
    j = np.clip(np.searchsorted(el_rows[order], el), 1, n_r - 1)
    lo, hi = order[j - 1], order[j]
    row = np.where(np.abs(el - el_rows[lo]) < np.abs(el - el_rows[hi]), lo, hi)
    step = np.median(np.abs(np.diff(el_rows)))
    ok = np.abs(el - el_rows[row]) <= 0.5 * step
    # colonne : azimut relatif à la colonne 0 dans le sens de rotation
    dlt = np.median(np.diff(col_az))                          # signé (négatif si rotation horaire)
    rel = np.angle(np.exp(1j * (az - off[row] - col_az[0])))
    col = np.round(np.mod(rel / dlt, n_c)).astype(np.int64) % n_c
    row, col, rng, pid = row[ok], col[ok], rng[ok], np.asarray(idx)[ok]
    grid = np.full((n_r, n_c), np.inf)
    win = np.full((n_r, n_c), -1, np.int64)
    o_ = np.lexsort((rng, row * n_c + col))
    first = o_[np.r_[True, (row * n_c + col)[o_][1:] != (row * n_c + col)[o_][:-1]]]
    grid[row[first], col[first]] = rng[first]
    win[row[first], col[first]] = pid[first]
    return grid, win


def main():
    import torch
    import yaml
    from scipy.spatial import cKDTree

    import gsplat
    from splatpipe.common.pcd import read_pcd
    from splatpipe.stage4.overfit_core import _load_lidar_model_root, _load_map_disks
    from splatpipe.train.data import WindowData, lidar_paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scene", default="")
    ap.add_argument("--tls", required=True)
    ap.add_argument("--offsets", default="0,1,2,4", help="décalages latéraux (m), à hauteur constante")
    ap.add_argument("--heights", default="1,2", help="décalages verticaux (m), sans décalage latéral")
    ap.add_argument("--every", type=int, default=6, help="un balayage de test sur N")
    ap.add_argument("--max-range", type=float, default=25.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"
    lm, _ = _load_lidar_model_root(args.root, dev)
    lid, _ = lidar_paths(args.root)
    model = json.load(open(f"{args.root}/lidar_model/{lid}.json"))
    if args.scene:
        z = np.load(args.scene)
        means, quats, scales, opac = (torch.from_numpy(z[k].astype(np.float32)).to(dev)
                                      for k in ("means", "quats", "scales", "opac"))
    else:
        means, quats, scales, opac, _ = _load_map_disks(dev, f"{args.root}/map/map_5cm_static.ply")
    col0 = torch.zeros(len(means), 1, 3, device=dev)
    t = read_pcd(args.tls)
    tls = np.stack([t["x"], t["y"], t["z"]], 1).astype(np.float64)
    M = means.cpu().numpy()
    lo, hi = M.min(0) - args.max_range, M.max(0) + args.max_range
    tls = tls[np.all((tls >= lo) & (tls <= hi), axis=1)]
    tt = cKDTree(tls)
    test = sorted(int(x) for x in yaml.safe_load(open(f"{args.root}/splits.yaml"))["lidars"][lid]["test_in"])
    data = WindowData(args.root, test)
    cases = [(f"lateral {float(x):g} m", np.array([0.0, float(x), 0.0])) for x in args.offsets.split(",")]
    cases += [(f"hauteur +{float(x):g} m", np.array([0.0, 0.0, float(x)])) for x in args.heights.split(",") if x]
    res = {k: {"err": [], "nerr": [], "cover": [], "n_poses": 0, "n_rejetees": 0} for k, _ in cases}
    for s in test[::args.every]:
        sw = data.sweep(s)
        T = np.asarray(data.traj.matrices([sw["t_mid"]])[0][0]) @ data.T_bl
        for d, vec in cases:
            for side in ((1,) if vec[1] == 0 else (1, -1)):
                Tv = T.copy()
                Tv[:3, 3] = T[:3, 3] + T[:3, :3] @ (vec * np.array([1.0, side, 1.0]))
                if np.any(vec) and tt.query(Tv[:3, 3])[0] < 0.3:
                    res[d]["n_rejetees"] += 1
                    continue
                g, win = tls_grid(tt, tls, Tv, model, args.max_range)
                V = torch.tensor(np.linalg.inv(Tv), dtype=torch.float32, device=dev)
                with torch.no_grad():
                    r, a, _ = gsplat.rasterization(
                        means, quats, scales, opac, col0, V[None], torch.eye(3, device=dev)[None], lm.n_columns,
                        lm.n_rows, camera_model="lidar", lidar_coeffs=lm, with_ut=True, with_eval3d=True,
                        packed=False, render_mode="RGB-Ed", global_z_order=False, far_plane=300.0, sh_degree=0)
                ds = r[0, ..., 3].cpu().numpy()
                al = a[0, ..., 0].cpu().numpy()
                has_t = np.isfinite(g)
                both = has_t & (al > 0.5) & (ds < args.max_range)
                e = (ds - g)[both]
                pw = tls[win[both]]
                _, ii = tt.query(pw, k=12, workers=-1)
                nb = tls[ii]
                c = np.einsum("nki,nkj->nij", nb - nb.mean(1)[:, None], nb - nb.mean(1)[:, None])
                nrm = np.linalg.eigh(c)[1][:, :, 0]
                u = (pw - Tv[:3, 3]) / np.linalg.norm(pw - Tv[:3, 3], axis=1, keepdims=True)
                res[d]["err"].append(e)
                res[d]["nerr"].append(np.abs(e) * np.abs((u * nrm).sum(1)))
                res[d]["cover"].append(float(both.sum() / max(has_t.sum(), 1)))
                res[d]["n_poses"] += 1
    out = {"scene": args.scene or "carte initiale", "max_range_m": args.max_range, "decalages": {}}
    for d, _ in cases:
        e = np.concatenate(res[d]["err"]) if res[d]["err"] else np.zeros(0)
        ne = np.concatenate(res[d]["nerr"]) if res[d]["nerr"] else np.zeros(0)
        out["decalages"][d] = {
            "n_poses": res[d]["n_poses"], "n_rejetees_dans_un_objet": res[d]["n_rejetees"],
            "apparies_10cm": round(float(np.mean(np.abs(e) <= 0.10)), 4) if len(e) else None,
            "mediane_abs_cm": round(float(np.median(np.abs(e))) * 100, 2) if len(e) else None,
            "mediane_signee_cm": round(float(np.median(e)) * 100, 2) if len(e) else None,
            "normale_mediane_cm": round(float(np.median(ne)) * 100, 2) if len(ne) else None,
            "normale_frac_3cm": round(float(np.mean(ne <= 0.03)), 4) if len(ne) else None,
            "couverture": round(float(np.mean(res[d]["cover"])), 4) if res[d]["cover"] else None}
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
