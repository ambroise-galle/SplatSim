"""DiTer++ — carte LiDAR 5 cm de la tranche (équivalent des étapes 1.5–1.7 d'Oxford) et comparaison au scan TLS.

Entrée : scène canonique écrite par `stage0/ingest_diterpp.py` (balayages deskewés dans le repère TLS).
Sortie : `map/map_5cm_static.ply` (même format que la carte Oxford : x,y,z,nx,ny,nz,n_points,n_scans,intensity,
last_echo_frac,flatness ; lignes triées par `pack_ijk`), `reports/stage1_map.json`.

Filtre « statique » : voxel vu dans ≥ `--min-scans` tours distincts (sur Oxford : DUFOMap). Sur la pelouse, peu
d'objets mobiles ; à vérifier sur les images (personnes).

Comparaison au TLS (`Lawn_GT.pcd`), par type de surface (normale du voxel) :
- précision : distance de chaque voxel au plus proche point TLS ;
- complétude : part des points TLS à ≤ 15 m de la trajectoire qui ont un voxel à ≤ 10 cm (borne basse : inclut les
  zones jamais vues depuis la trajectoire) ;
- décalage vertical signé du sol (herbe) : voxel − plan TLS local.

  venv/bin/python -m splatpipe.stage1.diterpp_map --scene data_local/diterpp/lawn/scene_upper_day_604_784 \
      --tls data_local/diterpp/lawn/Lawn_GT.pcd
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

VOXEL = 0.05


def map_5cm(xyz, intensity, scan_id, origins, voxel=VOXEL, k_normals=16):
    """Copie de `stage1.deskew_map.map_5cm` (ce module importe modal) : centroïde, normale (ACP des k centroïdes
    voisins, orientée vers les capteurs), nb de points, nb de tours distincts, intensité moyenne, planéité."""
    from scipy.spatial import cKDTree

    from splatpipe.stage1.maptools import knn_normals, pack_ijk

    keys = pack_ijk(np.floor(xyz / voxel).astype(np.int64))
    order = np.lexsort((scan_id, keys))
    ks, ss = keys[order], scan_id[order]
    new = np.r_[True, ks[1:] != ks[:-1]]
    starts = np.flatnonzero(new)
    n = np.diff(np.r_[starts, len(ks)])
    new_scan = new | np.r_[True, ss[1:] != ss[:-1]]
    n_scans = np.add.reduceat(new_scan.astype(np.int64), starts)
    first_scan, last_scan = ss[starts], ss[np.r_[starts[1:], len(ss)] - 1]
    del ks, ss, new, new_scan, keys
    p = xyz[order]
    cen = np.add.reduceat(p, starts, axis=0) / n[:, None]
    view = origins[scan_id[order]] - p
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    view = np.add.reduceat(view, starts, axis=0)
    del p
    inten = np.add.reduceat(intensity[order].astype(np.float64), starts) / n
    tree = cKDTree(cen, balanced_tree=False, compact_nodes=False)
    nrm, flat = knn_normals(cen, tree, k=k_normals)
    flip = np.sum(nrm * view, axis=1) < 0
    nrm[flip] *= -1
    return {"x": cen[:, 0].astype(np.float32), "y": cen[:, 1].astype(np.float32), "z": cen[:, 2].astype(np.float32),
            "nx": nrm[:, 0], "ny": nrm[:, 1], "nz": nrm[:, 2], "n_points": n.astype(np.uint32),
            "n_scans": n_scans.astype(np.uint32), "intensity": inten.astype(np.float32),
            "last_echo_frac": np.zeros(len(n), np.float32), "flatness": flat,
            "_first_scan": first_scan.astype(np.int32), "_last_scan": last_scan.astype(np.int32)}


def carve_free(scene: str, files: list[str], keys_sorted: np.ndarray, nrm: np.ndarray, step: float = 0.025,
               margin: float = 0.25, max_range: float = 40.0, min_cos: float = 0.3, every: int = 1) -> np.ndarray:
    """Espace libre (GPU) : pour chaque tour, les rayons origine → point sont échantillonnés tous les `step` m
    jusqu'à `margin` m avant l'impact ; un voxel occupé traversé compte comme « vu libre » dans ce tour si le rayon
    le traverse franchement (|cos(rayon, normale)| ≥ `min_cos` : un rayon rasant le sol de la pelouse n'est pas une
    preuve). Renvoie le nombre de tours où chaque voxel a été vu libre."""
    import torch

    from splatpipe.stage1.maptools import KEY_BITS, KEY_OFF

    dev = "cuda"
    ks = torch.from_numpy(keys_sorted).to(dev)
    nr = torch.from_numpy(nrm.astype(np.float32)).to(dev)
    free = torch.zeros(len(ks), dtype=torch.int32, device=dev)
    big = 1 << 30
    f_first = torch.full((len(ks),), big, dtype=torch.int32, device=dev)
    f_last = torch.full((len(ks),), -1, dtype=torch.int32, device=dev)
    for i, f in enumerate(files[::every]):
        with np.load(f) as z:
            p = torch.from_numpy(z["xyz"].astype(np.float64)).to(dev)
            a = torch.from_numpy((z["t_rel"] / max(float(z["t_end"] - z["t_start"]), 1e-6)).astype(np.float64)).to(dev)
            o0 = torch.from_numpy(z["T_map_lidar_start"][:3, 3]).to(dev)
            o1 = torch.from_numpy(z["T_map_lidar_end"][:3, 3]).to(dev)
        o = o0 + a[:, None] * (o1 - o0)
        v = p - o
        r = v.norm(dim=1)
        k = (r <= max_range) & (r > margin + step)
        o, u, r = o[k], v[k] / r[k, None], r[k]
        n_s = ((r - margin) / step).floor().long()
        tot = int(n_s.sum())
        seen = []
        for c0 in range(0, len(r), 4096):                      # paquets de rayons (mémoire GPU)
            c1 = min(c0 + 4096, len(r))
            ns = n_s[c0:c1]
            ray = torch.repeat_interleave(torch.arange(c0, c1, device=dev), ns)
            first = torch.cumsum(ns, 0) - ns
            kk = torch.arange(len(ray), device=dev) - torch.repeat_interleave(first, ns) + 1
            q = o[ray] + u[ray] * (kk.double() * step)[:, None]
            ijk = torch.floor(q / 0.05).long() + KEY_OFF
            key = (ijk[:, 0] << (2 * KEY_BITS)) | (ijk[:, 1] << KEY_BITS) | ijk[:, 2]
            j = torch.searchsorted(ks, key).clamp(max=len(ks) - 1)
            hit = ks[j] == key
            j, ray = j[hit], ray[hit]
            cos = (u[ray].float() * nr[j]).sum(1).abs()
            seen.append(j[cos >= min_cos])
        if seen:
            js = torch.unique(torch.cat(seen))
            free[js] += 1
            f_first[js] = torch.minimum(f_first[js], torch.tensor(i * every, dtype=torch.int32, device=dev))
            f_last[js] = i * every
        if i % 200 == 0:
            print(f"[carve] {i}/{len(files[::every])} échantillons {tot}", flush=True)
    return free.cpu().numpy(), f_first.cpu().numpy(), f_last.cpu().numpy()


def height_above_ground(xyz: np.ndarray, keep: np.ndarray, cell: float = 0.5, q: float = 0.02) -> np.ndarray:
    """Hauteur au-dessus du sol local : grille `cell` m, sol = quantile `q` des z des voxels gardés de la cellule."""
    ij = np.floor(xyz[:, :2] / cell).astype(np.int64)
    key = (ij[:, 0] + 1_000_000) * 2_000_000 + (ij[:, 1] + 1_000_000)
    _, inv = np.unique(key, return_inverse=True)
    kk = np.flatnonzero(keep)
    order = kk[np.lexsort((xyz[kk, 2], inv[kk]))]          # voxels gardés triés par (cellule, z)
    ii = inv[order]
    starts = np.flatnonzero(np.r_[True, ii[1:] != ii[:-1]])
    cnt = np.diff(np.r_[starts, len(ii)])
    ground = np.full(inv.max() + 1, np.nan)
    ground[ii[starts]] = xyz[order[starts + np.floor(q * (cnt - 1)).astype(np.int64)], 2]
    return xyz[:, 2] - ground[inv]


def evaluate(scene: str, tls_path: str, cols: dict) -> dict:
    """Carte vs TLS. Distances au PLAN TLS local (k = 16 voisins, voisinage plan λmin/Σλ < 0,01 et rayon < 15 cm),
    séparées murs / sol par la normale TLS : la distance au plus proche point TLS est gonflée par l'espacement du
    scan et par le centroïde du voxel. Sol : écart signé par tranche de temps (pelouse puis route)."""
    from scipy.spatial import cKDTree

    from splatpipe.common.pcd import read_pcd

    t = read_pcd(tls_path)
    tls = np.stack([t["x"], t["y"], t["z"]], 1).astype(np.float64)
    cen = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    lo, hi = cen.min(0) - 5, cen.max(0) + 5
    tls = tls[np.all((tls >= lo) & (tls <= hi), axis=1)]
    tt = cKDTree(tls)
    rep = {}
    sp = tt.query(tls[::50], k=2, workers=-1)[0][:, 1]
    rep["tls_espacement_cm_mediane"] = float(np.median(sp) * 100)
    d1 = tt.query(cen, workers=-1)[0]
    rep["voxel_vers_point_tls"] = {"mediane_cm": float(np.median(d1) * 100), "frac_10cm": float((d1 < 0.10).mean())}
    dp = np.full(len(cen), np.nan)
    sgn = np.full(len(cen), np.nan)
    nzt = np.full(len(cen), np.nan)
    for a in range(0, len(cen), 500_000):
        q = cen[a:a + 500_000]
        dd, ii = tt.query(q, k=16, workers=-1)
        nb = tls[ii]
        mu = nb.mean(1)
        c = np.einsum("nki,nkj->nij", nb - mu[:, None], nb - mu[:, None]) / 16
        lam, vec = np.linalg.eigh(c)
        n0 = vec[:, :, 0] * np.where(vec[:, 2:3, 0] < 0, -1, 1)
        planar = (lam[:, 0] / lam.sum(1).clip(1e-12) < 0.01) & (dd[:, -1] < 0.15)
        s_ = np.einsum("ni,ni->n", q - mu, n0)
        dp[a:a + 500_000] = np.where(planar, np.abs(s_), np.nan)
        sgn[a:a + 500_000] = np.where(planar, s_, np.nan)
        nzt[a:a + 500_000] = np.abs(n0[:, 2])
    ok = np.isfinite(dp)
    for lab, m in (("murs", ok & (nzt < 0.3)), ("sol", ok & (nzt > 0.9))):
        rep[f"plan_{lab}"] = {"frac_voxels": float(m.mean()), "mediane_cm": float(np.median(dp[m]) * 100),
                              "p90_cm": float(np.percentile(dp[m], 90) * 100), "frac_2cm": float((dp[m] < 0.02).mean()),
                              "frac_5cm": float((dp[m] < 0.05).mean())}
    # sol : écart signé (positif = carte LiDAR au-dessus du TLS) par tranche de 30 s le long de la trajectoire
    tr = np.load(f"{scene}/traj/spline.npz")
    tp, tt_ = tr["pos"][::40], tr["t"][::40]
    g = ok & (nzt > 0.9)
    jt = cKDTree(tp).query(cen[g], workers=-1)[1]
    tv = tt_[jt] - tt_[0]
    rep["sol_ecart_signe_cm_par_30s"] = {
        f"{int(b)}-{int(b) + 30} s": {"n": int(((tv >= b) & (tv < b + 30)).sum()),
                                      "mediane": float(np.median(sgn[g][(tv >= b) & (tv < b + 30)]) * 100)}
        for b in range(0, int(tv.max()) + 1, 30) if ((tv >= b) & (tv < b + 30)).sum() > 100}
    near_traj = cKDTree(tp).query(tls, workers=-1)[0] <= 15.0
    dc = cKDTree(cen).query(tls[near_traj], workers=-1)[0]
    rep["completude_tls_15m"] = {"n_points_tls": int(near_traj.sum()), "frac_10cm": float((dc < 0.10).mean()),
                                 "frac_20cm": float((dc < 0.20).mean())}
    return rep


def corridor_mask(scene, cen, keep, n_scans, free, args):
    """Objets qui accompagnent le robot (personne qui marche à côté ou derrière lui, cf. note) : voxels à moins de
    `corridor_m` m (horizontalement) de la trajectoire, à plus de 10 cm du sol, vus libres franchement dans ≥ 5 tours
    et touchés dans moins de 30 % des tours où ils sont observés. Hors couloir, rien n'est retiré (le feuillage, vu à
    travers, serait pris pour du dynamique)."""
    from scipy.spatial import cKDTree

    hits = n_scans.astype(np.float64)
    hag = height_above_ground(cen, keep)
    tr = np.load(f"{scene}/traj/spline.npz")["pos"][::10]
    dtr = cKDTree(tr[:, :2]).query(cen[:, :2], workers=-1)[0]
    dyn = keep & (dtr < args.corridor_m) & (hag > 0.1) & (free >= 5) & (hits / (hits + free) < 0.3)
    return dyn, {"mode": "couloir", "corridor_m": args.corridor_m, "n_dynamiques": int(dyn.sum()),
                 "frac_voxels_couloir": float((keep & (dtr < args.corridor_m) & (hag > 0.1)).sum() / max(keep.sum(), 1))}


def dynamic_mask(cen, keep, n_scans, h_first, h_last, free, f_first, f_last, args):
    """Objets mobiles : voxel touché pendant une courte période (≤ `dyn_hit_span` tours), vu libre franchement
    (≥ `dyn_min_free` tours, part des touches < `dyn_max_ratio`) AVANT ou APRÈS cette période (≥ 2 s d'écart), et à
    plus de `dyn_min_hag` m du sol (la couche d'herbe, traversée par les rayons rasants, n'est pas un objet mobile).
    Le feuillage, semi-transparent, est touché pendant toute la durée où il est visible : critère temporel."""
    hits = n_scans.astype(np.float64)
    hag = height_above_ground(cen, keep)
    span = (h_last - h_first).astype(np.int64)
    before = (f_first >= 0) & (f_first <= h_first - 20)
    after = (f_last >= 0) & (f_last >= h_last + 20)
    dyn = (keep & (free >= args.dyn_min_free) & (hits / (hits + free) < args.dyn_max_ratio)
           & (span <= args.dyn_hit_span) & (before | after) & (hag > args.dyn_min_hag))
    info = {"dyn_min_free": args.dyn_min_free, "dyn_max_ratio": args.dyn_max_ratio,
            "dyn_hit_span_tours": args.dyn_hit_span, "dyn_min_hag_m": args.dyn_min_hag,
            "frac_voxels_vus_libres": float((free > 0).mean()), "n_dynamiques": int(dyn.sum()),
            "hauteur_sol_dynamiques_m_p10_p50_p90": np.percentile(hag[dyn], [10, 50, 90]).round(2).tolist()
            if dyn.any() else None}
    return dyn, info


def main():
    from splatpipe.stage1.maptools import pack_ijk, write_ply

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tls", default="", help="carte de référence (DiTer++) ; sans : pas de comparaison")
    ap.add_argument("--min-scans", type=int, default=3)
    ap.add_argument("--max-range", type=float, default=40.0)
    ap.add_argument("--carve", action="store_true", help="retire les voxels vus libres (objets mobiles), GPU")
    ap.add_argument("--dyn-min-free", type=int, default=5)
    ap.add_argument("--dyn-max-ratio", type=float, default=0.2, help="dynamique si hits/(hits+libre) < ratio")
    ap.add_argument("--dyn-hit-span", type=int, default=40, help="durée max des touches (tours, 10 Hz)")
    ap.add_argument("--dyn-min-hag", type=float, default=0.3, help="hauteur min au-dessus du sol (m)")
    ap.add_argument("--corridor-m", type=float, default=0.0,
                    help="avec --carve : ne retire que les voxels vus libres dans ce couloir autour de la trajectoire")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    out_ply = f"{args.scene}/map/map_5cm_static.ply"
    if args.eval_only:
        from splatpipe.stage4.overfit_core import _ply_header

        off, n, props = _ply_header(out_ply)
        mm = np.memmap(out_ply, dtype=np.dtype([(nm, "<" + t) for nm, t in props]), mode="r", offset=off, shape=(n,))
        cols = {nm: np.asarray(mm[nm]) for nm, _ in props}
        rep = json.load(open(f"{args.scene}/reports/stage1_map.json")) if os.path.exists(
            f"{args.scene}/reports/stage1_map.json") else {}
        rep.update(evaluate(args.scene, args.tls, cols))
        json.dump(rep, open(f"{args.scene}/reports/stage1_map.json", "w"), indent=1, ensure_ascii=False)
        print(json.dumps(rep, indent=1, ensure_ascii=False))
        return
    os.makedirs(f"{args.scene}/map", exist_ok=True)
    os.makedirs(f"{args.scene}/reports", exist_ok=True)
    from splatpipe.train.data import lidar_paths

    files = sorted(glob.glob(f"{args.scene}/{lidar_paths(args.scene)[1]}/*.npz"))
    xyz, inten, sid, orig = [], [], [], []
    for i, f in enumerate(files):
        with np.load(f) as z:
            p = z["xyz"]
            o = 0.5 * (z["T_map_lidar_start"][:3, 3] + z["T_map_lidar_end"][:3, 3])
            k = np.linalg.norm(p - o, axis=1) <= args.max_range
            xyz.append(p[k])
            inten.append(z["intensity"][k])
            sid.append(np.full(int(k.sum()), i, np.int32))
            orig.append(o)
    xyz = np.concatenate(xyz).astype(np.float64)
    inten = np.concatenate(inten)
    sid = np.concatenate(sid)
    orig = np.array(orig)
    n_pts = len(xyz)
    m = map_5cm(xyz, inten, sid, orig, voxel=VOXEL)
    del xyz, inten, sid
    rep = {"n_tours": len(files), "n_points": int(n_pts), "n_voxels_bruts": int(len(m["x"])),
           "min_scans": args.min_scans}
    keep = m["n_scans"] >= args.min_scans
    if args.carve:
        cen = np.stack([m["x"], m["y"], m["z"]], 1).astype(np.float64)
        keys = pack_ijk(np.floor(cen / VOXEL).astype(np.int64))
        assert np.all(np.diff(keys) > 0), "voxels non triés par clé"
        free, f_first, f_last = carve_free(args.scene, files, keys, np.stack([m["nx"], m["ny"], m["nz"]], 1),
                                           max_range=args.max_range)
        np.savez_compressed(f"{args.scene}/map/carve_counts.npz", free=free, f_first=f_first, f_last=f_last,
                            n_scans=m["n_scans"], h_first=m["_first_scan"], h_last=m["_last_scan"],
                            xyz=cen.astype(np.float32))
        if args.corridor_m > 0:
            dyn, info = corridor_mask(args.scene, cen, keep, m["n_scans"], free, args)
        else:
            dyn, info = dynamic_mask(cen, keep, m["n_scans"], m["_first_scan"], m["_last_scan"], free, f_first,
                                     f_last, args)
        m["n_free"] = free.astype(np.uint32)
        rep["espace_libre"] = info
        keep &= ~dyn
    for k in ("_first_scan", "_last_scan"):
        m.pop(k)
    cols = {k: v[keep] for k, v in m.items()}
    rep["n_voxels_statiques"] = int(keep.sum())
    os.makedirs(f"{args.scene}/map", exist_ok=True)
    write_ply(out_ply, cols,
              comments=[f"voxels 5 cm, n_scans >= {args.min_scans}, portee <= {args.max_range} m"
                        + (", espace libre (carve)" if args.carve else ""), f"scene {os.path.basename(args.scene)}"])
    if args.tls:
        rep.update(evaluate(args.scene, args.tls, cols))
    os.makedirs(f"{args.scene}/reports", exist_ok=True)
    json.dump(rep, open(f"{args.scene}/reports/stage1_map.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
