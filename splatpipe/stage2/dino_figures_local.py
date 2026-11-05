"""Figures + métriques M3 en local, version économe mémoire (pic ~4 Go).

Entrée : superpoint_features_desktop.npz (features fp16 par superpoint, par caméra pour la surface).
Sorties : logs/stage2/material/m3_local/ — metrics.json, cos_hist, pca_top + 2 zooms, retrieval.

Lancement :  PYTHONPATH=. .venv/bin/python splatpipe/stage2/dino_figures_local.py
"""

from __future__ import annotations

import argparse
import json
import os
import time

import sys

import numpy as np

sys.path.insert(0, "/home/ambroise/dino")

SEQ = "2024-03-13-observatory-quarter-01"
SP_NPZ = "data_local/scene/reports/stage2_material/superpoints.npz"
M2_SEM = "data_local/scene/reports/stage2_material/m2/superpoint_semantics.npz"
MAP_PLY = "data_local/scene/map/map_5cm_static.ply"
OUT_DIR = "logs/stage2/material/m3_local"
GROUP_NAMES = {0: "sky", 1: "person", 2: "bike", 3: "vehicle", 4: "foliage", 5: "ground-naturel",
               6: "ground-pave", 7: "batiment", 8: "vitrage", 9: "grille-barriere", 10: "poteaux",
               11: "roche-eau", 12: "mobilier-urbain", 13: "autre"}
QUERY_GROUPS = [2, 4, 7, 6, 12]  # bike, foliage, batiment, ground-pave, mobilier-urbain
CHUNK = 400_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/superpoint_features_desktop.npz")
    ap.add_argument("--nn", type=int, default=50)
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from splatpipe.stage1.maptools import pack_ijk

    def read_ply_points(path, fields):
        """PLY binaire little-endian (copie locale ; superpoints.py importe Modal)."""
        ply_t = {"float": "f4", "double": "f8", "uchar": "u1", "char": "i1", "ushort": "u2",
                 "short": "i2", "uint": "u4", "int": "i4"}
        with open(path, "rb") as f:
            lines = []
            while True:
                ln = f.readline()
                if not ln:
                    raise ValueError("en-tete PLY inacheve")
                lines.append(ln)
                if ln.strip() == b"end_header":
                    break
            offset = f.tell()
        fmt, n, props = None, 0, []
        for ln in lines:
            t = ln.decode("ascii", "replace").strip().split()
            if not t:
                continue
            if t[0] == "format":
                fmt = t[1]
            elif t[0] == "element" and t[1] == "vertex":
                n = int(t[2])
            elif t[0] == "property":
                props.append((t[2], ply_t[t[1]]))
        assert fmt == "binary_little_endian", fmt
        arr = np.fromfile(path, dtype=np.dtype(props), count=n, offset=offset)
        return {k: arr[k] for k in fields}


    def read_ply_points(path, fields):
        """PLY binaire little-endian (copie locale ; superpoints.py importe Modal)."""
        ply_t = {"float": "f4", "double": "f8", "uchar": "u1", "char": "i1", "ushort": "u2",
                 "short": "i2", "uint": "u4", "int": "i4"}
        with open(path, "rb") as f:
            lines = []
            while True:
                ln = f.readline()
                if not ln:
                    raise ValueError("en-tete PLY inacheve")
                lines.append(ln)
                if ln.strip() == b"end_header":
                    break
            offset = f.tell()
        fmt, n, props = None, 0, []
        for ln in lines:
            t = ln.decode("ascii", "replace").strip().split()
            if not t:
                continue
            if t[0] == "format":
                fmt = t[1]
            elif t[0] == "element" and t[1] == "vertex":
                n = int(t[2])
            elif t[0] == "property":
                props.append((t[2], ply_t[t[1]]))
        assert fmt == "binary_little_endian", fmt
        arr = np.fromfile(path, dtype=np.dtype(props), count=n, offset=offset)
        return {k: arr[k] for k in fields}

    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    # --- chargement par blocs : feat_mean (centrée + normalisée) et compteurs
    with np.load(args.npz) as z:
        n_sp = int(z["m2_dominant_group"].shape[0])
        feat_cam16 = z["feat_cam"]
        cnt_cam = z["cnt_cam_surf"]
        cnt_all = z["cnt_all"].astype(np.int64)
        surf_idx = z["surf_idx"]
        m2_dom = z["m2_dominant_group"]
    with np.load(args.npz) as z:
        feat_sum16 = z["feat_sum"]
        fin_all = np.isfinite(feat_sum16).all(axis=1)
    seen_all = cnt_all > 0
    okm = seen_all & fin_all
    feat_mean = np.zeros((n_sp, 768), np.float16)
    for a0 in range(0, n_sp, CHUNK):
        a1 = min(a0 + CHUNK, n_sp)
        m = okm[a0:a1]
        if not m.any():
            continue
        blk = feat_sum16[a0:a1][m].astype(np.float32) / np.maximum(cnt_all[a0:a1][m], 1)[:, None]
        feat_mean[a0:a1][m] = blk.astype(np.float16)
    del feat_sum16

    # moyenne globale mu (sur les features centrables), puis centrage + normalisation
    mu = np.zeros(768, np.float64)
    for a0 in range(0, n_sp, CHUNK):
        a1 = min(a0 + CHUNK, n_sp)
        mu += feat_mean[a0:a1][okm[a0:a1]].astype(np.float64).sum(0)
    mu /= max(int(okm.sum()), 1)
    for a0 in range(0, n_sp, CHUNK):
        a1 = min(a0 + CHUNK, n_sp)
        m = okm[a0:a1]
        blk = feat_mean[a0:a1][m].astype(np.float32) - mu.astype(np.float32)
        blk /= np.maximum(np.linalg.norm(blk, axis=1, keepdims=True), 1e-6)
        feat_mean[a0:a1][m] = blk.astype(np.float16)
    print("feat_mean centrée + normalisée", flush=True)

    # --- cohérence intra-superpoint : cosinus inter-caméras après centrage PAR CAMÉRA
    fnc = np.zeros_like(feat_cam16, dtype=np.float16)  # surface seulement (102 794, 3, 768)
    for k in range(3):
        blk = feat_cam16[:, k, :].astype(np.float32)
        blk[~np.isfinite(blk)] = 0
        blk -= mu.astype(np.float32)
        blk /= np.maximum(np.linalg.norm(blk, axis=1, keepdims=True), 1e-6)
        fnc[:, k, :] = blk.astype(np.float16)
    del feat_cam16
    cos_all, cos_rand = [], []
    surf_ok = fin_all[surf_idx]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        m = np.flatnonzero((cnt_cam[:, i] >= 8) & (cnt_cam[:, j] >= 8) & surf_ok)
        for a0 in range(0, len(m), 100_000):
            sel = m[a0:a0 + 300_000]
            ai = fnc[sel, i].astype(np.float32)
            aj = fnc[sel, j].astype(np.float32)
            cs = (ai * aj).sum(1)
            cos_all.append(cs)
            perm = rng_perm(len(cs))
            ajp = aj[perm]
            ajp /= np.maximum(np.linalg.norm(ajp, axis=1, keepdims=True), 1e-6)
            cos_rand.append((ai * ajp).sum(1))
        print(f"paire ({i},{j}) : {int(len(m)):,} sp comparables", flush=True)
    cos_all = np.concatenate(cos_all)
    cos_rand = np.concatenate(cos_rand)
    del fnc
    rep = {"cos_intra_sp_mean": round(float(cos_all.mean()), 4),
           "cos_intra_sp_p10": round(float(np.percentile(cos_all, 10)), 4),
           "cos_random_mean": round(float(cos_rand.mean()), 4),
           "cos_gap": round(float(cos_all.mean() - cos_rand.mean()), 4),
           "n_comparable_pairs": int(len(cos_all)),
           "coverage_surface": round(float((cnt_all[surf_idx] > 0).mean()), 4),
           "coverage_all": round(float(seen_all.mean()), 4),
           "non_finite_sp_all": int((~fin_all).sum())}

    # --- PCA-RGB : mu/matrice centrage appliqués par blocs
    cols = read_ply_points(MAP_PLY, ["x", "y", "z"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    from splatpipe.stage1.maptools import pack_ijk

    keys = pack_ijk(np.floor(xyz / 0.05).astype(np.int64))
    o = np.argsort(keys, kind="stable")
    xyz_s = xyz[o].astype(np.float32)
    with np.load(SP_NPZ) as z:
        sp_id = z["sp_id"]
        sizes = np.bincount(z["sp_id"], minlength=n_sp)
    seen_s = seen_all[sp_id]
    surf_s = seen_s & (sizes[sp_id] >= 20)
    # PCA / retrieval au niveau SUPERPOINT (103 k) — pas au niveau voxel (8,6 M -> OOM)
    surf_sp = np.flatnonzero(sizes >= 20)
    fin_surf_sp = fin_all[surf_sp]
    Fc = feat_mean[surf_sp].astype(np.float32)
    Fc[~fin_surf_sp] = 0
    Fc = Fc - Fc.mean(0, keepdims=True)
    sub = rng.choice(len(Fc), min(60_000, len(Fc)), replace=False)
    _, _, Vt = np.linalg.svd(Fc[sub], full_matrices=False)
    var = (Vt @ Fc[sub].T).var(axis=1)
    var = var / var.sum()
    P = Fc @ Vt[:3].T
    lo, hi = np.percentile(P, [1, 99], axis=0)
    Pn = np.clip((P - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    rgb_sp = np.full((n_sp, 3), 0.85, np.float32)
    oksp = fin_surf_sp
    rgb_sp[surf_sp[oksp]] = Pn[oksp]
    rgbvox = rgb_sp[sp_id]

    # --- retrieval : superpoints de surface, features centrées normalisées
    Fs = feat_mean[surf_sp].astype(np.float32)
    Fs[~fin_surf_sp] = 0
    Fs /= np.maximum(np.linalg.norm(Fs, axis=1, keepdims=True), 1e-6)
    surf_dom = m2_dom[surf_sp]
    queries = {}
    for g in QUERY_GROUPS:
        cand = np.flatnonzero((surf_dom == g) & (cnt_all[surf_sp] >= 30))
        if len(cand):
            queries[g] = int(cand[rng.integers(len(cand))])

    idxr = rng.choice(n_sp, 1_800_000, replace=False)

    def scatter(ax, P2, vcol, title, s=0.08):
        ax.scatter(P2[:, 0], P2[:, 1], c=vcol, s=s, marker=".", rasterized=True, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9, loc="left")
        ax.tick_params(labelsize=7)

    P2 = xyz_s[idxr][:, :2]
    # figure PCA
    fig, ax = plt.subplots(figsize=(10, 9), facecolor="white")
    scatter(ax, P2, rgbvox[idxr].astype(np.float32),
            f"M3 — PCA(RGB) des features DINOv2 par superpoint (couverture surface "
            f"{rep['coverage_surface']*100:.0f} %)")
    ax.set_xlabel("x map (m)", fontsize=8)
    ax.set_ylabel("y map (m)", fontsize=8)
    fig.savefig(f"{OUT_DIR}/m3_pca_top.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # histogramme cosinus
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    bins = np.linspace(-1, 1, 80)
    ax.hist(cos_rand, bins=bins, alpha=0.6, label="paires aléatoires", color="#bbbbbb")
    ax.hist(cos_all, bins=bins, alpha=0.7, label="paires intra-superpoint (caméras)", color="#2a78d6")
    ax.set_xlabel("cosinus inter-caméras (centré par caméra)", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title(f"cohérence intra-superpoint : {rep['cos_intra_sp_mean']} vs aléatoire "
                 f"{rep['cos_random_mean']}", fontsize=9, loc="left")
    fig.savefig(f"{OUT_DIR}/m3_cos_hist.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # retrieval : 5 requêtes
    n_q = len(queries)
    fig, axes = plt.subplots(n_q, 1, figsize=(9, 4.0 * n_q), facecolor="white")
    retrieval = {}
    for ax2, (g, q) in zip(np.atleast_1d(axes), queries.items()):
        cs = Fs @ Fs[q]
        nn = np.argsort(-cs)[: args.nn]
        nn_set = np.zeros(len(Fs), bool)
        nn_set[nn] = True
        same = int((surf_dom[nn] == g).sum())
        retrieval[GROUP_NAMES[g]] = {"query_sp": int(surf_sp[q]), "nn": args.nn,
                                     "same_group_nn": same}
        vcol = np.full((n_sp, 3), 0.92, np.float32)
        vsurf = np.full((len(Fs), 3), 0.92, np.float32)
        vsurf[nn] = (0.95, 0.35, 0.12)
        vsurf[q] = (1.0, 0.0, 0.0)
        vv = np.full((n_sp, 3), 0.92, np.float32)
        vv[surf_sp] = vsurf
        vv[surf_sp[q]] = (1.0, 0.0, 0.0)
        ax2.scatter(P2[:, 0], P2[:, 1], c=vv[idxr], s=0.08, marker=".",
                    rasterized=True, linewidths=0)
        ax2.set_aspect("equal")
        ax2.set_title(f"requête « {GROUP_NAMES[g]} » (sp {surf_sp[q]}) — "
                      f"{same}/{args.nn} NN du même groupe M2", fontsize=9, loc="left")
        ax2.tick_params(labelsize=6)
    fig.savefig(f"{OUT_DIR}/m3_retrieval.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    rep["retrieval"] = {k: v for k, v in retrieval.items()}
    rep["timing_s"] = round(time.time() - t0, 1)
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def rng_perm(n: int):
    return np.random.default_rng(1).permutation(n)


if __name__ == "__main__":
    main()
