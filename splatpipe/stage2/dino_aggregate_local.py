"""Agrégation LOCALE des features DINOv2 produites sur le desktop (`superpoint_features_desktop.npz`).

Calcule : moyennes par superpoint, cohérence intra-superpoint (cosinus inter-caméras vs aléatoire),
couverture, rendu PCA-RGB de la carte (dessus + 2 zooms), démo de récupération (requêtes auto-pickées
par classe M2 dominante -> 50 plus proches voisins, colorés sur la carte).

Lancement :  PYTHONPATH=. .venv/bin/python splatpipe/stage2/dino_aggregate_local.py --npz <chemin npz>
Sorties :    logs/stage2/material/m3_local/ (rapport json + figures)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

SEQ = "2024-03-13-observatory-quarter-01"
MAP_PLY = "data_local/scene/map/map_5cm_static.ply"
SP_NPZ = "data_local/scene/reports/stage2_material/superpoints.npz"
M2_SEM = "data_local/scene/reports/stage2_material/m2/superpoint_semantics.npz"
OUT_DIR = "logs/stage2/material/m3_local"
GROUP_NAMES = {0: "sky", 1: "person", 2: "bike", 3: "vehicle", 4: "foliage", 5: "ground-naturel",
               6: "ground-pave", 7: "batiment", 8: "vitrage", 9: "grille-barriere", 10: "poteaux",
               11: "roche-eau", 12: "mobilier-urbain", 13: "autre"}
QUERY_GROUPS = [2, 4, 6, 7, 12]  # bike, foliage, ground-pave, batiment, mobilier-urbain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/superpoint_features_desktop.npz")
    ap.add_argument("--n-queries", type=int, default=5)
    ap.add_argument("--nn", type=int, default=50)
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
                    raise ValueError(f"en-tête PLY inachevé : {path}")
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
    read_ply_full = read_ply_points  # fonction locale définie plus haut

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    with np.load(args.npz) as z:
        n_sp = int(z["m2_dominant_group"].shape[0])
        feat_sum16 = z["feat_sum"]
        feat_cam16 = z["feat_cam"]
        cnt_cam = z["cnt_cam_surf"]
        cnt_all = z["cnt_all"].astype(np.int64)
        surf_idx = z["surf_idx"]
        m2_dom = z["m2_dominant_group"]
        m2_wsum = z["m2_wsum"].astype(np.float32)
        n_vues = z["n_vues"]
        cam_bits = z["cam_bits"]
    seen_all = cnt_all > 0
    fin_all = np.isfinite(feat_sum16).all(axis=1)   # lecture seulement
    fin_surf = np.isfinite(feat_cam16).all(axis=(1, 2))
    surf_lut_full = None

    # moyennes
    feat_mean = np.zeros((n_sp, 768), np.float16)
    for a0 in range(0, n_sp, 600_000):
        a1 = min(a0 + 600_000, n_sp)
        m = seen_all[a0:a1] & fin_all[a0:a1]
        if not m.any():
            continue
        block = feat_sum16[a0:a1][m].astype(np.float32) / np.maximum(cnt_all[a0:a1][m], 1)[:, None]
        feat_mean[a0:a1][m] = block.astype(np.float16)
    del feat_sum16
    okm = seen_all & fin_all
    mu = feat_mean[okm].astype(np.float32).mean(0)
    for a0 in range(0, n_sp, 600_000):
        a1 = min(a0 + 600_000, n_sp)
        m = okm[a0:a1]
        blk = feat_mean[a0:a1][m].astype(np.float32) - mu
        blk /= np.maximum(np.linalg.norm(blk, axis=1, keepdims=True), 1e-6)
        feat_mean[a0:a1][m] = blk.astype(np.float16)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(f"{OUT_DIR}/superpoint_features.npz", feat=feat_mean,
                        feat_cam=feat_cam16, cnt_all=cnt_all.astype(np.uint32), cnt_cam=cnt_cam,
                        n_vues=n_vues, m2_dominant_group=m2_dom.astype(np.int16),
                        m2_wsum=m2_wsum)
    print("superpoint_features.npz sauvegardé", flush=True)

    # cohérence intra-superpoint : cosinus inter-caméras APRÈS centrage (les features DINOv2 brutes
    # partagent une direction commune — cosinus aléatoire ~0.74 sinon, la métrique ne mesurerait rien)
    mu = feat_mean[okm].mean(0)
    feat_mean[okm] -= mu
    nn2 = np.maximum(np.linalg.norm(feat_mean[okm], axis=1, keepdims=True), 1e-6)
    feat_mean[okm] /= nn2
    fnc = np.zeros_like(feat_cam16, dtype=np.float16)
    for k in range(3):
        blk = feat_cam16[:, k, :].astype(np.float32)
        blk[~np.isfinite(blk)] = 0
        blk -= mu
        blk /= np.maximum(np.linalg.norm(blk, axis=1, keepdims=True), 1e-6)
        fnc[:, k, :] = blk.astype(np.float16)
    wsum_c = cnt_cam
    cos_all, cos_rand = [], []
    cos_random_mean = []
    rngl = np.random.default_rng(1)
    cam_off = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        m = (wsum_c[:, i] >= 8) & (wsum_c[:, j] >= 8) & fin_surf
        idx_m = np.flatnonzero(m)
        # centrage PAR CAMÉRA : chaque fisheye voit un contenu systématiquement différent
        # (avant/gauche/droite) -> ses nuages sont décalés ; on retire le décalage moyen
        # de chaque caméra AVANT le cosinus, sinon le décalage domine la mesure
        mi = fnc[idx_m, i].astype(np.float32).mean(0)
        mj = fnc[idx_m, j].astype(np.float32).mean(0)
        cam_off.append((round(float(np.linalg.norm(mi)), 3), round(float(np.linalg.norm(mj)), 3),
                        round(float((mi @ mj) / max(np.linalg.norm(mi) * np.linalg.norm(mj), 1e-9)), 3)))
        for a0 in range(0, len(idx_m), 300_000):
            sel = idx_m[a0:a0 + 300_000]
            ai = fnc[sel, i].astype(np.float32) - mi
            aj = fnc[sel, j].astype(np.float32) - mj
            ai /= np.maximum(np.linalg.norm(ai, axis=1, keepdims=True), 1e-6)
            aj /= np.maximum(np.linalg.norm(aj, axis=1, keepdims=True), 1e-6)
            cs = (ai * aj).sum(1)
            cos_all.append(cs)
            perm = rngl.permutation(len(cs))
            ajp = aj[perm]
            ajp /= np.maximum(np.linalg.norm(ajp, axis=1, keepdims=True), 1e-6)
            cos_rand.append((ai * ajp).sum(1))
        print(f"paire ({i},{j}) : {int(m.sum()):,} sp comparables", flush=True)
    cos_all = np.concatenate(cos_all)
    cos_rand = np.concatenate(cos_rand)
    rep = {
        "n_superpoints": n_sp,
        "coverage_surface": round(float((cnt_all[surf_idx] > 0).mean()), 4),
        "coverage_all": round(float(seen_all.mean()), 4),
        "cos_intra_sp_mean": round(float(cos_all.mean()), 4),
        "cos_intra_sp_p10": round(float(np.percentile(cos_all, 10)), 4),
        "cos_random_mean": round(float(cos_rand.mean()), 4),
        "cos_gap": round(float(cos_all.mean() - cos_rand.mean()), 4),
        "n_comparable_pairs": int(len(cos_all)),
        "non_finite_sp": {"all": int((~fin_all).sum()), "surface": int((~fin_surf).sum())},
        "timing_s": round(time.time() - t0, 1),
    }
    Fd = feat_mean[seen_all & fin_all].astype(np.float32)
    subd = Fd[rngl.integers(0, len(Fd), 20_000)]
    _, sv, _ = np.linalg.svd(subd, full_matrices=False)
    var = sv**2 / (sv**2).sum()
    rep["svd_variance_pc1_5"] = [round(float(v), 4) for v in var[:5]]
    rep["effective_rank"] = round(float(np.exp(-(var * np.log(var + 1e-12)).sum())), 1)
    print("spectre PC1-5:", rep["svd_variance_pc1_5"], "| rang effectif:", rep["effective_rank"])
    print(json.dumps(rep, indent=1))

    # rendus : PCA-RGB de la carte (dessus + 2 zooms) + histogramme cosinus
    with np.load(SP_NPZ) as z:
        sp_id = z["sp_id"]
        sizes = np.bincount(z["sp_id"], minlength=n_sp)
    seen_s = seen_all[sp_id]
    cols = read_ply_full(MAP_PLY, ["x", "y", "z"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    from splatpipe.stage1.maptools import pack_ijk

    keys = pack_ijk(np.floor(xyz / 0.05).astype(np.int64))
    o = np.argsort(keys, kind="stable")
    xyz_s = xyz[o].astype(np.float32)
    feat_v = np.zeros((n_sp, 768), np.float32)
    feat_v[seen_all & fin_all] = feat_mean[seen_all & fin_all]
    feat_s = feat_v[sp_id]
    surf_s = seen_s & (sizes[sp_id] >= 20)
    F = feat_s[surf_s]
    F = F - F.mean(0, keepdims=True)
    sub = np.random.default_rng(0).choice(len(F), min(60_000, len(F)), replace=False)
    _, _, Vt = np.linalg.svd(F[sub], full_matrices=False)
    P = F @ Vt[:3].T
    lo, hi = np.percentile(P, [1, 99], axis=0)
    Pn = np.clip((P - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    rgbp = np.full((n_sp, 3), 0.85, np.float32)
    rgbp[surf_s] = Pn
    rgbvox = rgbp[sp_id]

    rng0 = np.random.default_rng(0)
    idxr = rng0.choice(n_sp, 1_800_000, replace=False)

    def render(ax, xa, ya, title):
        ax.scatter(xyz_s[idxr, xa], xyz_s[idxr, ya], c=rgbvox[idxr], s=0.08, marker=".",
                   rasterized=True, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9, loc="left")
        ax.tick_params(labelsize=7)

    fig, ax = plt.subplots(figsize=(10, 9), facecolor="white")
    render(ax, 0, 1, f"M3 [local] — PCA(RGB) des features DINOv2 par superpoint de surface")
    ax.set_xlabel("x map (m)", fontsize=8)
    ax.set_ylabel("y map (m)", fontsize=8)
    fp = f"{OUT_DIR}/m3_pca_top.png"
    fig.savefig(fp, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    bins = np.linspace(-1, 1, 80)
    ax.hist(cos_rand, bins=bins, alpha=0.6, label="paires aléatoires", color="#bbbbbb")
    ax.hist(cos_all, bins=bins, alpha=0.7, label="paires intra-superpoint (caméras)", color="#2a78d6")
    ax.axvline(0.7, color="#c8380d", linestyle="--", linewidth=1, label="cible 0,7")
    ax.set_xlabel("cosinus inter-caméras", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title(f"cohérence intra-superpoint : {rep['cos_intra_sp_mean']} vs aléatoire "
                 f"{rep['cos_random_mean']}", fontsize=9, loc="left")
    fc = f"{OUT_DIR}/m3_cos_hist.png"
    fig.savefig(fc, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # récupération : requêtes = sp de surface dominés par certains groupes M2, 50-NN cosinus
    Fs = feat_s[surf_s]
    Fs = Fs / np.maximum(np.linalg.norm(Fs, axis=1, keepdims=True), 1e-6)
    surf_dom = m2_dom[surf_s]
    fig, axes = plt.subplots(len(QUERY_GROUPS), 1, figsize=(9, 4.2 * len(QUERY_GROUPS)), facecolor="white")
    retrieval = {}
    for qi, (ax2, g) in enumerate(zip(np.atleast_1d(axes), QUERY_GROUPS)):
        cand = np.flatnonzero((surf_dom == g) & (cnt_all[surf_s] >= 30))
        if len(cand) == 0:
            continue
        q = cand[rng0.integers(len(cand))]
        cs = Fs @ Fs[q]
        nn = np.argsort(-cs)[: args.nn]
        nn_set = np.zeros(len(Fs), bool)
        nn_set[nn] = True
        same_group = int((surf_dom[nn] == g).sum())
        retrieval[GROUP_NAMES[g]] = {"query_sp": int(surf_idx[q]), "nn": args.nn,
                                     "same_group_nn": same_group}
        col = np.full((len(Fs), 3), 0.9, np.float32)
        col[nn] = np.array([0.95, 0.3, 0.1])
        col[q] = np.array([1.0, 0.0, 0.0])
        vcol = col[surf_s][sp_id]
        ax2.scatter(xyz_s[idxr, 0], xyz_s[idxr, 1], c=vcol[idxr], s=0.08, marker=".",
                    rasterized=True, linewidths=0)
        ax2.set_aspect("equal")
        ax2.set_title(f"requête « {GROUP_NAMES[g]} » (sp {surf_idx[q]}) — "
                      f"{same_group}/{args.nn} NN du même groupe M2", fontsize=9, loc="left")
        ax2.tick_params(labelsize=6)
    fr = f"{OUT_DIR}/m3_retrieval.png"
    fig.savefig(fr, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    rep["retrieval"] = retrieval
    np.savez_compressed(f"{OUT_DIR}/superpoint_features.npz", feat=feat_mean.astype(np.float16),
                        feat_cam=cam_mean, cnt=cnt.astype(np.uint32), cnt_cam=cnt_cam, n_vues=n_vues,
                        cam_bits=cam_bits, m2_dominant_group=m2_dom.astype(np.int16),
                        m2_wsum=m2_wsum)
    with open(f"{OUT_DIR}/m3_local_report.json", "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print("figures écrites dans", OUT_DIR, f"({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
