"""M5 — ablations A–H sur labels humains M4 (tourne sur la desktop, sans Modal, sklearn requis).

Ablations : A géométrie | B A+physique | C M2 | D DINO | E B+C | F C+D | G B+C+D | H B+D.
Primaire : macro-F1 test + IC bootstrap 95 % (10 000, paires) ; ΔF1 = abl − C (globale,
par classe, par strate n_vues). Règle stride 1 : macro-F1(F) − macro-F1(C) ≥ +0,03
et IC bootstrap excluant 0.

Features DINO : l'agrégat desktop `feat_sum` (fp16, somme de tokens L2-normalisés) est centré
par la moyenne globale μ (sp avec cnt_all > 0) puis L2-normalisé par sp — même recette que les
métriques M3 ; sp sans feature -> vecteur nul (imputation identique pour toutes les ablations).
AUCUNE feature n'est ajustée sur le test : StandardScaler + PCA-32 sont ajustés sur le train
(folds LOBO pour l'hyperparamètre, puis train complet pour l'évaluation finale unique).

Deps desktop :  venv/bin/pip install scikit-learn
Lancement :
  cd /home/ambroise/dino && venv/bin/python splatpipe/stage2/m5_ablations.py \
      --labels out/m4/labels.csv
Sorties : out/m5/results.json, out/m5/confusion_<abl>.png, out/m5/delta_f1.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/ambroise/dino")
from splatpipe.stage1.maptools import pack_ijk

DATA_DIR = os.environ.get("DINO_DATA_DIR", "/home/ambroise/dino/data_local/scene")
OUT_DIR = os.environ.get("DINO_OUT_DIR", "/home/ambroise/dino/out")
SP_NPZ = f"{DATA_DIR}/reports/stage2_material/superpoints.npz"
M2_SEM = f"{DATA_DIR}/reports/stage2_material/m2/superpoint_semantics.npz"
FEAT_NPZ = f"{OUT_DIR}/superpoint_features_desktop.npz"
MAP_PLY = f"{DATA_DIR}/map/map_5cm_static.ply"
M4_DIR = f"{OUT_DIR}/m4"
M5_DIR = f"{OUT_DIR}/m5"
MATERIALS = ["foliage", "trunk", "grass", "metal", "concrete", "brick", "rock", "glass", "other"]
ABLATIONS = ["A", "B", "C", "D", "E", "F", "G", "H"]
DINO_ABLS = {"D", "F", "G", "H"}
STRATA = [(0, 10, "faible"), (10, 40, "moyen"), (40, 10**9, "eleve")]
N_BOOT = 10_000


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


def grouped_minmax(vals, sp_s, order):
    """min/max par sp (valeurs triées par sp, `order` = tri secondaire par valeur) — pas de ufunc.at."""
    v, s2 = vals[order], sp_s[order]
    starts = np.flatnonzero(np.r_[True, s2[1:] != s2[:-1]])
    ends = np.r_[starts[1:], len(v)]
    out_min = np.zeros(int(sp_s[-1]) + 1)
    out_max = np.full(int(sp_s[-1]) + 1, -np.inf)
    out_min[s2[starts]] = v[starts]
    out_max[s2[starts]] = v[ends - 1]
    return out_min, out_max


def dino_labeled_rows(lab_ids):
    """-> (D[len(lab_ids),768] centrée μ + L2, cnt_all[lab_ids], n_vues[lab_ids]).

    μ : moyenne des agrégats (feat_sum/cnt_all) sur les sp avec cnt_all > 0.
    feat_sum (fp16, 3,5 Go) est décompressé UNE fois (un accès npz = décompression totale).
    """
    lab_ids = np.asarray(lab_ids, np.int64)
    with np.load(FEAT_NPZ) as z:
        feat_sum = z["feat_sum"].copy()   # (n_sp, 768) fp16 — copie : assainissement
        cnt_all = z["cnt_all"]
        n_vues_lab = z["n_vues"][lab_ids].copy()
    # 149 lignes (gros sps) débordent du fp16 (max 65504) -> inf, ce qui empoisonnerait
    # μ (et donc TOUTES les features) : clip, puis μ sur les lignes finies.
    bad = ~np.isfinite(feat_sum).all(1)
    if bad.any():
        blk_bad = feat_sum[bad]
        feat_sum[bad] = np.where(np.isfinite(blk_bad), blk_bad, np.float16(65504))
    n = cnt_all.shape[0]
    CH = 300_000
    s, npos = np.zeros(768), 0
    for a in range(0, n, CH):
        b = min(a + CH, n)
        blk = feat_sum[a:b].astype(np.float32)
        m = cnt_all[a:b] > 0
        s += blk[m].sum(0, dtype=np.float64)
        npos += int(m.sum())
    mu = (s / max(npos, 1)).astype(np.float32)
    D = feat_sum[lab_ids].astype(np.float32)
    del feat_sum
    has = cnt_all[lab_ids] > 0
    D[has] -= mu
    D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-6)
    D[~has] = 0.0
    return D, cnt_all[lab_ids].copy(), n_vues_lab


def macro_f1(y_true, y_pred):
    """(macro-F1, {classe: f1}) — classes présentes dans y_true ∪ y_pred (convention sklearn)."""
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    f1s = {}
    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        f1s[c] = 2 * tp / max(2 * tp + fp + fn, 1)
    return float(np.mean(list(f1s.values()))), f1s


def load_labels(path):
    """-> (pass1 {sp_id: (material, unsure)}, pass2 {sp_id: material})."""
    p1, p2 = {}, {}
    with open(path) as f:
        for row in csv.DictReader(f):
            mat = row["material"].strip()
            if not mat:
                continue
            sid, p = int(row["sp_id"]), (row["pass"].strip() or "1")
            if p == "2":
                p2.setdefault(sid, mat)
            else:
                p1.setdefault(sid, (mat, int(row["unsure"].strip() or 0)))
    return p1, p2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=f"{M4_DIR}/labels.csv")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(M5_DIR, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # ---------------------------------------------------------------- labels + split
    labels, labels_p2 = load_labels(args.labels)
    lab_ids = np.array(sorted(labels.keys()), dtype=np.int64)
    y_lab = np.array([labels[i][0] for i in lab_ids])
    unknown = sorted(set(y_lab.tolist()) - set(MATERIALS))
    if unknown:
        raise SystemExit(f"matériaux inconnus dans labels.csv : {unknown} (taxonomie : {MATERIALS})")
    with open(f"{M4_DIR}/splits.json") as f:
        splits = json.load(f)
    test_sp = np.array(splits["test_sp"], dtype=np.int64)
    is_test_row = np.isin(lab_ids, test_sp)
    rows_te, rows_tr = np.flatnonzero(is_test_row), np.flatnonzero(~is_test_row)
    y_te, y_tr = y_lab[rows_te], y_lab[rows_tr]
    n_te = len(rows_te)
    print(f"labels : {len(lab_ids)} (train {len(rows_tr)}, test {n_te}) ; "
          f"répétitions pass 2 : {len(labels_p2)}", flush=True)
    if n_te == 0 or len(rows_tr) == 0:
        raise SystemExit("labels.csv incomplet : annoter au moins le train et le test")

    # ---------------------------------------------------------------- tables sp (depuis la carte)
    # sp_id (superpoints.npz) est aligné aux LIGNES du PLY (fichier déjà trié par clés
    # pack_ijk) mais non monotone (ids attribués dans un autre parcours) : regrouper par
    # argsort(sp_id) — jamais reduceat sur l'ordre fichier.
    cols = read_ply_points(MAP_PLY, ["x", "y", "z", "nx", "ny", "nz", "intensity", "last_echo_frac"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    nrm = np.stack([cols["nx"], cols["ny"], cols["nz"]], 1).astype(np.float64)
    inten, lef = cols["intensity"].astype(np.float64), cols["last_echo_frac"].astype(np.float64)
    o = np.argsort(pack_ijk(np.floor(xyz / 0.05).astype(np.int64)), kind="stable")
    xyz_s, nrm_s, inten_s, lef_s = xyz[o], nrm[o], inten[o], lef[o]
    del xyz, nrm, inten, lef, cols
    with np.load(SP_NPZ) as z:
        sp_id = z["sp_id"]
    n_sp = int(sp_id.max()) + 1
    cnt = np.bincount(sp_id, minlength=n_sp).astype(np.float64)
    og = np.argsort(sp_id, kind="stable")
    sp_g = sp_id[og]
    starts = np.flatnonzero(np.r_[True, sp_g[1:] != sp_g[:-1]])
    assert sp_g[0] == 0 and len(starts) == n_sp, "sp_id non aligné aux lignes du PLY ?"
    n_tok = np.diff(np.r_[starts, len(sp_g)]).astype(np.float64)
    xyz_g, nrm_g, inten_g, lef_g = xyz_s[og], nrm_s[og], inten_s[og], lef_s[og]
    del xyz_s, nrm_s, inten_s, lef_s, sp_id, og
    cen = np.zeros((n_sp, 3))
    cen[sp_g[starts]] = np.add.reduceat(xyz_g, starts, axis=0) / cnt[:, None]
    z_m = np.add.reduceat(xyz_g[:, 2], starts) / n_tok
    z2 = np.add.reduceat(xyz_g[:, 2] ** 2, starts) / n_tok
    z_std = np.sqrt(np.maximum(z2 - z_m**2, 0))
    z_min, z_max = grouped_minmax(xyz_g[:, 2], sp_g, np.lexsort((xyz_g[:, 2], sp_g)))
    x_min, x_max = grouped_minmax(xyz_g[:, 0], sp_g, np.lexsort((xyz_g[:, 0], sp_g)))
    y_min, y_max = grouped_minmax(xyz_g[:, 1], sp_g, np.lexsort((xyz_g[:, 1], sp_g)))
    del xyz_g
    n_mean = np.zeros((n_sp, 3))
    n_mean[sp_g[starts]] = np.add.reduceat(nrm_g, starts, axis=0) / cnt[:, None]
    dot = np.einsum("ij,ij->i", nrm_g, n_mean[sp_g])
    n_var = np.add.reduceat((1.0 - np.abs(dot)) ** 2, starts) / n_tok
    del nrm_g, dot
    i_m = np.add.reduceat(inten_g, starts) / n_tok
    i2 = np.add.reduceat(inten_g**2, starts) / n_tok
    lef_m = np.add.reduceat(lef_g, starts) / n_tok
    del inten_g, lef_g, sp_g, starts

    with np.load(M2_SEM) as z:
        dom = z["dominant_group"][lab_ids]
        wsum = z["wsum"][lab_ids].astype(np.float64)

    # ---------------------------------------------------------------- features (lignes labellisées)
    D, cnt_lab, n_vues_lab = dino_labeled_rows(lab_ids)
    geo_A = np.stack([np.log(np.maximum(cnt[lab_ids], 1)), z_m[lab_ids], z_std[lab_ids],
                      (z_max - z_min)[lab_ids], (x_max - x_min)[lab_ids], (y_max - y_min)[lab_ids],
                      n_mean[lab_ids, 0], n_mean[lab_ids, 1], n_mean[lab_ids, 2],
                      n_var[lab_ids]], 1).astype(np.float32)
    phys = np.stack([i_m[lab_ids], np.sqrt(np.maximum(i2 - i_m**2, 0))[lab_ids],
                     lef_m[lab_ids]], 1).astype(np.float32)
    m2_feat = np.zeros((len(lab_ids), 15), np.float32)
    for g in range(14):
        m2_feat[dom == g, g] = 1.0
    m2_feat[:, 14] = np.clip(wsum / np.maximum(cnt[lab_ids], 1), 0, 1)
    FEATS = {
        "A": geo_A, "B": np.hstack([geo_A, phys]), "C": m2_feat, "D": D,
        "E": np.hstack([geo_A, phys, m2_feat]),
        "F": np.hstack([m2_feat, D]),
        "G": np.hstack([geo_A, phys, m2_feat, D]),
        "H": np.hstack([geo_A, phys, D]),
    }
    print(f"features prêtes : {len(lab_ids)} sp ; DINO dense {int((cnt_lab > 0).sum())}/{len(lab_ids)}",
          flush=True)

    # ---------------------------------------------------------------- classifieur + CV LOBO
    def fit_predict(abl, c_reg, tr, te):
        X = FEATS[abl]
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        if abl in DINO_ABLS:
            mu = Xtr.mean(0)
            Xtr = Xtr - mu
            _, _, vt = np.linalg.svd(Xtr, full_matrices=False)
            vt = vt[:32]
            Xtr, Xte = Xtr @ vt.T, (Xte - mu) @ vt.T
        clf = LogisticRegression(max_iter=2000, C=c_reg, class_weight="balanced")
        clf.fit(Xtr, y_tr[tr])
        return clf.predict(Xte)

    from scipy.cluster.vq import kmeans2

    _, block_lab = kmeans2(cen[lab_ids[rows_tr]], 6, minit="++", seed=0)
    rows_by_block = [np.flatnonzero(block_lab == bi) for bi in range(6)]
    print("blocs LOBO :", [len(b) for b in rows_by_block], flush=True)

    results = {abl: {} for abl in ABLATIONS}
    preds = {}
    for abl in ABLATIONS:
        best_c, best_cv = 1.0, -1.0
        for c_reg in (0.1, 1.0, 10.0):
            f1s = []
            for bi in range(6):
                te_loc = rows_by_block[bi]
                tr_loc = np.concatenate([rows_by_block[b] for b in range(6) if b != bi])
                if len(te_loc) == 0 or len(np.unique(y_tr[tr_loc])) < 2:
                    continue
                yp = fit_predict(abl, c_reg, tr_loc, te_loc)
                f1s.append(macro_f1(y_tr[te_loc], yp)[0])
            if f1s and np.mean(f1s) > best_cv:
                best_cv, best_c = float(np.mean(f1s)), c_reg
        yp_te = fit_predict(abl, best_c, rows_tr, rows_te)
        preds[abl] = yp_te
        mf1, f1_cls = macro_f1(y_te, yp_te)
        bacc = float(np.mean([(yp_te[y_te == c] == c).mean() for c in sorted(set(y_te.tolist()))]))
        results[abl] = {"C_reg": best_c, "cv_macro_f1": round(best_cv, 4),
                        "test_macro_f1": round(mf1, 4), "test_bal_acc": round(bacc, 4),
                        "f1_par_classe": {k: round(v, 3) for k, v in sorted(f1_cls.items())}}
        print(f"{abl} : C={best_c}, CV {best_cv:.3f}, test macro-F1 {mf1:.3f}", flush=True)
        classes = sorted(set(y_te.tolist()) | set(yp_te.tolist()))
        cm = np.zeros((len(classes), len(classes)))
        for a, b in zip(y_te, yp_te):
            cm[classes.index(a), classes.index(b)] += 1
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor="white")
        im = ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_title(f"ablation {abl} — macro-F1 {mf1:.3f} (test, n={n_te})", fontsize=9, loc="left")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.savefig(f"{M5_DIR}/confusion_{abl}.png", dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ---------------------------------------------------------------- ΔF1 vs C : bootstrap pairé
    rng = np.random.default_rng(0)
    boot = {abl: np.empty(N_BOOT) for abl in ABLATIONS if abl != "C"}
    for i in range(N_BOOT):
        idx = rng.integers(0, n_te, n_te)
        f_c = macro_f1(y_te[idx], preds["C"][idx])[0]
        for abl in boot:
            boot[abl][i] = macro_f1(y_te[idx], preds[abl][idx])[0] - f_c
    results["delta_vs_C"] = {}
    for abl, d in boot.items():
        results["delta_vs_C"][abl] = {
            "mean": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
        }
    d_f = boot["F"]
    results["decision_stride1"] = {
        "regle": "macro-F1(F) − macro-F1(C) ≥ +0.03 ET IC95 excluant 0",
        "delta_mean": round(float(d_f.mean()), 4),
        "ci95": results["delta_vs_C"]["F"]["ci95"],
        "verdict": ("LANCER" if (d_f.mean() >= 0.03 and np.percentile(d_f, 2.5) > 0)
                    else "NE PAS LANCER"),
    }

    # ΔF1 par strate n_vues (point)
    results["delta_vs_C_par_strate"] = {"F": {}, "D": {}}
    for lo, hi, name in STRATA:
        m = (n_vues_lab[rows_te] >= lo) & (n_vues_lab[rows_te] < hi)
        if m.sum() < 5:
            continue
        f_c = macro_f1(y_te[m], preds["C"][m])[0]
        for abl in ("D", "F"):
            f_a = macro_f1(y_te[m], preds[abl][m])[0]
            results["delta_vs_C_par_strate"][abl][name] = {
                "n": int(m.sum()), "f1_C": round(f_c, 3), "f1_" + abl: round(f_a, 3),
                "delta": round(f_a - f_c, 3)}

    # cohérence annotation pass 2
    if labels_p2:
        agree = sum(labels[i][0] == m2 for i, m2 in labels_p2.items() if i in labels)
        results["self_consistency_pass2"] = {
            "n": sum(1 for i in labels_p2 if i in labels), "agree": int(agree),
            "frac": round(agree / max(sum(1 for i in labels_p2 if i in labels), 1), 3)}

    results["timing_s"] = round(time.time() - t0, 1)
    with open(f"{M5_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    xs = ABLATIONS
    ys = [results[a]["test_macro_f1"] for a in xs]
    ax.bar(xs, ys, color=["#999"] * 3 + ["#47c"] * 4 + ["#999"], alpha=0.85)
    for i, v in enumerate(ys):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("macro-F1 test")
    ax.set_title("M5 — ablations (test spatial séparé, labels humains)", fontsize=10, loc="left")
    ax.set_ylim(0, 1)
    fig.savefig(f"{M5_DIR}/delta_f1.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
