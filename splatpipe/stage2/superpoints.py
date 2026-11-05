"""Tâche 2.5.1 (jalon M1 du plan 2.5) — superpoints du nuage statique.

Méthode (v3, après deux tentatives documentées). Chaque voxel 5 cm de `map/map_5cm_static.ply` (points
dyn == 0 uniquement) est relié à ses voisins de grille (26-adjacence, chaque paire une fois : demi-voisinage
de 13 décalages). Deux écarts par arête : normale (degrés ; normales orientées capteur, donc deux faces
opposées d'un mur s'écartent de ~180°) et intensité (absolu).

  v1 — composantes connexes des arêtes sous les seuils du plan (25°, 2·IQR) : FAIL réel — chaînage transitif
       (un superpoint de 82 % de la carte par pas de < 25°) ; et 2·IQR = 212 inopérant (IQR mesuré = 105,8).
       Archivé : `logs/stage2/material/v1_cc_fail/`.
  v2 — Felzenszwalb–Huttenlocher (poids w = θ/25° + ΔI/30) : rejeté sans run réel — le diagnostic montre un
       bruit de poids intra-surface médian ≈ 0,1–0,3 (θ stable p50 = 2,5° mais p75 = 3,8°, plus ΔI) : le
       critère adaptatif min(int+k/|C|) gèle la croissance à quelques voxels dès que k/|C| < bruit, et aucun
       k unique ne convient (sélectif au départ ⇒ k < 4 ; croissance à 500 voxels ⇒ k > 250).
  v3 — **cœur fort + rattachement lâche**, fondé sur le diagnostic (`logs/stage2/material/diagnose.json`) :
       sur surfaces stables (n_scans ≥ 3, flatness < 0,15) les écarts de normale k16 sont p50 = 2,5°,
       p90 = 5,3° ; sur toutes les arêtes p50 = 25° (végétation/bords dominants). Le balayage CC montre
       qu'à θ ≤ 5° le plus grand composant ne couvre que 1,0 % des voxels (pas de blob) : un seuil de cœur
       bas coupe net le chaînage.
       1. Cœur : composantes connexes des arêtes θ ≤ `theta_core_deg` ET ΔI ≤ `dint_core` ;
       2. Îlots (< `min_size` voxels) fusionnés par vagues dans le composant voisin le plus peuplé, mais
          seulement à travers des arêtes « lâches » θ ≤ `theta_loose_deg` ET ΔI ≤ `dint_loose` — les
          fragments de végétation réels restent donc des superpoints petits au lieu d'être vidangés dans
          le sol/les murs (v1 faisait ça) ;
       3. Les voxels sans aucune voisine (3,6 % — bruit volant) restent des superpoints de taille 1,
          comptés à part.

Deux jeux de paramètres évalués dans la même passe (arêtes et écarts calculés une fois) :
  A « défaut » : cœur 6° / ΔI 30 ; lâche 25° / ΔI 60 ; min 20 voxels ;
  B « fin »    : cœur 4° / ΔI 20 ; lâche 25° / ΔI 60 ; min 20 voxels.
Décision A/B documentée dans `superpoints_params.json` (champ `chosen`, posé après inspection visuelle via
`finalize_main` ; les fichiers canoniques `superpoints.npz` / `superpoints_colored.ply` sont la copie du jeu
retenu, pour les jalons M2+).

Sorties dans `reports/stage2_material/` : `superpoints_<set>.npz` (clés de voxel triées → sp_id),
`superpoints_<set>.ply` (couleur aléatoire par superpoint, inspection CloudCompare), `superpoints_params.json`,
rendus PNG (dessus + deux profils). Copies locales : `logs/stage2/material/`.

Lancement :
  .venv/bin/modal run splatpipe/stage2/superpoints.py::sp_main
  .venv/bin/modal run splatpipe/stage2/superpoints.py::finalize_main --chosen A_defaut
"""


from __future__ import annotations

import numpy as np

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume
from splatpipe.stage1.images import map_image

app = modal.App("splatsim-stage2-superpoints")
vol = data_volume()

SEQ = "2024-03-13-observatory-quarter-01"
MAP_PLY = "map/map_5cm_static.ply"
OUT_SUB = "reports/stage2_material"
VOXEL = 0.05
SETS = {
    "A_defaut": {"theta_core_deg": 6.0, "dint_core": 30.0, "theta_loose_deg": 25.0, "dint_loose": 60.0,
                 "min_size": 20},
    "B_fin": {"theta_core_deg": 4.0, "dint_core": 20.0, "theta_loose_deg": 25.0, "dint_loose": 60.0,
              "min_size": 20},
}


# =============================================================================
# Noyau numpy (importable et testable hors Modal)
# =============================================================================

def _half_offsets() -> list[tuple[int, int, int]]:
    """13 décalages couvrant la 26-adjacence une seule fois par paire (di>0, ou di=0&dj>0, ou di=dj=0&dk>0)."""
    return [
        (di, dj, dk)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        for dk in (-1, 0, 1)
        if (di, dj, dk) != (0, 0, 0) and (di > 0 or (di == 0 and dj > 0) or (di == 0 and dj == 0 and dk > 0))
    ]


def read_ply_points(path: str, fields: list[str]) -> dict:
    """PLY binaire little-endian, un élément `vertex` → colonnes demandées (décalage lu dans l'en-tête)."""
    ply_t = {"float": "f4", "double": "f8", "uchar": "u1", "char": "i1", "ushort": "u2", "short": "i2", "uint": "u4", "int": "i4"}
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
            props.append((t[2], ply_t[t[1]]))  # np.dtype : liste de (nom, type)
    if fmt != "binary_little_endian":
        raise ValueError(f"format PLY non géré : {fmt}")
    arr = np.fromfile(path, dtype=np.dtype(props), count=n, offset=offset)
    return {k: arr[k] for k in fields}


def edge_pass(keys: np.ndarray, normals: np.ndarray, intensity: np.ndarray) -> dict:
    """Arêtes de la 26-adjacence entre voxels (clés `pack_ijk` déjà calculées).

    Renvoie, en ordre de clés triées : `keys_sorted`, `ei`/`ej` (int32, paires non orientées via le
    demi-voisinage), `ang_deg` (écart de normale) et `d_int` (écart absolu d'intensité), float32.
    """
    from splatpipe.stage1.maptools import pack_ijk, unpack_keys

    n = len(keys)
    o = np.argsort(keys, kind="stable")
    ks = keys[o]
    nrm = np.ascontiguousarray(normals[o], dtype=np.float32)
    inten = np.ascontiguousarray(intensity[o], dtype=np.float32)
    ijk = unpack_keys(ks)
    ei_l, ej_l, ang_l, di_l = [], [], [], []
    for off in _half_offsets():
        nk = pack_ijk(ijk + np.array(off, dtype=np.int64))
        pos = np.searchsorted(ks, nk)
        np.minimum(pos, n - 1, out=pos)
        m = ks[pos] == nk
        i = np.flatnonzero(m).astype(np.int32)
        j = pos[m].astype(np.int32)
        dot = np.einsum("ij,ij->i", nrm[i], nrm[j])
        ei_l.append(i)
        ej_l.append(j)
        ang_l.append(np.degrees(np.arccos(np.clip(dot.astype(np.float32), -1.0, 1.0))))
        di_l.append(np.abs(inten[i] - inten[j]))
    return {
        "keys_sorted": ks,
        "ei": np.concatenate(ei_l),
        "ej": np.concatenate(ej_l),
        "ang_deg": np.concatenate(ang_l),
        "d_int": np.concatenate(di_l),
    }


def region_grow(n_vox: int, ei: np.ndarray, ej: np.ndarray) -> np.ndarray:
    """Composantes connexes du graphe (arêtes déjà filtrées, non orientées) → étiquettes int32 compactées."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    g = coo_matrix((np.ones(len(ei), np.int8), (ei, ej)), shape=(n_vox, n_vox))
    _, lab = connected_components(g, directed=False)
    return lab.astype(np.int32)


def merge_slivers(labels: np.ndarray, ei: np.ndarray, ej: np.ndarray, min_size: int, max_rounds: int = 20) -> tuple[np.ndarray, int]:
    """Fusionne, par vagues, les étiquettes de < `min_size` voxels dans leur voisin d'adjacence le plus peuplé.

    Renvoie (étiquettes compactées 0..M-1, nb de fusions). Les unions sont appliquées via un mini union-find
    pour éviter les cycles ; les voisins sont parcourus par tranches dichotomiques (pas de balayage O(n²)).
    """
    lab = labels.astype(np.int64)
    k = int(labels.max()) + 1
    parent = np.arange(k, dtype=np.int64)

    def find(x: int) -> int:
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:  # compression de chemin
            parent[x], x = r, parent[x]
        return r

    n_merged = 0
    for _ in range(max_rounds):
        roots = parent[np.arange(k)]
        for _ in range(64):  # aplatissement vectorisé de la forêt
            nxt = parent[roots]
            if np.array_equal(nxt, roots):
                break
            roots = nxt
        lab = roots[lab]
        sizes = np.bincount(lab, minlength=k)
        small = np.unique(lab[np.isin(lab, np.flatnonzero(sizes < min_size))])
        small = small[sizes[small] > 0]
        if len(small) == 0:
            break
        pl, ql = lab[ei], lab[ej]
        m = pl != ql
        src = np.concatenate([pl[m], ql[m]])
        dst = np.concatenate([ql[m], pl[m]])
        so = np.argsort(src, kind="stable")
        src, dst = src[so], dst[so]
        lo = np.searchsorted(src, small, "left")
        hi = np.searchsorted(src, small, "right")
        unions = []
        for r, a, b in zip(small, lo, hi):  # plus petits d'abord
            if b <= a:
                continue  # îlot sans voisin
            nb = dst[a:b]
            u, cnt = np.unique(nb, return_counts=True)
            big = u[sizes[u] >= min_size]
            pool, key = (big, sizes[big]) if len(big) else (u, cnt)  # sinon : voisin le plus fréquent
            unions.append((int(r), int(pool[np.argmax(key)])))
        if not unions:
            break
        for r, t in unions:
            rr, rt = find(r), find(t)
            if rr != rt:
                if sizes[rr] >= sizes[rt]:
                    parent[rt] = rr
                else:
                    parent[rr] = rt
                n_merged += 1
    roots = parent[np.arange(k)]
    for _ in range(64):
        nxt = parent[roots]
        if np.array_equal(nxt, roots):
            break
        roots = nxt
    lab = roots[lab]
    _, lab_c = np.unique(lab, return_inverse=True)
    return lab_c.astype(np.int32), n_merged


def set_metrics(labels: np.ndarray, ei: np.ndarray, ej: np.ndarray, ang: np.ndarray, d_int: np.ndarray,
                min_size: int) -> dict:
    """Statistiques d'un partitionnement : tailles, îlots résiduels, cohérence intra-arête."""
    sizes = np.bincount(labels)
    sizes = sizes[sizes > 0]
    li, lj = labels[ei], labels[ej]
    same = li == lj
    a_in, a_out = ang[same], ang[~same]
    d_in, d_out = d_int[same], d_int[~same]

    def s(x: np.ndarray) -> dict:
        return {"n": int(len(x)), "mean": float(x.mean()) if len(x) else 0.0,
                "p50": float(np.median(x)) if len(x) else 0.0, "p90": float(np.percentile(x, 90)) if len(x) else 0.0}

    return {
        "n_superpoints": int(len(sizes)),
        "size": {"median": float(np.median(sizes)), "p90": float(np.percentile(sizes, 90)),
                 "max": int(sizes.max()), "min": int(sizes.min())},
        "largest_voxel_frac": float(sizes.max() / sizes.sum()),
        "n_residual_slivers": int((sizes < min_size).sum()),
        "sliver_voxel_frac": float(sizes[sizes < min_size].sum() / sizes.sum()),
        "intra_edge_angle_deg": s(a_in),
        "inter_edge_angle_deg": s(a_out),
        "intra_edge_dint": s(d_in),
        "inter_edge_dint": s(d_out),
    }


def palette(n: int, seed: int = 0) -> np.ndarray:
    """Couleurs aléatoires bien différenciées (teinte pas d'or, S/V variés) → uint8 [n,3]."""
    rng = np.random.default_rng(seed)
    h = (np.arange(n) * 0.618033988749895) % 1.0
    s = rng.uniform(0.55, 0.95, n)
    v = rng.uniform(0.75, 1.0, n)
    i = np.floor(h * 6.0).astype(int) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])
    return (np.stack([r, g, b], 1) * 255).astype(np.uint8)


def render_views(xyz: np.ndarray, rgb: np.ndarray, out_prefix: str, title: str, max_pts: int = 1_800_000) -> list[str]:
    """Rendus orthographiques : dessus (x,y), profils (x,z) et (y,z). Renvoie les chemins écrits."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(xyz) > max_pts:
        idx = np.random.default_rng(0).choice(len(xyz), max_pts, replace=False)
    else:
        idx = np.arange(len(xyz))
    p, c = xyz[idx], rgb[idx].astype(np.float32) / 255.0
    files = []
    for name, (a, b), (la, lb) in (
        ("top", (0, 1), ("x map (m)", "y map (m)")),
        ("side_xz", (0, 2), ("x map (m)", "z map (m)")),
        ("side_yz", (1, 2), ("y map (m)", "z map (m)")),
    ):
        fig, ax = plt.subplots(figsize=(10, 9), facecolor="white")
        ax.scatter(p[:, a], p[:, b], c=c, s=0.06, marker=".", rasterized=True, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel(la, fontsize=8)
        ax.set_ylabel(lb, fontsize=8)
        ax.tick_params(labelsize=7)
        f = f"{out_prefix}_{name}.png"
        fig.savefig(f, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        files.append(f)
    return files


# =============================================================================
# Modal
# =============================================================================

@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=16, memory=64 * 1024, timeout=3 * 3600)
def compute(seq: str = SEQ, set_names: list[str] | None = None) -> dict:
    import json
    import os
    import time

    import numpy as np

    from splatpipe.stage1.maptools import pack_ijk, scene_root, write_ply
    from splatpipe.stage2.superpoints import (
        MAP_PLY, OUT_SUB, SETS, VOXEL, edge_pass, merge_slivers, palette, read_ply_points, region_grow,
        render_views, set_metrics,
    )

    t0 = time.time()
    set_names = set_names or list(SETS)
    root = scene_root(DATA_MOUNT, seq)
    out_dir = f"{root}/{OUT_SUB}"
    os.makedirs(out_dir, exist_ok=True)

    cols = read_ply_points(f"{root}/{MAP_PLY}", ["x", "y", "z", "nx", "ny", "nz", "intensity", "n_scans", "flatness"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    nrm = np.stack([cols["nx"], cols["ny"], cols["nz"]], 1)
    inten = cols["intensity"]
    keys = pack_ijk(np.floor(xyz / VOXEL).astype(np.int64))
    n_scans, flat = cols["n_scans"], cols["flatness"]
    del cols
    rep: dict = {"stage": "2.5.1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "n_voxels": int(len(keys)), "voxel": VOXEL, "map_ply": MAP_PLY,
                 "iqr_intensity": float(np.percentile(inten, 75) - np.percentile(inten, 25))}
    print(f"[superpoints] {len(keys)} voxels, IQR intensité = {rep['iqr_intensity']:.2f}", flush=True)

    t1 = time.time()
    order = np.argsort(keys, kind="stable")
    eg = edge_pass(keys, nrm, inten)
    ks = eg["keys_sorted"]
    xyz_s = xyz[order].astype(np.float32)
    rep["edge_pass_s"] = round(time.time() - t1, 1)
    print(f"[superpoints] arêtes : {len(eg['ei'])/1e6:.1f} M en {rep['edge_pass_s']} s", flush=True)

    # Diagnostics (motivent la reformulation v2 : CC → FH, IQR → ΔI absolu)
    deg = np.bincount(eg["ei"], minlength=len(ks))
    stable = (n_scans[order] >= 3) & (flat[order] < 0.15)
    st_e = stable[eg["ei"]] & stable[eg["ej"]]
    ang, dInt = eg["ang_deg"], eg["d_int"]

    def pct(x, q) -> float:
        return float(np.percentile(x, q)) if len(x) else 0.0

    rep["diagnostics"] = {
        "n_voxels_without_edge": int((deg == 0).sum()),
        "frac_voxels_without_edge": float((deg == 0).mean()),
        "angle_deg": {f"p{q}": pct(ang, q) for q in (5, 25, 50, 75, 90, 95)},
        "angle_deg_stable_surfaces": {f"p{q}": pct(ang[st_e], q) for q in (50, 75, 90, 95)},
        "frac_edges_below": {str(t): float((ang < t).mean()) for t in (10, 15, 20, 25, 30, 45, 90)},
        "d_int": {f"p{q}": pct(dInt, q) for q in (50, 90, 99)},
        "frac_dint_above_30": float((dInt > 30).mean()),
        "intensity_voxels": {f"p{q}": pct(inten, q) for q in (1, 25, 50, 75, 99)},
    }
    rep["diagnostics"]["note"] = ("v1 (composantes connexes, seuils du plan) : 1 superpoint géant = 82 % de la "
                                  "carte (chaînage transitif), 2·IQR = 212 inopérant ; d'où FH + ΔI absolu (v2).")
    print(json.dumps(rep["diagnostics"], indent=1), flush=True)

    rep["sets"] = {}
    figures: dict[str, bytes] = {}
    for name in set_names:
        p = SETS[name]
        t2 = time.time()
        core = (ang <= p["theta_core_deg"]) & (dInt <= p["dint_core"])
        loose = (ang <= p["theta_loose_deg"]) & (dInt <= p["dint_loose"])
        lab0 = region_grow(len(ks), eg["ei"][core], eg["ej"][core])
        lab, n_merged = merge_slivers(lab0, eg["ei"][loose], eg["ej"][loose], p["min_size"])
        m = set_metrics(lab, eg["ei"], eg["ej"], ang, dInt, p["min_size"])
        m["params"] = p | {"iqr": rep["iqr_intensity"]}
        m["core_edge_frac"] = float(core.mean())
        m["n_merged_slivers"] = int(n_merged)
        m["compute_s"] = round(time.time() - t2, 1)
        rep["sets"][name] = m
        print(f"[superpoints] {name} : {m['n_superpoints']} sp, plus grand = "
              f"{m['largest_voxel_frac']*100:.1f} % des voxels, médiane {m['size']['median']} voxels, "
              f"{m['n_merged_slivers']} îlots fusionnés, {m['compute_s']} s", flush=True)

        np.savez_compressed(f"{out_dir}/superpoints_{name}.npz", keys=ks, sp_id=lab)
        rgb = palette(int(lab.max()) + 1)
        write_ply(f"{out_dir}/superpoints_{name}.ply",
                  {"x": xyz_s[:, 0], "y": xyz_s[:, 1], "z": xyz_s[:, 2],
                   "red": rgb[lab, 0], "green": rgb[lab, 1], "blue": rgb[lab, 2], "sp_id": lab.astype(np.uint32)},
                  comments=[f"SplatSim 2.5.1 superpoints set={name} "
                            f"coeur={p['theta_core_deg']}deg/{p['dint_core']}DI "
                            f"lache={p['theta_loose_deg']}deg/{p['dint_loose']}DI min={p['min_size']}",
                            f"n_superpoints={m['n_superpoints']} voxels_5cm={len(ks)} repere=map (GLIM)"])
        figs = render_views(xyz_s, rgb[lab], f"{out_dir}/superpoints_{name}", title=
                            f"2.5.1 superpoints [{name}] — {m['n_superpoints']} sp, "
                            f"médiane {m['size']['median']:.0f} voxels")
        for f in figs:
            figures[os.path.basename(f)] = open(f, "rb").read()

    rep["timing_s"] = round(time.time() - t0, 1)
    with open(f"{out_dir}/superpoints_params.json", "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    vol.commit()
    return {"report": json.loads(json.dumps(rep)), "figures": figures}


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=2, memory=16 * 1024, timeout=1800)
def finalize(chosen: str, rationale: str, seq: str = SEQ) -> dict:
    """Pose la décision A/B : met à jour le JSON et copie le jeu retenu vers les noms canoniques (M2+)."""
    import json
    import os
    import shutil
    import time

    from splatpipe.stage1.maptools import scene_root
    from splatpipe.stage2.superpoints import OUT_SUB, SETS

    if chosen not in SETS:
        raise ValueError(f"jeu inconnu : {chosen}")
    root = scene_root(DATA_MOUNT, seq)
    out_dir = f"{root}/{OUT_SUB}"
    with open(f"{out_dir}/superpoints_params.json") as f:
        rep = json.load(f)
    rep["chosen"] = {"set": chosen, "rationale": rationale,
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(f"{out_dir}/superpoints_params.json", "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    for src, dst in ((f"{out_dir}/superpoints_{chosen}.npz", f"{out_dir}/superpoints.npz"),
                     (f"{out_dir}/superpoints_{chosen}.ply", f"{out_dir}/superpoints_colored.ply")):
        if os.path.exists(dst):
            os.remove(dst)
        shutil.copy(src, dst)
    for legacy in ("superpoints_B_serre.npz", "superpoints_B_serre.ply"):  # reliques v1 (ancien nom de jeu)
        lp = f"{out_dir}/{legacy}"
        if os.path.exists(lp):
            os.remove(lp)
    vol.commit()
    return {"chosen": chosen, "rationale": rationale, "canonical": ["superpoints.npz", "superpoints_colored.ply"]}


@app.local_entrypoint()
def sp_main(seq: str = SEQ):
    import json
    import os

    out = compute.remote(seq)
    os.makedirs("logs/stage2/material", exist_ok=True)
    with open("logs/stage2/material/superpoints.json", "w") as f:
        json.dump(out["report"], f, indent=1, ensure_ascii=False)
    for name, b in out["figures"].items():
        open(f"logs/stage2/material/{name}", "wb").write(b)
    r = out["report"]
    print(json.dumps({"n_voxels": r["n_voxels"], "iqr_intensity": r["iqr_intensity"],
                      "sets": {k: {kk: v[kk] for kk in ("n_superpoints", "size", "largest_voxel_frac",
                                                        "n_residual_slivers", "sliver_voxel_frac",
                                                        "intra_edge_angle_deg", "n_merged_slivers",
                                                        "core_edge_frac", "compute_s")}
                               for k, v in r["sets"].items()}}, indent=1, ensure_ascii=False))


@app.local_entrypoint()
def finalize_main(chosen: str, rationale: str = "", seq: str = SEQ):
    import json

    out = finalize.remote(chosen, rationale, seq)
    print(json.dumps(out, indent=1, ensure_ascii=False))


# =============================================================================
# Auto-test local (sans Modal) : scène synthétique à limites connues
# =============================================================================

def _self_test() -> None:
    import tempfile

    from splatpipe.stage1.maptools import pack_ijk, write_ply
    from splatpipe.stage2.superpoints import (
        SETS, edge_pass, merge_slivers, read_ply_points, region_grow, set_metrics,
    )

    rng = np.random.default_rng(0)
    v = 0.05

    def vox(i, j, k):
        return np.stack([i * v + 0.012, j * v + 0.012, k * v + 0.012], 1)  # centroïde à l'écart des faces

    pts, nrm, inten = [], [], []
    ground = [(i, j, 0) for i in range(40) for j in range(40) if not (5 <= i < 11 and 5 <= j < 11)]
    wall = [(40, j, k) for j in range(20) for k in range(1, 21)]      # plan x, normale -x (90° du sol)
    patch = [(i, j, 0) for i in range(5, 11) for j in range(5, 11)]   # 36 voxels sombres à la place du sol
    sliver = [(i, 39, 1) for i in range(20, 25)]                      # 5 voxels inclinés 45° (hors seuil lâche)
    chain = [(60 + d, 60 + d, 0) for d in range(4)]                   # escalade diagonale (26-adjacence seule)
    iso = [(70, 70, 0)]                                               # îlot isolé (aucun voisin)
    sliver2 = [(i, 0, 1) for i in range(20, 25)]                      # 5 voxels à 15° (rattachables au sol)
    for voxels, n, i0 in ((ground, (0, 0, 1), 100.0), (wall, (-1, 0, 0), 100.0), (patch, (0, 0, 1), 40.0),
                          (sliver, (0, np.sqrt(0.5), np.sqrt(0.5)), 100.0), (chain, (0, 0, 1), 100.0),
                          (iso, (0, 0, 1), 100.0), (sliver2, (0, np.sin(np.radians(15)), np.cos(np.radians(15))), 100.0)):
        pts.append(vox(np.array(voxels)[:, 0], np.array(voxels)[:, 1], np.array(voxels)[:, 2]))
        nrm.append(np.tile(n, (len(voxels), 1)))
        inten.append(i0 + rng.uniform(-1, 1, len(voxels)))
    xyz = np.concatenate(pts)
    grp = np.repeat(np.arange(7), [len(g) for g in (ground, wall, patch, sliver, chain, iso, sliver2)])
    normals = np.concatenate(nrm).astype(np.float32)
    intensity = np.concatenate(inten).astype(np.float32)
    keys = pack_ijk(np.floor(xyz / v).astype(np.int64))
    assert len(np.unique(keys)) == len(keys)
    o = np.argsort(keys, kind="stable")
    grp_s = grp[o]  # groupes dans l'ordre des clés triées (espace de travail du graphe)

    eg = edge_pass(keys, normals, intensity)
    p = SETS["A_defaut"]
    iqr = float(np.percentile(intensity, 75) - np.percentile(intensity, 25))
    assert 0 < iqr < 6, iqr  # la tache sombre ne doit pas dominer l'IQR
    core = (eg["ang_deg"] <= p["theta_core_deg"]) & (eg["d_int"] <= p["dint_core"])
    loose = (eg["ang_deg"] <= p["theta_loose_deg"]) & (eg["d_int"] <= p["dint_loose"])
    assert core.sum() > 0 and loose.sum() > core.sum()
    lab0 = region_grow(len(keys), eg["ei"][core], eg["ej"][core])
    # avant fusion : sol, mur, tache, bavure 45°, chaîne, isolé, bavure 15°
    assert lab0.max() == 6, np.bincount(lab0)
    # 26-adjacence : la chaîne diagonale forme UNE composante (et n'en fusionne aucune autre)
    assert np.unique(lab0[grp_s == 4]).tolist() == [lab0[grp_s == 4][0]]

    lab, n_merged = merge_slivers(lab0, eg["ei"][loose], eg["ej"][loose], p["min_size"])
    n_sp = lab.max() + 1
    assert n_sp == 6, (n_sp, np.bincount(lab))  # sol(+15°), mur, tache, 45°, chaîne, isolé
    assert np.unique(lab[grp_s == 6]).tolist() == [lab[grp_s == 0][0]], "la bavure 15° doit être fusionnée dans le sol"
    assert len(np.unique(lab[grp_s == 3])) == 1, "la bavure 45° doit rester un superpoint séparé"
    assert len(np.unique(lab[grp_s == 0])) == 1 and len(np.unique(lab[grp_s == 2])) == 1
    m = set_metrics(lab, eg["ei"], eg["ej"], eg["ang_deg"], eg["d_int"], p["min_size"])
    assert m["n_residual_slivers"] == 3, m  # bavure 45° + chaîne + isolé
    # intra dominé par les arêtes internes (0°) ; ~25 soudures 15° noyées dedans.
    # « inter » mélange des coupes à 0° (intensité : sol↔tache) et 45/90° (normale) → borne basse uniquement.
    assert m["intra_edge_angle_deg"]["mean"] < 1.0 and m["inter_edge_angle_deg"]["mean"] > 30
    assert n_merged == 1
    # PLY : aller-retour écriture/lecture (format de map_5cm_static.ply)
    with tempfile.TemporaryDirectory() as dd:
        write_ply(f"{dd}/t.ply", {"x": np.arange(3, dtype=np.float32), "y": np.zeros(3, np.float32),
                                  "z": np.ones(3, np.float32), "nx": np.ones(3, np.float32),
                                  "ny": np.zeros(3, np.float32), "nz": np.zeros(3, np.float32),
                                  "n_points": np.arange(3, dtype=np.uint32), "n_scans": np.ones(3, np.uint32),
                                  "intensity": np.full(3, 7.5, np.float32)})
        back = read_ply_points(f"{dd}/t.ply", ["x", "intensity", "n_scans"])
        assert back["x"].dtype == np.float32 and back["x"].tolist() == [0.0, 1.0, 2.0]
        assert back["intensity"].tolist() == [7.5] * 3 and back["n_scans"].tolist() == [1, 1, 1]
    print("splatpipe.stage2.superpoints : auto-test OK")


if __name__ == "__main__":
    _self_test()


# =============================================================================
# Diagnostic v2 (choix de la règle de croissance sur données réelles)
# =============================================================================

@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=16, memory=64 * 1024, timeout=2 * 3600)
def diagnose(seq: str = SEQ) -> dict:
    """Distribution des poids d'arête + balayage CC (plus grand composant vs seuil), normales d'origine vs k=48."""
    import json
    import time

    import numpy as np
    from scipy.spatial import cKDTree

    from splatpipe.stage1.maptools import knn_normals, pack_ijk, scene_root
    from splatpipe.stage2.superpoints import (
        MAP_PLY, VOXEL, edge_pass, read_ply_points,
    )

    t0 = time.time()
    root = scene_root(DATA_MOUNT, seq)
    cols = read_ply_points(f"{root}/{MAP_PLY}", ["x", "y", "z", "nx", "ny", "nz", "intensity", "n_scans", "flatness"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    nrm0 = np.stack([cols["nx"], cols["ny"], cols["nz"]], 1)
    inten = cols["intensity"]
    n_scans, flat = cols["n_scans"], cols["flatness"]
    keys = pack_ijk(np.floor(xyz / VOXEL).astype(np.int64))
    order = np.argsort(keys, kind="stable")
    ks, xyz_s = keys[order], xyz[order]
    nrm0_s, inten_s = nrm0[order], inten[order]
    ns_s, fl_s = n_scans[order], flat[order]

    def angle_stats(nrm: np.ndarray, tag: str, rep: dict) -> None:
        eg = edge_pass(ks, nrm, inten_s)
        ang = eg["ang_deg"]
        stable = (ns_s >= 3) & (fl_s < 0.15)
        st_e = stable[eg["ei"]] & stable[eg["ej"]]
        d_int = eg["d_int"]
        d = {f"p{q}": float(np.percentile(ang, q)) for q in (5, 25, 50, 75, 90, 95)}
        d["p_stable"] = {f"p{q}": float(np.percentile(ang[st_e], q)) for q in (50, 75, 90, 95)}
        d["frac_below"] = {str(t): float((ang < t).mean()) for t in (5, 8, 10, 12, 15, 20, 25, 30)}
        d["d_int_p50_p90_p99"] = [float(np.percentile(d_int, q)) for q in (50, 90, 99)]
        d["frac_dint_above"] = {str(t): float((d_int > t).mean()) for t in (10, 30, 60)}
        d["frac_voxels_without_edge"] = float((np.bincount(eg["ei"], minlength=len(ks)) == 0).mean())
        rep[tag] = d
        print(f"[diag] {tag} : " + json.dumps(d), flush=True)
        # Balayage CC : composantes (nb, part du plus grand) vs seuil de normale
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        sweep = {}
        for th in (5, 8, 10, 12, 15, 20, 25):
            adm = ang <= th
            i, j = eg["ei"][adm], eg["ej"][adm]
            _, lab = connected_components(coo_matrix((np.ones(len(i), np.int8), (i, j)),
                                                     shape=(len(ks), len(ks))), directed=False)
            sizes = np.bincount(lab)
            sizes = sizes[sizes > 0]
            sweep[str(th)] = {"n_comp": int(len(sizes)), "largest_frac": float(sizes.max() / sizes.sum()),
                              "p50_size": float(np.median(sizes)), "n_ge20": int((sizes >= 20).sum())}
            print(f"[diag] {tag} CC θ≤{th}° : {sweep[str(th)]}", flush=True)
        rep[tag]["cc_sweep"] = sweep

    rep: dict = {"stage": "2.5.1-diagnostic", "n_voxels": int(len(ks)),
                 "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    angle_stats(nrm0_s, "normales_k16_carte", rep)
    t1 = time.time()
    tree = cKDTree(xyz_s, balanced_tree=False, compact_nodes=False)
    nrm48, flat48 = knn_normals(xyz_s, tree, k=48)
    rep["knn48_s"] = round(time.time() - t1, 1)
    angle_stats(nrm48, "normales_k48_recalc", rep)
    rep["intensity_voxels_p1_p25_p50_p75_p99"] = [float(np.percentile(inten_s, q)) for q in (1, 25, 50, 75, 99)]
    rep["timing_s"] = round(time.time() - t0, 1)
    return rep


@app.local_entrypoint()
def diagnose_main(seq: str = SEQ):
    import json
    import os

    rep = diagnose.remote(seq)
    os.makedirs("logs/stage2/material", exist_ok=True)
    with open("logs/stage2/material/diagnose.json", "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


# =============================================================================
# Rendus de revue (gris = îlots < min_size, couleurs = superpoints de surface)
# =============================================================================

@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=8, memory=48 * 1024, timeout=3600)
def render_review(seq: str = SEQ, set_names: list[str] | None = None) -> dict:
    import os
    import time

    import numpy as np

    from splatpipe.stage1.maptools import pack_ijk, scene_root
    from splatpipe.stage2.superpoints import (
        MAP_PLY, OUT_SUB, SETS, VOXEL, palette, read_ply_points, _render_review_views,
    )

    root = scene_root(DATA_MOUNT, seq)
    out_dir = f"{root}/{OUT_SUB}"
    cols = read_ply_points(f"{root}/{MAP_PLY}", ["x", "y", "z"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    keys = pack_ijk(np.floor(xyz / VOXEL).astype(np.int64))
    order = np.argsort(keys, kind="stable")
    xyz_s = xyz[order].astype(np.float32)
    figures: dict[str, bytes] = {}
    info = {}
    for name in set_names or list(SETS):
        with np.load(f"{out_dir}/superpoints_{name}.npz") as z:
            lab = z["sp_id"]
        sizes = np.bincount(lab)
        keep = sizes >= SETS[name]["min_size"]
        recol = np.zeros((len(sizes), 3), np.uint8)
        recol[keep] = palette(int(keep.sum()), seed=1)
        recol[~keep] = (200, 200, 200)
        rgb = recol[lab]
        n_sp_surface = int(keep.sum())
        vox_in_surface = float(sizes[keep].sum() / sizes.sum())
        info[name] = {"n_sp_total": int(len(sizes)), "n_sp_surface": n_sp_surface,
                      "vox_in_surface_frac": vox_in_surface}
        print(f"[revue] {name} : {n_sp_surface:,} sp de surface (≥ {SETS[name]['min_size']} voxels) "
              f"couvrant {vox_in_surface*100:.1f} % des voxels", flush=True)
        # crops (repère map, lus sur la vue de dessus) : racks à vélos + bloc ouest arboré
        crops = {"zoom_racks": ((-38, -2), (-14, 14)), "zoom_ouest": ((-75, -45), (-62, -32))}
        figs = _render_review_views(xyz_s, rgb, f"{out_dir}/superpoints_{name}_review",
                                    title=f"2.5.1 [{name}] — {n_sp_surface:,} sp de surface (gris = petits/bruit)",
                                    crops=crops)
        for f in figs:
            figures[os.path.basename(f)] = open(f, "rb").read()
    vol.commit()
    return {"figures": figures, "info": info}


def _render_review_views(xyz, rgb, out_prefix, title, crops, max_pts=1_800_000):
    """Vue de dessus + deux profils + zooms ; gris = petits superpoints (déjà codé dans rgb)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(xyz) > max_pts:
        idx = np.random.default_rng(0).choice(len(xyz), max_pts, replace=False)
    else:
        idx = np.arange(len(xyz))
    p, c = xyz[idx], rgb[idx].astype(np.float32) / 255.0
    order = np.argsort(c.sum(1))  # les points sombres (traits) derrière, les couleurs devant
    p, c = p[order], c[order]
    files = []
    views = [("top", (0, 1), ("x map (m)", "y map (m)"), None),
             ("side_xz", (0, 2), ("x map (m)", "z map (m)"), None),
             ("side_yz", (1, 2), ("y map (m)", "z map (m)"), None)]
    for cname, (xr, yr) in crops.items():
        views.append((cname, (0, 1), ("x map (m)", "y map (m)"), (xr, yr)))
    for name, (a, b), (la, lb), lims in views:
        fig, ax = plt.subplots(figsize=(11, 10), facecolor="white")
        ax.scatter(p[:, a], p[:, b], c=c, s=0.08 if lims is None else 0.35, marker=".", rasterized=True,
                   linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel(la, fontsize=8)
        ax.set_ylabel(lb, fontsize=8)
        ax.tick_params(labelsize=7)
        if lims is not None:
            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])
        f = f"{out_prefix}_{name}.png"
        fig.savefig(f, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        files.append(f)
    return files


@app.local_entrypoint()
def render_review_main(seq: str = SEQ):
    import json
    import os

    out = render_review.remote(seq)
    os.makedirs("logs/stage2/material", exist_ok=True)
    for name, b in out["figures"].items():
        open(f"logs/stage2/material/{name}", "wb").write(b)
    print(json.dumps(out["info"], indent=1, ensure_ascii=False))


# =============================================================================
# Rendu « jugeable » : ombrage par normales + frontières de superpoints surlignées
# =============================================================================

@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=16, memory=64 * 1024, timeout=3600)
def render_shaded(seq: str = SEQ, set_name: str = "A_defaut") -> dict:
    """Grayscale ombré (Lambert, |n·L|) pour lire la géométrie ; voxels de frontière entre deux superpoints
    peints avec la couleur de leur superpoint (vitrail) — l'œil vérifie que les frontières suivent les bords
    d'objets. Produit aussi l'histogramme de tailles des superpoints de surface."""
    import os
    import time

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from splatpipe.stage1.maptools import pack_ijk, scene_root
    from splatpipe.stage2.superpoints import (
        MAP_PLY, OUT_SUB, SETS, VOXEL, edge_pass, palette, read_ply_points,
    )


    t0 = time.time()
    root = scene_root(DATA_MOUNT, seq)
    out_dir = f"{root}/{OUT_SUB}"
    cols = read_ply_points(f"{root}/{MAP_PLY}", ["x", "y", "z", "nx", "ny", "nz"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    nrm = np.stack([cols["nx"], cols["ny"], cols["nz"]], 1)
    keys = pack_ijk(np.floor(xyz / VOXEL).astype(np.int64))
    order = np.argsort(keys, kind="stable")
    xyz_s = xyz[order].astype(np.float32)
    nrm_s = nrm[order]
    del cols, xyz, nrm, keys, order

    with np.load(f"{out_dir}/superpoints_{set_name}.npz") as z:
        keys_sorted = z["keys"]
        lab = z["sp_id"]
    assert int(len(keys_sorted)) == len(xyz_s)
    sizes = np.bincount(lab)
    n_surface = int((sizes >= SETS[set_name]["min_size"]).sum())
    sz = sizes[sizes >= SETS[set_name]["min_size"]]

    # frontières : voxels ayant au moins un voisin de grille d'un autre superpoint
    eg = edge_pass(keys_sorted, nrm_s, np.zeros(len(xyz_s), np.float32))
    diff = lab[eg["ei"]] != lab[eg["ej"]]
    bnd = np.zeros(len(xyz_s), bool)
    bnd[eg["ei"][diff]] = True
    bnd[eg["ej"][diff]] = True

    # style vitrail : intérieur clair teinté par superpoint, frontière = même teinte assombrie (trait lisible)
    L = np.array([0.55, 0.45, 0.70], np.float32)
    L /= np.linalg.norm(L)
    shade = 0.55 + 0.45 * np.abs(nrm_s @ L)
    pal = palette(int(lab.max()) + 1, seed=1).astype(np.float32) / 255.0
    small = sizes < SETS[set_name]["min_size"]
    pal[small] = 0.62  # unités fines : gris neutre
    rgb = np.full((len(xyz_s), 3), 0.40, np.float32) * shade[:, None]  # intérieurs : gris ombré
    rgb[bnd] = pal[lab[bnd]] * 0.38  # frontières : teinte du superpoint, sombre (trait lisible)

    figures: dict[str, bytes] = {}

    def fig_to_bytes(fig, name):
        f = f"{out_dir}/superpoints_{set_name}_shaded_{name}.png"
        fig.savefig(f, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures[os.path.basename(f)] = open(f, "rb").read()

    idx = np.arange(len(xyz_s)) if len(xyz_s) <= 1_800_000 else np.random.default_rng(0).choice(
        len(xyz_s), 1_800_000, replace=False)
    interior, border = idx[~bnd[idx]], idx[bnd[idx]]  # frontières dessinées au-dessus, en plus gros
    views = [
        ("top", (0, 1), ("x map (m)", "y map (m)"), None, 0.10, 0.30),
        ("side_xz", (0, 2), ("x map (m)", "z map (m)"), None, 0.10, 0.30),
        ("side_yz", (1, 2), ("y map (m)", "z map (m)"), None, 0.10, 0.30),
        ("zoom_racks", (0, 1), ("x map (m)", "y map (m)"), ((-38, -2), (-14, 14)), 0.5, 1.9),
        ("zoom_ouest", (0, 1), ("x map (m)", "y map (m)"), ((-75, -45), (-62, -32)), 0.5, 1.9),
        ("zoom_kiosque", (0, 1), ("x map (m)", "y map (m)"), ((-14, 4), (-26, -8)), 0.5, 1.9),
        ("zoom_facade", (0, 2), ("x map (m)", "z map (m)"), ((-14, 4), (0, 14)), 0.5, 1.9),
    ]
    for name, (a, b), (la, lb), lims, si, sb in views:
        fig, ax = plt.subplots(figsize=(11, 10), facecolor="white")
        ax.scatter(xyz_s[interior, a], xyz_s[interior, b], c=np.clip(rgb[interior], 0, 1), s=si,
                   marker=".", rasterized=True, linewidths=0)
        ax.scatter(xyz_s[border, a], xyz_s[border, b], c=np.clip(rgb[border], 0, 1), s=sb,
                   marker=".", rasterized=True, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(f"2.5.1 [{set_name}] — ombrage normales, frontières de superpoints surlignées "
                     f"({n_surface:,} sp de surface)", fontsize=9, loc="left")
        ax.set_xlabel(la, fontsize=8)
        ax.set_ylabel(lb, fontsize=8)
        ax.tick_params(labelsize=7)
        if lims is not None:
            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])
        fig_to_bytes(fig, name)

    # histogramme de tailles (échelle log)
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="white")
    bins = np.logspace(np.log10(20), np.log10(sz.max()), 40)
    ax.hist(sz, bins=bins, color="#2a78d6", alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("taille d'un superpoint de surface (voxels 5 cm ; 400 voxels ≈ 1 m²)", fontsize=8)
    ax.set_ylabel("nombre de superpoints", fontsize=8)
    ax.set_title(f"tailles des {n_surface:,} superpoints de surface — médiane {np.median(sz):.0f} voxels "
                 f"({np.median(sz)*0.0025:.2f} m²), max {sz.max():,}", fontsize=9, loc="left")
    ax.tick_params(labelsize=7)
    fig_to_bytes(fig, "size_hist")

    stats = {"n_surface": n_surface, "size_p25": float(np.percentile(sz, 25)),
             "size_p50": float(np.median(sz)), "size_p75": float(np.percentile(sz, 75)),
             "size_p90": float(np.percentile(sz, 90)), "size_max": int(sz.max()),
             "area_m2_p50": float(np.median(sz) * 0.0025), "area_m2_p90": float(np.percentile(sz, 90) * 0.0025),
             "boundary_voxel_frac": float(bnd.mean())}
    stats["timing_s"] = round(time.time() - t0, 1)
    print(f"[ombré] {stats}", flush=True)
    vol.commit()
    return {"figures": figures, "stats": stats}


@app.local_entrypoint()
def render_shaded_main(set_name: str = "A_defaut"):
    import json
    import os

    out = render_shaded.remote(SEQ, set_name)
    os.makedirs("logs/stage2/material", exist_ok=True)
    for name, b in out["figures"].items():
        open(f"logs/stage2/material/{name}", "wb").write(b)
    print(json.dumps(out["stats"], indent=1, ensure_ascii=False))
