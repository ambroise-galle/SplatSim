"""Tâches 2.5.2–2.5.3 (jalon M2 du plan 2.5) — lifting « tout-horizon » des labels Mask2Former.

Remplace le vote ±30 ms de la 2.3 par une association de visibilité **tous instants confondus**, pour les
surfaces statiques. Pour chaque image des 3 caméras (5 608 poses × 3) :
  1. préfiltrage des voxels par portée (≤ 55 m du centre optique) ;
  2. projection fisheye (`FisheyeCamera.project`, marge 1 px) ;
  3. **z-buffer basse résolution** du nuage lui-même (cellules 4×4 px, profondeur = distance le long du
     rayon, min par cellule) ; un voxel est visible si sa profondeur est à ±10 cm du premier plan de sa
     cellule — l'auto-occlusion du nuage ;
  4. rejets : cos d'incidence < 0,25 ; pixels masqués ciel ou personne (`masks/` bits 1 et 2).
Chaque voxel visible vote pour son **superpoint** : histogramme de classes ADE20K (150 dims) pondéré par le
cos d'incidence, histogramme regroupé **par caméra** (14 groupes, pour l'accord inter-vues), histogramme
d'incidence (6 pas de 15°), n_vues et masque de caméras. Le commutateur de la 2.3 reste disponible :
`mode="window"` restreint les images à ±`t_window_s` du milieu des tours (expériences dynamiques) ; le mode
par défaut est `"all"` (tout-horizon).

Agrégation (`aggregate`) : classe dominante par superpoint (150 et groupe), part du dominant, couverture
voxel (héritage superpoint → voxel), fraction multi-vues (≥ 3 vues), accord inter-caméras sur les groupes
(paires de caméras comparables, poids ≥ 0,5 chacune), fuite ciel ; PLY carte colorée par classe dominante +
rendus. Comparaison à la 2.3 : 41 % → cible ≥ 70 % de voxels avec ≥ 1 classe.

Lancement :
  .venv/bin/modal run splatpipe/stage2/lift_semantics.py::m2_pilot_main   # ~30 images, contrôle
  .venv/bin/modal run splatpipe/stage2/lift_semantics.py::m2_main         # 8 conteneurs, complet
  .venv/bin/modal run splatpipe/stage2/lift_semantics.py::aggregate_main  # si besoin de rejouer l'agrégat
"""

from __future__ import annotations

import numpy as np

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume
from splatpipe.stage1.images import map_image

app = modal.App("splatsim-stage2-lift-m2")
vol = data_volume()

SEQ = "2024-03-13-observatory-quarter-01"
SP_NPZ = "reports/stage2_material/superpoints.npz"  # clés triées → sp_id (M1, jeu retenu)
MAP_PLY = "map/map_5cm_static.ply"
OUT_BASE = "reports/stage2_material"
MAX_RANGE_M = 55.0
COS_MIN = 0.25
ZCELL = 4
DEPTH_TOL = 0.10
MARGIN_PX = 1
BIT_SKY, BIT_PERSON = 1, 2
INC_EDGES = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])

# regroupement des 150 classes ADE20K (noms de `semantics/classes.json`, en minuscules débarrassées des espaces)
GROUPS = [
    ("sky", ["sky"]),
    ("person", ["person", "apparel"]),
    ("bike", ["bicycle", "minibike"]),
    ("vehicle", ["car", "bus", "truck", "van", "boat", "airplane", "ship"]),
    ("foliage", ["tree", "palm", "plant", "flower"]),
    ("ground-naturel", ["grass", "field", "earth", "sand", "dirt track", "land", "hill"]),
    ("ground-pave", ["road", "sidewalk", "path", "runway", "floor"]),
    ("batiment", ["wall", "building", "house", "skyscraper", "tower", "bridge", "column", "stairs", "step", "stairway"]),
    ("vitrage", ["windowpane", "glass", "screen door", "mirror"]),
    ("grille-barriere", ["fence", "railing", "bannister", "awning", "canopy"]),
    ("poteaux", ["pole", "streetlight", "traffic light"]),
    ("roche-eau", ["rock", "mountain", "water", "river", "lake", "sea", "waterfall", "fountain", "swimming pool"]),
    ("mobilier-urbain", ["bench", "ashcan", "signboard", "booth", "barrel", "box", "sculpture"]),
    ("autre", []),
]
# palette de rendu, même ordre que GROUPS + « sans classe »
PALETTE = np.array([
    (60, 160, 255), (220, 30, 30), (255, 150, 0), (255, 210, 60), (30, 140, 40), (150, 210, 70),
    (90, 90, 90), (135, 150, 170), (120, 200, 250), (150, 90, 50), (130, 60, 160), (70, 130, 180),
    (210, 170, 60), (170, 170, 170), (235, 235, 235)], np.uint8)


def _group_map(id2label: dict[str, str]) -> np.ndarray:
    name_to_g = {n: i for i, (g, names) in enumerate(GROUPS) for n in names}
    out = np.full(150, len(GROUPS) - 1, np.int64)  # défaut : « autre »
    for k, v in id2label.items():
        out[int(k)] = name_to_g.get(v.strip().lower(), len(GROUPS) - 1)
    return out


def lift_image() -> modal.Image:
    """Image CPU légère : numpy/scipy/matplotlib/pyyaml + pillow (lecture des PNG sémantiques)."""
    return (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            "numpy==2.3.3", "scipy==1.16.2", "matplotlib==3.10.6", "pyyaml==6.0.2", "pillow==11.3.0"
        )
        .env({"MPLBACKEND": "Agg"})
        .add_local_python_source("splatpipe")
    )


# =============================================================================
# Worker : accumulation sur une tranche d'images
# =============================================================================

@app.function(image=lift_image(), volumes={DATA_MOUNT: vol}, cpu=8, memory=32 * 1024, timeout=4 * 3600)
def worker(part: int, n_parts: int, tag: str = "m2", limit_images: int = 0, mode: str = "all",
           t_window_s: float = 0.03, seq: str = SEQ) -> dict:
    import json
    import os
    import time

    from PIL import Image

    from splatpipe.common.camera_model import FisheyeCamera
    from splatpipe.common.scene import Scene
    from splatpipe.stage1.maptools import pack_ijk, scene_root
    from splatpipe.stage2.lift_semantics import (
        BIT_PERSON, BIT_SKY, COS_MIN, DEPTH_TOL, GROUPS, INC_EDGES, MARGIN_PX, MAP_PLY, MAX_RANGE_M,
        OUT_BASE, SP_NPZ, ZCELL, _group_map,
    )
    from splatpipe.stage2.superpoints import read_ply_points

    t0 = time.time()
    root = scene_root(DATA_MOUNT, seq)
    out_dir = f"{root}/{OUT_BASE}/{tag}"
    os.makedirs(out_dir, exist_ok=True)

    cols = read_ply_points(f"{root}/{MAP_PLY}", ["x", "y", "z", "nx", "ny", "nz"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    nrm = np.stack([cols["nx"], cols["ny"], cols["nz"]], 1).astype(np.float64)
    keys = pack_ijk(np.floor(xyz / 0.05).astype(np.int64))
    order = np.argsort(keys, kind="stable")
    xyz_s = xyz[order]
    nrm_s = nrm[order]
    del cols, xyz, nrm, keys, order
    with np.load(f"{root}/{SP_NPZ}") as z:
        sp_id = z["sp_id"]
    n_sp = int(sp_id.max()) + 1

    with open(f"{root}/semantics/classes.json") as f:
        gmap = _group_map(json.load(f)["id2label"])
    n_groups = len(GROUPS)

    with open(f"{root}/poses/cameras.json") as f:
        poses = json.load(f)["cameras"]
    sc = Scene("oxford_spires", seq, data_root=DATA_MOUNT)
    cam_models = {c: FisheyeCamera.from_dict(sc.camera(c)) for c in poses}

    scan_t = None
    if mode == "window":  # commutateur 2.3 : images proches d'un tour (expériences dynamiques)
        d = f"{root}/lidar/hesai_deskewed"
        scan_t = np.array([int(s[:-4]) / 1e9 + 0.05 for s in sorted(os.listdir(d)) if s.endswith(".npz")])

    hist = np.zeros((n_sp, 150), np.float32)
    hist_cam = np.zeros((n_sp, len(poses), n_groups), np.float32)
    inc_hist = np.zeros((n_sp, len(INC_EDGES) - 1), np.uint32)
    n_vues = np.zeros(n_sp, np.uint32)
    cam_bits = np.zeros(n_sp, np.uint8)
    n_imgs = n_vis = n_pix_rejected = 0
    per_cam_imgs = {}

    # reprise après préemption : la liste des images est déterministe ; un checkpoint (accumulateurs +
    # nb d'images déjà faites, dans le même ordre) est sauvé toutes les 200 images et consommé à la fin.
    ckpt_path = f"{out_dir}/part_{part}_ckpt.npz"
    n_done_start = 0
    if os.path.exists(ckpt_path):
        with np.load(ckpt_path) as z:
            hist, hist_cam, inc_hist = z["hist"], z["hist_cam"], z["inc_hist"]
            n_vues, cam_bits = z["n_vues"], z["cam_bits"]
            n_imgs, n_vis, n_pix_rejected = int(z["n_imgs"]), int(z["n_vis"]), int(z["n_pix_rejected"])
            n_done_start = int(z["n_done"])
        print(f"[reprise] part {part} : {n_done_start} images déjà faites", flush=True)
    n_done = n_done_start

    def save_ckpt():
        np.savez_compressed(ckpt_path, hist=hist, hist_cam=hist_cam, inc_hist=inc_hist, n_vues=n_vues,
                            cam_bits=cam_bits, n_imgs=n_imgs, n_vis=n_vis, n_pix_rejected=n_pix_rejected,
                            n_done=n_done)
        vol.commit()  # sinon la préemption perd le checkpoint

    for k, (c, cam_poses) in enumerate(sorted(poses.items())):
        stamps = sorted(int(s) for s in cam_poses)
        stamps = stamps[part::n_parts]
        if limit_images:
            stamps = stamps[:limit_images]
        per_cam_imgs[c] = len(stamps)
        model = cam_models[c]
        cw = (model.width + ZCELL - 1) // ZCELL
        ch = (model.height + ZCELL - 1) // ZCELL
        for st in stamps:
            if n_done < n_done_start:  # images déjà comptabilisées dans le checkpoint (ordre déterministe)
                n_done += 1
                continue
            pose = cam_poses[str(st)]
            if scan_t is not None:
                t_img = float(pose["t_start"])
                if np.abs(scan_t - t_img).min() > t_window_s:
                    n_done += 1
                    continue
            spath = f"{root}/semantics/{c}/{st}.png"
            mpath = f"{root}/masks/{c}/{st}.png"
            if not (os.path.exists(spath) and os.path.exists(mpath)):
                n_done += 1
                continue
            sem = np.asarray(Image.open(spath))
            msk = np.asarray(Image.open(mpath))
            T_mc = np.asarray(pose["T_map_cam_start"], np.float64)
            R, cc = T_mc[:3, :3].T, T_mc[:3, 3]  # R = rotation carte→caméra (T_map_cam est caméra→carte)

            d3 = xyz_s - cc
            dist2 = np.einsum("ij,ij->i", d3, d3)
            cand = np.flatnonzero(dist2 <= MAX_RANGE_M**2)
            if len(cand) == 0:
                continue
            pc = d3[cand] @ R.T  # repère caméra
            uv, valid, _ = model.project(pc, margin=MARGIN_PX)
            del d3
            j = np.flatnonzero(valid)
            if len(j) == 0:
                continue
            cand, pc, uv = cand[j], pc[j], uv[j]
            depth = np.sqrt(np.einsum("ij,ij->i", pc, pc))
            view = pc / depth[:, None]
            view_w = view @ R
            cos_i = np.einsum("ij,ij->i", nrm_s[cand], -view_w)
            keep = cos_i >= COS_MIN
            if not keep.any():
                continue
            cand, depth, cos_i, view = cand[keep], depth[keep], cos_i[keep], view[keep]
            ui = uv[keep, 0].round().astype(np.int64)
            vi = uv[keep, 1].round().astype(np.int64)
            cls = sem[vi, ui].astype(np.int64)
            mval = msk[vi, ui]
            bad = ((mval & (BIT_SKY | BIT_PERSON)) != 0) | (cls >= 150)
            n_pix_rejected += int(bad.sum())
            ok = ~bad
            if not ok.any():
                continue
            cand, depth, cos_i, ui, vi, cls = (a[ok] for a in (cand, depth, cos_i, ui, vi, cls))

            # z-buffer 4×4 : premier plan par cellule, visibilité à ±10 cm
            cell = (vi // ZCELL) * cw + (ui // ZCELL)
            dmin = np.full(ch * cw, np.inf)
            np.minimum.at(dmin, cell, depth)
            vis = depth <= dmin[cell] + DEPTH_TOL
            if not vis.any():
                continue
            spv = sp_id[cand[vis]]
            w = cos_i[vis].astype(np.float32)
            clsv = cls[vis]
            np.add.at(hist, (spv, clsv), w)
            np.add.at(hist_cam, (spv, k, gmap[clsv]), w)
            ang = np.degrees(np.arccos(np.clip(cos_i[vis], -1.0, 1.0)))
            b = np.clip(np.searchsorted(INC_EDGES, ang, side="right") - 1, 0, len(INC_EDGES) - 2)
            np.add.at(inc_hist, (spv, b), 1)
            cnt = np.bincount(spv, minlength=n_sp)
            n_vues += cnt.astype(np.uint32)
            cam_bits[np.flatnonzero(cnt)] |= 1 << k
            n_imgs += 1
            n_vis += int(vis.sum())
            n_done += 1
            if n_done % 200 == 0:
                save_ckpt()

    if n_done % 200 != 0:
        save_ckpt()
    np.savez_compressed(f"{out_dir}/part_{part}.npz", hist=hist, hist_cam=hist_cam, inc_hist=inc_hist,
                        n_vues=n_vues, cam_bits=cam_bits)
    os.remove(ckpt_path)  # part écrite : checkpoint consommé
    vol.commit()
    return {"part": part, "n_images": n_imgs, "n_vis_vox": n_vis, "n_pix_rejected": n_pix_rejected,
            "per_cam_images": per_cam_imgs, "s": round(time.time() - t0, 1)}


# =============================================================================
# Agrégation : sommation des parts, classe dominante, couverture, accords, sorties
# =============================================================================

@app.function(image=lift_image(), volumes={DATA_MOUNT: vol}, cpu=16, memory=96 * 1024, timeout=3 * 3600)
def aggregate(n_parts: int, tag: str = "m2", seq: str = SEQ, make_figures: bool = True) -> dict:
    import json
    import os
    import time

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from splatpipe.stage1.maptools import pack_ijk, scene_root, write_ply
    from splatpipe.stage2.lift_semantics import GROUPS, OUT_BASE, PALETTE
    from splatpipe.stage2.superpoints import read_ply_points, _render_review_views

    t0 = time.time()
    root = scene_root(DATA_MOUNT, seq)
    out_dir = f"{root}/{OUT_BASE}/{tag}"
    os.makedirs(out_dir, exist_ok=True)

    with np.load(f"{root}/{SP_NPZ}") as z:
        sp_id = z["sp_id"]
        sp_keys = z["keys"]
    n_sp = int(sp_id.max()) + 1

    hist = np.zeros((n_sp, 150), np.float32)
    hist_cam = None
    inc_hist = np.zeros((n_sp, len(INC_EDGES) - 1), np.uint32)
    n_vues = np.zeros(n_sp, np.uint32)
    cam_bits = np.zeros(n_sp, np.uint8)
    for p in range(n_parts):
        f = f"{out_dir}/part_{p}.npz"
        if not os.path.exists(f):
            raise FileNotFoundError(f)
        with np.load(f) as z:
            hist += z["hist"]
            inc_hist += z["inc_hist"]
            n_vues += z["n_vues"]
            cam_bits |= z["cam_bits"]
            if hist_cam is None:
                hist_cam = z["hist_cam"].astype(np.float32)
            else:
                hist_cam += z["hist_cam"]
        os.remove(f)  # parts consommées (l'agrégat est reproductible en relançant les workers)

    wsum = hist.sum(1)
    seen = wsum > 0
    dom = np.full(n_sp, -1, np.int32)
    dom[seen] = hist[seen].argmax(1)
    dom_frac = np.zeros(n_sp, np.float32)
    dom_frac[seen] = hist[np.arange(n_sp)[seen], dom[seen]] / wsum[seen]
    gmap = _group_map(json.load(open(f"{root}/semantics/classes.json"))["id2label"])
    hist_g = np.zeros((n_sp, len(GROUPS)), np.float32)
    for g in range(len(GROUPS)):
        hist_g[:, g] = hist[:, gmap == g].sum(1)
    dom_g = np.full(n_sp, -1, np.int32)
    dom_g[seen] = hist_g[seen].argmax(1)

    # accord inter-caméras : paires de caméras comparables (poids groupe ≥ 0,5 de part et d'autre)
    wcam = hist_cam.sum(2)
    pairs = [(0, 1), (0, 2), (1, 2)]
    agree = comp = 0
    argm = np.full((n_sp, len(pairs)), -1, np.int32)
    for a, (i, j) in enumerate(pairs):
        ok = (wcam[:, i] >= 0.5) & (wcam[:, j] >= 0.5)
        ai = hist_cam[:, i, :].argmax(1)
        aj = hist_cam[:, j, :].argmax(1)
        comp += int(ok.sum())
        agree += int((ai[ok] == aj[ok]).sum())

    vox_class = np.where(seen[sp_id], dom_g[sp_id], -1)
    cov = float((vox_class >= 0).mean())
    rep = {
        "stage": "2.5.2/2.5.3", "tag": tag,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_superpoints": n_sp, "n_voxels": int(len(sp_id)),
        "coverage_voxel": cov, "coverage_superpoint": float(seen.mean()),
        "multiview_frac_voxel": None,
        "sky_leak_voxel": float((vox_class == 0).mean()),
        "agree_pairs_total": comp, "agree_frac": (agree / comp) if comp else None,
        "dominant_groups": {},
    }
    mv = n_vues[sp_id] >= 3  # fraction multi-vues en voxels : n_vues du superpoint ≥ 3
    rep["multiview_frac_voxel"] = float(mv.mean())
    for g, (name, _) in enumerate(GROUPS):
        n = int((dom_g == g).sum())
        if n:
            rep["dominant_groups"][name] = {"n_superpoints": n,
                                            "wfrac": float(hist_g[seen, g].sum() / hist_g[seen].sum())}
    rep["timing_s"] = round(time.time() - t0, 1)

    np.savez_compressed(f"{out_dir}/superpoint_semantics.npz", hist=hist, hist_g=hist_g,
                        hist_cam=hist_cam.astype(np.float32), inc_hist=inc_hist, n_vues=n_vues,
                        cam_bits=cam_bits, wsum=wsum, dominant_class=dom, dominant_group=dom_g,
                        dominant_frac=dom_frac)

    # carte PLY colorée par classe dominante + rendus (héritage superpoint → voxel)
    figures: dict[str, bytes] = {}
    if make_figures:
        cols = read_ply_points(f"{root}/map/map_5cm_static.ply", ["x", "y", "z"])
        xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
        keys = pack_ijk(np.floor(xyz / 0.05).astype(np.int64))
        o = np.argsort(keys, kind="stable")
        xyz_s = xyz[o].astype(np.float32)
        assert np.array_equal(keys[o], sp_keys)
        pal = PALETTE.astype(np.float32)
        rgb = pal[np.where(vox_class >= 0, vox_class, len(GROUPS))]
        comments = [f"SplatSim 2.5.2/3 carte coloree par classe Mask2Former dominante par superpoint ({tag})",
                    "groupes : " + ", ".join(f"{i}={n}" for i, (n, _) in enumerate(GROUPS)) +
                    f" ; {len(GROUPS)}=sans classe"]
        write_ply(f"{out_dir}/map_colored.ply",
                  {"x": xyz_s[:, 0], "y": xyz_s[:, 1], "z": xyz_s[:, 2],
                   "red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2],
                   "group": np.where(vox_class >= 0, vox_class, len(GROUPS)).astype(np.uint8)},
                  comments=comments)
        crops = {"zoom_racks": ((-38, -2), (-14, 14)), "zoom_ouest": ((-75, -45), (-62, -32))}
        figs = _render_review_views(
            xyz_s, rgb, f"{out_dir}/map_{tag}", title=f"M2 [{tag}] — classe Mask2Former dominante "
            f"(couverture {cov*100:.1f} %, accord inter-caméras {100*rep['agree_frac']:.1f} %)", crops=crops)
        for f in figs:
            figures[os.path.basename(f)] = open(f, "rb").read()
        # figure de couverture avant/après
        fig, ax = plt.subplots(figsize=(5.5, 3.4), facecolor="white")
        names = ["2.3 (vote ±30 ms)", f"M2 [{tag}]"]
        vals = [41.0, cov * 100]
        ax.bar(names, vals, color=["#bbbbbb", "#2a78d6"], width=0.55)
        for i, v in enumerate(vals):
            ax.text(i, v + 1.5, f"{v:.1f} %", ha="center", fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_ylabel("voxels avec ≥ 1 classe (%)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_title("couverture sémantique voxel", fontsize=9, loc="left")
        ff = f"{out_dir}/coverage.png"
        fig.savefig(ff, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures["coverage.png"] = open(ff, "rb").read()

    with open(f"{out_dir}/report.json", "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    vol.commit()
    return {"report": json.loads(json.dumps(rep)), "figures": figures}


# =============================================================================
# Orchestration distante (survit aux déconnexions du client local, cf. --detach)
# =============================================================================

@app.function(image=lift_image(), volumes={DATA_MOUNT: vol}, cpu=2, memory=8 * 1024, timeout=6 * 3600)
def orchestrate(n_parts: int, tag: str = "m2", seq: str = SEQ) -> dict:
    """Lance les workers en parallèle puis l'agrégation, entièrement côté serveur.

    (`starmap` depuis un conteneur bute sur un bug Modal « aclose(): asynchronous generator » — on passe par
    des `.remote` parallèles dans un pool de threads.)"""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n_parts) as ex:
        outs = list(ex.map(lambda p: worker.remote(p, n_parts, tag=tag, seq=seq), range(n_parts)))
    agg = aggregate.remote(n_parts, tag=tag, seq=seq)
    return {"workers": outs, "report": agg["report"]}


# =============================================================================
# Entrées locales
# =============================================================================

@app.local_entrypoint()
def m2_pilot_main(n_images: int = 30):
    import json

    out = worker.remote(0, 1, tag="m2_pilot", limit_images=n_images)
    agg = aggregate.remote(1, tag="m2_pilot")
    print(json.dumps({"worker": out, "report": agg["report"]}, indent=1, ensure_ascii=False))


@app.local_entrypoint()
def m2_main(n_parts: int = 8):
    """Lancement détaché recommandé : `modal run --detach splatpipe/stage2/lift_semantics.py::m2_main`
    puis récupération locale des livrables par `fetch_main` (tout est écrit sur le Volume)."""
    import json

    out = orchestrate.remote(n_parts)
    print(json.dumps(out["report"], indent=1, ensure_ascii=False))


@app.local_entrypoint()
def aggregate_main(tag: str = "m2", n_parts: int = 8):
    import json
    import os

    agg = aggregate.remote(n_parts, tag=tag)
    os.makedirs("logs/stage2/material", exist_ok=True)
    for name, b in agg["figures"].items():
        open(f"logs/stage2/material/{name}", "wb").write(b)
    with open(f"logs/stage2/material/{tag}_report.json", "w") as f:
        json.dump(agg["report"], f, indent=1, ensure_ascii=False)
    print(json.dumps(agg["report"], indent=1, ensure_ascii=False))


@app.local_entrypoint()
def fetch_main(tag: str = "m2"):
    """Récupère localement rapport + figures d'un tag déjà agrégé (run détaché)."""
    import json
    import os

    @app.function(image=lift_image(), volumes={DATA_MOUNT: vol}, timeout=600)
    def _fetch(tag: str) -> dict:
        import os

        out_dir = f"{DATA_MOUNT}/scenes/oxford_spires/{SEQ}/{OUT_BASE}/{tag}"
        files = {}
        rp = f"{out_dir}/report.json"
        if os.path.exists(rp):
            files["report.json"] = open(rp, "rb").read()
        for f in sorted(os.listdir(out_dir)):
            if f.endswith(".png"):
                files[f] = open(f"{out_dir}/{f}", "rb").read()
        return files

    os.makedirs("logs/stage2/material", exist_ok=True)
    files = _fetch.remote(tag)
    for name, b in files.items():
        open(f"logs/stage2/material/{name if name != 'report.json' else f'{tag}_report.json'}", "wb").write(b)
    print(json.dumps(json.loads(files["report.json"]), indent=1, ensure_ascii=False))


@app.local_entrypoint()
def worker_one_main(part: int, n_parts: int = 8, tag: str = "m2", mode: str = "all"):
    """Met en file **côté serveur** (`spawn`, pas `remote`) un worker qui écrit `part_<part>.npz` sur le
    Volume puis sort — aucun lien de durée de vie avec le client local (réseau du Mac instable ce soir)."""
    h = worker.spawn(part, n_parts, tag=tag, mode=mode)
    print(f"spawned part {part}: {h.object_id}")


@app.function(image=lift_image(), volumes={DATA_MOUNT: vol}, cpu=8, memory=64 * 1024, timeout=3600)
def diag_agreement(tag: str = "m2", seq: str = SEQ) -> dict:
    """Diagnostic de l'accord inter-caméras : effet du seuil de poids, restriction aux sp de surface,
    et matrice de confusion des désaccords (paires de groupes)."""
    import json

    import numpy as np

    from splatpipe.stage1.maptools import scene_root
    from splatpipe.stage2.lift_semantics import GROUPS, OUT_BASE

    root = scene_root(DATA_MOUNT, seq)
    with np.load(f"{root}/{OUT_BASE}/{tag}/superpoint_semantics.npz") as z:
        hist_g, hist_cam, wsum = z["hist_g"], z["hist_cam"].astype(np.float32), z["wsum"]
    with np.load(f"{root}/{OUT_BASE}/superpoints.npz") as z:
        sizes = np.bincount(z["sp_id"])
    sizes = sizes[: hist_g.shape[0]]
    dom_g = np.full(hist_g.shape[0], -1, np.int32)
    seen = wsum > 0
    dom_g[seen] = hist_g[seen].argmax(1)
    surface = (sizes >= 20) & seen
    wcam = hist_cam.sum(2)
    ng = hist_g.shape[1]
    argm = np.stack([hist_cam[:, i, :].argmax(1) for i in range(3)], 1)  # (n_sp, 3)

    out = {"thresholds": {}, "confusion_top": []}
    for th in (0.5, 2.0, 5.0, 20.0, 50.0):
        tot = ok = 0
        for i, j in ((0, 1), (0, 2), (1, 2)):
            m = (wcam[:, i] >= th) & (wcam[:, j] >= th)
            tot += int(m.sum())
            ok += int((argm[m, i] == argm[m, j]).sum())
        out["thresholds"][str(th)] = {"pairs": tot, "agree": round(ok / tot, 4) if tot else None}

    tot = ok = 0
    conf = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        m = (wcam[:, i] >= 5.0) & (wcam[:, j] >= 5.0) & surface
        tot += int(m.sum())
        a, b = argm[m, i], argm[m, j]
        ok += int((a == b).sum())
        for x, y in zip(a.tolist(), b.tolist()):
            if x != y:
                key = tuple(sorted((x, y)))
                conf[key] = conf.get(key, 0) + 1
    out["surface_strong_agree"] = {"pairs": tot, "agree": round(ok / tot, 4) if tot else None}
    for (x, y), n in sorted(conf.items(), key=lambda kv: -kv[1])[:12]:
        out["confusion_top"].append({"a": GROUPS[x][0], "b": GROUPS[y][0], "n": n})
    # accord « confiant » : chaque caméra dominée à ≥60 % par son groupe (exclut les sp de frontière mixtes)
    for th, margin in ((5.0, 0.6), (20.0, 0.6)):
        tot = ok = 0
        conf_pairs = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            wi, wj = wcam[:, i], wcam[:, j]
            pi = hist_cam[:, i, :].max(1) / np.maximum(wi, 1e-9)
            pj = hist_cam[:, j, :].max(1) / np.maximum(wj, 1e-9)
            m = (wi >= th) & (wj >= th) & (pi >= margin) & (pj >= margin)
            tot += int(m.sum())
            ok += int((argm[m, i] == argm[m, j]).sum())
        out["thresholds"][f"confiant th={th} marge={margin}"] = {"pairs": tot,
                                                                 "agree": round(ok / tot, 4) if tot else None}

    # cohérence avec le consensus fusionné : argmax par caméra == groupe dominant fusionné
    for th in (5.0, 20.0):
        tot = ok = 0
        for i in range(3):
            m = wcam[:, i] >= th
            tot += int(m.sum())
            ok += int((argm[m, i] == dom_g[m]).sum())
        out["thresholds"][f"vs consensus th={th}"] = {"views": tot,
                                                      "agree": round(ok / tot, 4) if tot else None}

    # accord par groupe (paires fortes) : qui s'accorde, qui se dispute
    per_g = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        m = (wcam[:, i] >= 5.0) & (wcam[:, j] >= 5.0) & surface
        a, b = argm[m, i], argm[m, j]
        eq = a == b
        for g in np.unique(np.r_[a, b]):
            k = (a == g) | (b == g)
            d = per_g.setdefault(GROUPS[int(g)][0], {"pairs": 0, "agree": 0})
            d["pairs"] += int(k.sum())
            d["agree"] += int((k & eq).sum())
    out["per_group_agree"] = {g: {"pairs": d["pairs"], "agree": round(d["agree"] / d["pairs"], 3)}
                              for g, d in sorted(per_g.items(), key=lambda kv: -kv[1]["pairs"])}

    # contrôle : couverture dans la boîte du toit du kiosque (zone blanche du rendu)
    from splatpipe.stage1.maptools import pack_ijk, scene_root as sr
    from splatpipe.stage2.superpoints import read_ply_points as _rp
    root2 = sr(DATA_MOUNT, seq)
    cols = _rp(f"{root2}/map/map_5cm_static.ply", ["x", "y", "z"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    keys2 = pack_ijk(np.floor(xyz / 0.05).astype(np.int64))
    o2 = np.argsort(keys2, kind="stable")
    in_box = ((xyz[o2, 0] > -25) & (xyz[o2, 0] < -3) & (xyz[o2, 1] > -12) & (xyz[o2, 1] < -6)
              & (xyz[o2, 2] > 2) & (xyz[o2, 2] < 9))
    with np.load(f"{root}/{OUT_BASE}/superpoints.npz") as z:
        sp_box = z["sp_id"][o2]
    seen_box = wsum[sp_box] > 0
    out["roof_box"] = {"n_voxels": int(in_box.sum()), "coverage": float(seen_box[in_box].mean()),
                       "groups": {GROUPS[int(g)][0]: int(n) for g, n in
                                  zip(*np.unique(dom_g[sp_box[in_box][seen_box[in_box]]], return_counts=True))}}

    # répartition des classes dominantes par caméra (sur sp comparables forts)
    out["per_cam_dominant"] = {}
    for i in range(3):
        m = wcam[:, i] >= 5.0
        u, c = np.unique(argm[m, i], return_counts=True)
        out["per_cam_dominant"][f"cam{i}"] = {GROUPS[int(x)][0]: int(n) for x, n in zip(u, c)}
    return out


@app.local_entrypoint()
def diag_m2_main(tag: str = "m2"):
    import json

    print(json.dumps(diag_agreement.remote(tag), indent=1, ensure_ascii=False))


@app.function(image=lift_image(), volumes={DATA_MOUNT: vol}, cpu=8, memory=48 * 1024, timeout=3600)
def render_colored(tag: str = "m2", seq: str = SEQ) -> dict:
    """Rendu lisible de la carte par classe dominante : palette assombrie, points plus denses."""
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from splatpipe.stage1.maptools import pack_ijk, scene_root
    from splatpipe.stage2.lift_semantics import GROUPS, OUT_BASE, PALETTE
    from splatpipe.stage2.superpoints import read_ply_points

    root = scene_root(DATA_MOUNT, seq)
    out_dir = f"{root}/{OUT_BASE}/{tag}"
    with np.load(f"{out_dir}/superpoint_semantics.npz") as z:
        dom_g, wsum = z["dominant_group"], z["wsum"]
    with np.load(f"{root}/{OUT_BASE}/superpoints.npz") as z:
        sp_id = z["sp_id"]
    cols = read_ply_points(f"{root}/map/map_5cm_static.ply", ["x", "y", "z"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    keys = pack_ijk(np.floor(xyz / 0.05).astype(np.int64))
    o = np.argsort(keys, kind="stable")
    xyz_s = xyz[o].astype(np.float32)
    vox = np.where(wsum[sp_id] > 0, dom_g[sp_id], -1)
    cov = float((vox >= 0).mean())
    pal = (PALETTE.astype(np.float32) * 0.72).astype(np.uint8)  # assombri pour la lisibilité
    rgb = pal[np.where(vox >= 0, vox, len(GROUPS))]
    if len(xyz_s) > 2_600_000:
        idx = np.random.default_rng(0).choice(len(xyz_s), 2_600_000, replace=False)
    else:
        idx = np.arange(len(xyz_s))
    p, c = xyz_s[idx], rgb[idx].astype(np.float32) / 255.0
    # les « sans classe » d'abord, sinon ils recouvrent les couleurs des classes
    none_m = (vox[idx] < 0)
    order = np.lexsort((c.sum(1), none_m))
    p, c = p[order], c[order]
    figures: dict[str, bytes] = {}
    views = [
        ("top", (0, 1), ("x map (m)", "y map (m)"), None, 0.14),
        ("side_xz", (0, 2), ("x map (m)", "z map (m)"), None, 0.14),
        ("zoom_racks", (0, 1), ("x map (m)", "y map (m)"), ((-38, -2), (-14, 14)), 0.8),
        ("zoom_ouest", (0, 1), ("x map (m)", "y map (m)"), ((-75, -45), (-62, -32)), 0.8),
        ("zoom_kiosque", (0, 1), ("x map (m)", "y map (m)"), ((-14, 4), (-26, -8)), 0.8),
    ]
    for name, (a, b), (la, lb), lims, s_ in views:
        fig, ax = plt.subplots(figsize=(11, 10), facecolor="white")
        ax.scatter(p[:, a], p[:, b], c=c, s=s_, marker=".", rasterized=True, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(f"M2 [{tag}] — classe dominante (assombrie) — couverture {cov*100:.1f} %", fontsize=9,
                     loc="left")
        ax.set_xlabel(la, fontsize=8)
        ax.set_ylabel(lb, fontsize=8)
        ax.tick_params(labelsize=7)
        if lims is not None:
            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])
        f = f"{out_dir}/mapcol_{tag}_{name}.png"
        fig.savefig(f, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures[os.path.basename(f)] = open(f, "rb").read()
    vol.commit()
    return {"figures": figures, "coverage": cov}


@app.local_entrypoint()
def render_colored_main(tag: str = "m2"):
    import json
    import os

    out = render_colored.remote(tag)
    os.makedirs("logs/stage2/material", exist_ok=True)
    for name, b in out["figures"].items():
        open(f"logs/stage2/material/{name}", "wb").write(b)
    print(json.dumps({"coverage": out["coverage"]}))


# =============================================================================
# Accord intra-triplet (test 2.3-comparable : mêmes instants, 3 caméras)
# =============================================================================

@app.function(image=lift_image(), volumes={DATA_MOUNT: vol}, cpu=8, memory=48 * 1024, timeout=4 * 3600)
def triplet_agree(part: int, n_parts: int, seq: str = SEQ) -> dict:
    """Pour chaque triplet synchronisé : projeter les voxels dans les 3 images (même z-buffer/rejets que le
    lifting), grouper par superpoint, et compter, par paire de caméras, les comparables (poids ≥ 0,5)
    d'accord et de désaccord sur le groupe dominant. Scalaire en sortie — le test direct du 86,9 % de la 2.3."""
    import json
    import os
    import time

    from PIL import Image

    from splatpipe.common.camera_model import FisheyeCamera
    from splatpipe.common.scene import Scene
    from splatpipe.stage1.maptools import pack_ijk, scene_root
    from splatpipe.stage2.lift_semantics import (
        BIT_PERSON, BIT_SKY, COS_MIN, DEPTH_TOL, GROUPS, MARGIN_PX, MAP_PLY, MAX_RANGE_M, OUT_BASE,
        SP_NPZ, ZCELL, _group_map,
    )
    from splatpipe.stage2.superpoints import read_ply_points

    t0 = time.time()
    root = scene_root(DATA_MOUNT, seq)
    cols = read_ply_points(f"{root}/{MAP_PLY}", ["x", "y", "z", "nx", "ny", "nz"])
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    nrm = np.stack([cols["nx"], cols["ny"], cols["nz"]], 1).astype(np.float64)
    keys = pack_ijk(np.floor(xyz / 0.05).astype(np.int64))
    o = np.argsort(keys, kind="stable")
    xyz_s, nrm_s = xyz[o], nrm[o]
    del cols, xyz, nrm, keys, o
    with np.load(f"{root}/{SP_NPZ}") as z:
        sp_id = z["sp_id"]
    n_sp = int(sp_id.max()) + 1

    with open(f"{root}/semantics/classes.json") as f:
        gmap = _group_map(json.load(f)["id2label"])
    sc = Scene("oxford_spires", seq, data_root=DATA_MOUNT)
    cam_models = {c: FisheyeCamera.from_dict(sc.camera(c)) for c in sc.camera_ids()}
    with open(f"{root}/poses/cameras.json") as f:
        poses = json.load(f)["cameras"]
    cams = sorted(poses)
    frames = sc.load_frames()
    idx = {c: sc.load_camera_index(c) for c in cams}

    trip = []
    for t in range(len(frames["frame_id"])):
        st = {c: int(idx[c]["stamp_ns"][frames[f"{c}_idx"][t]]) for c in cams}
        if all(str(st[c]) in poses[c] for c in cams):
            trip.append((t, st))
    trip = trip[part::n_parts]

    w = [np.zeros(n_sp, np.float32) for _ in cams]        # poids par caméra (réinitialisés par triplet)
    gh = [np.zeros((n_sp, len(GROUPS)), np.float32) for _ in cams]
    pair_tot = [0, 0, 0]
    pair_ok = [0, 0, 0]
    vox_tot = [0, 0, 0]
    vox_ok = [0, 0, 0]
    n_trip = 0
    for t, st in trip:
        seen_pts = {}
        for k, c in enumerate(cams):
            T_mc = np.asarray(poses[c][str(st[c])]["T_map_cam_start"], np.float64)
            R, cc = T_mc[:3, :3].T, T_mc[:3, 3]  # R = rotation carte→caméra (T_map_cam est caméra→carte)
            sem = np.asarray(Image.open(f"{root}/semantics/{c}/{st[c]}.png"))
            msk = np.asarray(Image.open(f"{root}/masks/{c}/{st[c]}.png"))
            model = cam_models[c]
            cw = (model.width + ZCELL - 1) // ZCELL
            d3 = xyz_s - cc
            cand = np.flatnonzero(np.einsum("ij,ij->i", d3, d3) <= MAX_RANGE_M**2)
            pc = d3[cand] @ R.T
            uv, valid, _ = model.project(pc, margin=MARGIN_PX)
            j = np.flatnonzero(valid)
            cand, pc, uv = cand[j], pc[j], uv[j]
            depth = np.sqrt(np.einsum("ij,ij->i", pc, pc))
            view_w = (pc / depth[:, None]) @ R
            cos_i = np.einsum("ij,ij->i", nrm_s[cand], -view_w)
            keep = cos_i >= COS_MIN
            cand, depth, cos_i, pc, uv = (a[keep] for a in (cand, depth, cos_i, pc, uv))
            ui = uv[:, 0].round().astype(np.int64)
            vi = uv[:, 1].round().astype(np.int64)
            cls = sem[vi, ui].astype(np.int64)
            bad = ((msk[vi, ui] & (BIT_SKY | BIT_PERSON)) != 0) | (cls >= 150)
            ok = ~bad
            cand, depth, cos_i, ui, vi, cls = (a[ok] for a in (cand, depth, cos_i, ui, vi, cls))
            cell = (vi // ZCELL) * cw + (ui // ZCELL)
            dmin = np.full(270 * cw, np.inf)
            np.minimum.at(dmin, cell, depth)
            vis = depth <= dmin[cell] + DEPTH_TOL
            cand, cos_i, cls = cand[vis], cos_i[vis], cls[vis]
            seen_pts[c] = (sp_id[cand], cos_i.astype(np.float32), gmap[cls].astype(np.int64))
            w[k].fill(0)
            gh[k].fill(0)
            np.add.at(w[k], seen_pts[c][0], seen_pts[c][1])
            np.add.at(gh[k], (seen_pts[c][0], seen_pts[c][2]), 1.0)  # [2] = groupe (int) ; [1] = cos (float)
        argm = [gh[k].argmax(1) for k in range(3)]
        for a, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            m = (w[i] >= 0.5) & (w[j] >= 0.5)
            pair_tot[a] += int(m.sum())
            pair_ok[a] += int((argm[i][m] == argm[j][m]).sum())
        # niveau voxel : même voxel vu par 2 caméras du triplet
        common = None
        ws = {}
        for k, c in enumerate(cams):
            spv, wv, gv = seen_pts[c]
            u, inv = np.unique(spv, return_inverse=True)
            wsum = np.zeros(len(u), np.float32)
            np.add.at(wsum, inv, wv)
            ws[c] = (u, inv, wsum, gv)
        for a, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            ui_, invi, wi_, gi_ = ws[cams[i]]
            uj_, invj, wj_, gj_ = ws[cams[j]]
            common, xi, xj = np.intersect1d(ui_, uj_, assume_unique=True, return_indices=True)
            if len(common) == 0:
                continue
            # groupe dominant de chaque superpoint pour cette caméra dans CE triplet
            ghi = np.zeros((len(ui_), len(GROUPS)), np.float32)
            np.add.at(ghi, (invi, gi_), 1.0)
            di = ghi.argmax(1)
            ghj = np.zeros((len(uj_), len(GROUPS)), np.float32)
            np.add.at(ghj, (invj, gj_), 1.0)
            dj = ghj.argmax(1)
            m = (wi_[xi] >= 0.5) & (wj_[xj] >= 0.5)
            vox_tot[a] += int(m.sum())
            vox_ok[a] += int((di[xi[m]] == dj[xj[m]]).sum())
        n_trip += 1
    out = {"part": part, "n_triplets": n_trip,
           "sp_pairs_tot": pair_tot, "sp_pairs_ok": pair_ok,
           "vox_pairs_tot": vox_tot, "vox_pairs_ok": vox_ok,
           "s": round(time.time() - t0, 1)}
    out_dir = f"{root}/{OUT_BASE}/triplet_agree"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/part_{part}.json", "w") as f:
        json.dump(out, f)
    vol.commit()
    print(json.dumps(out), flush=True)
    return out


@app.function(image=lift_image(), volumes={DATA_MOUNT: vol}, cpu=2, memory=8 * 1024, timeout=1800)
def triplet_sum(n_parts: int) -> dict:
    import json
    import os

    from splatpipe.stage1.maptools import scene_root
    from splatpipe.stage2.lift_semantics import OUT_BASE

    root = scene_root(DATA_MOUNT, seq=SEQ)
    out_dir = f"{root}/{OUT_BASE}/triplet_agree"
    parts = []
    for p in range(n_parts):
        with open(f"{out_dir}/part_{p}.json") as f:
            parts.append(json.load(f))
    tot = [sum(o["sp_pairs_tot"][a] for o in parts) for a in range(3)]
    ok = [sum(o["sp_pairs_ok"][a] for o in parts) for a in range(3)]
    vtot = [sum(o["vox_pairs_tot"][a] for o in parts) for a in range(3)]
    vok = [sum(o["vox_pairs_ok"][a] for o in parts) for a in range(3)]
    res = {"n_triplets": sum(o["n_triplets"] for o in parts),
           "sp_agree_per_pair": [round(o / t, 4) if t else None for o, t in zip(ok, tot)],
           "sp_agree_all": round(sum(ok) / sum(tot), 4) if sum(tot) else None,
           "vox_agree_per_pair": [round(o / t, 4) if t else None for o, t in zip(vok, vtot)],
           "vox_agree_all": round(sum(vok) / sum(vtot), 4) if sum(vtot) else None,
           "n_sp_pairs": sum(tot), "n_vox_pairs": sum(vtot)}
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.local_entrypoint()
def triplet_spawn_main(part: int, n_parts: int = 6):
    triplet_agree.spawn(part, n_parts)
    print(f"spawned triplet part {part}")


@app.local_entrypoint()
def triplet_sum_main(n_parts: int = 6):
    import json

    print(json.dumps(triplet_sum.remote(n_parts), indent=1, ensure_ascii=False))
