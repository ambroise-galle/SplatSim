"""Sous-étape 2.3 — sémantique LiDAR : projection des images dans les tours, vote par point, union dynamique.

Méthode. Pour chaque tour corrigé `lidar/hesai_deskewed/<stamp>.npz` et chaque caméra : on prend l'image du
triplet dont l'exposition est la plus proche du milieu du tour (|Δt| ≤ 60 ms), on projette les points dont le
temps propre est à ±30 ms de l'exposition (protocole T2.1) et à portée ≤ 55 m, et on lit, au pixel :
  - la classe ADE20K de `semantics/<cam>/<stamp>.png` (vote),
  - les bits de `masks/<cam>/<stamp>.png` (champ de bits : ciel 1, personne 2, véhicule 4, végétation 8).
Chaque point accumule ≤ 3 votes (un par caméra). Sortie `semantics_lidar/<stamp>.npz`, même ordre que `xyz` :
  ade     uint8  classe ADE20K majoritaire (255 = aucun vote) ; égalité → priorité cam0 > cam1 > cam2
  n_votes uint8  nombre de caméras ayant voté (0–3)
  agree   uint8  0 = 1 vote, 1 = tous d'accord, 2 = en désaccord
  bits    uint8  union des bits de masque vus
  dyn     uint8  **union 2.2/2.3 décidée au T2.1** : étiquette DUFOMap OU bit « personne » projeté
Agrégats par tour (et index.json) : accord inter-caméras (≥ 2 votes) sur la classe et sur le bit végétation,
part de points « ciel » (fuite attendue ≈ 0 : les retours ne frappent pas le ciel), fractions par bit.

`map_sem_main` agrège ensuite les points d'entraînement **statiques** (dyn == 0) par voxel 5 cm (même grille
que `map_5cm_static.ply`) : classe majoritaire par nombre de points, part de végétation, n_votes — écrit
`map/map_5cm_static_sem.ply` (repère map) pour l'initialisation et les régularisations de l'étape 4.

Lancement :
  .venv/bin/modal run splatpipe/stage2/lidar_semantics.py::sem_main                 # tous les tours
  .venv/bin/modal run splatpipe/stage2/lidar_semantics.py::map_sem_main             # carte sémantique 5 cm
"""

from __future__ import annotations

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume
from splatpipe.stage1.images import map_image

app = modal.App("splatsim-stage2-sem-lidar")
vol = data_volume()

SEQ = "2024-03-13-observatory-quarter-01"
OUT_SUB = "semantics_lidar"
TIME_WINDOW_S = 0.03
FRAME_WINDOW_S = 0.06
MARGIN_PX = 2
MAX_RANGE_M = 55.0
BIT_SKY, BIT_PERSON, BIT_VEHICLE, BIT_VEGETATION = 1, 2, 4, 8


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=2, memory=32 * 1024, timeout=3 * 3600)
def project_chunk(stamps: list[int], seq: str = SEQ) -> dict:
    import json
    import os
    import time
    from collections import Counter

    import numpy as np
    from PIL import Image

    from splatpipe.common.camera_model import FisheyeCamera
    from splatpipe.common.scene import Scene

    t0 = time.time()
    sc = Scene("oxford_spires", seq, data_root=DATA_MOUNT)
    cams = sc.camera_ids()
    with open(sc.path("poses", "cameras.json")) as f:
        poses = json.load(f)["cameras"]
    idx = {c: sc.load_camera_index(c) for c in cams}
    cam_models = {c: FisheyeCamera.from_dict(sc.camera(c)) for c in cams}
    cam_times = {c: idx[c]["t"].astype(np.float64) for c in cams}
    cam_stamps = {c: idx[c]["stamp_ns"] for c in cams}

    dd = sc.path("lidar", "hesai_deskewed")
    dl = sc.path("lidar", "dyn_labels_dufo")
    os.makedirs(sc.path(OUT_SUB), exist_ok=True)

    totals = Counter()
    per_scan = []
    errors = []
    for st in stamps:
        out_path = f"{sc.path(OUT_SUB)}/{st}.npz"
        if os.path.exists(out_path):  # reprise : tour déjà projeté
            continue
        try:
            ts = st / 1e9
            with np.load(f"{dd}/{st}.npz") as z:
                xyz, t_rel = z["xyz"], z["t_rel"]
                t_start = float(z["t_start"])
            with np.load(f"{dl}/{st}.npz") as z:
                dyn_dufo = z["dyn"].astype(bool)
            t_pt = t_start + t_rel.astype(np.float64)
            rng3 = np.linalg.norm(xyz.astype(np.float64), axis=1)
            n_pts = len(xyz)
            votes_ade = np.full((n_pts, 3), 255, np.uint8)
            votes_bits = np.zeros((n_pts, 3), np.uint8)
            for k, c in enumerate(cams):
                if c not in poses:
                    continue
                j = int(np.argmin(np.abs(cam_times[c] - (ts + 0.05))))
                if abs(float(cam_times[c][j]) - (ts + 0.05)) > FRAME_WINDOW_S:
                    continue
                key = str(int(cam_stamps[c][j]))
                if key not in poses[c]:
                    continue
                mpath = sc.path("masks", c, f"{key}.png")
                spath = sc.path("semantics", c, f"{key}.png")
                if not (os.path.exists(mpath) and os.path.exists(spath)):
                    continue
                t_cam = poses[c][key]["t_start"]
                T_mc = np.array(poses[c][key]["T_map_cam_start"])
                mask = np.asarray(Image.open(mpath))
                sem = np.asarray(Image.open(spath))
                keep = (np.abs(t_pt - t_cam) <= TIME_WINDOW_S) & (rng3 <= MAX_RANGE_M)
                if not keep.any():
                    continue
                uv, valid, _ = cam_models[c].project(
                    (xyz[keep].astype(np.float64) - T_mc[:3, 3]) @ T_mc[:3, :3], margin=MARGIN_PX)
                ui = uv[valid].round().astype(int)
                H, W = mask.shape
                inb = (ui[:, 0] >= 0) & (ui[:, 0] < W) & (ui[:, 1] >= 0) & (ui[:, 1] < H)
                if not inb.any():
                    continue
                sel = np.flatnonzero(keep)[valid][inb]
                votes_ade[sel, k] = sem[ui[inb, 1], ui[inb, 0]]
                votes_bits[sel, k] = mask[ui[inb, 1], ui[inb, 0]]

            # vote majoritaire : première caméra qui vote dans l'ordre cam0 > cam1 > cam2 (les 3 voient
            # presque toujours la même classe ; en cas d'égalité stricte la priorité tranche)
            ade = np.full(n_pts, 255, np.uint8)
            n_votes = (votes_ade != 255).sum(1).astype(np.uint8)
            for k in range(3):
                take = (votes_ade[:, k] != 255) & (ade == 255)
                ade[take] = votes_ade[take, k]
            # accord : tous les votes identiques (2 ou 3 votes)
            has = [votes_ade[:, k] != 255 for k in range(3)]
            eq01 = has[0] & has[1] & (votes_ade[:, 0] == votes_ade[:, 1])
            eq02 = has[0] & has[2] & (votes_ade[:, 0] == votes_ade[:, 2])
            eq12 = has[1] & has[2] & (votes_ade[:, 1] == votes_ade[:, 2])
            all_eq = (eq01 & eq12) | (eq01 & ~has[2]) | (eq02 & ~has[1]) | (eq12 & ~has[0])
            agree = np.zeros(n_pts, np.uint8)
            agree[n_votes >= 2] = np.where(all_eq[n_votes >= 2], 1, 2).astype(np.uint8)
            # union des bits et union dynamique (T2.1 : DUFOMap ∪ personne)
            bits = np.zeros(n_pts, np.uint8)
            for k in range(3):
                bits |= votes_bits[:, k]
            person = (bits & BIT_PERSON) > 0
            dyn_any = dyn_dufo | person
            np.savez_compressed(
                out_path, ade=ade, n_votes=n_votes, agree=agree, bits=bits,
                dyn=dyn_any.astype(np.uint8), stamp_ns=np.int64(st))

            veg_votes = (votes_bits & BIT_VEGETATION) > 0
            n_vot = veg_votes.sum(1)
            multi = n_votes >= 2
            veg_unanim = multi & ((n_vot == 0) | (n_vot == n_votes))
            veg_m = (bits & BIT_VEGETATION) > 0
            per_scan.append({
                "stamp": int(st), "n_points": int(n_pts), "n_votes_mean": float(n_votes.mean()),
                "frac_person": float(person.mean()), "frac_veg": float(veg_m.mean()),
                "frac_vehicle": float(((bits & BIT_VEHICLE) > 0).mean()),
                "frac_sky": float(((bits & BIT_SKY) > 0).mean()), "frac_dyn_any": float(dyn_any.mean()),
                "n_multi": int(multi.sum()), "n_agree": int((agree == 1).sum()),
                "n_veg_multi": int((multi & veg_m).sum()),
                "n_veg_unanim": int(veg_unanim[veg_m & multi].sum()) if (multi & veg_m).sum() else 0,
            })
            for k, v in (("n_person", person.sum()), ("n_veg", veg_m.sum()),
                         ("n_sky", ((bits & BIT_SKY) > 0).sum()), ("n_dyn_any", dyn_any.sum()),
                         ("n_multi", multi.sum()), ("n_agree", (agree == 1).sum()),
                         ("n_veg_multi_unanim", (veg_m & multi & veg_unanim).sum())):
                totals[k] += int(v)
        except Exception as e:  # un mauvais tour ne doit pas tuer le chunk ; le tour sera refait à la reprise
            errors.append({"stamp": int(st), "error": f"{type(e).__name__}: {e}"[:300]})
    vol.commit()
    return {"per_scan": per_scan, "n_scans": len(stamps), "errors": errors, "totals": dict(totals),
            "chunk_s": round(time.time() - t0, 1)}


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=1, memory=2048, timeout=600)
def deskewed_stamps(seq: str = SEQ) -> list[int]:
    import os

    from splatpipe.stage1.maptools import scene_root

    root = scene_root(DATA_MOUNT, seq)
    return sorted(int(s[:-4]) for s in os.listdir(f"{root}/lidar/hesai_deskewed") if s.endswith(".npz"))


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=4, memory=64 * 1024, timeout=2 * 3600)
def aggregate_all(seq: str = SEQ) -> dict:
    """Ré-agrège les totaux sur TOUS les sidecars existants (les reprises partielles ne comptent que
    les tours qu'elles ont traités)."""
    import json
    import os
    import time

    import numpy as np

    from splatpipe.common.scene import Scene

    t0 = time.time()
    sc = Scene("oxford_spires", seq, data_root=DATA_MOUNT)
    sem_dir = sc.path(OUT_SUB)
    stamps = sorted(int(s[:-4]) for s in os.listdir(sem_dir) if s.endswith(".npz"))
    tot: dict = {}
    per_scan = []
    n_pts_tot = 0
    for st in stamps:
        with np.load(f"{sem_dir}/{st}.npz") as z:
            ade, nv, agree, bits, dyn = z["ade"], z["n_votes"], z["agree"], z["bits"], z["dyn"]
        n = len(ade)
        n_pts_tot += n
        multi = nv >= 2
        veg = (bits & BIT_VEGETATION) > 0
        person = (bits & BIT_PERSON) > 0
        for k, v in (("n_person", person.sum()), ("n_veg", veg.sum()),
                     ("n_sky", ((bits & BIT_SKY) > 0).sum()), ("n_dyn_any", dyn.sum()),
                     ("n_multi", multi.sum()), ("n_agree", (agree == 1).sum())):
            tot[k] = tot.get(k, 0) + int(v)
        per_scan.append({"stamp": int(st), "n_points": int(n), "n_votes_mean": float(nv.mean()),
                         "frac_veg": float(veg.mean()), "frac_person": float(person.mean()),
                         "frac_dyn_any": float(dyn.mean())})
    multi = tot.get("n_multi", 0)
    summary = {"n_scans": len(stamps), "n_points": n_pts_tot,
               "protocol": {"time_window_s": TIME_WINDOW_S, "frame_window_s": FRAME_WINDOW_S,
                            "margin_px": MARGIN_PX, "max_range_m": MAX_RANGE_M},
               "frac": {k[2:].lower(): round(tot.get(k, 0) / n_pts_tot, 5)
                        for k in ("n_person", "n_veg", "n_sky", "n_dyn_any")},
               "multi_vote_frac": round(multi / n_pts_tot, 4),
               "class_agree_frac_of_multi": round(tot.get("n_agree", 0) / max(multi, 1), 4),
               "totals": tot, "aggregate_s": round(time.time() - t0, 1)}
    return {"summary": summary, "per_scan": per_scan}


@app.local_entrypoint()
def aggregate_main(seq: str = SEQ):
    import json
    import os

    out = aggregate_all.remote(seq)
    s = out["summary"]
    with open("logs/stage2/sem_lidar.json", "w") as f:
        json.dump(s | {"per_scan": out["per_scan"]}, f, indent=1)
    print(json.dumps(s, indent=1))


@app.local_entrypoint()
def sem_main(n_chunks: int = 24, seq: str = SEQ):
    import json
    import os
    import time

    import numpy as np

    stamps = deskewed_stamps.remote(seq)
    chunks = [c.tolist() for c in np.array_split(np.array(stamps, dtype=np.int64), n_chunks)]
    t0 = time.time()
    outs = list(project_chunk.map(chunks, kwargs={"seq": seq}, return_exceptions=True))
    per_scan, errors = [], []
    for o in outs:
        if isinstance(o, Exception):
            errors.append({"chunk_error": str(o)[:500]})
            continue
        per_scan += o["per_scan"]
        errors += o.get("errors", [])
    totals = {}
    for o in outs:
        if isinstance(o, Exception):
            continue
        for k, v in o["totals"].items():
            totals[k] = totals.get(k, 0) + v
    n_tot = sum(p["n_points"] for p in per_scan)
    multi = totals.get("n_multi", 0)
    summary = {
        "n_scans": len(stamps), "n_points": n_tot, "wall_s": round(time.time() - t0, 1),
        "protocol": {"time_window_s": TIME_WINDOW_S, "frame_window_s": FRAME_WINDOW_S,
                     "margin_px": MARGIN_PX, "max_range_m": MAX_RANGE_M},
        "frac": {k[2:].lower(): round(totals.get(k, 0) / n_tot, 5)
                 for k in ("n_person", "n_veg", "n_sky", "n_dyn_any")},
        "multi_vote_frac": round(multi / n_tot, 4),
        "class_agree_frac_of_multi": round(totals.get("n_agree", 0) / max(multi, 1), 4),
        "totals": totals,
    }
    # part des points multi-votes ET végétation dont les caméras sont unanimes :
    n_veg_multi = sum(p["n_veg_multi"] for p in per_scan)
    n_veg_unanim = sum(p["n_veg_unanim"] for p in per_scan)
    summary["veg_unanim_frac_of_multi_veg"] = round(n_veg_unanim / max(n_veg_multi, 1), 4)
    summary["n_veg_multi"] = n_veg_multi
    write_sem_index.remote(seq, summary)
    os.makedirs("logs", exist_ok=True)
    with open("logs/stage2/sem_lidar.json", "w") as f:
        json.dump(summary | {"per_scan": per_scan, "errors": errors}, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_scan"} | {"n_errors": len(errors)}, indent=1))


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=1, memory=2048, timeout=600)
def write_sem_index(seq: str, summary: dict):
    import json

    from splatpipe.common.scene import Scene

    sc = Scene("oxford_spires", seq, data_root=DATA_MOUNT)
    doc = dict(summary)
    doc["description"] = (
        "Sémantique LiDAR par point (même ordre que xyz de lidar/hesai_deskewed/<stamp>.npz) : "
        "ade uint8 (classe ADE20K majoritaire, 255 = sans vote), n_votes uint8 (0–3 caméras), "
        "agree uint8 (0 = 1 vote, 1 = accord, 2 = désaccord), bits uint8 (union des masques 2.1 : "
        "ciel 1, personne 2, véhicule 4, végétation 8), dyn uint8 = DUFOMap OU bit personne (union T2.1).")
    with open(sc.path(OUT_SUB, "index.json"), "w") as f:
        json.dump(doc, f, indent=1)
    vol.commit()


# =============================================================================
# Carte sémantique 5 cm (points statiques d'entraînement, vote par voxel)
# =============================================================================

GROUPS = {"vegetation": 3, "person": 1, "vehicle": 2, "sky": 0}
GROUP_COLORS = {"vegetation": (40, 170, 60), "person": (230, 40, 40), "vehicle": (235, 200, 40),
                "sky": (60, 140, 235), "structure": (150, 150, 150)}


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=16, memory=128 * 1024, timeout=3 * 3600)
def map_sem_build(seq: str = SEQ) -> dict:
    import json
    import os
    import time

    import numpy as np
    from PIL import Image

    from splatpipe.stage1.maptools import group_sorted, lidar_splits, list_scans, pack_ijk, scene_root, write_ply

    T0 = time.time()
    root = scene_root(DATA_MOUNT, seq)
    stamps, paths = list_scans(root)
    train = set(lidar_splits(root)["train"].tolist())
    sem_dir = f"{root}/{OUT_SUB}"
    have = {int(s[:-4]) for s in os.listdir(sem_dir) if s.endswith(".npz")}
    desk = {int(s[:-4]) for s in os.listdir(f"{root}/lidar/hesai_deskewed") if s.endswith(".npz")}
    idx = [i for i, s in enumerate(stamps) if int(s) in train and int(s) in desk and int(s) in have]

    xyz_l, ade_l, bits_l, nv_l = [], [], [], []
    for i in idx:
        st = int(stamps[i])
        with np.load(f"{root}/lidar/hesai_deskewed/{st}.npz") as z:
            xyz_l.append(z["xyz"])
        with np.load(f"{sem_dir}/{st}.npz") as z:
            ade_l.append(z["ade"])
            bits_l.append(z["bits"])
            nv_l.append(z["n_votes"])
    xyz = np.concatenate(xyz_l).astype(np.float64)
    ade = np.concatenate(ade_l)
    bits = np.concatenate(bits_l)
    nv = np.concatenate(nv_l)
    veg = (bits & BIT_VEGETATION) > 0
    info = {"n_scans": len(idx), "n_points": int(len(xyz))}

    ijk = np.floor(xyz / 0.05).astype(np.int64)
    keys = pack_ijk(ijk)
    del ijk
    order, starts, ukeys = group_sorted(keys)
    counts = np.diff(np.r_[starts, len(keys)]).astype(np.int64)
    cen = np.add.reduceat(xyz[order], starts, axis=0) / counts[:, None]
    info["n_voxels"] = int(len(ukeys))

    # classe majoritaire par voxel (nombre de points ; égalité → plus petit id ADE).
    # paire (clé, ade) en dtype structuré : keys tient sur 63 bits, on ne peut pas les décaler de 8.
    voted = ade != 255
    dt = np.dtype([("k", np.int64), ("a", np.uint8)])
    pairs = np.empty(int(voted.sum()), dtype=dt)
    pairs["k"] = keys[voted]
    pairs["a"] = ade[voted]
    uk, ucnt = np.unique(pairs, return_counts=True)
    del pairs, voted
    vox, cls = uk["k"], uk["a"].astype(np.int64)
    o = np.lexsort((cls, -ucnt, vox))
    vox, cls, ucnt = vox[o], cls[o], ucnt[o]
    first = np.r_[True, vox[1:] != vox[:-1]]
    vox_maj, cls_maj, cnt_maj = vox[first], cls[first], ucnt[first]
    pos = np.searchsorted(ukeys, vox_maj)
    ade_vox = np.full(len(ukeys), 255, np.uint8)
    ade_vox[pos] = cls_maj.astype(np.uint8)
    del vox, cls, ucnt, vox_maj, cnt_maj, pos

    # agrégats par voxel (reduceat dans l'ordre groupé de group_sorted)
    veg_frac = np.add.reduceat(veg[order].astype(np.float64), starts) / counts
    nv_mean = np.add.reduceat(nv[order].astype(np.float64), starts) / counts

    def frac_bit(b: int) -> np.ndarray:
        return np.add.reduceat((((bits & b) > 0)[order]).astype(np.float64), starts)

    groups = {name: frac_bit(1 << b) for name, b in
              (("vegetation", 3), ("person", 1), ("vehicle", 2), ("sky", 0))}
    del keys, order
    # couleur = groupe dominant parmi les voxels VOTÉS (sinon structure) ; égalité → végétation d'abord
    stack = np.stack([groups["vegetation"], groups["person"], groups["vehicle"], groups["sky"]])
    any_sem = stack.max(0) > 0
    dom = stack.argmax(0)
    dom[~any_sem] = -1
    names = ["vegetation", "person", "vehicle", "sky"]
    rgb = np.array([GROUP_COLORS["structure"]] * len(ukeys), np.uint8)
    for gi, name in enumerate(names):
        rgb[dom == gi] = GROUP_COLORS[name]
    veg_frac_v = groups["vegetation"]

    ply = f"{root}/map/map_5cm_static_sem.ply"
    cols = {"x": cen[:, 0].astype(np.float32), "y": cen[:, 1].astype(np.float32), "z": cen[:, 2].astype(np.float32),
            "ade": ade_vox, "veg_frac": veg_frac_v.astype(np.float32), "n_points": counts.astype(np.uint32),
            "n_votes_mean": nv_mean.astype(np.float32), "red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2]}
    comments = ["SplatSim etape 2.3 carte semantique (voxel 0.05 m, points statiques dyn==0, tours train)",
                "ade = classe ADE20K majoritaire (255 = sans vote), veg_frac = part vegetation, n_votes_mean 0-3",
                "rgb = groupe dominant : vert vegetation, rouge personne, jaune véhicule, bleu ciel, gris structure",
                "meme grille que map_5cm_static.ply (floor(xyz/0.05))"]
    info["ply_bytes"] = write_ply(ply, cols, comments=comments)
    n_sem = int(any_sem.sum())
    info["n_voxels_with_votes"] = n_sem
    info["frac_vox_with_votes"] = float(n_sem / len(ukeys))
    info["dominant_among_voted"] = {name: float((dom[any_sem] == gi).mean()) for gi, name in enumerate(names)}
    info["frac_vox_with_class"] = float((ade_vox != 255).mean())
    json.dump(info, open(f"{root}/reports/stage2_sem_map.json", "w"), indent=1)
    vol.commit()
    info["total_s"] = round(time.time() - T0, 1)
    return info


@app.local_entrypoint()
def map_sem_main(seq: str = SEQ):
    import json

    print(json.dumps(map_sem_build.remote(seq), indent=1))


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=8, memory=64 * 1024, timeout=1800)
def render_topview(seq: str = SEQ, n_sample: int = 2_500_000) -> dict:
    """Vue de dessus de la carte sémantique (sous-échantillon), couleurs = groupe dominant."""
    import os
    import struct

    import numpy as np
    from PIL import Image

    from splatpipe.common.scene import Scene

    sc = Scene("oxford_spires", seq, data_root=DATA_MOUNT)
    p = sc.path("map", "map_5cm_static_sem.ply")
    with open(p, "rb") as f:
        hdr, n = [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            hdr.append(line)
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line == "end_header":
                break
        hdr_size = f.tell()
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("ade", "u1"), ("veg_frac", "<f4"),
                       ("n_points", "<u4"), ("n_votes_mean", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        step = max(1, n // n_sample)
        xs, ys, cs = [], [], []
        for k in range(0, n, step):
            f.seek(hdr_size + dt.itemsize * k)
            a = np.fromfile(f, dtype=dt, count=1)[0]
            xs.append(a["x"]), ys.append(a["y"]), cs.append((a["red"], a["green"], a["blue"]))
        xs, ys = np.array(xs), np.array(ys)
        cols = np.array(cs, np.uint8)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    o = np.random.default_rng(0).permutation(len(xs))
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#fcfcfb")
    ax.scatter(xs[o], ys[o], c=cols[o] / 255.0, s=0.06, rasterized=True, linewidths=0)
    ax.set_facecolor("#fcfcfb")
    ax.set_aspect("equal")
    ax.set_title("2.3 — carte sémantique statique 5 cm (voxels votés)\nvert végétation · jaune véhicules · rouge personnes · bleu fuite ciel · gris structure",
                 fontsize=10, color="#1f1f1e", loc="left")
    ax.set_xlabel("x map (m)", fontsize=8, color="#5f5e5a")
    ax.set_ylabel("y map (m)", fontsize=8, color="#5f5e5a")
    ax.tick_params(colors="#5f5e5a", labelsize=8)
    fig.savefig("/tmp/sem_topview.png", dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    out = sc.path("reports", "stage2_sem_topview.png")
    Image.open("/tmp/sem_topview.png").save(out)
    vol.commit()
    return {"n_shown": int(len(xs)), "n_voxels": n, "png": os.path.basename(out),
            "png_bytes": os.path.getsize(out)}


@app.local_entrypoint()
def map_sem_view_main(seq: str = SEQ):
    import json
    import os

    out = render_topview.remote(seq)
    os.makedirs("logs/stage2/dufo", exist_ok=True)
    print(json.dumps(out, indent=1))
