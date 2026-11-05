"""Sous-étape 2.2 — retrait des points dynamiques par DUFOMap sur les tours corrigés (`lidar/hesai_deskewed`).

Méthode. DUFOMap (BSD-3, liaisons Python `dufomap==1.1.1`, paramètres de l'article : `resolution=0.1`,
`d_s=0.2`, `d_p=2`) construit une carte d'occupation par lancer de rayons : un voxel traversé par un rayon est
de l'espace libre, un voxel touché est un obstacle. Un point est « dynamique » si son voxel est observé libre
à une autre date. Les tours sont déjà en repère `map` (`cloud_transform=False`) ; l'origine des rayons est la
position capteur au milieu du tour ((T_start+T_end)/2). Les deux échos sont gardés comme touches.

Entrée : `lidar/hesai_deskewed/<stamp>.npz` (xyz, repère map ; origines `T_map_lidar_start/end`).
Sorties :
  lidar/dyn_labels_dufo/<stamp>.npz   sidecar : `dyn` uint8 par point (0 statique, 1 dynamique), même ordre
                                      que `xyz` du tour correspondant ; + index.json (résumé, paramètres) ;
  map/map_5cm_static.ply              carte 5 cm des seuls points statiques (même format que map_5cm.ply),
                                      repère map — voir `static_map_main` ci-dessous.

Contrainte des liaisons : pas de save/load de la carte → construction + étiquetage dans le même conteneur.
`run()` est appelé tour par tour (construction), puis `segment()` tour par tour (étiquetage).

Lancement :
  .venv/bin/modal run splatpipe/stage2/dufomap_run.py::pilot_main                # 1 tour sur 20, minuterie
  .venv/bin/modal run splatpipe/stage2/dufomap_run.py::full_main                 # tous les tours
  .venv/bin/modal run splatpipe/stage2/dufomap_run.py::static_map_main           # carte statique 5 cm
"""

from __future__ import annotations

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume
from splatpipe.stage2.images import dufomap_image

app = modal.App("splatsim-stage2-dufomap")
vol = data_volume()

SEQ = "2024-03-13-observatory-quarter-01"
DESK_SUB = "lidar/hesai_deskewed"
OUT_SUB = "lidar/dyn_labels_dufo"
RESOLUTION = 0.1
D_S = 0.2
D_P = 2  # int exigé par les liaisons pybind11


def _list_deskewed(root: str) -> list[int]:
    import json
    import os

    with open(f"{root}/{DESK_SUB}/index.json") as f:
        idx = json.load(f)
    stamps = sorted(int(s[:-4]) for s in os.listdir(f"{root}/{DESK_SUB}") if s.endswith(".npz"))
    return stamps


@app.function(image=dufomap_image(), volumes={DATA_MOUNT: vol}, cpu=16, memory=128 * 1024, timeout=3 * 3600)
def build_and_label(stamps: list[int], resolution: float = RESOLUTION, d_s: float = D_S, d_p: int = D_P,
                    out_sub: str = OUT_SUB, seq: str = SEQ, label: bool = True) -> dict:
    import json
    import os
    import time

    import numpy as np
    from dufomap import dufomap as Dufo

    from splatpipe.stage1.maptools import scene_root

    t0 = time.time()
    root = scene_root(DATA_MOUNT, seq)
    dufo = Dufo(resolution, d_s, d_p, num_threads=0)
    build_s, origins_used = [], []
    n_build_pts = 0
    for st in stamps:
        t1 = time.time()
        with np.load(f"{root}/{DESK_SUB}/{st}.npz") as z:
            xyz = z["xyz"]
            T0_, T1_ = z["T_map_lidar_start"], z["T_map_lidar_end"]
        origin = T0_.copy()
        origin[:3, 3] = 0.5 * (T0_[:3, 3] + T1_[:3, 3])
        dufo.run(xyz, origin, cloud_transform=False)
        origins_used.append(float(np.linalg.norm(origin[:3, 3])))
        build_s.append(time.time() - t1)
        n_build_pts += len(xyz)
    t_build = time.time() - t0
    dufo.oncePropagateCluster(if_propagate=True, if_cluster=False)

    out = {"stamps": [], "dyn_frac": [], "n_points": [], "segment_s": []}
    if label:
        os.makedirs(f"{root}/{out_sub}", exist_ok=True)
        t2 = time.time()
        for st in stamps:
            t3 = time.time()
            with np.load(f"{root}/{DESK_SUB}/{st}.npz") as z:
                xyz = z["xyz"]
                T0_, T1_ = z["T_map_lidar_start"], z["T_map_lidar_end"]
            origin = T0_.copy()
            origin[:3, 3] = 0.5 * (T0_[:3, 3] + T1_[:3, 3])
            dyn = dufo.segment(xyz, origin, cloud_transform=False)
            np.savez_compressed(f"{root}/{out_sub}/{st}.npz", dyn=dyn.astype(np.uint8), stamp_ns=np.int64(st),
                                resolution=np.float32(resolution), d_s=np.float32(d_s), d_p=np.float32(d_p))
            out["stamps"].append(int(st))
            out["dyn_frac"].append(float(dyn.mean()))
            out["n_points"].append(int(len(xyz)))
            out["segment_s"].append(round(time.time() - t3, 3))
        t_label = time.time() - t2
        vol.commit()
    else:
        t_label = 0.0

    fr = np.array(out["dyn_frac"]) if out["dyn_frac"] else np.array([0.0])
    summary = {
        "params": {"resolution": resolution, "d_s": d_s, "d_p": d_p},
        "n_scans": len(stamps), "n_points": int(n_build_pts),
        "build_s": round(t_build, 1), "label_s": round(t_label, 1),
        "build_ms_per_scan": round(1000 * t_build / max(len(stamps), 1), 1),
        "dyn_frac": {"mean": float(fr.mean()), "median": float(np.median(fr)), "p90": float(np.percentile(fr, 90)),
                     "max": float(fr.max())},
        "origin_norm_m": {"min": float(min(origins_used)), "max": float(max(origins_used))},
        "n_labeled": len(out["stamps"]),
    }
    return summary | {"per_scan": out}


@app.function(image=dufomap_image(), volumes={DATA_MOUNT: vol}, cpu=1, memory=2048, timeout=600)
def deskewed_stamps(seq: str = SEQ, stride: int = 1) -> list[int]:
    from splatpipe.stage1.maptools import scene_root

    root = scene_root(DATA_MOUNT, seq)
    return _list_deskewed(root)[::stride]


@app.local_entrypoint()
def pilot2_main(stride: int = 20, seq: str = SEQ, resolution: float = RESOLUTION, d_s: float = D_S, d_p: int = D_P):
    import json

    stamps = deskewed_stamps.remote(seq, stride)
    out = build_and_label.remote(stamps, resolution, d_s, d_p, out_sub=f"{OUT_SUB}_pilot", seq=seq)
    with open("logs/stage2/dufomap_pilot.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_scan"}, indent=1))


@app.local_entrypoint()
def full_main(seq: str = SEQ, resolution: float = RESOLUTION, d_s: float = D_S, d_p: int = D_P):
    """Tous les tours : la carte DOIT être construite d'un seul tenant (pas de save/load des liaisons)."""
    import json
    import time

    stamps = deskewed_stamps.remote(seq, 1)
    t0 = time.time()
    out = build_and_label.remote(stamps, resolution, d_s, d_p, out_sub=OUT_SUB, seq=seq)
    write_index.remote(seq, OUT_SUB, resolution, d_s, d_p)
    with open("logs/stage2/dufomap.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_scan"} | {"wall_s": round(time.time() - t0, 1)}, indent=1))


@app.function(image=dufomap_image(), volumes={DATA_MOUNT: vol}, cpu=1, memory=2048, timeout=600)
def write_index(seq: str, out_sub: str, resolution: float, d_s: float, d_p: float):
    import json
    import os

    from splatpipe.stage1.maptools import scene_root

    root = scene_root(DATA_MOUNT, seq)
    doc = {
        "description": ("Étiquettes dynamiques DUFOMap par point (0 statique, 1 dynamique), même ordre que xyz de "
                        f"{DESK_SUB}/<stamp>.npz ; carte construite sur tous les tours corrigés (train + test_in)."),
        "params": {"resolution": resolution, "d_s": d_s, "d_p": d_p, "origin": "position capteur au milieu du tour",
                   "package": "dufomap==1.1.1"},
        "n_files": len([f for f in os.listdir(f"{root}/{out_sub}") if f.endswith(".npz")]),
    }
    with open(f"{root}/{out_sub}/index.json", "w") as f:
        json.dump(doc, f, indent=1)
    vol.commit()
