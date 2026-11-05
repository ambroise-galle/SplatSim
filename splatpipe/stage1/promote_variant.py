"""Sous-étape 1.7a — promotion d'une variante de trajectoire en sorties canoniques de l'étape 1.

Après un essai `final_variant_main` / `deskew_main` / `maps_main --suffix` / `evaluate_variant_main` réussi,
remplace les fichiers canoniques par la variante suffixée, en gardant les anciens sous un nom de secours :

  traj/spline.npz                  ← traj/spline_calib.npz        (ancien → traj/spline_calib_r4.npz)
  traj/final.tum                   ← traj/final_calib.tum         (ancien → traj/final_calib_r4.tum)
  traj/final_to_world.yaml         ← ..._calib.yaml               (ancien → ..._calib_r4.yaml)
  map/map_5cm.ply                  ← map/map_5cm_calib.ply        (ancien → map/map_5cm_calib_r4.ply)
  reports/.../t12/glim_final.npz   ← glim_final_calib.npz         (ancien → glim_final_calib_r4.npz)
  lidar/hesai_deskewed/            ← lidar/hesai_deskewed_calib/  (ancien → lidar/hesai_deskewed_calib_r4/, échange)

et, sur `--clean-eval`, supprime `map/eval/` (nuages 1 cm intermédiaires du protocole T1.3, régénérables par
`maps_main`). Ne touche jamais à `/data/raw/` ni aux trajectoires fournies (`traj/gt.tum`, `traj/glim*`).

Lancement :
  .venv/bin/modal run splatpipe/stage1/promote_variant.py::promote_main --suffix _calib
"""

from __future__ import annotations

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume
from splatpipe.stage1.images import map_image

app = modal.App("splatsim-stage1-promote")
vol = data_volume()

SEQ = "2024-03-13-observatory-quarter-01"
FILE_PAIRS = [
    ("traj/spline{suffix}.npz", "traj/spline.npz"),
    ("traj/final{suffix}.tum", "traj/final.tum"),
    ("traj/final_to_world{suffix}.yaml", "traj/final_to_world.yaml"),
    ("map/map_5cm{suffix}.ply", "map/map_5cm.ply"),
    ("reports/stage1_map/t12/glim_final{suffix}.npz", "reports/stage1_map/t12/glim_final.npz"),
]
DESKED_PAIR = ("lidar/hesai_deskewed", "lidar/hesai_deskewed{suffix}")


@app.function(image=map_image(), volumes={DATA_MOUNT: vol}, cpu=1, memory=2048, timeout=3600)
def promote(suffix: str, backup_suffix: str = "_calib_r4", seq: str = SEQ, clean_eval: bool = True) -> dict:
    import json
    import os
    import shutil

    from splatpipe.stage1.maptools import scene_root

    root = scene_root(DATA_MOUNT, seq)
    actions = []

    def split_ext(p: str) -> tuple[str, str]:
        return os.path.splitext(p)

    for src_t, dst in FILE_PAIRS:
        src = src_t.format(suffix=suffix)
        if not os.path.exists(f"{root}/{src}"):
            actions.append({"action": "skip_missing_source", "src": src})
            continue
        if os.path.exists(f"{root}/{dst}"):
            b, e = split_ext(dst)
            bak = f"{b}{backup_suffix}{e}"
            if os.path.exists(f"{root}/{bak}"):
                actions.append({"action": "skip_backup_exists", "backup": bak})
                os.remove(f"{root}/{dst}")
            else:
                os.rename(f"{root}/{dst}", f"{root}/{bak}")
                actions.append({"action": "backup", "from": dst, "to": bak})
        shutil.copyfile(f"{root}/{src}", f"{root}/{dst}")
        actions.append({"action": "promote", "from": src, "to": dst, "bytes": os.path.getsize(f"{root}/{dst}")})

    can_d, var_d = DESKED_PAIR[0], DESKED_PAIR[1].format(suffix=suffix)
    if os.path.exists(f"{root}/{var_d}"):
        if os.path.exists(f"{root}/{can_d}"):
            bak_d = can_d + backup_suffix
            if os.path.exists(f"{root}/{bak_d}"):
                shutil.rmtree(f"{root}/{bak_d}")
            os.rename(f"{root}/{can_d}", f"{root}/{bak_d}")
            actions.append({"action": "backup_dir", "from": can_d, "to": bak_d})
        os.rename(f"{root}/{var_d}", f"{root}/{can_d}")
        actions.append({"action": "promote_dir", "from": var_d, "to": can_d})

    eval_dir = f"{root}/map/eval"
    if clean_eval and os.path.isdir(eval_dir):
        n = sum(len(fs) for _, _, fs in os.walk(eval_dir))
        sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(eval_dir) for f in fs)
        shutil.rmtree(eval_dir)
        actions.append({"action": "delete_map_eval", "n_files": n, "bytes": sz})

    doc = {"suffix": suffix, "backup_suffix": backup_suffix, "actions": actions,
           "note": "Sorties canoniques de l'étape 1 remplacées par la variante ; anciens fichiers en secours sous '"
                   + backup_suffix + "'. map/eval régénérable par maps_main."}
    with open(f"{root}/reports/promotion{suffix}.json", "w") as f:
        json.dump(doc, f, indent=1)
    vol.commit()
    return doc


@app.local_entrypoint()
def promote_main(suffix: str = "_calib", backup_suffix: str = "_calib_r4", seq: str = SEQ, clean_eval: bool = True):
    import json

    print(json.dumps(promote.remote(suffix, backup_suffix, seq, clean_eval), indent=1))
