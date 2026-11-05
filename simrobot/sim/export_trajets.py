"""Trajectoires et poses du jeu de données → `trajets.json` de la scène (repère physique = monde recentré, z haut),
pour l'affichage dans la vue (touche J).

- caméras GaussGym (`transforms.json` de chaque tuile, dédoublonnées par caméra et horodatage) : position, axe de
  visée, axe « haut » de l'image, champ (fl_x, w, h). `transform_matrix` = caméra → mission, convention OpenGL
  (x droite, y haut, visée −z) ; monde = FLIP · mission ;
- trajectoire DLIO de l'ANYmal (base), recalée sur les caméras (`lidar_terrain.register`).

  PYTHONPATH=. venv/bin/python simrobot/sim/export_trajets.py --tiles-dir data_local/simrobot/grandtour/HOB-1 \
      --lidar-dir data_local/simrobot/grandtour/HOB-1/lidar --scene out/simrobot/scenes/HOB-1_full
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.dirname(__file__))
from gaussgym_export import FLIP  # noqa: E402


def cameras(tiles, centre):
    seen, out = set(), {}
    for t in tiles:
        for f in json.load(open(f"{t}/transforms.json"))["frames"]:
            cam = os.path.basename(f["file_path"]).rsplit("_", 1)[0]          # hdr_front, hdr_left, hdr_right
            key = (cam, f["timestamp"])
            if key in seen:
                continue
            seen.add(key)
            s, ns = str(f["timestamp"]).split("_")
            T = np.array(f["transform_matrix"], float)
            c = out.setdefault(cam, {"t": [], "p": [], "f": [], "u": [],
                                     "fov_y_deg": float(np.degrees(2 * np.arctan(float(f["h"]) / 2 / float(f["fl_y"])))),
                                     "aspect": float(f["w"]) / float(f["h"])})
            c["t"].append(int(s) + int(ns) * 1e-9)
            c["p"].append(FLIP @ T[:3, 3] - centre)
            c["f"].append(FLIP @ -T[:3, 2])
            c["u"].append(FLIP @ T[:3, 1])
    for c in out.values():
        o = np.argsort(c["t"])
        c["t"] = np.round(np.asarray(c["t"])[o] - min(c["t"]), 3).tolist()
        for k, nd in (("p", 3), ("f", 4), ("u", 4)):
            c[k] = np.round(np.asarray(c[k])[o], nd).tolist()
    return out


def dlio(tiles, lidar_dir, centre):
    from lidar_terrain import register, zarr_group

    R, t, reg = register(tiles, lidar_dir)
    odo = zarr_group(glob.glob(f"{lidar_dir}/dlio_map_odometry.tar")[0], "dlio_map_odometry")
    ts, p = odo("timestamp").astype(float), odo("pose_pos").astype(float)
    pw = (p @ R.T + t) @ FLIP.T - centre
    return {"t": np.round(ts - ts[0], 3).tolist(), "p": np.round(pw, 3).tolist(), "recalage": reg["residu_m"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", required=True)
    ap.add_argument("--lidar-dir", required=True)
    ap.add_argument("--scene", required=True)
    a = ap.parse_args()
    meta = json.load(open(f"{a.scene}/scene.json"))
    tiles = [f"{a.tiles_dir}/{t}" for t in meta["tuiles"]]
    centre = np.array(meta["centre_monde_m"])
    out = {"cameras": cameras(tiles, centre), "dlio": dlio(tiles, a.lidar_dir, centre),
           "reperes": "physique = FLIP · mission − centre_monde_m (z haut) ; f = axe de visée, u = haut de l'image"}
    json.dump(out, open(f"{a.scene}/trajets.json", "w"), separators=(",", ":"))
    print(json.dumps({"cameras": {k: len(v["t"]) for k, v in out["cameras"].items()}, "dlio": len(out["dlio"]["t"]),
                      "recalage_m": out["dlio"]["recalage"],
                      "taille_Mo": round(os.path.getsize(f"{a.scene}/trajets.json") / 1e6, 2)}))


if __name__ == "__main__":
    main()
