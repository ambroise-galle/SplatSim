"""Prépare une scène de simulation à partir de tuiles GaussGym (desktop), dans le repère MONDE (z haut) :
monde = FLIP · mission (180° autour de x ; le repère des poses GaussGym a z vers le BAS, cf. gaussgym_export.FLIP).

- `scene.ply`   : splat assemblé (Voronoï), recentré sur `centre`, passé en y haut pour l'affichage (SH tournées), sans
                  les gaussiennes à moins de `--ghost-radius` des caméras (le robot / boîtier capteur reconstruit en
                  « fantôme » le long du chemin) ;
- `terrain.npz` : champ de hauteur 10 cm (repère monde recentré) : carte LiDAR DLIO si `--lidar-dir` (sol + troncs en
                  obstacles, cf. lidar_terrain.py), sinon maillages NKSR (sol seul, couloir de ~6 m) ;
- `scene.json`  : centre (monde), départs (position au sol + cap) le long du chemin, `ply_to_physics` (matrice 3×3 :
                  coordonnées du PLY → physique recentrée) et `splat_view_rotation_deg` (rotation à appliquer à
                  l'entité splat dans la vue).

Affichage = (x, z, −y) de la physique (y haut).

  PYTHONPATH=. venv/bin/python simrobot/sim/prepare_scene.py --tiles data_local/simrobot/grandtour/HOB-1/* \
      --lidar-dir data_local/simrobot/grandtour/HOB-1/lidar --out out/simrobot/scenes/HOB-1_5

  # mission entière, tuiles téléchargées à la volée et supprimées après traitement (disque limité) :
  PYTHONPATH=. venv/bin/python simrobot/sim/prepare_scene.py --hf-mission 2024-11-14-11-17-02 \
      --tiles-dir data_local/simrobot/grandtour/HOB-1 --lidar-dir data_local/simrobot/grandtour/HOB-1/lidar \
      --out out/simrobot/scenes/HOB-1_full --per-tile --sh-degree 1 --delete-downloaded
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.dirname(__file__))
from gaussgym_export import FLIP, load_tile_world, rotate_all, tile_cameras_world, write_ply  # noqa: E402
from terrain import heightfield, height_at, read_mesh_ply  # noqa: E402

R_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)     # (x, y, z) → (x, z, −y)
HF = "https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset"


def hf_tiles(mission):
    import urllib.request

    t = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/leggedrobotics/grand_tour_dataset/tree/"
                                         f"main/{mission}/splats"))
    return sorted(x["path"].split("/")[-1] for x in t if x["type"] == "directory")


def hf_get(mission, tile_dir, rel):
    """Télécharge `splats/<tuile>/<rel>` s'il manque (fichier .part puis renommage : pas de fichier tronqué)."""
    import urllib.request

    dst = f"{tile_dir}/{rel}"
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    urllib.request.urlretrieve(f"{HF}/resolve/main/{mission}/splats/{os.path.basename(tile_dir)}/{rel}", dst + ".part")
    os.replace(dst + ".part", dst)
    return dst


def time_ordered_cameras(tiles):
    """Positions (repère monde) de la caméra avant, dans l'ordre du temps, toutes tuiles confondues."""
    fr = []
    for t in tiles:
        for f in json.load(open(f"{t}/transforms.json"))["frames"]:
            if "hdr_front" in f["file_path"]:
                s, ns = str(f["timestamp"]).split("_")
                fr.append((int(s) + int(ns) * 1e-9, np.array(f["transform_matrix"], float)[:3, 3]))
    fr.sort(key=lambda x: x[0])
    return np.array([p for _, p in fr]) @ FLIP.T


def path_starts(P, terrain, every=8.0, look=2.0):
    """Départs tous les `every` m le long des positions caméra P (ordre du temps), cap = direction réellement
    parcourue sur les `look` m suivants (l'axe de la caméra ne convient pas : le robot tourne parfois sur place)."""
    starts, last = [], None
    for i, p in enumerate(P):
        if last is not None and np.linalg.norm(p[:2] - last) <= every:
            continue
        d = np.linalg.norm(P[i:, :2] - p[:2], axis=1)
        j = np.flatnonzero(d >= look)
        if len(j) == 0:
            break
        v = P[i + j[0], :2] - p[:2]
        z = float(height_at(terrain, p[None, :2])[0])
        starts.append({"xyz": [round(float(p[0]), 3), round(float(p[1]), 3), round(z, 3)],
                       "yaw": round(float(np.arctan2(v[1], v[0])), 4)})
        last = p[:2]
    return starts


def safe_starts(starts, terrain, radius=0.7, max_range=0.35):
    """Écarte les départs où le sol varie de plus de `max_range` m dans un disque de `radius` m (empreinte du robot) :
    passerelle étroite, talus. Le robot y apparaît à cheval sur le vide (HÖB-1 complète, départ 6 : plateau de 1,2 m
    de large, 1,6 m au-dessus du sol alentour)."""
    out = []
    for st in starts:
        x, y = st["xyz"][:2]
        a, r = np.meshgrid(np.linspace(0, 2 * np.pi, 16, endpoint=False), np.linspace(0, radius, 5))
        pts = np.stack([x + r.ravel() * np.cos(a.ravel()), y + r.ravel() * np.sin(a.ravel())], 1)
        h = height_at(terrain, pts)
        if h.max() - h.min() <= max_range:
            out.append(st)
    return out


def main():
    from scipy.spatial import cKDTree

    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", default=None, help="dossiers de tuiles déjà téléchargées")
    ap.add_argument("--hf-mission", default=None, help="mission GrandTour : liste et télécharge les tuiles")
    ap.add_argument("--tiles-dir", default=None, help="avec --hf-mission : dossier local des tuiles")
    ap.add_argument("--per-tile", action="store_true",
                    help="un PLY par tuile (tiles_ply/) au lieu d'un scene.ply : mémoire bornée, entrée directe de "
                         "splat-transform")
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--delete-downloaded", action="store_true", help="supprimer chaque splat.ply une fois traité")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lidar-dir", default=None, help="carte DLIO + odométrie : terrain LiDAR (recommandé)")
    ap.add_argument("--ghost-radius", type=float, default=0.45)
    ap.add_argument("--min-opacity", type=float, default=0.05)
    ap.add_argument("--pad", type=float, default=6.0, help="marge du terrain autour des caméras (m)")
    ap.add_argument("--no-splat", action="store_true",
                    help="ne pas réécrire scene.ply (terrain, départs et métadonnées seulement)")
    ap.add_argument("--splat-check", default=None,
                    help="PLY du splat (repère du PLY de la scène, ex. merged.ply) : un obstacle doit y être visible "
                         "(sinon : traînées de personnes, murs invisibles)")
    ap.add_argument("--obstacles", default="troncs", choices=["troncs", "cellules"],
                    help="troncs : cercles par tranche + continuité verticale (défaut) ; cellules : ancienne méthode")
    ap.add_argument("--legacy-mission-ply", action="store_true",
                    help="scene.ply existant exporté depuis le repère mission (z bas) avant le correctif du 09/10")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.hf_mission:
        tiles = [f"{a.tiles_dir}/{t}" for t in hf_tiles(a.hf_mission)]
        for t in tiles:                                        # petits fichiers d'abord : Voronoï et départs
            hf_get(a.hf_mission, t, "transforms.json")
            hf_get(a.hf_mission, t, "splatfacto/dataparser_transforms.json")
        print(f"{len(tiles)} tuiles", flush=True)
    else:
        tiles = sorted(t.rstrip("/") for t in a.tiles if os.path.exists(f"{t}/splatfacto/splat.ply"))
    cams = [tile_cameras_world(t) for t in tiles]
    allc = np.concatenate(cams)
    owner = np.concatenate([np.full(len(c), i) for i, c in enumerate(cams)])
    tree = cKDTree(allc)
    centre = allc.mean(0)

    # ---------------- terrain (avant le splat : il sert aussi à placer les départs)
    res = 0.1
    lo = allc[:, :2].min(0) - a.pad - centre[:2]
    n = np.ceil((allc[:, :2].max(0) + a.pad - centre[:2] - lo) / res).astype(int)
    reg = None
    if a.lidar_dir:
        from lidar_terrain import ground_and_obstacles, lidar_points_world, register

        R, t, reg = register(tiles, a.lidar_dir)
        print(json.dumps({"recalage": reg}), flush=True)
        P = lidar_points_world(a.lidar_dir, R, t) - centre
        splat = None
        if a.splat_check:
            from stems import load_splat_points

            splat = load_splat_points(a.splat_check, FLIP @ R_YUP.T if a.legacy_mission_ply else R_YUP.T)
        terrain = ground_and_obstacles(P, lo, res, n[0], n[1], free_xy=time_ordered_cameras(tiles) - centre,
                                       method=a.obstacles, splat=splat)
        source = "lidar_dlio"
    else:
        meshes = [(V @ FLIP.T, F) for V, F in (read_mesh_ply(f"{t}/mesh.ply") for t in tiles)]
        terrain = heightfield(meshes, res=res)
        terrain["lo"] = terrain["lo"] - centre[:2]
        terrain["H"] = terrain["H"] - np.float32(centre[2])
        source = "nksr"
    np.savez_compressed(f"{a.out}/terrain.npz", **terrain)

    # ---------------- splat
    n_ghost, n_gauss, splat_files = None, None, None
    if not a.no_splat:
        parts, n_ghost, n_gauss, splat_files = [], 0, 0, []
        if a.per_tile:
            os.makedirs(f"{a.out}/tiles_ply", exist_ok=True)
        for i, t in enumerate(tiles):
            if a.hf_mission:
                hf_get(a.hf_mission, t, "splatfacto/splat.ply")
            G = load_tile_world(t)
            d, j = tree.query(G["means"])
            ghost = d < a.ghost_radius
            keep = (owner[j] == i) & (G["opac"] >= a.min_opacity) & ~ghost
            n_ghost += int(((owner[j] == i) & ghost).sum())
            Gk = {k: v[keep] for k, v in G.items()}
            del G
            n_gauss += int(keep.sum())
            if a.per_tile:
                rotate_all(Gk, R_YUP, centre)
                name = f"tiles_ply/{os.path.basename(t)}.ply"
                write_ply(f"{a.out}/{name}", Gk, sh_degree=a.sh_degree, normals=False)
                splat_files.append(name)
            else:
                parts.append(Gk)
            if a.delete_downloaded:
                os.remove(f"{t}/splatfacto/splat.ply")
            print(f"[{i + 1}/{len(tiles)}] {os.path.basename(t)} : {int(keep.sum())} gardées / {len(keep)}", flush=True)
        if not a.per_tile:
            G = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            rotate_all(G, R_YUP, centre)
            write_ply(f"{a.out}/scene.ply", G, sh_degree=a.sh_degree)
            splat_files = ["scene.ply"]

    # PLY → physique recentrée, et rotation de l'entité splat dans la vue (le moteur affiche le PLY tel quel)
    if a.legacy_mission_ply:
        # ancien export : PLY = R_YUP (p_mission − c_mission) ; physique = FLIP (p_mission − c_mission)
        ply_to_phys = FLIP @ R_YUP.T
        view_rot = [180, 0, 0]                                 # R_YUP · FLIP · R_YUPᵀ = 180° autour de x
    else:
        ply_to_phys = R_YUP.T
        view_rot = [0, 0, 0]

    ground = dict(terrain, H=terrain.get("sol", terrain["H"]))            # départs posés sur le sol, pas sur les piliers
    starts = safe_starts(path_starts(time_ordered_cameras(tiles) - centre, ground), terrain)
    old = json.load(open(f"{a.out}/scene.json")) if os.path.exists(f"{a.out}/scene.json") else {}
    meta = {
        "tuiles": [os.path.basename(t) for t in tiles],
        "centre_monde_m": centre.round(4).tolist(),
        "n_gaussiennes": n_gauss if n_gauss is not None else old.get("n_gaussiennes"),
        "fantomes_retires": n_ghost if n_ghost is not None else old.get("fantomes_retires"),
        "fichiers_splat": splat_files if splat_files is not None else old.get("fichiers_splat"),
        "sh_degre": a.sh_degree if splat_files is not None else old.get("sh_degre"),
        "departs": starts,
        "terrain": {"source": source, "forme": list(terrain["H"].shape), "res_m": res,
                    "couvert_frac": round(float(terrain["couvert"].mean()), 3),
                    "obstacles_frac": round(float(terrain.get("obstacle", np.zeros(1)).mean()), 4)},
        "recalage_lidar": reg,
        "ply_to_physics": ply_to_phys.tolist(),
        "splat_view_rotation_deg": view_rot,
        "reperes": "monde = FLIP · mission GaussGym (z haut) ; physique = monde recentré ; affichage = (x, z, -y)",
    }
    json.dump(meta, open(f"{a.out}/scene.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("departs", "recalage_lidar")} | {"n_departs": len(starts)}))


if __name__ == "__main__":
    main()
