"""Accès données pour le trainer 4.1 — fenêtre glissante (Oxford Spires ; DiTer++ depuis le 08/10 : LiDAR et caméras
lus dans `sensors.yaml`).

Tout le chargement lourd reste en CPU (uint8 / numpy) ; seuls les tenseurs du pas
courant passent sur GPU (piège §8.3 : av_trainer pré-empile tout sur GPU en float32
≈ 67 Go/min de séquence — interdit ici).

Une `WindowData` indexe :
  - des balayages LiDAR deskewed (`<stamp>.npz`) + leur sidecar dyn 2.3
    (`semantics_lidar/<stamp>.npz`, repli : n_scans de la carte) ;
  - par balayage, l'image la plus proche de chaque caméra (± 60 ms) avec son masque
    ciel (bit 1 de `masks/<cam>/<stamp>.png`), sa pose (`poses/cameras.json`,
    obturateur global) et ses intrinsèques OPENCV_FISHEYE.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np

def lidar_paths(root: str) -> tuple[str, str]:
    """(id du LiDAR, dossier des balayages deskewés) : premier LiDAR de `sensors.yaml` ; `hesai_qt64` →
    `lidar/hesai_deskewed` (Oxford), `ouster_os1_64` → `lidar/ouster_deskewed` (DiTer++)."""
    import yaml

    lid = next(iter(yaml.safe_load(open(f"{root}/sensors.yaml"))["lidars"]))
    return lid, f"lidar/{lid.split('_')[0]}_deskewed"


SWEEP_FIELDS = ("xyz", "ret", "row", "col", "t_rel", "t_start", "t_end",
                "T_map_lidar_start", "T_map_lidar_end")
FRAME_TOL_NS = 60_000_000        # ± 60 ms autour du milieu du balayage


class WindowData:
    def __init__(self, root: str, stamps: list[int], holdout_every: int = 0,
                 cam_names: tuple[str, ...] | None = None):
        import yaml

        self.root = root
        self.stamps = sorted(stamps)
        sensors = yaml.safe_load(open(f"{root}/sensors.yaml"))
        self.cam_names = list(cam_names) if cam_names else list(sensors["cameras"])
        self.lidar_id, self.deskew_dir = lidar_paths(root)
        self.holdout = set()
        if holdout_every > 0:      # 1 balayage sur N en évaluation, jamais optimisé
            self.holdout = set(self.stamps[::holdout_every])
        self.train_stamps = [s for s in self.stamps if s not in self.holdout]

        self.cams_cfg = {c: sensors["cameras"][c] for c in self.cam_names}
        self.poses = json.load(open(f"{root}/poses/cameras.json"))["cameras"]
        self._frame_index = {}
        for c in self.cam_names:
            idx = []
            with open(f"{root}/cameras/{c}/index.csv") as f:
                for r in csv.DictReader(f):
                    idx.append((int(r["stamp_ns"]), r["rel_path"]))
            idx.sort()
            self._frame_index[c] = idx
        self.traj = None           # chargé paresseusement (spline)
        self.T_bl = np.asarray(sensors["lidars"][self.lidar_id]["T_base_lidar"], float)
        self._sweeps: dict[int, dict] = {}
        self._dyn_cache: dict[int, np.ndarray | None] = {}

    # ------------------------------------------------------------------ spline
    def origins(self, t_seconds: np.ndarray) -> np.ndarray:
        """Origines lidar (repère carte) aux instants demandés — via la spline."""
        from splatpipe.common.trajectory import Trajectory

        if self.traj is None:
            self.traj = Trajectory.load(f"{self.root}/traj/spline.npz")
        Tmb, _ = self.traj.matrices(list(map(float, t_seconds)))
        return (np.asarray(Tmb) @ self.T_bl[None])[:, :3, 3]

    # ------------------------------------------------------------------ sweeps
    def sweep_light(self, stamp: int) -> float:
        """t_mid du balayage, sans tout charger (pour l'indexation des fenêtres)."""
        with np.load(f"{self.root}/{self.deskew_dir}/{stamp}.npz") as z:
            return 0.5 * (float(z["t_start"]) + float(z["t_end"]))

    def sweep(self, stamp: int) -> dict:
        """Tableaux numpy du balayage (chargés une fois) : cellules premier écho,
        distance mesurée par point (origine à son propre temps), cellules libres,
        masque de perte (hors dyn et ≤ RANGE_MAX). Cache plafonné (RAM 15 Go)."""
        if stamp in self._sweeps:
            return self._sweeps[stamp]
        if len(self._sweeps) > 400:
            self._sweeps.clear()
        with np.load(f"{self.root}/{self.deskew_dir}/{stamp}.npz") as z:
            d = {k: z[k] for k in SWEEP_FIELDS}
        xyz = d["xyz"].astype(np.float32)
        first = d["ret"].astype(int) == 0
        t_pt = float(d["t_start"]) + d["t_rel"].astype(np.float64)
        rng = np.full(len(xyz), np.nan, dtype=np.float64)
        rng[first] = np.linalg.norm(xyz[first].astype(np.float64)
                                    - self.origins(t_pt[first]), axis=1)

        # dyn : sidecar 2.3 par point, repli n_scans de la carte (fait côté trainer)
        dyn_pt = None
        sc = f"{self.root}/semantics_lidar/{stamp}.npz"
        if os.path.exists(sc):
            with np.load(sc) as z:
                if "dyn" in z.files:
                    dyn_pt = z["dyn"].astype(bool)

        n_grid = int(d["row"].max()) + 1, int(d["col"].max()) + 1
        occupied = np.zeros(n_grid, bool)
        occupied[d["row"][first], d["col"][first]] = True
        rr, cc = np.meshgrid(np.arange(n_grid[0]), np.arange(n_grid[1]), indexing="ij")
        free_rows, free_cols = rr[~occupied], cc[~occupied]     # cellules sans retour

        self._sweeps[stamp] = dict(stamp=int(stamp), xyz=xyz, first=first, rng=rng,
                                   row=d["row"].astype(np.int64), col=d["col"].astype(np.int64),
                                   dyn_pt=dyn_pt, free_rows=free_rows, free_cols=free_cols,
                                   T0=np.asarray(d["T_map_lidar_start"], float),
                                   T1=np.asarray(d["T_map_lidar_end"], float),
                                   t_mid=0.5 * (float(d["t_start"]) + float(d["t_end"])))
        self._sweeps[stamp]["o_mid"] = self.origins(np.array([self._sweeps[stamp]["t_mid"]]))[0]
        return self._sweeps[stamp]

    # ------------------------------------------------------------------ frames
    def frame(self, cam: str, t_mid_ns: int) -> dict | None:
        """Image la plus proche du temps demandé (± FRAME_TOL_NS), avec pose/K/ciel."""
        idx = self._frame_index[cam]
        lo, hi = 0, len(idx)
        while lo < hi:                       # bissection sur la liste triée
            mid = (lo + hi) // 2
            if idx[mid][0] < t_mid_ns:
                lo = mid + 1
            else:
                hi = mid
        best = None
        for j in (lo - 1, lo):
            if 0 <= j < len(idx) and (best is None or abs(idx[j][0] - t_mid_ns) < abs(best[0] - t_mid_ns)):
                best = idx[j]
        if best is None or abs(best[0] - t_mid_ns) > FRAME_TOL_NS:
            return None
        st, rel = best
        entry = self.poses[cam].get(str(st))
        if entry is None:
            return None
        msk = f"{self.root}/masks/{cam}/{st}.png"
        return dict(cam=cam, stamp=st, rel=rel, T_wc=np.asarray(entry["T_map_cam_start"], float),
                    sky_path=msk if os.path.exists(msk) else None)

    def frames_for_sweep(self, stamp: int) -> list[dict]:
        sw = self.sweep(stamp)
        t_mid_ns = int(sw["t_mid"] * 1e9)
        out = []
        for c in self.cam_names:
            fr = self.frame(c, t_mid_ns)
            if fr is not None:
                out.append(fr)
        return out
