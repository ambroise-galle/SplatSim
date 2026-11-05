"""Accès à une scène au format canonique (contrat de données §1 du plan, version 0.1).

Racine : `<data_root>/scenes/<dataset>/<seq>/` (par défaut `data_root = /data`, le Volume Modal).

Contenu produit par l'étape 0 (`splatpipe/stage0/ingest_oxford.py`) :

    meta.yaml                    dataset, séquence, conventions, sources
    sensors.yaml                 caméras (intrinsèques + T_base_cam), lidars (T_base_lidar, modèle de faisceaux),
                                 imu (T_base_imu, bruits), time_offsets, derived (valeurs de contrôle)
    cameras/<cam>/index.csv      stamp_ns,t,rel_path  (rel_path relatif à data_root ; images non copiées)
    cameras/frames.csv           triplets synchronisés : frame_id,stamp_ns,t,<cam>_idx…,<cam>_rel_path…
    imu/imu.npz                  t (float64 s), stamp_ns (int64), acc (N×3 m/s²), gyro (N×3 rad/s), repère imu
    traj/gt.tum                  T_world_base (world = scan laser fixe)
    traj/{vilens,hba,colmap}.tum trajectoires dans leurs propres repères locaux (pas `world`)
    splits.yaml                  train / test_in / test_out (frames caméra et scans LiDAR)
    reports/<étape>.json
    lidar/, lidar_model/         écrits par l'ingestion LiDAR (autre module)

Exemple :

    from splatpipe.common.scene import Scene
    sc = Scene("oxford_spires", "2024-03-13-observatory-quarter-01")
    K = sc.camera("cam0"); T_base_cam0 = sc.T_base("cam0")
    idx = sc.load_camera_index("cam0"); img = sc.abs_path(idx["rel_path"][0])
    t, pos, quat = sc.load_traj("gt")
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field

import numpy as np

DATA_ROOT = "/data"
CONTRACT_VERSION = "0.1"


@dataclass
class Scene:
    dataset: str
    seq: str
    data_root: str = DATA_ROOT
    _cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ chemins
    @property
    def root(self) -> str:
        return f"{self.data_root}/scenes/{self.dataset}/{self.seq}"

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    @property
    def meta_path(self) -> str:
        return self.path("meta.yaml")

    @property
    def sensors_path(self) -> str:
        return self.path("sensors.yaml")

    @property
    def splits_path(self) -> str:
        return self.path("splits.yaml")

    @property
    def frames_path(self) -> str:
        return self.path("cameras", "frames.csv")

    @property
    def imu_path(self) -> str:
        return self.path("imu", "imu.npz")

    def camera_index_path(self, cam: str) -> str:
        return self.path("cameras", cam, "index.csv")

    def traj_path(self, name: str) -> str:
        """`name` ∈ {gt, vilens, hba, colmap, odom, final, …} → `traj/<name>.tum`."""
        return self.path("traj", f"{name}.tum")

    def report_path(self, stage: str) -> str:
        return self.path("reports", f"{stage}.json")

    def lidar_dir(self, lidar_id: str) -> str:
        return self.path("lidar", lidar_id)

    def abs_path(self, rel_path: str) -> str:
        """Chemin relatif à `data_root` (colonne `rel_path`) → chemin absolu."""
        return os.path.join(self.data_root, rel_path)

    def makedirs(self) -> None:
        for d in ("cameras", "imu", "traj", "reports"):
            os.makedirs(self.path(d), exist_ok=True)

    # ------------------------------------------------------------------ YAML
    @staticmethod
    def _load_yaml(path: str) -> dict:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f)

    def load_meta(self) -> dict:
        return self._load_yaml(self.meta_path)

    def load_sensors(self) -> dict:
        if "sensors" not in self._cache:
            self._cache["sensors"] = self._load_yaml(self.sensors_path)
        return self._cache["sensors"]

    def load_splits(self) -> dict:
        return self._load_yaml(self.splits_path)

    # ------------------------------------------------------------------ capteurs
    def camera_ids(self) -> list[str]:
        return sorted(self.load_sensors()["cameras"])

    def camera(self, cam: str) -> dict:
        """Intrinsèques et extrinsèques d'une caméra (dict de sensors.yaml)."""
        return self.load_sensors()["cameras"][cam]

    def K(self, cam: str) -> np.ndarray:
        c = self.camera(cam)
        return np.array([[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]])

    def dist_coeffs(self, cam: str) -> np.ndarray:
        c = self.camera(cam)
        return np.array([c["k1"], c["k2"], c["k3"], c["k4"]])

    def lidar(self, lidar_id: str) -> dict:
        return self.load_sensors()["lidars"][lidar_id]

    def T_base(self, sensor: str) -> np.ndarray:
        """`T_base_<sensor>` (4×4) pour une caméra, un lidar ou `imu`."""
        s = self.load_sensors()
        if sensor in s.get("cameras", {}):
            return np.asarray(s["cameras"][sensor]["T_base_cam"], dtype=np.float64)
        if sensor in s.get("lidars", {}):
            return np.asarray(s["lidars"][sensor]["T_base_lidar"], dtype=np.float64)
        if sensor == "imu":
            return np.asarray(s["imu"]["T_base_imu"], dtype=np.float64)
        raise KeyError(f"capteur inconnu : {sensor}")

    def time_offset(self, sensor: str) -> float:
        """Décalage à ajouter à l'horodatage du capteur pour obtenir le temps de référence (s)."""
        return float(self.load_sensors().get("time_offsets", {}).get(sensor, 0.0))

    # ------------------------------------------------------------------ données
    def load_camera_index(self, cam: str) -> dict[str, np.ndarray]:
        """→ {'stamp_ns': int64[N], 't': float64[N], 'rel_path': str[N]} trié par temps."""
        return _read_csv_columns(self.camera_index_path(cam), {"stamp_ns": np.int64, "t": np.float64})

    def load_frames(self) -> dict[str, np.ndarray]:
        """Triplets synchronisés → colonnes de `cameras/frames.csv` (frame_id, stamp_ns, t, <cam>_idx, <cam>_rel_path)."""
        types = {"frame_id": np.int64, "stamp_ns": np.int64, "t": np.float64}
        for cam in self.camera_ids():
            types[f"{cam}_idx"] = np.int64
        return _read_csv_columns(self.frames_path, types)

    def load_imu(self) -> dict[str, np.ndarray]:
        """→ {'t', 'stamp_ns', 'acc', 'gyro'} (repère imu, m/s² et rad/s)."""
        with np.load(self.imu_path) as z:
            return {k: z[k] for k in z.files}

    def load_traj(self, name: str = "gt", ns: bool = False):
        """→ `(t, pos [N,3], quat_xyzw [N,4])` ; `t` en secondes float64 ou en nanosecondes int64 si `ns=True`."""
        from splatpipe.common.geometry import read_tum, read_tum_ns

        return read_tum_ns(self.traj_path(name)) if ns else read_tum(self.traj_path(name))

    def load_report(self, stage: str) -> dict:
        with open(self.report_path(stage)) as f:
            return json.load(f)

    def write_report(self, stage: str, report: dict) -> str:
        os.makedirs(self.path("reports"), exist_ok=True)
        p = self.report_path(stage)
        with open(p, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return p


def _read_csv_columns(path: str, types: dict) -> dict[str, np.ndarray]:
    """Lit un CSV à en-tête en colonnes numpy (types donnés, `str` sinon)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    cols = list(zip(*body)) if body else [() for _ in header]
    out = {}
    for name, col in zip(header, cols):
        dtype = types.get(name)
        out[name] = np.array(col, dtype=dtype) if dtype is not None else np.array(col, dtype=str)
    return out
