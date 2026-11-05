"""Modèle de faisceaux d'un LiDAR tournant structuré, aux conventions du rendu LiDAR de gsplat.

Référence : gsplat commit 28e794c, `gsplat/cuda/_lidar.py`
(`RowOffsetStructuredSpinningLidarModelParameters`), `gsplat/cuda/include/Lidars.cuh`
et `gsplat/cuda/_torch_lidars.py`. Conventions reproduites ici (numpy, float64) :

- Repère capteur : rayon unitaire `(cos az·cos el, sin az·cos el, sin el)`.
  `az = atan2(y, x)` (0 sur +x, positif vers +y, donc sens trigonométrique vu de +z),
  `el = asin(z/‖p‖)` (positif vers +z). Origine des rayons = centre du capteur.
- Élément (row, col) : `el = row_elevations_rad[row]`,
  `az = column_azimuths_rad[col] + row_azimuth_offsets_rad[row]`, azimut ramené dans (−π, π].
- Lignes : `row_elevations_rad` strictement **décroissant** (row 0 = faisceau le plus haut).
- Colonnes : `column_azimuths_rad` triés dans le **sens de rotation** et sans faire un tour complet
  (CLOCKWISE : azimut décroissant ; COUNTER_CLOCKWISE : croissant). Colonne 0 = première tirée.
- Temps : temps relatif d'un élément = `col / (n_cols − 1)` ∈ [0, 1] ; 0 ↔ pose de début
  (`viewmats`), 1 ↔ pose de fin (`viewmats_rs`). Toutes les lignes d'une colonne sont supposées
  simultanées.
- Grilles de distance : tableaux `[n_rows, n_cols]` indexés `[row, col]` (row-major), comme
  `gt_dist` dans `examples/av_trainer.py` et l'image rendue `[H = n_rows, W = n_cols]`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CLOCKWISE = "CLOCKWISE"
COUNTER_CLOCKWISE = "COUNTER_CLOCKWISE"
TWO_PI = 2.0 * math.pi

# Codes de `return_index` pour les points qui ne sont pas rangés dans un créneau de la grille.
RET_DUPLICATE = -1  # copie exacte (même distance et intensité) d'un écho déjà rangé dans la même cellule
RET_OVERFLOW = -2  # écho distinct au-delà de `max_returns` (collision de créneau)
RET_INVALID = -3  # distance nulle, négative ou non finie


# ----------------------------------------------------------------------------- angles


def wrap_pi(a):
    """Ramène un angle dans (−π, π] (convention `element_to_image_point` de gsplat)."""
    a = np.asarray(a, dtype=np.float64)
    w = np.remainder(a + math.pi, TWO_PI) - math.pi  # [−π, π)
    return np.where(w <= -math.pi, w + TWO_PI, w)


def normalize_angle(a, start: float = 0.0):
    """Équivalent numpy de `gsplat.cuda._lidar.normalize_angle` : résultat dans [start, start + 2π)."""
    return np.remainder(np.asarray(a, dtype=np.float64) - start, TWO_PI) + start


def relative_clock_rotation(angle_ref, angle, direction: str):
    """Rotation signée de `angle_ref` vers `angle` dans le sens `direction` (non normalisée)."""
    if direction == CLOCKWISE:
        return np.asarray(angle_ref, dtype=np.float64) - angle
    if direction == COUNTER_CLOCKWISE:
        return np.asarray(angle, dtype=np.float64) - angle_ref
    raise ValueError(f"sens de rotation inconnu : {direction}")


def relative_angle(angle_ref, angle, direction: str):
    """Angle parcouru de `angle_ref` à `angle` dans le sens `direction`, dans [0, 2π)."""
    return np.remainder(relative_clock_rotation(angle_ref, angle, direction), TWO_PI)


def sensor_angles_to_rays(azimuth, elevation):
    """(az, el) → rayons unitaires (..., 3), formule de `sensor_angles_to_rays` de gsplat."""
    az = np.asarray(azimuth, dtype=np.float64)
    el = np.asarray(elevation, dtype=np.float64)
    ce = np.cos(el)
    return np.stack([np.cos(az) * ce, np.sin(az) * ce, np.sin(el)], axis=-1)


def points_to_sensor_angles(xyz):
    """Points (..., 3) → (azimut, élévation, distance), mêmes définitions que gsplat."""
    p = np.asarray(xyz, dtype=np.float64)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    rho = np.hypot(x, y)
    return np.arctan2(y, x), np.arctan2(z, rho), np.sqrt(rho * rho + z * z)


# ----------------------------------------------------------------------------- modèle


def _arr(v, dtype=np.float64):
    return None if v is None else np.asarray(v, dtype=dtype)


@dataclass
class StructuredSpinningLidarModel:
    """LiDAR tournant structuré à décalage d'azimut par ligne (gsplat `RowOffsetStructuredSpinningLidarModelParameters`).

    Champs propres à splatpipe (non utilisés par gsplat) : `max_returns`, `row_ring_ids`
    (ring du capteur pour chaque ligne), `row_time_offsets_s` (retard de tir de chaque ligne dans une
    colonne), `notes`.
    """

    name: str
    row_elevations_rad: np.ndarray
    column_azimuths_rad: np.ndarray
    row_azimuth_offsets_rad: np.ndarray
    spinning_frequency_hz: float
    spinning_direction: str
    max_returns: int = 1
    row_ring_ids: np.ndarray | None = None
    row_time_offsets_s: np.ndarray | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.row_elevations_rad = _arr(self.row_elevations_rad)
        self.column_azimuths_rad = _arr(self.column_azimuths_rad)
        self.row_azimuth_offsets_rad = _arr(self.row_azimuth_offsets_rad)
        self.row_ring_ids = _arr(self.row_ring_ids, np.int64)
        self.row_time_offsets_s = _arr(self.row_time_offsets_s)
        self.spinning_frequency_hz = float(self.spinning_frequency_hz)
        self.max_returns = int(self.max_returns)
        self.validate()
        # Angles des colonnes relatifs à la colonne 0, dans le sens de rotation : [0, 2π), croissants.
        self._col_rel = relative_angle(self.column_azimuths_rad[0], self.column_azimuths_rad, self.spinning_direction)
        step = np.diff(self._col_rel)
        self._uniform_step = float(np.median(step)) if len(step) and np.ptp(step) < 1e-9 else None

    # -- propriétés
    @property
    def n_rows(self) -> int:
        return int(self.row_elevations_rad.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.column_azimuths_rad.shape[0])

    @property
    def ring_to_row(self) -> np.ndarray:
        """Table ring → row (−1 si le ring n'existe pas)."""
        if self.row_ring_ids is None:
            return np.arange(self.n_rows)
        lut = np.full(int(self.row_ring_ids.max()) + 1, -1, dtype=np.int64)
        lut[self.row_ring_ids] = np.arange(self.n_rows)
        return lut

    def validate(self) -> None:
        """Mêmes assertions que le constructeur gsplat (+ cohérence des champs splatpipe)."""
        assert self.spinning_direction in (CLOCKWISE, COUNTER_CLOCKWISE), self.spinning_direction
        el, col, off = self.row_elevations_rad, self.column_azimuths_rad, self.row_azimuth_offsets_rad
        assert el.ndim == 1 and col.ndim == 1 and off.shape == el.shape
        rel_el = relative_angle(el[0], el, CLOCKWISE)
        assert np.all(np.diff(rel_el) > 0), "élévations des lignes : ordre décroissant exigé (row 0 = haut)"
        assert np.all(np.abs(el - el[0]) < TWO_PI)
        rel_col = relative_angle(col[0], col, self.spinning_direction)
        assert np.all(np.diff(rel_col) > 0), "azimuts des colonnes : tri dans le sens de rotation exigé"
        assert np.all(np.abs(col - col[0]) < TWO_PI), "azimuts des colonnes : pas de tour complet"
        if self.row_ring_ids is not None:
            assert self.row_ring_ids.shape == el.shape and len(np.unique(self.row_ring_ids)) == len(el)
        if self.row_time_offsets_s is not None:
            assert self.row_time_offsets_s.shape == el.shape
        assert self.max_returns >= 1

    # -- sérialisation
    def to_dict(self) -> dict:
        def lst(a):
            return None if a is None else [float(v) if a.dtype.kind == "f" else int(v) for v in a]

        return {
            "type": "RowOffsetStructuredSpinningLidarModelParameters",
            "name": self.name,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "spinning_frequency_hz": self.spinning_frequency_hz,
            "spinning_direction": self.spinning_direction,
            "max_returns": self.max_returns,
            "row_elevations_rad": lst(self.row_elevations_rad),
            "column_azimuths_rad": lst(self.column_azimuths_rad),
            "row_azimuth_offsets_rad": lst(self.row_azimuth_offsets_rad),
            "row_ring_ids": lst(self.row_ring_ids),
            "row_time_offsets_s": lst(self.row_time_offsets_s),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StructuredSpinningLidarModel":
        m = cls(
            name=d["name"],
            row_elevations_rad=d["row_elevations_rad"],
            column_azimuths_rad=d["column_azimuths_rad"],
            row_azimuth_offsets_rad=d["row_azimuth_offsets_rad"],
            spinning_frequency_hz=d["spinning_frequency_hz"],
            spinning_direction=d["spinning_direction"],
            max_returns=d.get("max_returns", 1),
            row_ring_ids=d.get("row_ring_ids"),
            row_time_offsets_s=d.get("row_time_offsets_s"),
            notes=d.get("notes", {}),
        )
        assert m.n_rows == d.get("n_rows", m.n_rows) and m.n_cols == d.get("n_cols", m.n_cols)
        return m

    def save_json(self, path: str | Path) -> str:
        text = json.dumps(self.to_dict(), indent=1)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
        return hashlib.sha1(text.encode()).hexdigest()

    @classmethod
    def load_json(cls, path: str | Path) -> "StructuredSpinningLidarModel":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def gsplat_kwargs(self) -> dict:
        """Arguments du constructeur gsplat (tableaux float32 à convertir en `torch.Tensor` sur le device).

        Exemple :
            kw = model.gsplat_kwargs()
            params = RowOffsetStructuredSpinningLidarModelParameters(
                row_elevations_rad=torch.from_numpy(kw["row_elevations_rad"]).cuda(), ...,
                spinning_frequency_hz=kw["spinning_frequency_hz"],
                spinning_direction=SpinningDirection[kw["spinning_direction"]])
        """
        return {
            "row_elevations_rad": self.row_elevations_rad.astype(np.float32),
            "column_azimuths_rad": self.column_azimuths_rad.astype(np.float32),
            "row_azimuth_offsets_rad": self.row_azimuth_offsets_rad.astype(np.float32),
            "spinning_frequency_hz": self.spinning_frequency_hz,
            "spinning_direction": self.spinning_direction,
        }

    # -- éléments → angles, rayons, points, temps
    def element_angles(self, row, col):
        """(row, col) → (azimut dans (−π, π], élévation), comme `element_to_image_point` de gsplat."""
        row = np.asarray(row, dtype=np.int64)
        col = np.asarray(col, dtype=np.int64)
        az = wrap_pi(self.column_azimuths_rad[col] + self.row_azimuth_offsets_rad[row])
        return az, self.row_elevations_rad[row]

    def element_rays(self, row, col):
        """(row, col) → rayons unitaires (..., 3) dans le repère capteur."""
        return sensor_angles_to_rays(*self.element_angles(row, col))

    def elements_to_points(self, row, col, rng):
        """(row, col, distance le long du rayon) → points xyz (..., 3), origine au centre du capteur."""
        return self.element_rays(row, col) * np.asarray(rng, dtype=np.float64)[..., None]

    def column_relative_time(self, col):
        """Temps relatif gsplat d'une colonne : col / (n_cols − 1) ∈ [0, 1]."""
        return np.asarray(col, dtype=np.float64) / max(self.n_cols - 1, 1)

    def element_times(self, col, t_start: float, t_end: float):
        """Temps absolu d'une colonne selon gsplat : interpolation linéaire entre t_start (col 0) et t_end (col n−1)."""
        return t_start + self.column_relative_time(col) * (t_end - t_start)

    def full_grid_rays(self):
        """Rayons unitaires de toute la grille, forme (n_rows, n_cols, 3)."""
        rr, cc = np.meshgrid(np.arange(self.n_rows), np.arange(self.n_cols), indexing="ij")
        return self.element_rays(rr, cc)

    # -- points → éléments
    def nearest_row_by_elevation(self, elevation):
        """Ligne dont l'élévation du modèle est la plus proche (élévations décroissantes)."""
        el = np.asarray(elevation, dtype=np.float64)
        asc = self.row_elevations_rad[::-1]  # croissant
        k = np.clip(np.searchsorted(asc, el), 1, len(asc) - 1)
        left_closer = (el - asc[k - 1]) <= (asc[k] - el)
        idx_asc = np.where(left_closer, k - 1, k)
        return (self.n_rows - 1) - idx_asc

    def nearest_column(self, row, azimuth):
        """Colonne dont l'azimut d'élément (col_az + offset[row]) est le plus proche de `azimuth`, pour la ligne donnée."""
        row = np.asarray(row, dtype=np.int64)
        base = self.column_azimuths_rad[0] + self.row_azimuth_offsets_rad[row]
        rel = relative_angle(base, azimuth, self.spinning_direction)  # [0, 2π)
        n = self.n_cols
        if self._uniform_step is not None:
            col = np.rint(rel / self._uniform_step).astype(np.int64)
            return np.where(col >= n, np.where(rel - self._col_rel[-1] < TWO_PI - rel, n - 1, 0), col)
        k = np.searchsorted(self._col_rel, rel)
        left = np.clip(k - 1, 0, n - 1)
        d_left = rel - self._col_rel[left]
        right_rel = np.where(k < n, self._col_rel[np.clip(k, 0, n - 1)], TWO_PI)
        right = np.where(k < n, k, 0)
        return np.where(d_left <= right_rel - rel, left, right).astype(np.int64)

    def points_to_elements(self, xyz, ring=None) -> dict:
        """Place des points (repère capteur) dans la grille.

        La ligne vient du champ `ring` s'il est fourni (via `row_ring_ids`), sinon de l'élévation la plus proche.
        Retourne `row`, `col`, `range`, `azimuth`, `elevation`, `az_residual` et `el_residual`
        (mesure − modèle, en radians ; azimut ramené dans (−π, π]).
        """
        az, el, rng = points_to_sensor_angles(xyz)
        if ring is None:
            row = self.nearest_row_by_elevation(el)
        else:
            row = self.ring_to_row[np.asarray(ring, dtype=np.int64)]
            if np.any(row < 0):
                raise ValueError("ring absent du modèle")
        col = self.nearest_column(row, az)
        az_m, el_m = self.element_angles(row, col)
        return {
            "row": row,
            "col": col,
            "range": rng,
            "azimuth": az,
            "elevation": el,
            "az_residual": wrap_pi(az - az_m),
            "el_residual": el - el_m,
        }


# ----------------------------------------------------------------------------- retours multiples


def assign_return_slots(row, col, rng, intensity, n_cols: int, max_returns: int, dedup_exact: bool = True):
    """Rang de retour de chaque point dans sa cellule (row, col).

    Convention : 0 = écho le plus proche, 1 = suivant, ... (tri par distance croissante).
    Les copies exactes (même distance et même intensité) d'un écho déjà rangé reçoivent `RET_DUPLICATE`,
    les échos au-delà de `max_returns` `RET_OVERFLOW`, les distances invalides `RET_INVALID`.
    """
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    rng = np.asarray(rng, dtype=np.float64)
    inten = np.asarray(intensity, dtype=np.float64)
    n = len(rng)
    ret = np.full(n, RET_INVALID, dtype=np.int8)
    valid = np.isfinite(rng) & (rng > 0)
    idx = np.nonzero(valid)[0]
    if len(idx) == 0:
        return ret
    key = row[idx] * n_cols + col[idx]
    order = idx[np.lexsort((inten[idx], rng[idx], key))]
    k = row[order] * n_cols + col[order]
    r = rng[order]
    i = inten[order]
    new_cell = np.r_[True, k[1:] != k[:-1]]
    if dedup_exact:
        dup = ~new_cell & np.r_[False, (r[1:] == r[:-1]) & (i[1:] == i[:-1])]
    else:
        dup = np.zeros(len(order), dtype=bool)
    # rang parmi les échos distincts de la cellule
    distinct = ~dup
    cum = np.cumsum(distinct)
    cell_start = np.maximum.accumulate(np.where(new_cell, np.arange(len(order)), 0))
    rank = cum - cum[cell_start]  # 0 pour le premier écho distinct de la cellule
    out = np.where(dup, RET_DUPLICATE, np.where(rank < max_returns, rank, RET_OVERFLOW))
    ret[order] = out.astype(np.int8)
    return ret


def fill_grids(row, col, ret, values: dict, n_rows: int, n_cols: int, max_returns: int, fill: dict | None = None) -> dict:
    """Remplit des grilles `[max_returns, n_rows, n_cols]` à partir des points rangés (ret ≥ 0)."""
    fill = fill or {}
    m = np.asarray(ret) >= 0
    r, c, s = np.asarray(row)[m], np.asarray(col)[m], np.asarray(ret)[m].astype(np.int64)
    out = {}
    for name, (v, dtype) in values.items():
        g = np.full((max_returns, n_rows, n_cols), fill.get(name, 0), dtype=dtype)
        g[s, r, c] = np.asarray(v)[m]
        out[name] = g
    return out
