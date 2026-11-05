"""Géométrie SE(3) minimale en numpy (sans dépendance obligatoire à scipy).

Conventions (voir `etapes/notes/conventions.md`) :
- `T_a_b` est une matrice 4×4 qui envoie un point du repère `b` dans le repère `a` : `p_a = T_a_b · p_b`.
- Quaternions dans l'ordre **xyzw** (Hamilton, comme TUM et Oxford Spires), normalisés, `w >= 0` en sortie.
- Trajectoire TUM : lignes `t tx ty tz qx qy qz qw` = `T_world_base(t)`.

Toutes les fonctions acceptent des tableaux « batch » : `(..., 3)` pour les translations,
`(..., 4)` pour les quaternions, `(..., 3, 3)` / `(..., 4, 4)` pour les matrices.

Auto-test : `.venv/bin/python -m splatpipe.common.geometry`
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Quaternions et rotations
# ---------------------------------------------------------------------------


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalise un quaternion xyzw et impose `w >= 0` (représentant canonique)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., 3:4] < 0, -q, q)


def quat_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    """Quaternion xyzw → matrice de rotation 3×3 (normalise d'abord)."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Matrice de rotation 3×3 → quaternion xyzw (méthode de Shepperd, `w >= 0`)."""
    R = np.asarray(R, dtype=np.float64)
    batch = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    # Choix du pivot le plus grand parmi (w, x, y, z) pour la stabilité numérique.
    k = np.argmax(np.stack([tr, m00, m11, m22], axis=1), axis=1)
    r = lambda i, j: R[:, i, j]  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        s0 = np.sqrt(np.maximum(1.0 + tr, 0.0)) * 2.0
        s1 = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 0.0)) * 2.0
        s2 = np.sqrt(np.maximum(1.0 - m00 + m11 - m22, 0.0)) * 2.0
        s3 = np.sqrt(np.maximum(1.0 - m00 - m11 + m22, 0.0)) * 2.0
        cands = np.stack(
            [
                np.stack([(r(2, 1) - r(1, 2)) / s0, (r(0, 2) - r(2, 0)) / s0, (r(1, 0) - r(0, 1)) / s0, 0.25 * s0], 1),
                np.stack([0.25 * s1, (r(1, 0) + r(0, 1)) / s1, (r(0, 2) + r(2, 0)) / s1, (r(2, 1) - r(1, 2)) / s1], 1),
                np.stack([(r(1, 0) + r(0, 1)) / s2, 0.25 * s2, (r(2, 1) + r(1, 2)) / s2, (r(0, 2) - r(2, 0)) / s2], 1),
                np.stack([(r(0, 2) + r(2, 0)) / s3, (r(2, 1) + r(1, 2)) / s3, 0.25 * s3, (r(1, 0) - r(0, 1)) / s3], 1),
            ],
            axis=1,
        )
    q = cands[np.arange(R.shape[0]), k]
    return quat_normalize(q).reshape(batch + (4,))


def rotation_angle(R: np.ndarray) -> np.ndarray:
    """Angle (rad, dans [0, π]) de la rotation R."""
    R = np.asarray(R, dtype=np.float64)
    c = (np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(c, -1.0, 1.0))


def rot_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Logarithme de SO(3) : matrice de rotation → vecteur rotation (axe · angle, rad), batch possible."""
    q = rot_to_quat(R)
    s = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., 3:4])
    # Petit angle : angle/s → 2 (développement limité), évite la division par zéro.
    scale = np.where(s > 1e-12, angle / np.where(s > 1e-12, s, 1.0), 2.0)
    return q[..., :3] * scale


def rotation_angle_between(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """Angle (rad) de la rotation relative `R_aᵀ · R_b` (distance géodésique sur SO(3))."""
    return rotation_angle(np.swapaxes(R_a, -1, -2) @ R_b)


def angle_between_vectors(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle (rad) entre deux vecteurs 3D (batch)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    c = np.sum(u * v, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.arccos(np.clip(c, -1.0, 1.0))


def is_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
    """Vrai si R est orthonormale (RᵀR = I) et de déterminant +1, à `tol` près."""
    R = np.asarray(R, dtype=np.float64)
    eye = np.eye(3)
    return bool(np.all(np.abs(np.swapaxes(R, -1, -2) @ R - eye) < tol) and np.all(np.abs(np.linalg.det(R) - 1) < tol))


# ---------------------------------------------------------------------------
# SE(3)
# ---------------------------------------------------------------------------


def se3(t: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """(translation, quaternion xyzw) → matrice homogène 4×4 (batch possible)."""
    t = np.asarray(t, dtype=np.float64)
    R = quat_to_rot(q_xyzw)
    T = np.zeros(R.shape[:-2] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def se3_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(R 3×3, t 3) → matrice homogène 4×4 (batch possible)."""
    R = np.asarray(R, dtype=np.float64)
    T = np.zeros(R.shape[:-2] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def se3_to_tq(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Matrice 4×4 → (translation, quaternion xyzw)."""
    T = np.asarray(T, dtype=np.float64)
    return T[..., :3, 3].copy(), rot_to_quat(T[..., :3, :3])


def inv(T: np.ndarray) -> np.ndarray:
    """Inverse exacte d'une transformation rigide 4×4 : `inv(T_a_b) = T_b_a`."""
    T = np.asarray(T, dtype=np.float64)
    Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -(Rt @ T[..., :3, 3:4])[..., 0]
    out[..., 3, 3] = 1.0
    return out


def compose(*Ts: np.ndarray) -> np.ndarray:
    """Produit matriciel de gauche à droite : `compose(T_a_b, T_b_c) = T_a_c`."""
    out = np.asarray(Ts[0], dtype=np.float64)
    for T in Ts[1:]:
        out = out @ np.asarray(T, dtype=np.float64)
    return out


def transform_points(T_a_b: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """Applique `T_a_b` à des points (N, 3) exprimés dans `b`."""
    T = np.asarray(T_a_b, dtype=np.float64)
    p = np.asarray(p_b, dtype=np.float64)
    return p @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------------------
# Trajectoires TUM
# ---------------------------------------------------------------------------


def parse_stamp_ns(s: str) -> int:
    """Texte `"<secondes>.<fraction>"` → entier de nanosecondes, sans passer par un float (exact).

    Exemples : `"1710338243.457672000"` → 1710338243457672000 ; `"12.5"` → 12500000000.
    """
    s = s.strip()
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    sec, _, frac = s.partition(".")
    if len(frac) > 9 and set(frac[9:]) - {"0"}:
        raise ValueError(f"précision sous la nanoseconde non supportée : {s}")
    ns = int(sec or "0") * 1_000_000_000 + int((frac + "000000000")[:9])
    return -ns if neg else ns


def format_stamp_ns(ns: int) -> str:
    """Entier de nanosecondes → `"<secondes>.<9 chiffres>"` (inverse exact de `parse_stamp_ns`)."""
    ns = int(ns)
    sign = "-" if ns < 0 else ""
    s, r = divmod(abs(ns), 1_000_000_000)
    return f"{sign}{s}.{r:09d}"


def read_tum_ns(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lit un fichier TUM → `(stamp_ns [N] int64, pos [N,3], quat_xyzw [N,4])`, dans l'ordre du fichier.

    Les temps sont lus depuis le texte (exacts à la nanoseconde). Lignes vides et commentaires `#` ignorés.
    Les quaternions sont normalisés (w >= 0).
    """
    stamps, vals = [], []
    with open(path) as f:
        for k, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"{path}:{k + 1}: {len(parts)} colonnes, 8 attendues (t tx ty tz qx qy qz qw)")
            stamps.append(parse_stamp_ns(parts[0]))
            vals.append([float(x) for x in parts[1:]])
    if not stamps:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3)), np.zeros((0, 4))
    a = np.asarray(vals, dtype=np.float64)
    return np.asarray(stamps, dtype=np.int64), a[:, 0:3], quat_normalize(a[:, 3:7])


def read_tum(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lit un fichier TUM → `(t [N] float64 en s, pos [N,3], quat_xyzw [N,4])`, dans l'ordre du fichier."""
    ns, pos, q = read_tum_ns(path)
    return ns_to_s(ns), pos, q


def ns_to_s(ns: np.ndarray) -> np.ndarray:
    """Nanosecondes int64 → secondes float64 (précision ~0,2 µs pour des temps Unix actuels)."""
    ns = np.asarray(ns, dtype=np.int64)
    return (ns // 1_000_000_000).astype(np.float64) + (ns % 1_000_000_000).astype(np.float64) * 1e-9


def write_tum(
    path: str, t: np.ndarray, pos: np.ndarray, quat_xyzw: np.ndarray, header: str | None = None
) -> None:
    """Écrit un fichier TUM.

    `t` en secondes (float) ou en nanosecondes (tableau d'entiers : écrit exactement, 9 décimales).
    Positions avec 9 décimales, quaternions normalisés (w >= 0) avec 12 décimales.
    """
    t = np.asarray(t)
    exact = np.issubdtype(t.dtype, np.integer)
    pos = np.asarray(pos, dtype=np.float64)
    q = quat_normalize(quat_xyzw)
    with open(path, "w") as f:
        if header:
            for line in header.strip().splitlines():
                f.write(f"# {line}\n")
        for i in range(len(t)):
            ts = format_stamp_ns(t[i]) if exact else f"{float(t[i]):.9f}"
            f.write(
                f"{ts} {pos[i, 0]:.9f} {pos[i, 1]:.9f} {pos[i, 2]:.9f} "
                f"{q[i, 0]:.12f} {q[i, 1]:.12f} {q[i, 2]:.12f} {q[i, 3]:.12f}\n"
            )


def clean_trajectory(
    t: np.ndarray, pos: np.ndarray, quat: np.ndarray, dup_tol: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Trie par temps et supprime les doublons temporels (Δt <= dup_tol, on garde la 1re occurrence).

    `t` en secondes (float, `dup_tol` par défaut 1e-6 s) ou en nanosecondes (entiers, défaut 1000 ns).
    Renvoie `(t, pos, quat, info)` où `info` compte les pas désordonnés et les doublons
    (et l'écart max de position/rotation entre doublons, pour savoir s'ils étaient identiques).
    """
    t = np.asarray(t)
    if dup_tol is None:
        dup_tol = 1000 if np.issubdtype(t.dtype, np.integer) else 1e-6
    n_in = len(t)
    n_unsorted = int(np.sum(np.diff(t) < 0)) if n_in > 1 else 0
    order = np.argsort(t, kind="stable")
    t, pos, quat = t[order], pos[order], quat[order]
    keep = np.ones(n_in, dtype=bool)
    if n_in > 1:
        keep[1:] = np.diff(t) > dup_tol
    dup_idx = np.nonzero(~keep)[0]
    max_dp, max_dr = 0.0, 0.0
    if len(dup_idx):
        dp = np.linalg.norm(pos[dup_idx] - pos[dup_idx - 1], axis=1)
        dr = rotation_angle_between(quat_to_rot(quat[dup_idx]), quat_to_rot(quat[dup_idx - 1]))
        max_dp, max_dr = float(dp.max()), float(np.degrees(dr.max()))
    info = {
        "n_in": n_in,
        "n_out": int(keep.sum()),
        "n_unsorted_steps": n_unsorted,
        "n_duplicates_removed": int(len(dup_idx)),
        "duplicates_max_dpos_m": max_dp,
        "duplicates_max_drot_deg": max_dr,
    }
    return t[keep], pos[keep], quat[keep], info


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """SLERP vectorisé entre quaternions xyzw (chemin le plus court). `alpha` dans [0, 1]."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    alpha = np.asarray(alpha, dtype=np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_t = np.sin(theta)
    small = sin_t < 1e-8
    safe = np.where(small, 1.0, sin_t)
    w0 = np.where(small, 1.0 - alpha, np.sin((1.0 - alpha) * theta) / safe)
    w1 = np.where(small, alpha, np.sin(alpha * theta) / safe)
    return quat_normalize(w0 * q0 + w1 * q1)


def interpolate_poses(
    t: np.ndarray, pos: np.ndarray, quat: np.ndarray, t_query: np.ndarray, max_gap_s: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpole une trajectoire (triée) aux instants `t_query`.

    Translation linéaire, rotation SLERP. Renvoie `(pos_q, quat_q, valid)` ; `valid` est faux hors de
    `[t[0], t[-1]]` ou si l'intervalle encadrant dépasse `max_gap_s` (valeurs alors extrapolées au bord).
    """
    t = np.asarray(t, dtype=np.float64)
    tq = np.atleast_1d(np.asarray(t_query, dtype=np.float64))
    if len(t) < 2:
        raise ValueError("au moins 2 poses sont nécessaires")
    i1 = np.clip(np.searchsorted(t, tq, side="right"), 1, len(t) - 1)
    i0 = i1 - 1
    dt = t[i1] - t[i0]
    alpha = np.clip((tq - t[i0]) / np.where(dt > 0, dt, 1.0), 0.0, 1.0)
    p = pos[i0] + alpha[:, None] * (pos[i1] - pos[i0])
    q = slerp(quat[i0], quat[i1], alpha)
    valid = (tq >= t[0]) & (tq <= t[-1])
    if max_gap_s is not None:
        valid &= dt <= max_gap_s
    return p, q, valid


# ---------------------------------------------------------------------------
# Auto-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    import os
    import tempfile

    rng = np.random.default_rng(0)
    n = 200
    q = quat_normalize(rng.normal(size=(n, 4)))
    R = quat_to_rot(q)
    assert is_rotation(R), "quat_to_rot ne donne pas des rotations"
    q2 = rot_to_quat(R)
    assert np.allclose(q, q2, atol=1e-10), "aller-retour quat→rot→quat"

    # Cas de référence : 90° autour de z envoie x sur y.
    Rz90 = quat_to_rot(np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]))
    assert np.allclose(Rz90 @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    # T_base_lidar du dataset : 180° autour de z.
    T_bl = se3([0, 0, 0.124], [0, 0, 1, 0])
    assert np.allclose(transform_points(T_bl, np.array([[1.0, 2.0, 3.0]])), [[-1.0, -2.0, 3.124]])
    # Rotations proches de 180° (pivot non-w de Shepperd).
    for axis in np.eye(3):
        qa = np.append(axis, 0.0)
        assert np.allclose(np.abs(rot_to_quat(quat_to_rot(qa))), np.abs(qa), atol=1e-12)

    t = rng.normal(size=(n, 3))
    T = se3(t, q)
    assert np.allclose(inv(T) @ T, np.eye(4), atol=1e-12)
    assert np.allclose(compose(T[0], inv(T[0]), T[1]), T[1], atol=1e-12)
    t_b, q_b = se3_to_tq(T)
    assert np.allclose(t_b, t) and np.allclose(q_b, q, atol=1e-10)
    p = rng.normal(size=(10, 3))
    assert np.allclose(transform_points(compose(T[0], T[1]), p), transform_points(T[0], transform_points(T[1], p)))

    # Angles.
    ang = rotation_angle(Rz90)
    assert abs(ang - np.pi / 2) < 1e-12
    assert np.allclose(rot_to_rotvec(Rz90), [0, 0, np.pi / 2], atol=1e-12)
    assert np.allclose(rot_to_rotvec(np.eye(3)), 0.0)
    rv = rot_to_rotvec(R)
    assert np.allclose(np.linalg.norm(rv, axis=1), rotation_angle(R), atol=1e-9)
    assert abs(rotation_angle_between(np.eye(3), Rz90) - np.pi / 2) < 1e-12
    assert abs(angle_between_vectors(np.array([1.0, 0, 0]), np.array([0, 2.0, 0])) - np.pi / 2) < 1e-12

    # SLERP : moitié d'une rotation de 90° autour de z = 45°.
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = rot_to_quat(Rz90)
    qm = slerp(q0, q1, 0.5)
    assert abs(rotation_angle(quat_to_rot(qm)) - np.pi / 4) < 1e-12
    # Chemin le plus court avec des signes opposés.
    assert np.allclose(slerp(q0, -q1, 0.5), qm, atol=1e-12)

    # TUM : écriture/lecture/tri/doublons/interpolation.
    tt = np.array([0.0, 1.0, 2.0, 3.0])
    pp = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    qz = np.stack([rot_to_quat(quat_to_rot([0, 0, np.sin(a / 2), np.cos(a / 2)])) for a in [0, 0.2, 0.4, 0.6]])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.tum")
        # désordre + doublon
        write_tum(path, tt[[0, 2, 1, 3, 3]], pp[[0, 2, 1, 3, 3]], qz[[0, 2, 1, 3, 3]], header="t tx ty tz qx qy qz qw")
        t_r, p_r, q_r = read_tum(path)
    t_c, p_c, q_c, info = clean_trajectory(t_r, p_r, q_r)
    assert info["n_duplicates_removed"] == 1 and info["n_unsorted_steps"] == 1
    assert np.allclose(t_c, tt) and np.allclose(p_c, pp) and np.allclose(q_c, qz, atol=1e-9)
    p_i, q_i, v = interpolate_poses(t_c, p_c, q_c, np.array([0.0, 1.5, 3.0, 4.0]))
    assert np.allclose(p_i[:3, 0], [0, 1.5, 3]) and list(v) == [True, True, True, False]
    assert abs(rotation_angle(quat_to_rot(q_i[1])) - 0.3) < 1e-9
    _, _, v2 = interpolate_poses(t_c, p_c, q_c, np.array([1.5]), max_gap_s=0.5)
    assert not v2[0]

    # Horodatages exacts en nanosecondes.
    for txt, ns in [("1710338243.457672000", 1710338243457672000), ("12.5", 12_500_000_000), ("7", 7_000_000_000)]:
        assert parse_stamp_ns(txt) == ns
    assert format_stamp_ns(1710338098043290436) == "1710338098.043290436"
    assert parse_stamp_ns(format_stamp_ns(-1_500_000_001)) == -1_500_000_001
    ns_arr = np.array([1710338098043290436, 1710338098052826000], dtype=np.int64)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ns.tum")
        write_tum(path, ns_arr, pp[:2], qz[:2])
        ns_r, _, _ = read_tum_ns(path)
    assert np.array_equal(ns_r, ns_arr)
    _, _, _, info_ns = clean_trajectory(ns_arr[[1, 0, 0]], pp[:3], qz[:3])
    assert info_ns["n_duplicates_removed"] == 1 and info_ns["n_out"] == 2
    assert abs(ns_to_s(ns_arr)[0] - 1710338098.043290436) < 1e-6

    # Comparaison optionnelle avec scipy.
    try:
        from scipy.spatial.transform import Rotation, Slerp

        assert np.allclose(Rotation.from_quat(q).as_matrix(), R, atol=1e-12)
        key = Rotation.from_quat(q[:2])
        s = Slerp([0.0, 1.0], key)(0.3).as_quat()
        s = s if s[3] >= 0 else -s
        assert np.allclose(slerp(q[0], q[1], 0.3), s, atol=1e-9)
        scipy_msg = "comparaison scipy OK"
    except ImportError:
        scipy_msg = "scipy absent, comparaison sautée"
    print(f"splatpipe.common.geometry : auto-test OK ({scipy_msg})")


if __name__ == "__main__":
    _self_test()
