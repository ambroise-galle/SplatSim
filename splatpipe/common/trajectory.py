"""Trajectoires continues : échantillons denses interpolables et mise en forme par l'IMU.

Contrat de `traj/spline.npz` (écrit par `Trajectory.save`, relu par `Trajectory.load`) :

    t      float64 [N]    temps Unix (s), strictement croissant (≈ 400 Hz pour la trajectoire finale)
    pos    float64 [N,3]  translation de T_<frame>_<body>(t)
    quat   float64 [N,4]  rotation xyzw (w >= 0, continuité de signe non garantie)
    frame  str            repère fixe (« map » = repère carte de GLIM, « world » = scan laser fixe)
    body   str            repère corps (« base »)
    meta   str            JSON (méthode, sources, paramètres)

Requête à un instant quelconque : translation linéaire + SLERP entre les deux échantillons encadrants
(`Trajectory.interp` / `Trajectory.matrices`). Avec un pas de 2,5 ms l'erreur d'interpolation est
négligeable (< 0,001° et < 0,1 mm pour un appareil porté à la main).

Construction (`imu_shaped`) : on part de poses d'ancrage (p. ex. GLIM à 10 Hz, ou la GT) et d'une
trajectoire « de forme » à haute fréquence (orientation intégrée du gyroscope, ou trajectoire IMU de GLIM).
Entre deux ancres k et k+1 :

    Δ_k(s) = inv(F(t_k)) · F(s)                              mouvement relatif mesuré par la forme
    E_k    = inv(Δ_k(t_{k+1})) · inv(A_k) · A_{k+1}          écart résiduel (petit)
    T(s)   = A_k · Δ_k(s) · Exp(α · Log(E_k)),  α = (s − t_k)/(t_{k+1} − t_k)

La trajectoire passe exactement par les ancres, est continue, et suit la forme entre elles. Un biais
constant du gyroscope donne une erreur linéaire en temps, entièrement absorbée par la correction E_k.
Rotation et translation sont traitées séparément (translation linéaire si la forme n'en a pas).

Lissage des ancres (`smooth_anchor_corrections`) : GLIM place ses trames de façon rigide dans chaque
sous-carte (T_map_imu = T_map_submap · T_submap_odom · T_odom_imu), donc la correction
C_k = A_k · inv(O_k) (O = odométrie) est constante par sous-carte et saute aux frontières. On lisse C_k
par un noyau gaussien (σ en secondes) puis Ã_k = C̃_k · O_k.

Auto-test : `.venv/bin/python -m splatpipe.common.trajectory`
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from splatpipe.common.geometry import interpolate_poses, quat_normalize, quat_to_rot, read_tum, rot_to_quat, write_tum

# ---------------------------------------------------------------------------
# Quaternions xyzw (Hamilton), vectorisés
# ---------------------------------------------------------------------------


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Produit de Hamilton a ⊗ b (xyzw), correspond à R_a · R_b."""
    ax, ay, az, aw = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.array(q, dtype=np.float64, copy=True)
    q[..., :3] *= -1
    return q


def quat_exp(rv: np.ndarray) -> np.ndarray:
    """Vecteur rotation (rad) → quaternion xyzw."""
    rv = np.asarray(rv, dtype=np.float64)
    th = np.linalg.norm(rv, axis=-1, keepdims=True)
    half = 0.5 * th
    k = np.where(th > 1e-12, np.sin(half) / np.where(th > 1e-12, th, 1.0), 0.5 - th**2 / 48.0)
    return np.concatenate([rv * k, np.cos(half)], axis=-1)


def quat_log(q: np.ndarray) -> np.ndarray:
    """Quaternion xyzw → vecteur rotation (rad), chemin le plus court."""
    q = quat_normalize(q)
    s = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    ang = 2.0 * np.arctan2(s, q[..., 3:4])
    k = np.where(s > 1e-12, ang / np.where(s > 1e-12, s, 1.0), 2.0)
    return q[..., :3] * k


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Applique la rotation q à des vecteurs v (batch compatible)."""
    qv = np.concatenate([np.asarray(v, dtype=np.float64), np.zeros(np.shape(v)[:-1] + (1,))], axis=-1)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[..., :3]


def quat_slerp_batch(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """SLERP q0→q1 par `alpha` (batch), via Exp/Log (chemin le plus court)."""
    d = quat_mul(quat_conj(q0), q1)
    d = np.where(d[..., 3:4] < 0, -d, d)
    return quat_mul(q0, quat_exp(np.asarray(alpha)[..., None] * quat_log(d)))


# ---------------------------------------------------------------------------
# Trajectoire échantillonnée
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    t: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    frame: str = "map"
    body: str = "base"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=np.float64)
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.quat = quat_normalize(self.quat)
        if len(self.t) > 1 and not np.all(np.diff(self.t) > 0):
            raise ValueError("temps non strictement croissants")

    # --------------------------------------------------------------- E/S
    @classmethod
    def from_tum(cls, path: str, frame: str = "map", body: str = "base") -> "Trajectory":
        t, p, q = read_tum(path)
        o = np.argsort(t, kind="stable")
        t, p, q = t[o], p[o], q[o]
        keep = np.r_[True, np.diff(t) > 1e-9]
        return cls(t[keep], p[keep], q[keep], frame=frame, body=body, meta={"source": path})

    @classmethod
    def load(cls, path: str) -> "Trajectory":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
            return cls(z["t"], z["pos"], z["quat"], frame=str(z["frame"]), body=str(z["body"]), meta=meta)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path, t=self.t, pos=self.pos, quat=self.quat, frame=np.array(self.frame), body=np.array(self.body),
            meta=np.array(json.dumps(self.meta, ensure_ascii=False)),
        )

    def write_tum(self, path: str, t_query: np.ndarray | None = None, header: str | None = None) -> None:
        if t_query is None:
            write_tum(path, self.t, self.pos, self.quat, header=header)
        else:
            p, q, v = self.interp(t_query)
            write_tum(path, np.asarray(t_query)[v], p[v], q[v], header=header)

    # --------------------------------------------------------------- requêtes
    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    def interp(self, t_query, max_gap_s: float | None = None):
        """→ (pos [M,3], quat [M,4], valid [M]) ; linéaire + SLERP entre échantillons."""
        return interpolate_poses(self.t, self.pos, self.quat, np.atleast_1d(t_query), max_gap_s=max_gap_s)

    def rot_trans(self, t_query, max_gap_s: float | None = None):
        """→ (R [M,3,3], p [M,3], valid [M])."""
        p, q, v = self.interp(t_query, max_gap_s)
        return quat_to_rot(q), p, v

    def matrices(self, t_query, max_gap_s: float | None = None):
        """→ (T_<frame>_<body> [M,4,4], valid [M])."""
        R, p, v = self.rot_trans(t_query, max_gap_s)
        T = np.zeros((len(p), 4, 4))
        T[:, :3, :3], T[:, :3, 3], T[:, 3, 3] = R, p, 1.0
        return T, v

    def transformed(self, T_new_frame: np.ndarray, T_body_newbody: np.ndarray | None = None, frame: str | None = None):
        """Trajectoire T_new_body' = T_new_frame · T(t) · T_body_newbody."""
        Rf, tf = T_new_frame[:3, :3], T_new_frame[:3, 3]
        qf = rot_to_quat(Rf)
        q = quat_mul(qf[None], self.quat)
        p = self.pos @ Rf.T + tf
        if T_body_newbody is not None:
            qb = rot_to_quat(T_body_newbody[:3, :3])
            p = p + quat_rotate(q, np.broadcast_to(T_body_newbody[:3, 3], p.shape))
            q = quat_mul(q, qb[None])
        return Trajectory(self.t.copy(), p, q, frame=frame or self.frame, body=self.body, meta=dict(self.meta))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def integrate_gyro(t: np.ndarray, gyro: np.ndarray, R_body_imu: np.ndarray | None = None) -> np.ndarray:
    """Orientation (repère fixe arbitraire) du corps aux instants `t` par intégration du gyroscope.

    `gyro` [N,3] rad/s dans le repère imu ; si `R_body_imu` est donné, l'orientation renvoyée est celle du
    corps : q_w_body(t) = q_w_imu(t) ⊗ q_imu_body. Intégration par le point milieu (ω moyen entre échantillons).
    Renvoie des quaternions xyzw [N,4] avec q[0] = q_imu_body (orientation de départ arbitraire).
    """
    t = np.asarray(t, dtype=np.float64)
    w = np.asarray(gyro, dtype=np.float64)
    dt = np.diff(t)
    w_mid = 0.5 * (w[1:] + w[:-1])
    dq = quat_exp(w_mid * dt[:, None])
    q = np.empty((len(t), 4))
    q[0] = [0, 0, 0, 1.0]
    # Produit cumulé (boucle Python sur N ≈ 1e5 : ~1 s ; renormalisation périodique).
    cur = np.array([0, 0, 0, 1.0])
    for i in range(len(dq)):
        cur = quat_mul(cur, dq[i])
        if i % 1000 == 0:
            cur /= np.linalg.norm(cur)
        q[i + 1] = cur
    q = quat_normalize(q)
    if R_body_imu is not None:
        q_imu_body = rot_to_quat(np.asarray(R_body_imu).T)
        q = quat_mul(q, q_imu_body[None])
    return q


def _interp_quat_at(t_src: np.ndarray, q_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    _, q, _ = interpolate_poses(t_src, np.zeros((len(t_src), 3)), q_src, t_query)
    return q


def imu_shaped(
    anchor_t: np.ndarray,
    anchor_pos: np.ndarray,
    anchor_quat: np.ndarray,
    shape_t: np.ndarray,
    shape_quat: np.ndarray,
    shape_pos: np.ndarray | None = None,
    out_t: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trajectoire continue passant par les ancres, mise en forme par une trajectoire haute fréquence.

    `shape_quat` : orientation du corps (repère fixe quelconque) aux instants `shape_t` ; `shape_pos` (optionnel) :
    position du corps dans ce même repère (sinon translation linéaire entre ancres).
    `out_t` : instants de sortie (défaut : union des instants de forme et d'ancrage dans [t_0, t_K]).
    Renvoie (t, pos, quat).
    """
    at = np.asarray(anchor_t, dtype=np.float64)
    ap = np.asarray(anchor_pos, dtype=np.float64)
    aq = quat_normalize(anchor_quat)
    st = np.asarray(shape_t, dtype=np.float64)
    if out_t is None:
        inside = st[(st > at[0]) & (st < at[-1])]
        out_t = np.unique(np.concatenate([at, inside]))
    s = np.asarray(out_t, dtype=np.float64)
    if st[0] > at[0] or st[-1] < at[-1]:
        raise ValueError("la trajectoire de forme doit couvrir les ancres")
    k = np.clip(np.searchsorted(at, s, side="right") - 1, 0, len(at) - 2)
    alpha = np.clip((s - at[k]) / (at[k + 1] - at[k]), 0.0, 1.0)

    Fq_anchor = _interp_quat_at(st, shape_quat, at)
    Fq_s = _interp_quat_at(st, shape_quat, s)
    # Rotation : R(s) = A_k · Δ_k(s) · Exp(α Log E_k)
    delta_s = quat_mul(quat_conj(Fq_anchor[k]), Fq_s)
    delta_end = quat_mul(quat_conj(Fq_anchor[:-1]), Fq_anchor[1:])  # Δ_k(t_{k+1})
    rel_anchor = quat_mul(quat_conj(aq[:-1]), aq[1:])
    E = quat_mul(quat_conj(delta_end), rel_anchor)
    E = np.where(E[:, 3:4] < 0, -E, E)
    logE = quat_log(E)
    q_out = quat_mul(quat_mul(aq[k], delta_s), quat_exp(alpha[:, None] * logE[k]))

    if shape_pos is None:
        p_out = ap[k] + alpha[:, None] * (ap[k + 1] - ap[k])
    else:
        sp = np.asarray(shape_pos, dtype=np.float64)
        Fp_anchor = np.stack([np.interp(at, st, sp[:, i]) for i in range(3)], 1)
        Fp_s = np.stack([np.interp(s, st, sp[:, i]) for i in range(3)], 1)
        # d_k(s) : déplacement de la forme depuis t_k, exprimé dans le corps en t_k
        d_s = quat_rotate(quat_conj(Fq_anchor[k]), Fp_s - Fp_anchor[k])
        d_end = quat_rotate(quat_conj(Fq_anchor[:-1]), Fp_anchor[1:] - Fp_anchor[:-1])
        resid = ap[1:] - ap[:-1] - quat_rotate(aq[:-1], d_end)  # repère fixe
        p_out = ap[k] + quat_rotate(aq[k], d_s) + alpha[:, None] * resid[k]
    return s, p_out, quat_normalize(q_out)


def smooth_anchor_corrections(
    anchor_t: np.ndarray,
    anchor_pos: np.ndarray,
    anchor_quat: np.ndarray,
    odom_pos: np.ndarray,
    odom_quat: np.ndarray,
    sigma_s: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Lisse la correction C_k = A_k · inv(O_k) par un noyau gaussien de largeur `sigma_s` → Ã_k = C̃_k · O_k.

    Rotation : moyenne pondérée dans l'espace tangent en C_k ; translation : moyenne pondérée.
    Renvoie (pos, quat, info) des ancres lissées.
    """
    at = np.asarray(anchor_t, dtype=np.float64)
    aq, oq = quat_normalize(anchor_quat), quat_normalize(odom_quat)
    cq = quat_mul(aq, quat_conj(oq))
    cp = np.asarray(anchor_pos) - quat_rotate(cq, np.asarray(odom_pos))
    if sigma_s <= 0:
        return np.asarray(anchor_pos, dtype=np.float64).copy(), aq.copy(), {"sigma_s": 0.0}
    n = len(at)
    cq_s = np.empty_like(cq)
    cp_s = np.empty_like(cp)
    lo = np.searchsorted(at, at - 4 * sigma_s)
    hi = np.searchsorted(at, at + 4 * sigma_s, side="right")
    for i in range(n):
        j = slice(lo[i], hi[i])
        w = np.exp(-0.5 * ((at[j] - at[i]) / sigma_s) ** 2)
        w /= w.sum()
        d = quat_mul(quat_conj(cq[i])[None], cq[j])
        d = np.where(d[:, 3:4] < 0, -d, d)
        cq_s[i] = quat_mul(cq[i], quat_exp((w[:, None] * quat_log(d)).sum(0)))
        cp_s[i] = (w[:, None] * cp[j]).sum(0)
    q_new = quat_normalize(quat_mul(cq_s, oq))
    p_new = cp_s + quat_rotate(cq_s, np.asarray(odom_pos))
    dq = quat_mul(quat_conj(aq), q_new)
    dang = np.degrees(np.linalg.norm(quat_log(dq), axis=1))
    dpos = np.linalg.norm(p_new - anchor_pos, axis=1)
    info = {
        "sigma_s": float(sigma_s),
        "anchor_change_rot_deg": {"median": float(np.median(dang)), "p95": float(np.percentile(dang, 95)), "max": float(dang.max())},
        "anchor_change_pos_m": {"median": float(np.median(dpos)), "p95": float(np.percentile(dpos, 95)), "max": float(dpos.max())},
    }
    return p_new, q_new, info


# ---------------------------------------------------------------------------
# Auto-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    import os
    import tempfile

    rng = np.random.default_rng(1)
    # Mouvement synthétique : rotation oscillante + translation courbe, gyro exact à 400 Hz.
    t = np.arange(0, 10, 0.0025)
    w = np.stack([0.4 * np.sin(2 * np.pi * 2.1 * t), 0.3 * np.cos(2 * np.pi * 1.3 * t), 0.2 + 0.1 * np.sin(t)], 1)
    q_true = integrate_gyro(t, w)
    p_true = np.stack([np.sin(0.3 * t) * 5, np.cos(0.2 * t) * 3, 0.05 * np.sin(2 * np.pi * 2 * t)], 1)
    # Ancres à 10 Hz ; gyroscope biaisé.
    ak = np.arange(0, len(t), 40)
    bias = np.array([0.004, -0.002, 0.003])
    q_shape = integrate_gyro(t, w + bias)
    s, p, q = imu_shaped(t[ak], p_true[ak], q_true[ak], t, q_shape)
    qi = _interp_quat_at(t, q_true, s)
    err = np.degrees(np.linalg.norm(quat_log(quat_mul(quat_conj(qi), q)), axis=1))
    # SLERP pur entre ancres pour comparaison
    _, q_sl, _ = interpolate_poses(t[ak], p_true[ak], q_true[ak], s)
    err_sl = np.degrees(np.linalg.norm(quat_log(quat_mul(quat_conj(qi), q_sl)), axis=1))
    assert err.max() < 0.002, err.max()
    assert err_sl.max() > 10 * err.max()
    # Passage exact par les ancres
    i = np.searchsorted(s, t[ak])
    assert np.allclose(p[i], p_true[ak]) and np.allclose(np.abs(np.sum(q[i] * q_true[ak], 1)), 1, atol=1e-10)
    # Forme avec translation (exacte) → erreur de position quasi nulle
    s2, p2, _ = imu_shaped(t[ak], p_true[ak], q_true[ak], t, q_true, shape_pos=p_true)
    pi = np.stack([np.interp(s2, t, p_true[:, j]) for j in range(3)], 1)
    assert np.abs(p2 - pi).max() < 1e-9
    # Lissage : corrections constantes par morceaux → la trajectoire lissée reste proche et continue.
    n = len(ak)
    seg = np.arange(n) // 30
    corr_rv = rng.normal(scale=np.radians(0.05), size=(seg.max() + 1, 3))[seg]
    corr_p = rng.normal(scale=0.02, size=(seg.max() + 1, 3))[seg]
    cq = quat_exp(corr_rv)
    A_q = quat_mul(cq, q_true[ak])
    A_p = quat_rotate(cq, p_true[ak]) + corr_p
    p_s, q_s, info = smooth_anchor_corrections(t[ak], A_p, A_q, p_true[ak], q_true[ak], sigma_s=1.0)
    assert info["anchor_change_rot_deg"]["max"] < 0.2
    # σ = 0 → identité
    p0, q0, _ = smooth_anchor_corrections(t[ak], A_p, A_q, p_true[ak], q_true[ak], sigma_s=0.0)
    assert np.allclose(p0, A_p)
    # Trajectory : sauvegarde / chargement / interpolation / changement de repère
    tr = Trajectory(s, p, q, frame="map", meta={"method": "test"})
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "spline.npz")
        tr.save(f)
        tr2 = Trajectory.load(f)
        tr2.write_tum(os.path.join(d, "x.tum"), t_query=t[ak][:5])
        tr3 = Trajectory.from_tum(os.path.join(d, "x.tum"))
    assert tr2.meta["method"] == "test" and np.allclose(tr2.pos, tr.pos) and len(tr3.t) == 5
    T, v = tr.matrices([s[5], s[5] + 1e-3, -1.0])
    assert v.tolist() == [True, True, False]
    Tw = np.eye(4)
    Tw[:3, :3] = quat_to_rot(quat_exp(np.array([0.1, -0.2, 0.3])))
    Tw[:3, 3] = [1, 2, 3]
    Tb = np.eye(4)
    Tb[:3, :3] = quat_to_rot(np.array([0, 0, 1.0, 0]))
    Tb[:3, 3] = [0, 0, 0.124]
    Tn, _ = tr.transformed(Tw, Tb, frame="world").matrices([s[100]])
    To, _ = tr.matrices([s[100]])
    assert np.allclose(Tn[0], Tw @ To[0] @ Tb, atol=1e-9)
    print(f"splatpipe.common.trajectory : auto-test OK (forme IMU max {err.max():.2e}°, SLERP max {err_sl.max():.3f}°)")


if __name__ == "__main__":
    _self_test()
