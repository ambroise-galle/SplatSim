"""Modèle de caméra `OPENCV_FISHEYE` (équidistant de Kalibr, k1–k4) en numpy, sans dépendance à OpenCV.

Conventions (voir `etapes/notes/conventions.md`) : repère optique OpenCV (x droite, y bas, z avant),
pixels (u, v) avec l'origine au **centre** du pixel (0, 0) en haut à gauche, comme OpenCV.

Modèle (identique à `cv2.fisheye.projectPoints` pour z > 0, sans skew) :

    θ   = atan2(√(x² + y²), z)                      angle au centre optique (rad), valide aussi pour z ≤ 0
    θ_d = θ · (1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
    u   = fx · θ_d · x / √(x² + y²) + cx
    v   = fy · θ_d · y / √(x² + y²) + cy

Remarque : OpenCV calcule `θ = atan(r)` avec `r = √(x²+y²)/z`, ce qui replie les points derrière la
caméra (z < 0) sur l'avant. Ici `atan2` garde θ ∈ [0, π] ; `project` renvoie un masque `valid` qui
exclut les angles où le polynôme n'est plus monotone (`theta_max`) et, si demandé, les pixels hors image.

Auto-test (comparaison à OpenCV si installé) : `python -m splatpipe.common.camera_model`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FisheyeCamera:
    """Caméra OPENCV_FISHEYE : intrinsèques `fx, fy, cx, cy`, distorsion `k1..k4`, taille `width × height`."""

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    k4: float
    width: int
    height: int

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, d: dict) -> "FisheyeCamera":
        """Depuis une entrée `cameras.<cam>` de `sensors.yaml` (clé `model` = OPENCV_FISHEYE si présente)."""
        model = d.get("model", "OPENCV_FISHEYE")
        if model != "OPENCV_FISHEYE":
            raise ValueError(f"modèle de caméra non supporté : {model}")
        return cls(*(float(d[k]) for k in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")),
                   int(d["width"]), int(d["height"]))

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def D(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.k3, self.k4])

    # ------------------------------------------------------------------ distorsion
    def distort_theta(self, theta: np.ndarray) -> np.ndarray:
        """θ → θ_d = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)."""
        t2 = np.asarray(theta, dtype=np.float64) ** 2
        return theta * (1.0 + t2 * (self.k1 + t2 * (self.k2 + t2 * (self.k3 + t2 * self.k4))))

    def _dtheta_d(self, theta: np.ndarray) -> np.ndarray:
        t2 = np.asarray(theta, dtype=np.float64) ** 2
        return 1.0 + t2 * (3 * self.k1 + t2 * (5 * self.k2 + t2 * (7 * self.k3 + t2 * 9 * self.k4)))

    @property
    def theta_max(self) -> float:
        """Plus grand θ ≤ π tel que θ_d reste strictement croissant sur [0, θ] (domaine inversible)."""
        th = np.linspace(0.0, np.pi, 20001)
        bad = np.nonzero(self._dtheta_d(th) <= 0)[0]
        return float(np.pi if len(bad) == 0 else th[max(bad[0] - 1, 0)])

    def undistort_theta(self, theta_d: np.ndarray, iters: int = 20) -> np.ndarray:
        """θ_d → θ par Newton (départ θ = θ_d), borné à [0, theta_max]."""
        td = np.asarray(theta_d, dtype=np.float64)
        th = td.copy()
        tmax = self.theta_max
        for _ in range(iters):
            f = self.distort_theta(th) - td
            th = np.clip(th - f / self._dtheta_d(th), 0.0, tmax)
        return th

    # ------------------------------------------------------------------ projection
    def project(self, p_cam: np.ndarray, margin: float | None = None, theta_limit: float | None = None):
        """Points (N, 3) du repère caméra → `(uv (N, 2), valid (N,), theta (N,))`.

        `valid` : θ < min(theta_max, theta_limit) et point non nul ; si `margin` n'est pas None,
        le pixel doit aussi être dans `[−0,5 + margin ; width − 0,5 − margin]` (idem en v).
        """
        p = np.asarray(p_cam, dtype=np.float64)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        rxy = np.hypot(x, y)
        theta = np.arctan2(rxy, z)
        td = self.distort_theta(theta)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(rxy > 1e-12, td / np.where(rxy > 1e-12, rxy, 1.0), 1.0 / np.where(z > 0, z, 1.0))
        u = self.fx * s * x + self.cx
        v = self.fy * s * y + self.cy
        tlim = self.theta_max if theta_limit is None else min(self.theta_max, theta_limit)
        valid = (theta < tlim) & (np.linalg.norm(p, axis=-1) > 0)
        if margin is not None:
            valid &= (u >= -0.5 + margin) & (u <= self.width - 0.5 - margin)
            valid &= (v >= -0.5 + margin) & (v <= self.height - 0.5 - margin)
        return np.stack([u, v], axis=-1), valid, theta

    def unproject(self, uv: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) → rayons unitaires (N, 3) dans le repère caméra (inverse de `project`)."""
        uv = np.asarray(uv, dtype=np.float64)
        mx = (uv[..., 0] - self.cx) / self.fx
        my = (uv[..., 1] - self.cy) / self.fy
        td = np.hypot(mx, my)
        th = self.undistort_theta(td)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(td > 1e-12, np.sin(th) / np.where(td > 1e-12, td, 1.0), 1.0)
        return np.stack([mx * k, my * k, np.cos(th)], axis=-1)


# ---------------------------------------------------------------------------
# Auto-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    rng = np.random.default_rng(0)
    # Intrinsèques de cam0 d'Oxford Spires (calibration fournie).
    cam = FisheyeCamera(698.1692094914263, 698.5147584873237, 712.7646767255006, 560.5751284820508,
                        -0.04114225009462773, -0.002421765984053725, 0.003400355152679783,
                        -0.0018518822347983275, 1440, 1080)
    n = 20000
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[:, 2] = np.abs(d[:, 2]) + 1e-3  # devant la caméra
    p = d * rng.uniform(0.5, 80.0, size=(n, 1))
    uv, valid, theta = cam.project(p)
    assert valid.all() and theta.max() < np.pi / 2

    # Aller-retour projection → rayon.
    rays = cam.unproject(uv)
    err = np.linalg.norm(rays - d / np.linalg.norm(d, axis=1, keepdims=True), axis=1)  # ≈ angle en rad
    assert err.max() < 1e-9, err.max()

    # Points derrière le plan image (z < 0, θ > 90°) : aller-retour tant que θ < theta_max.
    th_b = rng.uniform(np.pi / 2, min(cam.theta_max, 2.8), size=500)
    ph_b = rng.uniform(-np.pi, np.pi, size=500)
    pb = np.stack([np.sin(th_b) * np.cos(ph_b), np.sin(th_b) * np.sin(ph_b), np.cos(th_b)], 1) * 7.0
    uvb, vb, _ = cam.project(pb)
    assert vb.all()
    assert np.abs(cam.unproject(uvb) - pb / 7.0).max() < 1e-9

    # Axe optique → point principal.
    uv0, _, _ = cam.project(np.array([[0.0, 0.0, 3.0]]))
    assert np.allclose(uv0, [[cam.cx, cam.cy]])

    # Marge d'image.
    _, vin, _ = cam.project(np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]]), margin=5)
    assert vin.tolist() == [True, False]

    try:
        import cv2
    except ImportError:
        print(f"splatpipe.common.camera_model : auto-test OK (OpenCV absent, comparaison sautée ; theta_max = "
              f"{np.degrees(cam.theta_max):.1f}°)")
        return

    # Comparaison à cv2.fisheye.projectPoints (points devant la caméra, repère caméra : rvec = tvec = 0).
    uv_cv, _ = cv2.fisheye.projectPoints(p.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cam.K, cam.D)
    diff = np.abs(uv_cv.reshape(-1, 2) - uv)
    assert diff.max() < 1e-6, diff.max()
    # Avec une pose quelconque : p_cam = R · p_monde + t.
    rvec = rng.normal(size=3) * 0.3
    tvec = rng.normal(size=3) * 0.1
    R, _ = cv2.Rodrigues(rvec)
    pw = (p - tvec) @ R  # p_cam = R pw + t
    uv_cv2, _ = cv2.fisheye.projectPoints(pw.reshape(-1, 1, 3), rvec, tvec, cam.K, cam.D)
    assert np.abs(uv_cv2.reshape(-1, 2) - uv).max() < 1e-6
    # Comparaison à cv2.fisheye.undistortPoints (coordonnées normalisées x/z, y/z).
    und = cv2.fisheye.undistortPoints(uv.reshape(-1, 1, 2), cam.K, cam.D,
                                      criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12))
    und = und.reshape(-1, 2)
    ours = rays[:, :2] / rays[:, 2:3]
    small = theta < np.radians(80)  # OpenCV perd en précision près de 90°
    rel = np.abs(und[small] - ours[small]).max()
    assert rel < 1e-6, rel
    print(f"splatpipe.common.camera_model : auto-test OK (OpenCV {cv2.__version__} : projection max |Δ| = "
          f"{diff.max():.2e} px, undistort max |Δ| = {rel:.2e} ; theta_max = {np.degrees(cam.theta_max):.1f}°)")


if __name__ == "__main__":
    _self_test()
