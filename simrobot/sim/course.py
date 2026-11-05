"""Circuit du mode jeu (`server.py --jeu`) : la vraie trajectoire de l'ANYmal (DLIO, `trajets.json`), rééchantillonnée
tous les 0,5 m, lissée sur ±1 m, coupée avant les repassages (la mission revient sur ses pas après ~250 m). Portes de
passage tous les 10 m, chevrons dans les virages (cap qui tourne de plus de 35° entre −2,5 et +2,5 m), posés côté
extérieur sur le sol physique. Tout est en repère physique (x, y, z haut).

  .venv/bin/python simrobot/sim/course.py --scene data_local/simrobot/scenes/HOB-1_full   # pilote automatique
"""

from __future__ import annotations

import json

import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def build_track(scene_dir, terrain, s_max=250.0, step=0.5, gate_every=10.0, turn_deg=35.0, depart_s=6.0):
    """Départ à 6 m : avant, une tache grise du splat (l'ANYmal y est resté immobile) et un obstacle de bord de sentier
    droit devant."""
    from terrain import height_at

    D = np.array(json.load(open(f"{scene_dir}/trajets.json"))["dlio"]["p"])[:, :2]
    seg = np.linalg.norm(np.diff(D, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    sk = np.arange(0, min(s_max, s[-1]), step)
    P = np.stack([np.interp(sk, s, D[:, 0]), np.interp(sk, s, D[:, 1])], 1)
    w = int(round(1.0 / step))                                   # lissage ±1 m
    ker = np.ones(2 * w + 1) / (2 * w + 1)
    Ps = np.stack([np.convolve(np.pad(P[:, k], w, mode="edge"), ker, mode="valid") for k in range(2)], 1)
    tang = np.gradient(Ps, axis=0)
    yaw = np.unwrap(np.arctan2(tang[:, 1], tang[:, 0]))
    z = height_at(terrain, Ps)
    n = len(Ps)
    h = int(round(2.5 / step))
    dyaw = np.array([yaw[min(k + h, n - 1)] - yaw[max(k - h, 0)] for k in range(n)])
    corner = np.abs(dyaw) > np.radians(turn_deg)
    chevrons = []
    k = 0
    while k < n:
        if not corner[k]:
            k += 1
            continue
        k1 = k
        while k1 < n and corner[k1]:
            k1 += 1
        sign = np.sign(dyaw[k:k1].sum())                         # > 0 : virage à gauche
        a = max(k - int(1.5 / step), 0)
        for j in range(a, k1, int(1.5 / step))[:5]:
            nrm = np.array([-np.sin(yaw[j]), np.cos(yaw[j])])    # normale gauche
            pos = Ps[j] - sign * 1.1 * nrm                       # côté extérieur
            chevrons.append({"x": float(pos[0]), "y": float(pos[1]), "z": float(height_at(terrain, pos[None])[0]),
                             "yaw": float(yaw[j]), "gauche": bool(sign > 0)})
        k = k1
    gates = [{"i": int(i), "s": float(sk[i])} for i in range(0, n, int(round(gate_every / step)))][1:]
    return {"pts": np.round(np.c_[Ps, z], 3).tolist(), "yaw": np.round(yaw, 4).tolist(), "step": step,
            "longueur": float(sk[-1]), "portes": [g for g in gates if g["s"] > depart_s + 2], "chevrons": chevrons,
            "depart_i": int(round(depart_s / step))}


def autopilot(scene_dir, v_cible=5.0, max_t=180.0):
    """Pilote automatique (poursuite pure) sur le circuit avec la voiture : faisabilité et temps de référence."""
    import sys

    from server import Sim

    sim = Sim(scene_dir, lidar=False, vehicule="voiture")
    T = build_track(scene_dir, sim.terrain)
    P, yaw = np.array(T["pts"]), np.array(T["yaw"])
    r = sim.rover
    i0 = T["depart_i"]
    r.reset({"xyz": P[i0].tolist(), "yaw": float(yaw[i0])})
    k, t, stuck, prev, flips = i0, 0.0, 0, r.data.qpos[:2].copy(), 0
    while t < max_t and k < len(P) - 2:
        x, y = r.data.qpos[:2]
        q = r.data.qpos[3:7]
        hd = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        k += int(np.argmin(np.linalg.norm(P[k:k + 40, :2] - [x, y], axis=1)))
        tgt = P[min(k + 4, len(P) - 1), :2]                       # 2 m devant
        err = wrap(np.arctan2(tgt[1] - y, tgt[0] - x) - hd)
        v = r.forward_speed()
        curv = abs(wrap(yaw[min(k + 8, len(P) - 1)] - yaw[k]))    # virage à venir (4 m)
        vt = v_cible / (1 + 3 * curv)
        r.control(float(np.clip(0.5 * (vt - v) + 0.3, -1, 1)), float(np.clip(2.0 * err, -1, 1)), False, False, 0.02)
        r.step(10)
        t += 0.02
        if r.upright() < 0.3:
            flips += 1
            r.reset({"xyz": P[k].tolist(), "yaw": float(yaw[k])})
        if int(round(t / 0.02)) % 100 == 0:
            stuck = stuck + 1 if np.linalg.norm(r.data.qpos[:2] - prev) < 0.3 else 0
            prev = r.data.qpos[:2].copy()
            if stuck >= 2:                                       # bloqué 4 s : replacé 2 m plus loin
                print(f"  bloqué à s = {k * T['step']:.0f} m", file=sys.stderr)
                k = min(k + 4, len(P) - 3)
                r.reset({"xyz": P[k].tolist(), "yaw": float(yaw[k])})
                stuck = 0
    return {"longueur_m": T["longueur"], "temps_s": round(t, 1), "fini": bool(k >= len(P) - 3), "retournements": flips,
            "chevrons": len(T["chevrons"]), "portes": len(T["portes"])}


if __name__ == "__main__":
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--v", type=float, default=5.0)
    a = ap.parse_args()
    print(json.dumps(autopilot(a.scene, a.v), ensure_ascii=False))
