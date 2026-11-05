"""Suivi automatique de la VRAIE trajectoire de l'ANYmal (DLIO, `trajets.json`), tronçon par tronçon entre les départs
de `scene.json` : poursuite à 0,6 m devant, 25 s maximum par tronçon, bloqué si le robot avance de moins de 15 cm en
3 s. Compte les tronçons franchis et les retournements, et donne la cause probable de chaque blocage.

`--traversees` : test ciblé des obstacles bas retypés par les étiquettes (buissons, reliefs) à moins de 3 m du trajet.
Le robot part à 2,5 m de l'obstacle, face à lui, depuis le point du trajet le plus proche, et roule tout droit à 1 m/s
pendant 8 s. On mesure s'il le dépasse (≥ 1 m au-delà de son centre), sa vitesse minimale et le freinage subi.

  .venv/bin/python simrobot/tools/follow_test.py --scene data_local/simrobot/scenes/HOB-1_full            # avec étiquettes
  .venv/bin/python simrobot/tools/follow_test.py --scene data_local/simrobot/scenes/HOB-1_full --sans-etiquettes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from server import Sim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--terrain", default="terrain.npz")
    ap.add_argument("--sans-etiquettes", action="store_true", help="terrain sans les étiquettes du splat (référence)")
    ap.add_argument("--sans-sol", action="store_true", help="obstacles typés, mais adhérence uniforme (point 1 seul)")
    ap.add_argument("--out", default=None, help="résultats JSON")
    ap.add_argument("--traversees", action="store_true", help="test ciblé : buissons et reliefs proches du trajet")
    a = ap.parse_args()
    if a.traversees:
        return traversees(a)
    t_start = time.time()
    sim = Sim(a.scene, a.terrain, etiquettes=not a.sans_etiquettes, sol=not a.sans_sol, lidar=False)
    r, S = sim.rover, sim.meta["departs"]
    D = np.array(json.load(open(f"{a.scene}/trajets.json"))["dlio"]["p"])
    seg = np.linalg.norm(np.diff(D[:, :2], axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    D = D[np.r_[True, np.diff(np.floor(s / 0.05)) > 0]]                 # densifié à ~5 cm
    tr, src = sim.terrain["troncs"], sim.terrain["troncs_source"]
    idx, j0 = [], 0
    for st in S:                                                         # index DLIO de chaque départ (monotone)
        dd = np.linalg.norm(D[j0:j0 + 4000, :2] - st["xyz"][:2], axis=1)
        j0 += int(np.argmin(dd))
        idx.append(j0)
    ok_n, flips, rows = 0, 0, []
    for i in range(len(S) - 1):
        sim.reset(i)
        for _ in range(500):
            r.control(0, 0, False, False, 0.002)
            r.step()
        a_, b = idx[i], idx[i + 1]
        if b <= a_:
            continue
        k, t, prev, stuck, end = a_, 0.0, r.data.qpos[:2].copy(), 0, D[b, :2]
        while t < 25:
            x, y = r.data.qpos[:2]
            q = r.data.qpos[3:7]
            yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            k += int(np.argmin(np.linalg.norm(D[k:min(b + 1, k + 60), :2] - [x, y], axis=1)))
            tgt = D[min(k + 12, b), :2]
            err = (np.arctan2(tgt[1] - y, tgt[0] - x) - yaw + np.pi) % (2 * np.pi) - np.pi
            r.control(float(np.clip(1.2 - abs(err), 0.3, 1.0)), float(np.clip(2.5 * err, -1, 1)), False, False, 0.02)
            r.step(10)
            t += 0.02
            if np.linalg.norm(r.data.qpos[:2] - end) < 0.6:
                break
            if int(round(t / 0.02)) % 50 == 0:
                stuck = stuck + 1 if np.linalg.norm(r.data.qpos[:2] - prev) < 0.15 else 0
                prev = r.data.qpos[:2].copy()
                if stuck >= 3:
                    break
        x, y = r.data.qpos[:2]
        up = r.upright()
        row = {"troncon": f"{i}->{i + 1}", "duree_s": round(t, 1)}
        if np.linalg.norm([x - end[0], y - end[1]]) < 0.6:
            ok_n += 1
            row["franchi"] = True
        else:
            flips += up < 0.3
            de = np.hypot(tr[:, 0] - x, tr[:, 1] - y) - tr[:, 2]
            jn = int(np.argmin(de)) if len(tr) else -1
            caps = sim.terrain.get("capsules", np.zeros((0, 7)))
            if len(caps):                                      # distance au bord de la capsule la plus proche
                A, B = caps[:, :3], caps[:, 3:6]
                pz = np.array([x, y, r.data.qpos[2]])
                ab = B - A
                tt = np.clip(np.einsum("ij,ij->i", pz - A, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
                dc = np.linalg.norm(pz - (A + tt[:, None] * ab), axis=1) - caps[:, 6]
                row["capsule_proche_m"] = round(float(dc.min()), 2)
            row.update({"franchi": False, "retourne": bool(up < 0.3),
                        "avance_m": round(float(np.linalg.norm([x - D[a_, 0], y - D[a_, 1]])), 1),
                        "obstacle_proche_m": round(float(de[jn]), 2) if jn >= 0 else None,
                        "obstacle_type": (["tronc", "tronc (cont.)", "obstacle bas"][int(src[jn])] if jn >= 0 else None),
                        "sol": sim.rover.ground.name(x, y) if sim.rover.ground else None,
                        "frein_veg_N": round(sim.rover.ground_info["frein_veg_N"])})
            print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
    res = {"variante": "sans étiquettes" if a.sans_etiquettes else ("obstacles typés" if a.sans_sol else
                                                                     "obstacles typés + sol"),
           "franchis": ok_n, "troncons": len(S) - 1, "retournements": int(flips),
           "duree_moyenne_franchis_s": round(float(np.mean([r_["duree_s"] for r_ in rows if r_["franchi"]])), 2),
           "calcul_s": round(time.time() - t_start, 1)}
    print(json.dumps(res, ensure_ascii=False))
    if a.out:
        json.dump({"resume": res, "troncons": rows}, open(a.out, "w"), ensure_ascii=False, indent=1)


def traversees(a):
    from scipy.spatial import cKDTree

    E = np.load(f"{a.scene}/terrain_etiquettes.npz")
    sim = Sim(a.scene, a.terrain, etiquettes=not a.sans_etiquettes, sol=not a.sans_sol, lidar=False)
    r = sim.rover
    T0 = np.load(f"{a.scene}/{a.terrain}")
    tr, typ = T0["troncs"], E["obs_type"]
    D = np.array(json.load(open(f"{a.scene}/trajets.json"))["dlio"]["p"])
    kd = cKDTree(D[:, :2])
    rows = []
    for kind, name in ((1, "buisson"), (2, "relief")):
        sel = np.where(typ == kind)[0]
        dist, j = kd.query(tr[sel, :2])
        sel, j = sel[dist < 3.0], j[dist < 3.0]
        for o, jj in zip(sel, j):
            x, y, rad, _ = tr[o]
            d = np.array([x, y]) - D[jj, :2]
            if np.linalg.norm(d) < 0.3:                       # obstacle sur le trajet : partir dans l'axe du trajet
                d = D[min(jj + 5, len(D) - 1), :2] - D[max(jj - 5, 0), :2]
            u = d / max(np.linalg.norm(d), 1e-6)
            yaw = float(np.arctan2(u[1], u[0]))
            p0 = None
            for back in (2.5, 3.5, 4.5):                       # départ dégagé de TOUT obstacle d'origine
                c = np.array([x, y]) - back * u
                if (np.hypot(tr[:, 0] - c[0], tr[:, 1] - c[1]) - tr[:, 2]).min() > 0.6:
                    p0 = c
                    break
            if p0 is None:
                continue
            r.reset({"xyz": [p0[0], p0[1], 0.0], "yaw": yaw})
            for _ in range(300):
                r.control(0, 0, False, False, 0.002)
                r.step()
            if r.upright() < 0.9:                              # départ instable (talus) : écarté
                continue
            vmin, fmax, t = 9.0, 0.0, 0.0
            while t < 8.0:
                r.control(0.5, 0.0, False, False, 0.02)
                r.step(10)
                t += 0.02
                prog = float((r.data.qpos[:2] - [x, y]) @ u)
                if abs(prog) < rad + 0.5:                      # avant du châssis à 0,42 m du centre
                    vmin = min(vmin, abs(r.state()["speed"]))
                    if r.ground is not None:
                        fmax = max(fmax, r.ground_info["frein_veg_N"])
                if prog > 1.0 or r.upright() < 0.3:
                    break
            prog = float((r.data.qpos[:2] - [x, y]) @ u)
            # cause d'arrêt : contact avec un cylindre rigide (géométrie du monde), freinage d'un buisson, ou relief
            m_, d_ = r.model, r.data
            rigide = any(m_.geom_bodyid[d_.contact[k].geom1] == 0 and m_.geom_type[d_.contact[k].geom1] == 5
                         or m_.geom_bodyid[d_.contact[k].geom2] == 0 and m_.geom_type[d_.contact[k].geom2] == 5
                         for k in range(d_.ncon))
            frein = r.ground_info["frein_veg_N"] if r.ground is not None else 0.0
            cause = ("dépassé" if prog > 1.0 else "retourné" if r.upright() < 0.3 else "obstacle rigide" if rigide
                     else "freiné (buisson)" if frein > 1 else "relief / pente")
            # relief sous le robot : dénivelé du sol physique dans 0,6 m, pente moyenne
            Hh = sim.terrain["H"]
            ci = int((r.data.qpos[0] - sim.terrain["lo"][0]) / float(sim.terrain["res"]))
            cj = int((r.data.qpos[1] - sim.terrain["lo"][1]) / float(sim.terrain["res"]))
            win = Hh[max(ci - 6, 0):ci + 7, max(cj - 6, 0):cj + 7]
            gyy, gxx = np.gradient(win, float(sim.terrain["res"]))
            rows.append({"type": name, "obstacle": int(o), "mu": round(float(E["obs_mu"][o]), 2), "cause": cause,
                         "x": round(float(r.data.qpos[0]), 2), "y": round(float(r.data.qpos[1]), 2),
                         "denivele_0.6m": round(float(np.ptp(win)), 2) if win.size else None,
                         "pente_deg_q90": round(float(np.degrees(np.arctan(np.percentile(np.hypot(gxx, gyy), 90)))), 1)
                         if win.size else None,
                         "atteint": bool(vmin < 9.0),
                         "haut_m": round(float(E["obs_haut"][o]), 2), "depasse": bool(prog > 1.0),
                         "retourne": bool(r.upright() < 0.3), "v_min": round(min(vmin, 9.0), 2),
                         "frein_max_N": round(fmax)})
    res = {"variante": "sans étiquettes" if a.sans_etiquettes else ("obstacles typés" if a.sans_sol else
                                                                     "obstacles typés + sol")}
    for name in ("buisson", "relief"):
        R = [x for x in rows if x["type"] == name]
        A = [x for x in R if x["atteint"]]
        res[name] = {"n": len(R), "atteints": len(A), "depasses": sum(x["depasse"] for x in R),
                     "retournes": sum(x["retourne"] for x in R),
                     "v_min_mediane_au_contact": round(float(np.median([x["v_min"] for x in A])), 2) if A else None,
                     "frein_max_median_N": round(float(np.median([x["frein_max_N"] for x in A]))) if A else None,
                     "causes": {c: sum(x["cause"] == c for x in R) for c in sorted({x["cause"] for x in R})}}
    print(json.dumps(res, ensure_ascii=False))
    if a.out:
        json.dump({"resume": res, "obstacles": rows}, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
