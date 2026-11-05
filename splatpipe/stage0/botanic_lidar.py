"""BotanicGarden — étape 0 LiDAR (Velodyne VLP-16) : extraction des balayages et de l'IMU, modèle de faisceaux gsplat,
contrôle de la convention de temps de la trajectoire de référence.

Faits établis sur les données (cf. etapes/notes/datasets/botanicgarden.md) :
- `/velodyne_points` : nuage NON organisé (1 × N), champs x y z intensity ring time ; `time` ∈ [−99,5 ; +1,3] ms par
  rapport à l'en-tête → l'en-tête est daté en FIN de tour ; repère `velodyne` = repère de la trajectoire de référence ;
- `ring` 0..15 ; ligne gsplat = ordre d'élévation décroissant (ligne 0 = faisceau le plus haut) ;
- colonnes : azimut quantifié au pas angulaire mesuré (≈ 0,2° à 10 Hz), relatif à l'azimut de coupure du tour, dans le
  sens de rotation ; temps linéaire en colonne.

  venv/bin/python -m splatpipe.stage0.botanic_lidar extract --bag data_local/botanicgarden/1018_00/1018_00_LIO.bag \
      --out data_local/botanicgarden/1018_00/lidar_raw
  venv/bin/python -m splatpipe.stage0.botanic_lidar model --raw data_local/botanicgarden/1018_00/lidar_raw
  venv/bin/python -m splatpipe.stage0.botanic_lidar timecheck --raw ... --gt data_local/botanicgarden/gt_pose/gt_1018_00_qnew.tum
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

NAME = "vlp16"
PC2 = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}


# ============================================================================= extraction
def extract(bag: str, out: str):
    from pathlib import Path

    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS1_NOETIC)
    os.makedirs(out, exist_ok=True)
    imu = {}
    n = 0
    with AnyReader([Path(bag)], default_typestore=ts) as r:
        conns = [c for c in r.connections if c.topic in ("/velodyne_points", "/imu/data", "/livox/imu", "/odom")]
        for con, t, raw in r.messages(connections=conns):
            m = r.deserialize(raw, con.msgtype)
            st = int(m.header.stamp.sec) * 1_000_000_000 + int(m.header.stamp.nanosec)
            if con.topic == "/velodyne_points":
                dt = np.dtype({"names": [f.name for f in m.fields], "formats": [PC2[f.datatype] for f in m.fields],
                               "offsets": [f.offset for f in m.fields], "itemsize": m.point_step})
                a = np.frombuffer(bytes(m.data), dtype=dt, count=m.width * m.height)
                np.savez_compressed(f"{out}/{st}.npz", xyz=np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float32),
                                    intensity=a["intensity"].astype(np.float32), ring=a["ring"].astype(np.uint8),
                                    time=a["time"].astype(np.float32), stamp_ns=np.int64(st))
                n += 1
            elif con.topic in ("/imu/data", "/livox/imu"):
                d = imu.setdefault(con.topic, {"t": [], "gyro": [], "acc": []})
                d["t"].append(st / 1e9)
                d["gyro"].append((m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z))
                d["acc"].append((m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z))
            elif con.topic == "/odom":
                d = imu.setdefault("/odom", {"t": [], "pos": [], "quat": []})
                d["t"].append(st / 1e9)
                p, q = m.pose.pose.position, m.pose.pose.orientation
                d["pos"].append((p.x, p.y, p.z))
                d["quat"].append((q.x, q.y, q.z, q.w))
    for k, d in imu.items():
        np.savez_compressed(f"{out}/{k.strip('/').replace('/', '_')}.npz", **{a: np.asarray(v) for a, v in d.items()})
    print(json.dumps({"balayages": n, "autres": {k: len(v["t"]) for k, v in imu.items()}}))


def load_raw(path: str) -> dict:
    z = np.load(path)
    t = z["time"].astype(np.float64)
    st = int(z["stamp_ns"])
    rng = np.linalg.norm(z["xyz"], axis=1)
    v = np.isfinite(rng) & (rng > 0)
    return dict(xyz=z["xyz"][v].astype(np.float64), ring=z["ring"][v].astype(np.int64), t_abs=st / 1e9 + t[v],
                intensity=z["intensity"][v], stamp_ns=st)


def raw_files(raw: str) -> list[str]:
    return sorted(f for f in glob.glob(f"{raw}/*.npz") if os.path.basename(f)[0].isdigit())


# ============================================================================= modèle
def _wrap(a):
    return np.angle(np.exp(1j * a))


def estimate_model(raw: str, n: int = 60):
    from splatpipe.common.lidar_model import CLOCKWISE, COUNTER_CLOCKWISE, StructuredSpinningLidarModel

    files = raw_files(raw)
    files = files[len(files) // (n + 1)::len(files) // (n + 1)][:n]
    S = [load_raw(f) for f in files]
    rep = {"n_sweeps": len(S)}
    # élévation par anneau (points > 3 m)
    P = np.concatenate([s["xyz"] for s in S])
    R = np.concatenate([s["ring"] for s in S])
    rho = np.linalg.norm(P, axis=1)
    el = np.arctan2(P[:, 2], np.hypot(P[:, 0], P[:, 1]))
    az = np.arctan2(P[:, 1], P[:, 0])
    rings = np.unique(R)
    el_ring = np.array([np.median(el[(R == r) & (rho > 3)]) for r in rings])
    order = np.argsort(-el_ring)                              # ligne 0 = le plus haut
    row_of_ring = np.empty(rings.max() + 1, np.int64)
    row_of_ring[rings[order]] = np.arange(len(rings))
    row_el = el_ring[order]
    # sens de rotation et vitesse angulaire : azimut vs temps dans chaque tour
    slopes, durs, az0 = [], [], []
    for s in S:
        t = s["t_abs"] - s["t_abs"].min()
        a = np.unwrap(np.arctan2(s["xyz"][:, 1], s["xyz"][:, 0])[np.argsort(t)])
        ts = np.sort(t)
        slopes.append(np.polyfit(ts, a, 1)[0])
        durs.append(ts[-1])
    w = float(np.median(slopes))                              # rad/s (signé)
    direction = COUNTER_CLOCKWISE if w > 0 else CLOCKWISE
    # pas angulaire : écart médian entre azimuts successifs d'un même anneau
    steps = []
    for s in S[:10]:
        for r in rings[:4]:
            k = s["ring"] == r
            a = np.arctan2(s["xyz"][k, 1], s["xyz"][k, 0])[np.argsort(s["t_abs"][k])]
            d = np.abs(_wrap(np.diff(a)))
            steps.append(np.median(d[d > 0]))
    step = float(np.median(steps))
    n_cols = int(round(2 * np.pi / step))
    dlt = 2 * np.pi / n_cols
    # azimut de coupure : azimut des premiers points des tours (temps minimal)
    first_az = []
    for s in S:
        k = np.argsort(s["t_abs"])[:50]
        first_az.append(np.angle(np.exp(1j * np.arctan2(s["xyz"][k, 1], s["xyz"][k, 0])).mean()))
    a0 = float(np.angle(np.exp(1j * np.array(first_az)).mean()))
    sgn = 1 if direction == COUNTER_CLOCKWISE else -1
    col_az = a0 + sgn * dlt * (np.arange(n_cols) + 0.5)       # centre des cellules
    model = StructuredSpinningLidarModel(
        name="Velodyne VLP-16 (BotanicGarden), estimé sur les nuages", row_elevations_rad=row_el,
        column_azimuths_rad=col_az, row_azimuth_offsets_rad=np.zeros(len(rings)), spinning_frequency_hz=abs(w) / (2 * np.pi),
        spinning_direction=direction, max_returns=1, row_ring_ids=rings[order], row_time_offsets_s=np.zeros(len(rings)),
        notes={"colonnes": "azimut relatif à la coupure, au pas mesuré ; ligne = anneau trié par élévation décroissante",
               "origine": "repère velodyne (centre du capteur)"})
    # contrôle : affectation (ligne, colonne) et résidus
    col = np.floor(_wrap(sgn * (az - a0)) % (2 * np.pi) / dlt).astype(np.int64) % n_cols
    row = row_of_ring[R]
    el_res = np.degrees(el - row_el[row])
    az_res = np.degrees(_wrap(az - col_az[col]))
    rep.update({"anneaux_elevation_deg": np.degrees(row_el).round(3).tolist(), "sens": direction,
                "frequence_hz": abs(w) / (2 * np.pi), "duree_tour_ms_mediane": float(np.median(durs) * 1e3),
                "pas_angulaire_deg": float(np.degrees(step)), "n_colonnes": n_cols, "azimut_coupure_deg": float(np.degrees(a0)),
                "residu_elevation_deg_p50_p99_au_dela_3m": np.percentile(np.abs(el_res[rho > 3]), [50, 99]).round(3).tolist(),
                "residu_azimut_deg_p99": float(np.percentile(np.abs(az_res), 99))})
    # collisions : plusieurs points dans la même cellule d'un tour
    coll = []
    i0 = 0
    for s in S:
        m_ = len(s["xyz"])
        cell = row[i0:i0 + m_] * n_cols + col[i0:i0 + m_]
        coll.append(1 - len(np.unique(cell)) / m_)
        i0 += m_
    rep["frac_points_cellule_partagee"] = float(np.mean(coll))
    return model, rep, row_of_ring


# ============================================================================= trajectoire
def gt_trajectory(path: str, dt: float = 0.0):
    """Trajectoire de référence (repère velodyne, une pose par tour, datée en fin de tour). Lignes non finies écartées
    (1018-00 : une pose NaN). `dt` est ajouté aux temps."""
    from splatpipe.common.trajectory import Trajectory

    g = np.loadtxt(path)
    g = g[np.isfinite(g).all(1)]
    return Trajectory(g[:, 0] + dt, g[:, 1:4], g[:, 4:8], frame="map", body="velodyne",
                      meta={"source": path, "dt_s": dt})


def shaped_trajectory(gt_path: str, imu_npz: str, dt: float):
    """GT recalée de `dt` + gyroscope Xsens entre les poses (même construction que DiTer++)."""
    from splatpipe.stage0.diterpp_lidar import build_trajectory, fit_imu

    gt = gt_trajectory(gt_path, dt)
    imu = np.load(imu_npz)
    fit = fit_imu(gt, imu["t"], imu["gyro"])
    return build_trajectory(gt, imu["t"], imu["gyro"], fit), fit


def deskew(sw: dict, traj) -> np.ndarray:
    tq, inv = np.unique(sw["t_abs"], return_inverse=True)
    T, _ = traj.matrices(tq)
    T = T[inv]
    return np.einsum("nij,nj->ni", T[:, :3, :3], sw["xyz"]) + T[:, :3, 3]


def timecheck(raw: str, gt_path: str, dts=(0.0, -0.025, -0.05), n_test: int = 30, gap_s: float = 5.0,
              radius: float = 30.0) -> dict:
    """Convention de temps de la GT sans carte de référence : pour chaque hypothèse (pose datée t + dt), carte des
    balayages (1 sur 3, 1 point sur 2) puis écart point → plan local de balayages de test à la carte construite SANS
    les balayages à moins de `gap_s` s d'eux. Murs (normale horizontale) et sol séparés."""
    from scipy.spatial import cKDTree

    files = raw_files(raw)
    imu_npz = f"{raw}/imu_data.npz"
    out = {}
    for dt in dts:
        traj, fit = shaped_trajectory(gt_path, imu_npz, dt)
        pts, tt = [], []
        for f in files[::3]:
            sw = load_raw(f)
            if sw["t_abs"].min() < traj.t_start or sw["t_abs"].max() > traj.t_end:
                continue
            rho = np.linalg.norm(sw["xyz"], axis=1)
            k = (rho > 1.5) & (rho < radius)
            sw = {a: (v[k] if isinstance(v, np.ndarray) and len(v) == len(rho) else v) for a, v in sw.items()}
            p = deskew(sw, traj)[::2]
            pts.append(p)
            tt.append(np.full(len(p), sw["stamp_ns"] / 1e9))
        P, TT = np.concatenate(pts), np.concatenate(tt)
        tests = files[len(files) // (n_test + 1)::len(files) // (n_test + 1)][:n_test]
        res = {"murs": [], "sol": []}
        for f in tests:
            sw = load_raw(f)
            if sw["t_abs"].min() < traj.t_start or sw["t_abs"].max() > traj.t_end:
                continue
            rho = np.linalg.norm(sw["xyz"], axis=1)
            k = (rho > 1.5) & (rho < radius)
            sw = {a: (v[k] if isinstance(v, np.ndarray) and len(v) == len(rho) else v) for a, v in sw.items()}
            q = deskew(sw, traj)
            far = np.abs(TT - sw["stamp_ns"] / 1e9) > gap_s
            tree = cKDTree(P[far])
            dd, ii = tree.query(q, k=12, workers=-1)
            nb = P[far][ii]
            mu = nb.mean(1)
            c = np.einsum("nki,nkj->nij", nb - mu[:, None], nb - mu[:, None]) / 12
            lam, vec = np.linalg.eigh(c)
            planar = (lam[:, 0] / lam.sum(1).clip(1e-12) < 0.01) & (dd[:, -1] < 0.3)
            n0 = vec[:, :, 0]
            d = np.abs(((q - mu) * n0).sum(1))
            res["murs"].append(d[planar & (np.abs(n0[:, 2]) < 0.3)])
            res["sol"].append(d[planar & (np.abs(n0[:, 2]) > 0.9)])
        out[f"{dt * 1000:+.0f} ms"] = {
            lab: {"n": int(sum(len(x) for x in v)), "mediane_cm": float(np.median(np.concatenate(v)) * 100),
                  "p90_cm": float(np.percentile(np.concatenate(v), 90) * 100)} for lab, v in res.items()}
        out[f"{dt * 1000:+.0f} ms"]["imu_dt_s"] = fit["dt_s"]
        print(f"dt {dt * 1000:+.0f} ms", json.dumps(out[f"{dt * 1000:+.0f} ms"]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "model", "timecheck"])
    ap.add_argument("--gt")
    ap.add_argument("--bag")
    ap.add_argument("--raw")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.stage == "extract":
        extract(a.bag, a.out)
    elif a.stage == "timecheck":
        out = a.out or os.path.join(os.path.dirname(a.raw.rstrip("/")), "stage0")
        rep = timecheck(a.raw, a.gt)
        json.dump(rep, open(f"{out}/timecheck.json", "w"), indent=1, ensure_ascii=False)
    else:
        out = a.out or os.path.join(os.path.dirname(a.raw.rstrip("/")), "stage0")
        os.makedirs(out, exist_ok=True)
        model, rep, _ = estimate_model(a.raw)
        model.save_json(f"{out}/{NAME}.json")
        json.dump(rep, open(f"{out}/{NAME}_estimation.json", "w"), indent=1, ensure_ascii=False)
        print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
