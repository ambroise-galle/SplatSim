"""Étape 0.2 bis — Le Hesai QT64 d'Oxford Spires enregistre-t-il plusieurs retours par faisceau ?

Lancement :
  .venv/bin/modal run splatpipe/stage0/probe_returns.py
"""

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume

app = modal.App("splatsim-stage0-probe-returns")
vol = data_volume()
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "pypcd4").add_local_python_source("splatpipe")

SEQ_DIR = f"{DATA_MOUNT}/raw/oxford_spires/sequences/2024-03-13-observatory-quarter-01"


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=2, memory=8192, timeout=1200)
def probe(n_scans: int = 5) -> dict:
    import os

    import numpy as np
    from pypcd4 import PointCloud

    files = sorted(f for f in os.listdir(f"{SEQ_DIR}/raw/lidar-clouds") if f.endswith(".pcd"))
    picks = [files[int(i)] for i in np.linspace(len(files) * 0.1, len(files) * 0.9, n_scans)]
    out = []
    for fname in picks:
        pc = PointCloud.from_path(f"{SEQ_DIR}/raw/lidar-clouds/{fname}")
        d = {name: pc.numpy((name,))[:, 0] for name in ("x", "y", "z", "intensity", "timestamp", "ring")}
        x, y, z, t, ring = d["x"], d["y"], d["z"], d["timestamp"], d["ring"].astype(int)
        rng = np.sqrt(x**2 + y**2 + z**2)
        az = np.degrees(np.arctan2(y, x)) % 360.0
        el = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))

        # Horodatages distincts par ring = nombre de tirs (colonnes) ; points par tir = retours.
        per_ring_shots, per_shot_counts, dual_dr = [], [], []
        for r in range(64):
            m = ring == r
            if not m.any():
                continue
            tr, rr = t[m], rng[m]
            order = np.argsort(tr, kind="stable")
            tr, rr = tr[order], rr[order]
            uniq, inv, cnt = np.unique(np.round(tr * 1e6).astype(np.int64), return_inverse=True, return_counts=True)
            per_ring_shots.append(len(uniq))
            per_shot_counts.append(cnt)
            for k in np.where(cnt == 2)[0][:2000]:
                v = rr[inv == k]
                dual_dr.append(abs(v[1] - v[0]))
        per_shot_counts = np.concatenate(per_shot_counts)
        dual_dr = np.asarray(dual_dr)

        t_rel = t - t.min()
        # Période de tir : pas médian entre horodatages distincts d'un même ring.
        m0 = ring == 20
        ts = np.unique(np.round(t[m0] * 1e6)) * 1e-6
        col_period = float(np.median(np.diff(ts))) if len(ts) > 2 else None
        az0 = np.sort(np.unique(np.round(az[m0], 2)))
        el_per_ring = [float(np.median(el[ring == r])) for r in range(64) if (ring == r).any()]

        out.append({
            "file": fname,
            "n_points": int(len(x)),
            "scan_duration_s": round(float(t_rel.max()), 5),
            "file_stamp_minus_tmin_s": round(float(fname.rsplit(".", 1)[0]) - float(t.min()), 6),
            "shots_per_ring_median": int(np.median(per_ring_shots)),
            "points_per_shot_hist": {int(k): int((per_shot_counts == k).sum()) for k in np.unique(per_shot_counts)},
            "dual_return_range_gap_m": {
                "n": int(len(dual_dr)),
                "median": float(np.median(dual_dr)) if len(dual_dr) else None,
                "p10": float(np.percentile(dual_dr, 10)) if len(dual_dr) else None,
                "frac_identical_lt_1cm": float((dual_dr < 0.01).mean()) if len(dual_dr) else None,
            },
            "column_period_s_ring20": col_period,
            "azimuth_step_deg_ring20_median": float(np.median(np.diff(az0))) if len(az0) > 2 else None,
            "elevation_per_ring_deg": [round(e, 3) for e in el_per_ring],
            "range_m": [float(rng.min()), float(np.percentile(rng, 50)), float(rng.max())],
        })
    return {"scans": out}


@app.local_entrypoint()
def main():
    import json
    import pathlib

    rep = probe.remote()
    pathlib.Path("logs/stage0/probe_returns.json").write_text(json.dumps(rep, indent=2))
    for s in rep["scans"]:
        s = dict(s)
        s.pop("elevation_per_ring_deg")
        print(json.dumps(s))
    print("elevations (1er scan):", rep["scans"][0]["elevation_per_ring_deg"])
