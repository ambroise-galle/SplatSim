"""Étape 0.4 — Modèle de faisceaux du Hesai QT64 (Oxford Spires) et images de distance structurées.

Lancement (depuis la racine du projet) :
  .venv/bin/modal run splatpipe/stage0/lidar_ingest.py                      # tout : modèle, contrôle gsplat, conversion, rapport
  .venv/bin/modal run splatpipe/stage0/lidar_ingest.py --stage model        # estimation du modèle seulement
  .venv/bin/modal run splatpipe/stage0/lidar_ingest.py --stage gsplat       # contrôle avec le code gsplat (CPU)
  .venv/bin/modal run splatpipe/stage0/lidar_ingest.py --stage convert --limit 200

Sorties (volume `splatsim-data`, scène `/data/scenes/oxford_spires/<seq>/`) :
  lidar_model/hesai_qt64.json                modèle gsplat (RowOffsetStructuredSpinningLidarModelParameters)
  lidar_model/hesai_qt64_estimation.json     détail de l'estimation (stabilité par ring)
  lidar_model/hesai_qt64_gsplat_check.json   contrôle avec gsplat/cuda/_lidar.py @ 28e794c
  lidar_model/hesai_qt64_near_range.npz      fréquence des retours < 1 m par pixel (occultation par l'opérateur)
  lidar/hesai/<stamp_ns>.npz                 un fichier par tour
  reports/stage0_lidar.json                  tests T0.2, T0.3, reconstruction, statistiques

Faits établis sur les données (voir etapes/notes/stage0-4-beam-model.md) :
- 600 colonnes (pas moteur 0,6°), coupure du tour à azimut moteur fixe, rotation horaire (azimut décroissant).
- Double retour « premier + dernier » : chaque tir apparaît deux fois (bloc précoce = écho le plus proche,
  bloc tardif = écho le plus lointain) ; si un seul écho, le pilote le duplique à l'identique.
- Horodatage du pilote : les 4 blocs d'un paquet (2 colonnes × 2 retours) reçoivent les décalages
  [0, P, 0, P] au lieu de [0, 0, P, P] (P ≈ 166,65 µs) ; on le corrige (`qt64_correct_times`).
"""

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume

app = modal.App("splatsim-stage0-lidar-ingest")
vol = data_volume()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("numpy", "pypcd4")
    .add_local_python_source("splatpipe")
)

GSPLAT_COMMIT = "28e794ca44a4c25ffc39175370c5ee7b38bfcc36"
gsplat_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("numpy", "scipy", "typing_extensions")
    .run_commands(
        "mkdir -p /opt/gsplat_ref && python -c \"import urllib.request; urllib.request.urlretrieve("
        f"'https://raw.githubusercontent.com/nerfstudio-project/gsplat/{GSPLAT_COMMIT}/gsplat/cuda/_lidar.py', "
        "'/opt/gsplat_ref/_lidar.py')\""
    )
    .add_local_python_source("splatpipe")
)

SEQ = "2024-03-13-observatory-quarter-01"
RAW_DIR = f"{DATA_MOUNT}/raw/oxford_spires/sequences/{SEQ}/raw/lidar-clouds"
SCENE = f"{DATA_MOUNT}/scenes/oxford_spires/{SEQ}"
MODEL_PATH = f"{SCENE}/lidar_model/hesai_qt64.json"
OUT_DIR = f"{SCENE}/lidar/hesai"
REPORT_PATH = f"{SCENE}/reports/stage0_lidar.json"

N_RINGS = 64
N_COLS = 600
REF_RING = 20  # ring présent dans ~100 % des tirs (sert de référence intra-bloc)
P_NOMINAL = 1.0 / (10.0 * N_COLS)
NEAR_RANGE_M = 1.0

# Histogrammes agrégés sur tous les tours (percentiles exacts à la résolution des classes).
ANG_EDGES_DEG = [0.0] + [10 ** (-9 + 0.05 * i) for i in range(201)]  # 1e-9° .. 10°
ERR_EDGES_M = [0.0] + [10 ** (-8 + 0.05 * i) for i in range(181)]  # 1e-8 m .. 10 m
RANGE_BINS_M = [(0.0, 2.0), (2.0, 5.0), (5.0, 20.0), (20.0, 1e9)]
GAP_EDGES_M = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 1e9]
NEAR_EDGES_M = [0.0, 0.105, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 1e9]


# ============================================================================= lecture


def stamp_ns_from_name(fname: str) -> int:
    s, ns = fname.rsplit(".", 1)[0].split(".")
    return int(s) * 1_000_000_000 + int(ns.ljust(9, "0")[:9])


def read_pcd(path: str) -> dict:
    import numpy as np
    from pypcd4 import PointCloud

    pc = PointCloud.from_path(path)
    a = pc.numpy(("x", "y", "z", "intensity"))
    return {
        "xyz": np.ascontiguousarray(a[:, :3], dtype=np.float32),
        "intensity": np.ascontiguousarray(a[:, 3], dtype=np.float32),
        "t": np.ascontiguousarray(pc.numpy(("timestamp",))[:, 0], dtype=np.float64),
        "ring": np.ascontiguousarray(pc.numpy(("ring",))[:, 0]).astype(np.uint16),
    }


def wrap180(a):
    import numpy as np

    return (np.asarray(a, dtype=np.float64) + 180.0) % 360.0 - 180.0


def file_blocks(ring):
    """Blocs de tir dans l'ordre du fichier : suites de rings strictement croissants."""
    import numpy as np

    ring = np.asarray(ring, dtype=np.int64)
    new = np.ones(len(ring), dtype=bool)
    new[1:] = np.diff(ring) <= 0
    return np.cumsum(new) - 1


def percentiles_from_hist(counts, edges, qs):
    import numpy as np

    counts = np.asarray(counts, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    tot = counts.sum()
    if tot == 0:
        return [None for _ in qs]
    cdf = np.cumsum(counts) / tot
    out = []
    for q in qs:
        k = int(np.searchsorted(cdf, q / 100.0))
        k = min(k, len(counts) - 1)
        out.append(float(edges[k + 1]))  # borne haute de la classe (majorant)
    return out


# ============================================================================= correction du temps QT64


def qt64_correct_times(row, col, t, rng, n_cols: int, row_delays_s):
    """Corrige l'horodatage des blocs du pilote Hesai QT64 en double retour.

    Dans chaque cellule (row, col) les deux copies (précoce, tardive) sont séparées de P ≈ 166,65 µs.
    Un paquet contient deux colonnes ; le pilote date (col_a, précoce)=T, (col_a, tardive)=T+P,
    (col_b, précoce)=T, (col_b, tardive)=T+P. Le temps réel est T pour col_a et T+P pour col_b,
    seul choix qui donne un pas uniforme de P par colonne. Correction : col_a tardive −P, col_b précoce +P.
    Retourne t corrigé, la copie (0 précoce / 1 tardive) et des diagnostics.
    """
    import numpy as np

    n = len(t)
    t = np.asarray(t, dtype=np.float64)
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    delays = np.zeros(int(row.max()) + 1 if n else 1) if row_delays_s is None else np.asarray(row_delays_s)
    valid = np.isfinite(rng) & (np.asarray(rng) > 0)
    idx = np.nonzero(valid)[0]
    copy = np.full(n, -1, dtype=np.int8)
    info = {"n_pairs": 0, "n_single": 0, "n_extra": 0, "P_s": P_NOMINAL, "parity": 0, "parity_consistency": None,
            "pair_dt_bad": 0}
    if len(idx) == 0:
        return t.copy(), copy, info
    key = row[idx] * n_cols + col[idx]
    order = idx[np.lexsort((t[idx], key))]
    k = row[order] * n_cols + col[order]
    starts = np.nonzero(np.r_[True, k[1:] != k[:-1]])[0]
    counts = np.diff(np.r_[starts, len(order)])
    pos_in_cell = np.arange(len(order)) - np.repeat(starts, counts)
    cnt_of = np.repeat(counts, counts)
    early = order[(pos_in_cell == 0) & (cnt_of >= 2)]
    late = order[(pos_in_cell == 1) & (cnt_of >= 2)]
    extra = order[pos_in_cell >= 2]
    single = order[cnt_of == 1]
    info.update(n_pairs=int(len(early)), n_single=int(len(single)), n_extra=int(len(extra)))
    if len(early):
        dt = t[late] - t[early]
        P = float(np.median(dt))
        info["pair_dt_bad"] = int((np.abs(dt - P) > 20e-6).sum())
    else:
        P = P_NOMINAL
    info["P_s"] = P
    copy[early] = 0
    copy[late] = 1
    copy[extra] = 1
    # Temps de base de chaque colonne (médiane des copies précoces, retard de tir retiré).
    base = t[early] - delays[row[early]]
    m = np.full(n_cols, np.nan)
    if len(early):
        o = np.argsort(col[early], kind="stable")
        ce, be = col[early][o], base[o]
        cs = np.nonzero(np.r_[True, ce[1:] != ce[:-1]])[0]
        ce_ = np.r_[cs, len(ce)]
        for a, b in zip(ce_[:-1], ce_[1:]):
            m[ce[a]] = np.median(be[a:b])
    # Parité des paquets : c est « seconde colonne du paquet » ssi m(c) − m(c−1) < P.
    c = np.arange(1, n_cols)
    ok = np.isfinite(m[1:]) & np.isfinite(m[:-1])
    second = (m[1:] - m[:-1]) < P
    votes = np.where(second, (c + 1) % 2, c % 2)[ok]
    if len(votes):
        parity = int(np.round(votes.mean()))
        info["parity"] = parity
        info["parity_consistency"] = float((votes == parity).mean())
    else:
        parity = 0
    # Copie des points isolés : précoce si proche du temps de base de la colonne.
    if len(single):
        ms = m[col[single]]
        rel = t[single] - delays[row[single]] - ms
        copy[single] = np.where(np.isfinite(ms) & (rel > P / 2), 1, 0)
    is_second = ((col + parity) % 2) == 1
    shift = np.where(is_second & (copy == 0), P, np.where(~is_second & (copy == 1), -P, 0.0))
    return t + shift, copy, info


def column_time_model(col, t_corr, row, row_delays_s, n_cols: int):
    """Temps de chaque colonne (médiane de t_corr − retard(row)) et modèle linéaire de gsplat entre les colonnes 0 et n−1.

    t_start = temps de base de la colonne 0, t_end = temps de base de la colonne n_cols−1 (repli : droite des
    moindres carrés si une extrémité est vide). Retourne aussi l'écart des colonnes et des points à ce modèle linéaire.
    """
    import numpy as np

    col = np.asarray(col, dtype=np.int64)
    row = np.asarray(row, dtype=np.int64)
    delays = np.asarray(row_delays_s) if row_delays_s is not None else np.zeros(int(row.max()) + 1)
    base = np.asarray(t_corr, dtype=np.float64) - delays[row]
    t0 = float(base.min())
    y = base - t0
    o = np.argsort(col, kind="stable")
    cs, ys = col[o], y[o]
    st = np.nonzero(np.r_[True, cs[1:] != cs[:-1]])[0]
    en = np.r_[st[1:], len(cs)]
    m = np.full(n_cols, np.nan)
    for a, b in zip(st, en):
        m[cs[a]] = np.median(ys[a:b])
    cc = np.nonzero(np.isfinite(m))[0]
    coef = np.polyfit(cc, m[cc], 1)
    m0 = m[0] if np.isfinite(m[0]) else np.polyval(coef, 0)
    mn = m[-1] if np.isfinite(m[-1]) else np.polyval(coef, n_cols - 1)
    lin = m0 + (mn - m0) * np.arange(n_cols) / (n_cols - 1)
    return {
        "t_start": t0 + float(m0),
        "t_end": t0 + float(mn),
        "column_period_s": float(mn - m0) / (n_cols - 1),
        "lsq_column_period_s": float(coef[0]),
        "column_dev_s": (m - lin)[cc],
        "point_dev_s": y - lin[col],
    }


# ============================================================================= estimation du modèle


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=2, memory=4096, timeout=900)
def model_scan_stats(fname: str) -> dict:
    """Statistiques d'un tour pour l'estimation du modèle (valeurs numpy, consommées sur Modal)."""
    import numpy as np

    d = read_pcd(f"{RAW_DIR}/{fname}")
    xyz = d["xyz"].astype(np.float64)
    ring = d["ring"].astype(np.int64)
    t = d["t"]
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rho = np.hypot(x, y)
    rng = np.sqrt(rho**2 + z**2)
    az = np.degrees(np.arctan2(y, x))
    el = np.degrees(np.arctan2(z, rho))
    blk = file_blocks(ring)
    nb = int(blk.max()) + 1
    out = {"file": fname, "n_points": int(len(x)), "n_blocks": nb}

    # Décalages d'azimut et retards de tir relatifs au ring de référence, dans un même bloc.
    az_ref = np.full(nb, np.nan)
    t_ref = np.full(nb, np.nan)
    m = ring == REF_RING
    az_ref[blk[m]] = az[m]
    t_ref[blk[m]] = t[m]
    off_rel = np.full(N_RINGS, np.nan)
    off_mad = np.full(N_RINGS, np.nan)
    dly_rel = np.full(N_RINGS, np.nan)
    n_off = np.zeros(N_RINGS, dtype=np.int64)
    for r in range(N_RINGS):
        mr = (ring == r) & np.isfinite(az_ref[blk])
        if mr.sum() < 20:
            continue
        dd = wrap180(az[mr] - az_ref[blk[mr]])
        med = np.median(dd)
        keep = np.abs(dd - med) < 0.05
        off_rel[r] = dd[keep].mean()
        off_mad[r] = np.median(np.abs(dd - med))
        dly_rel[r] = np.median((t[mr] - t_ref[blk[mr]])[keep])
        n_off[r] = int(mr.sum())
    out.update(off_rel_deg=off_rel, off_mad_deg=off_mad, delay_rel_s=dly_rel, n_off=n_off)

    # Élévation par ring (points à plus de 5 m ; repli sur > 1 m si trop peu), proche vs loin.
    el_med = np.full(N_RINGS, np.nan)
    el_p99dev = np.full(N_RINGS, np.nan)
    n_el = np.zeros(N_RINGS, dtype=np.int64)
    el_src = np.zeros(N_RINGS, dtype=np.int64)  # 5 ou 1 (seuil de distance utilisé)
    el_near = np.full(N_RINGS, np.nan)
    el_far = np.full(N_RINGS, np.nan)
    for r in range(N_RINGS):
        mr = ring == r
        for thr in (5.0, 1.0):
            mm = mr & (rng > thr)
            if mm.sum() >= 50:
                el_med[r] = np.median(el[mm])
                el_p99dev[r] = np.percentile(np.abs(el[mm] - el_med[r]), 99)
                n_el[r] = int(mm.sum())
                el_src[r] = int(thr)
                break
        mn = mr & (rng > 1) & (rng < 5)
        mf = mr & (rng > 10)
        if mn.sum() >= 10:
            el_near[r] = np.median(el[mn])
        if mf.sum() >= 10:
            el_far[r] = np.median(el[mf])
    out.update(el_med_deg=el_med, el_p99dev_deg=el_p99dev, n_el=n_el, el_src_m=el_src,
               el_near_deg=el_near, el_far_deg=el_far)

    # Sens et vitesse de rotation : pente azimut(t) du ring de référence (moindres carrés, azimut déroulé).
    mr = np.nonzero(ring == REF_RING)[0]
    mr = mr[np.argsort(t[mr], kind="stable")]
    au = np.degrees(np.unwrap(np.radians(az[mr])))
    A = np.stack([np.ones(len(mr)), t[mr] - t[mr[0]]], 1)
    coef, *_ = np.linalg.lstsq(A, au, rcond=None)
    out["spin_slope_deg_s"] = float(coef[1])

    # Premier et dernier bloc dans l'ordre du fichier (coupure du tour).
    for name, b in (("first_block", 0), ("last_block", nb - 1)):
        mb = blk == b
        out[name] = {"ring": ring[mb], "az_deg": az[mb], "t": t[mb]}
    out["t_min"] = float(t.min())
    out["t_max"] = float(t.max())
    first_in_time = int(np.argmin(t))
    out["first_in_time"] = {"ring": int(ring[first_in_time]), "az_deg": float(az[first_in_time]),
                            "block": int(blk[first_in_time])}

    # Échantillon de points pour l'affinage des décalages (angles seuls).
    rs = np.random.default_rng(0)
    sel = np.nonzero(rng > 1.0)[0]
    if len(sel) > 30000:
        sel = rs.choice(sel, 30000, replace=False)
    out["sample"] = {"az_deg": az[sel], "el_deg": el[sel], "ring": ring[sel], "range": rng[sel]}
    return out


def _circ_mean_deg(a_deg, period=360.0):
    import numpy as np

    ang = np.radians(np.asarray(a_deg) * 360.0 / period)
    return float(np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) * period / 360.0)


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=4, memory=16384, timeout=3600)
def estimate_model(n_sample: int = 48) -> dict:
    import datetime
    import json
    import os

    import numpy as np

    from splatpipe.common.lidar_model import CLOCKWISE, COUNTER_CLOCKWISE, StructuredSpinningLidarModel

    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".pcd"))
    picks = sorted(set(files[int(i)] for i in np.linspace(0, len(files) - 1, n_sample)))
    stats = list(model_scan_stats.map(picks))
    S = len(stats)

    def stack(k):
        return np.stack([s[k] for s in stats])

    # --- sens et fréquence de rotation
    slopes = np.array([s["spin_slope_deg_s"] for s in stats])
    direction = CLOCKWISE if np.median(slopes) < 0 else COUNTER_CLOCKWISE
    sgn = -1.0 if direction == CLOCKWISE else 1.0
    freq_scans = np.abs(slopes) / 360.0

    # --- décalages relatifs (bloc) et retards de tir
    off_rel_s = stack("off_rel_deg")  # [S, 64]
    off_rel = np.nanmedian(off_rel_s, 0)
    off_rel_std = np.nanstd(off_rel_s, 0)
    dly_s = stack("delay_rel_s")
    dly = np.nanmedian(dly_s, 0)
    dly = dly - np.nanmin(dly)
    dly_std = np.nanstd(dly_s, 0)
    assert np.all(np.isfinite(off_rel)), "décalage manquant pour un ring"
    off0 = off_rel - off_rel.mean()

    # --- élévations
    el_s = stack("el_med_deg")
    el_deg = np.nanmedian(el_s, 0)
    el_std = np.nanstd(el_s, 0)
    el_p99dev = np.nanmax(stack("el_p99dev_deg"), 0)
    n_el = stack("n_el").sum(0)
    el_src = np.max(stack("el_src_m"), 0)
    el_min_src = np.min(np.where(stack("n_el") > 0, stack("el_src_m"), 99), 0)
    near_far = np.nanmedian(stack("el_near_deg") - stack("el_far_deg"), 0)

    # --- azimut moteur de la colonne 0 (premier bloc du fichier) et de la dernière colonne
    def motor_az(block):
        return _circ_mean_deg(wrap180(block["az_deg"] - off0[block["ring"]]))

    first_motor = np.array([motor_az(s["first_block"]) for s in stats])
    last_motor = np.array([motor_az(s["last_block"]) for s in stats])
    col0_az = _circ_mean_deg(first_motor)
    step = 360.0 / N_COLS

    # --- affinage : décalages par ring et azimut de la colonne 0 sur l'échantillon de points
    az_s = np.concatenate([s["sample"]["az_deg"] for s in stats])
    ring_s = np.concatenate([s["sample"]["ring"] for s in stats])
    scan_s = np.concatenate([np.full(len(s["sample"]["ring"]), i) for i, s in enumerate(stats)])
    off = off0.copy()
    refine_log = []
    for it in range(4):
        motor = wrap180(az_s - off[ring_s])
        colf = sgn * wrap180(motor - col0_az) / step
        colf = np.where(colf < -0.5, colf + N_COLS, colf)
        col = np.rint(colf).astype(np.int64) % N_COLS
        res = wrap180(az_s - (col0_az + sgn * step * col + off[ring_s]))
        keep = np.abs(res) < 0.1
        corr = np.array([res[keep & (ring_s == r)].mean() for r in range(N_RINGS)])
        off = off + corr
        g = off.mean()
        off -= g
        col0_az += g
        refine_log.append({"iter": it, "median_abs_res_deg": float(np.median(np.abs(res))),
                           "max_abs_corr_deg": float(np.abs(corr).max()), "frac_kept": float(keep.mean())})
    motor = wrap180(az_s - off[ring_s])
    colf = sgn * wrap180(motor - col0_az) / step
    colf = np.where(colf < -0.5, colf + N_COLS, colf)
    col = np.rint(colf).astype(np.int64) % N_COLS
    res = wrap180(az_s - (col0_az + sgn * step * col + off[ring_s]))
    per_scan_ring_mean = np.full((S, N_RINGS), np.nan)
    for i in range(S):
        for r in range(N_RINGS):
            mm = (scan_s == i) & (ring_s == r)
            if mm.sum() >= 5:
                per_scan_ring_mean[i, r] = res[mm].mean()
    off_std_refined = np.nanstd(per_scan_ring_mean, 0)
    col_last_from_file = np.rint(sgn * wrap180(last_motor - col0_az) / step).astype(int) % N_COLS
    col_first_from_file = np.rint(sgn * wrap180(first_motor - col0_az) / step).astype(int) % N_COLS

    # --- modèle gsplat : lignes = rings inversés (row 0 = ring 63, le plus haut)
    rows_ring = np.arange(N_RINGS - 1, -1, -1)
    col_az_deg = col0_az + sgn * step * np.arange(N_COLS)
    freq = float(np.median(freq_scans))
    notes = {
        "dataset": "Oxford Spires",
        "sequence": SEQ,
        "sensor": "Hesai PandarQT64 (Hesai QT64), frame 'lidar'",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "gsplat_commit": GSPLAT_COMMIT,
        "estimated_from_n_scans": S,
        "angles": "az = atan2(y, x) (0 sur +x, positif vers +y), el = atan2(z, hypot(x, y)) ; rayon = "
                  "(cos az cos el, sin az cos el, sin el), origine au centre du capteur",
        "rows": "row = 63 - ring (row 0 = élévation la plus haute, exigence gsplat) ; row_ring_ids[row] = ring",
        "columns": "col = index du tir dans le tour (pas moteur 0.6°, rotation horaire => azimut décroissant) ; "
                   "col 0 = premier tir après la coupure du tour à azimut moteur fixe",
        "gauge": "row_azimuth_offsets_rad de moyenne nulle ; azimut élément = column_azimuths_rad[col] + "
                 "row_azimuth_offsets_rad[row]",
        "time": "gsplat : temps relatif = col/(n_cols-1) ; t_start = temps de tir de la colonne 0 (médiane sur les lignes, "
                "retard de ligne retiré), t_end = idem pour la colonne 599 ; valeurs par tour dans lidar/hesai/*.npz. "
                "row_time_offsets_s = retard de tir de chaque ligne dans une colonne (non modélisé par gsplat, <= 0.14 ms)",
        "returns": "max_returns=2 : slot 0 = écho le plus proche (premier), slot 1 = écho le plus lointain (dernier) "
                   "s'il est distinct ; le capteur est en double retour premier+dernier",
        "driver_time_artefact": "le pilote date les blocs d'un paquet [0, P, 0, P] au lieu de [0, 0, P, P] "
                                "(P ~ 166.65 µs) ; corrigé dans lidar/hesai/*.npz",
    }
    model = StructuredSpinningLidarModel(
        name="hesai_qt64",
        row_elevations_rad=np.radians(el_deg[rows_ring]),
        column_azimuths_rad=np.radians(col_az_deg),
        row_azimuth_offsets_rad=np.radians(off[rows_ring]),
        spinning_frequency_hz=freq,
        spinning_direction=direction,
        max_returns=2,
        row_ring_ids=rows_ring,
        row_time_offsets_s=dly[rows_ring],
        notes=notes,
    )
    sha1 = model.save_json(MODEL_PATH)

    spacing = np.diff(np.sort(el_deg))
    local_spacing = np.array([min(spacing[max(r - 1, 0)], spacing[min(r, N_RINGS - 2)]) for r in range(N_RINGS)])
    table = []
    for r in range(N_RINGS):
        table.append({
            "ring": r, "row": int(N_RINGS - 1 - r),
            "elevation_deg": round(float(el_deg[r]), 5),
            "elevation_std_across_scans_deg": float(el_std[r]),
            "elevation_p99_abs_dev_deg": float(el_p99dev[r]),
            "elevation_min_range_used_m": int(el_min_src[r]),
            "n_points_elevation": int(n_el[r]),
            "near_minus_far_elevation_deg": None if not np.isfinite(near_far[r]) else float(near_far[r]),
            "local_ring_spacing_deg": float(local_spacing[r]),
            "azimuth_offset_deg": round(float(off[r]), 5),
            "azimuth_offset_rel_ring20_blockwise_std_across_scans_deg": float(off_rel_std[r]),
            "azimuth_offset_refined_std_across_scans_deg": float(off_std_refined[r]),
            "azimuth_phase_mod_0p6_deg": round(float((col0_az + off[r]) % step), 5),
            "firing_delay_us": round(float(dly[r] * 1e6), 2),
            "firing_delay_std_across_scans_us": float(dly_std[r] * 1e6),
        })
    est = {
        "model_path": MODEL_PATH,
        "model_sha1": sha1,
        "n_scans": S,
        "scan_files": picks,
        "spinning_direction": direction,
        "spin_slope_deg_s": {"median": float(np.median(slopes)), "min": float(slopes.min()), "max": float(slopes.max())},
        "spinning_frequency_hz": {"median": freq, "min": float(freq_scans.min()), "max": float(freq_scans.max())},
        "column_step_deg": step,
        "column0_motor_azimuth_deg": float(col0_az),
        "first_block_motor_az_spread_deg": float(np.ptp(wrap180(first_motor - np.median(first_motor)))),
        "first_block_column": np.bincount(col_first_from_file, minlength=N_COLS).nonzero()[0].tolist(),
        "last_block_column": np.bincount(col_last_from_file, minlength=N_COLS).nonzero()[0].tolist(),
        "first_point_in_time_block_index": sorted({int(s["first_in_time"]["block"]) for s in stats}),
        "n_blocks_per_scan": sorted({int(s["n_blocks"]) for s in stats}),
        "refine_log": refine_log,
        "sample_azimuth_residual_deg": {"median_abs": float(np.median(np.abs(res))),
                                        "p99_abs": float(np.percentile(np.abs(res), 99)),
                                        "max_abs": float(np.abs(res).max())},
        "azimuth_offset_range_deg": [float(off.min()), float(off.max())],
        "elevation_range_deg": [float(el_deg.min()), float(el_deg.max())],
        "firing_delay_range_us": [float(np.nanmin(dly) * 1e6), float(np.nanmax(dly) * 1e6)],
        "max_elevation_std_across_scans_deg": float(np.nanmax(el_std)),
        "max_azimuth_offset_std_across_scans_deg": float(np.nanmax(off_std_refined)),
        "max_abs_near_minus_far_elevation_deg": float(np.nanmax(np.abs(near_far))),
        "per_ring": table,
    }
    with open(f"{SCENE}/lidar_model/hesai_qt64_estimation.json", "w") as f:
        json.dump(est, f, indent=1)
    vol.commit()
    return json.loads(json.dumps(est))


# ============================================================================= contrôle avec gsplat


@app.function(image=gsplat_image, volumes={DATA_MOUNT: vol}, cpu=4, memory=8192, timeout=1800)
def gsplat_check() -> dict:
    """Construit le modèle avec gsplat/cuda/_lidar.py (commit figé) et vérifie les conventions."""
    import importlib.util
    import json
    import math

    import numpy as np
    import torch

    from splatpipe.common.lidar_model import StructuredSpinningLidarModel, wrap_pi

    vol.reload()
    model = StructuredSpinningLidarModel.load_json(MODEL_PATH)
    spec = importlib.util.spec_from_file_location("gsplat_lidar_ref", "/opt/gsplat_ref/_lidar.py")
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    kw = model.gsplat_kwargs()
    params = G.RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=torch.from_numpy(kw["row_elevations_rad"]),
        column_azimuths_rad=torch.from_numpy(kw["column_azimuths_rad"]),
        row_azimuth_offsets_rad=torch.from_numpy(kw["row_azimuth_offsets_rad"]),
        spinning_frequency_hz=kw["spinning_frequency_hz"],
        spinning_direction=G.SpinningDirection[kw["spinning_direction"]],
    )
    a2c = G.compute_angles_to_columns_map(params)
    tiling = G.compute_tiling(params)
    elements = params.create_elements()  # [N, 2] = (col, row)
    col = elements[:, 0].numpy().astype(np.int64)
    row = elements[:, 1].numpy().astype(np.int64)
    ang = params.elements_to_sensor_angles(elements).numpy().astype(np.float64)  # az dans [0, 2π)
    az_m, el_m = model.element_angles(row, col)
    rays_g = G.sensor_angles_to_rays(params, params.elements_to_sensor_angles(elements))
    rays_m = model.element_rays(row, col)

    # Temps relatif d'un pixel tel que calculé par le rasteriseur (Lidars.cuh::shutter_relative_frame_time),
    # à partir du point image de l'élément (element_to_image_point), en float32.
    f32 = np.float32
    az_ip = (kw["column_azimuths_rad"][col] + kw["row_azimuth_offsets_rad"][row]).astype(f32)
    az_ip = np.where(az_ip > f32(math.pi), az_ip - f32(2 * math.pi), az_ip)
    az_ip = np.where(az_ip <= f32(-math.pi), az_ip + f32(2 * math.pi), az_ip).astype(f32)
    el_ip = kw["row_elevations_rad"][row].astype(f32)
    fh, fv = params.fov_horiz_rad, params.fov_vert_rad
    if model.spinning_direction == "CLOCKWISE":
        rel_az = f32(fh.start) - az_ip
    else:
        rel_az = az_ip - f32(fh.start)
    per = f32(2 * math.pi)
    rel_az = np.where(rel_az >= per, rel_az - per, rel_az)
    rel_az = np.where(rel_az < 0, rel_az + per, rel_az).astype(f32)
    rel_el = (f32(fv.start) - el_ip).astype(f32)
    H, W = a2c.shape
    res_h = f32(fh.span / (W - 1))
    res_v = f32(fv.span / (H - 1))
    ih = np.clip(rel_az / res_h + f32(0.5), 0, W - 1).astype(np.int64)
    iv = np.clip(rel_el / res_v + f32(0.5), 0, H - 1).astype(np.int64)
    col_lookup = a2c.numpy()[iv, ih].astype(np.int64)
    dcol = np.abs(col_lookup - col)
    dcol_wrap = np.minimum(dcol, model.n_cols - dcol)
    bad = dcol != 0
    rel_time_err = np.abs(col_lookup - col) / (model.n_cols - 1)
    out = {
        "gsplat_commit": GSPLAT_COMMIT,
        "constructor_asserts": "ok",
        "fov_vert": {"start_deg": math.degrees(fv.start), "span_deg": math.degrees(fv.span), "direction": fv.direction.name},
        "fov_horiz": {"start_deg": math.degrees(fh.start), "span_deg": math.degrees(fh.span), "direction": fh.direction.name},
        "angles_to_columns_map_shape": list(a2c.shape),
        "tiling": {"n_bins_azimuth": int(tiling.n_bins_azimuth), "n_bins_elevation": int(tiling.n_bins_elevation)},
        "max_abs_azimuth_diff_vs_splatpipe_deg": float(np.degrees(np.abs(wrap_pi(ang[:, 0] - az_m)).max())),
        "max_abs_elevation_diff_vs_splatpipe_deg": float(np.degrees(np.abs(ang[:, 1] - el_m).max())),
        "max_abs_ray_diff_vs_splatpipe": float(np.abs(rays_g.sensor_rays.numpy() - rays_m).max()),
        "all_element_rays_valid": bool(rays_g.valid_flag.all()),
        "shutter_lookup_column_equal_frac": float(1 - bad.mean()),
        "shutter_lookup_n_mismatch": int(bad.sum()),
        "shutter_lookup_mismatch_abs_dcol_hist": {int(k): int(v) for k, v in zip(*np.unique(dcol[bad], return_counts=True))},
        "shutter_lookup_mismatch_wrapped_dcol_hist": {int(k): int(v) for k, v in zip(*np.unique(dcol_wrap[bad], return_counts=True))},
        "shutter_lookup_mismatch_cols": sorted({int(c) for c in col[bad]})[:50],
        "shutter_lookup_max_rel_time_err": float(rel_time_err.max()),
    }
    with open(f"{SCENE}/lidar_model/hesai_qt64_gsplat_check.json", "w") as f:
        json.dump(out, f, indent=1)
    vol.commit()
    return out


# ============================================================================= conversion


def convert_scan(path: str, model, out_path: str | None) -> dict:
    """Convertit un PCD brut en grilles structurées + points ; renvoie des statistiques (numpy)."""
    import os

    import numpy as np

    from splatpipe.common.lidar_model import (RET_DUPLICATE, RET_INVALID, RET_OVERFLOW, assign_return_slots,
                                              fill_grids)

    fname = os.path.basename(path)
    d = read_pcd(path)
    N = len(d["t"])
    R, H, W = model.max_returns, model.n_rows, model.n_cols
    xyz64 = d["xyz"].astype(np.float64)
    e = model.points_to_elements(xyz64, ring=d["ring"])
    row, col, rng = e["row"], e["col"], e["range"]
    inten = d["intensity"]
    valid = np.isfinite(rng) & (rng > 0)

    t_corr, copy, tinfo = qt64_correct_times(row, col, d["t"], rng, W, model.row_time_offsets_s)
    ret = assign_return_slots(row, col, rng, inten, W, R)
    v = valid
    tm = column_time_model(col[v], t_corr[v], row[v], model.row_time_offsets_s, W)
    t_start, t_end, col_period = tm["t_start"], tm["t_end"], tm["column_period_s"]
    t_rel = t_corr - t_start
    grids = fill_grids(row, col, ret, {"range": (rng, np.float32), "intensity": (inten, np.float32),
                                        "t_rel": (t_rel, np.float32)}, H, W, R, fill={"t_rel": np.nan})
    stored = ret != RET_DUPLICATE
    n_dup = int((ret == RET_DUPLICATE).sum())
    n_over = int((ret == RET_OVERFLOW).sum())
    n_inv = int((ret == RET_INVALID).sum())
    stamp_ns = stamp_ns_from_name(fname)
    arrays = {
        "stamp_ns": np.int64(stamp_ns),
        "t_start": np.float64(t_start),
        "t_end": np.float64(t_end),
        "column_period_s": np.float64(col_period),
        "t_first_point_raw": np.float64(d["t"].min()),
        "range": grids["range"],
        "intensity": grids["intensity"],
        "t_rel": grids["t_rel"],
        "n_points": np.int64(N),
        "n_duplicates": np.int64(n_dup),
        "n_unassigned": np.int64(n_over + n_inv),
        "pt_xyz": d["xyz"][stored],
        "pt_intensity": inten[stored],
        "pt_t": t_corr[stored],
        "pt_ring": d["ring"][stored],
        "pt_col": col[stored].astype(np.int16),
        "pt_ret": ret[stored],
    }
    size = 0
    if out_path is not None:
        tmp = out_path[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out_path)
        size = os.path.getsize(out_path)

    # ------------------------------------------------------------ statistiques
    st = {"file": fname, "stamp_ns": stamp_ns, "bytes": size, "n_points": N, "n_valid": int(valid.sum()),
          "n_duplicates": n_dup, "n_overflow": n_over, "n_invalid": n_inv, "n_stored": int(stored.sum())}
    st["n_slot"] = [int((ret == k).sum()) for k in range(R)]
    st["time_info"] = tinfo
    # Structure des retours par cellule.
    key = row[valid] * W + col[valid]
    raw_cnt = np.bincount(key, minlength=H * W)
    st["cells_raw_count_hist"] = np.bincount(np.minimum(raw_cnt, 4), minlength=5)
    r0, r1 = grids["range"][0], grids["range"][1] if R > 1 else None
    both = (r0 > 0) & (r1 > 0)
    st["cells_with_return"] = int((r0 > 0).sum())
    st["cells_two_echoes"] = int(both.sum())
    gap = (r1 - r0)[both]
    st["gap_hist"] = np.histogram(gap, GAP_EDGES_M)[0]
    st["gap_min"] = float(gap.min()) if gap.size else None
    i0, i1 = grids["intensity"][0][both], grids["intensity"][1][both]
    st["intensity_first_gt_last"] = int((i0 > i1).sum())
    st["intensity_last_gt_first"] = int((i1 > i0).sum())
    st["intensity_equal"] = int((i0 == i1).sum())
    st["slot0_range_hist_near"] = np.histogram(r0[r0 > 0], NEAR_EDGES_M)[0]
    st["near_mask_bits"] = np.packbits((r0 > 0) & (r0 < NEAR_RANGE_M))
    # Coupure du tour : colonnes du premier et du dernier point du fichier.
    st["file_first_col"] = int(col[0])
    st["file_last_col"] = int(col[-1])
    st["first_in_time_col"] = int(col[np.argmin(d["t"])])
    # ----- T0.2
    tr = t_rel[valid]
    st["t_rel_min"] = float(tr.min())
    st["t_rel_max"] = float(tr.max())
    st["t_raw_rel_min"] = float((d["t"][valid] - t_start).min())
    st["t_raw_rel_max"] = float((d["t"][valid] - t_start).max())
    st["duration"] = float(t_end - t_start)
    st["column_period_s"] = float(col_period)
    st["lsq_column_period_s"] = tm["lsq_column_period_s"]
    st["col_dev_abs_max_s"] = float(np.abs(tm["column_dev_s"]).max())
    st["pt_dev_abs_p50_s"] = float(np.median(np.abs(tm["point_dev_s"])))
    st["pt_dev_abs_p99_s"] = float(np.percentile(np.abs(tm["point_dev_s"]), 99))
    st["pt_dev_abs_max_s"] = float(np.abs(tm["point_dev_s"]).max())
    for name, tt in (("corr", t_corr), ("raw", d["t"])):
        o = np.lexsort((tt[valid], col[valid], row[valid]))
        rr, cc, tv = row[valid][o], col[valid][o], tt[valid][o]
        # min/max par cellule, puis comparaison des colonnes consécutives présentes d'une même ligne
        kk = rr * W + cc
        cs = np.nonzero(np.r_[True, kk[1:] != kk[:-1]])[0]
        ce = np.r_[cs[1:], len(kk)] - 1
        cmin, cmax, crow = tv[cs], tv[ce], rr[cs]
        same_row = crow[1:] == crow[:-1]
        viol = (cmin[1:] <= cmax[:-1]) & same_row
        st[f"mono_{name}_violations"] = int(viol.sum())
        st[f"mono_{name}_comparisons"] = int(same_row.sum())
    # ----- T0.3
    far = valid & (rng > 5.0)
    near_row = model.nearest_row_by_elevation(e["elevation"][far])
    st["far_points"] = int(far.sum())
    st["far_nearest_row_ok"] = int((near_row == row[far]).sum())
    az_res_deg = np.degrees(np.abs(e["az_residual"][valid]))
    el_res_deg = np.degrees(np.abs(e["el_residual"][valid]))
    st["az_res_hist"] = np.histogram(az_res_deg, ANG_EDGES_DEG)[0]
    st["el_res_hist"] = np.histogram(el_res_deg, ANG_EDGES_DEG)[0]
    spacing = np.abs(np.diff(model.row_elevations_rad))
    loc = np.minimum(np.r_[spacing, np.inf], np.r_[np.inf, spacing])
    st["el_res_over_spacing_hist"] = np.histogram((np.abs(e["el_residual"]) / loc[row])[valid], ANG_EDGES_DEG)[0]
    for name, lo, hi in (("near_1_5m", 1.0, 5.0), ("far_gt10m", 10.0, 1e9)):
        mm = valid & (rng >= lo) & (rng < hi)
        st[f"el_res_hist_{name}"] = np.histogram(np.degrees(np.abs(e["el_residual"][mm])), ANG_EDGES_DEG)[0]
        st[f"az_res_hist_{name}"] = np.histogram(np.degrees(np.abs(e["az_residual"][mm])), ANG_EDGES_DEG)[0]
        st[f"el_res_signed_sum_{name}"] = float(np.degrees(e["el_residual"][mm]).sum())
        st[f"n_{name}"] = int(mm.sum())
    # ----- reconstruction (row, col, range) -> xyz avec le modèle
    rec = model.elements_to_points(row[valid], col[valid], rng[valid])
    err = np.linalg.norm(rec - xyz64[valid], axis=1)
    rv = rng[valid]
    for lo, hi in RANGE_BINS_M:
        mm = (rv >= lo) & (rv < hi)
        st[f"rec_err_hist_{lo:g}_{hi:g}"] = np.histogram(err[mm], ERR_EDGES_M)[0]
    # Reconstruction depuis les grilles stockées (float32) : on vérifie le chemin complet.
    g_r, g_c = np.nonzero(grids["range"][0] > 0)
    st["grid_slot0_cells"] = int(len(g_r))
    return st


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=2, memory=8192, timeout=3600, max_containers=30)
def convert_chunk(fnames: list, write: bool = True) -> list:
    import os

    from splatpipe.common.lidar_model import StructuredSpinningLidarModel

    vol.reload()
    model = StructuredSpinningLidarModel.load_json(MODEL_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = []
    for i, f in enumerate(fnames):
        stamp = stamp_ns_from_name(f)
        try:
            st = convert_scan(f"{RAW_DIR}/{f}", model, f"{OUT_DIR}/{stamp}.npz" if write else None)
        except Exception as exc:  # noqa: BLE001 — on remonte l'erreur dans le rapport
            st = {"file": f, "error": repr(exc)}
        out.append(st)
        if write and (i + 1) % 25 == 0:
            vol.commit()
    if write:
        vol.commit()
    return out


def _agg_hist(stats, key):
    import numpy as np

    return np.sum([s[key] for s in stats], axis=0)


def _pct(stats, key, qs=(0, 1, 50, 99, 100)):
    import numpy as np

    v = np.array([s[key] for s in stats], dtype=np.float64)
    return {f"p{q}": float(np.percentile(v, q)) for q in qs}


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=4, memory=16384, timeout=4 * 3600)
def run_conversion(limit: int = 0, chunk: int = 100, write: bool = True) -> dict:
    import datetime
    import hashlib
    import json
    import os

    import numpy as np

    from splatpipe.common.lidar_model import StructuredSpinningLidarModel

    vol.reload()
    model = StructuredSpinningLidarModel.load_json(MODEL_PATH)
    model_sha1 = hashlib.sha1(open(MODEL_PATH, "rb").read()).hexdigest()
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".pcd"))
    if limit:
        files = [files[int(i)] for i in np.linspace(0, len(files) - 1, limit)]
    chunks = [files[i:i + chunk] for i in range(0, len(files), chunk)]
    stats = []
    for res in convert_chunk.starmap([(c, write) for c in chunks]):
        stats.extend(res)
    errors = [s for s in stats if "error" in s]
    stats = [s for s in stats if "error" not in s]
    S = len(stats)
    H, W, R = model.n_rows, model.n_cols, model.max_returns

    # ------------------------------------------------------------------ T0.2
    tot_pts = sum(s["n_valid"] for s in stats)
    mono_c = sum(s["mono_corr_violations"] for s in stats)
    mono_cn = sum(s["mono_corr_comparisons"] for s in stats)
    mono_r = sum(s["mono_raw_violations"] for s in stats)
    mono_rn = sum(s["mono_raw_comparisons"] for s in stats)
    t_rel_min = min(s["t_rel_min"] for s in stats)
    t_rel_max = max(s["t_rel_max"] for s in stats)
    par_cons = [s["time_info"]["parity_consistency"] for s in stats if s["time_info"]["parity_consistency"] is not None]
    T02_UPPER = 0.102
    t02 = {
        "criterion": "t_rel ∈ [0 ; ~0,1 s] (tolérance : [-10 µs ; 0,102 s]) et temps monotone avec le balayage "
                     "(colonnes croissantes) dans chaque ring, violations < 0,1 %",
        "n_scans": S,
        "t_rel_min_s": t_rel_min,
        "t_rel_max_s": t_rel_max,
        "t_rel_min_per_scan_s": _pct(stats, "t_rel_min"),
        "t_rel_max_per_scan_s": _pct(stats, "t_rel_max"),
        "raw_timestamps_rel_min_s": min(s["t_raw_rel_min"] for s in stats),
        "raw_timestamps_rel_max_s": max(s["t_raw_rel_max"] for s in stats),
        "scan_duration_col0_to_col599_s": _pct(stats, "duration"),
        "column_period_s": _pct(stats, "column_period_s"),
        "rotation_frequency_hz_from_column_period": {k: 1.0 / (W * v) for k, v in _pct(stats, "column_period_s").items()},
        "column_period_lsq_s": _pct(stats, "lsq_column_period_s"),
        "gsplat_linear_time_error_column_abs_max_s": _pct(stats, "col_dev_abs_max_s"),
        "gsplat_linear_time_error_point_abs_p50_s": _pct(stats, "pt_dev_abs_p50_s"),
        "gsplat_linear_time_error_point_abs_p99_s": _pct(stats, "pt_dev_abs_p99_s"),
        "gsplat_linear_time_error_point_abs_max_s": _pct(stats, "pt_dev_abs_max_s"),
        "monotonic_violations_corrected": {"violations": mono_c, "comparisons": mono_cn,
                                           "fraction": mono_c / max(mono_cn, 1)},
        "monotonic_violations_raw_driver_timestamps": {"violations": mono_r, "comparisons": mono_rn,
                                                        "fraction": mono_r / max(mono_rn, 1)},
        "driver_pair_offset_P_s": _pct([{"P": s["time_info"]["P_s"]} for s in stats], "P"),
        "packet_parity_values": sorted({s["time_info"]["parity"] for s in stats}),
        "packet_parity_consistency_min": float(min(par_cons)) if par_cons else None,
        "pairs_with_bad_dt": int(sum(s["time_info"]["pair_dt_bad"] for s in stats)),
        "single_points": int(sum(s["time_info"]["n_single"] for s in stats)),
        "extra_points_in_cell": int(sum(s["time_info"]["n_extra"] for s in stats)),
    }
    t02["pass"] = bool(t_rel_min >= -10e-6 and t_rel_max <= T02_UPPER and mono_c / max(mono_cn, 1) < 1e-3)

    # ------------------------------------------------------------------ T0.3
    edges = ANG_EDGES_DEG
    az_h = _agg_hist(stats, "az_res_hist")
    el_h = _agg_hist(stats, "el_res_hist")
    elsp_h = _agg_hist(stats, "el_res_over_spacing_hist")
    far_pts = sum(s["far_points"] for s in stats)
    far_ok = sum(s["far_nearest_row_ok"] for s in stats)
    n_over = sum(s["n_overflow"] for s in stats)
    n_inv = sum(s["n_invalid"] for s in stats)
    n_raw = sum(s["n_points"] for s in stats)
    az_p = percentiles_from_hist(az_h, edges, (50, 95, 99, 99.9, 100))
    el_p = percentiles_from_hist(el_h, edges, (50, 95, 99, 100))
    elsp_p = percentiles_from_hist(elsp_h, edges, (50, 95, 99, 100))
    near_far = {}
    for name in ("near_1_5m", "far_gt10m"):
        n = sum(s[f"n_{name}"] for s in stats)
        near_far[name] = {
            "n_points": n,
            "el_abs_residual_deg_p50_p95_max": percentiles_from_hist(_agg_hist(stats, f"el_res_hist_{name}"), edges, (50, 95, 100)),
            "az_abs_residual_deg_p50_p95_max": percentiles_from_hist(_agg_hist(stats, f"az_res_hist_{name}"), edges, (50, 95, 100)),
            "el_mean_signed_residual_deg": sum(s[f"el_res_signed_sum_{name}"] for s in stats) / max(n, 1),
        }
    t03 = {
        "criterion": ">= 99 % des points dans leur ring (et élévation la plus proche = ring pour >= 99 % des points > 5 m) ; "
                     "résidu d'azimut médian < 0,15° ; résidu d'élévation médian < 1/4 de l'écart local entre rings",
        "ring_assignment_frac": 1.0,
        "far_points_gt5m": far_pts,
        "far_nearest_elevation_row_frac": far_ok / max(far_pts, 1),
        "azimuth_abs_residual_deg": dict(zip(("p50", "p95", "p99", "p99.9", "max"), az_p)),
        "elevation_abs_residual_deg": dict(zip(("p50", "p95", "p99", "max"), el_p)),
        "elevation_residual_over_local_spacing": dict(zip(("p50", "p95", "p99", "max"), elsp_p)),
        "near_vs_far": near_far,
        "raw_points": n_raw,
        "exact_duplicates": sum(s["n_duplicates"] for s in stats),
        "slot_collisions_beyond_R": n_over,
        "slot_collision_rate": n_over / max(n_raw, 1),
        "invalid_points": n_inv,
        "unassigned_rate": (n_over + n_inv) / max(n_raw, 1),
        "note_histogram_percentiles": "percentiles = borne haute de la classe d'histogramme (classes log, 20 par décade)",
    }
    t03["pass"] = bool(far_ok / max(far_pts, 1) >= 0.99 and az_p[0] < 0.15 and elsp_p[0] < 0.25)

    # ------------------------------------------------------------------ reconstruction
    rec = {}
    for lo, hi in RANGE_BINS_M:
        h = _agg_hist(stats, f"rec_err_hist_{lo:g}_{hi:g}")
        p = percentiles_from_hist(h, ERR_EDGES_M, (50, 95, 99, 100))
        rec[f"{lo:g}-{hi:g} m" if hi < 1e8 else f">{lo:g} m"] = {"n": int(h.sum()), "median_m": p[0], "p95_m": p[1],
                                                                 "p99_m": p[2], "max_m": p[3]}
    rec["criterion"] = "information (T4.0b) : erreur 3D entre le point brut et r·rayon(row, col) du modèle"

    # ------------------------------------------------------------------ structure des retours
    cells_hist = _agg_hist(stats, "cells_raw_count_hist")
    gap_h = _agg_hist(stats, "gap_hist")
    ret_struct = {
        "cells_total": S * H * W,
        "cells_by_raw_point_count": {str(k) if k < 4 else ">=4": int(v) for k, v in enumerate(cells_hist)},
        "cells_with_return": int(sum(s["cells_with_return"] for s in stats)),
        "cells_two_distinct_echoes": int(sum(s["cells_two_echoes"] for s in stats)),
        "slot_points": [int(sum(s["n_slot"][k] for s in stats)) for k in range(R)],
        "range_gap_last_minus_first_hist_m": {"edges": GAP_EDGES_M[:-1] + ["inf"], "counts": gap_h.tolist()},
        "range_gap_percentiles_m": dict(zip(("p1", "p10", "p50", "p90", "p99"),
                                            percentiles_from_hist(gap_h, GAP_EDGES_M, (1, 10, 50, 90, 99)))),
        "range_gap_min_m": min(s["gap_min"] for s in stats if s["gap_min"] is not None),
        "intensity_first_gt_last": int(sum(s["intensity_first_gt_last"] for s in stats)),
        "intensity_last_gt_first": int(sum(s["intensity_last_gt_first"] for s in stats)),
        "intensity_equal": int(sum(s["intensity_equal"] for s in stats)),
        "slot0_range_hist": {"edges": NEAR_EDGES_M[:-1] + ["inf"], "counts": _agg_hist(stats, "slot0_range_hist_near").tolist()},
    }
    ret_struct["frac_cells_two_echoes_among_returns"] = ret_struct["cells_two_distinct_echoes"] / max(ret_struct["cells_with_return"], 1)

    # ------------------------------------------------------------------ coupure du tour, occultation proche
    cut = {
        "file_first_col_values": {int(k): int(v) for k, v in zip(*np.unique([s["file_first_col"] for s in stats], return_counts=True))},
        "file_last_col_values": {int(k): int(v) for k, v in zip(*np.unique([s["file_last_col"] for s in stats], return_counts=True))},
        "first_in_time_col_values": {int(k): int(v) for k, v in zip(*np.unique([s["first_in_time_col"] for s in stats], return_counts=True))},
    }
    near_count = np.zeros(H * W, dtype=np.int64)
    for s in stats:
        near_count += np.unpackbits(s["near_mask_bits"])[: H * W]
    near_freq = (near_count / max(S, 1)).reshape(H, W).astype(np.float32)
    near = {"threshold_m": NEAR_RANGE_M,
            "pixels_freq_gt_5pct": int((near_freq > 0.05).sum()),
            "pixels_freq_gt_50pct": int((near_freq > 0.5).sum())}
    cols_hit = np.nonzero((near_freq > 0.5).any(0))[0]
    if len(cols_hit):
        az_cols = np.degrees(model.column_azimuths_rad[cols_hit] + np.median(model.row_azimuth_offsets_rad))
        near["columns_freq_gt_50pct_minmax"] = [int(cols_hit.min()), int(cols_hit.max()), int(len(cols_hit))]
        near["approx_azimuth_deg_minmax"] = [float(((az_cols + 180) % 360 - 180).min()), float(((az_cols + 180) % 360 - 180).max())]
        rows_hit = np.nonzero((near_freq > 0.5).any(1))[0]
        near["rows_freq_gt_50pct_minmax"] = [int(rows_hit.min()), int(rows_hit.max())]
        near["elevation_deg_minmax"] = [float(np.degrees(model.row_elevations_rad[rows_hit]).min()),
                                        float(np.degrees(model.row_elevations_rad[rows_hit]).max())]
    if write:
        np.savez_compressed(f"{SCENE}/lidar_model/hesai_qt64_near_range.npz", freq=near_freq,
                            threshold_m=np.float32(NEAR_RANGE_M), n_scans=np.int64(S))

    # ------------------------------------------------------------------ stockage
    sizes = np.array([s["bytes"] for s in stats], dtype=np.float64)
    storage = {"n_files": int((sizes > 0).sum()), "total_bytes": int(sizes.sum()), "total_gib": float(sizes.sum() / 2**30),
               "per_scan_mb": {"mean": float(sizes.mean() / 1e6) if S else None, "max": float(sizes.max() / 1e6) if S else None},
               "points_stored": int(sum(s["n_stored"] for s in stats))}

    counts = {"points_per_scan": _pct(stats, "n_points"), "valid_points": tot_pts,
              "cells_with_return_per_scan": _pct(stats, "cells_with_return")}
    flagged = []
    for s in stats:
        ti = s["time_info"]
        why = []
        if s["file_first_col"] != 0 or s["file_last_col"] != W - 1:
            why.append(f"coupure: premier col {s['file_first_col']}, dernier col {s['file_last_col']}")
        if ti["parity_consistency"] is None or ti["parity_consistency"] < 0.99:
            why.append(f"parité paquets {ti['parity_consistency']}")
        if ti["pair_dt_bad"] or ti["n_extra"]:
            why.append(f"paires anormales {ti['pair_dt_bad']}, points en trop {ti['n_extra']}")
        if not (-10e-6 <= s["t_rel_min"] and s["t_rel_max"] <= T02_UPPER):
            why.append(f"t_rel hors bornes [{s['t_rel_min']:.6f}, {s['t_rel_max']:.6f}]")
        if s["n_points"] < 0.5 * np.median([x["n_points"] for x in stats]):
            why.append(f"peu de points {s['n_points']}")
        if why:
            flagged.append({"file": s["file"], "why": why})
    report = {
        "step": "stage0_lidar",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "sequence": SEQ,
        "gsplat_commit": GSPLAT_COMMIT,
        "model_path": MODEL_PATH,
        "model_sha1": model_sha1,
        "output_dir": OUT_DIR,
        "npz_layout": {
            "t_start, t_end": "float64 s Unix : temps de la colonne 0 et de la colonne n_cols-1 (poses viewmats / viewmats_rs)",
            "column_period_s": "float64", "t_first_point_raw": "float64 (horodatage brut minimal, = nom du fichier)",
            "range, intensity": "float32 [R=2, 64, 600] (0 = pas de retour) ; slot 0 = écho le plus proche, slot 1 = écho distinct le plus lointain",
            "t_rel": "float32 [R, 64, 600] s depuis t_start (NaN si pas de retour), horodatage corrigé",
            "n_points, n_duplicates, n_unassigned": "int64 (points bruts du PCD, copies exactes non stockées, points non rangés)",
            "pt_xyz, pt_intensity, pt_t, pt_ring, pt_col, pt_ret": "points du PCD sans les copies exactes : float32 N×3 (repère lidar), float32, "
                                                                     "float64 (temps corrigé), uint16, int16, int8 (0/1 = slot, -2 débordement, -3 invalide)",
            "row": "row = 63 - ring",
        },
        "counts": counts,
        "errors": errors,
        "flagged_scans": flagged,
        "tests": {"T0.2": t02, "T0.3": t03, "reconstruction": rec},
        "return_structure": ret_struct,
        "scan_cut": cut,
        "near_range_self_occlusion": near,
        "storage": storage,
    }
    for extra, fn in (("model_estimation", "hesai_qt64_estimation.json"), ("gsplat_check", "hesai_qt64_gsplat_check.json")):
        p = f"{SCENE}/lidar_model/{fn}"
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            d.pop("per_ring", None)
            d.pop("scan_files", None)
            report[extra] = d
    if write and not limit:
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=1)
        vol.commit()
    return json.loads(json.dumps(report))


@app.function(image=image, volumes={DATA_MOUNT: vol}, timeout=300)
def read_json(path: str) -> dict:
    import json

    vol.reload()
    with open(path) as f:
        return json.load(f)


@app.local_entrypoint()
def main(stage: str = "all", limit: int = 0, n_sample: int = 48, chunk: int = 100, write: bool = True):
    import json
    import pathlib

    logs = pathlib.Path("logs")
    logs.mkdir(exist_ok=True)
    if stage in ("model", "all"):
        est = estimate_model.remote(n_sample)
        (logs / "stage0_lidar_model_estimation.json").write_text(json.dumps(est, indent=1))
        s = {k: v for k, v in est.items() if k not in ("per_ring", "scan_files")}
        print(json.dumps(s, indent=1))
    if stage in ("gsplat", "all"):
        gs = gsplat_check.remote()
        (logs / "stage0_lidar_gsplat_check.json").write_text(json.dumps(gs, indent=1))
        print(json.dumps(gs, indent=1))
    if stage in ("convert", "all"):
        rep = run_conversion.remote(limit, chunk, write)
        name = "stage0_lidar.json" if not limit else f"stage0_lidar_limit{limit}.json"
        (logs / name).write_text(json.dumps(rep, indent=1))
        print(json.dumps({"T0.2": rep["tests"]["T0.2"], "T0.3": rep["tests"]["T0.3"],
                          "reconstruction": rep["tests"]["reconstruction"], "storage": rep["storage"],
                          "errors": rep["errors"][:5]}, indent=1))
