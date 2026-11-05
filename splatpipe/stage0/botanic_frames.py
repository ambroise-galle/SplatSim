"""BotanicGarden : horodatage des images pleine résolution (imagezip) et identification des caméras.

Les archives d'images (1920×1200 à 40 Hz, TIFF) n'ont aucun horodatage : nom = `img_<dossier>_<série>_seq_<n>_<vue>.tif`
avec un compteur de vues continu. Les bags VLIO contiennent les mêmes caméras réduites (960×600 à 10 Hz) avec leur
temps ROS. On apparie les deux par corrélation normalisée (images ramenées à 240×150 en niveaux de gris) :
- `scan`  : lecture SÉQUENTIELLE du bag VLIO (rosbags refuse son index : `read_index_data` assert) → pour chaque sujet
            image, temps d'en-tête de chaque message et vignette 240×150 (`<out>/vlio_thumbs.npz`) ;
- `match` : pour chaque dossier de caméra, vues 1, 11, 21… comparées à toutes les vignettes de chaque sujet → sujet de
            la caméra (gray/rgb, left/right), relation vue ↔ temps (ajustement linéaire, contrôle du pas de 25 ms et des
            résidus sur toute la séquence) → `<out>/frames_<dossier>.csv` (vue, stamp_ns, chemin) et `cameras.json`.

  venv/bin/python -m splatpipe.stage0.botanic_frames scan  --bag data_local/botanicgarden/1018_00/1018_00_VLIO.bag \
      --out data_local/botanicgarden/1018_00/frames
  venv/bin/python -m splatpipe.stage0.botanic_frames match --imgs data_local/botanicgarden/1018_00/1018_dalsa_garden_short \
      --out data_local/botanicgarden/1018_00/frames
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct

import numpy as np

from splatpipe.stage0.rosbag_stream import _fields, _records

TW, TH = 240, 150


def iter_bag(path: str):
    """(topic, type, t_bag, raw) de chaque message, en lisant les blocs dans l'ordre du fichier (sans l'index)."""
    conns = {}
    with open(path, "rb") as f:
        assert f.read(13) == b"#ROSBAG V2.0\n"
        while True:
            hb = f.read(4)
            if len(hb) < 4:
                return
            hl = struct.unpack("<I", hb)[0]
            h = _fields(f.read(hl))
            dl = struct.unpack("<I", f.read(4))[0]
            op = h["op"][0]
            if op != 5:                                    # en-tête du bag, index, infos de blocs : sautés
                f.seek(dl, 1)
                continue
            data = f.read(dl)
            comp = h.get("compression", b"none")
            if comp == b"lz4":
                import lz4.frame
                data = lz4.frame.decompress(data)
            elif comp == b"bz2":
                import bz2
                data = bz2.decompress(data)
            for ih, idata in _records(data):
                iop = ih["op"][0]
                if iop == 7:
                    cid = struct.unpack("<I", ih["conn"])[0]
                    conns[cid] = (ih["topic"].decode(), _fields(idata).get("type", b"").decode())
                elif iop == 2:
                    cid = struct.unpack("<I", ih["conn"])[0]
                    if cid in conns:
                        sec, nsec = struct.unpack("<II", ih["time"])
                        yield conns[cid][0], conns[cid][1], sec + nsec * 1e-9, idata


def to_gray_thumb(img: np.ndarray, encoding: str = "mono8") -> np.ndarray:
    import cv2

    enc = encoding.lower()
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if enc.startswith("rgb") else cv2.COLOR_BGR2GRAY)
    elif enc.startswith("bayer"):
        code = {"bayer_rggb8": cv2.COLOR_BayerRG2GRAY, "bayer_bggr8": cv2.COLOR_BayerBG2GRAY,
                "bayer_gbrg8": cv2.COLOR_BayerGB2GRAY, "bayer_grbg8": cv2.COLOR_BayerGR2GRAY}[enc]
        g = cv2.cvtColor(img, code)
    else:
        g = img
    return cv2.resize(g, (TW, TH), interpolation=cv2.INTER_AREA)


def scan(bag: str, out: str):
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS1_NOETIC)
    os.makedirs(out, exist_ok=True)
    stamps, thumbs, info = {}, {}, {}
    for topic, typ, t, raw in iter_bag(bag):
        if typ not in ("sensor_msgs/Image", "sensor_msgs/CompressedImage"):
            continue
        m = ts.deserialize_ros1(raw, typ.replace("/", "/msg/", 1))
        st = int(m.header.stamp.sec) * 1_000_000_000 + int(m.header.stamp.nanosec)
        if typ == "sensor_msgs/CompressedImage":
            import cv2
            img = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_UNCHANGED)
            enc = "bgr8" if img.ndim == 3 else "mono8"
        else:
            enc = m.encoding
            ch = {"mono8": 1, "rgb8": 3, "bgr8": 3}.get(enc.lower(), 1)
            a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, :m.width * ch]
            img = a.reshape(m.height, m.width, ch) if ch == 3 else a
        stamps.setdefault(topic, []).append(st)
        thumbs.setdefault(topic, []).append(to_gray_thumb(img, enc))
        info.setdefault(topic, {"type": typ, "encoding": enc, "width": int(img.shape[1]), "height": int(img.shape[0])})
    np.savez_compressed(f"{out}/vlio_thumbs.npz",
                        **{f"st::{k}": np.array(v, np.int64) for k, v in stamps.items()},
                        **{f"im::{k}": np.stack(v) for k, v in thumbs.items()})
    for k in info:
        s = np.array(stamps[k])
        info[k].update(n=len(s), debut_ns=int(s[0]), fin_ns=int(s[-1]), pas_median_ms=float(np.median(np.diff(s)) / 1e6))
    json.dump(info, open(f"{out}/vlio_topics.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


def _ncc(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Corrélation normalisée d'une vignette (H, W) avec un paquet (N, H, W)."""
    a = a.astype(np.float32).ravel()
    a = (a - a.mean()) / (a.std() + 1e-6)
    B = B.reshape(len(B), -1).astype(np.float32)
    B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-6)
    return B @ a / len(a)


def match(imgs: str, out: str, probe_step: int = 10, n_probe: int = 60):
    import cv2

    z = np.load(f"{out}/vlio_thumbs.npz")
    topics = sorted(k[4:] for k in z.files if k.startswith("im::"))
    TH_ = {t: z[f"im::{t}"] for t in topics}
    ST = {t: z[f"st::{t}"] for t in topics}
    cams = {}
    for d in sorted(glob.glob(f"{imgs}/**/c*/", recursive=True)):
        files = sorted(glob.glob(f"{d}*.tif"), key=lambda p: int(p.rsplit("_", 1)[1][:-4]))
        if not files:
            continue
        name = os.path.basename(d.rstrip("/"))
        frame_no = np.array([int(p.rsplit("_", 1)[1][:-4]) for p in files])

        def thumb(i):
            im = cv2.imread(files[i], cv2.IMREAD_UNCHANGED)
            return to_gray_thumb(im, "bgr8" if im.ndim == 3 else "mono8"), im.ndim == 3
        probes = list(range(0, min(len(files), probe_step * n_probe), probe_step))
        probes += list(range(len(files) // 2, min(len(files), len(files) // 2 + probe_step * 20), probe_step))
        probes += list(range(max(0, len(files) - probe_step * 20), len(files), probe_step))
        best = {}
        color = None
        for i in probes:
            th, color = thumb(i)
            for t in topics:
                c = _ncc(th, TH_[t])
                k = int(np.argmax(c))
                best.setdefault(t, []).append((frame_no[i], k, float(c[k])))
        # sujet de la caméra : meilleure corrélation médiane
        topic = max(topics, key=lambda t: np.median([b[2] for b in best[t]]))
        b = np.array(best[topic])
        fr, k, cc = b[:, 0], b[:, 1].astype(int), b[:, 2]
        t_k = ST[topic][k] / 1e9
        good = cc > 0.95
        A = np.c_[np.ones(good.sum()), fr[good]]
        coef, *_ = np.linalg.lstsq(A, t_k[good], rcond=None)
        res_ms = (t_k[good] - A @ coef) * 1e3          # gonflé par les arrêts du robot (images identiques)
        n_bag = len(ST[topic])
        phase, phase_scores = None, {}
        check_min = None
        if abs(len(files) - 4 * n_bag) <= 3:
            # correspondance exacte : vue d'indice 4k + φ (0-based) ↔ image k du bag (l'archive peut finir jusqu'à 3 vues
            # plus tôt) ; φ par corrélation moyenne sur 120 images réparties (les passages en mouvement départagent)
            n_al = min(n_bag, (len(files) - 1) // 4 + 1)
            ks = np.linspace(0, n_al - 2, 120).astype(int)
            for ph in range(4):
                phase_scores[ph] = float(np.mean([_ncc(thumb(4 * kk + ph)[0], TH_[topic][kk:kk + 1])[0] for kk in ks]))
            phase = max(phase_scores, key=phase_scores.get)
            n_al = min(n_bag, (len(files) - 1 - phase) // 4 + 1)
            idx_al = 4 * np.arange(n_al) + phase                  # indices (0-based) des vues alignées
            t_al = ST[topic][:n_al].astype(np.float64)
            # contrôle sur toute la séquence (une vue manquante au milieu décalerait la suite) : corrélation minimale
            kc = np.linspace(0, n_al - 1, 200).astype(int)
            check_min = float(min(_ncc(thumb(4 * kk + phase)[0], TH_[topic][kk:kk + 1])[0] for kk in kc))
            stamp = np.interp(np.arange(len(files)), idx_al, t_al)
            ext = np.arange(len(files)) > idx_al[-1]              # après la dernière vue alignée : pas de 25 ms
            stamp[ext] = t_al[-1] + (np.arange(len(files))[ext] - idx_al[-1]) * 25e6
            pre = np.arange(len(files)) < idx_al[0]
            stamp[pre] = t_al[0] - (idx_al[0] - np.arange(len(files))[pre]) * 25e6
            stamp = stamp.astype(np.int64)
        else:
            stamp = ((coef[0] + coef[1] * frame_no) * 1e9).astype(np.int64)
        with open(f"{out}/frames_{name}.csv", "w") as fo:
            fo.write("frame,stamp_ns,path\n")
            for f_, s_, p_ in zip(frame_no, stamp, files):
                fo.write(f"{f_},{s_},{os.path.abspath(p_)}\n")
        cams[name] = {"topic": topic, "couleur": bool(color), "n_vues": len(files), "n_bag": n_bag,
                      "methode": "exacte (4 vues par image du bag)" if phase is not None else "ajustement linéaire",
                      "phase": phase, "scores_phase": phase_scores, "controle_correlation_min_200": check_min,
                      "pas_median_ms": float(np.median(np.diff(stamp)) / 1e6),
                      "correlation_mediane": float(np.median(cc)), "frac_sondes_appariees": float(good.mean()),
                      "pas_s": float(coef[1]), "frequence_hz": float(1 / coef[1]) if coef[1] else None,
                      "residu_ms_max": float(np.abs(res_ms).max()) if len(res_ms) else None,
                      "premiere_vue_ns": int(stamp[0]), "derniere_vue_ns": int(stamp[-1]),
                      "autres_sujets_correlation": {t: float(np.median([x[2] for x in best[t]])) for t in topics}}
        print(name, json.dumps(cams[name], ensure_ascii=False), flush=True)
    json.dump(cams, open(f"{out}/cameras.json", "w"), indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["scan", "match"])
    ap.add_argument("--bag")
    ap.add_argument("--imgs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.stage == "scan":
        scan(a.bag, a.out)
    else:
        match(a.imgs, a.out)


if __name__ == "__main__":
    main()
