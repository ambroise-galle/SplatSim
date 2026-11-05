"""Extraction EN FLUX d'une tranche de temps d'un bag ROS1 distant (lecture partielle HTTP), sans télécharger le bag.

Pensé pour DiTer++ (bags de 70–84 Go sur Dropbox, qui accepte les requêtes `Range`) :
  1. l'index du bag (fin de fichier, quelques Mo) donne la position et l'intervalle de temps de chaque bloc
     (`chunk`) ; on le lit une fois (`--index`, produit par `index_main`) ;
  2. on demande d'un seul tenant la plage d'octets des blocs de la tranche, et on analyse le flux à la volée :
     enregistrements ROS1 (op 5 = bloc, op 7 = connexion, op 2 = message, op 4 = index de bloc) ;
  3. seuls les sujets voulus sont décodés (`rosbags`) et écrits sous forme compacte ; le reste est jeté.
Reprise automatique sur coupure réseau (on redemande à partir du dernier bloc complet).

Sorties (`--out`) :
  lidar/<stamp_ns>.npz   nuage par balayage : xyz (float32, repère capteur), t_rel (s), ring, intensity,
                         reflectivity, range (m) — champs présents dans le message ;
  cam/<stamp_ns>.jpg     images couleur (JPEG q95) ; camera_info.json (1er message CameraInfo) ;
  imu_<nom>.npz          t, gyro, acc (et orientation) par IMU ;
  meta.json              sujets, intervalle, comptes, octets lus.

  .venv/bin/python -m splatpipe.stage0.rosbag_stream index --url URL --out index.json
  .venv/bin/python -m splatpipe.stage0.rosbag_stream extract --url URL --index index.json --t0 T0 --t1 T1 --out DIR
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import time

import numpy as np

DEFAULT_TOPICS = {
    "/agent1/ouster/points": "lidar",
    "/agent1/front/color/image_raw": "cam",
    "/agent1/front/color/camera_info": "caminfo",
    "/agent1/ouster/imu": "imu_ouster",
    "/agent1/imu/data": "imu_body",
}
PC2_TYPES = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}


def _fields(h: bytes) -> dict:
    f, q = {}, 0
    while q < len(h):
        n = struct.unpack_from("<I", h, q)[0]
        q += 4
        k, v = h[q:q + n].split(b"=", 1)
        q += n
        f[k.decode()] = v
    return f


def _records(buf: bytes):
    """Itère (header_dict, data) sur un tampon d'enregistrements ROS1 complets."""
    p = 0
    while p + 4 <= len(buf):
        hl = struct.unpack_from("<I", buf, p)[0]
        h = _fields(buf[p + 4:p + 4 + hl])
        p += 4 + hl
        dl = struct.unpack_from("<I", buf, p)[0]
        yield h, buf[p + 4:p + 4 + dl]
        p += 4 + dl


# ----------------------------------------------------------------------------------------------- index
def read_index(url: str) -> dict:
    import requests

    head = requests.get(url, headers={"Range": "bytes=0-4095"}, timeout=60).content
    assert head[:13] == b"#ROSBAG V2.0\n", head[:20]
    hl = struct.unpack_from("<I", head, 13)[0]
    f = _fields(head[17:17 + hl])
    index_pos = struct.unpack("<Q", f["index_pos"])[0]
    tail = requests.get(url, headers={"Range": f"bytes={index_pos}-"}, timeout=600).content
    conns, chunks = {}, []
    for h, d in _records(tail):
        op = h["op"][0]
        if op == 7:
            cid = struct.unpack("<I", h["conn"])[0]
            conns[cid] = {"topic": h["topic"].decode(), "type": _fields(d).get("type", b"").decode()}
        elif op == 6:
            s, e = struct.unpack("<II", h["start_time"]), struct.unpack("<II", h["end_time"])
            chunks.append([struct.unpack("<Q", h["chunk_pos"])[0], s[0] + s[1] * 1e-9, e[0] + e[1] * 1e-9])
    chunks.sort()
    return {"url": url, "index_pos": index_pos, "conns": conns, "chunks": chunks}


# ----------------------------------------------------------------------------------------------- décodage
class Sink:
    def __init__(self, out: str, topics: dict):
        from rosbags.typesys import Stores, get_typestore

        self.ts = get_typestore(Stores.ROS1_NOETIC)
        self.out, self.topics = out, topics
        os.makedirs(f"{out}/lidar", exist_ok=True)
        os.makedirs(f"{out}/cam", exist_ok=True)
        self.imu = {}
        self.counts = {}
        self.caminfo = None

    def handle(self, topic: str, rostype: str, stamp: float, raw: bytes):
        kind = self.topics[topic]
        msg = self.ts.deserialize_ros1(raw, rostype.replace("/", "/msg/", 1))
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if kind == "lidar":
            self._pc2(msg)
        elif kind == "cam":
            self._image(msg)
        elif kind == "caminfo" and self.caminfo is None:
            self.caminfo = {"width": int(msg.width), "height": int(msg.height), "distortion_model": msg.distortion_model,
                            "D": [float(x) for x in msg.D], "K": [float(x) for x in msg.K],
                            "R": [float(x) for x in msg.R], "P": [float(x) for x in msg.P],
                            "frame_id": msg.header.frame_id}
        elif kind.startswith("imu"):
            a = self.imu.setdefault(kind, {"t": [], "gyro": [], "acc": [], "quat": []})
            a["t"].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            a["gyro"].append((msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z))
            a["acc"].append((msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z))
            a["quat"].append((msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w))

    @staticmethod
    def _stamp_ns(msg) -> int:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _pc2(self, msg):
        dt = np.dtype({"names": [f.name for f in msg.fields], "formats": [PC2_TYPES[f.datatype] for f in msg.fields],
                       "offsets": [f.offset for f in msg.fields], "itemsize": msg.point_step})
        arr = np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)
        out = {"xyz": np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float32),
               "height": np.int32(msg.height), "width": np.int32(msg.width), "frame_id": np.array(msg.header.frame_id)}
        names = arr.dtype.names
        if "t" in names:
            out["t_rel"] = (arr["t"].astype(np.float64) * 1e-9).astype(np.float32)   # Ouster : ns depuis le début
        for k in ("ring", "intensity", "reflectivity", "range", "ambient"):
            if k in names:
                v = arr[k]
                out[k] = (v.astype(np.float32) * 1e-3) if k == "range" else v.copy()   # range Ouster en mm
        np.savez_compressed(f"{self.out}/lidar/{self._stamp_ns(msg)}.npz", **out)

    def _image(self, msg):
        from PIL import Image

        enc = msg.encoding.lower()
        a = np.frombuffer(bytes(msg.data), np.uint8)
        if enc in ("rgb8", "bgr8"):
            img = a.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
            if enc == "bgr8":
                img = img[..., ::-1]
        elif enc in ("mono8",):
            img = a.reshape(msg.height, msg.step)[:, :msg.width]
        else:
            raise ValueError(f"encodage image non géré : {msg.encoding}")
        Image.fromarray(np.ascontiguousarray(img)).save(f"{self.out}/cam/{self._stamp_ns(msg)}.jpg", quality=95)

    def close(self, meta: dict):
        for k, a in self.imu.items():
            np.savez_compressed(f"{self.out}/{k}.npz", **{n: np.asarray(v) for n, v in a.items()})
        if self.caminfo:
            json.dump(self.caminfo, open(f"{self.out}/camera_info.json", "w"), indent=1)
        meta["counts"] = self.counts
        json.dump(meta, open(f"{self.out}/meta.json", "w"), indent=1)


# ----------------------------------------------------------------------------------------------- flux
class _Stream:
    """Lecteur d'octets sur une réponse HTTP en flux, avec compteur de position absolue."""

    def __init__(self, resp, pos0: int):
        self.it = resp.iter_content(chunk_size=8 << 20)
        self.buf = bytearray()
        self.pos = pos0

    def read(self, n: int) -> bytes:
        while len(self.buf) < n:
            try:
                self.buf += next(self.it)
            except StopIteration:
                raise EOFError
        out = bytes(self.buf[:n])
        del self.buf[:n]
        self.pos += n
        return out


def extract(url, index, t0, t1, out, topics=None):
    import requests

    topics = topics or DEFAULT_TOPICS
    os.makedirs(out, exist_ok=True)
    chunks = [c for c in index["chunks"] if c[2] >= t0 and c[1] <= t1]
    all_pos = [c[0] for c in index["chunks"]]
    last = chunks[-1][0]
    after = next((p for p in all_pos if p > last), index["index_pos"])
    pos, end = chunks[0][0], after - 1
    conn_info = {int(k): v for k, v in index["conns"].items()}
    sink = Sink(out, topics)
    t_start, n_chunks, n_bytes = time.time(), 0, 0
    while pos <= end:
        try:
            r = requests.get(url, headers={"Range": f"bytes={pos}-{end}"}, stream=True, timeout=120)
            r.raise_for_status()
            s = _Stream(r, pos)
            while s.pos <= end:
                rec_start = s.pos
                hl = struct.unpack("<I", s.read(4))[0]
                h = _fields(s.read(hl))
                dl = struct.unpack("<I", s.read(4))[0]
                data = s.read(dl)
                if h["op"][0] == 5:                                  # bloc : messages + connexions
                    comp = h.get("compression", b"none")
                    if comp == b"lz4":
                        import lz4.frame
                        data = lz4.frame.decompress(data)
                    elif comp == b"bz2":
                        import bz2
                        data = bz2.decompress(data)
                    for ih, idata in _records(data):
                        op = ih["op"][0]
                        if op == 7:
                            cid = struct.unpack("<I", ih["conn"])[0]
                            conn_info.setdefault(cid, {"topic": ih["topic"].decode(),
                                                       "type": _fields(idata).get("type", b"").decode()})
                        elif op == 2:
                            cid = struct.unpack("<I", ih["conn"])[0]
                            sec, nsec = struct.unpack("<II", ih["time"])
                            tt = sec + nsec * 1e-9
                            ci = conn_info.get(cid)
                            if ci and ci["topic"] in topics and t0 <= tt <= t1:
                                try:
                                    sink.handle(ci["topic"], ci["type"], tt, idata)
                                except Exception as e:                 # message illisible : on le saute
                                    sink.counts["erreurs"] = sink.counts.get("erreurs", 0) + 1
                                    if sink.counts["erreurs"] <= 3:
                                        print(f"[message ignoré] {ci['topic']}: {type(e).__name__}: {e}", flush=True)
                    n_chunks += 1
                    n_bytes = s.pos - chunks[0][0]
                    pos = s.pos                                      # reprise possible à partir d'ici
                    if n_chunks % 200 == 0:
                        el = time.time() - t_start
                        print(f"{n_chunks}/{len(chunks)} blocs, {n_bytes/1e9:.2f} Go, {n_bytes/1e6/max(el,1):.1f} Mo/s, "
                              f"{json.dumps(sink.counts)}", flush=True)
                else:
                    pos = s.pos
                if rec_start == s.pos:
                    break
        except EOFError:
            break
        except Exception as e:                                       # coupure réseau : reprise au dernier bloc complet
            print(f"[reprise] {type(e).__name__}: {e} — à partir de l'octet {pos}", flush=True)
            time.sleep(5)
    meta = {"url": url, "t0": t0, "t1": t1, "n_chunks": n_chunks, "bytes": n_bytes,
            "seconds": round(time.time() - t_start, 1), "topics": topics}
    sink.close(meta)
    print(json.dumps(meta, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index")
    a.add_argument("--url", required=True)
    a.add_argument("--out", required=True)
    b = sub.add_parser("extract")
    b.add_argument("--url", required=True)
    b.add_argument("--index", required=True)
    b.add_argument("--t0", type=float, required=True)
    b.add_argument("--t1", type=float, required=True)
    b.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "index":
        idx = read_index(args.url)
        json.dump(idx, open(args.out, "w"))
        T0, T1 = idx["chunks"][0][1], idx["chunks"][-1][2]
        print(f"{len(idx['chunks'])} blocs, {T1 - T0:.1f} s ; sujets : {sorted(v['topic'] for v in idx['conns'].values())}")
    else:
        extract(args.url, json.load(open(args.index)), args.t0, args.t1, args.out)


if __name__ == "__main__":
    main()
