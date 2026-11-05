"""Contrôle qualité d'un morceau de splat GaussGym / GrandTour (splatfacto) : rendu gsplat aux poses des photos
(vues d'entraînement et vues tenues à l'écart par la répartition nerfstudio « fraction » 90/10), PSNR / SSIM contre
les photos brutes redressées, puis vues nouvelles décalées (latéral, hauteur) pour juger la tenue hors trajectoire.

Photos : extraites à distance de `images/<cam>.tar` (requêtes HTTP Range, sans télécharger l'archive), redressées
fisheye équidistant → sténopé avec les intrinsèques de `transforms.json`.

  venv/bin/python simrobot/tools/gaussgym_check.py --chunk data_local/simrobot/grandtour/HOB-1/09dfce26 \
      --mission 2024-11-14-11-17-02 --out out/simrobot/check_HOB-1_09dfce26
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request

import numpy as np

HF = "https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset/resolve/main"


# ---------------------------------------------------------------- PLY splatfacto
def read_splat_ply(path):
    with open(path, "rb") as f:
        props = []
        n = 0
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        data = np.fromfile(f, dtype=np.float32, count=n * len(props)).reshape(n, len(props))
    col = {p: i for i, p in enumerate(props)}
    g = lambda *names: data[:, [col[k] for k in names]]
    n_rest = sum(p.startswith("f_rest_") for p in props)
    rest = g(*[f"f_rest_{i}" for i in range(n_rest)]).reshape(n, 3, n_rest // 3).transpose(0, 2, 1)
    return {
        "means": g("x", "y", "z"),
        "sh": np.concatenate([g("f_dc_0", "f_dc_1", "f_dc_2")[:, None], rest], 1),
        "opac": 1 / (1 + np.exp(-g("opacity")[:, 0])),
        "scales": np.exp(g("scale_0", "scale_1", "scale_2")),
        "quats": g("rot_0", "rot_1", "rot_2", "rot_3"),
    }


# ---------------------------------------------------------------- photos dans l'archive distante
def _headers(buf, base):
    """(nom, taille, offset données) des en-têtes ustar alignés sur 512 dans buf (offset absolu base)."""
    out = []
    for o in range(0, len(buf) - 512, 512):
        h = buf[o:o + 512]
        if h[257:262] == b"ustar":
            name = h[:100].rstrip(b"\0").decode(errors="replace")
            size = int(h[124:136].rstrip(b"\0 ") or b"0", 8)
            out.append((name, size, base + o + 512))
    return out


def camera_timestamps(mission, cam):
    """Horodatages (s) des images de `images/<cam>.tar`, dans l'ordre des index, lus dans `data/<cam>.tar` (zarr
    compressé blosc). La numérotation des PNG GaussGym ne suit PAS toujours l'index de l'archive (vrai pour HÖB-1,
    faux pour LMB-1) : apparier par horodatage."""
    import io
    import tarfile

    import numcodecs

    buf = urllib.request.urlopen(f"{HF}/{mission}/data/{cam}.tar").read()
    tf = tarfile.open(fileobj=io.BytesIO(buf))
    za = json.load(tf.extractfile(f"{cam}/timestamp/.zarray"))
    raw = tf.extractfile(f"{cam}/timestamp/0").read()
    codec = numcodecs.get_codec(za["compressor"])
    return np.frombuffer(codec.decode(raw), dtype=np.dtype(za["dtype"]))[:za["shape"][0]]


def fetch_tar_members(url, indices, guess_stride=587_000, win=6 << 20):
    """Membres `<cam>/<idx:06d>.jpeg` d'une archive tar distante, par requêtes Range : on estime l'offset, on lit
    une fenêtre, on recale la foulée moyenne sur les en-têtes trouvés, et on recommence."""
    got = {}
    head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
    total = int(head.headers["Content-Length"])
    stride, anchor = guess_stride, (0, 0)  # (index, offset) connu
    for idx in sorted(indices):
        for _ in range(8):
            est = anchor[1] + (idx - anchor[0]) * stride
            start = min(max(0, (est - win // 2) // 512 * 512), max(0, (total - win) // 512 * 512))
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + win - 1}"})
            buf = urllib.request.urlopen(req).read()
            hs = [(int(re.search(r"(\d+)\.jpe?g$", n).group(1)), s, o) for n, s, o in _headers(buf, start)
                  if re.search(r"(\d+)\.jpe?g$", n)]
            if not hs:
                stride = int(stride * 1.01)
                continue
            hit = [h for h in hs if h[0] == idx]
            i0, _, o0 = hs[0]
            if i0 > 0 and i0 != anchor[0]:
                stride = (o0 - anchor[1]) // (i0 - anchor[0]) if i0 != anchor[0] else stride
                anchor = (i0, o0)
            if hit:
                _, s, o = hit[0]
                if o - start + s <= len(buf):
                    got[idx] = buf[o - start:o - start + s]
                else:
                    req = urllib.request.Request(url, headers={"Range": f"bytes={o}-{o + s - 1}"})
                    got[idx] = urllib.request.urlopen(req).read()
                break
    return got


# ---------------------------------------------------------------- géométrie
def c2w_opencv(frame, dp):
    """Pose caméra → repère du splat (repère dataparser nerfstudio), convention OpenCV (z avant, y bas)."""
    T = np.array(frame["transform_matrix"], float)
    T[:3, 1:3] *= -1                               # OpenGL → OpenCV
    A = np.eye(4)
    A[:3] = np.array(dp["transform"], float)
    T = A @ T
    T[:3, 3] *= dp["scale"]
    return T


def ssim(a, b):
    import cv2
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    s_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    s_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    s_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (s_a + s_b + C2))
    return float(m.mean())


def main():
    import cv2
    import torch
    import yaml
    from gsplat import rasterization

    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--mission", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"

    tr = json.load(open(f"{a.chunk}/transforms.json"))
    dp = json.load(open(f"{a.chunk}/splatfacto/dataparser_transforms.json"))
    G = read_splat_ply(f"{a.chunk}/splatfacto/splat.ply")
    t = {k: torch.tensor(v, device=dev, dtype=torch.float32) for k, v in G.items()}
    t["quats"] = t["quats"] / t["quats"].norm(dim=1, keepdim=True)
    sh_deg = int(np.sqrt(G["sh"].shape[1])) - 1
    s = dp["scale"]

    frames = tr["frames"]
    # répartition nerfstudio « fraction » : linspace sur l'ordre trié des fichiers. GaussGym règle
    # train_split_fraction = 1,0 → AUCUNE vue tenue à l'écart (toutes les vues sont des vues d'entraînement).
    frac = float(re.search(r"train_split_fraction:\s*([\d.]+)", open(f"{a.chunk}/splatfacto/config.yml").read()).group(1))
    order = np.argsort([f["file_path"] for f in frames])
    n = len(frames)
    i_train = np.linspace(0, n - 1, int(np.ceil(n * frac))).astype(int)
    i_eval = np.setdiff1d(np.arange(n), i_train)
    split = {int(order[i]): "train" for i in i_train} | {int(order[i]): "eval" for i in i_eval}

    def render(c2w, fr, W, H):
        K = torch.tensor([[float(fr["fl_x"]), 0, float(fr["cx"])], [0, float(fr["fl_y"]), float(fr["cy"])],
                          [0, 0, 1]], device=dev, dtype=torch.float32)
        vm = torch.tensor(np.linalg.inv(c2w), device=dev, dtype=torch.float32)
        img, alpha, _ = rasterization(t["means"], t["quats"], t["scales"], t["opac"], t["sh"], vm[None], K[None],
                                      W, H, sh_degree=sh_deg, rasterize_mode="antialiased",
                                      render_mode="RGB+ED", near_plane=0.01 * s, far_plane=1e3)
        rgb = img[0, ..., :3].clamp(0, 1).cpu().numpy()
        dep = img[0, ..., 3].cpu().numpy() / s                   # profondeur en mètres
        return (rgb * 255).astype(np.uint8)[..., ::-1], dep, alpha[0, ..., 0].cpu().numpy()

    # ---- vues réelles : quelques vues d'entraînement + toutes les vues tenues à l'écart (par caméra)
    pick = {}
    for cam in ("hdr_front", "hdr_left", "hdr_right"):
        idx = [i for i in range(n) if cam in frames[i]["file_path"]]
        ev = [i for i in idx if split[i] == "eval"]
        trn = [i for i in idx if split[i] == "train"]
        pick[cam] = ev[:3] + trn[len(trn) // (a.n_train + 1)::len(trn) // (a.n_train + 1)][:a.n_train]
    rep = []
    for cam, ids in pick.items():
        meta = yaml.safe_load(urllib.request.urlopen(f"{HF}/{a.mission}/metadata/{cam}_caminfo.yaml"))["camera_info"]
        K0 = np.array(meta["K"], float).reshape(3, 3)
        D0 = np.array(meta["D"], float)
        ts = camera_timestamps(a.mission, cam)
        num = {}
        for i in ids:
            sec, nsec = str(frames[i]["timestamp"]).split("_")
            tf = int(sec) + int(nsec) * 1e-9
            j = int(np.argmin(np.abs(ts - tf)))
            if abs(ts[j] - tf) < 0.005:
                num[i] = j
        ids = [i for i in ids if i in num]
        photos = fetch_tar_members(f"{HF}/{a.mission}/images/{cam}.tar", list(num.values()))
        for i in ids:
            fr = frames[i]
            W, H = int(fr["w"]), int(fr["h"])
            Kn = np.array([[float(fr["fl_x"]), 0, float(fr["cx"])], [0, float(fr["fl_y"]), float(fr["cy"])], [0, 0, 1]])
            raw = cv2.imdecode(np.frombuffer(photos[num[i]], np.uint8), cv2.IMREAD_COLOR)
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(K0, D0, np.eye(3), Kn, (W, H), cv2.CV_32FC1)
            photo = cv2.remap(raw, m1, m2, cv2.INTER_LINEAR)
            valid = cv2.remap(np.full(raw.shape[:2], 255, np.uint8), m1, m2, cv2.INTER_NEAREST) > 0
            rgb, dep, alpha = render(c2w_opencv(fr, dp), fr, W, H)
            v = valid & (alpha > 0.5)
            mse = np.mean((rgb[v].astype(float) - photo[v].astype(float)) ** 2)
            psnr = 10 * np.log10(255 ** 2 / mse)
            gs = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            r = {"cam": cam, "img": num[i], "split": split[i], "psnr": round(psnr, 2),
                 "ssim": round(ssim(gs(rgb) * v, gs(photo) * v), 3), "couverture": round(float(v.mean()), 3)}
            rep.append(r)
            print(json.dumps(r), flush=True)
            sheet = np.concatenate([photo, rgb], 1)
            sheet = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2))
            cv2.putText(sheet, f"{cam} {num[i]} [{split[i]}] photo | rendu  PSNR {psnr:.1f} dB", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imwrite(f"{a.out}/{split[i]}_{cam}_{num[i]}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # ---- vues nouvelles : caméra avant au milieu du morceau, décalée latéralement et en hauteur
    fi = [i for i in range(n) if "hdr_front" in frames[i]["file_path"]]
    fr = frames[fi[len(fi) // 2]]
    W, H = int(fr["w"]), int(fr["h"])
    T0 = c2w_opencv(fr, dp)
    # « haut » du repère du splat : nerfstudio orientation_method=up aligne la verticale moyenne des caméras sur +z
    up = np.array([0, 0, 1.0])
    fwd = T0[:3, 2] - up * (T0[:3, 2] @ up)
    fwd /= np.linalg.norm(fwd)
    left = np.cross(up, fwd)
    tiles = []
    for name, dl, dz in [("trajectoire", 0, 0), ("lateral_1m", 1, 0), ("lateral_2m", 2, 0), ("lateral_-2m", -2, 0),
                         ("haut_1m", 0, 1), ("haut_2m", 0, 2)]:
        T = T0.copy()
        T[:3, 3] += s * (dl * left + dz * up)
        if dz > 0:                                   # regarder un peu vers le bas
            ang = np.deg2rad(15 * dz)
            Rx = np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]])
            T[:3, :3] = T[:3, :3] @ Rx
        rgb, dep, alpha = render(T, fr, W, H)
        im = cv2.resize(rgb, (W // 3, H // 3))
        cv2.putText(im, name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(im)
    grid = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
    cv2.imwrite(f"{a.out}/vues_nouvelles.jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # ---- statistiques du morceau (en mètres, repère du splat / scale)
    P = G["means"] / s
    cams = np.array([c2w_opencv(f, dp)[:3, 3] for f in frames]) / s
    summ = {
        "n_gaussiennes": int(len(P)),
        "etendue_cameras_m": np.ptp(cams, 0).round(2).tolist(),
        "etendue_splat_p5_p95_m": (np.percentile(P, 95, 0) - np.percentile(P, 5, 0)).round(1).tolist(),
        "frac_gauss_a_moins_de_5m_des_cameras": None,
        "psnr_eval_median": float(np.median([r["psnr"] for r in rep if r["split"] == "eval"] or [np.nan])),
        "psnr_train_median": float(np.median([r["psnr"] for r in rep if r["split"] == "train"] or [np.nan])),
    }
    from scipy.spatial import cKDTree
    d, _ = cKDTree(cams).query(P[::10])
    summ["frac_gauss_a_moins_de_5m_des_cameras"] = round(float((d < 5).mean()), 3)
    print(json.dumps(summ), flush=True)
    json.dump({"vues": rep, "resume": summ}, open(f"{a.out}/report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
