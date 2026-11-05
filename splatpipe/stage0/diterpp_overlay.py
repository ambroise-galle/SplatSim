"""DiTer++ : contrôle visuel de la calibration caméra ↔ LiDAR et de la synchronisation.

Carte de référence colorée (`Lawn_GT.pcd`) rendue (z-buffer) dans la caméra couleur, pose
T_map_cam(t) = T_map_lidar(t) · T_lidar_camera, avec T_map_lidar = trajectoire de référence (pose du LiDAR dans
le repère de la carte, vérifié : balayages à 3–7 cm de la carte) interpolée au temps de l'image.
Deux jeux d'intrinsèques comparés : document des auteurs (plumb_bob avec distorsion) et `camera_info` du pilote
(distorsion nulle). Planche : photo | rendu (auteurs) | rendu (pilote), et score de bords.

  .venv/bin/python -m splatpipe.stage0.diterpp_overlay --dir data_local/diterpp/lawn
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw

T_LIDAR_CAMERA = np.array([[-0.00960827, -0.00551823, 0.999939, 0.207485],
                           [-0.999944, -0.00440598, -0.00963264, 0.0441344],
                           [0.00445887, -0.999975, -0.00547559, -0.0340388],
                           [0, 0, 0, 1.0]])
K_AUTEURS = dict(fx=602.759355073482, fy=603.104346263694, cx=317.332303454975, cy=252.658513638755,
                 D=[0.0941101147405658, -0.112593630936320, 0.000621077448723540, -0.000148779248023108,
                    -0.321163183583000])


def fold_radius(D, r_max: float = 3.0) -> float:
    """Rayon normalisé (non distordu) où r·(1 + k1 r² + k2 r⁴ + k3 r⁶) cesse de croître : au-delà, le polynôme
    radial se replie (k3 < 0) et des points hors champ retomberaient dans l'image. Auteurs DiTer++ : 0,86 ; coin de
    l'image : 0,67."""
    k1, k2, _, _, k3 = D
    r = np.linspace(0.0, r_max, 30001)
    dr = 1 + 3 * k1 * r ** 2 + 5 * k2 * r ** 4 + 7 * k3 * r ** 6
    bad = np.flatnonzero(dr <= 0)
    return float(r[bad[0]]) if len(bad) else r_max


def project(pc, K):
    """Projection sténopé + plumb_bob (k1, k2, p1, p2, k3), bornée au domaine monotone du polynôme."""
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    k1, k2, p1, p2, k3 = K["D"]
    lim = fold_radius(K["D"])
    out = (x * x + y * y) > lim * lim
    x, y = np.where(out, 1e3, x), np.where(out, 1e3, y)
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return K["fx"] * xd + K["cx"], K["fy"] * yd + K["cy"]


def render(xyz, rgb, T_map_cam, K, W, H, max_range=200.0):
    d = xyz - T_map_cam[:3, 3]
    near = np.einsum("ij,ij->i", d, d) < max_range ** 2
    pc = d[near] @ T_map_cam[:3, :3]
    ok = pc[:, 2] > 0.3
    pc, col = pc[ok], rgb[near][ok]
    u, v = project(pc, K)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, col = u[ok].astype(int), v[ok].astype(int), pc[ok, 2], col[ok]
    cell = v * W + u
    o = np.lexsort((z, cell))
    first = o[np.r_[True, cell[o][1:] != cell[o][:-1]]]
    img = np.full((H, W, 3), 0, np.uint8)
    dep = np.full((H, W), np.inf)
    img[v[first], u[first]] = col[first]
    dep[v[first], u[first]] = z[first]
    return img, dep


def edge_score(photo_gray, dep):
    """Gradient moyen de la photo sur les bords de profondeur du rendu (plus haut = mieux aligné)."""
    gx = np.abs(np.diff(photo_gray, axis=1, prepend=photo_gray[:, :1]))
    gy = np.abs(np.diff(photo_gray, axis=0, prepend=photo_gray[:1]))
    G = np.hypot(gx, gy)
    fin = np.isfinite(dep)
    e = np.zeros_like(fin)
    a, b = dep[:, 1:], dep[:, :-1]
    ok = fin[:, 1:] & fin[:, :-1]
    jump = np.zeros_like(ok)
    jump[ok] = np.abs(a[ok] - b[ok]) > 0.15 * np.minimum(a[ok], b[ok])
    e[:, 1:] |= jump
    return float(G[e].mean() / max(G.mean(), 1e-6)) if e.sum() > 50 else float("nan")


def cam_time_offset(d: str, sl: str, gt_name: str, n: int = 12, dts=None, radius: float = 80.0) -> dict:
    """Décalage temporel de la caméra : score de bords (photo vs rendu de la carte TLS) en fonction du décalage
    ajouté au temps des images, avec la trajectoire retenue (GT recalée + gyroscope)."""
    from splatpipe.common.pcd import read_pcd, rgb_from_float
    from splatpipe.stage0.diterpp_lidar import GT_DT_S, build_trajectory, fit_imu, gt_trajectory

    dts = np.round(np.arange(-0.10, 0.1001, 0.01), 3) if dts is None else dts
    gt1 = gt_trajectory(f"{d}/{gt_name}", GT_DT_S)
    imu = np.load(f"{d}/{sl}/imu_body.npz")
    traj = build_trajectory(gt1, imu["t"], imu["gyro"], fit_imu(gt1, imu["t"], imu["gyro"]))
    m = read_pcd(f"{d}/Lawn_GT.pcd")
    xyz = np.stack([m["x"], m["y"], m["z"]], 1).astype(np.float64)
    rgb = rgb_from_float(m["rgb"])
    imgs = sorted(glob.glob(f"{d}/{sl}/cam/*.jpg"))
    # sensibilité ∝ vitesse angulaire : images aux rotations les plus rapides (gyro lissé 0,1 s), espacées ≥ 2 s
    ti = np.array([int(os.path.basename(f)[:-4]) / 1e9 for f in imgs])
    wn = np.linalg.norm(imu["gyro"], axis=1)
    ker = np.ones(50) / 50
    rate = np.interp(ti, imu["t"], np.convolve(wn, ker, mode="same"))
    chosen = []
    for i in np.argsort(-rate):
        if all(abs(ti[i] - ti[j]) > 2.0 for j in chosen):
            chosen.append(i)
        if len(chosen) == n:
            break
    imgs = [imgs[i] for i in sorted(chosen)]
    rates = [float(rate[i]) for i in sorted(chosen)]
    ci = json.load(open(f"{d}/{sl}/camera_info.json"))
    W, H = ci["width"], ci["height"]
    scores = np.zeros((len(imgs), len(dts)))
    for i, f in enumerate(imgs):
        t0 = int(os.path.basename(f)[:-4]) / 1e9
        gray = np.asarray(Image.open(f).convert("L"), np.float64)
        c = traj.interp(np.array([t0]))[0][0]
        near = np.einsum("ij,ij->i", xyz - c, xyz - c) < radius ** 2
        for j, dt in enumerate(dts):
            T = traj.matrices(np.array([t0 + dt]))[0][0] @ T_LIDAR_CAMERA
            _, dep = render(xyz[near], rgb[near], T, K_AUTEURS, W, H, max_range=radius)
            scores[i, j] = edge_score(gray, dep)
        print(os.path.basename(f), "meilleur dt", float(dts[int(np.nanargmax(scores[i]))]), flush=True)
    mean = np.nanmean(scores / np.nanmax(scores, axis=1, keepdims=True), axis=0)
    best = float(dts[int(np.nanargmax(mean))])
    per = [float(dts[int(np.nanargmax(r))]) for r in scores]
    return {"dts": dts.tolist(), "score_moyen_normalise": mean.tolist(), "meilleur_dt_s": best,
            "meilleur_par_image": per, "vitesse_angulaire_rad_s": rates, "n_images": len(imgs)}


def main():
    from scipy.spatial.transform import Rotation, Slerp

    from splatpipe.common.pcd import read_pcd, rgb_from_float

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data_local/diterpp/lawn")
    ap.add_argument("--slice", default="upper_day_604_784")
    ap.add_argument("--gt", default="Lawn_upper_day_gt.tum")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--dt", type=float, default=0.0, help="décalage ajouté au temps des images (s)")
    ap.add_argument("--find-dt", action="store_true", help="estime le décalage temporel de la caméra")
    args = ap.parse_args()
    if args.find_dt:
        rep = cam_time_offset(args.dir, args.slice, args.gt)
        os.makedirs(f"{args.dir}/{args.slice}/stage0", exist_ok=True)
        json.dump(rep, open(f"{args.dir}/{args.slice}/stage0/cam_time_offset.json", "w"), indent=1)
        print(json.dumps({k: rep[k] for k in ("meilleur_dt_s", "meilleur_par_image")}))
        return
    m = read_pcd(f"{args.dir}/Lawn_GT.pcd")
    xyz = np.stack([m["x"], m["y"], m["z"]], 1).astype(np.float64)
    rgb = rgb_from_float(m["rgb"])
    g = np.loadtxt(f"{args.dir}/{args.gt}")
    sl = Slerp(g[:, 0], Rotation.from_quat(g[:, 4:8]))
    ci = json.load(open(f"{args.dir}/{args.slice}/camera_info.json"))
    W, H = ci["width"], ci["height"]
    K_pilote = dict(fx=ci["K"][0], fy=ci["K"][4], cx=ci["K"][2], cy=ci["K"][5], D=[0, 0, 0, 0, 0])
    imgs = sorted(glob.glob(f"{args.dir}/{args.slice}/cam/*.jpg"))
    out_dir = f"{args.dir}/{args.slice}/overlay"
    os.makedirs(out_dir, exist_ok=True)
    rep = []
    for f in imgs[len(imgs) // (args.n + 1)::len(imgs) // (args.n + 1)][:args.n]:
        t = int(os.path.basename(f)[:-4]) / 1e9 + args.dt
        T = np.eye(4)
        T[:3, :3] = sl([t]).as_matrix()[0]
        T[:3, 3] = [np.interp(t, g[:, 0], g[:, k]) for k in (1, 2, 3)]
        T_map_cam = T @ T_LIDAR_CAMERA
        photo = np.asarray(Image.open(f).convert("RGB"))
        gray = photo.mean(2)
        row = {"image": os.path.basename(f)}
        tiles = [photo]
        for name, K in (("auteurs", K_AUTEURS), ("pilote", K_pilote)):
            img, dep = render(xyz, rgb, T_map_cam, K, W, H)
            row[name] = round(edge_score(gray, dep), 3)
            tiles.append(img)
        rep.append(row)
        sheet = Image.new("RGB", (3 * W, H + 20), "white")
        for i, tl in enumerate(tiles):
            sheet.paste(Image.fromarray(tl), (i * W, 20))
        ImageDraw.Draw(sheet).text((5, 4), f"photo | carte TLS, calib auteurs ({row['auteurs']}) | calib pilote ({row['pilote']})",
                                   fill=(0, 0, 0))
        sheet.save(f"{out_dir}/{os.path.basename(f)}")
        print(json.dumps(row), flush=True)
    json.dump(rep, open(f"{out_dir}/overlay.json", "w"), indent=1)


if __name__ == "__main__":
    main()
