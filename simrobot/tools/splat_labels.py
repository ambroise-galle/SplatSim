"""Étiquetage sémantique a posteriori du splat (desktop, gsplat + transformers).

1. `bakeoff` : quelques rendus du splat segmentés par plusieurs modèles ADE20K, côte à côte (choix du modèle).
2. `segment` : rendus aux poses des caméras GaussGym (avant, gauche, droite), segmentés par le modèle retenu →
   `vues/<cam>_<i>.png` (classe ADE20K par pixel) + `conf/<cam>_<i>.png` (probabilité de la classe, 0–255).
3. `lift` : report sur les gaussiennes. Pour chaque vue, on rend une image à K canaux dont la couleur de chaque
   gaussienne est une inconnue nulle ; la dérivée de Σ_pixels <rendu, one-hot(classe) × confiance> par rapport à la
   couleur d'une gaussienne est exactement Σ_pixels w_i(p) × vote(p), où w_i(p) = T_i α_i est son poids dans le
   mélange du pixel (occlusion comprise). Une rétropropagation par vue donne donc les votes pondérés de toutes les
   gaussiennes. Une vue sur 10 est mise de côté pour `check`.
4. `check` : rendu des étiquettes reportées aux poses mises de côté, comparé à la segmentation de ces vues (accord
   par pixel, IoU par classe) + planche.

Les rendus étant faits sur le splat lui-même, image et gaussiennes sont alignées par construction (pas d'erreur de
pose ou de calibration, contrairement aux photos).

  cd ~/dino && HF_HOME=/mnt/data/dino/hf_cache PYTHONPATH=. venv/bin/python simrobot/tools/splat_labels.py segment \
      --scene out/simrobot/scenes/HOB-1_full --out out/simrobot/labels/HOB-1_full
"""

from __future__ import annotations

import argparse
import colorsys
import glob
import json
import os
import time
import zlib

import numpy as np
import torch

CAMS = ("hdr_front", "hdr_left", "hdr_right")
MODELS = {
    "eomt3": "tue-mps/eomt-dinov3-ade-semantic-large-512",
    "m2f": "facebook/mask2former-swin-large-ade-semantic",
    "oneformer": "shi-labs/oneformer_ade20k_swin_large",
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SKY_BG = (0.66, 0.73, 0.80)

# couleurs lisibles pour les classes ADE20K de la forêt ; les autres : couleur tirée du nom
PALETTE = {
    "tree": (30, 105, 35), "plant": (150, 205, 60), "grass": (205, 235, 95), "earth": (135, 95, 55),
    "land": (180, 125, 75), "path": (225, 185, 125), "dirt track": (200, 150, 90), "rock": (128, 128, 128),
    "water": (35, 120, 255), "sky": (160, 205, 255), "field": (235, 220, 120), "flower": (255, 120, 200),
    "mountain": (110, 90, 125), "hill": (150, 130, 105), "sand": (235, 215, 165), "person": (255, 0, 0),
    "fence": (255, 150, 0), "wall": (120, 120, 185), "building": (185, 75, 75), "road": (85, 85, 85),
    "sidewalk": (175, 175, 175), "pole": (255, 255, 0), "signboard": (255, 0, 255), "palm": (70, 150, 70),
    "bridge": (200, 100, 0), "river": (0, 80, 200), "lake": (0, 60, 180), "sea": (0, 40, 160),
    "floor": (160, 140, 120), "canopy": (60, 165, 125),
}


def short(name):
    return name.split(",")[0].strip()


def class_color(name):
    s = short(name)
    if s in PALETTE:
        return PALETTE[s]
    h = (zlib.crc32(s.encode()) % 1000) / 1000
    return tuple(int(255 * c) for c in colorsys.hsv_to_rgb(h, 0.65, 0.95))


def palette_lut(names):
    lut = np.zeros((256, 3), np.uint8)
    for i, n in enumerate(names):
        lut[i] = class_color(n)
    lut[255] = (0, 0, 0)
    return lut


# ---------------------------------------------------------------------------------------------------- splat, poses

def load_gaussians(ply, dev="cuda", sh=True):
    with open(ply, "rb") as f:
        props, n = [], 0
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                props.append((line.split()[-1], "<f4"))
            elif line == "end_header":
                break
        V = np.fromfile(f, dtype=np.dtype(props), count=n)
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32), device=dev)  # noqa: E731
    G = {"means": t(np.stack([V["x"], V["y"], V["z"]], 1))}
    q = t(np.stack([V[f"rot_{i}"] for i in range(4)], 1))
    G["quats"] = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    G["scales"] = torch.exp(t(np.stack([V[f"scale_{i}"] for i in range(3)], 1)))
    G["opac"] = torch.sigmoid(t(V["opacity"]))
    if sh:
        nrest = sum(1 for p, _ in props if p.startswith("f_rest_"))
        dc = t(np.stack([V[f"f_dc_{i}"] for i in range(3)], 1))[:, None, :]
        rest = t(np.stack([V[f"f_rest_{i}"] for i in range(nrest)], 1)).view(n, 3, nrest // 3).transpose(1, 2)
        G["sh"] = torch.cat([dc, rest], 1).contiguous()
        G["sh_degree"] = int(round(np.sqrt(nrest // 3 + 1))) - 1
    del V
    return G


def c2w_from(p, f, u):
    f, u = np.asarray(f, float), np.asarray(u, float)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = np.cross(f, u), -u, f, p     # OpenCV : x droite, y bas, z avant
    return M


def views(scene, W, H, stride=4, holdout=10, cams=CAMS):
    """Poses des caméras GaussGym (repère physique) → liste de vues {nom, cam, i, viewmat (repère du PLY), K,
    test}. Une vue sur `holdout` (dans l'ordre du trajet) est mise de côté."""
    meta = json.load(open(f"{scene}/scene.json"))
    tr = json.load(open(f"{scene}/trajets.json"))["cameras"]
    M = np.eye(4)
    M[:3, :3] = np.asarray(meta["ply_to_physics"]).T                     # physique → PLY
    out = []
    for cam in cams:
        c = tr[cam]
        fy = H / 2 / np.tan(np.radians(c["fov_y_deg"]) / 2)
        K = np.array([[fy, 0, W / 2], [0, fy, H / 2], [0, 0, 1]])
        for k, i in enumerate(range(0, len(c["p"]), stride)):
            out.append({"nom": f"{cam}_{i:04d}", "cam": cam, "i": i, "K": K, "test": k % holdout == holdout // 2,
                        "viewmat": np.linalg.inv(M @ c2w_from(c["p"][i], c["f"][i], c["u"][i]))})
    return out


def tens(a, dev="cuda"):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)


def render_rgb(G, v, W, H, dev="cuda"):
    from gsplat import rasterization

    with torch.no_grad():
        img, alpha, _ = rasterization(G["means"], G["quats"], G["scales"], G["opac"], G["sh"],
                                      tens(v["viewmat"], dev)[None], tens(v["K"], dev)[None], W, H,
                                      sh_degree=G["sh_degree"], near_plane=0.05, far_plane=200.0, packed=True,
                                      backgrounds=torch.tensor([SKY_BG], device=dev))
    return (img[0].clamp(0, 1) * 255).byte().cpu().numpy(), alpha[0, ..., 0].cpu().numpy()


GSPLAT_CHANNELS = (1, 2, 3, 4, 5, 6, 8, 9, 16, 17, 21, 23, 24, 32, 33, 64, 65, 128, 129, 256, 257, 512, 513)


def padded(K):
    """Nombre de canaux compilé dans gsplat (Config.h) immédiatement ≥ K."""
    return next(c for c in GSPLAT_CHANNELS if c >= K)


def cull(G, v, W, H, far=200.0, dev="cuda"):
    """Indices des gaussiennes qui peuvent toucher l'image (avant, à moins de `far`, dans le cône élargi de 3 σ)."""
    Vm = tens(v["viewmat"], dev)
    pc = G["means"] @ Vm[:3, :3].T + Vm[:3, 3]
    r = 3 * G["scales"].max(1).values
    K = v["K"]
    tx, ty = W / 2 / K[0, 0] * 1.05, H / 2 / K[1, 1] * 1.05
    z = pc[:, 2]
    ok = (z > 0.05) & (z < far) & (pc[:, 0].abs() <= z * tx + r) & (pc[:, 1].abs() <= z * ty + r)
    return ok.nonzero().squeeze(1)


def contributors(G, v, W, H, dev="cuda"):
    """Indices des gaussiennes de poids de mélange non nul dans la vue (passe à 1 canal + rétropropagation : la
    dérivée du rendu sommé est Σ_p T_i α_i). Les autres (cachées, ou sous le seuil d'opacité de gsplat) ne changent
    aucun pixel : les retirer laisse le rendu identique, et la passe à K canaux tient en mémoire (14 M de gaussiennes
    dans le cône, ~1,3 M contributrices)."""
    from gsplat import rasterization

    idx = cull(G, v, W, H)
    f1 = torch.zeros(len(idx), 1, device=dev, requires_grad=True)
    img, _, _ = rasterization(G["means"][idx], G["quats"][idx], G["scales"][idx], G["opac"][idx], f1,
                              tens(v["viewmat"])[None], tens(v["K"])[None], W, H, near_plane=0.05, far_plane=200.0,
                              packed=True)
    img.sum().backward()
    return idx[f1.grad[:, 0] > 0], len(idx)


# ---------------------------------------------------------------------------------------------------- segmenteurs

class Segmenter:
    """Segmentation sémantique ADE20K (150 classes). `scores(img)` → (150, H, W) scores de classe (sommes des
    masques pondérés par la probabilité de classe, comme le post-traitement de transformers)."""

    def __init__(self, name, dev="cuda", tile=None, fp16=True):
        import transformers as tf

        self.name, self.dev, self.tile, self.fp16 = name, dev, tile, fp16
        mid = MODELS[name]
        if name == "eomt3":
            self.model = tf.EomtDinov3ForUniversalSegmentation.from_pretrained(mid)
            self.proc = tf.AutoImageProcessor.from_pretrained(mid)
        elif name == "m2f":
            self.model = tf.Mask2FormerForUniversalSegmentation.from_pretrained(mid)
            self.proc = tf.AutoImageProcessor.from_pretrained(mid)
        elif name == "oneformer":
            self.model = tf.OneFormerForUniversalSegmentation.from_pretrained(mid)
            self.proc = tf.OneFormerProcessor.from_pretrained(mid)
        self.model = self.model.to(dev).eval()
        cfg = self.model.config
        self.names = [cfg.id2label[i] for i in range(len(cfg.id2label))]

    def _queries_to_scores(self, cls, masks, size):
        masks = torch.nn.functional.interpolate(masks.float(), size=size, mode="bilinear", align_corners=False)
        return torch.einsum("bqc,bqhw->bchw", cls.float().softmax(-1)[..., :-1], masks.sigmoid())

    def scores(self, img):
        H, W = img.shape[:2]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            if self.name == "eomt3" and self.tile:
                # fenêtres de `tile` px à la résolution du rendu (≈ le champ d'une photo ADE20K), recouvrement moyenné
                T = self.tile
                x = torch.as_tensor(img, device=self.dev).permute(2, 0, 1)[None].float() / 255
                x = (x - MEAN.to(self.dev)) / STD.to(self.dev)
                xs = np.linspace(0, W - T, max(1, int(np.ceil((W - T) / (T * 0.6))) + 1)).round().astype(int)
                ys = np.linspace(0, H - T, max(1, int(np.ceil((H - T) / (T * 0.6))) + 1)).round().astype(int)
                acc = torch.zeros(len(self.names), H, W, device=self.dev)
                cnt = torch.zeros(1, H, W, device=self.dev)
                boxes = [(y0, x0) for y0 in ys for x0 in xs]
                for b in range(0, len(boxes), 3):
                    bb = boxes[b:b + 3]
                    out = self.model(pixel_values=torch.cat([x[..., y0:y0 + T, x0:x0 + T] for y0, x0 in bb]))
                    s = self._queries_to_scores(out.class_queries_logits, out.masks_queries_logits, (T, T))
                    for k, (y0, x0) in enumerate(bb):
                        acc[:, y0:y0 + T, x0:x0 + T] += s[k]
                        cnt[:, y0:y0 + T, x0:x0 + T] += 1
                return acc / cnt
            if self.name == "oneformer":
                inp = self.proc(images=img, task_inputs=["semantic"], return_tensors="pt").to(self.dev)
            elif self.name == "m2f":
                inp = self.proc(images=img, size={"shortest_edge": 640, "longest_edge": 2560},
                                return_tensors="pt").to(self.dev)
            else:
                inp = self.proc(images=img, return_tensors="pt").to(self.dev)
            out = self.model(**inp)
            r = self.proc.post_process_semantic_segmentation(out, target_sizes=[(H, W)],
                                                             **({"return_segmentation_scores": True}
                                                                if self.name == "eomt3" else {}))[0]
            if self.name == "eomt3":
                return r.segmentation_scores.float()
            # Mask2Former / OneFormer : argmax seulement (comparaison visuelle)
            return torch.nn.functional.one_hot(r.long(), len(self.names)).permute(2, 0, 1).float()

    def label(self, img):
        s = self.scores(img)
        tot = s.sum(0).clamp_min(1e-6)
        p, lab = s.max(0)
        return lab.byte().cpu().numpy(), (p / tot * 255).round().byte().cpu().numpy()


# ---------------------------------------------------------------------------------------------------- planches

def font(size=18):
    from PIL import ImageFont

    for fp in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return None


def sheet(rows, titles, legend, names, out, scale=0.5):
    """rows : listes d'images RGB uint8 de même taille ; legend : ids de classes à afficher."""
    from PIL import Image, ImageDraw

    h, w = rows[0][0].shape[:2]
    w2, h2, gap = int(w * scale), int(h * scale), 4
    leg_h = 26 * ((len(legend) + 5) // 6) + 8
    S = Image.new("RGB", (len(titles) * (w2 + gap), 28 + len(rows) * (h2 + gap) + leg_h), (20, 20, 24))
    d = ImageDraw.Draw(S)
    f = font(17)
    for c, t in enumerate(titles):
        d.text((c * (w2 + gap) + 6, 5), t, fill=(240, 240, 240), font=f)
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            S.paste(Image.fromarray(im).resize((w2, h2), Image.NEAREST if c else Image.BILINEAR),
                    (c * (w2 + gap), 28 + r * (h2 + gap)))
    y0 = 28 + len(rows) * (h2 + gap) + 4
    cw = S.width // 6
    for k, cid in enumerate(legend):
        x, y = (k % 6) * cw + 6, y0 + (k // 6) * 26
        d.rectangle([x, y + 3, x + 18, y + 19], fill=class_color(names[cid]))
        d.text((x + 24, y + 1), f"{cid} {short(names[cid])}", fill=(230, 230, 230), font=f)
    S.save(out, quality=90)
    print(out, S.size)


def blend(img, lab, lut, a=0.0):
    col = lut[lab]
    return col if a == 0 else (img * a + col * (1 - a)).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------- commandes

def cmd_bakeoff(a):
    W, H = a.W, a.H
    V = [v for v in views(a.scene, W, H, stride=1) if v["cam"] == "hdr_front"]
    allv = views(a.scene, W, H, stride=1)
    picks = []
    n = len(V)
    for k, frac in enumerate(np.linspace(0.06, 0.94, a.n)):
        cam = CAMS[k % 3]
        i = int(frac * n)
        picks.append(next(v for v in allv if v["cam"] == cam and v["i"] == i))
    G = load_gaussians(f"{a.scene}/merged.ply")
    imgs = [render_rgb(G, v, W, H)[0] for v in picks]
    del G
    torch.cuda.empty_cache()
    configs = [("eomt3", 512, "EoMT-DINOv3-L, fenêtres 512"), ("eomt3", None, "EoMT-DINOv3-L, image réduite"),
               ("m2f", None, "Mask2Former Swin-L"), ("oneformer", None, "OneFormer Swin-L")]
    cols, names, stats = [], None, {}
    for name, tile, title in configs:
        S = Segmenter(name, tile=tile)
        names = S.names
        S.label(imgs[0])
        torch.cuda.synchronize()
        t0 = time.time()
        labs = [S.label(im)[0] for im in imgs]
        torch.cuda.synchronize()
        stats[title] = round((time.time() - t0) / len(imgs), 3)
        cols.append((title, labs))
        del S
        torch.cuda.empty_cache()
    lut = palette_lut(names)
    frac = np.zeros(256)
    for _, labs in cols:
        for lab in labs:
            frac += np.bincount(lab.ravel(), minlength=256) / lab.size / (len(labs) * len(cols))
    legend = [int(c) for c in np.argsort(-frac) if frac[c] > 0.003]
    rows = [[im] + [lut[labs[r]] for _, labs in cols] for r, im in enumerate(imgs)]
    sheet(rows, ["rendu du splat"] + [t for t, _ in cols], legend, names, a.out)
    # accord entre modèles (même classe au pixel)
    agree = {}
    for t, labs in cols[1:]:
        agree[t] = round(float(np.mean([np.mean(l1 == l2) for l1, l2 in zip(cols[0][1], labs)])), 3)
    print(json.dumps({"s_par_image": stats, "accord_avec_" + cols[0][0]: agree,
                      "classes": {short(names[c]): round(float(frac[c]), 3) for c in legend}}, ensure_ascii=False))


def read_png(path):
    from PIL import Image

    return np.asarray(Image.open(path))


def cmd_segment(a):
    from PIL import Image

    os.makedirs(f"{a.out}/rgb", exist_ok=True)
    os.makedirs(f"{a.out}/lab", exist_ok=True)
    os.makedirs(f"{a.out}/conf", exist_ok=True)
    params = {"W": a.W, "H": a.H, "stride": a.stride, "holdout": a.holdout, "model": MODELS[a.model], "tile": a.tile}
    if os.path.exists(f"{a.out}/params.json"):
        old = json.load(open(f"{a.out}/params.json"))
        assert old == params, f"paramètres différents de ceux du dossier : {old}"
    json.dump(params, open(f"{a.out}/params.json", "w"))
    V = views(a.scene, a.W, a.H, a.stride, a.holdout)
    todo = [v for v in V if not os.path.exists(f"{a.out}/lab/{v['nom']}.png")]
    print(f"{len(V)} vues, {len(V) - len(todo)} déjà faites (reprise), {sum(v['test'] for v in V)} mises de côté",
          flush=True)
    G = load_gaussians(f"{a.scene}/merged.ply")
    S = Segmenter(a.model, tile=a.tile)
    json.dump(S.names, open(f"{a.out}/classes.json", "w"))
    t0, hist = time.time(), np.zeros(256)
    for k, v in enumerate(todo):
        img, alpha = render_rgb(G, v, a.W, a.H)
        lab, conf = S.label(img)
        Image.fromarray(img).save(f"{a.out}/rgb/{v['nom']}.jpg", quality=92)
        Image.fromarray(conf).save(f"{a.out}/conf/{v['nom']}.png")
        Image.fromarray(lab).save(f"{a.out}/lab/{v['nom']}.png")
        hist += np.bincount(lab.ravel(), minlength=256)
        if k % 100 == 0 or k == len(todo) - 1:
            top = np.argsort(-hist)[:8]
            print(f"{k + 1}/{len(todo)} {(time.time() - t0) / (k + 1):.2f} s/vue — "
                  + ", ".join(f"{short(S.names[c])} {hist[c] / hist.sum():.1%}" for c in top if hist[c]), flush=True)
    print("FIN_SEGMENT", flush=True)


def class_hist(out, V):
    h = np.zeros(256)
    for v in V:
        if not v["test"]:
            lab = read_png(f"{out}/lab/{v['nom']}.png")
            h += np.bincount(lab.ravel(), weights=read_png(f"{out}/conf/{v['nom']}.png").ravel() / 255.0,
                             minlength=256)
    return h / h.sum()


def cmd_lift(a):
    p = json.load(open(f"{a.out}/params.json"))
    names = json.load(open(f"{a.out}/classes.json"))
    W, H = p["W"], p["H"]
    V = views(a.scene, W, H, p["stride"], p["holdout"])
    hist = class_hist(a.out, V)
    order = [int(c) for c in np.argsort(-hist) if hist[c] > 0]
    classes = [c for c in order if hist[c] >= a.min_frac][:a.K - 1]
    rest = float(sum(hist[c] for c in order if c not in classes))
    print(f"{len(classes)} classes gardées ({1 - rest:.4%} des pixels pondérés), {len(order) - len(classes)} "
          f"regroupées en « autre » ({rest:.4%}) :", ", ".join(f"{short(names[c])} {hist[c]:.2%}" for c in classes),
          flush=True)
    K = len(classes) + 1
    dev = "cuda"
    lut = torch.full((256,), K - 1, dtype=torch.long, device=dev)
    lut[torch.tensor(classes, device=dev)] = torch.arange(K - 1, device=dev)
    G = load_gaussians(f"{a.scene}/merged.ply", sh=False)
    N = G["means"].shape[0]
    acc = torch.zeros(N, K, device=a.acc)
    from gsplat import rasterization

    train = [v for v in V if not v["test"]]
    t0 = time.time()
    for k, v in enumerate(train):
        lab = lut[torch.as_tensor(read_png(f"{a.out}/lab/{v['nom']}.png"), device=dev).long()]
        conf = torch.as_tensor(read_png(f"{a.out}/conf/{v['nom']}.png"), device=dev).float() / 255
        idx, n_cone = contributors(G, v, W, H)
        feat = torch.zeros(len(idx), padded(K), device=dev, requires_grad=True)
        img, _, _ = rasterization(G["means"][idx], G["quats"][idx], G["scales"][idx], G["opac"][idx], feat,
                                  tens(v["viewmat"])[None], tens(v["K"])[None], W, H, near_plane=0.05,
                                  far_plane=200.0, packed=True)
        Y = torch.nn.functional.one_hot(lab, padded(K)).float() * conf[..., None]
        (img[0] * Y).sum().backward()
        g = feat.grad[:, :K]
        nz = g.sum(1) > 0
        acc.index_add_(0, idx[nz].to(a.acc), g[nz].to(a.acc))
        del feat, img, g
        if k % 200 == 0 or k == len(train) - 1:
            print(f"{k + 1}/{len(train)} {(time.time() - t0) / (k + 1):.3f} s/vue, {n_cone / 1e6:.1f} M dans le cône, "
                  f"{len(idx) / 1e6:.2f} M contributrices, {int(nz.sum()) / 1e6:.2f} M votantes, "
                  f"GPU max {torch.cuda.max_memory_allocated() / 1e9:.1f} Go",
                  flush=True)
    tot = acc.sum(1)
    best_w, best = acc.max(1)
    purity = (best_w / tot.clamp_min(1e-12))
    ade = torch.tensor(classes + [254], dtype=torch.uint8, device=acc.device)[best]
    seen = tot > a.min_weight
    ade[~seen] = 255
    frac = (acc / tot.clamp_min(1e-12)[:, None] * 255).round().to(torch.uint8)
    np.savez(f"{a.out}/gaussiennes.npz", label=ade.cpu().numpy(), purity=(purity * 255).round().byte().cpu().numpy(),
             weight=tot.half().cpu().numpy(), classes=np.array(classes + [254]), names=np.array(names),
             note="label : classe ADE20K par gaussienne (ligne de merged.ply) ; 254 = autre classe ADE (rare), "
                  "255 = jamais vue ; purity = part du vote gagnant ; weight = Σ poids de mélange × confiance")
    np.save(f"{a.out}/votes_frac.npy", frac.cpu().numpy())
    lab_np, op = ade.cpu().numpy(), G["opac"].cpu().numpy()
    stats = {"gaussiennes": N, "vues": len(train), "jamais_vues": round(float((lab_np == 255).mean()), 4),
             "purete_mediane": round(float(purity[seen].median()), 3), "classes_K": K,
             "par_classe": {}}
    for c in classes + [254]:
        m = lab_np == c
        if m.any():
            stats["par_classe"][short(names[c]) if c < 254 else "autre"] = {
                "n": int(m.sum()), "part": round(float(m.mean()), 4), "part_opacite": round(float(op[m].sum() / op.sum()), 4)}
    json.dump(stats, open(f"{a.out}/lift_stats.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))
    print("FIN_LIFT", flush=True)


def cmd_check(a):
    from gsplat import rasterization

    p = json.load(open(f"{a.out}/params.json"))
    names = json.load(open(f"{a.out}/classes.json"))
    W, H = p["W"], p["H"]
    names_sel = a.names.split(",") if a.names else None
    V = [v for v in views(a.scene, W, H, p["stride"], p["holdout"])
         if (v["nom"] in names_sel if names_sel else v["test"])]
    tag = f"_{a.tag}" if a.tag else ""
    L = np.load(f"{a.out}/gaussiennes.npz")
    classes = [int(c) for c in L["classes"]]
    K = len(classes)
    dev = "cuda"
    lut = torch.full((256,), K - 1, dtype=torch.long, device=dev)
    lut[torch.tensor(classes[:-1], device=dev)] = torch.arange(K - 1, device=dev)
    back = np.array(classes[:-1] + [254], np.uint8)
    G = load_gaussians(f"{a.scene}/merged.ply", sh=False)
    frac = torch.as_tensor(np.load(f"{a.out}/votes_frac.npy"))                # CPU, uint8
    conf_m = np.zeros((K, K), np.int64)
    picks = set(range(len(V))) if names_sel else set(np.linspace(0, len(V) - 1, a.n_sheet).round().astype(int).tolist())
    rows, lutc = [], palette_lut(names)
    lutc[254] = (60, 60, 60)
    for k, v in enumerate(V):
        idx, _ = contributors(G, v, W, H)
        f = torch.zeros(len(idx), padded(K), device=dev)
        f[:, :K] = frac[idx.cpu()].to(dev).float() / 255
        with torch.no_grad():
            img, alpha, _ = rasterization(G["means"][idx], G["quats"][idx], G["scales"][idx], G["opac"][idx], f,
                                          tens(v["viewmat"])[None], tens(v["K"])[None], W, H, near_plane=0.05,
                                          far_plane=200.0, packed=True)
        pred = img[0, ..., :K].argmax(-1)
        ok = alpha[0, ..., 0] > 0.5
        seg = lut[torch.as_tensor(read_png(f"{a.out}/lab/{v['nom']}.png"), device=dev).long()]
        conf_m += np.bincount((seg[ok] * K + pred[ok]).cpu().numpy(), minlength=K * K).reshape(K, K)
        if k in picks:
            rgb = read_png(f"{a.out}/rgb/{v['nom']}.jpg")
            s_np, p_np, ok_np = back[seg.cpu().numpy()], back[pred.cpu().numpy()], ok.cpu().numpy()
            p_img = lutc[p_np]
            p_img[~ok_np] = 0
            diff = rgb // 3
            diff[ok_np & (s_np != p_np)] = (255, 40, 40)
            rows.append([rgb, lutc[s_np], p_img, diff])
    tp = np.diag(conf_m).astype(float)
    iou = tp / np.maximum(conf_m.sum(0) + conf_m.sum(1) - tp, 1)
    sup = conf_m.sum(1) / conf_m.sum()
    res = {"vues_mises_de_cote": len(V), "accord_pixels": round(float(tp.sum() / conf_m.sum()), 4),
           "mIoU_classes_>0.1%": round(float(iou[sup > 0.001].mean()), 4),
           "par_classe": {(short(names[c]) if c < 254 else "autre"): {"part": round(float(sup[i]), 4),
                                                                       "IoU": round(float(iou[i]), 3)}
                          for i, c in enumerate(classes) if sup[i] > 0.0005}}
    res["vues"] = "choisies : " + a.names if names_sel else "mises de côté"
    json.dump(res, open(f"{a.out}/check{tag}.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))
    legend = [c for i, c in enumerate(classes[:-1]) if sup[i] > 0.002]
    sheet(rows, ["rendu (vue choisie)" if names_sel else "rendu (vue mise de côté)", "segmentation de ce rendu",
                 "étiquettes reportées, re-rendues", "désaccords (rouge)"], legend, names, f"{a.out}/check{tag}.jpg")


def cmd_export(a):
    """merged.ply recoloré par classe (SH d'ordre 0 seulement) → PLY pour les niveaux de détail du visualiseur."""
    L = np.load(f"{a.out}/gaussiennes.npz")
    names = [str(n) for n in L["names"]]
    lut = palette_lut(names).astype(np.float32) / 255
    lut[254] = (0.45, 0.45, 0.45)                          # autre classe ADE20K (rare)
    lut[255] = (0.25, 0.25, 0.25)                          # jamais vue depuis les poses de capture
    src = f"{a.scene}/merged.ply"
    with open(src, "rb") as f:
        header, props, n = b"", [], 0
        while True:
            line = f.readline()
            header += line
            s = line.decode().strip()
            if s.startswith("element vertex"):
                n = int(s.split()[-1])
            elif s.startswith("property"):
                props.append((s.split()[-1], "<f4"))
            elif s == "end_header":
                break
        V = np.fromfile(f, dtype=np.dtype(props), count=n)
    assert n == len(L["label"])
    col = lut[L["label"]]
    for i in range(3):
        V[f"f_dc_{i}"] = (col[:, i] - 0.5) / 0.28209479
    for p, _ in props:
        if p.startswith("f_rest_"):
            V[p] = 0
    with open(a.dst, "wb") as f:
        f.write(header)
        V.tofile(f)
    if a.legend:
        cnt = np.bincount(L["label"], minlength=256) / n
        leg = [{"id": int(c), "nom": short(names[c]) if c < 254 else ("autre" if c == 254 else "jamais vue"),
                "rgb": [int(round(255 * x)) for x in lut[c]], "part": round(float(cnt[c]), 4)}
               for c in np.argsort(-cnt) if cnt[c] >= 0.0005]
        os.makedirs(os.path.dirname(a.legend) or ".", exist_ok=True)
        json.dump(leg, open(a.legend, "w"), ensure_ascii=False)
    print(a.dst, n)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bakeoff")
    b.add_argument("--scene", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=6)
    b.add_argument("--W", type=int, default=1152)
    b.add_argument("--H", type=int, default=768)
    s = sub.add_parser("segment")
    s.add_argument("--scene", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--model", default="eomt3", choices=list(MODELS))
    s.add_argument("--tile", type=int, default=512, help="fenêtres glissantes (EoMT) ; 0 = image réduite")
    s.add_argument("--stride", type=int, default=4, help="une pose sur N par caméra")
    s.add_argument("--holdout", type=int, default=10)
    s.add_argument("--W", type=int, default=1152)
    s.add_argument("--H", type=int, default=768)
    li = sub.add_parser("lift")
    li.add_argument("--scene", required=True)
    li.add_argument("--out", required=True)
    li.add_argument("--K", type=int, default=32, help="canaux : K−1 classes les plus fréquentes + « autre »")
    li.add_argument("--min-frac", type=float, default=1e-5)
    li.add_argument("--min-weight", type=float, default=1e-3)
    li.add_argument("--acc", default="cuda", choices=["cuda", "cpu"])
    c = sub.add_parser("check")
    c.add_argument("--scene", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--n-sheet", type=int, default=6)
    c.add_argument("--names", default=None, help="vues choisies (noms séparés par des virgules) au lieu des mises de côté")
    c.add_argument("--tag", default=None, help="suffixe des fichiers de sortie")
    e = sub.add_parser("export")
    e.add_argument("--scene", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--dst", required=True, help="PLY recoloré par classe")
    e.add_argument("--legend", default=None, help="légende pour le visualiseur (lod_sem/legende.json)")
    a = ap.parse_args()
    if getattr(a, "tile", None) == 0:
        a.tile = None
    {"bakeoff": cmd_bakeoff, "segment": cmd_segment, "lift": cmd_lift, "check": cmd_check,
     "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
