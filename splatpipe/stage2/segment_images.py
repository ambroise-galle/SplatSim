"""Sous-étape 2.1 — Segmentation sémantique des images (ADE20K) sur les GPU Modal.

Modèle : `facebook/mask2former-swin-large-ade-semantic` (voir `splatpipe/stage2/images.py` pour le choix et
la licence). Les classes sont retrouvées **par nom** dans `model.config.id2label` (pas d'identifiant codé en
dur) : `sky` → bit 0 ; `person` (+ `rider` si présent) → bit 1 ; véhicules potentiellement dynamiques
(`car`, `bus`, `truck`, `van`, `bicycle`, `motorbike`/`minibike`) → bit 2 (non masqués par défaut) ;
végétation (`tree`, `grass`, `plant`, `flower`, `palm`, `field` si présent) → bit 3.

Le processeur d'images redimensionne à `PROC_SIZE` (rapport 4:3 conservé, multiple de 32) ; les logits sont
ensuite ré-échantillonnés à la résolution d'origine (1440×1080) par
`Mask2FormerImageProcessor.post_process_semantic_segmentation` (upsampling bilinéaire + argmax), donc jamais
de argmax à basse résolution.

Sorties (par triplet de `cameras/frames.csv`, ≈ 5 632 × 3 ≈ 16 896 images) :
  semantics/<cam>/<stamp_ns>.png   uint8, id de classe ADE20K par pixel (pleine résolution)
  masks/<cam>/<stamp_ns>.png       uint8, bits ciel/personne/véhicule/végétation
  semantics/classes.json           id2label, définition des bits, modèle, révision, prétraitement

Reprise : une image dont les deux fichiers de sortie existent déjà est sautée (vérifié avant chaque appel du
modèle). Le volume est commité régulièrement (tous les `commit_every` images par conteneur, et à la fin).

Lancement (depuis la racine du projet) :
  .venv/bin/modal run splatpipe/stage2/segment_images.py --stage meta                      # id2label, table de bits, licence
  .venv/bin/modal run splatpipe/stage2/segment_images.py --stage pilot --n-pilot 30         # minuterie, extrapolation
  .venv/bin/modal run splatpipe/stage2/segment_images.py --stage full --n-chunks 6          # job complet, reprise automatique
  .venv/bin/modal run splatpipe/stage2/segment_images.py --stage sizes                      # tailles réelles écrites

Ne touche ni `/data/raw` (lecture seule), ni `lidar/`, `lidar_model/`, `traj/`, `map/`, `sensors.yaml`,
`cameras/`, `splits.yaml` (lecture seule). Écrit uniquement sous `semantics/`, `masks/`, `reports/`.
"""

from __future__ import annotations

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume
from splatpipe.stage2.images import segmentation_image

app = modal.App("splatsim-stage2-segment")
vol = data_volume()
image = segmentation_image()

DATASET = "oxford_spires"
SEQ = "2024-03-13-observatory-quarter-01"
CAMS = ("cam0", "cam1", "cam2")

MODEL_ID = "facebook/mask2former-swin-large-ade-semantic"
MODEL_LICENSE_NOTE = (
    "Code Mask2Former (facebookresearch/Mask2Former) : licence MIT. Fiche Hugging Face du point de contrôle : "
    "license=other (aucun texte de licence dédié plus restrictif trouvé lors de la vérification du 17/09/2026). "
    "Usage recherche uniquement, comme le reste du projet."
)

# Résolution donnée au processeur (rapport 4:3 comme les images 1440×1080, multiple de 32 pour size_divisor).
PROC_H = 864
PROC_W = 1152

# Bits des masques (définition figée dans le rapport et `semantics/classes.json`).
BIT_SKY = 0
BIT_PERSON = 1
BIT_VEHICLE = 2
BIT_VEGETATION = 3
SKY_KEYWORDS = {"sky"}
PERSON_KEYWORDS = {"person", "rider"}
VEHICLE_KEYWORDS = {"car", "bus", "truck", "van", "bicycle", "motorbike", "minibike"}
VEGETATION_KEYWORDS = {"tree", "grass", "plant", "flower", "palm", "field"}

N_ADE20K = 150  # nombre de classes attendu (SceneParsing150) ; sert de garde-fou dans `meta`


def scene(seq: str):
    from splatpipe.common.scene import Scene

    return Scene(DATASET, seq, data_root=DATA_MOUNT)


def build_class_table(id2label: dict) -> tuple[list, dict]:
    """id2label ({id(int|str): nom}, éventuellement `"a;b;c"`) → (table de bits [n_classes], détail par id).

    Chaque classe est retrouvée **par nom** : le nom ADE20K peut contenir des synonymes séparés par `;`
    (convention Hugging Face pour ce jeu de données), comparés en minuscules à des ensembles de mots-clés.
    """
    id2label = {int(k): v for k, v in id2label.items()}
    n = max(id2label) + 1
    table = [0] * n
    detail = {}
    for idx, label in sorted(id2label.items()):
        names = {tok.strip().lower() for tok in label.split(";")}
        bit = 0
        matched = []
        if names & SKY_KEYWORDS:
            bit |= 1 << BIT_SKY
            matched.append("sky")
        if names & PERSON_KEYWORDS:
            bit |= 1 << BIT_PERSON
            matched.append("person")
        if names & VEHICLE_KEYWORDS:
            bit |= 1 << BIT_VEHICLE
            matched.append("vehicle")
        if names & VEGETATION_KEYWORDS:
            bit |= 1 << BIT_VEGETATION
            matched.append("vegetation")
        table[idx] = bit
        detail[idx] = {"label": label, "matched_bits": matched}
    return table, detail


def _all_items(sc) -> list[dict]:
    """Une entrée par image des 5 632 triplets de `cameras/frames.csv` (≈ 16 896 images)."""
    frames = sc.load_frames()
    items = []
    n = len(frames["frame_id"])
    for i in range(n):
        stamp_ns = int(frames["stamp_ns"][i])
        for cam in CAMS:
            items.append({"cam": cam, "stamp_ns": stamp_ns, "rel_path": str(frames[f"{cam}_rel_path"][i])})
    return items


# =============================================================================
# Métadonnées du modèle (CPU, léger : config seulement, pas les poids)
# =============================================================================


@app.function(image=image, timeout=300)
def get_class_meta(revision: str | None = None) -> dict:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=revision)
    id2label = {int(k): v for k, v in cfg.id2label.items()}
    table, detail = build_class_table(id2label)
    sha = None
    try:
        from huggingface_hub import HfApi

        sha = HfApi().model_info(MODEL_ID, revision=revision or "main").sha
    except Exception as e:  # pas bloquant : on documente juste l'échec
        sha = f"<inconnu : {e!r}>"
    return {
        "model_id": MODEL_ID,
        "revision_requested": revision,
        "revision_sha": sha,
        "license_note": MODEL_LICENSE_NOTE,
        "n_labels": len(id2label),
        "id2label": id2label,
        "bit_table": table,
        "bit_detail": detail,
        "bits": {"sky": BIT_SKY, "person": BIT_PERSON, "vehicle": BIT_VEHICLE, "vegetation": BIT_VEGETATION},
        "keywords": {"sky": sorted(SKY_KEYWORDS), "person": sorted(PERSON_KEYWORDS),
                     "vehicle": sorted(VEHICLE_KEYWORDS), "vegetation": sorted(VEGETATION_KEYWORDS)},
        "proc_size": {"height": PROC_H, "width": PROC_W},
    }


@app.function(image=image, volumes={DATA_MOUNT: vol}, timeout=300)
def write_classes_json(meta: dict, seq: str = SEQ) -> str:
    import json
    import os

    sc = scene(seq)
    out_dir = sc.path("semantics")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "model_id": meta["model_id"],
        "revision_sha": meta["revision_sha"],
        "license_note": meta["license_note"],
        "preprocessing": {
            "resize": {"height": PROC_H, "width": PROC_W, "note": "aspect 4:3 conservé (1440×1080), "
                       "multiple de 32 (size_divisor)"},
            "postprocessing": "Mask2FormerImageProcessor.post_process_semantic_segmentation : upsampling "
                               "bilinéaire des logits à 1440×1080 puis argmax (jamais d'argmax à basse résolution)",
            "dtype": "fp16 (modèle et pixel_values)",
        },
        "n_labels": meta["n_labels"],
        "id2label": meta["id2label"],
        "bits": meta["bits"],
        "bit_detail": meta["bit_detail"],
        "keywords": meta["keywords"],
    }
    path = f"{out_dir}/classes.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    vol.commit()
    return path


# =============================================================================
# Inférence
# =============================================================================


def _load_model(revision: str | None):
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=revision)
    processor.size = {"height": PROC_H, "width": PROC_W}
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID, revision=revision)
    model.eval().to("cuda").half()
    return processor, model


def _run_batch(processor, model, pil_images: list, target_hw: tuple[int, int]):
    import torch

    inputs = processor(images=pil_images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to("cuda", dtype=torch.float16)
    pixel_mask = inputs.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to("cuda")
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    sizes = [target_hw] * len(pil_images)
    preds = processor.post_process_semantic_segmentation(outputs, target_sizes=sizes)
    return [p.to(torch.uint8).cpu().numpy() for p in preds]


def _process_items(items: list[dict], batch_size: int, seq: str, revision: str | None, class_table: list,
                   commit_every: int, n_load_workers: int = 16) -> dict:
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from PIL import Image

    sc = scene(seq)
    processor, model = _load_model(revision)
    table = np.asarray(class_table, dtype=np.uint8)
    n_classes = len(table)

    def _load(rel_path):
        return Image.open(sc.abs_path(rel_path)).convert("RGB")

    n_done = n_skip = n_err = 0
    errors: list = []
    hist = np.zeros((len(CAMS), n_classes), dtype=np.int64)
    t0 = time.time()
    t_infer = t_load = t_save = 0.0
    since_commit = 0
    pool = ThreadPoolExecutor(max_workers=n_load_workers)

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        todo = []
        for it in batch:
            sem_path = sc.path("semantics", it["cam"], f"{it['stamp_ns']}.png")
            mask_path = sc.path("masks", it["cam"], f"{it['stamp_ns']}.png")
            if os.path.exists(sem_path) and os.path.exists(mask_path):
                n_skip += 1
                continue
            todo.append((it, sem_path, mask_path))
        if not todo:
            continue
        try:
            tl = time.time()
            # Lecture parallélisée (threads) : le Volume Modal est monté en réseau, la latence par fichier
            # domine largement le décodage JPEG lui-même (mesuré : ~0,35 s/image en séquentiel).
            imgs = list(pool.map(_load, [it["rel_path"] for it, _, _ in todo]))
            h, w = imgs[0].height, imgs[0].width
            t_load += time.time() - tl
            ti = time.time()
            preds = _run_batch(processor, model, imgs, (h, w))
            t_infer += time.time() - ti
        except Exception as e:  # noqa: BLE001 - on isole l'erreur par lot, on continue
            for it, _, _ in todo:
                errors.append({"cam": it["cam"], "stamp_ns": it["stamp_ns"], "error": repr(e)[:300]})
            n_err += len(todo)
            continue
        ts = time.time()
        for (it, sem_path, mask_path), pred in zip(todo, preds):
            os.makedirs(os.path.dirname(sem_path), exist_ok=True)
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            # compress_level bas : PNG rapide à écrire, encore ~5-10x plus petit qu'un format non compressé
            # (grandes zones uniformes : ciel, route, façades), la vitesse d'écriture prime sur la taille ici.
            Image.fromarray(pred, mode="L").save(sem_path, compress_level=1)
            bits = table[pred]
            Image.fromarray(bits, mode="L").save(mask_path, compress_level=1)
            ci = CAMS.index(it["cam"])
            hist[ci] += np.bincount(pred.reshape(-1), minlength=n_classes)
            n_done += 1
            since_commit += 1
        t_save += time.time() - ts
        if since_commit >= commit_every:
            vol.commit()
            since_commit = 0
    vol.commit()
    pool.shutdown(wait=False)
    dt = time.time() - t0
    return {
        "n_items": len(items), "n_done": n_done, "n_skip": n_skip, "n_err": n_err,
        "errors": errors[:20], "n_errors_total": len(errors),
        "runtime_s": round(dt, 1), "inference_s": round(t_infer, 1), "load_s": round(t_load, 1),
        "save_s": round(t_save, 1),
        "images_per_s": round(n_done / dt, 3) if dt > 0 else None,
        "hist": hist.tolist(),
    }


@app.function(image=image, volumes={DATA_MOUNT: vol}, gpu="L4", cpu=8, memory=16384, timeout=1800, retries=0)
def pilot_run(n: int = 30, batch_size: int = 4, seq: str = SEQ, revision: str | None = None,
             class_table: list | None = None, n_load_workers: int = 8, seed: int | None = None) -> dict:
    import random
    import time

    sc = scene(seq)
    items = _all_items(sc)
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
    rng = random.Random(seed)
    sample = rng.sample(items, min(n, len(items)))
    res = _process_items(sample, batch_size, seq, revision, class_table, commit_every=10**9,
                         n_load_workers=n_load_workers)
    n_total = len(items)
    hours_needed = None
    if res["images_per_s"]:
        hours_needed = round(n_total / res["images_per_s"] / 3600, 3)
    res["n_total_dataset_images"] = n_total
    res["extrapolated_gpu_hours_single_container"] = hours_needed
    res["extrapolated_cost_usd_at_0.80_per_hour"] = round(hours_needed * 0.80, 2) if hours_needed else None
    return res


@app.function(image=image, volumes={DATA_MOUNT: vol}, gpu="L4", cpu=8, memory=16384, timeout=3 * 3600, retries=2)
def process_chunk(items: list[dict], batch_size: int = 4, seq: str = SEQ, revision: str | None = None,
                  class_table: list | None = None, commit_every: int = 200, n_load_workers: int = 8) -> dict:
    return _process_items(items, batch_size, seq, revision, class_table, commit_every, n_load_workers=n_load_workers)


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=2, memory=4096, timeout=1800)
def list_items(seq: str = SEQ) -> list[dict]:
    return _all_items(scene(seq))


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=2, memory=4096, timeout=1800)
def output_sizes(seq: str = SEQ) -> dict:
    import os

    sc = scene(seq)
    out = {}
    for sub in ("semantics", "masks"):
        n_files, total_bytes = 0, 0
        root = sc.path(sub)
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".png"):
                    n_files += 1
                    total_bytes += os.path.getsize(os.path.join(dirpath, fn))
        out[sub] = {"n_files": n_files, "total_bytes": total_bytes,
                    "mean_bytes": round(total_bytes / n_files, 1) if n_files else None}
    return out


# =============================================================================
# Orchestration locale
# =============================================================================


def _chunkify(items: list, n_chunks: int) -> list[list]:
    n = len(items)
    size = -(-n // n_chunks)  # arrondi supérieur
    return [items[i:i + size] for i in range(0, n, size)]


@app.local_entrypoint()
def main(stage: str = "meta", seq: str = SEQ, n_pilot: int = 30, batch_size: int = 4, n_chunks: int = 6,
         revision: str | None = None, commit_every: int = 200, n_load_workers: int = 8):
    import json
    import pathlib
    import time

    pathlib.Path("logs").mkdir(exist_ok=True)

    if stage == "meta":
        meta = get_class_meta.remote(revision)
        pathlib.Path("logs/stage2/classes.json").write_text(
            json.dumps({k: v for k, v in meta.items() if k != "bit_detail"} | {"bit_detail": meta["bit_detail"]},
                       indent=2, ensure_ascii=False))
        print(json.dumps({"model_id": meta["model_id"], "revision_sha": meta["revision_sha"],
                          "n_labels": meta["n_labels"], "bits": meta["bits"]}, indent=2, ensure_ascii=False))
        print("classes -> bits (non nuls) :")
        for idx, d in meta["bit_detail"].items():
            if d["matched_bits"]:
                print(f"  {idx:>3} {d['label']:<40} {d['matched_bits']}")
        print("Détail complet : logs/stage2/classes.json")
        return

    if stage == "pilot":
        meta = get_class_meta.remote(revision)
        sha = meta["revision_sha"]
        print(f"modèle {meta['model_id']} révision figée : {sha}")
        t0 = time.time()
        res = pilot_run.remote(n=n_pilot, batch_size=batch_size, seq=seq, revision=sha, class_table=meta["bit_table"],
                              n_load_workers=n_load_workers)
        res["wall_s_including_cold_start"] = round(time.time() - t0, 1)
        pathlib.Path("logs/stage2/pilot.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    if stage == "full":
        meta = get_class_meta.remote(revision)
        sha = meta["revision_sha"]
        print(f"modèle {meta['model_id']} révision figée : {sha}")
        cpath = write_classes_json.remote(meta, seq)
        print(f"classes.json écrit : {cpath}")
        items = list_items.remote(seq)
        print(f"{len(items)} images à traiter (3 caméras × triplets de frames.csv)")
        chunks = _chunkify(items, n_chunks)
        t0 = time.time()
        results = list(process_chunk.map(
            chunks,
            kwargs={"batch_size": batch_size, "seq": seq, "revision": sha, "class_table": meta["bit_table"],
                    "commit_every": commit_every, "n_load_workers": n_load_workers},
            return_exceptions=True,
        ))
        wall_s = time.time() - t0
        good = [r for r in results if isinstance(r, dict)]
        bad = [repr(r)[:500] for r in results if not isinstance(r, dict)]
        n_classes = len(meta["bit_table"])
        import numpy as np

        hist = np.zeros((len(CAMS), n_classes), dtype=np.int64)
        for r in good:
            hist += np.asarray(r["hist"], dtype=np.int64)
        errors = [e for r in good for e in r["errors"]]
        gpu_seconds = sum(r["runtime_s"] for r in good)  # somme sur les conteneurs = GPU-secondes consommées
        report = {
            "stage": "full", "seq": seq, "model_id": meta["model_id"], "revision_sha": sha,
            "n_chunks": len(chunks), "n_chunk_failures": len(bad), "chunk_failures": bad,
            "n_items": len(items), "n_done": sum(r["n_done"] for r in good),
            "n_skip": sum(r["n_skip"] for r in good), "n_err": sum(r["n_err"] for r in good),
            "n_errors_total": sum(r["n_errors_total"] for r in good), "errors_sample": errors[:20],
            "wall_s": round(wall_s, 1), "gpu_seconds_sum": round(gpu_seconds, 1),
            "gpu_hours_sum": round(gpu_seconds / 3600, 3),
            "cost_usd_at_0.80_per_hour": round(gpu_seconds / 3600 * 0.80, 2),
            "batch_size": batch_size, "proc_size": {"height": PROC_H, "width": PROC_W},
            "class_hist_by_cam": {cam: hist[i].tolist() for i, cam in enumerate(CAMS)},
        }
        write_run_report.remote(report, seq)
        pathlib.Path("logs/stage2/segment_run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps({k: v for k, v in report.items() if k != "class_hist_by_cam"}, indent=2, ensure_ascii=False))
        return

    if stage == "sizes":
        sizes = output_sizes.remote(seq)
        print(json.dumps(sizes, indent=2, ensure_ascii=False))
        return

    raise SystemExit(f"stage inconnu : {stage}")


@app.function(image=image, volumes={DATA_MOUNT: vol}, timeout=300)
def write_run_report(report: dict, seq: str = SEQ) -> str:
    sc = scene(seq)
    path = sc.write_report("stage2_segment_run", report)
    vol.commit()
    return path
