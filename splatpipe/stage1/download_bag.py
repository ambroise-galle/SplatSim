"""Étape 1.1 — Téléchargement du bag ROS 2 d'une séquence Oxford Spires dans le Volume Modal.

Seul ajout autorisé sous `/data/raw` : `sequences/<seq>/raw/ros2bag/<bag>/` (arborescence Hugging Face).

Lancement :
  .venv/bin/modal run splatpipe/stage1/download_bag.py --seq 2024-03-13-observatory-quarter-01
"""

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume

app = modal.App("splatsim-stage1-download-bag")
vol = data_volume()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub[hf_xet]>=0.34")
    .add_local_python_source("splatpipe")
)

REPO = "ori-drs/oxford_spires_dataset"
ROOT = f"{DATA_MOUNT}/raw/oxford_spires"


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=8, memory=8192, timeout=3 * 3600)
def download(seq: str) -> dict:
    import os
    import time

    from huggingface_hub import snapshot_download

    pattern = f"sequences/{seq}/raw/ros2bag/*"
    t0 = time.time()
    snapshot_download(REPO, repo_type="dataset", allow_patterns=[pattern], local_dir=ROOT, max_workers=8)
    dt = time.time() - t0
    # Le cache de métadonnées de huggingface_hub (.cache/huggingface) reste à la racine locale : sans effet.
    vol.commit()
    files = {}
    base = f"{ROOT}/sequences/{seq}/raw/ros2bag"
    for dirpath, _, names in os.walk(base):
        for n in names:
            p = os.path.join(dirpath, n)
            files[p] = os.path.getsize(p)
    return {"download_s": round(dt, 1), "files": files}


@app.local_entrypoint()
def main(seq: str = "2024-03-13-observatory-quarter-01"):
    import json

    print(json.dumps(download.remote(seq), indent=2))
