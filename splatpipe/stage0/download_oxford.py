"""Étape 0.1 — Téléchargement d'une séquence Oxford Spires dans le Volume Modal.

Lancement :
  .venv/bin/modal run splatpipe/stage0/download_oxford.py --seq 2024-03-13-observatory-quarter-01 --site observatory-quarter
"""

import modal

from splatpipe.modal_common import DATA_MOUNT, data_volume

app = modal.App("splatsim-stage0-download")
vol = data_volume()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unzip")
    .uv_pip_install("huggingface_hub[hf_xet]>=0.34")
    .add_local_python_source("splatpipe")
)

REPO = "ori-drs/oxford_spires_dataset"
ROOT = f"{DATA_MOUNT}/raw/oxford_spires"


@app.function(image=image, volumes={DATA_MOUNT: vol}, cpu=8, memory=8192, timeout=4 * 3600)
def download(seq: str, site: str) -> dict:
    import os
    import subprocess
    import time

    from huggingface_hub import snapshot_download

    patterns = [
        f"sequences/{seq}/raw/images.zip",
        f"sequences/{seq}/raw/imu.csv",
        f"sequences/{seq}/raw/lidar-clouds.zip",
        f"sequences/{seq}/processed/trajectory/*",
        f"sequences/{seq}/processed/vilens-slam/*",
        "calibration/*.yaml",
        "calibration/README.md",
        "calibration/calibration_target.txt",
        f"ground_truth_map/{site}/*5cm*.pcd",
        f"ground_truth_map/{site}/FinalizeReport.pdf",
    ]
    t0 = time.time()
    snapshot_download(REPO, repo_type="dataset", allow_patterns=patterns, local_dir=ROOT, max_workers=8)
    t_dl = time.time() - t0
    vol.commit()

    # Décompression à côté des archives (on garde les zips pour pouvoir recommencer).
    seq_dir = f"{ROOT}/sequences/{seq}"
    unzipped = {}
    for rel in ["raw/images.zip", "raw/lidar-clouds.zip", "processed/vilens-slam/undist-clouds.zip"]:
        z = f"{seq_dir}/{rel}"
        if not os.path.exists(z):
            continue
        out = os.path.dirname(z)
        listing = subprocess.run(["unzip", "-Z1", z], capture_output=True, text=True).stdout.splitlines()
        subprocess.run(["unzip", "-q", "-n", z, "-d", out], check=True)
        unzipped[rel] = {"entries": len(listing), "first": listing[:3]}
        vol.commit()

    sizes = {}
    for dirpath, _, files in os.walk(ROOT):
        for f in files:
            if f.endswith((".zip", ".pcd", ".csv", ".txt", ".yaml")):
                p = os.path.join(dirpath, f)
                if f.endswith(".zip") or "ground_truth" in p or dirpath.endswith(("trajectory", "raw", "vilens-slam", "calibration")):
                    sizes[p.replace(ROOT + "/", "")] = os.path.getsize(p)
    return {"download_s": round(t_dl, 1), "unzipped": unzipped, "files": sizes}


@app.local_entrypoint()
def main(seq: str = "2024-03-13-observatory-quarter-01", site: str = "observatory-quarter"):
    import json

    print(json.dumps(download.remote(seq, site), indent=2))
