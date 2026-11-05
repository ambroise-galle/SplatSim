"""Étape 4.0c — enveloppe Modal du surapprentissage (cœur : `overfit_core.py`).

Lancement :  .venv/bin/modal run splatpipe/stage4/overfit_tour_image.py
Sorties :    logs/stage4/overfit_tour_image.json + figure.
"""

from pathlib import Path

import modal

from splatpipe.images import gsplat_image

app = modal.App("splatsim-stage4-overfit")
vol = modal.Volume.from_name("splatsim-data", create_if_missing=True, version=2)
DATA_MOUNT = "/data"
SEQ = "2024-03-13-observatory-quarter-01"

_PKG = Path(__file__).resolve().parents[1]
image = (
    gsplat_image()
    .uv_pip_install("matplotlib==3.10.6")
    .add_local_file(_PKG / "__init__.py", "/root/splatpipe/__init__.py")
    .add_local_file(_PKG / "images.py", "/root/splatpipe/images.py")
    .add_local_dir(_PKG / "common", "/root/splatpipe/common", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(_PKG / "stage4", "/root/splatpipe/stage4", ignore=["**/__pycache__", "**/*.pyc"])
)


@app.function(image=image, volumes={DATA_MOUNT: vol}, gpu="L4", timeout=50 * 60)
def run_main(seq: str = SEQ, n_iters: int = 400, r_opt: float = 35.0,
             render_scale: float = 1.0) -> dict:
    from splatpipe.stage4.overfit_core import run_overfit

    return run_overfit(f"{DATA_MOUNT}/scenes/oxford_spires/{seq}", dev="cuda",
                       n_iters=n_iters, r_opt=r_opt, render_scale=render_scale)


@app.local_entrypoint()
def main(seq: str = SEQ, n_iters: int = 400, r_opt: float = 35.0, render_scale: float = 1.0):
    import json
    import os

    out = run_main.remote(seq, n_iters, r_opt, render_scale)
    os.makedirs("logs/stage4", exist_ok=True)
    fig = out.pop("_figure_rgb", None)
    with open("logs/stage4/overfit_tour_image.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    if fig:
        open("logs/stage4/overfit_tour_image_rgb.png", "wb").write(fig)
    print(json.dumps({k: v for k, v in out.items() if k != "hist"}, indent=1, ensure_ascii=False))
