"""Test GPU rapide : driver NVIDIA, CUDA, PyTorch, calcul sur GPU.

Lancement : .venv/bin/modal run splatpipe/smoke_gpu.py
"""

import modal

app = modal.App("splatsim-smoke-gpu")

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128"
)


@app.function(gpu="L4", image=image, timeout=600)
def check_gpu() -> dict:
    import subprocess
    import time

    import torch

    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert torch.cuda.is_available(), "CUDA indisponible dans PyTorch"
    a = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        b = a @ a
    torch.cuda.synchronize()
    dt = time.time() - t0

    return {
        "nvidia_smi": smi,
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "matmul_4096_x10_s": round(dt, 3),
        "result_finite": bool(torch.isfinite(b).all().item()),
    }


@app.local_entrypoint()
def main():
    import json

    print(json.dumps(check_gpu.remote(), indent=2))
