"""Images Modal partagées par les étapes GPU de SplatPipeline.

`gsplat_image()` : environnement de l'étape 4 (entraînement gsplat), gsplat compilé depuis les sources.

Choix (détails et mesures dans etapes/notes/stage4-gsplat-env.md) :
- gsplat épinglé sur le commit 28e794c (main, 03/09/2026). La version PyPI (1.5.3) ne contient pas
  le rendu LiDAR natif (`camera_model="lidar"`, PR #890, mars 2026) : compilation depuis les sources.
- Base `nvidia/cuda:12.9.1-devel-ubuntu22.04` + torch 2.9.1+cu129 : la combinaison de la CI GPU de gsplat
  (CUDA 12.9, torch 2.9.1 cu129, build AOT avec les mêmes BUILD_CAMERA_WRAPPERS/NUM_CHANNELS).
  torch 2.8.0 ne compile PAS ce commit : `csrc/Utils.cpp` (C++ hôte) appelle la surcharge C++
  `cudaEventCreate(&ev, cudaEventDisableTiming)` déclarée dans `cuda_runtime.h`, que les en-têtes c10 de
  torch 2.8 n'incluent pas (erreur « too many arguments », en CUDA 12.8 comme 12.9). Pilote Modal 580 : OK.
  Python 3.12 ajouté par Modal (`add_python`).
- Compilation AOT pendant la construction de l'image (`pip install -e .` = setup.py), pas de JIT au
  démarrage des conteneurs. Pas de GPU au build : `TORCH_CUDA_ARCH_LIST="8.0;8.9"` fixe les
  architectures ciblées (A100 ; L4 / L40S). Une autre carte (H100 = 9.0) exigerait de recompiler.
- `BUILD_CAMERA_WRAPPERS=1` : nécessaire aux tests pytest de gsplat (test_cameras, test_rasterization).
- Les extensions « JIT uniquement » de gsplat (sensors, geometry, scene) sont aussi précompilées dans
  `TORCH_EXTENSIONS_DIR=/opt/torch_extensions` pour éviter une compilation au premier appel.
- `CC/CXX/LDSHARED/LDCXXSHARED` forcés sur gcc/g++ : sinon l'édition de liens appelle `clang++` (absent).
- setuptools épinglé (80.9.0, < 82) : sans risque avec torch 2.9 et compatible torch < 2.9 (`pkg_resources`).
- `NUM_CHANNELS` réduit (12 valeurs au lieu de 23) : chaque noyau 3DGUT est instancié pour chaque nombre
  de canaux × taille de tuile × options ; avec la liste par défaut, les noyaux avant 3DGUT « from world »
  n'avaient pas fini après 60 min (contre ~8 min avec cette liste). Liste = CI GPU de gsplat (pytest) + 2 et 5
  (profondeur ou RGB+profondeur avec un signal supplémentaire, ex. intensité LiDAR). Un autre nombre de
  canaux lève « Unsupported number of color channels » : l'ajouter ici et reconstruire.
- `NVCC_FLAGS="--threads 2"` : nvcc génère le code des deux architectures en parallèle.
- Étape de compilation en `run_function` pour réserver CPU et mémoire (nvcc sur les noyaux 3DGUT est
  gourmand) ; `MAX_JOBS` limite le parallélisme. Lancer le premier build avec `modal run --detach` :
  la compilation dure longtemps et un client local déconnecté arrête l'app (et le build).
- Toutes les versions Python sont épinglées ; seuls les paquets apt (git, build-essential) suivent
  l'image de base Ubuntu 22.04.
"""

import modal

CUDA_BASE_IMAGE = "nvidia/cuda:12.9.1-devel-ubuntu22.04"
PYTHON_VERSION = "3.12"
TORCH_VERSION = "2.9.1"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu129"

GSPLAT_REPO = "https://github.com/nerfstudio-project/gsplat.git"
GSPLAT_COMMIT = "28e794ca44a4c25ffc39175370c5ee7b38bfcc36"
GSPLAT_DIR = "/opt/gsplat"
GSPLAT_BUILD_INFO = "/opt/gsplat_build_info.json"
TORCH_EXTENSIONS_DIR = "/opt/torch_extensions"

CUDA_ARCH_LIST = "8.0;8.9"
NUM_CHANNELS = "1,2,3,4,5,6,8,21,23,24,32,128"
BUILD_CPU = 32
BUILD_MEMORY_MIB = 128 * 1024
BUILD_MAX_JOBS = "16"

# Dépendances Python épinglées : build (setuptools, ninja, wheel), runtime gsplat (numpy, jaxtyping,
# rich, nvtx, typing_extensions), extra « lidar » (scipy), tests pytest de gsplat.
PY_DEPS = [
    "setuptools==80.9.0",
    "wheel==0.45.1",
    "ninja==1.13.0",
    "packaging==25.0",
    "numpy==2.3.3",
    "scipy==1.16.2",
    "jaxtyping==0.3.2",
    "rich==14.1.0",
    "nvtx==0.2.13",
    "typing_extensions==4.15.0",
    "pyyaml==6.0.2",
    "pillow==11.3.0",
    "tqdm==4.67.1",
    "imageio==2.37.2",
    "pytest==8.4.2",
    "pytest-check==2.5.3",
    "pytest-env==1.1.5",
    "typeguard==4.4.4",
]


def _compile_gsplat():
    """Étape de build : compile gsplat (AOT) et précompile les extensions JIT-only."""
    import glob
    import json
    import os
    import subprocess
    import time

    info = {
        "nproc": os.cpu_count(),
        "max_jobs": os.environ.get("MAX_JOBS"),
        "num_channels": os.environ.get("NUM_CHANNELS"),
        "nvcc_flags": os.environ.get("NVCC_FLAGS"),
        "arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    }
    info["nvcc"] = subprocess.run(
        ["/usr/local/cuda/bin/nvcc", "--version"], capture_output=True, text=True
    ).stdout.strip().splitlines()[-2:]
    info["cxx"] = subprocess.run(["c++", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    info["env_CXX"] = os.environ.get("CXX")

    t0 = time.time()
    subprocess.run(
        ["python", "-m", "pip", "install", "--no-build-isolation", "--no-deps", "-v", "-e", "."],
        cwd=GSPLAT_DIR,
        check=True,
    )
    info["setup_py_build_s"] = round(time.time() - t0, 1)
    info["inplace_extensions"] = sorted(
        os.path.relpath(p, GSPLAT_DIR) for p in glob.glob(f"{GSPLAT_DIR}/gsplat/**/*.so", recursive=True)
    )
    if not any(p.endswith("gsplat/csrc.so") or "/csrc." in p for p in info["inplace_extensions"]):
        # Repli : certaines versions de setuptools ne placent pas l'extension dans l'arbre source.
        subprocess.run(["python", "setup.py", "build_ext", "--inplace"], cwd=GSPLAT_DIR, check=True)
        info["inplace_extensions"] = sorted(
            os.path.relpath(p, GSPLAT_DIR) for p in glob.glob(f"{GSPLAT_DIR}/gsplat/**/*.so", recursive=True)
        )

    # Extensions chargées par JIT uniquement (setup.py ne les construit pas).
    jit_builds = {
        "sensors": "from gsplat.sensors.kernels._backend import _get_backend; _get_backend()",
        "geometry": "from gsplat.geometry.kernels._backend import _get_backend; _get_backend()",
        "scene": "from gsplat.scene.kernels._backend import _get_backend; _get_backend()",
    }
    info["jit_prebuild"] = {}
    for name, code in jit_builds.items():
        t1 = time.time()
        r = subprocess.run(["python", "-c", code], cwd="/", capture_output=True, text=True)
        info["jit_prebuild"][name] = {
            "ok": r.returncode == 0,
            "seconds": round(time.time() - t1, 1),
            "stderr_tail": r.stderr[-1500:] if r.returncode else "",
        }
        print(f"[jit prebuild] {name}: rc={r.returncode} ({time.time() - t1:.0f}s)", flush=True)
        if r.returncode:
            print(r.stdout[-3000:], r.stderr[-3000:], flush=True)

    info["total_build_s"] = round(time.time() - t0, 1)
    info["commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=GSPLAT_DIR, capture_output=True, text=True
    ).stdout.strip()
    with open(GSPLAT_BUILD_INFO, "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2), flush=True)


def gsplat_image() -> modal.Image:
    """Image CUDA 12.9 + torch 2.9.1+cu129 + gsplat@28e794c compilé (A100/L4/L40S)."""
    return (
        modal.Image.from_registry(CUDA_BASE_IMAGE, add_python=PYTHON_VERSION)
        .apt_install("git", "build-essential")
        .uv_pip_install(f"torch=={TORCH_VERSION}", index_url=TORCH_INDEX_URL)
        .uv_pip_install(*PY_DEPS)
        .run_commands(
            f"git clone {GSPLAT_REPO} {GSPLAT_DIR}",
            f"cd {GSPLAT_DIR} && git checkout {GSPLAT_COMMIT} && git submodule update --init --recursive",
        )
        .env(
            {
                "CUDA_HOME": "/usr/local/cuda",
                # Le Python ajouté par Modal (python-build-standalone) a CXX=clang++ dans sysconfig :
                # forcer gcc/g++ (présents via build-essential) pour la compilation et l'édition de liens.
                "CC": "gcc",
                "CXX": "g++",
                "LDSHARED": "gcc -shared",
                "LDCXXSHARED": "g++ -shared",
                "TORCH_CUDA_ARCH_LIST": CUDA_ARCH_LIST,
                "TORCH_EXTENSIONS_DIR": TORCH_EXTENSIONS_DIR,
                "BUILD_CAMERA_WRAPPERS": "1",
                "NUM_CHANNELS": NUM_CHANNELS,
                "NVCC_FLAGS": "--threads 2",
                "MAX_JOBS": BUILD_MAX_JOBS,
            }
        )
        .run_function(_compile_gsplat, cpu=BUILD_CPU, memory=BUILD_MEMORY_MIB, timeout=4 * 3600)
    )
