#!/bin/bash
# Environnement gsplat (ninja du venv dans le PATH : compilation parallèle ; échec JIT = arrêt)
# Environnement gsplat@28e794c sur Thunder (L40, sm_89) : CUDA 12.9 à côté du 13.0 de l'image + torch 2.9.1+cu129
set -e
exec >> ~/setup_env.log 2>&1
echo "== $(date +%T) CUDA 12.9"
cd /tmp && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-12-9 libglm-dev > /dev/null
/usr/local/cuda-12.9/bin/nvcc --version | tail -1
echo "== $(date +%T) venv + torch"
cd ~ && uv venv -q --python 3.12 ~/venv
uv pip install -q --python ~/venv/bin/python "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu129
uv pip install -q --python ~/venv/bin/python setuptools==80.9.0 wheel==0.45.1 packaging ninja numpy==2.3.3 scipy==1.16.2 \
    jaxtyping==0.3.2 rich typing_extensions pyyaml pillow tqdm imageio
echo "== $(date +%T) gsplat"
git clone -q https://github.com/nerfstudio-project/gsplat.git ~/gsplat && cd ~/gsplat
git checkout -q 28e794ca44a4c25ffc39175370c5ee7b38bfcc36 && git submodule update --init --recursive -q
export CUDA_HOME=/usr/local/cuda-12.9 PATH=$HOME/venv/bin:/usr/local/cuda-12.9/bin:$PATH CC=gcc CXX=g++ \
    TORCH_CUDA_ARCH_LIST=8.9 BUILD_CAMERA_WRAPPERS=1 NUM_CHANNELS=1,2,3,4,5,6,8,21,23,24,32,128 \
    NVCC_FLAGS="--threads 2" MAX_JOBS=8 TORCH_EXTENSIONS_DIR=$HOME/torch_extensions
~/venv/bin/python -m pip --version >/dev/null 2>&1 || uv pip install -q --python ~/venv/bin/python pip
~/venv/bin/python -m pip install --no-build-isolation --no-deps -q -e .
for m in sensors geometry scene; do
  ~/venv/bin/python -c "from gsplat.$m.kernels._backend import _get_backend; _get_backend()"; echo "jit $m ok"
done
~/venv/bin/python -c "import torch, gsplat; print('OK', torch.__version__, gsplat.__version__, torch.cuda.get_device_name())"
echo "== $(date +%T) FIN"
