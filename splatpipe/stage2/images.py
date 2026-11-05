"""Image Modal pour la sous-étape 2.1 (segmentation sémantique) — séparée de `splatpipe/images.py` (gsplat).

Choix :
- `facebook/mask2former-swin-large-ade-semantic` (Mask2Former, backbone Swin-Large, tête sémantique ADE20K
  150 classes). Code du dépôt `facebookresearch/Mask2Former` sous licence MIT ; la fiche Hugging Face du
  point de contrôle affiche `license: other` sans texte de licence dédié plus restrictif trouvé (vérifié le
  17/09/2026). Usage recherche uniquement ici, comme le reste du projet (cf. `planA-splatpipeline.md` §5).
- Pas d'image CUDA "devel" : Mask2Former (transformers) utilise un noyau CUDA optionnel pour l'attention
  déformable multi-échelle du décodeur de pixels ; sans `nvcc`/`ninja` il retombe automatiquement sur une
  implémentation pure PyTorch (avertissement, pas d'erreur). Base `debian_slim` + roue PyTorch cu128, comme
  `splatpipe/smoke_gpu.py` (déjà testé sur L4).
- `transformers==4.46.3` : version connue pour exister sur PyPI (nov. 2024), avec le support Mask2Former et
  `post_process_semantic_segmentation` (upsampling des logits à la résolution d'origine + argmax).
"""

import modal

TORCH_VERSION = "2.8.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
TRANSFORMERS_VERSION = "4.46.3"

PY_DEPS = [
    "transformers==4.46.3",
    "pillow==11.3.0",
    "numpy==2.3.3",
    "scipy==1.16.2",  # requis par Mask2FormerLoss (instanciée même en inférence)
    "huggingface_hub==0.35.3",
    "hf_transfer==0.1.9",
    "pyyaml==6.0.2",  # sensors.yaml (splatpipe.common.scene)
    "opencv-python-headless==4.10.0.84",  # dessin des superpositions QA (splatpipe/stage2/check_masks.py)
]


def segmentation_image() -> modal.Image:
    """CUDA runtime (via la roue torch cu128) + transformers, pour l'inférence Mask2Former sur L4."""
    return (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(f"torch=={TORCH_VERSION}", index_url=TORCH_INDEX_URL)
        .uv_pip_install(*PY_DEPS)
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/opt/hf_cache"})
        .add_local_python_source("splatpipe")
    )


def dufomap_image() -> modal.Image:
    """Image CPU pour la sous-étape 2.2 (retrait des points dynamiques).

    `dufomap==1.1.1` : roues manylinux cp312 (liaisons pybind11, UFOMap embarqué), CPU uniquement. La lecture des
    tours `splatpipe.stage1.maptools` tire numpy/scipy/pyyaml. Les liaisons n'exposent pas de save/load : la carte
    est construite et interrogée dans le même conteneur (voir `dufomap_run.py`).
    """
    return (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install("dufomap==1.1.1", "numpy==2.3.3", "scipy==1.16.2", "pyyaml==6.0.2")
        .add_local_python_source("splatpipe")
    )
