"""Images Modal de l'étape 1 (GLIM sans affichage, évaluation de trajectoires).

`glim_image()` : image Docker Hub officielle `koide3/glim_ros2:jazzy_cuda12.5` (GLIM + glim_ros2 compilés
avec CUDA, ROS 2 Jazzy, Ubuntu 24.04), épinglée par condensat. Choix (détails : etapes/notes/stage1-glim.md) :
- Jazzy plutôt que Humble : le bag Oxford Spires est écrit par la bibliothèque Python `rosbags`
  (métadonnées version 8) ; rosbag2 de Jazzy lit la version 8, celui de Humble s'arrête à la version 5/6.
- CUDA 12.5 plutôt que 13.1 : le pilote des L4 de Modal couvre CUDA 12.x sans dépendre d'un pilote ≥ 590.
- GLIM est installé dans `/root/ros2_ws/install` ; l'environnement ROS se charge avec
  `source /opt/ros/jazzy/setup.bash && source /root/ros2_ws/install/setup.bash` (voir `ROS_ENV`).
- `add_python="3.12"` : Python autonome de Modal (dans /usr/local) pour exécuter les fonctions ; les outils ROS
  gardent `/usr/bin/python3` (shebangs). On y ajoute numpy, pyyaml et `rosbags` (lecture directe du bag
  pour les contrôles, sans passer par ROS).

`eval_image()` : conteneur CPU léger pour T1.1 (numpy, scipy, matplotlib, evo épinglés).

`map_image()` : conteneur CPU pour 1.4–1.6 (carte, T1.2, T1.3) : numpy/scipy/matplotlib, `pye57` (scans laser fixes
E57), `huggingface_hub`, et l'outil C++ `octomap_tools` (liboctomap de Debian, compilé avec OpenMP) qui reproduit
le filtrage « espace inconnu » du benchmark de reconstruction Oxford Spires.
"""

import os

import modal

GLIM_IMAGE_REF = (
    "koide3/glim_ros2:jazzy_cuda12.5@sha256:236c7f1be5a179ea0aefcd1d76c9ac7b79b60c5c6f27d8da07ad49f9fda759ab"
)
ROS_DISTRO = "jazzy"
ROS_ENV = f"source /opt/ros/{ROS_DISTRO}/setup.bash && source /root/ros2_ws/install/setup.bash"
GLIM_SRC_CONFIG = "/root/ros2_ws/src/glim/config"  # configuration par défaut de la version compilée

PY_DEPS_GLIM = ["numpy==2.3.3", "pyyaml==6.0.2", "rosbags==0.11.5"]
PY_DEPS_EVAL = [
    "numpy==2.3.3",
    "scipy==1.16.2",
    "matplotlib==3.10.6",
    "pyyaml==6.0.2",
    "evo==1.37.1",
]


def glim_image() -> modal.Image:
    return (
        modal.Image.from_registry(GLIM_IMAGE_REF, add_python="3.12")
        .entrypoint([])
        .uv_pip_install(*PY_DEPS_GLIM)
        .env({"ROS_DISTRO": ROS_DISTRO, "ROS_LOG_DIR": "/tmp/ros_log", "RCUTILS_COLORIZED_OUTPUT": "0"})
        .add_local_python_source("splatpipe")
    )


def eval_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(*PY_DEPS_EVAL)
        .env({"MPLBACKEND": "Agg"})
        .add_local_python_source("splatpipe")
    )


PY_DEPS_MAP = [
    "numpy==2.3.3",
    "scipy==1.16.2",
    "matplotlib==3.10.6",
    "pyyaml==6.0.2",
    "pye57==0.4.19",
    "huggingface_hub[hf_xet]>=0.34,<2",
]
OCTOMAP_TOOLS_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "octomap_tools.cpp")


def map_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("g++", "liboctomap-dev", "libgomp1")
        .uv_pip_install(*PY_DEPS_MAP)
        .add_local_file(OCTOMAP_TOOLS_SRC, "/opt/splatsim/octomap_tools.cpp", copy=True)
        .run_commands(
            "g++ -O3 -fopenmp -o /usr/local/bin/octomap_tools /opt/splatsim/octomap_tools.cpp -loctomap -loctomath",
            "dpkg -s liboctomap-dev | grep Version > /opt/splatsim/octomap_version.txt",
        )
        .env({"MPLBACKEND": "Agg"})
        .add_local_python_source("splatpipe")
    )
