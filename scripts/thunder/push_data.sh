#!/bin/bash
# Depuis le Mac : données dérivées desktop → instance Thunder (flux tar via le Mac, ~2,1 Go, ~5 min).
# Les images brutes viennent de Hugging Face sur l'instance (get_images.sh), pas d'ici.
# Usage : scripts/thunder/push_data.sh [hôte ssh de l'instance, défaut tnr-0]
set -e
H=${1:-tnr-0}
ssh desktop 'cd ~/dino/data_local && tar -cf - scene/map scene/lidar/hesai_deskewed scene/semantics_lidar \
    scene/masks scene/poses scene/cameras scene/lidar_model scene/traj scene/sensors.yaml scene/splits.yaml \
    scene/reports/stage4/map_colors_lift.npz scene/reports/stage2_material/superpoints.npz \
    scene/reports/stage2_material/m4/labels_m2_consensus.npz' \
  | ssh "$H" 'mkdir -p ~/dino/data_local && tar -x -C ~/dino/data_local'
git -C "$(dirname "$0")/../.." archive HEAD splatpipe | ssh "$H" 'mkdir -p ~/dino && tar -x -C ~/dino'
ssh "$H" 'du -sh ~/dino/data_local/scene; ls ~/dino/data_local/scene/lidar/hesai_deskewed | wc -l'
