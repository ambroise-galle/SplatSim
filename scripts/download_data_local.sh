#!/bin/zsh
# Télécharge (~1,1 Go, une fois) les données nécessaires au run local du test intra-triplet.
# Nécessite un jeton Modal valide : .venv/bin/modal token new
set -e
SEQ=2024-03-13-observatory-quarter-01
V=splatsim-data
R="scenes/oxford_spires/$SEQ"
D=data_local/scene
mkdir -p $D/map $D/poses $D/cameras/cam0 $D/cameras/cam1 $D/cameras/cam2 $D/reports/stage2_material $D/semantics $D/masks
.venv/bin/modal volume get --force $V $R/sensors.yaml $D/sensors.yaml
.venv/bin/modal volume get --force $V $R/poses/cameras.json $D/poses/cameras.json
.venv/bin/modal volume get --force $V $R/semantics/classes.json $D/semantics/classes.json
.venv/bin/modal volume get --force $V $R/cameras/frames.csv $D/cameras/frames.csv
for c in cam0 cam1 cam2; do
  .venv/bin/modal volume get --force $V $R/cameras/$c/index.csv $D/cameras/$c/index.csv
done
.venv/bin/modal volume get --force $V $R/map/map_5cm_static.ply $D/map/map_5cm_static.ply
.venv/bin/modal volume get --force $V $R/reports/stage2_material/superpoints.npz $D/reports/stage2_material/superpoints.npz
.venv/bin/modal volume get --force $V $R/semantics $D/semantics
.venv/bin/modal volume get --force $V $R/masks $D/masks
echo "téléchargement terminé"
