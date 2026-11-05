#!/bin/bash
# Images brutes Oxford Spires (séquence de travail) depuis Hugging Face → ~/dino/data_local/scene/rawimgs/{cam}/
set -e
exec >> ~/get_images.log 2>&1
echo "== $(date +%T) téléchargement"
SEQ=2024-03-13-observatory-quarter-01
uvx --from "huggingface_hub[cli]" hf download ori-drs/oxford_spires_dataset --repo-type dataset \
    --include "sequences/$SEQ/raw/images.zip" --local-dir ~/hf
ls -la ~/hf/sequences/$SEQ/raw/
unzip -l ~/hf/sequences/$SEQ/raw/images.zip | head -8
mkdir -p ~/dino/data_local/scene/rawimgs && cd ~/dino/data_local/scene/rawimgs
unzip -q -o ~/hf/sequences/$SEQ/raw/images.zip && rm -rf ~/hf
for c in cam0 cam1 cam2; do echo "$c $(ls $c | wc -l) images"; done
echo "== $(date +%T) FIN téléchargement"
