#!/bin/bash
# Niveaux de détail en streaming (streamed SOG) pour une scène préparée en tuiles : fusion + tri spatial (Morton),
# décimations adaptatives 50/25/10/3/1 %, assemblage lod/lod-meta.json. Usage : build_lod.sh <dossier de scène>
# Variante depuis un PLY déjà fusionné (ex. splat recoloré par classe) : build_lod.sh <dossier> <source.ply> <sortie>
set -e
cd "$1"
export PATH=$HOME/dino/node/bin:$PATH
ST=$HOME/dino/simrobot/viewer/node_modules/.bin/splat-transform
SRC=${2:-merged.ply}
OUT=${3:-lod}
TAG=$(basename "$OUT")
if [ -z "$2" ]; then date; $ST -q -w tiles_ply/*.ply merged.ply -m; ls -la merged.ply; fi
for p in 50 25 10 3 1; do date; $ST -q -w "$SRC" --decimate-adaptive ${p}% ${TAG}_${p}.ply; ls -la ${TAG}_${p}.ply; done
date; $ST -w "$SRC" -l 0 ${TAG}_50.ply -l 1 ${TAG}_25.ply -l 2 ${TAG}_10.ply -l 3 ${TAG}_3.ply -l 4 ${TAG}_1.ply -l 5 "$OUT"/lod-meta.json --filter-nan 2>&1 | grep -v "spatially incoherent" | tail -5
date; du -sh "$OUT"; echo FIN_LOD
