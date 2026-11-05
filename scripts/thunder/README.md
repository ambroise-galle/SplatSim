# Entraîner sur Thunder Compute (`tnr`)

Premier run le 25/09 : L40 48 Go, 8 vCPU, 64 Go de RAM, image `base` (≈ 0,87 $/h). L'A6000 (0,35 $/h) était
indisponible. **Fenêtre de 3 000 pas : 4,5 min, contre 9–11 min sur la desktop (RTX 2070 Super).**

1. `tnr create --gpu l40 --num-gpus 1 --vcpus 8 --template base --disk 100 --yes --json`, puis attendre
   `"status": "RUNNING"` (`tnr status --json`). Une fois `tnr connect 0` lancé, l'alias SSH `tnr-0` existe dans
   `~/.ssh/config` (la commande ouvre un shell interactif : la couper une fois l'alias créé).
2. `scp scripts/thunder/{setup_env,get_images}.sh tnr-0:~/`, puis lancer les deux en arrière-plan sur l'instance
   (`setsid nohup ~/setup_env.sh …`) :
   - `get_images.sh` : 7 Go d'images depuis Hugging Face en ~1 min, puis
     décompression dans `~/dino/data_local/scene/rawimgs/` ;
   - `setup_env.sh` : CUDA 12.9 **à côté** du CUDA 13 de l'image (ne pas toucher au 13), venv, torch 2.9.1+cu129,
     gsplat@28e794c pour sm_89.
3. `scripts/thunder/push_data.sh` depuis le Mac : données dérivées + code.
4. Lancer : `cd ~/dino && setsid nohup ~/venv/bin/python -m splatpipe.train.full --root data_local/scene
   --out-dir out/train_full_v3 > train_full_v3.log 2>&1 < /dev/null &`.
   Suivi : `ssh tnr-0 'cd ~/dino && ~/venv/bin/python -m splatpipe.train.progress train_full_v3.log'`.
5. Rapatrier `out/train_full_v3/`, puis **supprimer l'instance** (`tnr delete 0`) : elle est facturée tant
   qu'elle existe.

Pièges rencontrés le 25/09 :
- sans `~/venv/bin` dans le PATH, pas de `ninja` : compilation séquentielle (~45 min au lieu de ~20) et échec
  des modules JIT de gsplat. `setup_env.sh` les met maintenant dans le PATH ; un échec JIT arrête le script.
  Le trainer 4.1 n'utilise pas ces modules.
- l'image `base` fournit torch 2.12+cu130 et CUDA 13 : gsplat@28e794c n'y est pas validé → venv dédié.
- les heures des journaux de l'instance sont en UTC.
