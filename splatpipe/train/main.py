"""CLI 4.1 — entraînement d'une fenêtre sur Oxford Spires (desktop, GPU local).

Exemple (desktop) :
  cd ~/dino && venv/bin/python -m splatpipe.train.main --root data_local/scene \
      --stamps-file data_local/scene/train_v1_stamps.txt --steps 1500 \
      --weights b0=1,b1=1,b2=0.05,b3=0.05,b4=1.0 \
      --out-dir out/train_v1
"""

from __future__ import annotations

import argparse
import json
import os


def parse_weights(s: str) -> dict[str, float]:
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, w = part.partition("=")
        out[name.strip().lower()] = float(w) if w else 1.0
    return out


def main() -> None:
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--root", required=True)
    pa.add_argument("--stamps-file", required=True, help="un stamp ns par ligne (balayages train)")
    pa.add_argument("--steps", type=int, default=1500)
    pa.add_argument("--r-opt", type=float, default=35.0)
    pa.add_argument("--holdout-every", type=int, default=10)
    pa.add_argument("--weights", default="b0=1,b1=1,b2=0.05,b3=0.05,b4=1.0",
                    help="pertes activées : nom=poids, séparées par des virgules")
    pa.add_argument("--eval-every", type=int, default=250)
    pa.add_argument("--out-dir", default="out/train_v1")
    pa.add_argument("--seed", type=int, default=0)
    a = pa.parse_args()

    stamps = [int(x) for x in open(a.stamps_file) if x.strip()]
    weights = parse_weights(a.weights)
    os.makedirs(a.out_dir, exist_ok=True)

    from splatpipe.train.trainer import WindowTrainer

    tr = WindowTrainer(a.root, stamps, r_opt=a.r_opt, holdout_every=a.holdout_every, seed=a.seed)
    res = tr.train(a.steps, weights, eval_every=a.eval_every, ckpt_dir=a.out_dir)
    res["cfg"] = {"root": a.root, "n_stamps": len(stamps), "steps": a.steps,
                  "r_opt": a.r_opt, "weights": weights, "holdout_every": a.holdout_every,
                  "n_opt": tr.n_opt}
    with open(f"{a.out_dir}/result.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
