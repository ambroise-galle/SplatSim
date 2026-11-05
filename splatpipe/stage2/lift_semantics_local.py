"""M2 en local (desktop), sans Modal : exécute `worker` et `aggregate` de `lift_semantics.py` tels quels.

`lift_semantics.py` est une app Modal ; ce lanceur remplace le paquet `modal` par une coquille vide
(décorateurs = identité, `vol.commit()` = rien) et fait pointer `DATA_MOUNT` sur une arborescence de liens
`<racine>/scenes/oxford_spires/<seq>` → `--scene`. Le calcul est donc exactement celui de Modal.

Depuis ~/dino (desktop) — une tranche d'images par processus, reprise automatique sur checkpoint :
  for p in 0 1 2 3; do
    nohup venv/bin/python splatpipe/stage2/lift_semantics_local.py worker --part $p --n-parts 4 > m2_part$p.log 2>&1 &
  done
  # quand les 4 parts sont finies :
  venv/bin/python splatpipe/stage2/lift_semantics_local.py aggregate --n-parts 4 > m2_aggregate.log 2>&1
Sorties : <scene>/reports/stage2_material/<tag>/ (par défaut tag=m2 : ÉCRASE l'ancien M2, invalide).
Test rapide : `worker --part 0 --n-parts 1 --tag m2_test --limit-images 6`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")


class _Any:
    """Objet factice : tout attribut ou appel renvoie un _Any ; appliqué à une fonction, il la renvoie."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        if len(a) == 1 and not k and isinstance(a[0], types.FunctionType):
            return a[0]
        return _Any()

    def __getattr__(self, name):
        return _Any()


def _install_modal_stub():
    stub = types.ModuleType("modal")
    stub.__getattr__ = lambda name: _Any()
    sys.modules["modal"] = stub


def _link_scene(scene: str, seq: str) -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(scene)), ".m2_data_root")
    d = os.path.join(base, "scenes", "oxford_spires")
    os.makedirs(d, exist_ok=True)
    link = os.path.join(d, seq)
    if not os.path.islink(link):
        os.symlink(os.path.abspath(scene), link)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["worker", "aggregate"])
    ap.add_argument("--scene", default="/home/ambroise/dino/data_local/scene")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--n-parts", type=int, default=4)
    ap.add_argument("--tag", default="m2")
    ap.add_argument("--limit-images", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    _install_modal_stub()
    import splatpipe.stage2.lift_semantics as ls

    ls.DATA_MOUNT = _link_scene(args.scene, ls.SEQ)
    if args.what == "worker":
        out = ls.worker(args.part, args.n_parts, tag=args.tag, limit_images=args.limit_images)
    else:
        out = ls.aggregate(args.n_parts, tag=args.tag, make_figures=not args.no_figures)
        out = out.get("report", out)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
