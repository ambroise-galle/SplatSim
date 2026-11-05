"""Avancement d'un entraînement `full.py` d'après son journal : fenêtres faites / total, fin estimée,
métriques de chaque fenêtre (LiDAR holdout, PSNR holdout) et RAM (journal du garde-fou).

Desktop, depuis ~/dino :   venv/bin/python -m splatpipe.train.progress train_full_v3.log
Depuis le Mac :            ssh desktop 'cd ~/dino && venv/bin/python -m splatpipe.train.progress train_full_v3.log'
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else "train_full_v3.log"
    lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
    total = None
    todo = None
    rows = []
    for ln in lines:
        m = re.match(r"scène : .* (\d+) fenêtres", ln)
        if m:
            total = int(m.group(1))
        m = re.match(r"fenêtres entraînées : \[(.*)\]", ln)
        if m:
            todo = len([x for x in m.group(1).split(",") if x.strip()])
        m = re.match(r"fenêtre\s+(\d+)/(\d+) : train (.*)", ln)
        if not m:
            continue
        parts = ("train " + m.group(3)).split(" | ")
        try:
            hold = json.loads(parts[1][len("holdout "):])
            rgb = json.loads(parts[2][len("rgb "):])
            stats = json.loads(parts[3])
            wall = float(parts[4].rstrip("s"))
        except (IndexError, ValueError):
            continue
        ps = [v["psnr_aligne"] for v in rgb.values() if v]
        rows.append(dict(w=int(m.group(1)), wall=wall, app=hold.get("matched_frac") if hold else None,
                         med=hold.get("median_abs_cm") if hold else None,
                         psnr=round(float(np.mean(ps)), 1) if ps else None,
                         opac=stats.get("opac_lt_0.05"), ram=stats.get("ram_max_go")))
    n_todo = todo or total or 0
    done = len(rows)
    started = os.path.getmtime(log) if not lines else os.stat(log).st_mtime
    print(f"journal : {log}")
    print(f"fenêtres terminées : {done}/{n_todo}")
    if rows:
        mean_w = float(np.mean([r["wall"] for r in rows]))
        left = max(n_todo - done, 0) * mean_w
        eta = time.strftime("%H:%M", time.localtime(time.time() + left)) if left else "—"
        print(f"durée moyenne par fenêtre : {mean_w / 60:.1f} min ; reste ≈ {left / 3600:.1f} h ; fin estimée ≈ {eta}")
        print("fenêtre | durée | LiDAR holdout appariés / médiane | PSNR holdout aligné | opacité<0,05 | RAM max")
        for r in rows:
            print(f"  {r['w']:2d}    | {r['wall'] / 60:4.1f} min | {r['app']} / {r['med']} cm | {r['psnr']} dB "
                  f"| {r['opac']} | {r['ram']} Go")
    if any(("Traceback" in ln) or ("out of memory" in ln) for ln in lines):
        print("⚠️ erreur dans le journal (Traceback / out of memory)")
    if any("peak_mem_mb" in ln for ln in lines):
        print("✅ run terminé")
    guard = os.path.join(os.path.dirname(os.path.abspath(log)), "ram_guard.log")
    if os.path.exists(guard):
        g = open(guard).read().splitlines()
        if g:
            print(f"garde-fou RAM : {g[-1]}" + (" — ⚠️ ARRÊT déclenché" if any("ARRET" in x for x in g) else ""))


if __name__ == "__main__":
    main()
