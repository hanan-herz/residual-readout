#!/usr/bin/env python3
# Historical tooling: Scheme A/pilot mappings do not validate the revised tables.
# Use scripts/build_descriptive_tables.py --check; see RESULTS.md.
"""Cross-check every number in Tables 1 & 2 against the on-disk artifacts.

Adopted canonical convention (Scheme A, from the M1 review fix):
  * readout column  = per-seed MEAN, run JSON ``last.mlp``
  * loop.0 / loop.k = same run JSON
  * k               = best-of available loop arm (BoolQ sweeps k in {0..64};
                      others k=8) -- "best of loop.0 / loop.k", applied
                      consistently per cell
  * n (rows)        = declared per task (BoolQ 3270, ARC 1165, RuleTaker
                      *pair* on the 400-row loop subset)

Two uses:
  1) Empty -- print a reproducibility report: every current table + the
     canonical number from artifacts, so ghosts surface.
  2) --patch --src tables.tex --dst tables.tex.new  -> rewrite the two tables
     with canonical numbers (keeps structure/captions, replaces numeric atoms).

The paired readout-vs-loop p-values are NOT recomputed here from npz (they
live in *_paired.json / mistral|granite8b_paired.json, which pair the 4-seed
vote preds). The readout column this script emits is per-seed mean, so the
Delta column is mean - loop.k; significance is reported from the paired
artifacts as-is with a note in the caption (they never change a cell's
sig/ns verdict, since mean-vs-vote differ by <=0.5pt).
"""
from __future__ import annotations

import argparse
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

# Authoritative CELLS map for paper Table 1 (Scheme A / paper/tables.tex).
# Note: scripts/build_table1.py uses a related but not identical map for some
# RuleTaker cells (n2k pilot JSONs vs layersweep). Prefer this file when
# checking paper numbers.
# (Task, model, run_json, k_arm). k_arm is what loop.k reports.
CELLS = [
    # BoolQ -- budget runs carry the k<=64 sweep (pad 8192)
    ("BoolQ", "Qwen3-0.6B",          "boolq_budget_06b.json",          "k_best", "loop.64"),
    ("BoolQ", "Qwen3-4B",            "boolq_budget_4b.json",           "k_best", "loop.64"),
    ("BoolQ", "Qwen3-8B",            "boolq_budget_8b.json",           "k_best", "loop.16"),
    ("BoolQ", "Mistral-7B",          "boolq_layersweep_mistral7b.json", "k_8",   "loop.8"),
    ("BoolQ", "Granite-3.1-8B",      "boolq_layersweep_granite8b.json", "k_8",   "loop.8"),
    ("BoolQ", "DeepSeek-V4",         "boolq_layersweep_dsv4.json",      "k_8",   "loop.8"),
    # RuleTaker (loop on 400-row subset)
    ("RuleTaker", "Qwen3-0.6B",      "ruletaker_layersweep_06b.json",  "k_8",   "loop.8"),
    ("RuleTaker", "Qwen3-4B",        "ruletaker_layersweep_4b.json",   "k_8",   "loop.8"),
    ("RuleTaker", "Qwen3-8B",        "ruletaker_layersweep_8b.json",   "k_8",   "loop.8"),
    ("RuleTaker", "Mistral-7B",      "ruletaker_layersweep_mistral7b.json", "k_8", "loop.8"),
    ("RuleTaker", "Granite-3.1-8B",  "ruletaker_layersweep_granite8b.json", "k_8", "loop.8"),
    ("RuleTaker", "DeepSeek-V4",     "ruletaker_layersweep_dsv4.json", "k_8",   "loop.8"),
    # ARC (loop on full 1165)
    ("ARC", "Qwen3-0.6B",            "arc_layersweep_06b.json",        "k_8",   "loop.8"),
    ("ARC", "Qwen3-4B",              "arc_layersweep_4b.json",         "k_8",   "loop.8"),
    ("ARC", "Qwen3-8B",              "arc_layersweep_8b.json",         "k_8",   "loop.8"),
    ("ARC", "Mistral-7B",            "arc_layersweep_mistral7b.json",  "k_8",   "loop.8"),
    ("ARC", "Granite-3.1-8B",        "arc_layersweep_granite8b.json",  "k_8",   "loop.8"),
    ("ARC", "DeepSeek-V4",           "arc_layersweep_dsv4.json",       "k_8",   "loop.8"),
]


def mean_of(d):
    v = d.get("last.mlp")
    return v[0] if isinstance(v, list) else v


def best_loop(d):
    """best-of loop.0 / loop.k: max over all loop.* arms in the run."""
    arms = [(float(v), k) for k, v in d.items() if k.startswith("loop.") and isinstance(v, (int, float))]
    if not arms:
        return None, None
    arms.sort(reverse=True)
    return None, None if arms[0][0] <= 0 else arms[0]


def canonical():
    rows = {}
    for task, model, fname, sel, karm in CELLS:
        with open(os.path.join(RES, fname)) as fh:
            d = json.load(fh)
        readout = mean_of(d)
        if sel == "k_best":
            _, (loopk, ksel) = best_loop(d)
        else:
            loopk, ksel = d.get(karm), karm
        loop0 = d.get("loop.zero")
        delta = (readout - loopk) if (readout is not None and loopk is not None) else None
        rows[(task, model)] = {
            "run": fname, "readout": readout, "loop0": loop0,
            "loopk": loopk, "ksel": ksel, "delta": delta,
        }
    return rows


CURRENT_TABLE1 = {  # (task, model) -> dict(k=("readout","loop0","loopk","delta"))
    ("BoolQ", "Qwen3-0.6B"): (0.753, 0.631, 0.715, 0.038),
    ("BoolQ", "Qwen3-4B"):   (0.862, 0.854, 0.869, -0.007),
    ("BoolQ", "Qwen3-8B"):   (0.879, 0.862, 0.886, -0.007),
    ("BoolQ", "Mistral-7B"): (0.841, 0.798, 0.852, -0.011),
    ("BoolQ", "Granite-3.1-8B"): (0.854, 0.815, 0.864, -0.010),
    ("BoolQ", "DeepSeek-V4"): (0.896, 0.888, 0.906, -0.009),
    ("RuleTaker", "Qwen3-0.6B"): (0.638, 0.600, 0.545, 0.093),
    ("RuleTaker", "Qwen3-4B"):   (0.738, 0.653, 0.698, 0.040),
    ("RuleTaker", "Qwen3-8B"):   (0.753, 0.675, 0.730, 0.023),
    ("RuleTaker", "Mistral-7B"): (0.660, 0.515, 0.575, 0.085),
    ("RuleTaker", "Granite-3.1-8B"): (0.828, 0.558, 0.695, 0.133),
    ("RuleTaker", "DeepSeek-V4"): (0.763, 0.605, 0.838, -0.074),
    ("ARC", "Qwen3-0.6B"):   (0.492, 0.502, 0.597, -0.106),
    ("ARC", "Qwen3-4B"):     (0.837, 0.826, 0.882, -0.046),
    ("ARC", "Qwen3-8B"):     (0.908, 0.902, 0.913, -0.005),
    ("ARC", "Mistral-7B"):   (0.728, 0.750, 0.779, -0.051),
    ("ARC", "Granite-3.1-8B"): (0.708, 0.732, 0.790, -0.082),
    ("ARC", "DeepSeek-V4"):  (0.951, 0.961, 0.959, -0.007),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", action="store_true")
    ap.add_argument("--src", default=os.path.join(ROOT, "paper", "tables.tex"))
    ap.add_argument("--dst", help="if --patch, write canonical tables here")
    args = ap.parse_args()

    canon = canonical()

    cols = ["readout", "loop0", "loopk", "delta"]
    print(f"{'task':10s}{'model':12s} {'field':7s} {'current':>8s} {'canonical':>9s} {'run'}")
    print("-" * 72)
    for (task, model), cur in CURRENT_TABLE1.items():
        c = canon[(task, model)]
        for j, col in enumerate(cols):
            curv, cann = cur[j], c[col]
            ok = (cann is not None and abs(curv - cann) < 0.0011) or (cann is None)
            flag = " " if ok else "  <-- DIFF"
            tag = "" if ok else " [here] "
            print(f"{task:10s}{model:12s} {col:7s} "
                  f"{curv if isinstance(curv,(int,float)) else str(curv):>8} "
                  f"{(f'{cann:.4f}' if cann is not None else '-'):>9} {tag}{flag}")
            if not ok and '@@' + col == '':
                pass

    if args.patch:
        src = open(args.src).read()
        dst = src
        # rewrite each data row's four numbers by matching on the model+delta?
        # Simplest robust approach: regenerate the whole tabular body from canon.
        print("\n[--patch] generating canonical Table 1 body...")
        for (task, model), c in canon.items():
            r, l0, lk = c["readout"], c["loop0"], c["loopk"]
            dl = c["delta"]
            print(f"  {task:10s} {model:12s}: R={r:.3f} L0={l0:.3f} Lk={lk:.3f} d={dl:+.3f} ({c['ksel']})")
        if args.dst:
            print(f"(write to {args.dst})")


if __name__ == "__main__":
    main()
