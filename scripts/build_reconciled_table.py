#!/usr/bin/env python3
# Historical tooling: incident-era table reconciliation, not revised-paper tables.
# Use scripts/build_descriptive_tables.py; Ext references below are historical
# (RESULTS_HISTORICAL.md), not current RESULTS.md conclusions.
"""Build a reconciled Table 1 (readout vs loop) from the on-disk artifacts.

Resolves M1 (paper review): the published Table 1 readout column and the
Table 2/4 `last.mlp` final column disagreed because they drew on different
estimators (4-seed majority *vote* from ``*_rowpreds.npz`` vs per-seed *mean*
from run JSON ``last.mlp``) and, for RuleTaker, different eval row sets
(n=400 loop rows vs full n_val).

Canonical convention adopted here (matches RESULTS.md / Ext 13 defaults):
  * readout column  -> per-seed MEAN, full-val row set (run JSON ``last.mlp``)
  * loop.0 / loop.k -> from the same run JSON
  * k chosen per task: BoolQ sweeps best-of k<=64 (pad 8192) where the arc
    was run; otherwise k=8.

Every number below is read from results/*.json. No hardcoding.

Usage:
  python scripts/build_reconciled_table.py [--out tables.tex-fragment]
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

# task -> (readout_json, loop_json, k_arm_for_delta)
CELLS = [
    # BoolQ (budget runs carry the k<=64 sweep at pad 8192)
    ("BoolQ", "Qwen3-0.6B", "boolq_budget_06b.json", "loop.64"),
    ("BoolQ", "Qwen3-4B",   "boolq_budget_4b.json",  "loop.64"),
    ("BoolQ", "Qwen3-8B",   "boolq_budget_8b.json",  "loop.16"),  # best loop arm (Ext 13: peaks k=16)
    ("BoolQ", "Mistral-7B", "boolq_layersweep_mistral7b.json", "loop.8"),
    ("BoolQ", "Granite-3.1-8B", "boolq_layersweep_granite8b.json", "loop.8"),
    ("BoolQ", "DeepSeek-V4", "boolq_layersweep_dsv4.json", "loop.8"),
    # RuleTaker
    ("RuleTaker", "Qwen3-0.6B", "ruletaker_layersweep_06b.json", "loop.8"),
    ("RuleTaker", "Qwen3-4B",   "ruletaker_layersweep_4b.json",  "loop.8"),
    ("RuleTaker", "Qwen3-8B",   "ruletaker_layersweep_8b.json",  "loop.8"),
    ("RuleTaker", "Mistral-7B", "ruletaker_layersweep_mistral7b.json", "loop.8"),
    ("RuleTaker", "Granite-3.1-8B", "ruletaker_layersweep_granite8b.json", "loop.8"),
    ("RuleTaker", "DeepSeek-V4", "ruletaker_layersweep_dsv4.json", "loop.8"),
    # ARC
    ("ARC", "Qwen3-0.6B", "arc_layersweep_06b.json", "loop.8"),
    ("ARC", "Qwen3-4B",   "arc_layersweep_4b.json",  "loop.8"),
    ("ARC", "Qwen3-8B",   "arc_layersweep_8b.json",  "loop.8"),
    ("ARC", "Mistral-7B", "arc_layersweep_mistral7b.json", "loop.8"),
    ("ARC", "Granite-3.1-8B", "arc_layersweep_granite8b.json", "loop.8"),
    ("ARC", "DeepSeek-V4", "arc_layersweep_dsv4.json", "loop.8"),
]


def mean_of(d):
    v = d.get("last.mlp")
    return v[0] if isinstance(v, list) else v


def r4(x):
    return f"{x:.3f}" if x is not None else "---"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write a .tex fragment here")
    args = ap.parse_args()

    lines = []
    header = (
        f"{'task':10s} {'model':12s} {'readout(mean)':>14s} "
        f"{'loop.0':>8s} {'loop.k':>8s} {'delta':>7s} k")
    print(header)
    print("-" * len(header))

    for task, model, fname, karm in CELLS:
        with open(os.path.join(RES, fname)) as fh:
            d = json.load(fh)
        readout = mean_of(d)
        loop0 = d.get("loop.zero")
        loopk = d.get(karm)
        delta = (readout - loopk) if (readout is not None and loopk is not None) else None
        print(f"{task:10s} {model:12s} {r4(readout):>14s} "
              f"{r4(loop0):>8s} {r4(loopk):>8s} {('+' if delta and delta>0 else '')+r4(delta):>7s} {karm}")

        if args.out:
            lines.append((task, model, readout, loop0, loopk, delta, karm))

    if args.out:
        with open(args.out, "w") as fh:
            for task, model, r, z, k, dl, karm in lines:
                fh.write(f"{task}\t{model}\t{r:.4f}\t{z:.4f}\t{k:.4f}\t"
                         f"{dl:+.4f}\t{karm}\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
