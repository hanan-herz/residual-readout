#!/usr/bin/env python3
# Historical tooling: incident-era estimator audit, not current source authority.
# Revised-paper source mappings and checks: scripts/build_descriptive_tables.py.
# See RESULTS.md; historical provenance is preserved in RESULTS_HISTORICAL.md.
"""Recover the canonical number per task x model cell from the artifacts.

Resolves the M1 "Table 1 vs Tables 2/4 disagree" complaint by computing, for
each cell, BOTH estimators from the on-disk artifacts:

  * mean   -> run JSON "last.mlp" (per-seed mean, the RESULTS/ext default)
  * vote   -> seed-ensemble majority vote, recomputed from *_rowpreds.npz

and reporting which run produced each "final-layer readout" number the paper
tables cite, so the tables can be made to agree.

Numbering convention documented in tables.tex header must be single-sourced:
this script is the single place that maps cell -> (minimal source set), so a
future re-run of the tables pulls identical numbers.

Usage:
  python scripts/extract_canonical.py [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "cloud_bench_cache")


# task -> model -> list of candidate run JSONs (kept in priority order).
# Layersweep runs are the canonical final-layer source (batch 8, pinned GPU);
# budget/plain runs are the alternate source used for the paired k-curve.
CELL_RUNS = {
    "boolq": {
        "0.6B":  ["boolq_layersweep_06b.json", "boolq_budget_06b.json", "results_20260808T162558_e83621.json"],
        "4B":    ["boolq_layersweep_4b.json",   "boolq_budget_4b.json",   "results_20260808T162241_cb93ed.json"],
        "8B":    ["boolq_layersweep_8b.json",   "boolq_budget_8b.json",   "results_20260808T154125_17f659.json"],
        "mistral": ["boolq_layersweep_mistral7b.json"],
        "granite": ["boolq_layersweep_granite8b.json"],
        "dsv4":  ["boolq_layersweep_dsv4.json"],
    },
    "ruletaker": {
        "0.6B":  ["ruletaker_layersweep_06b.json", "ruletaker_qwen06b_n2k.json"],
        "4B":    ["ruletaker_layersweep_4b.json",  "ruletaker_qwen4b_n2k.json"],
        "8B":    ["ruletaker_layersweep_8b.json",  "ruletaker_qwen8b_n2k.json"],
        "mistral": ["ruletaker_layersweep_mistral7b.json"],
        "granite": ["ruletaker_layersweep_granite8b.json"],
        "dsv4":  ["ruletaker_layersweep_dsv4.json", "ruletaker_dsv4_n2k.json"],
    },
    "arc": {
        "0.6B":  ["arc_layersweep_06b.json", "arc_qwen06b.json"],
        "4B":    ["arc_layersweep_4b.json",  "arc_qwen4b.json"],
        "8B":    ["arc_layersweep_8b.json",  "arc_qwen8b.json"],
        "mistral": ["arc_layersweep_mistral7b.json"],
        "granite": ["arc_layersweep_granite8b.json"],
        "dsv4":  ["arc_layersweep_dsv4.json", "arc_dsv4.json"],
    },
}

ORDER = ["0.6B", "4B", "8B", "mistral", "granite", "dsv4"]


def load_json(rel):
    p = os.path.join(RESULTS, rel)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def mlp_mean(d):
    v = d.get("last.mlp")
    if v is None:
        return None
    return v[0] if isinstance(v, list) else v


def find_rowpreds(task, model, layer_tag, pad, need="vote"):
    """Locate a rowpreds npz for the cell. Returns (path, keys) or (None, None)."""
    # cache path layout: cloud_bench_cache/<slug>/<task>/cache/...
    slug_map = {
        "0.6B": "Qwen__Qwen3-0.6B", "4B": "Qwen__Qwen3-4B", "8B": "Qwen__Qwen3-8B",
        "mistral": "mistralai__Mistral-7B-v0.3",
        "granite": "ibm-granite__granite-3.1-8b-base",
        "dsv4": "deepseek-ai__DeepSeek-V4-Flash-0731",
    }
    slug = slug_map[model]
    taskdir = os.path.join(CACHE, slug, task, "cache")
    if not os.path.isdir(taskdir):
        return None, None
    candidates = sorted(glob.glob(os.path.join(taskdir, "*rowpreds*.npz")))
    return candidates, None


def vote_from_npz(path, key="readout.mlp"):
    if path is None or not os.path.exists(path):
        return None, None
    try:
        z = np.load(path)
    except Exception:
        return None, None
    if key not in z.files:
        return None, None
    return float((z[key] == z["y_true"]).mean()), sorted(z.files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    out = {"cells": []}
    # emit every task x model in paper order
    for task in ["boolq", "ruletaker", "arc"]:
        for model in ORDER:
            if model not in CELL_RUNS[task]:
                continue
            cand = CELL_RUNS[task][model]
            loaded = {c: load_json(c) for c in cand}
            present = {c: d for c, d in loaded.items() if d is not None}

            # canonical final-layer number: prefer layersweep first-listed
            canonical_file = next((c for c in cand if present.get(c)), None)
            cd = present.get(canonical_file) if canonical_file else None

            means = {c: mlp_mean(d) for c, d in present.items()}
            canonical_mean = means.get(canonical_file)

            # candidate rowpreds for vote
            npz_path, _ = find_rowpreds(task, model, None, None)
            vote, keys = None, None
            if npz_path:
                # prefer a budget/loop-match npz carrying readout.mlp at final tap
                for p in npz_path:
                    v, k = vote_from_npz(p, "readout.mlp")
                    if v is not None:
                        vote, keys = v, k
                        npz_path = p
                        break

            cell = {
                "task": task, "model": model,
                "canonical_run": canonical_file,
                "n_val": cd.get("meta", {}).get("n_val") if cd else None,
                "loop_val": cd.get("meta", {}).get("loop_val") if cd else None,
                "last.mlp.mean": canonical_mean,
                "all_runs_mean": means,
                "vote": vote,
                "vote_source": os.path.relpath(npz_path, ROOT) if npz_path else None,
                "vote_keys": keys,
            }
            out["cells"].append(cell)

    if args.json:
        json.dump(out, open(args.json, "w"), indent=2)

    # ---- printable summary ----
    print(f"{'task':10s} {'model':8s} {'run':40s} {'mean':>7s} {'vote':>6s}  nval/loop")
    print("-" * 100)
    for c in out["cells"]:
        mean = f"{c['last.mlp.mean']:.4f}" if c["last.mlp.mean"] is not None else "-"
        vote = f"{c['vote']:.4f}" if c["vote"] is not None else "-"
        run = (c["canonical_run"] or "-")
        nv, lv = c["n_val"], c["loop_val"]
        print(f"{c['task']:10s} {c['model']:8s} {run:40s} {mean:>7s} {vote:>6s}  {nv}/{lv}")
    return out


if __name__ == "__main__":
    main()
