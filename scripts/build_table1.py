#!/usr/bin/env python3
# Historical tooling: best-loop tables, not the revised-paper reporting convention.
# Use scripts/build_descriptive_tables.py (--check to validate); see RESULTS.md.
"""build_table1.py — regenerate Table 1 (readout vs loop) from canonical runs.

Reads the canonical (or *_rerun) run JSONs + their per-row preds npz, and
emits a Table-1 row set where EVERY number is derived from artifacts:

  * Readout column   -> run JSON ``last.mlp`` (per-seed mean, full val)
  * loop.0 / loop.k  -> run JSON loop arms
  * loop.k selection -> best-of loop.0 / loop.k (the strong baseline)
  * Delta            -> readout.mean - loop.k (same columns shown)
  * p (McNemar)      -> recomputed on the shared loop rows from the
                        *_rowpreds.npz (readout vote vs loop.k on identical
                        rows); falls back to the run's stored paired stat

Sources per cell default to the canonical runs, but any cell can be pointed
at a *_rerun.json (the full-val loop re-runs). Run
  python scripts/build_table1.py --out paper/tables_row1.tex --report
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "cloud_bench_cache")

SLUG = {
    "Qwen3-0.6B": "Qwen__Qwen3-0.6B", "Qwen3-4B": "Qwen__Qwen3-4B",
    "Qwen3-8B": "Qwen__Qwen3-8B", "Mistral-7B": "mistralai__Mistral-7B-v0.3",
    "Granite-3.1-8B": "ibm-granite__granite-3.1-8b-base",
    "DeepSeek-V4-Flash": "deepseek-ai__DeepSeek-V4-Flash-0731",
}

# task -> model -> (run_json, loop_json). loop_json overrides the loop arms /
# loop_val source when the readout run and the full-val loop run differ
# (BoolQ Mistral/Granite: layersweep has the readout+placement, the *_rerun
# file has the full-3270 loop). Otherwise loop_json == run_json.
#
# For paper Table 1 verification, prefer scripts/verify_tables.py CELLS
# (Scheme A / paper/tables.tex, RuleTaker layersweep). This builder's
# RuleTaker rows use the 10k/4k JSONs (same source as the paper figures).
CELLS = [
    ("BoolQ", "Qwen3-0.6B", "boolq_budget_06b.json", None),
    ("BoolQ", "Qwen3-4B", "boolq_budget_4b.json", None),
    ("BoolQ", "Qwen3-8B", "boolq_budget_8b.json", None),
    ("BoolQ", "Mistral-7B", "boolq_layersweep_mistral7b.json", "boolq_mistral7b_rerun.json"),
    ("BoolQ", "Granite-3.1-8B", "boolq_layersweep_granite8b.json", "boolq_granite8b_rerun.json"),
    ("BoolQ", "DeepSeek-V4-Flash", "boolq_layersweep_dsv4.json", None),
    ("RuleTaker", "Qwen3-0.6B", "ruletaker_qwen06b_n10k.json", None),
    ("RuleTaker", "Qwen3-4B", "ruletaker_qwen4b_n10k.json", None),
    ("RuleTaker", "Qwen3-8B", "ruletaker_qwen8b_n10k.json", None),
    ("RuleTaker", "Mistral-7B", "ruletaker_mistral7b_n10k.json", None),
    ("RuleTaker", "Granite-3.1-8B", "ruletaker_granite8b_n10k.json", None),
    ("RuleTaker", "DeepSeek-V4-Flash", "ruletaker_dsv4_n10k.json", None),
    ("ARC", "Qwen3-0.6B", "arc_layersweep_06b.json", None),
    ("ARC", "Qwen3-4B", "arc_layersweep_4b.json", None),
    ("ARC", "Qwen3-8B", "arc_layersweep_8b.json", None),
    ("ARC", "Mistral-7B", "arc_layersweep_mistral7b.json", None),
    ("ARC", "Granite-3.1-8B", "arc_layersweep_granite8b.json", None),
    ("ARC", "DeepSeek-V4-Flash", "arc_layersweep_dsv4.json", None),
]

TASK_DIR = {"BoolQ": "boolq", "RuleTaker": "ruletaker", "ARC": "arc"}


def mean_of(d):
    v = d.get("last.mlp")
    return v[0] if isinstance(v, list) else v


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2.0 * stats.binom.cdf(k, n, 0.5)))


def paired_p(task, model, karm):
    """Find a rowpreds npz for the cell and compute McNemar of
    readout.mlp vs karm on shared rows. Returns (p, n) or (None, None)."""
    cdir = os.path.join(CACHE, SLUG[model], TASK_DIR[task], "cache")
    if not os.path.isdir(cdir):
        return None, None
    # prefer the largest (full-val) rowpreds npz
    cands = sorted(glob.glob(os.path.join(cdir, "*rowpreds*.npz")))
    best = None
    for p in cands:
        try:
            z = np.load(p)
        except Exception:
            continue
        if "readout.mlp" in z.files and karm in z.files:
            if best is None or len(z["y_true"]) > best[2]:
                best = (p, z, len(z["y_true"]))
    if best is None:
        return None, None
    _, z, n = best
    a = z["readout.mlp"] == z["y_true"]
    b = z[karm] == z["y_true"]
    p = mcnemar_exact(int((a & ~b).sum()), int((~a & b).sum()))
    return p, n


def star(p):
    if p is None:
        return ""
    return "\\pstarstarstar" if p < 1e-3 else "\\pstarstar" if p < 1e-2 else "\\pstar" if p < 5e-2 else ""


def tex_p(p):
    """Format a p-value in the paper's scientific style, e.g. $2.4\\mathrm{e}{-05}$."""
    if p is None:
        return "---"
    if p >= 0.001:
        return f"{p:.3f}".rstrip("0").rstrip(".")
    import math
    exp = int(math.floor(math.log10(p)))
    mant = p / 10 ** exp
    return f"${mant:.1f}\\mathrm{{e}}{{{exp:03d}}}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    rows = []
    for task, model, fname, loop_fname in CELLS:
        path = os.path.join(RES, fname)
        if not os.path.exists(path):
            rows.append((task, model, fname, None))
            continue
        d = json.load(open(path))
        readout = mean_of(d)
        # loop arms may come from a separate full-val loop run
        loop_src = json.load(open(os.path.join(RES, loop_fname))) if loop_fname else d
        loops = {k: v for k, v in loop_src.items()
                 if k.startswith("loop.") and isinstance(v, (int, float))}
        loop0 = loops.get("loop.zero")
        loop8 = loops.get("loop.8")
        # best-of-loop: strongest loop arm shown (the fair baseline). For the
        # BoolQ sweep this is the best k; elsewhere loop.8 (or loop.zero if it
        # beats loop.8, e.g. RuleTaker 0.6B where few-shot hurts).
        best_arm = max(loops, key=lambda k: loops[k]) if loops else None
        bestv = loops.get(best_arm)
        delta = (readout - bestv) if (readout is not None and bestv is not None) else None
        p, n = paired_p(task, model, best_arm) if best_arm else (None, None)
        rows.append((task, model, fname,
                     dict(readout=readout, loop0=loop0, loop8=loop8,
                          best_arm=best_arm, bestv=bestv,
                          delta=delta, p=p, n=n)))

    print(f"{'task':10s}{'model':16s}{'readout':>9s}{'loop0':>8s}{'loop8':>8s}{'best':>7s}{'bestv':>8s}{'delta':>8s}{'p':>12s}{'n':>7s}")
    print("-" * 100)
    for task, model, fname, r in rows:
        if r is None:
            print(f"{task:10s}{model:16s}  MISSING {fname}")
            continue
        pstr = f"{r['p']:.1e}" if r['p'] is not None else "-"
        nstr = str(r['n']) if r['n'] else "-"
        bstr = r['best_arm'].replace('loop.', '')
        print(f"{task:10s}{model:16s}{r['readout']:>9.4f}{r['loop0']:>8.4f}"
              f"{(r['loop8'] if r['loop8'] is not None else float('nan')):>8.4f}"
              f"{bstr:>7s}{r['bestv']:>8.4f}{r['delta']:>+8.3f}{pstr:>12s}{nstr:>7s}")

    if args.out:
        with open(args.out, "w") as fh:
            for task, model, fname, r in rows:
                if r is None:
                    continue
                readout, loop0, loop8 = r["readout"], r["loop0"], r["loop8"]
                bestv, delta, p = r["bestv"], r["delta"], r["p"]
                barm = r["best_arm"].replace("loop.", "")
                sig = p is not None and p < 0.05
                # bold the winning column (readout if delta>0, best loop if delta<0)
                rc = f"\\textbf{{{readout:.3f}}}" if delta > 0 else f"{readout:.3f}"
                bc = f"\\textbf{{{bestv:.3f}}}" if delta < 0 else f"{bestv:.3f}"
                l8 = f"{loop8:.3f}" if loop8 is not None else "---"
                # negatives go in math mode (proper minus), positives plain
                dnum = f"${delta:+.3f}$" if delta < 0 else f"{delta:+.3f}"
                dcell = f"\\sd{{{dnum}}}{star(p)}" if sig else dnum
                pcell = tex_p(p)
                fh.write(f" & {model} & {rc} & {loop0:.3f} & {l8} & "
                         f"{bc}\\,({barm}) & {dcell} & {pcell} \\\\\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
