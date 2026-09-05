# Historical tooling: best-loop selection, not the revised fixed-eight-shot view.
# Use scripts/build_descriptive_figures.py; see RESULTS.md for current caveats.
"""BoolQ headline chart: one-pass readout vs the fair autoregressive loop.

All six models, full-val (9427 train / 3270 val), read from the canonical
artifacts (single source of truth, same as scripts/build_table1.py):

  readout  = last.mlp (per-seed mean)
  loop.0   = zero-shot fair loop
  best     = best loop arm (best k in {0..64} for Qwen sweep, k=8 elsewhere)

Saves boolq_results.png.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.build_table1 import CELLS, RES, mean_of

# canonical BoolQ cells, in plot order (ascending scale, cross-family, MoE)
BOOLQ = [(task, model, fname, loop_fname)
         for (task, model, fname, loop_fname) in CELLS if task == "BoolQ"]


def from_cell(fname, loop_fname):
    d = json.load(open(os.path.join(RES, fname)))
    readout = mean_of(d)
    loop_src = json.load(open(os.path.join(RES, loop_fname))) if loop_fname else d
    loops = {k: v for k, v in loop_src.items()
             if k.startswith("loop.") and isinstance(v, (int, float))}
    best_arm = max(loops, key=lambda k: loops[k])
    return readout, loops.get("loop.zero"), loops[best_arm], best_arm.replace("loop.", "")


names, mlp_v, zero_v, best_v, best_arm = [], [], [], [], []
for task, model, fname, loop_fname in BOOLQ:
    m, z, b, arm = from_cell(fname, loop_fname)
    names.append(model.replace("-Flash", ""))
    mlp_v.append(m); zero_v.append(z); best_v.append(b); best_arm.append(arm)

x = np.arange(len(names))
w = 0.26
fig, ax = plt.subplots(figsize=(10.5, 5.2))


def bar(off, vals, label, color):
    ax.bar(x + off, vals, w, label=label, color=color, edgecolor="white", linewidth=0.5)
    for xi, v in enumerate(vals):
        ax.text(xi + off, v + 0.006, f"{v:.3f}", ha="center", va="bottom",
                fontsize=7.5, color="#222")


bar(-w, mlp_v,  "readout (one-pass, per-seed mean)", "#2c7fb8")
bar(0.0, zero_v, "loop.zero (fair zero-shot)", "#d95f02")
bar(+w, best_v, "best loop (few-shot)", "#31a354")
# annotate which loop arm won per cell
for xi, (b, arm) in enumerate(zip(best_v, best_arm)):
    ax.text(xi + w, b - 0.02, f"k={arm}", ha="center", va="top",
            fontsize=6.5, color="#1a5c1a")

ax.axhline(0.62, color="k", ls="--", lw=0.8)
ax.text(len(names) - 0.5, 0.624, "base rate ~0.62", ha="right", fontsize=7.5, color="#444")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("BoolQ validation accuracy")
ax.set_title("One-pass residual readout vs fair autoregressive loop (BoolQ)\n"
             "full val (9427 train / 3270 val), identical rows · best loop arm annotated",
             fontsize=10)
ax.legend(fontsize=8.5, loc="lower right")

plt.tight_layout()
out = "paper/figures/boolq_results.png"
plt.savefig(out, dpi=150)
print("saved", out)
print(f"models: {names}")
