"""Historical plot; not a revised-paper figure or current reporting convention.

Use scripts/build_descriptive_figures.py for the current manuscript.

Head-to-head: BoolQ (full-val) · RuleTaker (10k/4k) · ARC-Challenge (full).

Same six models, same three bars:
  readout (last.mlp, per-seed mean) · loop.zero · best loop

All numbers read from the canonical artifacts (single source of truth, same
as scripts/build_table1.py) — readout.mean, loop.zero, and the best loop arm
(best k for the BoolQ sweep, k=8 elsewhere).

Protocols differ by design —
  BoolQ:      train 9427 / val 3270 / loop 3270
  RuleTaker:  train 10000 / val 4000 / loop 4000  (10k/4k)
  ARC:        train 1117 / test 1165 / loop 1165 (full Challenge, 4-way)

Saves:
  head_to_head_three_tasks.png
"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.build_table1 import CELLS, RES, mean_of

ROOT = Path(__file__).resolve().parent

# canonical cells grouped by task, each (model, readout, loop.zero, best loop)
def _by_task(task):
    out = []
    for t, model, fname, loop_fname in CELLS:
        if t != task:
            continue
        d = json.load(open(os.path.join(RES, fname)))
        readout = mean_of(d)
        loop_src = json.load(open(os.path.join(RES, loop_fname))) if loop_fname else d
        loops = {k: v for k, v in loop_src.items()
                 if k.startswith("loop.") and isinstance(v, (int, float))}
        best_arm = max(loops, key=lambda k: loops[k])
        out.append((model.replace("-Flash", ""), readout,
                    loops.get("loop.zero"), loops[best_arm]))
    return out


BOOLQ = _by_task("BoolQ")
RULETAKER = _by_task("RuleTaker")
ARC = _by_task("ARC")

COLORS = {
    "mlp": "#2c7fb8",
    "zero": "#d95f02",
    "k8": "#31a354",
}


def panel(ax, rows, *, title, ylabel, base_rate=None, note=None,
          ylim=(0.5, 1.0), show_legend=True):
    names = [r[0] for r in rows]
    mlp_v = [r[1] for r in rows]
    zero_v = [r[2] for r in rows]
    best_v = [r[3] for r in rows]

    x = np.arange(len(names))
    w = 0.26

    def bar(off, vals, label, color):
        ax.bar(
            x + off, vals, w, label=label, color=color,
            edgecolor="white", linewidth=0.5,
        )
        for xi, v in enumerate(vals):
            ax.text(
                xi + off, v + 0.006, f"{v:.3f}",
                ha="center", va="bottom", fontsize=6.5, color="#222",
            )

    bar(-w, mlp_v, "readout (one-pass)", COLORS["mlp"])
    bar(0.0, zero_v, "loop.zero", COLORS["zero"])
    bar(+w, best_v, "best loop (few-shot)", COLORS["k8"])

    if base_rate is not None:
        ax.axhline(base_rate, color="k", ls="--", lw=0.8)
        ax.text(
            len(names) - 0.5, base_rate + 0.008,
            f"chance/base ~{base_rate:.2f}",
            ha="right", fontsize=6.5, color="#444",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=18, ha="right")
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9.5)
    if note:
        ax.text(
            0.5, -0.26, note, transform=ax.transAxes,
            ha="center", va="top", fontsize=7, color="#444",
        )
    if show_legend:
        ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharey=False)

    panel(
        axes[0],
        BOOLQ,
        title="BoolQ — full validation",
        ylabel="Accuracy",
        base_rate=0.62,
        note="train 9427 · val 3270 · loop full · Yes/No",
        ylim=(0.55, 1.0),
        show_legend=True,
    )
    panel(
        axes[1],
        RULETAKER,
        title="RuleTaker — 10k/4k",
        ylabel="Accuracy",
        base_rate=0.50,
        note="train 10000 · val 4000 · loop 4000 · in-context rules",
        ylim=(0.50, 0.90),
        show_legend=False,
    )
    panel(
        axes[2],
        ARC,
        title="ARC-Challenge — full test",
        ylabel="Accuracy",
        base_rate=0.25,
        note="train 1117 · test 1165 · loop full · A–D · parametric",
        ylim=(0.20, 1.05),
        show_legend=False,
    )

    fig.suptitle(
        "One-pass residual readout vs fair autoregressive loop\n"
        "BoolQ (passage) · RuleTaker (in-context rules) · ARC (parametric knowledge)",
        fontsize=11.5,
        y=1.03,
    )
    fig.tight_layout()
    out = ROOT / "paper/figures/head_to_head_three_tasks.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out.resolve())


if __name__ == "__main__":
    main()
