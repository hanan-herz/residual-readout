# Historical tooling: best-loop/significance framing is not a revised-paper task law.
# Use scripts/build_descriptive_figures.py; see RESULTS.md for current caveats.
"""Task-law summary figure — 3x6 heat panel (readout - best loop, 18 cells).

Rows = task (BoolQ / RuleTaker 10k/4k / ARC-Challenge), cols = model
(Qwen3-0.6B / 4B / 8B / Mistral-7B / Granite-3.1-8B / DeepSeek-V4-Flash).
Each cell is colored by Delta = readout - best loop (paired, same rows) and
annotated with the delta and McNemar significance stars. The "task law" reads
off the panel directly: green everywhere except three red cells — small
models on parametric ARC, the frontier MoE on serial RuleTaker, and
Mistral-7B on BoolQ.

The (delta, p) values are read from the canonical results/*.json artifacts
(single source of truth) via the same logic as scripts/build_table1.py:
delta = readout.mean - best loop arm, McNemar p recomputed on the shared
full-val rows from the *_rowpreds.npz. Output: tasklaw_summary.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.build_table1 import CELLS, mean_of, paired_p
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")

TASKS = ["boolq", "ruletaker", "arc"]
MODELS = [("Qwen3-0.6B", "Qwen3-0.6B"), ("Qwen3-4B", "Qwen3-4B"), ("Qwen3-8B", "Qwen3-8B"),
          ("Mistral-7B", "Mistral-7B"), ("Granite-3.1-8B", "Granite-3.1-8B"),
          ("DeepSeek-V4-Flash", "DeepSeek-V4")]
TASK_TITLE = {"boolq": "BoolQ", "ruletaker": "RuleTaker 10k/4k",
              "arc": "ARC-Challenge"}


def load_cells():
    """Return {(task_lower, model): (delta, p)} from canonical artifacts."""
    out = {}
    for task, model, fname, loop_fname in CELLS:
        d = json.load(open(os.path.join(RES, fname)))
        readout = mean_of(d)
        loop_src = json.load(open(os.path.join(RES, loop_fname))) if loop_fname else d
        loops = {k: v for k, v in loop_src.items()
                 if k.startswith("loop.") and isinstance(v, (int, float))}
        best_arm = max(loops, key=lambda k: loops[k])
        bestv = loops[best_arm]
        delta = readout - bestv
        p, _ = paired_p(task, model, best_arm)
        out[(task.lower(), model)] = (delta, p)
    return out


CANONICAL = load_cells()


def cell(task, suf):
    # suf is the model display name in MODELS
    model = dict((m, m) for m, _ in MODELS)[suf]
    delta, p = CANONICAL[(task, model)]
    return (delta, p, None, None)


def stars(p):
    if p is None:
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


def main():
    Z = np.zeros((len(TASKS), len(MODELS)))
    P = np.full((len(TASKS), len(MODELS)), np.nan)
    for i, t in enumerate(TASKS):
        for j, (suf, _) in enumerate(MODELS):
            delta, p, _, _ = cell(t, suf)
            Z[i, j] = delta
            P[i, j] = p if p is not None else np.nan

    lim = max(0.12, np.nanmax(np.abs(Z)))
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    im = ax.imshow(Z, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m for _, m in MODELS], fontsize=9)
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels([TASK_TITLE[t] for t in TASKS], fontsize=10)
    ax.xaxis.set_ticks_position("top")

    for i in range(len(TASKS)):
        for j in range(len(MODELS)):
            dv = Z[i, j]
            pv = P[i, j]
            s = stars(pv) if not np.isnan(pv) else ""
            ax.text(j, i, f"{dv:+.3f}{s}", ha="center", va="center",
                    fontsize=9, fontweight="bold" if s else "normal",
                    color="black")

    ax.set_title("readout − best loop (paired, same rows)  ·  *p<0.05  **p<0.01  ***p<0.001",
                 fontsize=10, pad=26)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(MODELS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(TASKS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("readout − best loop (accuracy)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig("paper/figures/tasklaw_summary.png", dpi=170, bbox_inches="tight")
    print("wrote tasklaw_summary.png")


if __name__ == "__main__":
    main()
