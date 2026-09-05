# Historical tooling: vote/best-arm significance, not revised-paper budget evidence.
# Use scripts/build_descriptive_figures.py; see RESULTS.md#samek-caveats.
"""Ext 13 k-curve: budget-matched loop vs one-pass readout (BoolQ, Qwen3 arc).

Three panels (0.6B / 4B / 8B). Loop accuracy plotted as a curve over
few-shot exemplars k (0, 8, 16, 32, 64, loop_pad_max=8192); the one-pass
readout as a flat reference (both the 4-seed majority vote a deployment uses
and the per-seed mean ± std reported elsewhere). Per panel, annotate the
paired significance (McNemar) of readout vs the best loop arm.

Inputs (from Arm A / Ext 13):
  results/boolq_budget_{06b,4b,8b}.json        loop arms + last.mlp per-seed
  results/boolq_budget_{06b,4b,8b}_paired.json  readout seed-vote + McNemar p

Saves boolq_budget_kcurve.png.
"""
import json
from copy import deepcopy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIZES = ["06b", "4b", "8b"]
LABELS = {"06b": "Qwen3-0.6B", "4b": "Qwen3-4B", "8b": "Qwen3-8B"}
KS = [0, 8, 16, 32, 64]

# best loop arm to test readout against, per scale (argmax over loop arms)
BEST = {"06b": "loop.64", "4b": "loop.64", "8b": "loop.16"}
STARS = {"06b": "***", "4b": "ns", "8b": "ns"}
COL_LOOP = "#1f77b4"
COL_READ = "#d62728"


def load(size):
    b = json.load(open(f"results/boolq_budget_{size}.json"))
    p = json.load(open(f"results/boolq_budget_{size}_paired.json"))
    loops = {k: b[k] for k, v in b.items() if k.startswith("loop.") and isinstance(v, (int, float))}
    # readout seed-vote = acc of readout.mlp (all paired rows share acc_a)
    vote = p["reports"][0]["acc_a"]
    mean, std = b["last.mlp"]
    return loops, vote, mean, std


def load_pair(size):
    p = json.load(open(f"results/boolq_budget_{size}_paired.json"))
    return {r["pair"]: r for r in p["reports"]}


fig, axes = plt.subplots(1, len(SIZES), figsize=(14.5, 4.6), sharey=False)

for ax, size in zip(axes, SIZES):
    loops, vote, mean, std = load(size)
    pairs = load_pair(size)

    xs = np.arange(len(KS))
    # loop curve (loop.zero for k=0, else loop.{k})
    ys = [loops["loop.zero" if k == 0 else f"loop.{k}"] for k in KS]
    ax.plot(xs, ys, "-o", color=COL_LOOP, lw=1.8, ms=5, zorder=3,
            label="loop (k exemplars, pad 8192)")
    for x, y, k in zip(xs, ys, KS):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5, color=COL_LOOP)

    # readout: seed-vote (solid) + per-seed mean ± std (dashed band)
    ax.axhline(vote, color=COL_READ, lw=1.8, zorder=2,
               label=f"readout (seed-vote {vote:.3f})")
    ax.axhline(mean, color=COL_READ, ls="--", lw=1.0, alpha=0.8,
               label=f"readout (per-seed mean ± std)")
    ax.fill_between(xs, mean - std, mean + std, color=COL_READ, alpha=0.10)

    # paired significance: readout vs best loop arm
    best = BEST[size]
    k_ix = KS.index(int(best.split(".")[1]))
    b_acc = ys[k_ix]
    r = pairs[f"readout.mlp vs {best}"]
    p = r["mcnemar_p"]
    pstr = f"p={p:.1e}" if p < 0.001 else f"p={p:.3f}"
    sym = "ns" if p >= 0.05 else ("***" if p < 0.001 else "*")
    ax.annotate(f"{r['diff']:+.3f} vs {best}\n{pstr} {sym}",
                xy=(k_ix, b_acc), xytext=(k_ix, b_acc - 0.012),
                ha="center", va="top", fontsize=8.5, color="black",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6",
                          alpha=0.85))

    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("few-shot exemplars $k$ (balanced)")
    ax.set_title(LABELS[size], fontsize=12)
    # per-panel y margin
    lo = min(vote, mean, *ys) - 0.02
    hi = max(vote, mean, *ys) + 0.02
    ax.set_ylim(lo, hi)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    if size == "8b":
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

fig.suptitle("Budget-matched loop k-curve vs one-pass readout — BoolQ (full val, n=3270, paired)",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("paper/figures/boolq_budget_kcurve.png", dpi=150, bbox_inches="tight")
print("saved boolq_budget_kcurve.png")
