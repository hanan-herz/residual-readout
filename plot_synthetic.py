# Historical tooling: separate synthetic exploration, not revised-paper evidence.
# See RESULTS_SYNTHETIC.md and RESULTS.md for the distinction.
"""Synthetic multi-hop mechanism figure (RESULTS_SYNTHETIC.md).

Two panels over Qwen3 scale (0.6B / 4B / 8B):
  left  — one-pass readouts (linear probe, MLP) vs the decoder loop
          (zero-shot, few-shot) on multi-hop transitivity;
  right — the MLP readout's margin over the few-shot decoder.

Numbers are hard-coded from RESULTS_SYNTHETIC.md (700 train / 175 val,
shuffled-facts transitivity); regenerate the md values before trusting this.
Output: synthetic_multihop.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ["Qwen3-0.6B", "Qwen3-4B", "Qwen3-8B"]

# RESULTS_SYNTHETIC.md headline table (val accuracy)
PROBE = [0.931, 0.966, 0.971]      # linear probe (last layer)
MLP = [0.909, 0.954, 0.966]        # MLP readout
DEC_ZS = [0.463, 0.474, 0.474]     # decoder zero-shot
DEC_FS = [0.491, 0.846, 0.680]     # decoder few-shot

COL_PROBE = "#7f7f7f"
COL_MLP = "#d62728"
COL_ZS = "#bbbbbb"
COL_FS = "#1f77b4"


def main():
    x = np.arange(len(MODELS))
    w = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8),
                                   gridspec_kw={"width_ratios": [2, 1]})

    ax1.bar(x - 1.5 * w, PROBE, w, label="linear probe (1-pass)", color=COL_PROBE)
    ax1.bar(x - 0.5 * w, MLP, w, label="MLP readout (1-pass)", color=COL_MLP)
    ax1.bar(x + 0.5 * w, DEC_ZS, w, label="decoder zero-shot", color=COL_ZS)
    ax1.bar(x + 1.5 * w, DEC_FS, w, label="decoder few-shot", color=COL_FS)
    ax1.axhline(0.5, color="#999999", lw=0.8, ls=":")
    ax1.set_xticks(x)
    ax1.set_xticklabels(MODELS, fontsize=9)
    ax1.set_ylim(0.4, 1.0)
    ax1.set_ylabel("val accuracy", fontsize=9)
    ax1.set_title("Multi-hop transitivity: one-pass readout vs decoder loop",
                  fontsize=10, pad=18)
    # Legend outside the axes so labels are not drawn over the bars.
    ax1.legend(
        fontsize=8,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
        ncols=4,
        columnspacing=1.0,
        handlelength=1.2,
        borderaxespad=0,
    )
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    margin = [m - f for m, f in zip(MLP, DEC_FS)]
    bars = ax2.bar(x, margin, 0.55, color=COL_MLP)
    ax2.axhline(0.0, color="#999999", lw=0.8)
    for xi, m in zip(x, margin):
        ax2.text(xi, m + 0.01, f"+{m:.2f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(MODELS, fontsize=9)
    ax2.set_ylabel("MLP − few-shot decoder", fontsize=9)
    ax2.set_title("One-pass margin over the loop", fontsize=10)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.suptitle(
        "The relation is in the residual; the loop never exceeds a one-pass readout",
        fontsize=10,
        y=1.08,
    )
    fig.tight_layout()
    fig.savefig(
        "paper/figures/synthetic_multihop.png",
        dpi=170,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    print("wrote synthetic_multihop.png")


if __name__ == "__main__":
    main()
