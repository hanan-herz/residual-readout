# Historical tooling: legacy ARC sources/framing, not revised-paper figures.
# Use scripts/build_descriptive_figures.py; see RESULTS.md for current caveats.
"""ARC-Challenge full: one-pass multi-class readout vs fair A–D loop.

Reads versioned shelf artifacts:

  Qwen3-0.6B  results/arc_qwen06b.json
  Qwen3-4B    results/arc_qwen4b.json
  Qwen3-8B    results/arc_qwen8b.json
  DeepSeek-V4 results/arc_dsv4.json

Saves arc_results.png under the repo root (full path printed on write).
"""
from pathlib import Path

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

ARTIFACTS = [
    ("Qwen3-0.6B", "results/arc_qwen06b.json"),
    ("Qwen3-4B", "results/arc_qwen4b.json"),
    ("Qwen3-8B", "results/arc_qwen8b.json"),
    ("DeepSeek-V4", "results/arc_dsv4.json"),
]


def _acc(x):
    return float(x[0]) if isinstance(x, (list, tuple)) else float(x)


def from_artifact(path: Path):
    r = json.loads(path.read_text())
    return _acc(r["last.mlp"]), float(r["loop.zero"]), float(r["loop.8"])


names, mlp_v, zero_v, k8_v = [], [], [], []
for name, rel in ARTIFACTS:
    m, z, k = from_artifact(ROOT / rel)
    names.append(name)
    mlp_v.append(m)
    zero_v.append(z)
    k8_v.append(k)

x = np.arange(len(names))
w = 0.26
fig, ax = plt.subplots(figsize=(9.0, 5.2))


def bar(off, vals, label, color):
    ax.bar(
        x + off, vals, w, label=label, color=color,
        edgecolor="white", linewidth=0.5,
    )
    for xi, v in enumerate(vals):
        ax.text(
            xi + off, v + 0.008, f"{v:.3f}",
            ha="center", va="bottom", fontsize=8, color="#222",
        )


bar(-w, mlp_v, "last.mlp (one-pass readout)", "#2c7fb8")
bar(0.0, zero_v, "loop.zero (fair zero-shot)", "#d95f02")
bar(+w, k8_v, "loop.8 (fair few-shot)", "#31a354")

ax.axhline(0.25, color="k", ls="--", lw=0.8)
ax.text(len(names) - 0.55, 0.26, "chance 0.25", ha="right", fontsize=8, color="#444")
ax.axhline(0.265, color="#888", ls=":", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylim(0.0, 1.05)
ax.set_ylabel("ARC-Challenge test accuracy (4-way)")
ax.set_title(
    "One-pass residual readout vs fair autoregressive loop (ARC-Challenge)\n"
    "full Challenge · train 1117 / test 1165 · A–D logprob loop · no passage",
    fontsize=10,
)
ax.legend(fontsize=8.5, loc="lower right")
plt.tight_layout()
out = ROOT / "paper/figures/arc_results.png"
plt.savefig(out, dpi=150)
print("saved", out)
