"""Plot the multi-head routing comparison and shuffle controls."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Matched multi-head pilot: 1,000 train / 400 validation per task.
SCALES = ["Qwen3-0.6B\n(L27)", "Qwen3-8B\n(L35)"]
ROUTERS = {
    "Gold task ceiling": ([1.000, 1.000], [0.620, 0.830]),
    "Task-ID linear": ([0.998, 1.000], [0.619, 0.830]),
    "Mahalanobis": ([0.993, 0.997], [0.618, 0.830]),
    "Naive max confidence": ([0.141, 0.577], [0.109, 0.780]),
}
ROUTER_COLORS = {
    "Gold task ceiling": "#222222",
    "Task-ID linear": "#2c7fb8",
    "Mahalanobis": "#31a354",
    "Naive max confidence": "#d95f02",
}


def _label_bars(ax, containers, decimals=2, fontsize=7):
    for container in containers:
        for bar in container:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.{decimals}f}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color="#222",
            )


def plot_scale_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharex=True)
    x = np.arange(len(SCALES))
    width = 0.19

    for ax, metric_index, title, ylabel in [
        (axes[0], 0, "Task routing accuracy", "Task accuracy"),
        (axes[1], 1, "Answer accuracy after routing", "Answer accuracy"),
    ]:
        containers = []
        for i, (name, values) in enumerate(ROUTERS.items()):
            vals = values[metric_index]
            bars = ax.bar(
                x + (i - 1.5) * width,
                vals,
                width,
                label=name,
                color=ROUTER_COLORS[name],
                edgecolor="white",
                linewidth=0.5,
            )
            containers.append(bars)
        _label_bars(ax, containers)
        if metric_index == 0:
            ax.axhline(1 / 3, color="#666", linestyle="--", linewidth=0.8)
            ax.text(
                1.48,
                1 / 3 + 0.025,
                "chance = .33",
                ha="right",
                fontsize=7,
                color="#555",
            )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_xticks(x)
        ax.set_xticklabels(SCALES, fontsize=9)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Multi-head single-pass routing: Qwen3 scale comparison\n"
        "Matched pilot · 1,000 train / 400 validation per task · 1,200 mixed examples",
        fontsize=12,
        y=1.03,
    )
    fig.tight_layout()
    path = OUT / "multihead_scale_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path)


SHUFFLE = {
    "0.6B L1": {
        "Linear": [1.000, 0.908, 0.912],
        "Mahalanobis": [1.000, 0.465, 0.450],
    },
    "8B L1": {
        "Linear": [0.997, 0.997, 0.970],
        "Mahalanobis": [1.000, 0.935, 0.387],
    },
    "8B L35": {
        "Linear": [1.000, 0.999, 0.998],
        "Mahalanobis": [0.997, 0.884, 0.939],
    },
}
CONDITIONS = ["Identity", "Word shuffle", "Token shuffle"]


def plot_shuffle_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)
    x = np.arange(len(CONDITIONS))
    width = 0.24
    colors = {"0.6B L1": "#7570b3", "8B L1": "#1b9e77", "8B L35": "#d95f02"}

    for ax, router in zip(axes, ["Linear", "Mahalanobis"]):
        containers = []
        for i, (setting, values) in enumerate(SHUFFLE.items()):
            bars = ax.bar(
                x + (i - 1) * width,
                values[router],
                width,
                label=setting,
                color=colors[setting],
                edgecolor="white",
                linewidth=0.5,
            )
            containers.append(bars)
        _label_bars(ax, containers, decimals=2, fontsize=7)
        ax.set_title(f"{router} router", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Task routing accuracy", fontsize=9)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Shuffle control: supervised routing is robust; Mahalanobis depends on layer",
        fontsize=12,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.01,
        "8B L1 Mahalanobis collapses under token shuffle; the same router is robust at L35",
        ha="center",
        fontsize=8.5,
        color="#444",
    )
    fig.tight_layout()
    path = OUT / "multihead_shuffle_layers.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    plot_scale_comparison()
    plot_shuffle_comparison()
