#!/usr/bin/env python3
"""Plot stored descriptive results only; no model loading, fitting, or tests."""
from __future__ import annotations

import json
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from build_descriptive_tables import BUDGET, CELLS, ROOT


def main():
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/build_descriptive_tables.py"),
                    "--check"], check=True)
    names = {name for _, _, readout, loop, sweep in CELLS
             for name in (readout, loop or readout, sweep)} | {name for _, name in BUDGET}
    sources = {name: json.loads((ROOT / "results" / name).read_text()) for name in names}
    out = ROOT / "paper/figures"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.alpha": 0.18,
        "pdf.fonttype": 42,
    })
    blue, orange, green = "#2c7fb8", "#d95f02", "#238b45"
    tasks = ("BoolQ", "RuleTaker", "ARC")
    models = [model for task, model, *_ in CELLS if task == "BoolQ"]
    labels = {"BoolQ": "BoolQ", "RuleTaker": "RuleTaker", "ARC": "ARC-Challenge"}
    metadata = {"CreationDate": None, "ModDate": None,
                "Subject": "Descriptive plots from stored results; no significance tests"}

    def save(fig, name):
        path = out / f"descriptive_{name}.pdf"
        fig.savefig(path, bbox_inches="tight", metadata=metadata)
        plt.close(fig)
        print(f"Wrote {path}")

    # Seed SDs here are 0.1-0.7 accuracy points: an error bar for them is
    # shorter than the marker that hides it, so plotting one promises an
    # encoding the figure cannot render. Report the bound in the caption
    # instead -- see the SEED-SD line printed below.
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 4.1), sharey=True)
    readout_sds = []
    for ax, task in zip(axes, tasks):
        for i, (_, model, readout, loop, _) in enumerate(c for c in CELLS if c[0] == task):
            d, scorer = sources[readout], sources[loop or readout]
            mean, sd = d["last.mlp"]
            assert 40 <= 100 * (mean - sd) <= 100 * (mean + sd) <= 100, (task, model)
            assert all(40 <= 100 * scorer[key] <= 100 for key in ("loop.zero", "loop.8")), (task, model)
            readout_sds.append(sd)
            ax.plot(100 * mean, i - 0.2, "o", color=blue, markersize=4, zorder=4)
            ax.plot(100 * scorer["loop.zero"], i, "s", color=orange, markersize=4)
            ax.plot(100 * scorer["loop.8"], i + 0.2, "^", color=green, markersize=5)
        ax.set(title=labels[task], xlabel="Accuracy (%)", xlim=(40, 100),
               xticks=[40, 60, 80, 100], yticks=range(6))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticklabels(models)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], color=color, marker=marker, linestyle="none", label=label)
               for color, marker, label in [(blue, "o", "Final readout (four-seed mean)"),
                                            (orange, "s", "Zero-shot scorer"),
                                            (green, "^", "Requested 8-shot scorer")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save(fig, "main")
    print(f"SEED-SD main: readout seed SD {100 * min(readout_sds):.2f}-"
          f"{100 * max(readout_sds):.2f}pp (not plotted; smaller than the markers)")

    # Same reasoning as Figure 1: these refit SDs are 0.1-0.4 accuracy points.
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.7), sharey=True)
    ks = (8, 16, 32, 64)
    refit_sds = []
    for ax, (model, name) in zip(axes, BUDGET):
        d = sources[name]
        for field, color, marker, label in [("budget", blue, "o", "First-n refit"),
                                             ("budget.balanced", green, "s", "Balanced refit")]:
            means, sds = zip(*(d[field][str(k)] for k in ks))
            assert all(40 <= 100 * (m - s) <= 100 * (m + s) <= 95
                       for m, s in zip(means, sds)), (model, field)
            refit_sds.extend(sds)
            ax.plot(ks, [100 * m for m in means], color=color, marker=marker,
                    markersize=4, label=label)
        assert all(40 <= 100 * d[f"loop.{k}"] <= 95 for k in ks), model
        assert 40 <= 100 * d["last.mlp"][0] <= 95, model
        ax.plot(ks, [100 * d[f"loop.{k}"] for k in ks], color=orange, marker="^",
                markersize=4, label="Requested-k scorer")
        ax.axhline(100 * d["last.mlp"][0], color="0.4", linestyle="--", linewidth=1,
                   label="9427-label readout mean")
        ax.set_xscale("log", base=2)
        ax.set(title=model, xlabel="Labels n / requested shots k", ylim=(40, 95),
               xticks=ks, xticklabels=[str(k) for k in ks])
    axes[0].set_ylabel("BoolQ accuracy (%)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    save(fig, "budget")
    print(f"SEED-SD budget: refit seed SD {100 * min(refit_sds):.2f}-"
          f"{100 * max(refit_sds):.2f}pp (not plotted; smaller than the markers)")

    # The swept-curve bands DO render here (7-10 points wide, alpha 0.18), so
    # unlike Figures 1-2 this figure keeps its SD encoding. The final-layer
    # diamond gets no error bar: a single point's SD would be occluded again.
    fig, axes = plt.subplots(6, 3, figsize=(8.4, 9.4), sharey=True)
    sweep_sds = []
    for task, model, _, _, sweep in CELLS:
        ax = axes[models.index(model), tasks.index(task)]
        d = sources[sweep]
        layers = sorted(d["layer_sweep"], key=int)
        means, sds = zip(*(d["layer_sweep"][layer]["mlp"] for layer in layers))
        assert all(20 <= 100 * (m - s) <= 100 * (m + s) <= 100
                   for m, s in zip(means, sds)), (task, model)
        xs = [int(layer) for layer in layers]
        sweep_sds.extend(sds)
        ax.plot(xs, [100 * m for m in means], color=blue, marker="o", markersize=3)
        ax.fill_between(xs, [100 * (m - s) for m, s in zip(means, sds)],
                        [100 * (m + s) for m, s in zip(means, sds)], color=blue, alpha=0.18)
        final = d["meta"]["n_layers"] - 1
        mean, sd = d["last.mlp"]
        assert 20 <= 100 * (mean - sd) <= 100 * (mean + sd) <= 100, (task, model)
        ax.plot(final, 100 * mean, "D", color=orange, markersize=4, zorder=4)
        ax.set(ylim=(20, 100), yticks=[20, 40, 60, 80, 100], xticks=[xs[0], final // 2, final])
        if task == "BoolQ":
            ax.set_ylabel(model + "\nAccuracy (%)", fontsize=9)
        if model == models[0]:
            ax.set_title(labels[task])
        if model == models[-1]:
            ax.set_xlabel("Layer (zero-based)")
    fig.legend(handles=[Line2D([], [], color=blue, marker="o", label="Swept MLP mean; band = ± seed SD"),
                        Line2D([], [], color=orange, marker="D", linestyle="none",
                               label="Final layer, same source run")],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=0.7)
    save(fig, "layers")
    print(f"SEED-SD layers: swept seed SD {100 * min(sweep_sds):.2f}-"
          f"{100 * max(sweep_sds):.2f}pp (plotted as bands)")


if __name__ == "__main__":
    main()
