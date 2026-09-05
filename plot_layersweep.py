# Historical tooling: selected-layer significance is not a revised-paper claim.
# Use scripts/build_descriptive_figures.py for descriptive sweeps; see RESULTS.md.
"""Readout probe-placement sweeps — full 3 x 6 matrix (Ext 14/15/16/17).

Grid: rows = task (BoolQ / RuleTaker / ARC-Challenge), cols = model
(Qwen3-0.6B / 4B / 8B / Mistral-7B / Granite-3.1-8B / DeepSeek-V4-Flash).
One-pass readout accuracy as a function of the residual-stream layer the
verdict head taps. Paired-McNemar annotation (best tap vs final layer) where
a *_paired.json exists; the final-layer readout (last.mlp) is drawn as a
diamond when it is not itself a swept layer. Loop arms are horizontal
references (best loop arm present in the run + loop.zero).

Inputs:  results/{boolq,arc}_layersweep_{06b,4b,8b,mistral7b,granite8b,dsv4}.json
         results/ruletaker_layersweep_{tag}_n10k.json
         (+ matching _paired.json / _n10k_paired.json where present)
Output:  layersweep_placement.png
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL_MLP = "#d62728"
COL_LIN = "#7f7f7f"
COL_LOOP = "#1f77b4"

TASKS = ["boolq", "ruletaker", "arc"]
MODELS = [("06b", "Qwen3-0.6B"), ("4b", "Qwen3-4B"), ("8b", "Qwen3-8B"),
          ("mistral7b", "Mistral-7B"), ("granite8b", "Granite-3.1-8B"),
          ("dsv4", "DeepSeek-V4")]
TASK_TITLE = {"boolq": "BoolQ", "ruletaker": "RuleTaker",
              "arc": "ARC-Challenge"}


def acc(x):
    return float(x[0]) if isinstance(x, (list, tuple)) else float(x)


def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


fig, axes = plt.subplots(len(TASKS), len(MODELS), figsize=(26.0, 12.5))

for row, task in enumerate(TASKS):
    for col, (tag, model) in enumerate(MODELS):
        ax = axes[row, col]
        suffix = "_n10k" if task == "ruletaker" else ""
        d = json.load(open(f"results/{task}_layersweep_{tag}{suffix}.json"))
        try:
            paired = {r["pair"]: r for r in json.load(
                open(f"results/{task}_layersweep_{tag}{suffix}_paired.json"))["reports"]}
        except FileNotFoundError:
            paired = {}

        ls = d["layer_sweep"]
        layers = sorted(ls.keys(), key=int)
        L = np.array([int(l) for l in layers])
        mlp = np.array([acc(ls[l]["mlp"]) for l in layers])
        mlp_sd = np.array([float(ls[l]["mlp"][1]) if isinstance(ls[l]["mlp"], (list, tuple)) and len(ls[l]["mlp"]) > 1 else 0.0
                           for l in layers])
        lin = np.array([acc(ls[l]["linear"]) for l in layers])

        # final-layer readout (last.mlp) — add as a diamond if not swept
        last_layer = int(d["meta"].get("n_layers", L.max() + 1) - 1)
        last_mlp = acc(d["last.mlp"])

        # plateau onset: first swept layer within 1pt of the final-layer readout
        onset = next((int(l) for l, v in zip(L, mlp) if v >= last_mlp - 0.01), None)
        if onset is not None and onset < last_layer:
            ax.axvspan(onset, last_layer, color=COL_MLP, alpha=0.07, zorder=0)
        if last_layer not in L:
            ax.scatter([last_layer], [last_mlp], marker="D", s=55,
                       facecolor="white", edgecolor=COL_MLP, lw=2.0, zorder=5,
                       label=f"final-layer readout L{last_layer}")

        best_i = int(np.argmax(mlp))

        ax.plot(L, mlp, "-o", color=COL_MLP, lw=2.0, ms=5, zorder=4,
                label="readout mlp (per-seed mean)")
        if mlp_sd.any():
            ax.fill_between(L, mlp - mlp_sd, mlp + mlp_sd, color=COL_MLP, alpha=0.15)
        ax.plot(L, lin, "--s", color=COL_LIN, lw=1.2, ms=4, zorder=3,
                label="readout linear probe")

        loop_keys = [k for k in d if k.startswith("loop.")]
        if loop_keys:
            best_loop = max(loop_keys, key=lambda k: acc(d[k]))
            ax.axhline(acc(d[best_loop]), color=COL_LOOP, ls="--", lw=1.3,
                       alpha=0.85, label=f"{best_loop} {acc(d[best_loop]):.3f}")
            if "loop.zero" in loop_keys and best_loop != "loop.zero":
                ax.axhline(acc(d["loop.zero"]), color=COL_LOOP, ls=":", lw=1.2,
                           alpha=0.6, label=f"loop.zero {acc(d['loop.zero']):.3f}")

        ax.axvline(L[best_i], color=COL_MLP, ls=":", lw=1.0, alpha=0.5)

        # paired McNemar annotation: best swept tap vs final swept layer
        final_swept = L[-1]
        pair_key = f"readout.mlp.L{L[best_i]} vs readout.mlp.L{final_swept}"
        if pair_key in paired and L[best_i] != final_swept:
            r = paired[pair_key]
            p, sym = r["mcnemar_p"], sig_stars(r["mcnemar_p"])
            pstr = f"p={p:.1e}" if p < 0.01 else f"p={p:.2f}"
            txt = (f"best L{L[best_i]}={mlp[best_i]:.3f} vs "
                   f"L{final_swept}={mlp[-1]:.3f}\n{r['diff']:+.3f}  {pstr} {sym}")
        else:
            txt = f"best swept L{L[best_i]} = {mlp[best_i]:.3f}"
        if onset is not None and onset < last_layer:
            txt += f"\nplateau from L{onset}/{last_layer} ({onset / last_layer * 100:.0f}%)"
        ax.annotate(txt, xy=(L[best_i], mlp[best_i]),
                    xytext=(0.97, 0.03), textcoords="axes fraction",
                    ha="right", va="bottom", fontsize=8, color="black",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.92),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.7))

        if row == len(TASKS) - 1:
            ax.set_xlabel("residual layer tapped")
        if col == 0:
            ax.set_ylabel(TASK_TITLE[task])
        ax.set_title(f"{model} · {TASK_TITLE[task]}", fontsize=10)
        ax.set_xticks(L)
        ref_lo = [acc(d[k]) for k in loop_keys] + list(lin)
        ref_lo = [acc(d[k]) for k in loop_keys] + list(lin)
        ref_hi = [acc(d[k]) for k in loop_keys]
        ax.set_ylim(min(ref_lo) - 0.02,
                    max(list(ref_hi) + [mlp.max(), last_mlp]) + 0.03)
        ax.grid(True, axis="y", ls=":", alpha=0.5)
        if row == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

fig.suptitle("Readout placement sweeps (all batch 8) — mid-depth taps significantly beat the final layer at 11/18 cells,\n"
             "never lose significantly; shaded band = plateau onset (first layer within 1pt of the final-layer readout)",
             fontsize=12.5, y=1.01)
fig.tight_layout()
fig.savefig("paper/figures/layersweep_placement.png", dpi=150, bbox_inches="tight")
print("saved layersweep_placement.png")
