# Historical tooling: configuration-derived strata are not question-level proof depth.
# Not revised-paper evidence; see RESULTS.md and scripts/build_descriptive_figures.py.
"""RuleTaker 10k/4k depth plot: per-depth MLP + loop when ``stratum_depth_loop`` exists.

Skips missing artifacts. Saves ruletaker_depth_strata.png.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

ARTIFACTS = [
    ("Qwen3-0.6B", "results/ruletaker_qwen06b_n10k.json"),
    ("Qwen3-4B", "results/ruletaker_qwen4b_n10k.json"),
    ("Qwen3-8B", "results/ruletaker_qwen8b_n10k.json"),
    ("DeepSeek-V4", "results/ruletaker_dsv4_n10k.json"),
]

DEPTH_ORDER = ["0", "1", "2", "3", "5", "NatLang"]
COLORS = ["#2c7fb8", "#41b6c4", "#238b45", "#e6550d"]


def _acc(x):
    if x is None:
        return np.nan
    return float(x[0]) if isinstance(x, (list, tuple)) else float(x)


def load(path: Path):
    r = json.loads(path.read_text())
    fair = r.get("stratum_depth_loop")
    full = r.get("stratum_depth") or {}
    mode = "fair" if fair else "legacy"
    src = fair or full

    def series(key):
        ys, ns = [], []
        for d in DEPTH_ORDER:
            info = (src or {}).get(d) or (src or {}).get(str(d))
            if not info:
                ys.append(np.nan)
                ns.append(0)
                continue
            ns.append(int(info.get("n") or 0))
            if key == "mlp":
                ys.append(_acc(info.get("mlp")))
            else:
                ys.append(_acc(info.get(key)))
        return ys, ns

    mlp_y, ns = series("mlp")
    z_y, _ = series("loop.zero") if fair else ([np.nan] * len(DEPTH_ORDER), ns)
    k8_y, _ = series("loop.8") if fair else ([np.nan] * len(DEPTH_ORDER), ns)
    return {
        "mode": mode,
        "mlp": mlp_y,
        "loop_zero": z_y,
        "loop_8": k8_y,
        "n": ns,
        "overall_mlp": _acc(r["last.mlp"]),
        "overall_matched": _acc(r.get("last.mlp.loop_matched")),
        "overall_z": float(r.get("loop.zero", np.nan)),
        "overall_k8": float(r.get("loop.8", np.nan)),
    }


def main():
    series = []
    ns_ref = None
    for name, rel in ARTIFACTS:
        p = ROOT / rel
        if not p.exists() or p.stat().st_size < 100:
            print("skip missing", rel)
            continue
        data = load(p)
        series.append((name, data))
        if ns_ref is None:
            ns_ref = data["n"]
        print(f"  {name}: mode={data['mode']} mlp={data['overall_mlp']:.3f} "
              f"k8={data['overall_k8']:.3f}")

    if not series:
        raise SystemExit("no result files found")

    # --- figure: overall bars + depth curves ---
    fig = plt.figure(figsize=(12.5, 9.5))
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)

    # Overall grouped bars
    names = [n for n, _ in series]
    x = np.arange(len(names))
    w = 0.22
    mlp_v = [d["overall_mlp"] for _, d in series]
    match_v = [d["overall_matched"] for _, d in series]
    z_v = [d["overall_z"] for _, d in series]
    k8_v = [d["overall_k8"] for _, d in series]

    def bars(off, vals, label, color):
        ax0.bar(x + off, vals, w, label=label, color=color,
                edgecolor="white", linewidth=0.5)
        for xi, v in enumerate(vals):
            if np.isnan(v):
                continue
            ax0.text(xi + off, v + 0.008, f"{v:.3f}", ha="center",
                     fontsize=7, color="#222")

    bars(-1.5 * w, mlp_v, "last.mlp (full val)", "#2c7fb8")
    bars(-0.5 * w, match_v, "mlp loop-matched", "#7bccc4")
    bars(0.5 * w, z_v, "loop.zero", "#d95f02")
    bars(1.5 * w, k8_v, "loop.8", "#31a354")
    ax0.axhline(0.5, color="k", ls="--", lw=0.7, alpha=0.5)
    ax0.set_xticks(x)
    ax0.set_xticklabels(names, fontsize=10)
    ax0.set_ylim(0.45, 1.0)
    ax0.set_ylabel("Accuracy")
    ax0.set_title(
        "RuleTaker 10k/4k — overall (train 10000 · val 4000 · loop 4000, identical rows)",
        fontsize=11,
    )
    ax0.legend(fontsize=8, loc="lower right", ncol=2)
    ax0.grid(axis="y", alpha=0.25)

    # Depth curves (fair same-row when available)
    xd = np.arange(len(DEPTH_ORDER))
    for (name, data), color in zip(series, COLORS):
        ax1.plot(xd, data["mlp"], marker="o", lw=2.2, ms=7, color=color,
                 label=f"{name} mlp")
        if data["mode"] == "fair":
            ax1.plot(xd, data["loop_8"], marker="s", lw=1.5, ms=5, color=color,
                     ls="--", label=f"{name} loop.8")
    ax1.axhline(0.5, color="k", ls="--", lw=0.7, alpha=0.5)
    labels = [f"d={d}\n(n={n})" for d, n in zip(DEPTH_ORDER, ns_ref or [0] * 6)]
    # n from fair loop set if present
    if series[0][1]["mode"] == "fair":
        labels = [f"d={d}\n(n={n})" for d, n in zip(DEPTH_ORDER, series[0][1]["n"])]
    ax1.set_xticks(xd)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0.45, 1.0)
    ax1.set_ylabel("Accuracy (loop_val rows)")
    ax1.set_xlabel("Depth — same rows for mlp vs loop.8 (stratum_depth_loop)")
    ax1.set_title(
        "Per-depth: solid+○ = mlp · dashed+□ = loop.8 (fair, same n per depth)",
        fontsize=11,
    )
    # compact legend: one entry per model style
    handles, labels_ = ax1.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, lab in zip(handles, labels_):
        if lab in seen:
            continue
        seen.add(lab)
        h2.append(h)
        l2.append(lab)
    ax1.legend(h2, l2, fontsize=7.5, loc="lower left", ncol=2, framealpha=0.95)
    ax1.grid(axis="y", alpha=0.25)
    ax1.text(
        0.5, -0.18,
        "loop_zero omitted from depth panel for clarity; overall bars include it. "
        "Depth n is among the full val 4000 (identical rows for mlp and loop).",
        transform=ax1.transAxes, ha="center", fontsize=8, color="#444",
    )

    fig.tight_layout()
    out = ROOT / "paper/figures/ruletaker_depth_strata.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out.resolve())


if __name__ == "__main__":
    main()
