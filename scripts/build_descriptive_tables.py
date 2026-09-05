#!/usr/bin/env python3
"""Build paper/tables.tex from explicit local JSONs, using only the stdlib.

No fitting, model loading, prediction files, statistical tests, or network I/O.
Run with --check to verify the committed tables, and --report for source counts
and unrounded descriptive values. Stored SDs use the producer's ddof=0.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# task, model, headline readout, loop override, placement sweep.
# The first four fields match build_table1.py CELLS; do not import that builder
# (it imports scientific packages and selects a best loop baseline).
CELLS = [
    ("BoolQ", "Qwen3-0.6B", "boolq_budget_06b.json", None, "boolq_layersweep_06b.json"),
    ("BoolQ", "Qwen3-4B", "boolq_budget_4b.json", None, "boolq_layersweep_4b.json"),
    ("BoolQ", "Qwen3-8B", "boolq_budget_8b.json", None, "boolq_layersweep_8b.json"),
    ("BoolQ", "Mistral-7B", "boolq_layersweep_mistral7b.json", "boolq_mistral7b_rerun.json", "boolq_layersweep_mistral7b.json"),
    ("BoolQ", "Granite-3.1-8B", "boolq_layersweep_granite8b.json", "boolq_granite8b_rerun.json", "boolq_layersweep_granite8b.json"),
    ("BoolQ", "DeepSeek-V4-Flash", "boolq_layersweep_dsv4.json", None, "boolq_layersweep_dsv4.json"),
    ("RuleTaker", "Qwen3-0.6B", "ruletaker_qwen06b_n10k.json", None, "ruletaker_layersweep_06b_n10k.json"),
    ("RuleTaker", "Qwen3-4B", "ruletaker_qwen4b_n10k.json", None, "ruletaker_layersweep_4b_n10k.json"),
    ("RuleTaker", "Qwen3-8B", "ruletaker_qwen8b_n10k.json", None, "ruletaker_layersweep_8b_n10k.json"),
    ("RuleTaker", "Mistral-7B", "ruletaker_mistral7b_n10k.json", None, "ruletaker_layersweep_mistral7b_n10k.json"),
    ("RuleTaker", "Granite-3.1-8B", "ruletaker_granite8b_n10k.json", None, "ruletaker_layersweep_granite8b_n10k.json"),
    ("RuleTaker", "DeepSeek-V4-Flash", "ruletaker_dsv4_n10k.json", None, "ruletaker_layersweep_dsv4_n10k.json"),
    ("ARC", "Qwen3-0.6B", "arc_layersweep_06b.json", None, "arc_layersweep_06b.json"),
    ("ARC", "Qwen3-4B", "arc_layersweep_4b.json", None, "arc_layersweep_4b.json"),
    ("ARC", "Qwen3-8B", "arc_layersweep_8b.json", None, "arc_layersweep_8b.json"),
    ("ARC", "Mistral-7B", "arc_layersweep_mistral7b.json", None, "arc_layersweep_mistral7b.json"),
    ("ARC", "Granite-3.1-8B", "arc_layersweep_granite8b.json", None, "arc_layersweep_granite8b.json"),
    ("ARC", "DeepSeek-V4-Flash", "arc_layersweep_dsv4.json", None, "arc_layersweep_dsv4.json"),
]
BUDGET = [
    ("Qwen3-0.6B", "boolq_samek_06b.json"),
    ("Qwen3-4B", "boolq_samek_4b.json"),
    ("Qwen3-8B", "boolq_samek_8b.json"),
]
COUNTS = {"boolq": (9427, 3270), "ruletaker": (10000, 4000), "arc": (1000, 1165)}
SHORT_LOOPS = {"boolq_layersweep_mistral7b.json", "boolq_layersweep_granite8b.json"}

CAPTIONS = {
    "main": r"""\textbf{Fixed final-layer readout and closed-set loop scores}
(accuracy). Readout is the stored \texttt{last.mlp} mean $\pm$ seed SD
(four seeds; SD uses divisor four, not a confidence interval).
\texttt{loop.zero} and \texttt{loop.8} are zero-shot and requested 8-shot
accuracies under the manuscript's product-of-token-probabilities heuristic,
not marginalized answer probabilities or free-form generation.
$\Delta=$ readout mean $-$ \texttt{loop.8}, with no baseline selection
or significance testing. Artifact-declared train/evaluation/loop counts:
BoolQ 9427/3270/3270; RuleTaker 10000/4000/4000;
ARC-Challenge 1000/1165/1165. BoolQ Mistral/Granite readouts use sweep
files and loops use full-evaluation reruns, not the sweeps' 405-row loops.
Readout token cap is 384; loop cap is 8192 for BoolQ Qwen and 2048
otherwise. These are different input and supervision protocols, not a
matched-label-budget comparison; requested demonstrations are not measured
retained counts.""",
    "placement": r"""\textbf{Descriptive readout placement sweeps} (accuracy).
Selected layer maximizes the stored MLP seed mean over recorded sweep
entries (including the final layer where swept). Layer indices are zero-based. Selected and
final means come from the same sweep JSON; the final column uses that
sweep file's \texttt{last.mlp}, not a value imported from Table~\ref{tab:main}.
ARC Qwen3-0.6B has no L27 sweep entry; its final mean is stored separately
in the same file.
These evaluation-selected maxima are optimistic, not held-out gains;
no significance tests are reported. Sweep train/evaluation counts are
9427/3270 (BoolQ), 10000/4000 (RuleTaker), and 1000/1165 (ARC-Challenge).
RuleTaker uses the separate 10k/4k sweep files because the headline
10k/4k files contain no sweep. Source runs and protocols may differ from
the headline sources even when displayed final means coincide.""",
    "controls": r"""\textbf{Descriptive control readouts} (accuracy).
All entries use the headline readout JSONs and train/evaluation counts
of Table~\ref{tab:main}. Shuffle is the stored mean for an MLP refit on
permuted training labels (\texttt{last.mlp.shufl}); context and final are
\texttt{ctx.mlp} and \texttt{last.mlp} seed means.
RP max, perm, and noise are the stored \texttt{last.randproj} values:
linear probes on Gaussian random projections, feature-column-permuted
inputs, and Gaussian noise inputs. RP dimension is selected on evaluation
accuracy from 64 and 256 using three-seed means; perm and noise use the
selected dimension. These are descriptive measurements, not leakage
tests or causal evidence.""",
    "budget": r"""\textbf{BoolQ label-budget descriptions} (accuracy).
First-$n$ and class-balanced columns are stored refit seed means from
\texttt{budget} and \texttt{budget.balanced} in the same model's
\texttt{samek} JSON; loop values are that JSON's \texttt{loop.k}.
$n=k$ compares artifact-declared readout label counts with requested
demonstration counts, not measured retained demonstration counts.
Exact subset identity is unavailable, so equal counts do not establish
identical labeled examples. The declared training pool has 9427 rows;
readout and loop evaluation counts are both 3270. Readout/loop token caps
are 384/8192. No significance tests are reported.""",
}


def table(label, columns, header, rows):
    return "\n".join([
        r"\begin{table*}[t]", r"\centering",
        r"\caption{" + CAPTIONS[label] + "}", r"\label{tab:" + label + "}",
        r"\small", r"\setlength{\tabcolsep}{4pt}%", r"\fittable{%",
        r"\begin{tabular}{@{}" + columns + "@{}}", r"\toprule",
        header + r" \\", r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular}%", "}", r"\end{table*}", "",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "paper/tables.tex")
    parser.add_argument("--check", action="store_true", help="Compare without writing")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    names = {name for _, _, readout, loop, sweep in CELLS
             for name in (readout, loop or readout, sweep)} | {name for _, name in BUDGET}
    sources = {}
    for name in sorted(names):
        with (ROOT / "results" / name).open() as handle:
            d = json.load(handle)
        m = d["meta"]
        counts = (m["n_train"], m["n_val"], m["loop_val"])
        train, val = COUNTS[m["task"]]
        assert counts == (train, val, 405 if name in SHORT_LOOPS else val), (name, counts)
        cap = 8192 if m["task"] == "boolq" and m["model_id"].startswith("Qwen/") else 2048
        assert (m["batch"], m["pad_max"], m["loop_pad_max"]) == (8, 384, cap), name
        assert m["val_split"] == ("validation" if m["task"] == "boolq" else "test"), name
        sources[name] = d
        if args.report:
            print(f"SOURCE {name}: train/val/loop={counts}, caps=384/{cap}")

    rows = {label: [] for label in CAPTIONS}
    deltas = {}
    previous = None
    for task, model, readout_file, loop_file, sweep_file in CELLS:
        d, loop, sweep = (sources[name] for name in
                          (readout_file, loop_file or readout_file, sweep_file))
        for source in (d, loop, sweep):
            assert source["meta"]["task"] == task.lower(), (task, model)
            assert source["meta"]["model_id"] == d["meta"]["model_id"], (task, model)
        assert loop["meta"]["loop_val"] == d["meta"]["n_val"], (task, model)
        mean, sd = d["last.mlp"]
        zero, eight = loop["loop.zero"], loop["loop.8"]
        delta = mean - eight
        layers = sweep["layer_sweep"]
        best = max(sorted(layers, key=int), key=lambda layer: layers[layer]["mlp"][0])
        final = str(sweep["meta"]["n_layers"] - 1)
        selected, final_mean = layers[best]["mlp"][0], sweep["last.mlp"][0]
        assert sweep["meta"]["head_file"] == f"heads/head_l{final}.npz", sweep_file
        if final in layers:
            assert layers[final]["mlp"] == sweep["last.mlp"], sweep_file
        assert int(best) == sweep["meta"]["layer_sweep_best"]["layer"], sweep_file
        rp = d["last.randproj"]
        controls = [d["last.mlp.shufl"][0], rp["max"], rp["perm"], rp["noise"],
                    d["ctx.mlp"][0], mean]
        assert rp["d"] in (64, 256), readout_file
        assert all(math.isfinite(v) and 0 <= v <= 1
                   for v in [mean, sd, zero, eight, selected, final_mean, *controls]), (task, model)

        if task != previous and previous is not None:
            for label in ("main", "placement", "controls"):
                rows[label].append(r"\midrule")
        display_task = "ARC-Challenge" if task == "ARC" else task
        prefix = f"{display_task if task != previous else ''} & {model} & "
        rows["main"].extend([
            f"% Readout: {readout_file}; loops: {loop_file or readout_file}",
            prefix + f"${mean:.3f} \\pm {sd:.3f}$ & {zero:.3f} & {eight:.3f} & ${delta:+.3f}$" + r" \\",
        ])
        rows["placement"].extend([
            f"% Sweep: {sweep_file}",
            prefix + f"L{best} & {selected:.3f} & L{final} & {final_mean:.3f}" + r" \\",
        ])
        rows["controls"].extend([
            f"% Readout: {readout_file}",
            prefix + " & ".join(f"{value:.3f}" for value in controls) + r" \\",
        ])
        deltas.setdefault(task, []).append(delta)
        previous = task
        if args.report:
            print(f"MAIN {task} {model}: mean={mean!r}, sd={sd!r}, zero={zero!r}, eight={eight!r}, delta={delta!r}")
            print(f"PLACEMENT {task} {model}: L{best}={selected!r}, L{final}={final_mean!r}, headline_final_difference={final_mean - mean!r}")
            print(f"CONTROLS {task} {model}: {controls!r}, rp_dimension={rp['d']}")

    for model, name in BUDGET:
        d = sources[name]
        m = d["meta"]
        parent_file = next(readout for task, cell_model, readout, _, _ in CELLS
                           if task == "BoolQ" and cell_model == model)
        assert m["parent_run"] == "results/" + parent_file, name
        assert m["parent_run_id"] == sources[parent_file]["meta"]["run_id"], name
        assert m["kind"] == "samek" and m["readout_n"] == "8,16,32,64", name
        assert m["budget_protocol"] == "first-n (budget) and class-balanced (budget.balanced)", name
        if rows["budget"]:
            rows["budget"].append(r"\midrule")
        rows["budget"].append(f"% Source: {name}")
        for k in (8, 16, 32, 64):
            first, balanced, loop = d["budget"][str(k)][0], d["budget.balanced"][str(k)][0], d[f"loop.{k}"]
            assert loop == sources[parent_file][f"loop.{k}"], name
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in (first, balanced, loop)), name
            rows["budget"].append(f"{model if k == 8 else ''} & {k} & {first:.3f} & {balanced:.3f} & {loop:.3f}" + r" \\")
            if args.report:
                print(f"BUDGET {model} k={k}: first={first!r}, balanced={balanced!r}, loop={loop!r}")

    preamble = r"""% Generated by python3 scripts/build_descriptive_tables.py; do not edit numbers.
% Sources are explicit local results/*.json mappings in that stdlib-only script.
% Requires booktabs and graphicx. Shared by the paper and standalone wrapper.
\newcommand{\fittable}[1]{%
  \sbox0{#1}%
  \ifdim\wd0>\textwidth
    \resizebox{\textwidth}{!}{\usebox0}%
  \else
    \usebox0
  \fi
}

"""
    text = preamble + "\n".join([
        table("main", "llcccc", r"Task & Model & Readout mean $\pm$ SD & \texttt{loop.zero} & \texttt{loop.8} & $\Delta$ vs.\ 8-shot", rows["main"]),
        table("placement", "llcccc", "Task & Model & Selected layer & Selected mean & Final layer & Final mean", rows["placement"]),
        table("controls", "llcccccc", "Task & Model & Shuffle & RP max & RP perm & RP noise & Context & Final", rows["controls"]),
        table("budget", "lcccc", r"Model & $n=k$ & First-$n$ refit & Class-balanced refit & \texttt{loop.k}", rows["budget"]),
    ])
    if args.report:
        for task, values in deltas.items():
            print(f"SUMMARY {task}: {sum(v > 0 for v in values)}/{len(values)} positive deltas vs loop.8; range={min(values):+.6f}..{max(values):+.6f}")
    if args.check:
        if args.out.read_text() != text:
            raise SystemExit(f"STALE: {args.out}; run this script without --check")
        print(f"OK: {args.out} matches {len(sources)} validated JSON sources")
    else:
        args.out.write_text(text)
        print(f"Wrote {args.out} from {len(sources)} validated JSON sources")


if __name__ == "__main__":
    main()
