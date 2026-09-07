#!/usr/bin/env python3
"""Build paper/tables.tex from explicit local JSONs, using only the stdlib.

No fitting, model loading, prediction files, statistical tests, or network I/O.
Run with --check to verify the committed tables, and --report for source counts
and unrounded descriptive values. Stored SDs use the producer's ddof=0.

--report also prints a crude layer-selection noise scale (see selection_noise);
it is arithmetic on stored accuracies for the prose in the placement section,
not a hypothesis test, and no table cell depends on it.
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
    "main": r"""\textbf{Fixed final-layer readout and closed-set scorer accuracies}
(accuracy). Readout is the stored \texttt{last.mlp} mean $\pm$ seed SD
(four seeds; SD uses divisor four, not a confidence interval).
\texttt{loop.zero} and \texttt{loop.8} are the stored artifact keys for the
zero-shot and requested 8-shot scorer arms (\S\ref{sec:scorer}); they are
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
\emph{Cand.}\ is the number of recorded sweep entries the maximum was taken
over, from 4 to 10 across cells. The optimism of a maximum grows with it, so
the column is needed to read the selected means. The noise scale quoted in
\S\ref{sec:placement} is the binomial standard error at the selected accuracy
on that cell's evaluation set, times the expected maximum of \emph{Cand.}\
independent standard normals.
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
RP max, its dimension, perm, and noise are the stored
\texttt{last.randproj} values:
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


# Expected maximum of L independent standard normals, L = 4..10 (Monte-Carlo,
# rounded). Used only to print a descriptive scale for how far the maximum of L
# noisy layer estimates sits above a single one; it is not a hypothesis test.
EMAX = {4: 1.029, 5: 1.163, 6: 1.267, 7: 1.352, 8: 1.423, 9: 1.485, 10: 1.539}


def selection_noise(accuracy, n_eval, n_candidates):
    """Scale of the upward bias in a maximum over ``n_candidates`` layer means.

    Binomial standard error at the observed accuracy times the expected maximum
    of that many independent standard normals. Deliberately crude: layer
    readouts are positively correlated across depth, which shrinks the real
    inflation, so this over-states it and is a conservative yardstick.
    """
    se = math.sqrt(accuracy * (1 - accuracy) / n_eval)
    return EMAX[n_candidates] * se


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
    placement = {}
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
        controls = [d["last.mlp.shufl"][0], rp["max"], rp["d"], rp["perm"],
                    rp["noise"], d["ctx.mlp"][0], mean]
        assert rp["d"] in (64, 256), readout_file
        assert all(math.isfinite(v) and 0 <= v <= 1
                   for v in [mean, sd, zero, eight, selected, final_mean,
                             controls[0], controls[1], controls[3], controls[4],
                             controls[5], controls[6]]), (task, model)

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
            prefix + f"{len(layers)} & L{best} & {selected:.3f} & L{final} & {final_mean:.3f}" + r" \\",
        ])
        rows["controls"].extend([
            f"% Readout: {readout_file}",
            prefix + f"{controls[0]:.3f} & {controls[1]:.3f} & {controls[2]} & "
            + " & ".join(f"{value:.3f}" for value in controls[3:]) + r" \\",
        ])
        deltas.setdefault(task, []).append(delta)
        previous = task
        if args.report:
            print(f"MAIN {task} {model}: mean={mean!r}, sd={sd!r}, zero={zero!r}, eight={eight!r}, delta={delta!r}")
            print(f"PLACEMENT {task} {model}: cand={len(layers)}, L{best}={selected!r}, L{final}={final_mean!r}, "
                  f"gap_pp={100 * (selected - final_mean):.2f}, "
                  f"selection_noise_pp={100 * selection_noise(selected, sweep["meta"]["n_val"], len(layers)):.2f}, "
                  f"headline_final_difference={final_mean - mean!r}")
            print(f"CONTROLS {task} {model}: {controls!r}")
        placement.setdefault(task, []).append(
            (selected - final_mean, selection_noise(selected, sweep["meta"]["n_val"], len(layers)), len(layers)))

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

    header = ("% Generated by python3 scripts/build_descriptive_tables.py; do not edit numbers.\n"
              "% Sources are explicit local results/*.json mappings in that stdlib-only script.\n")
    # \fittable lives in its own file because the manuscript inputs the four
    # per-table files separately; \providecommand keeps a double input harmless.
    macros = header + r"""% Requires booktabs and graphicx. Input once before any table file.
\providecommand{\fittable}[1]{%
  \sbox0{#1}%
  \ifdim\wd0>\textwidth
    \resizebox{\textwidth}{!}{\usebox0}%
  \else
    \usebox0
  \fi
}
"""
    preamble = header + "% Shared by the paper and the standalone wrapper.\n\\input{tables_macros.tex}\n\n"
    built = {
        "main": table("main", "llcccc", r"Task & Model & Readout mean $\pm$ SD & \texttt{loop.zero} & \texttt{loop.8} & $\Delta$ vs.\ 8-shot", rows["main"]),
        "placement": table("placement", "llccccc", "Task & Model & Cand. & Selected layer & Selected mean & Final layer & Final mean", rows["placement"]),
        "controls": table("controls", "llccccccc", "Task & Model & Shuffle & RP max & RP d & RP perm & RP noise & Context & Final", rows["controls"]),
        "budget": table("budget", "lcccc", r"Model & $n=k$ & First-$n$ refit & Class-balanced refit & \texttt{loop.k}", rows["budget"]),
    }
    # tables.tex keeps all four for the standalone main_tables.tex wrapper. The
    # manuscript inputs the per-table files instead, so each wide float is
    # declared next to the text that discusses it; declaring all four up front
    # fills the float queue and pushes every figure past its own reference.
    text = preamble + "\n".join(built.values())
    outputs = {args.out: text, args.out.with_name("tables_macros.tex"): macros}
    for label, body in built.items():
        outputs[args.out.with_name(f"table_{label}.tex")] = header + body
    if args.report:
        for task, values in deltas.items():
            print(f"SUMMARY {task}: {sum(v > 0 for v in values)}/{len(values)} positive deltas vs loop.8; range={min(values):+.6f}..{max(values):+.6f}")
        for task, values in placement.items():
            scales = [100 * noise for _, noise, _ in values]
            print(f"PLACEMENT-SUMMARY {task}: cand={sorted({c for *_, c in values})}, "
                  f"noise scale {min(scales):.1f}-{max(scales):.1f}pp, "
                  f"{sum(gap > noise for gap, noise, _ in values)}/{len(values)} gaps exceed it")
        flat = [v for values in placement.values() for v in values]
        scales = [100 * noise for _, noise, _ in flat]
        print(f"PLACEMENT-SUMMARY total: noise scale {min(scales):.1f}-{max(scales):.1f}pp, "
              f"{sum(gap > noise for gap, noise, _ in flat)}/{len(flat)} gaps exceed it")
    if args.check:
        for path, body in outputs.items():
            if not path.exists() or path.read_text() != body:
                raise SystemExit(f"STALE: {path}; run this script without --check")
        print(f"OK: {len(outputs)} table files match {len(sources)} validated JSON sources")
    else:
        for path, body in outputs.items():
            path.write_text(body)
        print(f"Wrote {len(outputs)} table files from {len(sources)} validated JSON sources")


if __name__ == "__main__":
    main()
