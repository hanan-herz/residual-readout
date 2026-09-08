# Figures and Tables

The current manuscript uses three descriptive vector PDFs and four tables.
Both builders use the explicit `CELLS` and `BUDGET` mappings in
`scripts/build_descriptive_tables.py`: 32 unique stored JSON sources under
`results/`, not a directory-wide glob. No model inference, classifier
fitting, or significance testing is performed by these builders.

## Current Outputs

| Output | Content | Builder |
|---|---|---|
| `paper/figures/descriptive_main.pdf` | Final-layer readout means, with separate zero-shot and requested-eight-shot scorer arms (seed SD stated in the caption, not drawn) | `scripts/build_descriptive_figures.py` |
| `paper/figures/descriptive_budget.pdf` | BoolQ first-n and balanced refits, requested-shot scores, and full-training reference | `scripts/build_descriptive_figures.py` |
| `paper/figures/descriptive_layers.pdf` | Recorded MLP layer sweeps and final-layer readouts from the same sweep sources | `scripts/build_descriptive_figures.py` |
| `paper/tables.tex` | All four tables in one file, in the manuscript's float order, for the standalone wrapper | `scripts/build_descriptive_tables.py` |
| `paper/table_{main,budget,placement,controls}.tex` | One table each; the manuscript inputs these so every wide float is declared beside the text that discusses it | `scripts/build_descriptive_tables.py` |
| `paper/tables_macros.tex` | `\fittable` definition, input once before any table file | `scripts/build_descriptive_tables.py` |
| `paper/main_tables.tex` | Standalone wrapper for `tables.tex` | LuaLaTeX, run from `paper/` |
| `paper/paper.pdf` | Public manuscript, including the current figures and tables | `bash paper/build.sh` |

All six generated table files come from one run of the builder and are checked
together; `--check` fails if any of them drifts from the stored JSONs.

The source mapping uses full-evaluation loop reruns for BoolQ Mistral and
Granite, separate RuleTaker 10k/4k placement sweeps, and the three BoolQ
`samek` files for label-budget refits. Source comments in the generated table
files identify each row's inputs. Seed SD is not a confidence interval, and in
Figures 1 and 2 it is stated in the caption rather than drawn, because at
0.06--0.71 accuracy points an error bar would be shorter than its own marker.
Evaluation-selected layer maxima are exploratory, not held-out gains; the
placement table reports how many candidates each maximum was taken over, and
`--report` prints the corresponding selection-noise scale.

## Check and Build

From the repository root, check the tables and list all source files using
only Python 3's standard library:

```bash
python3 -B scripts/build_descriptive_tables.py --check --report
```

Regenerate the tables or figures from the stored artifacts:

```bash
python3 -B scripts/build_descriptive_tables.py
python3 -B scripts/build_descriptive_figures.py
```

The figure builder requires Matplotlib and checks that the tables are current
before plotting. `bash paper/build.sh` performs that check, regenerates the
figures, and compiles `paper/paper.pdf` with LuaLaTeX and BibTeX. See
[`README.md`](README.md) for installation. These commands rebuild descriptive
outputs, not the experiments that produced the stored results.

## Historical Artifacts

The older `plot_*.py` scripts and PNGs are historical, not inputs to the
current manuscript. `scripts/build_table1.py`, `scripts/verify_tables.py`,
`scripts/extract_canonical.py`, and `scripts/build_reconciled_table.py` use
older reporting conventions; they do not validate the revised tables.
Some historical plots use hardcoded synthetic arrays or different source
mappings and paired-test statistics.

The older 2k-protocol manuscript, two standalone synthetic renderings, and
duplicate head-to-head alias were removed from the current tree; their
previous versions remain in Git history. The `probe_results.png` composite
and `paper/figures/head_to_head_three_tasks.png` remain historical artifacts.
Their generators write only the retained outputs under `paper/figures/`.

The old results narrative is preserved in root
[`RESULTS_HISTORICAL.md`](RESULTS_HISTORICAL.md), with a current summary in
[`RESULTS.md`](RESULTS.md). The canonical public paper remains
`paper/paper.pdf`.
