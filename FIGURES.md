# Figures and Tables

The current manuscript uses three descriptive vector PDFs and four tables.
Both builders use the explicit `CELLS` and `BUDGET` mappings in
`scripts/build_descriptive_tables.py`: 32 unique stored JSON sources under
`results/`, not a directory-wide glob. No model inference, classifier
fitting, or significance testing is performed by these builders.

## Current Outputs

| Output | Content | Builder |
|---|---|---|
| `paper/figures/descriptive_main.pdf` | Final-layer readout means and seed SDs, with separate zero-shot and requested-eight-shot scorer arms | `scripts/build_descriptive_figures.py` |
| `paper/figures/descriptive_budget.pdf` | BoolQ first-n and balanced refits, requested-shot scores, and full-training reference | `scripts/build_descriptive_figures.py` |
| `paper/figures/descriptive_layers.pdf` | Recorded MLP layer sweeps and final-layer readouts from the same sweep sources | `scripts/build_descriptive_figures.py` |
| `paper/tables.tex` | Main comparison, placement sweeps, controls, and BoolQ label budgets | `scripts/build_descriptive_tables.py` |
| `paper/main_tables.tex` | Standalone wrapper for `tables.tex` | LuaLaTeX, run from `paper/` |
| `paper/paper.pdf` | Public manuscript, including the current figures and tables | `bash paper/build.sh` |

The source mapping uses full-evaluation loop reruns for BoolQ Mistral and
Granite, separate RuleTaker 10k/4k placement sweeps, and the three BoolQ
`samek` files for label-budget refits. Source comments in `paper/tables.tex`
identify each row's inputs. Seed SD is not a confidence interval;
evaluation-selected layer maxima are exploratory, not held-out gains.

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
`paper/paper.pdf`; `paper/paper-revised.pdf` is an ignored local copy.
