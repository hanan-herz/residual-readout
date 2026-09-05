# residual-readout-bench

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

**Supervised Residual Readouts versus Few-Shot Answer-Token Scoring**

A descriptive comparison of supervised classifiers on frozen final-token
residual states with zero-/few-shot answer-token scoring across six models
on BoolQ, RuleTaker, and ARC-Challenge. The methods use different amounts of
task supervision and different prompt lengths. The stored scorer sums
log-probabilities of alternative answer tokens, rather than marginalizing
their probability mass; corrected-baseline performance is unmeasured.

Against the fixed requested-eight-shot arm, final-layer readout means are
higher in 5/6 RuleTaker cases, 2/6 BoolQ cases, and 0/6 ARC cases. In the
existing BoolQ 8/16/32/64-label refits, both refit protocols are below the
corresponding requested-shot scorer in all 12 model/budget settings. These
are descriptive comparisons, not significance, equivalence, or
supervision-matched superiority findings.

| Resource | Location |
|---|---|
| Public paper | [`paper/paper.pdf`](paper/paper.pdf), with LaTeX sources in [`paper/`](paper/) |
| Repository | https://github.com/hanan-herz/residual-readout-bench |
| Author | Hanan Herzog, [ORCID](https://orcid.org/0009-0009-0464-7112), hanan.herzog@gmail.com |
| Citation | [`CITATION.cff`](CITATION.cff) |
| Figures and tables | [`FIGURES.md`](FIGURES.md) |

## Quickstart (no GPU)

Run from the repository root. Checking the current tables requires only
Python 3's standard library, with no package installation or network access:

```bash
python3 -B scripts/build_descriptive_tables.py --check
```

Add `--report` to list the 32 explicitly mapped JSON sources and unrounded
descriptive values. This checks `paper/tables.tex` against stored artifacts;
it does not rerun experiments or validate their original execution.

### Paper build

Install Matplotlib in a virtual environment and a TeX distribution providing
LuaLaTeX, BibTeX, and the packages used by the manuscript:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install matplotlib
bash paper/build.sh
```

The build checks the tables, regenerates the three descriptive PDF figures,
and compiles the public `paper/paper.pdf`. It does not load models or fit
classifiers. `paper/paper-revised.pdf` is an ignored local revision copy,
not a second public paper.

### Experiments (separate setup)

The experiment harness needs additional dependencies, model weights,
datasets, and suitable compute. In an activated experiment environment:

```bash
python3 -m pip install -r requirements.txt
python3 -B bench.py --local --help
```

`bench.py` currently imports Modal unconditionally, including for local
runs and help, so `modal` is included in the requirements. Put `--local`
first; local model selection uses `--size` (for example, `0.6B`), and tasks
use `--task boolq`, `--task ruletaker`, or `--task arc`. Help does not run an
experiment. Cloud runs additionally require Modal setup; image dependencies
are defined in `bench.py`.

The historical Modal application identifier `vector-is-the-answer-bench`
is unchanged; it is not the GitHub repository name. Package versions are
not a lockfile for the original runs.

These instructions do not guarantee exact reproduction of stored runs.
In particular, the BoolQ cache-refit generator and exact balanced subset
identities have not been recovered. See the paper for protocol limitations.

## Layout and History

| Path | Role |
|---|---|
| `paper/` | Current manuscript, generated tables, and figures |
| `scripts/build_descriptive_*.py` | Current table and figure builders |
| `bench.py`, `common.py`, `paired_test.py` | Experiment harness and historical paired-test tooling |
| `tasks/` | BoolQ, RuleTaker, and ARC loaders |
| `results/` | Stored experiment JSON artifacts |
| `cloud/` | Modal serving helpers |

[`RESULTS.md`](RESULTS.md) summarizes the current results; root
[`RESULTS_HISTORICAL.md`](RESULTS_HISTORICAL.md) preserves the old narrative.
Older plotting and table tools are historical, not validators for the
revised manuscript; see `FIGURES.md` for their status.

## License and Citation

Code, data, and manuscript: CC BY 4.0 (attribution required).
See [`LICENSE`](LICENSE).

```bibtex
@misc{herzog2026residualreadout,
  title={Supervised Residual Readouts versus Few-Shot Answer-Token Scoring},
  author={Herzog, Hanan},
  year={2026},
  howpublished={GitHub},
  url={https://github.com/hanan-herz/residual-readout-bench}
}
```
