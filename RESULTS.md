# Results: Revised-Paper Summary

The [main paper PDF](paper/paper.pdf) reports a retrospective comparison of
supervised classifiers on frozen final-token residual states and an implemented
zero-/few-shot answer-token scorer. The central finding is a regime map:
readouts are strongest relative to scoring on RuleTaker and BoolQ, while
scoring has its clearest advantage on ARC when the residual readout is weak.
Both methods use one prompt forward pass, not iterative generation; their
supervision and inputs differ. No new inference or classifier fitting was
performed for the revision.

## Current Results

The main readout statistic is the fixed final-layer `last.mlp` mean over four
training seeds, with seed SD (`ddof=0`), not a confidence interval or ensemble
accuracy. `loop.zero` and requested-eight-shot `loop.8` are reported separately;
differences use `loop.8`, not the best available loop arm.

| Task | Training / evaluation / loop rows | Readout mean higher than `loop.8` |
|---|---|---|
| BoolQ | 9427 / 3270 / 3270 | 2 of 6 models |
| RuleTaker | 10000 / 4000 / 4000 | 5 of 6 models |
| ARC-Challenge | 1000 / 1165 / 1165 | 0 of 6 models |

These are numerical comparisons supporting a regime pattern, not significance
or equivalence findings.
BoolQ uses its validation split; RuleTaker uses a sampled official test subset;
ARC uses four-choice test items. Readout token caps are 384; scorer caps are
8192 for Qwen BoolQ and 2048 otherwise. Requested shots are not measured
retained demonstration counts after truncation.

The scorer sums log-probabilities over alternative answer-token IDs, ranking
their product rather than marginalizing answer probability. Corrected scorer
performance is unknown. The results do not establish supervision-independent
superiority, a reasoning mechanism, or latency/early-exit benefits. Layer
maxima are evaluation-selected and optimistic, not held-out gains; controls
are descriptive, not leakage audits or causal evidence.

## Samek Caveats

The stored BoolQ `samek` artifacts for [0.6B](results/boolq_samek_06b.json),
[4B](results/boolq_samek_4b.json), and [8B](results/boolq_samek_8b.json) include
first-`n` and class-balanced refits at 8, 16, 32, and 64 labels. In all 12
model/budget settings, both refit means are below the corresponding
requested-shot scorer. Thus the full-training readout advantage is not
established as independent of supervision.

Equal declared counts (`n=k`) do not certify identical labeled examples or
retained demonstrations. Exact balanced subset identities and the cache-refit
generator have not been recovered. These are existing measurements, not a
fully reproduced identical-example experiment; no significance tests are used.

## Sources and History

- [Explicit source mappings](scripts/build_descriptive_tables.py): `CELLS` maps headline readouts, loop overrides, and separate placement sweeps; `BUDGET` maps the `samek` sources.
- [Generated paper tables](paper/tables.tex): source filenames are included in comments. BoolQ Mistral/Granite loops use full-evaluation reruns, not the sweeps' 405-row loops. RuleTaker headline and sweep files both use 10k/4k but remain distinct sources.
- [Revision notes](PAPER_REVISION.md): methodology changes and paper/figure build instructions.
- [Historical results log](RESULTS_HISTORICAL.md): preserved incidents, provenance, and superseded conclusions, not current evidence summaries.
- [Synthetic exploration](RESULTS_SYNTHETIC.md): separate historical work, not evidence for the revised paper.

Validate the current tables against the stored JSON sources without writing:

```sh
python3 -B scripts/build_descriptive_tables.py --check --report
```

The older table builders, validators, and plotters carry historical-tooling
notices; they are not the revised-paper reproduction path. In particular,
`scripts/rerun_canonical.sh` is historical incident repair and can overwrite
stored source files, including loop reruns used by the revised paper.
