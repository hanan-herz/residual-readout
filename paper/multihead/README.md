# Multi-head paper artifacts

This directory contains figures and plotting code for the multi-head
single-pass routing pilot. These artifacts are separate from the canonical
single-head paper figures under `paper/figures/`.

## Contents

- `figures/multihead_scale_comparison.png` — clean routing and answer
  accuracy for Qwen3-0.6B vs Qwen3-8B.
- `figures/multihead_shuffle_layers.png` — shuffle robustness for the
  supervised task-ID probe and Mahalanobis router.
- `plot_multihead.py` — deterministic figure generator.

Regenerate both figures from the repository root with:

```bash
python paper/multihead/plot_multihead.py
```

The experimental writeup is `RESULTS_ROUTING.md`; the multi-head JSON
artifacts remain under `results/` with `multihead_` or `shuffle_control_`
names.
