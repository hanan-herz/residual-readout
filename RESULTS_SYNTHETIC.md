# Synthetic-task results — Latent-Representation Hypothesis

> **Separate historical exploration, not revised-paper evidence.** The original
> measurements and interpretations below are retained for provenance, not as
> current conclusions. See the [current results summary](RESULTS.md), including
> its [samek caveats](RESULTS.md#samek-caveats), and the
> [revised paper](paper/paper.pdf) for the supported descriptive comparison.

Mechanism experiments on synthetic tasks (multi-hop transitivity, arithmetic,
semantic roles, closed-set extraction, latent-loop drift). Small-n (700 train /
175 val unless noted). For the externally-validated headline see `RESULTS.md`
(BoolQ, up to 9427/3270). Interpretation notes are private and not in this repo.

Code: `probe_main.py` (transitivity), `matched.py` / `boundary.py` /
`add_verify.py` / `semantic.py` (Exp 5), `extract.py` (Exp 6), `drift.py`
(Exp 7). Plot: `paper/figures/probe_results.png`. Models `Qwen3-{0.6B,4B,8B}` fp16 on Apple M5.

## Task — multi-hop transitivity with shuffled facts + distractors

Froze the model; captured the last-token residual at several layers; **never
decoded an answer**. A linear logistic probe predicts the transitive relation.
Baselines on identical inputs: LM head/decoder (greedy yes/no, zero-shot and
few-shot), a surface position oracle, a label-shuffle null, and a
random-Gaussian-projection probe (selectivity control).

## Numbers (val accuracy; 700 train / 175 val)

| Model | Latent probe (last layer) | MLP readout | Decoder zero-shot | Decoder few-shot | Surface oracle | Label-shuffle null |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | **0.931** | **0.909** | 0.463 | 0.491 | 0.547 | 0.480 |
| Qwen3-4B   | **0.966** | **0.954** | 0.474 | 0.846 | 0.531 | 0.440 |
| Qwen3-8B   | **0.971** | **0.966** | 0.474 | 0.680 | 0.531 | 0.526 |

Random-projection probe (selectivity; mean over 5 runs):

| Model | dim 16 | dim 64 | dim 256 |
|---|---|---|---|
| Qwen3-0.6B | 0.682 | 0.871 | 0.931 |
| Qwen3-4B   | 0.926 | 0.930 | 0.952 |
| Qwen3-8B   | 0.896 | 0.928 | 0.967 |

**Reading.** The relation is linearly decodable from the frozen residual
(0.93–0.97, no generated token). The zero-shot decoder sits at chance — but that
is a conditioning artifact: a 2-demo few-shot budget surfaces the inference
(0.846 at 4B). So the loop *can* produce the answer — it just never exceeds a
single-pass readout of the same state (Exp 4). Surface/shuffle controls are at
chance; randproj retains 0.90+ at dim 64–256 (genuinely linear structure, not a
coordinate overfit). Probe accuracy rises with depth (earlier layers do more at
larger scale).

## Experiment 4 — capacity ceiling: is the loop doing irreducible work?

Does the loop compute at depth, or was the linear probe the bottleneck? Feed the
*same* last-layer vector to a nonlinear (MLP) one-pass readout.

| Model | MLP (single pass) | Decoder (few-shot) | MLP − decoder |
|---|---|---|---|
| Qwen3-0.6B | 0.909 | 0.491 | **+0.42** |
| Qwen3-4B   | 0.954 | 0.846 | **+0.11** |
| Qwen3-8B   | 0.966 | 0.680 | **+0.29** |

Per-hop the MLP holds ≈0.93–1.00 across hops at 4B/8B, at-or-above the decoder
at essentially every cell (including the deep cells where the decoder was
"catching up").

**Reading.** A nonlinear single-pass readout matches or beats the loop at all
serial depths and scales — the loop is **not** doing irreducible serial
computation here; it re-derives, less reliably, what the latent already holds.

### Serial-depth boundary (per-hop, 700/175, ~26/hop-cell)

| Model | hops | n | latent probe | decoder (few-shot) | shuffle |
|---|------|---|------------|---------|---------|
| 4B | 2 | 27 | 1.000 | 0.963 | 0.444 |
| 4B | 3 | 28 | 0.964 | 0.857 | 0.536 |
| 4B | 4 | 30 | 0.967 | 0.700 | 0.500 |
| 4B | 5 | 29 | 0.862 | 0.759 | 0.310 |
| 4B | 6 | 32 | 1.000 | 0.844 | 0.344 |
| 4B | 7 | 29 | 1.000 | 0.966 | 0.517 |
| 8B | 2 | 27 | 1.000 | 0.889 | 0.556 |
| 8B | 3 | 28 | 0.964 | 0.821 | 0.571 |
| 8B | 4 | 30 | 0.933 | 0.633 | 0.533 |
| 8B | 5 | 29 | 0.966 | 0.552 | 0.586 |
| 8B | 6 | 32 | 0.969 | 0.531 | 0.531 |
| 8B | 7 | 29 | 1.000 | 0.690 | 0.379 |
| 0.6B | 2 | 27 | 1.000 | 0.481 | 0.444 |
| 0.6B | 3 | 28 | 1.000 | 0.643 | 0.571 |
| 0.6B | 4 | 30 | 0.900 | 0.467 | 0.567 |
| 0.6B | 5 | 29 | 0.828 | 0.379 | 0.448 |
| 0.6B | 6 | 32 | 0.875 | 0.500 | 0.375 |
| 0.6B | 7 | 29 | 1.000 | 0.483 | 0.483 |

**Reading.** The probe is representation-resident at every depth (0.83–1.00).
The 4B decoder rises with depth (0.70→0.97) and 8B *declines* (0.89→0.53) — but
single-pass MLPs hold 0.91–1.00 at those same depths, so decoder
depth-sensitivity is a brittle readout phenomenon, not a latent property. The
decisive boundary is readout capacity, not decoder depth.

## Experiment 5 — matched supervision + deconfounds (`matched.py`, `boundary.py`, `add_verify.py`)

Closes Exps 1–4's two confounds (probe ~700 labels vs decoder 2 demos; probe
last-token vs loop full-context), then stress-tests for shortcuts. Seed-swept
MLP (`mean ± std`, 5 seeds).

### 5a — full-context, budget-matched readouts

| Qwen3 | last.linear | last.mlp | ctx.linear | ctx.mlp | mlp-shuffle null |
|---|---|---|---|---|---|
| 0.6B | 0.937 | 0.905 ± 0.008 | 0.806 | 0.677 ± 0.005 | 0.493 |
| 4B   | 0.968 | 0.960 ± 0.000 | 0.904 | 0.824 ± 0.011 | 0.523 |

**Reading.** Giving a one-pass readout the *whole context* (mean-pooled) hurts
(ctx.mlp 0.677/0.824 < last.mlp 0.905/0.960) — the last token already holds the
relation; the loop's context-re-reading adds nothing at matched supervision.

### 5b — format deconfound (reword `taller` → `outranks`, identical labels)

| Qwen3 | linear in-distr | linear → reworded | mlp in-distr | mlp → reworded | surface oracle |
|---|---|---|---|---|---|
| 0.6B | 0.937 | **0.857** | 0.905 ± 0.008 | 0.726 ± 0.019 | 0.52 |
| 4B   | 0.968 | **0.872** | 0.960 ± 0.000 | 0.896 ± 0.000 | 0.50 |

**Reading.** The readout transfers to a reworded relation (0.86–0.90 ≫ 0.50
chance) — it reads the semantic ranking, not the string `taller` (a surface
component exists, but semantic dominates).

### 5c — MLP is the honest metric (seed stability)

Single MLP seeds are pathological at small n (seed0 = 0.53 at n=300 vs 0.95
otherwise). At per-depth 700 it is stable: **0.909 ± 0.008 over 8 seeds (min
0.891, max 0.920)** on 0.6B. **Caveat: read every MLP number as a seed-swept
distribution (±0.01–0.02), never a single seed below ~500 examples.**

### 5d — arithmetic deconfound (`add_verify.py`, "Is A+B==C?", 6–15-digit)

The headline 0.869 linear probe **collapses within each digit-length bin**
(14: 0.453; 15: 0.562) — a length/magnitude shortcut, not addition. The **MLP
holds across both bins (0.837 / 0.876, overall 0.857)** and survives randproj
(dim 16/64/256: 0.75/0.84/0.86).

**Reading / correction.** Only the *nonlinear* readout genuinely carries the
sum per-length; the linear number was a length confound — a worked example of
why the selective controls are necessary.

### 5e — boundary hunt (`boundary.py`, iterated modular maps, K=2–6, parity label)

| size | linear | mlp | shuffle-null | surface-oracle | loop (few-shot) |
|---|---|---|---|---|---|
| 0.6B | 0.60 | 0.46 | 0.42 | 0.60 | 0.53 |
| 4B   | 0.52 | 0.55 | 0.53 | 0.60 | (as 0.6B) |

**Reading — honest negative.** The loop also sits at chance (0.53): this
compositional task is not in the frozen base models' weights, so neither channel
holds it and the game doesn't discriminate latent-vs-loop. On frozen base models
no "loop wins" cell is exhibitable (the loop's reasoning is parasitic on latent
knowledge); demonstrating genuine loop computation requires a CoT/RL-trained
model, which this thesis excludes.

### 5f — semantic role decomposition (`semantic.py`, "The <agent> <action>s the <patient>")

| role | 0.6B | 4B | chance | surface-pos | shuffle |
|---|---|---|---|---|---|
| agent | 1.000 | 1.000 | 0.083–0.15 | ≈chance | ≈chance |
| patient | 1.000 | 1.000 | 0.15 | ≈chance | ≈chance |
| action | 1.000 | 1.000 | 0.10 | ≈chance | ≈chance |

Agent survives randproj (dim 16/64: 0.997+/1.000).

**Reading.** Role decomposition is cleanly in the latent — including roles whose
words are *earlier* than the final token. **But** role *binding* under argument
inversion ("The {patient} {action}s the {agent}.") collapses to ≈0.00 (chance
0.083): the readout is **order-locked, not an order-invariant compositional
binder**. The transitivity probes missed this because their labels are symmetric
under the two orderings — so the latent is a semantic substrate in the
*decodable-structure* sense, with a real boundary at generative cross-ordering
composition.

## Experiment 6 — closed-set extraction (`extract.py`)

Product-facing claim: structured extraction needn't require token generation
when the output target is a *defined set*. A) relation classification (4 closed
relations); B) object slot-fill (object truncated away, forcing recovery of the
(subject, relation)→object binding from the latent).

One-pass MLP (reword split) vs baselines (0.6B n=700/200; 4B n=300/150):

| Task | Model | one-pass MLP (reword ±sd) | surface-BOW (reword) | loop (few-shot) | shuffle |
|---|---|---|---|---|---|
| A relation | 0.6B | **0.995 ± 0.010** | 0.850 | 0.433* | 0.205 |
| A relation | 4B | **0.933 ± 0.017** | 0.860 | 0.733 | 0.380 |
| B slot-fill | 0.6B | 0.568 ± 0.021 | 0.560 | 0.067* | 0.085 |
| B slot-fill | 4B | **0.513 ± 0.062** | 0.380 | **0.027** | 0.100 |

*low-n / prompting-decode artifact. Chance: A 0.25, B 0.125. Randproj (A, best
layer): 0.6B 1.000 all dims; 4B 0.78/0.93/0.997 at dim 16/64/256.

**Reading.** The one-pass readout matches/beats the loop on every axis
(relation 0.93–0.995; slot-fill 4B 0.51 vs loop 0.03) and is reword-invariant.
Caveats: 4B is small-n; the loop's near-zero on slot-fill is partly a
prompting/decode artifact (greedy few-shot decode under-states the loop);
closed-set only — open-world extraction is out of scope.

## Experiment 7 — latent-loop off-manifold drift (`drift.py`)

If we loop the latent (COCONUT-style refeed of the last-layer residual as the
next input embedding), does it drift off the manifold a pass-0 readout expects?

| metric /‖h0‖ (0.6B; 4B same) | latent-loop start→end | natural trajectory | reading |
|---|---|---|---|
| hidden norm | 1.01 → 1.30 (bounded) | ~1.0 (bounded) | **no blow-up** (step distance decays) |
| off-manifold dist (PCA-64 recon err) | 0 → **1.09 → 1.40** | bounded ≤ 0.87 | **drifts off** the real subspace |
| cos(h_k, h_0) | 1.0 → **~0.0 by k≈3** | holds ~0.5 | **fully decorrelates** from seed |

PCA(64) of 200 real residuals captures 100% of variance (near-low-rank manifold).

**Reading.** Not a catastrophic divergence (norm bounded) but real off-manifold
drift: a readout trained on pass-0 residuals will not transfer to a looped state
— the mechanical reason latent looping must retrain. Latent-loop serving is not
a frozen-model option; the single-pass readout carries no drift risk. `drift.py`
is parked.

## Global caveats (Exps 1–7)

- Single synthetic task family (transitivity + variations); not evidence for
  reasoning behavior broadly. See `RESULTS.md` for the public-benchmark test.
- Probes are linear / one-hidden-layer on the *last* layer; the
  loop-buys-computation hypothesis is not exhausted (deeper probes / other tasks
  could still find a loop-win depth).
- The few-shot demo is itself transitive, so it teaches the reasoning pattern;
  the MLP result does not depend on it (no demos).
- MLP point estimates carry ±0.01–0.02 (5c); hop-level cells are n≈29.
