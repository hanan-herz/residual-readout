# Multi-head single-pass routing — pilot results

> Status: **branch pilot** (`exp/multihead-single-pass`), Qwen3-0.6B and Qwen3-8B.
> Canonical paper numbers remain `paper/tables.tex`; nothing here feeds them.
> All artifacts: `results/multihead_*.json`, `results/shuffle_control_*.json`.
> Code: `multihead_route.py`, `embedding_control.py`, `shuffle_control.py`,
> `modal_multihead.py`.

## Question

After one frozen forward pass, can we *select* which closed-set answer head
(BoolQ / RuleTaker / ARC) to serve — without knowing the task in advance?
This is the serving-composition question on top of the paper's single-head
claim: given a registry of heads over the same residual space, how cheaply
can we route to the right one?

## Routers compared

| router | signal | deployable? |
|---|---|---|
| `gold_task_ceiling` | gold task label | no — analytical upper bound only |
| `naive_max_conf` | highest *answer* confidence across heads | yes, but fails |
| `mahalanobis` | nearest per-task train-residual manifold | yes, unsupervised |
| `task_id_linear` / `task_id_mlp` | trained probe: which task is this | yes, needs task labels |

## Findings

### 1. Task identity separates at layer 1 — and it is computation, not embedding

At the true embedding (layer 0, `embed_tokens` output), task routing is
exactly chance (0.333 for every router). At layer 1 it is ~perfect:

| layer | mahalanobis task-acc | task_id_linear | naive_max_conf | ceiling answer acc |
|---|---|---|---|---|
| 0 (embed) | 0.333 | 0.333 | 0.333 | 0.478 |
| 1 | 1.000 | 1.000 | 0.047 | 0.458 |
| 5 | 1.000 | 0.998 | 0.068 | 0.463 |
| 9 | 1.000 | 1.000 | 0.140 | 0.487 |
| 13 | 1.000 | 1.000 | 0.260 | 0.512 |
| 18 | 1.000 | 0.997 | 0.105 | 0.548 |
| 23 | 0.995 | 1.000 | 0.307 | 0.595 |
| 27 | 0.995 | 0.997 | 0.227 | 0.567 |

(t400/v200, uniform refit-on-pilot-train heads, `*_refit.json`)

The embedding-at-chance control means the first transformer block **creates**
the separation; it is not a tokenizer/prompt-template leak.

### 2. Answer quality rises with depth and plateaus at the top

Ceiling answer acc climbs 0.458 → 0.567 over layers 1→27 (pilot). The small
pilot showed an apparent dip at L27 (0.595@23 → 0.567@27); the bigger run
(t1000/v400, n=1200) resolves it as noise: **0.618 vs 0.620**. The honest
claim is *rises then plateaus*, not "final layer is worse".

### 3. Routing is free; the ceiling is the heads

At every depth, `mahalanobis` and `task_id_*` answer accuracy ≈
`gold_task_ceiling` (e.g. L27: 0.605 vs 0.607). Perfect routing would gain
~0.2pt — the bottleneck is answer-head quality, not selection.

### 4. Naive max-confidence routing is worse than chance

`naive_max_conf` task-acc 0.05–0.31 across depths (chance 0.333). Heads are
overconfident out-of-distribution (mean conf ~0.9–0.99 while wrong). "Pick
the most confident head" is not a router.

### 5. The separating signal is content, not format (shuffle control)

Layer-1 residuals re-extracted on shuffled prompts (t400/v200):

| condition | task_id_linear | mahalanobis |
|---|---|---|
| identity | 1.000 | 1.000 |
| word_shuffle | 0.908 | 0.465 |
| token_shuffle | 0.912 | 0.450 |

A trained probe separates tasks at ~0.91 even from a bag of tokens — the
signal is order-free **lexical content**, not sequential or template
structure. Mahalanobis collapses: its manifold geometry is built largely
from sequential structure. **Implication for deployment: route with the
trained task-ID probe; Mahalanobis is fine on real inputs but its perfect
score measures a more brittle signal.**

### 6. Per-task depth optima do not coincide (full-val layer sweeps)

From `results/*layersweep_06b.json` (real heads, full val):

| task | argmax layer | acc | at L18 | at L23 | at L26/27 |
|---|---|---|---|---|---|
| BoolQ | **18** | 0.761 | 0.761 | 0.747 | 0.746 (L27) |
| RuleTaker | **18** | 0.688 | 0.688 | 0.664 | 0.650 (L27) |
| ARC | **26** | 0.500 | 0.351 | — | 0.500 (L26) |

BoolQ/RuleTaker peak at ~2/3 depth and *decay* deeper; ARC wants the deepest
tap. Forcing one shared tap costs up to 15 points (ARC at L18) — per-task
depth is required, not optional.

## Serving architecture this supports

```
forward pass ── L1 ──► task-ID probe routes (~free, content-robust)
     BoolQ     ──► continue to L18 ──► head answers
     RuleTaker ──► continue to L18 ──► head answers
     ARC       ──► continue to L26 ──► head answers
```

One pass, truncated per task at its own optimum (avg ≈ 74% of full depth),
no KV-cache, router ~0 cost. Heads are portable artifacts scoped to
(model, layer); the closed set is extendable at zero router training cost
— a new task means training its answer head and caching a few hundred
residuals for the routing distribution.

**Training independence.** The router and the answer heads are trained
fully independently — no joint training anywhere:

- The task-ID probe is a single closed-set classifier
  `(L1 residual → task label)`; it never sees the answer heads.
- Each answer head is trained per-task, per-layer on its own residuals
  and answer labels, orthogonal to every other head and to the probe.

The only coupling is at serving time, where the probe's output selects
the head. This is what makes heads portable: a new task's head is trained
standalone and dropped in. The one asymmetry between routers is the cost
of extending the closed set with task N+1 — the supervised probe must be
retrained on N+1 classes, while Mahalanobis needs nothing beyond the new
task's L1 residual bank. Mahalanobis pays for that convenience with the
order-fragility shown in section 7: at L1 it misroutes order-scrambled or
off-manifold inputs, so the supervised probe is the practical L1 router
despite the retrain-on-extension cost.

### 7. Cross-scale replication: Qwen3-8B (Modal A100-80GB)

Pilot at L35, t1000/v400 (`modal_multihead.py`, uniform refit heads):

| router | 8B | 0.6B (L27 pilot) |
|---|---|---|
| gold_task_ceiling | 1.000 / **0.830** | 1.000 / 0.607 |
| task_id_linear | **1.000** / 0.830 | 0.997 / 0.605 |
| mahalanobis | **0.997** / 0.830 | 0.995 / 0.605 |
| naive_max_conf | 0.577 / 0.780 | 0.142 / 0.118 |

Routing ≈ ceiling at 8B too; task-ID perfect; naive max-conf still fails
(better calibrated than 0.6B but far from usable). Embedding control at
8B also replicates: routing at layer 0 is chance (task-ID 0.340,
mahalanobis 0.340, ceiling ans 0.465) — separation is created by the
first transformer block at both scales.

8B shuffle control. Two layers measured to test whether the
order-fragility of the unsupervised router is a layer effect:

| layer | condition | linear | maha | (0.6B L1 linear / maha) |
|---|---|---|---|---|
| L1 | identity | 0.997 | 1.000 | 1.000 / 1.000 |
| L1 | word_shuffle | 0.997 | 0.935 | 0.908 / 0.465 |
| L1 | token_shuffle | 0.970 | 0.387 | 0.912 / 0.450 |
| L35 | identity | 1.000 | 0.997 | — |
| L35 | word_shuffle | 0.999 | 0.884 | — |
| L35 | token_shuffle | 0.998 | 0.939 | — |

The supervised linear probe is nearly fully order-invariant at 8B
(0.997→0.997→0.970 at L1; 1.000→0.999→0.998 at L35) — content-not-format
strengthens with scale.

Mahalanobis order-fragility is a **layer** effect, not a scale effect. At
L1 the unsupervised router collapses under full token-shuffle (0.387,
near chance 0.333) at 8B just as it did at 0.6B (0.450): the early-layer
manifold encodes token order/position, so a bag-of-shuffled-tokens input
lands off-manifold. At L35 the same router is order-robust (0.939) —
deeper residual geometry becomes order-invariant. Practical consequence:
an unsupervised L1 router works for clean inputs but cannot be trusted to
route arbitrary/OOD text; if order-robust routing is required, route from
a deeper tap (or use the supervised probe, which is order-robust at any
layer).

### 8. Adaptive serving path at 8B

We evaluated the proposed production path directly on a matched mixed-traffic
pilot: 1,000 training and 400 validation examples per task, 1,200 validation
examples total. The L1 task-ID router selects the answer tap independently for
each task, using the optima from the 8B single-head sweeps: BoolQ L30,
RuleTaker L24, and ARC L35. The comparison is against running every task to
L35. Heads were refit on the same matched pilot residuals for this evaluation.

| router | task acc | adaptive taps answer acc | shared L35 answer acc |
|---|---:|---:|---:|
| gold task ceiling | 1.000 | 0.848 | 0.829 |
| task-ID linear (L1) | 0.998 | **0.849** | 0.829 |
| Mahalanobis (L1) | 1.000 | 0.848 | 0.829 |

Per-task answer accuracy under gold routing:

| task | adaptive tap | adaptive | shared L35 | gain |
|---|---:|---:|---:|---:|
| BoolQ | L30 | 0.870 | 0.848 | **+2.25pt** |
| RuleTaker | L24 | 0.760 | 0.725 | **+3.50pt** |
| ARC | L35 | 0.915 | 0.915 | +0.00pt |

The adaptive path improves mixed-task answer accuracy by **1.9 points**
(0.829→0.848) while stopping at an average of 30.7 of 36 transformer
blocks (**85.2% of full depth**, a 14.8% depth/FLOP proxy reduction). The
accuracy gain is broad: earlier task-specific taps improve BoolQ and
RuleTaker, while ARC appropriately runs to the final layer. This is a
measured depth proxy, not a wall-clock latency benchmark; actual savings
also depend on batching, scheduling, and implementation overhead.

## Caveats

- The routing pilot uses Qwen3-0.6B and Qwen3-8B; its matched controls use
  400 train / 200 val per task, while the adaptive 8B path uses 1000 / 400.
- Task-ID router is closed-set: no abstain. Open-world inputs silently
  route to the nearest task — see `gate_ar.py` for the coveredness gate.
- Cross-scale: pilot-level findings, embedding control, depth optima, and
  shuffle controls are now replicated at 8B (sections 7–8).
- 8B per-task depth optima from full sweeps: BoolQ L30, RuleTaker L24,
  ARC L35 — the same "optima do not coincide" pattern as 0.6B.
- BoolQ/RuleTaker L18 peaks vs L27 are ~1–4pt at full val — real-looking
  and consistent across two tasks, but modest.
- `multihead_route.py` confusion plumbing is hardcoded to 3 tasks
  (generalize for N heads).

## Next steps

- Cross-model check (4B, then DSV4) of: separation at L1, content-not-format,
  per-task depth optima split.
- N-head generalization (TASKS list, confusion matrix sizing).
- Big-N at more layers if the depth-plateau claim needs hardening.
- Measure wall-clock serving latency and batching overhead for the adaptive
  path; the current 14.8% figure is a depth/FLOP proxy only.
- Paper: multi-head result is ready as a routing/serving appendix or note;
  canonical single-head paper tables remain unchanged.
