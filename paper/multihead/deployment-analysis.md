# Deployment analysis: multi-head residual classification

This note captures the systems and commercial interpretation of the
multi-head single-pass results. Measured findings remain in
`RESULTS.md` and `RESULTS_ROUTING.md`; claims about production latency,
throughput, and economics below are deployment hypotheses unless explicitly
marked as measured.

## Core deployment idea

The useful combination is:

```text
already-deployed frozen causal language model
  -> shared forward to an early residual tap
  -> task router when task identity is not explicit
  -> continue to the selected task's answer tap
  -> small task-specific classification head
```

For the verified 8B adaptive pilot:

```text
L1 router
  -> BoolQ     -> L30 head
  -> RuleTaker -> L24 head
  -> ARC       -> L35 head
```

This combines two separate advantages:

1. **Direct residual classification:** replace autoregressive answer
   generation with a fixed-size classifier readout.
2. **Adaptive continuation:** avoid running every task to the deepest layer
   when its answer quality peaks earlier.

The second is an optimization on top of the first; the larger conceptual
change is replacing generated classification answers with direct residual
readouts.

## What the existing results already establish

### No-autoregressive classification

The canonical single-head artifacts compare residual heads with
autoregressive answer loops:

| Task/model | One-pass residual readout | Zero-shot loop | Few-shot loop |
|---|---:|---:|---:|
| BoolQ 0.6B | 0.746 | 0.631 | 0.645 |
| BoolQ 8B | 0.878 | 0.862 | 0.876 |
| RuleTaker 0.6B | 0.650 | 0.600 | 0.545 |
| RuleTaker 8B | 0.758 | 0.668 | 0.713 |
| ARC 0.6B | 0.487 | 0.495 | 0.596 |
| ARC 8B | 0.910 | 0.904 | 0.920 |

These numbers support a scoped claim: direct residual classification can
match or exceed the autoregressive interface on several closed-set tasks,
while ARC shows the boundary where a few-shot loop can still be stronger,
especially at small scale.

The readout path is:

```text
input context -> one model forward -> residual -> small head -> class
```

The generated-answer path is:

```text
input context -> forward -> generate token -> update KV cache
              -> generate next token -> stop and parse
```

The readout therefore has no generated answer tokens, decode loop, output
KV-cache growth, stop-token handling, or text parsing. This is the main
commercial distinction from a real generation call.

Important protocol distinction: the fair scoring loop in `bench.py` also uses
one context forward to score a fixed answer set, so readout-versus-scoring-loop
is approximately FLOP-equal. The generation savings apply to an actual
autoregressive decode loop, not to the no-generation scoring baseline.

### Adaptive depth

The matched 8B adaptive artifact is:

`results/multihead_adaptive_Qwen__Qwen3-8B_t1000_v400.json`

| Router | Task accuracy | Adaptive answer accuracy | Shared L35 answer accuracy |
|---|---:|---:|---:|
| Gold task ceiling | 1.000 | 0.848 | 0.829 |
| Task-ID linear at L1 | 0.998 | 0.849 | 0.829 |
| Mahalanobis at L1 | 1.000 | 0.848 | 0.829 |

Per-task gold-routed results:

| Task | Adaptive tap | Adaptive | Shared L35 | Gain |
|---|---:|---:|---:|---:|
| BoolQ | L30 | 0.870 | 0.848 | +2.25 points |
| RuleTaker | L24 | 0.760 | 0.725 | +3.50 points |
| ARC | L35 | 0.915 | 0.915 | 0.00 points |

The adaptive path improves mixed accuracy from 0.829 to 0.848 while using
30.7 of 36 transformer blocks on average, or 85.2% of full depth. This is a
14.8% depth/FLOP proxy reduction, not a measured wall-clock latency saving.

## Batching and scheduling

The batching assumption is correct: for efficient continuation, requests
should generally be grouped by selected target depth after the early router.

```text
Stage 1:
  mixed batch -> run to L1 -> task router

Stage 2:
  L24 bucket -> continue to L24
  L30 bucket -> continue to L30
  L35 bucket -> continue to L35
```

This creates a synchronization point after L1. The runtime must balance
compute savings against the cost of small buckets and queueing.

### Runtime options

#### Bucketed continuation

Route at L1, group requests by target depth, and continue each bucket
independently. This is the natural high-throughput design when traffic is
large enough to fill the buckets.

#### Deepest-layer fallback

If buckets are too small, merge them and run the group to L35. This preserves
GPU utilization and latency predictability at the cost of adaptive-depth
savings.

#### Continuous or ragged scheduling

At higher traffic volume, maintain active groups by layer and move requests
between depth buckets. This can improve utilization but requires inference
engine support for hidden-state continuation and layer-wise scheduling.

### What controls the real benefit

The 85.2% depth number does not automatically imply 14.8% lower latency.
Actual serving results depend on:

- traffic proportions across target depths;
- batch size in each depth bucket;
- wait time while forming buckets;
- scatter/gather and synchronization overhead;
- GPU utilization and kernel-launch overhead;
- whether the inference engine can resume from an intermediate hidden state;
- fallback policy for low-volume buckets.

At low traffic or tiny batches, running the entire group to L35 may be faster.
At sufficiently high and stable traffic, bucketed continuation should expose
more of the depth/FLOP reduction.

## Why classification is the strongest target

This approach is cleaner for classification-style prefill workloads than for
open-ended generation:

- one input sequence is processed once;
- no token-by-token output loop is required;
- there is no growing generation KV cache;
- output size is fixed and small;
- the task head emits a deterministic class or score;
- layer-bucket scheduling is simpler than continuous token-generation
  scheduling.

Likely target workloads include moderation, intent classification,
policy decisions, fraud/risk checks, compliance screening, document and
clause classification, agent/tool selection, RAG verification, and
LLM-as-a-judge calls.

The strongest cases are workloads where a large model's knowledge or long
context is useful, but the required output is a closed-form decision. A small
specialist classifier remains cheaper for narrow tasks that do not need the
large model or its context.

## When routing is needed

If the API already specifies the task, there is no need to infer the task:
select the head directly. The router matters for mixed, task-unknown traffic.

Preferred order for deployment:

1. explicit endpoint or schema field;
2. controlled keyword or format routing;
3. validated latent task router for genuinely mixed free-form traffic.

The task router should be closed-set and default-deny. Low-confidence,
off-manifold, or unsupported inputs should escalate to the autoregressive
path or another fallback rather than silently selecting a head.

Mahalanobis is worth keeping as a comparison because it tests a training-free
extension path. On clean closed-set traffic it matches the supervised router,
but the L1 shuffle control shows order fragility:

- 8B L1 identity: 1.000;
- 8B L1 word shuffle: 0.935;
- 8B L1 token shuffle: 0.387;
- 8B L35 token shuffle: 0.939.

Therefore the supervised L1 probe is the practical default, while Mahalanobis
is a useful zero-training baseline for clean traffic and extensibility.

## Positioning and novelty scope

The individual ingredients are not new in isolation:

- shared backbone plus task heads is standard;
- early exit is established;
- task-ID routing exists in continual learning;
- adapter and expert routing are established;
- prototype and geometric routing have precedent.

The useful systems combination is narrower:

> A frozen open-weight causal model serves mixed closed-set classification
> traffic through independently trained residual readout heads; an early
> probe identifies the task, the same forward pass continues to a
> task-specific answer depth, and the output is emitted without
> autoregressive generation.

This should be positioned as a deployment combination and measured systems
result, not as the invention of early exit or task routing individually.

## Required production benchmark

Before making a latency or throughput claim, compare these paths under
realistic arrival patterns and batch sizes:

| Path | Purpose |
|---|---|
| Separate autoregressive task calls | Current generative baseline |
| Shared model to L35 plus task head | No-generation baseline |
| Adaptive routing with bucketed continuation | Proposed path |
| Adaptive routing with small-bucket fallback | Practical scheduler policy |

Measure:

- end-to-end latency and tail latency;
- throughput at fixed quality;
- GPU utilization;
- queueing time after L1 routing;
- peak memory and model-copy count;
- generated-token count and KV-cache allocation;
- accuracy and abstain/escalation rate.

Until this benchmark exists, report 85.2% as a depth/FLOP proxy and describe
no-autoregressive output as the established architectural difference, without
claiming a specific combined wall-clock speedup.

## Source files

- `RESULTS.md` — canonical single-head readout versus loop results.
- `RESULTS_ROUTING.md` — routing, adaptive-depth, and Mahalanobis results.
- `results/multihead_adaptive_Qwen__Qwen3-8B_t1000_v400.json` — adaptive 8B
  artifact.
- `paper/routing-latent.md` — deployment routing and fallback notes.
- `implications.md` — broader commercial interpretation.
- `NOTES.md` — conceptual residual-versus-decoding analysis.
