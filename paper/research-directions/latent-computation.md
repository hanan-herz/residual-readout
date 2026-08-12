# New directions: frozen LLMs as latent computational substrates

**Status:** research memo, August 2026

## Core shift

The current project treats a frozen causal language model as a feature extractor
for task-specific classification heads. A broader direction is to treat it as
the language-processing stage of a small latent computer:

```text
text
  -> frozen causal LLM to a selected layer
  -> token-level residual memory
  -> small trainable module that computes in continuous space
  -> task-native output
```

The output need not be a class label or generated text. It could be a scalar,
a ranking, a set of evidence spans, a proof graph, a constraint assignment, an
action, or a plan. Intermediate computation would occur in continuous states,
without repeatedly projecting through the vocabulary and generating reasoning
tokens.

The most promising version is not simply a larger classification head. It is a
weight-shared module that repeatedly attends to the frozen token
representations and refines one or more latent solution states before emitting
a structured answer.

## Why the current results motivate this

The existing experiments provide several useful premises:

- Frozen causal-model residuals contain substantial task-relevant information.
- Useful information is available before the model's final layer.
- Different tasks prefer different answer depths.
- Task identity becomes nearly perfectly readable immediately after the first
  Transformer layer, permitting early selection among independently trained
  modules.
- On Qwen3-8B, the verified adaptive multi-head path improved mixed answer
  accuracy from `0.829` to `0.849` while executing `30.7/36 = 85.2%` average
  model depth.
- Direct readouts avoid answer-string priors, tokenization effects, and prompt
  interface failures that can materially affect autoregressive scoring.
- Independently trained heads can share one frozen model without joint
  training.

These findings support the idea that a deployed language model can serve as a
shared representational substrate for modular downstream computation.

They do **not** yet establish that:

- a residual state can be fed back into the frozen model usefully;
- repeated latent updates perform additional reasoning rather than merely add
  head capacity;
- latent computation can match generated chain-of-thought on difficult tasks;
- a pooled final-token vector retains enough information for structured
  outputs;
- the depth proxy measured so far translates directly into wall-clock or cost
  savings.

The next work should test these claims rather than treating them as conclusions
of the current study.

## Recommended direction: a structured latent reasoner

### Architecture

For input tokens `x`, extract a frozen token-level residual matrix at selected
layer `l`:

```text
H_l(x) in R^(sequence length x model width)
```

Project it into a small head dimension and treat it as read-only memory. Create
one or more trainable latent solution slots and update them recurrently:

```text
Z_0 = learned or input-conditioned latent slots
Z_(t+1) = Refine(Z_t, H_l(x))
output = Decode(Z_T, H_l(x))
```

A practical `Refine` block can use:

1. cross-attention from latent slots to frozen token residuals;
2. a residual update and normalization;
3. a small feed-forward network;
4. shared weights across recurrent steps.

Weight sharing matters. Stacking several different layers only tests whether a
larger head helps. Applying the same transition repeatedly tests whether an
iterative computation learned at one depth can continue improving with more
steps.

The system would have two independent compute axes:

- **backbone depth:** which frozen LLM layer supplies the representation;
- **latent depth:** how many recurrent refinements the selected module runs.

An early task router could select both the task module and its preferred
backbone depth. A halting head or a stability criterion could later select the
number of latent refinements per input.

### First target: proofs rather than labels

ProofWriter, RuleTaker with proof annotations, or ProsQA would provide a clean
first experiment. Instead of predicting only `True` or `False`, the module
should emit:

- the final verdict;
- which input facts and rules participate in the proof;
- proof-step ordering or graph edges;
- optionally a halting probability.

Rules and facts should be selected through pointer distributions over the
input, not regenerated as text. The output is then a structured object whose
validity can be checked mechanically.

This is meaningfully beyond classification because the system must identify,
compose, and order multiple input elements. It is also a stronger test of
latent reasoning than answer accuracy alone: a high answer score cannot hide
an invalid proof.

### Decisive evidence

The most convincing result would show all of the following:

1. Increasing recurrent steps improves accuracy primarily on deeper proofs.
2. The improvement survives comparison with a parameter- and FLOP-matched
   non-recurrent head.
3. The model generalizes to proof depths longer than those seen during
   training.
4. Predicted proofs are valid, not merely correlated evidence sets.
5. Direct structured inference is materially faster than autoregressive
   chain-of-thought at similar answer accuracy.
6. Different inputs use different amounts of latent computation without a
   major accuracy loss.

A gain from a larger head with no relationship between recurrence and proof
depth would be useful engineering, but it would not demonstrate latent
reasoning.

## Experimental plan

### Phase 0: inexpensive architecture check

Reuse existing residual banks where possible and compare:

- linear head;
- existing MLP head;
- parameter-matched feed-forward head;
- small recurrent latent head over the stored representation.

This can reveal optimization problems cheaply, but it is not a strong
reasoning experiment. A single pooled vector has already compressed the token
sequence, and repeated processing of that vector may only learn a more complex
decision boundary.

### Phase 1: token-level residual memory

Run fresh extraction that retains token-level residuals and token boundaries.
Start with a modest frozen backbone and a proof dataset whose outputs are
mechanically verifiable.

Suggested initial configuration:

- one selected mid-to-late residual layer plus an early-layer comparison;
- projected head width around 256;
- 4 to 8 latent slots;
- one weight-shared cross-attention refinement block;
- 1, 2, 4, and 8 recurrent steps;
- answer, proof-node, and proof-edge losses;
- deep supervision on intermediate recurrent states as an ablation.

### Required baselines

- Current last-token linear and MLP readouts.
- Mean or attention pooling followed by an MLP.
- A one-step latent-query head.
- An unshared stack with matched parameters and approximately matched FLOPs.
- A ReLIT-like answer-only recurrent head.
- Frozen-model direct answer and generated chain-of-thought.
- A task-specific encoder or small model trained under comparable supervision.

### Required ablations

- Final-token vector versus complete token residual memory.
- Backbone layer and recurrent step count.
- Shared versus unshared refinement weights.
- Fixed versus input-conditioned latent initialization.
- Answer-only versus proof supervision.
- One latent slot versus multiple slots.
- Fixed recurrence versus adaptive halting.
- Normal examples versus shuffled rules, distractor-heavy inputs, and longer
  proofs.

### Metrics

- Answer accuracy and macro-F1.
- Exact proof match.
- Proof-node and proof-edge F1.
- Mechanical proof validity.
- Performance stratified by gold proof depth.
- Generalization to longer proofs.
- Trainable parameter count.
- End-to-end latency, throughput, peak memory, and measured FLOPs.
- Accuracy as a function of both backbone depth and recurrent steps.

## Commercial direction: encode a document once, answer many fixed queries

ContractNLI suggests a systems-oriented version:

```text
contract encoded once
  -> cached token-level residual memory
  -> many lightweight requirement queries
  -> statuses + evidence spans + severity scores
```

For a fixed compliance schema, each requirement can have a learned latent query
or independently trained module. One contract encoding could then produce all
required findings in parallel rather than expanding the contract into many
full LLM prompts.

Potential outputs include:

- clause presence or absence;
- entailment and contradiction status;
- evidence spans;
- normalized risk or severity scores;
- relations among clauses;
- a set of missing obligations.

This is most compelling when an application asks many stable questions of the
same long document. Its economic claim is not merely “no generated tokens,”
but **one expensive document encoding followed by many cheap structured
queries**.

For arbitrary natural-language queries, the query must also be encoded. That
could use a separate frozen query pass or a smaller query encoder, so the
encode-once advantage is strongest for fixed or slowly changing schemas.

## Coconut-like frozen feedback

A closer adaptation of Coconut would feed learned continuous thoughts back
into the frozen LLM:

```text
prompt -> frozen LLM -> residual
       -> trainable projection and normalization
       -> append as soft thought embedding
       -> frozen LLM again
       -> repeat
       -> direct output head
```

Only the projection, optional thought controller, and output head would train.
The vocabulary would be bypassed during intermediate steps.

A projection is necessary even when hidden-state and embedding dimensions
match. A standard frozen LLM was not trained to consume its own hidden states
as token embeddings, and the two spaces need not have compatible
distributions.

This direction is conceptually close to Coconut, but less attractive as the
first experiment because every thought invokes the large backbone again. The
steps are sequential, training is more difficult, and answer-only supervision
may not teach useful recurrent dynamics. An external recurrent module over one
frozen residual extraction is cheaper and provides cleaner attribution.

## Other task-native outputs

### Regression and assessment

A latent module can directly emit continuous values rather than textualized
numbers:

- semantic similarity;
- translation or code quality;
- risk and severity;
- reward-model scores;
- judge scores;
- calibrated quantiles or intervals.

This avoids the mismatch between token likelihood and numerical distance.
However, regression by itself is only incrementally beyond the current project
unless recurrent refinement or multi-output structure contributes something
new.

### Parallel span and relation extraction

Multiple latent queries can act like object queries in set prediction and emit
an unordered set of:

```text
(start token, end token, type, score, relations)
```

This would support named entities, obligations, contradictions, claims,
evidence, and document events without autoregressive JSON generation. It is a
natural bridge from multi-head classification to structured extraction.

### Policy and planning

A more distant direction is to use the frozen LLM only to encode a textual
state, then train task-native transition, value, and policy modules:

```text
text state -> frozen residual memory
           -> action-conditioned latent transition
           -> imagined latent futures
           -> value/policy output
```

The system would output actions rather than language and could plan by rolling
latent states forward. This is the most conceptually different destination,
but the current results provide no evidence that frozen residuals support
accurate action-conditioned transitions. It should follow, not precede, a
successful structured-reasoning experiment.

## Nearby work and novelty constraints

The broad idea of frozen or latent reasoning is now active, so “frozen LLM plus
recursive head” is not by itself a credible novelty claim.

| Work | Frozen main LLM | Intermediate token generation | Final output | Relation |
|---|---:|---:|---|---|
| [Coconut](https://arxiv.org/abs/2412.06769) | No | No | Generated answer | Feeds a hidden state back as the next continuous input and trains with a curriculum. |
| [SoftCoT](https://arxiv.org/abs/2502.12134) | Yes | No for soft thoughts | Generated reasoning and answer | Projects hidden states from a small assistant into a frozen LLM's input space. |
| [Differentiable Cache Augmentation](https://arxiv.org/abs/2412.17747) | Yes | No for augmentation | Generated continuation | A trained coprocessor adds latent embeddings to the frozen decoder's key-value cache. |
| [RELISH](https://arxiv.org/abs/2604.01206) | Yes | No | Scalar | Iteratively cross-attends a learned latent state to frozen token representations for regression. |
| [ReLIT](https://arxiv.org/abs/2608.08113) | Yes | No during reasoning | Single logical answer through a projection head | Adds a recursive latent block to frozen TinyLlama representations; posted August 8, 2026 and especially close to this direction. |

A potentially distinctive combination would need to emphasize and test:

1. an unmodified frozen causal LLM;
2. task-specific residual depths;
3. early routing among independently trained modules;
4. recurrent latent computation outside the backbone;
5. direct, mechanically verifiable structured outputs;
6. adaptive compute across both backbone depth and latent iterations;
7. measured latency and throughput benefits;
8. modular addition of new tasks without retraining existing modules.

The novelty would be this complete system and its empirical behavior, not the
existence of latent recurrence alone.

## Failure modes and honest interpretation

- **Representation bottleneck:** a frozen layer may discard information needed
  for the task; no downstream recurrence can recover it.
- **Capacity masquerading as reasoning:** improvements may result from a larger
  head rather than iterative computation.
- **Shortcut proofs:** evidence selection can correlate with labels without
  forming a valid proof.
- **No extrapolation:** a recurrent module may work only at the step counts used
  in training.
- **Causal-token limitation:** early token states cannot incorporate later
  context; structured heads must attend over all token positions rather than
  assume every position is contextualized bidirectionally.
- **Systems mismatch:** nominal FLOP savings may fail to improve latency because
  small recurrent modules, dynamic halting, and mixed depths complicate
  batching.
- **Supervision mismatch:** generated reasoning baselines may benefit from broad
  reasoning pretraining, while a small head receives only task labels.

Negative findings would still be useful if they isolate whether the limiting
factor is frozen representation quality, token-level information extraction,
recurrent optimization, structured supervision, or serving overhead.

## Recommended immediate experiment

Build a token-memory proof reasoner on a modest Qwen model:

1. Extract complete token residuals from one early and one task-effective
   layer.
2. Train a small weight-shared latent-query module to predict both answer and
   proof structure.
3. Sweep 1, 2, 4, and 8 recurrent steps.
4. Compare against parameter-matched non-recurrent heads and generated
   reasoning.
5. Stratify every result by proof depth and evaluate on longer held-out proofs.
6. Measure actual latency and throughput.

The central question is:

> Can an unmodified frozen causal language model provide intermediate
> representations that a small, independently trained recurrent module
> converts into task-native structured solutions, with reasoning compute
> scaling through latent iterations instead of generated tokens?

A positive answer would move the project from efficient classification toward
a modular architecture for language-conditioned computation without language
as the output interface.
