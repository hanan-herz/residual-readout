# Literature and positioning: multi-head single-pass classification

This note records the literature scan for techniques related to the
multi-head single-pass experiment. It is a research note, not yet a final
paper related-work section.

## Working claim

The project combines the following elements:

```text
frozen causal language model
  -> early residual task router
  -> independently trained task-specific readout heads
  -> task-specific later-layer stopping
```

The closest literature covers these components separately, but the scan did
not identify an exact match for the full combination, especially the use of an
early residual task probe to select independently trained readout heads and
per-task answer taps in a single frozen causal-language-model pass.

## Closest comparisons

| Work | Shared/frozen backbone | Task-specific heads | Infers task at test time | Adaptive depth | Relation |
|---|---:|---:|---:|---:|---|
| **DeeBERT** (Xin et al., ACL 2020) | Mostly frozen during exit training | Yes, intermediate classifiers | No; one task at a time | Yes | Closest efficiency precedent |
| **FastBERT** (Liu et al., ACL 2020) | Partly frozen/staged | Yes | No | Yes | Early-exit precedent |
| **PABEE** (Zhou et al., NeurIPS 2020) | Staged | Yes | No | Yes | Early-exit and robustness precedent |
| **Latif & Zhai** (2024/2025) | Frozen backbone | Task-specific adapters/heads | Usually task ID supplied | No per-task depth | Closest frozen-backbone deployment setup |
| **L2R** (Araujo et al., EMNLP 2024) | Frozen PLM with isolated PEFT modules | Task/module-specific | Yes, trained router | Not the main focus | Closest modular language-model routing |
| **Eidetic Learning** (Dronen & Balestriero, 2025) | Shared body with task-isolated capacity | Yes | Yes: hidden state to task classifier to head | No | Closest explicit task-ID/head routing |
| **PHATGOOSE** (Muqeeth et al., ICML 2024) | Frozen backbone plus PEFT experts | Adapter experts | Yes, learned gates | No | Related expert routing, but token-wise/internal |
| **MMoE / PLE** | Shared trainable experts | Task-specific towers | Usually task known | No | General multi-task routing precedent |

## 1. Early-exit classification

### DeeBERT

**DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference**
(Xin et al., ACL 2020) adds lightweight classifiers after Transformer layers.
An input exits when an intermediate classifier is sufficiently confident. It
reports up to roughly 40% inference-time savings on GLUE-style classification.

Overlap with this project:

- pretrained/frozen Transformer backbone;
- lightweight intermediate classifiers;
- one forward pass with early stopping;
- layer-dependent answer quality.

Difference:

- DeeBERT normally handles one downstream task at a time;
- its exit decision is based on answer confidence;
- it does not infer which task-specific head to select;
- the task is generally known before inference.

This project adds:

```text
L1 residual -> which task? -> which head and which later tap?
```

### FastBERT and PABEE

**FastBERT: a Self-distilling BERT with Adaptive Inference Time** (Liu et
al., ACL 2020) improves intermediate exits with self-distillation.

**BERT Loses Patience: Fast and Robust Inference with Early Exit** (Zhou et
al., NeurIPS 2020) uses prediction stability/patience rather than only
confidence.

These are precedents for input-adaptive computation, but they do not
primarily study mixed-task routing among independently trained heads.

## 2. Frozen backbone with task-specific heads

### Efficient Multi-Task Inferencing with a Shared Backbone and Lightweight
Task-Specific Adapters for Automatic Scoring

Latif and Zhai (2024/2025) use a shared frozen backbone with task-specific
adapters or heads for 27 automated-scoring tasks. They report lower memory and
latency than separate fully fine-tuned models.

Overlap:

- one shared frozen backbone;
- independently specialized task modules;
- lightweight task-specific prediction components;
- deployment and memory-efficiency motivation.

Difference:

- task identity is generally available to the system;
- the required task module is loaded rather than inferred from an early
  residual;
- there is no per-task residual-layer optimum or L1 routing experiment;
- the backbone is a shared feature extractor rather than an adaptive-depth
  one-pass path.

This is a useful citation for the modular frozen-backbone setup, but not for
the complete routing result.

## 3. Automatic task routing to task-specific heads

### Eidetic Learning

**Eidetic Learning: an Efficient and Provable Solution to Catastrophic
Forgetting** (Dronen and Balestriero, 2025) is the closest explicit
architectural analogy:

```text
shared body -> task-classification head -> selected task-specific head
```

The paper predicts task identity from hidden states and routes each example
to its corresponding classifier. It is designed for continual learning and
uses structured capacity isolation to prevent forgetting.

Overlap:

- task identity is inferred rather than supplied;
- hidden representation to task classifier to task-specific head;
- modular/sequential task-head training;
- one shared body with automatic routing.

Difference:

- backbone capacity is explicitly partitioned and modified;
- the goal is continual-learning retention;
- it is not centered on frozen causal-LLM residual readouts;
- routing is not an early L1 probe followed by a task-specific answer tap;
- there is no task-specific adaptive stopping experiment.

### L2R

**Learning to Route for Dynamic Adapter Composition in Continual Learning with
Language Models** (Araujo, Moens, and Tuytelaars, EMNLP 2024) trains task-
specific PEFT modules separately, then trains routers to compose or select
them using a small memory of examples.

Overlap:

- independently trained task modules;
- post-hoc routing;
- language-model backbone;
- separation between specialist training and router training.

Difference:

- routed objects are adapters that modify internal Transformer computation;
- routing is composition/weighting rather than selecting a readout head;
- it does not test early residual task routing followed by continuation to a
  task-specific answer layer;
- it requires router training and memory examples.

## 4. Expert and multi-task routing

### PHATGOOSE

**Learning to Route Among Specialized Experts for Zero-Shot Generalization**
(Muqeeth et al., ICML 2024) routes among independently trained PEFT experts,
often at token and layer level.

It is relevant for independently trained specialists, a frozen base model,
post-hoc routing, and modular expert pools. It differs because it changes
internal model computation by activating or combining experts; this project
selects one answer head after shared residual computation.

### MMoE and PLE

**Modeling Task Relationships in Multi-task Learning with Multi-gate
Mixture-of-Experts** (Ma et al., KDD 2018) and **Progressive Layered
Extraction** (Tang et al., RecSys 2020) are foundational multi-task routing
methods. They use shared experts and task-specific gates/towers.

They are useful background, but generally train the backbone/experts jointly,
usually know the task or use task-conditioned gates, and do not use a frozen
causal model's residual stream as a post-hoc task registry.

## What appears distinctive about this combination

The likely novelty is not any single component. Each component has precedent:

- shared backbone plus task heads: standard;
- early exit: established;
- task-ID routing: established in continual learning;
- adapter/expert routing: established;
- geometric/prototype routing: established in related forms.

The more specific combination appears less represented:

1. frozen open-weight causal LLM;
2. answer heads trained independently per task and per layer;
3. task identity decoded from an early residual;
4. one-pass continuation to a task-specific optimal answer layer;
5. router and answer heads trained independently;
6. Mahalanobis as a training-free extensibility baseline;
7. direct comparison of supervised routing and geometric routing under
   shuffle/off-manifold controls.

A defensible positioning statement is:

> Prior work studies early exits for a known task, modular adapters or
> experts for task composition, and task-ID routing in continual learning.
> We study the combination of early residual task identification with
> independently trained per-task readout heads on a frozen causal language
> model, including task-specific depth selection within one forward pass.

## Priority citations

If the related-work section is short, prioritize these:

1. **DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference** — Xin
   et al., ACL 2020. <https://arxiv.org/abs/2004.12993>
2. **FastBERT: a Self-distilling BERT with Adaptive Inference Time** — Liu
   et al., ACL 2020. <https://arxiv.org/abs/2004.02178>
3. **BERT Loses Patience: Fast and Robust Inference with Early Exit** — Zhou
   et al., NeurIPS 2020. <https://arxiv.org/abs/2006.04152>
4. **Efficient Multi-Task Inferencing with a Shared Backbone and Lightweight
   Task-Specific Adapters for Automatic Scoring** — Latif and Zhai,
   2024/2025. <https://arxiv.org/abs/2412.21065>
5. **Learning to Route for Dynamic Adapter Composition in Continual Learning
   with Language Models** — Araujo et al., EMNLP 2024.
   <https://arxiv.org/abs/2408.09053>
6. **Eidetic Learning: an Efficient and Provable Solution to Catastrophic
   Forgetting** — Dronen and Balestriero, 2025.
   <https://arxiv.org/abs/2502.09500>

Useful surveys:

- **A Survey of Early Exit Deep Neural Networks in NLP** (2025):
  <https://arxiv.org/abs/2501.07670>
- **A Survey on Dynamic Neural Networks for Natural Language Processing**
  (2022): <https://arxiv.org/abs/2202.07101>

## Scope note

This scan is a positioning note based on the linked abstracts/pages and is
not a systematic review. Before submission, verify final venue metadata,
publication versions, exact experimental details, and whether any newer work
has appeared that directly combines early residual task routing with
post-hoc readout-head selection.
