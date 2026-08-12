"""Layer-0 control: task routing on the PRE-computation embedding.

Is the layer-1 task separation (maha ~1.0, no labels) present already at
the *embedding* (before any transformer block)? If yes, the "tasks separate
super early" finding is partly prompt-format leak through the embedding;
if no, the first block does the separating — a stronger story.

Layer-0 here = `model.model.embed_tokens` output (true embedding), NOT
`layers[0]` (which is after the first block). Reuses the multihead routers /
heads / eval verbatim so numbers are comparable to the depth sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from common import (
    CACHE_DIR,
    RESULTS_DIR,
    cache_path,
    cache_vectors,
    load_cached_vectors,
    load_model,
    model_slug,
    resolve_model,
    task_dirs,
)
from tasks import get_task
from bench import encode_labels, fmt_example
from multihead_route import (
    TASKS,
    evaluate_router,
    fit_answer_head_local,
    fit_mahalanobis,
    fit_task_id_linear,
    fit_task_id_mlp,
    head_forward,
    mahalanobis_score,
    predict_task_mlp,
    route_mahalanobis,
    route_naive_max_conf,
)


def embed_vectors(model, tok, texts, batch=24, label=""):
    """Capture the embedding output (before all transformer blocks)."""
    dev = model.device
    last_out, mean_out = [], []

    def forward_batch(sub):
        enc = tok(sub, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(dev)
        emb = model.model.embed_tokens(enc["input_ids"])
        am = enc["attention_mask"].cpu()
        lens = am.shape[1] - 1 - am.flip(1).argmax(1)
        last = emb.detach().float().cpu()[torch.arange(emb.size(0)), lens]
        mean = (emb.detach().float().cpu() * am.float().unsqueeze(-1)).sum(1) \
            / am.float().sum(1).unsqueeze(-1)
        last_out.append(last)
        mean_out.append(mean)

    for done in range(0, len(texts), batch):
        forward_batch(texts[done:done + batch])
        if (done + batch) % 20 == 0:
            print(f"  [{label}] {min(done + batch, len(texts))}/{len(texts)}")
    return torch.cat(last_out, 0).numpy(), torch.cat(mean_out, 0).numpy()


def main():
    ap = argparse.ArgumentParser(description="Layer-0 embedding routing control")
    ap.add_argument("--size", default="0.6B")
    ap.add_argument("--max-train", type=int, default=400)
    ap.add_argument("--max-val", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    probe_dev = device if device != "cpu" else "cpu"

    print(f"layer-0 embedding control: model={args.size} device={device}")

    model_path = os.path.join(CACHE_DIR, model_slug(args.size), "embed_l0", "cache")
    banks = {}
    # Cache path must not collide with real layers; use a dedicated tag.
    tag0 = "_l0embed"
    for t in TASKS:
        dirs = task_dirs(CACHE_DIR, args.size, t)
        cached = load_cached_vectors(
            args.max_train, args.max_val, ["last_tr", "last_va", "ytr", "yva"],
            cache_dir=dirs["cache"], tag=tag0)
        if cached is not None:
            print(f"  [{t}] embed cache hit")
            banks[t] = {
                "last_tr": cached["last_tr"].astype(np.float32),
                "last_va": cached["last_va"].astype(np.float32),
                "ytr": np.asarray(cached["ytr"]),
                "yva": np.asarray(cached["yva"]),
            }
            continue

        if not banks:
            print(f"loading {resolve_model(args.size)} on {device}...")
            model, tok = load_model(args.size, device=device)

        spec = get_task(t)
        rng = np.random.default_rng(0)
        train, val, _, mt, mv = spec.load(args.max_train, args.max_val, rng)
        ytr = encode_labels(train)
        yva = encode_labels(val)
        t0 = time.time()
        last_tr, _ = embed_vectors(model, tok, [fmt_example(r) for r in train],
                                    args.batch, label=f"{t}.tr")
        last_va, _ = embed_vectors(model, tok, [fmt_example(r) for r in val],
                                    args.batch, label=f"{t}.va")
        print(f"  [{t}] embed extract done in {time.time()-t0:.0f}s")
        cache_vectors(args.max_train, args.max_val,
                      last_tr=last_tr, last_va=last_va, ytr=ytr, yva=yva,
                      cache_dir=dirs["cache"], tag=tag0)
        banks[t] = {"last_tr": last_tr.astype(np.float32),
                    "last_va": last_va.astype(np.float32),
                    "ytr": ytr, "yva": yva}

    # Mixed val / train
    Xva = np.concatenate([banks[t]["last_va"] for t in TASKS], 0)
    y_ans = np.concatenate([banks[t]["yva"].astype(np.int64) for t in TASKS], 0)
    y_task = np.concatenate(
        [np.full(len(banks[t]["yva"]), i, dtype=np.int64) for i, t in enumerate(TASKS)], 0)
    Xtr = np.concatenate([banks[t]["last_tr"] for t in TASKS], 0)
    ytr_task = np.concatenate(
        [np.full(len(banks[t]["last_tr"]), i, dtype=np.int64) for i, t in enumerate(TASKS)], 0)
    n = len(Xva)

    # Refit heads on embedding (uniform protocol)
    heads = {}
    for t in TASKS:
        n_cls = 4 if t == "arc" else 2
        heads[t] = fit_answer_head_local(
            banks[t]["last_tr"], banks[t]["ytr"],
            banks[t]["last_va"], banks[t]["yva"], n_cls, probe_dev)
    preds, confs = {}, {}
    for t in TASKS:
        p, c, _ = head_forward(heads[t], Xva)
        preds[t], confs[t] = p, c

    # Routers
    maha = {t: fit_mahalanobis(banks[t]["last_tr"]) for t in TASKS}
    maha_scores = {t: mahalanobis_score(maha[t], Xva) for t in TASKS}
    sc, clf = fit_task_id_linear(Xtr, ytr_task)
    tid_lin = clf.predict(sc.transform(Xva))
    tid_lin_proba = clf.predict_proba(sc.transform(Xva))
    tid_lin_conf = tid_lin_proba.max(axis=1)
    net = fit_task_id_mlp(Xtr, ytr_task, probe_dev)
    tid_mlp, tid_mlp_conf = predict_task_mlp(net, Xva)

    gold_ceiling = y_task.copy()
    naive = route_naive_max_conf(confs)
    maha_route = route_mahalanobis(maha_scores)
    n_classes_of = {t: heads[t]["n_classes"] for t in TASKS}

    results = []
    for name, route, rconf in [
        ("gold_task_ceiling", gold_ceiling, np.ones(n)),
        ("naive_max_conf", naive,
         np.stack([confs[t] for t in TASKS], 1).max(1)),
        ("mahalanobis", maha_route,
         (lambda s: (np.exp(s - s.max(1, keepdims=True))
                     / np.exp(s - s.max(1, keepdims=True)).sum(1, keepdims=True)).max(1))(
             np.stack([maha_scores[t] for t in TASKS], 1))),
        ("task_id_linear", tid_lin, tid_lin_conf),
        ("task_id_mlp", tid_mlp, tid_mlp_conf),
    ]:
        ev = evaluate_router(
            name, route.astype(np.int64), y_task, preds, y_ans, n_classes_of,
            router_conf=rconf, abstain_thresh=0.7 if name != "gold_task_ceiling" else None)
        results.append(ev)

    chance_task = 1.0 / len(TASKS)
    ceiling_ans = next(r for r in results if r["router"] == "gold_task_ceiling")["answer_acc_strict"]
    summary = {
        "meta": {
            "model": args.size,
            "model_id": resolve_model(args.size),
            "layer": 0,  # embedding
            "layer_note": "embed_tokens output (pre-transformer-block)",
            "head_protocol": "refit-on-pilot-train",
            "max_train": args.max_train,
            "max_val": args.max_val,
            "n_mixed_val": n,
            "tasks": list(TASKS),
            "chance_task_acc": chance_task,
        },
        "routers": results,
        "gold_task_ceiling_answer": ceiling_ans,
    }
    out_path = os.path.join(
        RESULTS_DIR, f"multihead_route_{model_slug(args.size)}_t{args.max_train}_v{args.max_val}_embed0.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\n[wrote] {out_path}")
    print("========== LAYER-0 (embedding) HEADLINE ==========")
    for r in results:
        print(f"  {r['router']:18s} task={r['task_acc']:.3f} ans={r['answer_acc_strict']:.3f}")
    print("gold-task ceiling answer (correct head always): "
          f"{summary['gold_task_ceiling_answer']:.3f}")


if __name__ == "__main__":
    main()