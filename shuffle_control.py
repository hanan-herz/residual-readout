"""Shuffle control: is layer-1 task separation format or content?

Applies three text conditions to the SAME samples and measures task
separability (task-ID linear probe + Mahalanobis) at layer 1:

  identity      baseline (should reproduce ~1.0 task acc)
  word_shuffle  scramble WORD order in the formatted prompt
                (tokens ~ preserved, order/format destroyed)
  token_shuffle scramble TOKEN ids after tokenizing
                (strongest: kills words, POS, most local structure)

If separation survives word_shuffle but dies on token_shuffle, the first
block keys on something token-local (not macro layout). If it dies on both,
the separating signal is the sequential/format structure. If it survives
both, it is content carried in the token stream.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np
import torch

from common import (
    RESULTS_DIR,
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
    fit_mahalanobis,
    fit_task_id_linear,
    head_forward,
    mahalanobis_score,
    route_mahalanobis,
)

SHUFFLES = ("identity", "word_shuffle", "token_shuffle")
LAYER = 1


def _word_shuffle(text: str, rng) -> str:
    # Keep "Answer:" trailer in place so the residual is a fair comparison;
    # shuffle the body words.
    if "\nAnswer:" in text:
        body, _, tail = text.rpartition("\nAnswer:")
    else:
        body, tail = text, ""
    words = body.split()
    rng.shuffle(words)
    return " ".join(words) + (("\nAnswer:" + tail) if tail else "")


def shuf_texts(model, tok, texts, cond, rng):
    if cond in ("identity", "token_shuffle"):
        return texts
    return [_word_shuffle(t, rng) for t in texts]


def extract(model, tok, texts, cond, label, rng):
    """Layer-1 last-token residual via a hook on layers[LAYER] (full forward,
    so rotary position embeddings etc. are handled by the wrapper).
    token_shuffle scrambles token ids before the model call."""
    dev = model.device
    last = []
    for i in range(0, len(texts), 24):
        sub = texts[i:i + 24]
        enc = tok(sub, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(dev)
        if cond == "token_shuffle":
            ids = enc["input_ids"].clone()
            # shuffle tokens within each row (excluding pad positions)
            for b in range(ids.size(0)):
                ln = int(enc["attention_mask"][b].sum())
                perm = rng.permutation(ln)
                ids[b, :ln] = ids[b, perm]
            enc["input_ids"] = ids
        buf = []

        def hook(m, inp, o):
            if isinstance(o, tuple):
                o = o[0]
            buf.append(o.detach().float().cpu())

        h = model.model.layers[LAYER].register_forward_hook(hook)
        with torch.no_grad():
            model(**enc)
        h.remove()
        hs = buf[0]
        am = enc["attention_mask"].cpu()
        lens = am.shape[1] - 1 - am.flip(1).argmax(1)
        last.append(hs[torch.arange(hs.size(0)), lens])
        if (i // 24) % 20 == 19:
            print(f"  [{label}] {min(i+24,len(texts))}/{len(texts)}")
    return torch.cat(last, 0).numpy()


def main():
    ap = argparse.ArgumentParser(description="Shuffle control for task separation")
    ap.add_argument("--size", default="0.6B")
    ap.add_argument("--max-train", type=int, default=400)
    ap.add_argument("--max-val", type=int, default=200)
    ap.add_argument("--cond", default=None, choices=list(SHUFFLES),
                    help="run only one condition")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else \
        ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"model={args.size} device={device} layer={LAYER}")

    # load samples once
    samples = {}
    for t in TASKS:
        spec = get_task(t)
        rngs = np.random.default_rng(0)
        tr, va, *_ = spec.load(args.max_train, args.max_val, rngs)
        samples[t] = {"tr": tr, "va": va, "ytr": encode_labels(tr),
                      "yva": encode_labels(va)}

    conds = [args.cond] if args.cond else list(SHUFFLES)
    model = tok = None
    results = {}
    for cond in conds:
        print(f"\n===== CONDITION: {cond} =====")
        if model is None:
            print(f"loading {resolve_model(args.size)} on {device}...")
            model, tok = load_model(args.size, device=device)
        rng = np.random.default_rng(123)  # fixed per run for reproducibility
        Xva_parts, Xtr_parts, yva_parts, ytr_task, yva_task, y_ans = [], [], [], [], [], []
        for ti, t in enumerate(TASKS):
            txtr = shuf_texts(model, tok,
                              [fmt_example(r) for r in samples[t]["tr"]], cond, rng)
            txva = shuf_texts(model, tok,
                              [fmt_example(r) for r in samples[t]["va"]], cond, rng)
            trh = extract(model, tok, txtr, cond, f"{t}.tr", rng)
            vah = extract(model, tok, txva, cond, f"{t}.va", rng)
            Xtr_parts.append(trh); Xva_parts.append(vah)
            ytr_task.append(np.full(len(trh), ti, dtype=np.int64))
            yva_task.append(np.full(len(vah), ti, dtype=np.int64))
            y_ans.append(samples[t]["yva"].astype(np.int64))
        Xtr = np.concatenate(Xtr_parts, 0)
        Xva = np.concatenate(Xva_parts, 0)
        Ytr_task = np.concatenate(ytr_task, 0)
        Yva_task = np.concatenate(yva_task, 0)
        Yans = np.concatenate(y_ans, 0)
        n = len(Xva)

        # task-ID linear (supervised upper-line)
        sc, clf = fit_task_id_linear(Xtr, Ytr_task)
        tid = clf.predict(sc.transform(Xva))
        tid_acc = float((tid == Yva_task).mean())

        # Mahalanobis (unsupervised)
        maha = {t: fit_mahalanobis(Xtr[Ytr_task == i])
                for i, t in enumerate(TASKS)}
        maha_scores = {t: mahalanobis_score(maha[t], Xva) for t in TASKS}
        m_route = route_mahalanobis(maha_scores)
        m_acc = float((m_route == Yva_task).mean())

        # oracle/gold-task ceiling via evaluate_router (needs heads; we only
        # need task routing here, so report task_acc from tid/route)
        results[cond] = {
            "task_acc_linear": tid_acc,
            "task_acc_mahalanobis": m_acc,
            "chance": 1.0 / len(TASKS),
            "n": n,
        }
        print(f"  [{cond}] linear task_acc={tid_acc:.3f}  "
              f"mahalanobis task_acc={m_acc:.3f}  (chance {1/len(TASKS):.3f})")

    if args.out:
        out = args.out
    else:
        out = os.path.join(
            RESULTS_DIR, f"shuffle_control_{model_slug(args.size)}_l{LAYER}_t{args.max_train}_v{args.max_val}.json")
    with open(out, "w") as fh:
        json.dump({"meta": {"size": args.size, "layer": LAYER,
                            "cond": conds, "wash": "seed 123"},
                   "results": results}, fh, indent=2, default=str)
    print(f"\n[wrote] {out}")
    print("===== SUMMARY =====")
    for cond, r in results.items():
        print(f"  {cond:16s} linear={r['task_acc_linear']:.3f} "
              f"maha={r['task_acc_mahalanobis']:.3f}")


if __name__ == "__main__":
    main()