"""Modal pilot: multi-head single-pass routing at Qwen3-8B.

Replicates the 0.6B exp/multihead-single-pass findings at a second scale:
  route_pilot       depth-sweep style pilot at one tap layer (default L35)
  embed_control     is task separation present at the embedding (layer 0)?
  shuffle_control   word/token-shuffle at L1: format vs content

Self-contained (helpers inlined) because the Modal image only ships
bench.py/common.py/probe_main.py/tasks. Residuals cache into the
bench-results volume under <slug>/<task>/cache/vec_t*_v*_l*_p384_b8.npz
via common.cache_vectors, so reruns skip extraction.

Run (from repo root):
  python modal_multihead.py pilot            --layer 35
  python modal_multihead.py embed
  python modal_multihead.py shuffle          --layer 1
  python modal_multihead.py all              (sequential)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import modal
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# --- Modal image: torch stack + the repo files the pilot imports ----------
_KEEP_ROOT = frozenset({"common.py", "probe_main.py", "modal_multihead.py"})


def _ignore(p) -> bool:
    parts = p.parts
    if not parts:
        return True
    if parts[0] in _KEEP_ROOT and len(parts) == 1:
        return False
    if parts[0] == "tasks":
        if "__pycache__" in parts or any(x.endswith(".pyc") for x in parts):
            return True
        return False
    return True


image = (
    modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu22.04",
                              add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda",
          "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .pip_install("torch", "transformers>=4.46", "datasets", "scikit-learn",
                 "numpy", "sentencepiece", "protobuf", "accelerate")
    .add_local_dir(HERE, "/root/pilot", ignore=_ignore)
)

app = modal.App("vector-is-the-answer-multihead")

VOLUME_MOUNT = "/data"
bench_volume = modal.Volume.from_name("bench-results", create_if_missing=True)

WEIGHTS_MOUNT = "/weights"
weights_volume = modal.Volume.from_name("model-weights", create_if_missing=True)

TASKS = ("boolq", "ruletaker", "arc")


# ============================================================ remote body ==
def _remote_body(layer: int, mode: str, size: str,
                 max_train: int, max_val: int, batch: int,
                 shuffle_cond: str | None):
    os.chdir("/root/pilot")
    sys.path.insert(0, "/root/pilot")
    os.environ["HF_HOME"] = os.path.join(VOLUME_MOUNT, "hf_home")
    import torch  # noqa
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler

    import common
    from common import (
        CACHE_DIR, cache_vectors, load_cached_vectors, load_model,
        model_slug, resolve_model, task_dirs)
    from tasks import get_task
    from probe_main import mlp_probe

    os.makedirs(CACHE_DIR, exist_ok=True)

    def fmt_example(r):
        passage = (r.get("passage") or "").strip()
        if passage:
            return f"Passage: {passage}\nQuestion: {r['question']}\nAnswer:"
        return f"Question: {r['question']}\nAnswer:"

    def row_label(r):
        a = r["answer"]
        return int(a) if not isinstance(a, (bool, np.bool_)) else int(a)

    def encode_labels(rows):
        return np.array([row_label(r) for r in rows], dtype=np.int64)

    # ---- heads (same npz schema as bench.save_heads) ----
    def fit_head(Xtr, ytr, n_cls):
        _, net, mu, sd = mlp_probe(
            Xtr, ytr, Xtr[:1], ytr[:1], seed=0, device="cuda",
            return_net=True, n_classes=n_cls)
        sd_ = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        return {
            "n_classes": n_cls, "multi": n_cls > 2,
            "mlp_w1": sd_["0.weight"].astype(np.float32),
            "mlp_b1": sd_["0.bias"].astype(np.float32),
            "mlp_w2": sd_["2.weight"].astype(np.float32),
            "mlp_b2": sd_["2.bias"].astype(np.float32),
            "mlp_mu": np.asarray(mu, dtype=np.float32),
            "mlp_sd": np.asarray(sd, dtype=np.float32),
        }

    def head_forward(head, X):
        z = (X - head["mlp_mu"]) / (head["mlp_sd"] + 1e-8)
        h = np.maximum(0.0, z @ head["mlp_w1"].T + head["mlp_b1"])
        logits = h @ head["mlp_w2"].T + head["mlp_b2"]
        if head["multi"]:
            m = logits.max(axis=1, keepdims=True)
            e = np.exp(logits - m)
            probs = e / e.sum(axis=1, keepdims=True)
            return probs.argmax(1).astype(np.int64), probs.max(1)
        logits = logits.reshape(-1)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        pred = (p1 > 0.5).astype(np.int64)
        return pred, np.maximum(p1, 1.0 - p1)

    # ---- routers ----
    def fit_maha(Xtr, reg=1e-3):
        mu = Xtr.mean(0)
        Xc = Xtr - mu
        cov = (Xc.T @ Xc) / max(len(Xtr) - 1, 1)
        return {"mu": mu.astype(np.float32),
                "diag": np.diag(cov).astype(np.float32) + reg}

    def maha_score(bank, X):
        z = (X - bank["mu"]) / np.sqrt(bank["diag"])
        return -0.5 * (z * z).sum(axis=1)

    # ---- residual extraction (pad_max=384 to match 0.6B pilot) ----
    PAD_MAX = 384

    def extract(model, tok, texts, layer, batch, label):
        dev = model.device
        last = []
        for i in range(0, len(texts), batch):
            sub = texts[i:i + batch]
            enc = tok(sub, return_tensors="pt", padding=True,
                      truncation=True, max_length=PAD_MAX).to(dev)
            buf = []

            def hook(m, inp, o):
                if isinstance(o, tuple):
                    o = o[0]
                buf.append(o.flatten(2).detach().float().cpu())

            h = model.model.layers[layer].register_forward_hook(hook)
            with torch.no_grad():
                model(**enc)
            h.remove()
            hs = buf[0]
            am = enc["attention_mask"].cpu()
            lens = am.shape[1] - 1 - am.flip(1).argmax(1)
            last.append(hs[torch.arange(hs.size(0)), lens])
            if (i // batch) % 40 == 39:
                print(f"  [{label}] {min(i+batch,len(texts))}/{len(texts)}",
                      flush=True)
        return torch.cat(last, 0).numpy().astype(np.float32)

    def extract_embed(model, tok, texts, batch, label):
        dev = model.device
        last = []
        for i in range(0, len(texts), batch):
            sub = texts[i:i + batch]
            enc = tok(sub, return_tensors="pt", padding=True,
                      truncation=True, max_length=PAD_MAX).to(dev)
            with torch.no_grad():
                emb = model.model.embed_tokens(enc["input_ids"])
            am = enc["attention_mask"].cpu()
            lens = am.shape[1] - 1 - am.flip(1).argmax(1)
            last.append(emb.detach().float().cpu()[
                torch.arange(emb.size(0)), lens])
            if (i // batch) % 40 == 39:
                print(f"  [{label}] {min(i+batch,len(texts))}/{len(texts)}",
                      flush=True)
        return torch.cat(last, 0).numpy().astype(np.float32)

    def shuffle_texts(texts, cond, rng):
        if cond == "identity":
            return texts
        out = []
        for t in texts:
            if "\nAnswer:" in t:
                body, _, _ = t.rpartition("\nAnswer:")
            else:
                body = t
            if cond == "word_shuffle":
                w = body.split()
                rng.shuffle(w)
                out.append(" ".join(w) + "\nAnswer:")
            elif cond == "token_shuffle":
                out.append("SHUFTOK:" + body)
            else:
                raise ValueError(cond)
        return out

    def extract_shuffled(model, tok, texts, layer, batch, label, rng):
        """extract, but SHUFTOK-marked texts get token-id shuffles."""
        dev = model.device
        last = []
        for i in range(0, len(texts), batch):
            sub = texts[i:i + batch]
            marked = [s.startswith("SHUFTOK") for s in sub]
            plain = [s[8:] if m_ else s for m_, s in zip(marked, sub)]
            enc = tok(plain, return_tensors="pt", padding=True,
                      truncation=True, max_length=PAD_MAX).to(dev)
            if any(marked):
                ids = enc["input_ids"].clone()
                for b in range(ids.size(0)):
                    if not marked[b]:
                        continue
                    ln = int(enc["attention_mask"][b].sum())
                    perm = rng.permutation(ln)
                    ids[b, :ln] = ids[b, perm]
                enc["input_ids"] = ids
            buf = []

            def hook(m, inp, o):
                if isinstance(o, tuple):
                    o = o[0]
                buf.append(o.flatten(2).detach().float().cpu())

            h = model.model.layers[layer].register_forward_hook(hook)
            with torch.no_grad():
                model(**enc)
            h.remove()
            hs = buf[0]
            am = enc["attention_mask"].cpu()
            lens = am.shape[1] - 1 - am.flip(1).argmax(1)
            last.append(hs[torch.arange(hs.size(0)), lens])
        return torch.cat(last, 0).numpy().astype(np.float32)

    # ---- task-ID probes ----
    def fit_tid_linear(Xtr, ytr):
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
        clf.fit(sc.transform(Xtr), ytr)
        return sc, clf

    # ============================================================ modes ==
    print(f"[multihead] mode={mode} size={size} layer={layer} "
          f"cond={shuffle_cond} t{max_train}/v{max_val}", flush=True)
    try:
        import subprocess
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"], text=True).strip()
        print(f"[multihead] nvidia-smi: {smi}", flush=True)
    except Exception as e:
        print(f"[multihead] nvidia-smi failed: {e}", flush=True)

    samples = {}
    for t in TASKS:
        spec = get_task(t)
        rng0 = np.random.default_rng(0)
        tr, va, _, mt, mv = spec.load(max_train, max_val, rng0)
        samples[t] = {"ytr": encode_labels(tr), "yva": encode_labels(va),
                      "texts_tr": [fmt_example(r) for r in tr],
                      "texts_va": [fmt_example(r) for r in va]}

    model = tok = None

    def banks_at_layer(L, cond=None):
        """Residual bank per task at layer L (or embed if L<0)."""
        nonlocal model, tok
        banks = {}
        rng = np.random.default_rng(123)
        for t in TASKS:
            dirs = task_dirs(VOLUME_MOUNT, size, t)
            cached = None
            if cond in (None, "identity") and L >= 0:
                cached = load_cached_vectors(
                    max_train, max_val,
                    ["last_tr", "last_va", "ytr", "yva"],
                    cache_dir=dirs["cache"], layer=L, pad_max=PAD_MAX,
                    batch=batch)
            if cached is not None:
                print(f"  [{t}] cache hit L{L}", flush=True)
                banks[t] = {
                    "last_tr": cached["last_tr"].astype(np.float32),
                    "last_va": cached["last_va"].astype(np.float32),
                    "ytr": np.asarray(cached["ytr"]),
                    "yva": np.asarray(cached["yva"]),
                }
                continue
            if model is None:
                weights_dir = os.path.join(WEIGHTS_MOUNT, model_slug(size))
                print(f"  loading {resolve_model(size)} "
                      f"(weights_dir={weights_dir}) ...", flush=True)
                model, tok = load_model(size, device="cuda",
                                        weights_dir=weights_dir)
            t0 = time.time()
            if cond == "token_shuffle":
                txtr = shuffle_texts(samples[t]["texts_tr"], cond, rng)
                txva = shuffle_texts(samples[t]["texts_va"], cond, rng)
                trh = extract_shuffled(model, tok, txtr, L, batch,
                                       f"{t}.tr", rng)
                vah = extract_shuffled(model, tok, txva, L, batch,
                                       f"{t}.va", rng)
            elif cond == "word_shuffle":
                txtr = shuffle_texts(samples[t]["texts_tr"], cond, rng)
                txva = shuffle_texts(samples[t]["texts_va"], cond, rng)
                trh = extract(model, tok, txtr, L, batch, f"{t}.tr")
                vah = extract(model, tok, txva, L, batch, f"{t}.va")
            elif L < 0:
                trh = extract_embed(model, tok, samples[t]["texts_tr"],
                                    batch, f"{t}.tr")
                vah = extract_embed(model, tok, samples[t]["texts_va"],
                                    batch, f"{t}.va")
            else:
                trh = extract(model, tok, samples[t]["texts_tr"], L, batch,
                              f"{t}.tr")
                vah = extract(model, tok, samples[t]["texts_va"], L, batch,
                              f"{t}.va")
            print(f"  [{t}] extract done in {time.time()-t0:.0f}s", flush=True)
            banks[t] = {"last_tr": trh, "last_va": vah,
                        "ytr": samples[t]["ytr"], "yva": samples[t]["yva"]}
            if cond in (None, "identity") and L >= 0:
                cache_vectors(max_train, max_val,
                              last_tr=trh, last_va=vah,
                              ytr=samples[t]["ytr"], yva=samples[t]["yva"],
                              cache_dir=dirs["cache"], layer=L,
                              pad_max=PAD_MAX, batch=batch)
        return banks

    def routing_eval(banks, fit_heads=True):
        heads = {}
        for t in TASKS:
            n_cls = 4 if t == "arc" else 2
            heads[t] = fit_head(banks[t]["last_tr"], banks[t]["ytr"], n_cls)
        Xva = np.concatenate([banks[t]["last_va"] for t in TASKS], 0)
        y_ans = np.concatenate([banks[t]["yva"].astype(np.int64)
                                for t in TASKS], 0)
        y_task = np.concatenate(
            [np.full(len(banks[t]["last_va"]), i, dtype=np.int64)
             for i, t in enumerate(TASKS)], 0)
        Xtr = np.concatenate([banks[t]["last_tr"] for t in TASKS], 0)
        ytr_task = np.concatenate(
            [np.full(len(banks[t]["last_tr"]), i, dtype=np.int64)
             for i, t in enumerate(TASKS)], 0)

        preds, confs = {}, {}
        for t in TASKS:
            preds[t], confs[t] = head_forward(heads[t], Xva)

        def router_answer(route):
            ok = np.zeros(len(route), dtype=bool)
            for ti, t in enumerate(TASKS):
                pick = route == ti
                if pick.any():
                    n_cls_t = 4 if t == "arc" else 2
                    ok[pick] = (preds[t][pick] == y_ans[pick])
            return float(ok.mean())

        results = {}
        ceiling_route = y_task.copy()
        results["gold_task_ceiling"] = {
            "task_acc": 1.0,
            "answer_acc_strict": router_answer(ceiling_route)}
        naive = np.stack([confs[t] for t in TASKS], 1).argmax(1)
        results["naive_max_conf"] = {
            "task_acc": float((naive == y_task).mean()),
            "answer_acc_strict": router_answer(naive)}
        maha = {t: fit_maha(banks[t]["last_tr"]) for t in TASKS}
        mscores = np.stack([maha_score(maha[t], Xva) for t in TASKS], 1)
        m_route = mscores.argmax(1)
        results["mahalanobis"] = {
            "task_acc": float((m_route == y_task).mean()),
            "answer_acc_strict": router_answer(m_route)}
        sc, clf = fit_tid_linear(Xtr, ytr_task)
        tid = clf.predict(sc.transform(Xva))
        results["task_id_linear"] = {
            "task_acc": float((tid == y_task).mean()),
            "answer_acc_strict": router_answer(tid)}
        results["n_mixed_val"] = int(len(Xva))
        return results

    out = {"meta": {"mode": mode, "size": size, "model_id": resolve_model(size),
                    "layer": layer, "shuffle_cond": shuffle_cond,
                    "max_train": max_train, "max_val": max_val,
                    "pad_max": PAD_MAX, "batch": batch,
                    "head_protocol": "refit-on-pilot-train",
                    "tasks": list(TASKS)}}

    if mode == "pilot":
        banks = banks_at_layer(layer)
        out["routers"] = routing_eval(banks)
    elif mode == "embed":
        banks = banks_at_layer(-1)
        out["routers"] = routing_eval(banks)
    elif mode == "shuffle":
        res = {}
        for cond in ("identity", "word_shuffle", "token_shuffle"):
            banks = banks_at_layer(layer, cond=cond)
            Xva = np.concatenate([banks[t]["last_va"] for t in TASKS], 0)
            y_task = np.concatenate(
                [np.full(len(banks[t]["last_va"]), i, dtype=np.int64)
                 for i, t in enumerate(TASKS)], 0)
            Xtr = np.concatenate([banks[t]["last_tr"] for t in TASKS], 0)
            ytr_task = np.concatenate(
                [np.full(len(banks[t]["last_tr"]), i, dtype=np.int64)
                 for i, t in enumerate(TASKS)], 0)
            sc, clf = fit_tid_linear(Xtr, ytr_task)
            tid = clf.predict(sc.transform(Xva))
            maha = {t: fit_maha(banks[t]["last_tr"]) for t in TASKS}
            mscores = np.stack([maha_score(maha[t], Xva) for t in TASKS], 1)
            res[cond] = {
                "task_acc_linear": float((tid == y_task).mean()),
                "task_acc_mahalanobis": float((mscores.argmax(1)
                                               == y_task).mean()),
                "chance": 1.0 / len(TASKS),
            }
            print(f"  [{cond}] linear={res[cond]['task_acc_linear']:.3f} "
                  f"maha={res[cond]['task_acc_mahalanobis']:.3f}", flush=True)
        out["results"] = res
    else:
        raise ValueError(mode)

    # persist JSON on the volume
    vol_dir = os.path.join(VOLUME_MOUNT, "multihead", model_slug(size))
    os.makedirs(vol_dir, exist_ok=True)
    tag = (f"_{mode}_l{layer}_t{max_train}_v{max_val}"
           + (f"_{shuffle_cond}" if shuffle_cond else ""))
    with open(os.path.join(vol_dir, f"pilot{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    bench_volume.commit()
    print(f"[multihead] wrote {vol_dir}/pilot{tag}.json", flush=True)
    return out


@app.function(image=image,
              volumes={VOLUME_MOUNT: bench_volume,
                       WEIGHTS_MOUNT: weights_volume},
              gpu="A100-80GB", timeout=3 * 60 * 60, cpu=16.0, memory=65536)
def run_multihead(mode: str, size="8B", layer=35, max_train=1000,
                  max_val=400, batch=16):
    """Single entrypoint (bench.py pattern): one function, mode selects.

    mode: pilot | embed | shuffle | all
    "all" runs pilot -> embed -> shuffle sequentially in ONE container,
    so the model is loaded once and reused across modes.
    """
    if mode == "all":
        return {
            "pilot": _remote_body(layer, "pilot", size, max_train, max_val,
                                  batch, None),
            "embed": _remote_body(-1, "embed", size, max_train, max_val,
                                  batch, None),
            "shuffle": _remote_body(1, "shuffle", size, 400, 200, batch,
                                    None),
        }
    if mode == "embed":
        return _remote_body(-1, "embed", size, max_train, max_val, batch,
                              None)
    if mode == "shuffle":
        return _remote_body(layer, "shuffle", size, max_train, max_val,
                            batch, None)
    return _remote_body(layer, "pilot", size, max_train, max_val, batch,
                        None)


# ============================================================ local CLI ==
def _show(name, out):
    print(f"\n===== {name} =====")
    if "routers" in out:
        for k, v in out["routers"].items():
            if k == "n_mixed_val":
                continue
            print(f"  {k:18s} task={v['task_acc']:.3f} "
                  f"ans={v['answer_acc_strict']:.3f}")
    if "results" in out:
        for c, r in out["results"].items():
            print(f"  {c:14s} linear={r['task_acc_linear']:.3f} "
                  f"maha={r['task_acc_mahalanobis']:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Modal 8B multi-head pilot")
    ap.add_argument("which", choices=["pilot", "embed", "shuffle", "all"])
    ap.add_argument("--size", default="8B")
    ap.add_argument("--layer", type=int, default=35)
    ap.add_argument("--max-train", type=int, default=1000)
    ap.add_argument("--max-val", type=int, default=400)
    args = ap.parse_args()

    with app.run():
        if args.which == "all":
            out = run_multihead.remote("all", size=args.size,
                                       layer=args.layer,
                                       max_train=args.max_train,
                                       max_val=args.max_val)
            for name in ("pilot", "embed", "shuffle"):
                _show(name, out[name])
        else:
            _show(args.which,
                  run_multihead.remote(args.which, size=args.size,
                                       layer=args.layer,
                                       max_train=args.max_train,
                                       max_val=args.max_val))


if __name__ == "__main__":
    main()