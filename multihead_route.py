"""Multi-head routing pilot: do task heads know *their* problem type?

Attaches all saved closed-set answer heads to the same final residual and
compares routers:

  oracle            gold task head (upper bound on answer acc)
  naive_max_conf    pick head with highest *answer* confidence
  mahalanobis       pick task whose train residual Gaussian fits best
  task_id_linear    logistic task-ID probe on residual
  task_id_mlp       small MLP task-ID probe on residual

Also reports the cross-task answer matrix (head applied OOD) and confidence
calibration by true task.

Usage (local, Qwen3-0.6B — residuals cached under cloud_bench_cache):

  python multihead_route.py --size 0.6B --max-train 400 --max-val 200

Larger pilot (slower first pass):

  python multihead_route.py --size 0.6B --max-train 1000 --max-val 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from common import load_model, n_layers as discover_layers, resolve_model
from probe_main import mlp_probe
from tasks import get_task
from common import (
    CACHE_DIR,
    PAD_MAX,
    RESULTS_DIR,
    cache_path,
    cache_vectors,
    load_cached_vectors,
    model_slug,
    task_dirs,
)
from bench import (
    encode_labels,
    fmt_example,
    to_vecs,
)

TASKS = ("boolq", "ruletaker", "arc")


# ------------------------------------------------------------------ heads --
def load_head_npz(path: str) -> dict:
    z = np.load(path)
    n_out = int(z["mlp_w2"].shape[0])
    return {
        "path": path,
        "n_classes": n_out if n_out > 1 else 2,
        "multi": n_out > 1,
        "mlp_w1": z["mlp_w1"].astype(np.float32),
        "mlp_b1": z["mlp_b1"].astype(np.float32),
        "mlp_w2": z["mlp_w2"].astype(np.float32),
        "mlp_b2": z["mlp_b2"].astype(np.float32),
        "mlp_mu": z["mlp_mu"].astype(np.float32),
        "mlp_sd": z["mlp_sd"].astype(np.float32),
    }


def find_head(size: str, task: str, layer: int) -> str | None:
    """Prefer task-tree head, then results/ shelf for BoolQ-era artifacts."""
    slug = model_slug(size)
    candidates = [
        os.path.join(CACHE_DIR, slug, task, "heads", f"head_l{layer}.npz"),
        os.path.join(RESULTS_DIR, f"head_{slug.replace('__', '_')}_l{layer}.npz"),
        # BoolQ shelf names used historically
        os.path.join(RESULTS_DIR, f"head_Qwen_Qwen3-{size}_l{layer}.npz"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def head_forward(head: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred, conf, probs).

    Binary: pred in {0,1}, conf = max(p, 1-p), probs shape [n, 2].
    Multi:  pred class index, conf = max softmax, probs [n, C].
    """
    z = (X - head["mlp_mu"]) / (head["mlp_sd"] + 1e-8)
    h = np.maximum(0.0, z @ head["mlp_w1"].T + head["mlp_b1"])
    logits = h @ head["mlp_w2"].T + head["mlp_b2"]  # [n, C] or [n, 1]
    if head["multi"]:
        # stable softmax
        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        probs = e / e.sum(axis=1, keepdims=True)
        pred = probs.argmax(axis=1).astype(np.int64)
        conf = probs.max(axis=1)
        return pred, conf, probs
    logits = logits.reshape(-1)
    p1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    probs = np.stack([1.0 - p1, p1], axis=1)
    pred = (p1 > 0.5).astype(np.int64)
    conf = np.maximum(p1, 1.0 - p1)
    return pred, conf, probs


# ----------------------------------------------------------- residual bank --
def ensure_residuals(
    size: str,
    task: str,
    max_train: int,
    max_val: int,
    model,
    tok,
    layer: int,
    batch: int,
    device: str,
) -> dict:
    """Load or extract last-token residuals for one task; return arrays + y."""
    dirs = task_dirs(CACHE_DIR, size, task)
    cache_dir = dirs["cache"]
    keys = ["last_tr", "last_va", "ytr", "yva"]
    cached = load_cached_vectors(
        max_train, max_val, keys, cache_dir=cache_dir, layer=layer,
        batch=batch)
    if cached is not None:
        print(f"  [{task}] cache hit {cache_path(cache_dir, max_train, max_val, layer=layer, batch=batch)}")
        return {
            "last_tr": cached["last_tr"].astype(np.float32),
            "last_va": cached["last_va"].astype(np.float32),
            "ytr": np.asarray(cached["ytr"]),
            "yva": np.asarray(cached["yva"]),
        }

    # Full ARC cache may exist at official sizes — take a prefix if it matches
    # train/val order (ARC load uses fixed seeds).
    if task == "arc":
        full = load_cached_vectors(
            1117, 1165, keys, cache_dir=cache_dir, layer=layer, batch=batch)
        if full is not None and max_train <= 1117 and max_val <= 1165:
            print(f"  [{task}] slicing full ARC cache → t{max_train}/v{max_val}")
            # Use the cache's own y — do NOT re-load a shuffled subsample
            # (load_arc(max_n=...) permutes rows; full cache is unshuffled order).
            ytr = np.asarray(full["ytr"][:max_train])
            yva = np.asarray(full["yva"][:max_val])
            out = {
                "last_tr": full["last_tr"][:max_train].astype(np.float32),
                "last_va": full["last_va"][:max_val].astype(np.float32),
                "ytr": ytr,
                "yva": yva,
            }
            # persist pilot-sized cache for reruns
            cache_vectors(
                max_train, max_val,
                last_tr=out["last_tr"], last_va=out["last_va"],
                ytr=ytr, yva=yva,
                cache_dir=cache_dir, layer=layer, batch=batch,
            )
            return out

    print(f"  [{task}] extracting residuals (train={max_train} val={max_val})...")
    spec = get_task(task)
    rng = np.random.default_rng(0)
    train, val, _, mt, mv = spec.load(max_train, max_val, rng)
    assert mt == max_train and mv == max_val, (mt, mv, max_train, max_val)
    ytr = encode_labels(train)
    yva = encode_labels(val)
    texts_tr = [fmt_example(r) for r in train]
    texts_va = [fmt_example(r) for r in val]
    t0 = time.time()
    last_tr, _ = to_vecs(model, tok, texts_tr, layer, batch=batch, label=f"{task}.tr")
    last_va, _ = to_vecs(model, tok, texts_va, layer, batch=batch, label=f"{task}.va")
    print(f"  [{task}] extract done in {time.time() - t0:.0f}s")
    cache_vectors(
        max_train, max_val,
        last_tr=last_tr, last_va=last_va, ytr=ytr, yva=yva,
        cache_dir=cache_dir, layer=layer, batch=batch,
    )
    return {
        "last_tr": last_tr.astype(np.float32),
        "last_va": last_va.astype(np.float32),
        "ytr": ytr,
        "yva": yva,
    }


def fit_answer_head_local(Xtr, ytr, Xva, yva, n_classes: int, device: str) -> dict:
    """Fit a fresh MLP answer head when no artifact is on disk."""
    _, net, mu, sd = mlp_probe(
        Xtr, ytr, Xva[:1], yva[:1], seed=0, device=device, return_net=True,
        n_classes=n_classes)
    sd_ = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    return {
        "path": "(refit)",
        "n_classes": n_classes,
        "multi": n_classes > 2,
        "mlp_w1": sd_["0.weight"].astype(np.float32),
        "mlp_b1": sd_["0.bias"].astype(np.float32),
        "mlp_w2": sd_["2.weight"].astype(np.float32),
        "mlp_b2": sd_["2.bias"].astype(np.float32),
        "mlp_mu": np.asarray(mu, dtype=np.float32),
        "mlp_sd": np.asarray(sd, dtype=np.float32),
    }


# -------------------------------------------------------------- mahalanobis --
def fit_mahalanobis(Xtr: np.ndarray, reg: float = 1e-3):
    mu = Xtr.mean(0)
    Xc = Xtr - mu
    d = Xtr.shape[1]
    cov = (Xc.T @ Xc) / max(len(Xtr) - 1, 1)
    cov = cov + reg * np.eye(d, dtype=np.float32)
    # diagonal approximation is more stable in d=1024 with n_train~400
    diag = np.diag(cov).astype(np.float32) + reg
    return {"mu": mu.astype(np.float32), "diag": diag}


def mahalanobis_score(bank: dict, X: np.ndarray) -> np.ndarray:
    """Higher = closer to train manifold (negative half squared distance)."""
    z = (X - bank["mu"]) / np.sqrt(bank["diag"])
    return -0.5 * (z * z).sum(axis=1)


# ----------------------------------------------------------------- routers --
def route_naive_max_conf(confs: dict[str, np.ndarray]) -> np.ndarray:
    """confs[task] shape [n] → task index per row."""
    stack = np.stack([confs[t] for t in TASKS], axis=1)  # [n, 3]
    return stack.argmax(axis=1)


def route_mahalanobis(scores: dict[str, np.ndarray]) -> np.ndarray:
    stack = np.stack([scores[t] for t in TASKS], axis=1)
    return stack.argmax(axis=1)


def fit_task_id_linear(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    # sklearn ≥1.8 dropped multi_class=; multinomial is default for lbfgs C>2.
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
    clf.fit(sc.transform(Xtr), ytr)
    return sc, clf


def fit_task_id_mlp(Xtr, ytr, device: str):
    # 3-way task id
    dummy_x, dummy_y = Xtr[:1], ytr[:1]
    _, net, _, _ = mlp_probe(
        Xtr, ytr, dummy_x, dummy_y, seed=0, device=device, return_net=True,
        n_classes=3, hidden=64, epochs=80)
    return net


@torch.no_grad()
def predict_task_mlp(net, X) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = net._probe_mu, net._probe_sd
    dev = next(net.parameters()).device
    xv = torch.tensor((X - mu) / (sd + 1e-8), dtype=torch.float32, device=dev)
    logits = net(xv).detach().cpu().numpy()
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    probs = e / e.sum(axis=1, keepdims=True)
    return probs.argmax(axis=1), probs.max(axis=1)


# ----------------------------------------------------------------- metrics --
def answer_correct(task: str, pred: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """Boolean per-row: answer head prediction matches gold labels.

    Only meaningful when head label space matches the task (same n_classes
    and same convention). Cross-schema always False.
    """
    if n_classes != len(np.unique(np.concatenate([y, pred]))) and n_classes == 4:
        # ARC head: gold must be 0..3
        if y.max() > 3:
            return np.zeros(len(y), dtype=bool)
    if task in ("boolq", "ruletaker"):
        # binary heads share Yes/No convention
        if y.max() > 1:
            return np.zeros(len(y), dtype=bool)
        return pred.astype(np.int64) == y.astype(np.int64)
    if task == "arc":
        if y.max() > 3:
            return np.zeros(len(y), dtype=bool)
        return pred.astype(np.int64) == y.astype(np.int64)
    return pred.astype(np.int64) == y.astype(np.int64)


def evaluate_router(
    name: str,
    route_idx: np.ndarray,
    gold_task: np.ndarray,
    preds: dict[str, np.ndarray],
    y_by_task_row: np.ndarray,
    n_classes_of: dict[str, int],
    confs: dict[str, np.ndarray] | None = None,
    abstain_thresh: float | None = None,
    router_conf: np.ndarray | None = None,
) -> dict:
    """route_idx: [n] in 0..2; gold_task same; y_by_task_row gold answer labels."""
    n = len(route_idx)
    task_hit = route_idx == gold_task
    ans_ok = np.zeros(n, dtype=bool)
    for i in range(n):
        t = TASKS[int(route_idx[i])]
        # only score answer if schema matches gold task
        gold_t = TASKS[int(gold_task[i])]
        if t != gold_t:
            ans_ok[i] = False
            continue
        ans_ok[i] = bool(preds[t][i] == y_by_task_row[i])

    out = {
        "router": name,
        "n": int(n),
        "task_acc": float(task_hit.mean()),
        "answer_acc_strict": float(ans_ok.mean()),
        "answer_acc_given_correct_route": float(ans_ok[task_hit].mean()) if task_hit.any() else float("nan"),
    }
    # confusion: predicted task vs gold
    cm = confusion_matrix(gold_task, route_idx, labels=[0, 1, 2])
    out["confusion"] = cm.tolist()
    out["confusion_labels"] = list(TASKS)

    if abstain_thresh is not None and router_conf is not None:
        keep = router_conf >= abstain_thresh
        out["abstain_thresh"] = abstain_thresh
        out["keep_rate"] = float(keep.mean())
        if keep.any():
            out["task_acc_kept"] = float(task_hit[keep].mean())
            out["answer_acc_strict_kept"] = float(ans_ok[keep].mean())
        else:
            out["task_acc_kept"] = float("nan")
            out["answer_acc_strict_kept"] = float("nan")
    return out


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description="Multi-head residual routing pilot")
    ap.add_argument("--size", default="0.6B")
    ap.add_argument("--max-train", type=int, default=400)
    ap.add_argument("--max-val", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (auto if omit)")
    ap.add_argument("--layer", type=int, default=None,
                    help="residual/head layer to use (default: infer from cached heads)")
    ap.add_argument("--refit-heads", action="store_true",
                    help="force ALL answer heads to be refit on the pilot train "
                         "(uniform protocol; do not load cached head artifacts)")
    ap.add_argument("--out", default=None, help="results json path")
    args = ap.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    probe_dev = device if device != "cpu" else "cpu"

    print(f"model={args.size} device={device} train/task={args.max_train} "
          f"val/task={args.max_val}")

    # Load model only if any residual bank is missing
    model = tok = None
    layer = args.layer

    def need_model():
        nonlocal model, tok, layer
        if model is not None:
            return
        print(f"loading {resolve_model(args.size)} on {device}...")
        model, tok = load_model(args.size, device=device)
        if layer is None:
            layer = discover_layers(model) - 1
            print(f"  n_layers={discover_layers(model)} last_layer={layer}")

    # Infer layer from existing heads if not given explicitly
    if layer is None:
        for t in TASKS:
            for L in (27, 35, 42):
                p = find_head(args.size, t, L)
                if p:
                    layer = L
                    break
            if layer is not None:
                break

    banks = {}
    for t in TASKS:
        # try cache without model first
        dirs = task_dirs(CACHE_DIR, args.size, t)
        cached = load_cached_vectors(
            args.max_train, args.max_val,
            ["last_tr", "last_va", "ytr", "yva"],
            cache_dir=dirs["cache"], layer=layer if layer is not None else 27,
            batch=args.batch,
        )
        if cached is None and t == "arc":
            # try full ARC cache at the requested layer
            cached_full = load_cached_vectors(
                1117, 1165, ["last_tr", "last_va", "ytr", "yva"],
                cache_dir=dirs["cache"], layer=layer, batch=args.batch)
            if cached_full is not None and layer is None:
                layer = 27
        if cached is None:
            need_model()
            if layer is None:
                layer = discover_layers(model) - 1
            banks[t] = ensure_residuals(
                args.size, t, args.max_train, args.max_val,
                model, tok, layer, args.batch, device)
        else:
            if layer is None:
                layer = 27
            print(f"  [{t}] cache hit")
            banks[t] = {
                "last_tr": cached["last_tr"].astype(np.float32),
                "last_va": cached["last_va"].astype(np.float32),
                "ytr": np.asarray(cached["ytr"]),
                "yva": np.asarray(cached["yva"]),
            }
            # still allow ARC full-slice path if pilot sizes differ
            if (banks[t]["last_tr"].shape[0] != args.max_train
                    or banks[t]["last_va"].shape[0] != args.max_val):
                need_model()
                banks[t] = ensure_residuals(
                    args.size, t, args.max_train, args.max_val,
                    model, tok, layer, args.batch, device)

    # Free GPU memory if we only needed extraction
    if model is not None:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Load or refit answer heads
    heads = {}
    for t in TASKS:
        path = find_head(args.size, t, layer) if not args.refit_heads else None
        n_cls = 4 if t == "arc" else 2
        if path is not None:
            print(f"  [{t}] head {path}")
            heads[t] = load_head_npz(path)
            # dimension check
            if heads[t]["mlp_mu"].shape[0] != banks[t]["last_tr"].shape[1]:
                print(f"  [{t}] head dim mismatch — refitting on pilot train")
                heads[t] = fit_answer_head_local(
                    banks[t]["last_tr"], banks[t]["ytr"],
                    banks[t]["last_va"], banks[t]["yva"],
                    n_cls, probe_dev)
        else:
            why = "forced --refit-heads" if args.refit_heads else "no saved head"
            print(f"  [{t}] {why} — refitting on pilot train")
            heads[t] = fit_answer_head_local(
                banks[t]["last_tr"], banks[t]["ytr"],
                banks[t]["last_va"], banks[t]["yva"],
                n_cls, probe_dev)

    # Stack mixed val set
    X_parts, y_ans, y_task, task_names_row = [], [], [], []
    for ti, t in enumerate(TASKS):
        X_parts.append(banks[t]["last_va"])
        y_ans.append(banks[t]["yva"].astype(np.int64))
        y_task.append(np.full(len(banks[t]["yva"]), ti, dtype=np.int64))
        task_names_row.extend([t] * len(banks[t]["yva"]))
    Xva = np.concatenate(X_parts, axis=0)
    y_ans = np.concatenate(y_ans, axis=0)
    y_task = np.concatenate(y_task, axis=0)
    n = len(Xva)
    print(f"\nmixed val: n={n} ({args.max_val} × {len(TASKS)} tasks)")

    # Stack mixed train for task-ID / mahalanobis
    Xtr_parts, ytr_task = [], []
    for ti, t in enumerate(TASKS):
        Xtr_parts.append(banks[t]["last_tr"])
        ytr_task.append(np.full(len(banks[t]["last_tr"]), ti, dtype=np.int64))
    Xtr = np.concatenate(Xtr_parts, axis=0)
    ytr_task = np.concatenate(ytr_task, axis=0)

    # Per-head predictions on the full mixed val residual bank
    preds, confs, probs = {}, {}, {}
    for t in TASKS:
        p, c, pr = head_forward(heads[t], Xva)
        preds[t], confs[t], probs[t] = p, c, pr

    # --- cross-task answer matrix (only when schema matches) ---
    print("\n=== Cross-task answer accuracy (head × true task) ===")
    cross = {t_h: {} for t_h in TASKS}
    for t_h in TASKS:
        for ti, t_true in enumerate(TASKS):
            mask = y_task == ti
            n_cls_h = heads[t_h]["n_classes"]
            n_cls_t = 4 if t_true == "arc" else 2
            if n_cls_h != n_cls_t:
                cross[t_h][t_true] = None  # schema mismatch
                print(f"  head={t_h:10s} on {t_true:10s}:  n/a (schema)")
                continue
            acc = float((preds[t_h][mask] == y_ans[mask]).mean())
            mean_conf = float(confs[t_h][mask].mean())
            cross[t_h][t_true] = {"acc": acc, "mean_conf": mean_conf, "n": int(mask.sum())}
            mark = " ← ID" if t_h == t_true else ""
            print(f"  head={t_h:10s} on {t_true:10s}:  acc={acc:.3f}  "
                  f"mean_conf={mean_conf:.3f}{mark}")

    # mean answer-confidence by (head, true task) — even across schema
    print("\n=== Mean answer-confidence (head × true task) ===")
    conf_mat = {t_h: {} for t_h in TASKS}
    for t_h in TASKS:
        row = []
        for ti, t_true in enumerate(TASKS):
            mask = y_task == ti
            mc = float(confs[t_h][mask].mean())
            conf_mat[t_h][t_true] = mc
            row.append(f"{mc:.3f}")
        print(f"  {t_h:10s} → boolq/ruletaker/arc: " + "  ".join(row))

    # Mahalanobis banks
    maha = {t: fit_mahalanobis(banks[t]["last_tr"]) for t in TASKS}
    maha_scores = {t: mahalanobis_score(maha[t], Xva) for t in TASKS}

    # Task-ID probes
    print("\nfitting task-ID probes...")
    sc, clf = fit_task_id_linear(Xtr, ytr_task)
    tid_lin = clf.predict(sc.transform(Xva))
    tid_lin_proba = clf.predict_proba(sc.transform(Xva))
    tid_lin_conf = tid_lin_proba.max(axis=1)

    net = fit_task_id_mlp(Xtr, ytr_task, probe_dev)
    tid_mlp, tid_mlp_conf = predict_task_mlp(net, Xva)

    # Oracle / naive / maha routes
    oracle = y_task.copy()
    naive = route_naive_max_conf(confs)
    maha_route = route_mahalanobis(maha_scores)

    n_classes_of = {t: heads[t]["n_classes"] for t in TASKS}

    results = []
    for name, route, rconf in [
        ("oracle", oracle, np.ones(n)),
        ("naive_max_conf", naive, np.stack([confs[t] for t in TASKS], 1).max(1)),
        ("mahalanobis", maha_route,
         # softmax over maha scores as pseudo-conf
         (lambda s: (np.exp(s - s.max(1, keepdims=True))
                     / np.exp(s - s.max(1, keepdims=True)).sum(1, keepdims=True)).max(1))(
             np.stack([maha_scores[t] for t in TASKS], 1))),
        ("task_id_linear", tid_lin, tid_lin_conf),
        ("task_id_mlp", tid_mlp, tid_mlp_conf),
    ]:
        # rebuild per-row preds from chosen head for answer scoring
        # evaluate_router uses preds[task][i] when route picks that task
        ev = evaluate_router(
            name, route.astype(np.int64), y_task, preds, y_ans, n_classes_of,
            router_conf=rconf, abstain_thresh=0.7 if name != "oracle" else None,
        )
        results.append(ev)
        print(f"\n=== router: {name} ===")
        print(f"  task_acc={ev['task_acc']:.3f}  "
              f"answer_acc_strict={ev['answer_acc_strict']:.3f}  "
              f"answer|correct_route={ev['answer_acc_given_correct_route']:.3f}")
        if "keep_rate" in ev:
            print(f"  abstain@0.7 keep={ev['keep_rate']:.3f}  "
                  f"task_acc_kept={ev.get('task_acc_kept', float('nan')):.3f}  "
                  f"ans_kept={ev.get('answer_acc_strict_kept', float('nan')):.3f}")
        print("  confusion (rows=gold, cols=pred):")
        print("           ", "  ".join(f"{t[:6]:>6s}" for t in TASKS))
        for i, t in enumerate(TASKS):
            print(f"  {t:10s}", "  ".join(f"{c:6d}" for c in ev["confusion"][i]))

    # Per-true-task breakdown for naive vs task_id
    print("\n=== Per true-task: naive_max_conf vs task_id_linear ===")
    breakdown = {}
    for name, route in [("naive_max_conf", naive), ("task_id_linear", tid_lin)]:
        breakdown[name] = {}
        for ti, t in enumerate(TASKS):
            mask = y_task == ti
            r = route[mask]
            task_acc = float((r == ti).mean())
            # answer when routed to self
            self_mask = r == ti
            if self_mask.any():
                # indices into full array
                idx = np.where(mask)[0][self_mask]
                ans = float((preds[t][idx] == y_ans[idx]).mean())
            else:
                ans = float("nan")
            # mean conf of each head on this true task
            mean_c = {h: float(confs[h][mask].mean()) for h in TASKS}
            breakdown[name][t] = {
                "task_acc": task_acc,
                "answer_when_self": ans,
                "mean_conf_by_head": mean_c,
                "route_hist": {
                    TASKS[j]: int((r == j).sum()) for j in range(3)
                },
            }
            print(f"  [{name}] true={t:10s} route_acc={task_acc:.3f}  "
                  f"hist={breakdown[name][t]['route_hist']}  "
                  f"confs={ {k: round(v,3) for k,v in mean_c.items()} }")

    # Chance baselines
    chance_task = 1.0 / len(TASKS)
    oracle_ans = next(r for r in results if r["router"] == "oracle")["answer_acc_strict"]

    summary = {
        "meta": {
            "model": args.size,
            "model_id": resolve_model(args.size),
            "layer": layer,
            "max_train": args.max_train,
            "max_val": args.max_val,
            "n_mixed_val": n,
            "tasks": list(TASKS),
            "heads": {t: heads[t]["path"] for t in TASKS},
            "head_protocol": "refit-on-pilot-train" if args.refit_heads
                             else "cached-or-refit",
            "chance_task_acc": chance_task,
        },
        "cross_task_answer": cross,
        "confidence_matrix": conf_mat,
        "routers": results,
        "per_task_breakdown": breakdown,
        "oracle_answer_acc": oracle_ans,
    }

    out_path = args.out or os.path.join(
        RESULTS_DIR, f"multihead_route_{model_slug(args.size)}_t{args.max_train}_v{args.max_val}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\n[wrote] {out_path}")

    print("\n========== HEADLINE ==========")
    print(f"chance task routing: {chance_task:.3f}")
    for r in results:
        print(f"  {r['router']:18s}  task={r['task_acc']:.3f}  "
              f"ans_strict={r['answer_acc_strict']:.3f}")
    print(f"oracle answer (correct head always): {oracle_ans:.3f}")
    print("==============================")
    return summary


if __name__ == "__main__":
    main()
