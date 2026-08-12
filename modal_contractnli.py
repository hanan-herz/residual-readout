"""Modal ContractNLI residual-readout versus autoregressive scoring.

Each row is one contract plus one of ContractNLI's 17 hypotheses. Residual
heads are fit on train contracts only; development selects the reported tap,
and test is held out. Residual banks and deployable head artifacts are saved
under the bench-results volume for later multi-head experiments.

Full run:
  modal run --detach modal_contractnli.py --size 0.6B \
    --train-docs 423 --dev-docs 61 --test-docs 123 --max-length 2048
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
_KEEP = frozenset({"modal_contractnli.py", "common.py", "probe_main.py"})


def _ignore(p) -> bool:
    parts = Path(p).parts
    return not parts or not (len(parts) == 1 and parts[0] in _KEEP)


image = (
    modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu22.04",
                              add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda",
          "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .pip_install("torch", "transformers>=4.46", "scikit-learn", "numpy",
                 "sentencepiece", "protobuf", "accelerate")
    .add_local_dir(HERE, "/root/contractnli", ignore=_ignore)
)

app = modal.App("vector-is-the-answer-contractnli")
DATA_MOUNT = "/data"
RESULT_VOLUME = modal.Volume.from_name("bench-results", create_if_missing=True)
WEIGHTS_MOUNT = "/weights"
WEIGHTS_VOLUME = modal.Volume.from_name("model-weights", create_if_missing=True)
DATA_URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
LABELS = ("Entailment", "Contradiction", "NotMentioned")


def _remote(size: str, train_docs: int, dev_docs: int, test_docs: int,
            max_length: int, batch: int):
    os.chdir("/root/contractnli")
    sys.path.insert(0, "/root/contractnli")
    os.environ["HF_HOME"] = os.path.join(DATA_MOUNT, "hf_home")

    import urllib.request
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    from common import load_model, model_slug, n_layers
    from probe_main import mlp_probe

    work = Path("/tmp/contractnli")
    work.mkdir(parents=True, exist_ok=True)
    archive = work / "contract-nli.zip"
    root = work / "contract-nli"
    if not (root / "train.json").exists():
        if not archive.exists():
            print(f"[data] downloading {DATA_URL}", flush=True)
            urllib.request.urlretrieve(DATA_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(work)

    rng = np.random.default_rng(0)
    label_map = {name: i for i, name in enumerate(LABELS)}

    def load_rows(split: str, n_docs: int | None = None):
        data = json.load(open(root / f"{split}.json"))
        docs = data["documents"]
        if n_docs is not None and n_docs < len(docs):
            if split == "train":
                idx = np.sort(rng.choice(len(docs), n_docs, replace=False))
                docs = [docs[int(i)] for i in idx]
            else:
                docs = docs[:n_docs]
        rows = []
        for doc in docs:
            anns = doc["annotation_sets"][0]["annotations"]
            for key, meta in data["labels"].items():
                rows.append({
                    "contract": doc["text"],
                    "hypothesis": meta["hypothesis"],
                    "y": label_map[anns[key]["choice"]],
                    "doc_id": str(doc["id"]),
                    "key": key,
                })
        return rows, docs

    train, train_docs_used = load_rows("train", train_docs)
    dev_rows, dev_docs_used = load_rows("dev", dev_docs)
    test_rows, test_docs_used = load_rows("test", test_docs)

    def prompt(row):
        return (f"Contract:\n{row['contract']}\n\n"
                f"Hypothesis: {row['hypothesis']}\nDecision:")

    row_sets = {"train": train, "dev": dev_rows, "test": test_rows}
    texts = {name: [prompt(r) for r in rows]
             for name, rows in row_sets.items()}
    ys = {name: np.asarray([r["y"] for r in rows], dtype=np.int64)
          for name, rows in row_sets.items()}

    weights_dir = os.path.join(WEIGHTS_MOUNT, model_slug(size))
    print(f"[model] loading Qwen/Qwen3-{size} from {weights_dir}", flush=True)
    model, tok = load_model(size, dtype=torch.float16, device="cuda",
                            weights_dir=weights_dir)
    tok.truncation_side = "left"
    depth = n_layers(model)
    taps = (1, depth - 1)
    dev = model.device

    def extract(text_list, label):
        out = {layer: [] for layer in taps}
        for start in range(0, len(text_list), batch):
            sub = text_list[start:start + batch]
            enc = tok(sub, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(dev)
            bufs = {layer: [] for layer in taps}

            def make_hook(layer):
                def hook(_module, _inputs, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    bufs[layer].append(output.detach().float().cpu())
                return hook

            hooks = [model.model.layers[layer].register_forward_hook(
                make_hook(layer)) for layer in taps]
            with torch.no_grad():
                model(**enc)
            for hook in hooks:
                hook.remove()
            last = enc["attention_mask"].sum(1).cpu() - 1
            rows_i = torch.arange(len(sub))
            for layer in taps:
                out[layer].append(bufs[layer][0][rows_i, last])
            done = min(start + batch, len(text_list))
            if start == 0 or done == len(text_list) or start // batch % 50 == 49:
                print(f"  [{label}] {done}/{len(text_list)}", flush=True)
        return {layer: torch.cat(out[layer]).numpy().astype(np.float32)
                for layer in taps}

    t0 = time.time()
    banks = {name: extract(texts[name], f"residual.{name}")
             for name in ("train", "dev", "test")}
    extraction_sec = time.time() - t0

    def metrics(y_true, pred):
        return {
            "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro")),
        }

    def mlp_predict(net, X, mu, sd):
        xt = torch.tensor((X - mu) / (sd + 1e-8), dtype=torch.float32,
                          device="cuda")
        with torch.no_grad():
            logits = net(xt).detach().cpu().numpy()
        return logits.argmax(1).astype(np.int64)

    metrics_out = {}
    heads_dir = Path(DATA_MOUNT) / "contractnli" / model_slug(size) / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(DATA_MOUNT) / "contractnli" / model_slug(size) / "residuals"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for layer in taps:
        Xtr, Xdev, Xtest = (banks[name][layer]
                            for name in ("train", "dev", "test"))
        ytr, ydev, ytest = (ys[name] for name in ("train", "dev", "test"))
        np.savez(cache_dir / f"residual_l{layer}.npz",
                 X_train=Xtr, y_train=ytr, X_dev=Xdev, y_dev=ydev,
                 X_test=Xtest, y_test=ytest,
                 train_doc_id=np.asarray([r["doc_id"] for r in train]),
                 train_hypothesis=np.asarray([r["key"] for r in train]),
                 dev_doc_id=np.asarray([r["doc_id"] for r in dev_rows]),
                 dev_hypothesis=np.asarray([r["key"] for r in dev_rows]),
                 test_doc_id=np.asarray([r["doc_id"] for r in test_rows]),
                 test_hypothesis=np.asarray([r["key"] for r in test_rows]))

        scaler = StandardScaler().fit(Xtr)
        linear = LogisticRegression(max_iter=3000, solver="lbfgs").fit(
            scaler.transform(Xtr), ytr)
        linear_metrics = {
            "dev": metrics(ydev, linear.predict(scaler.transform(Xdev))),
            "test": metrics(ytest, linear.predict(scaler.transform(Xtest))),
        }

        print(f"  [head] fitting MLP L{layer}", flush=True)
        dev_pred, net, mu, sd = mlp_probe(
            Xtr, ytr, Xdev, ydev, seed=0, device="cuda", return_net=True,
            n_classes=3, epochs=100)
        test_pred = mlp_predict(net, Xtest, mu, sd)
        mlp_metrics = {
            "dev": metrics(ydev, np.asarray(dev_pred, dtype=np.int64)),
            "test": metrics(ytest, test_pred),
        }
        state = {k: v.detach().cpu().numpy().astype(np.float32)
                 for k, v in net.state_dict().items()}
        np.savez(heads_dir / f"head_l{layer}.npz",
                 mlp_w1=state["0.weight"], mlp_b1=state["0.bias"],
                 mlp_w2=state["2.weight"], mlp_b2=state["2.bias"],
                 mlp_mu=np.asarray(mu, dtype=np.float32),
                 mlp_sd=np.asarray(sd, dtype=np.float32),
                 n_classes=np.asarray(3, dtype=np.int64))
        metrics_out[f"layer_{layer}"] = {
            "linear": linear_metrics,
            "mlp": mlp_metrics,
        }

    # Score complete candidate label strings, not only their first token.
    # Each candidate is a separate context forward; this is a clean closed-set
    # autoregressive scoring baseline for a three-class, multi-token label set.
    label_token_ids = [tok(" " + label, add_special_tokens=False)["input_ids"]
                       for label in LABELS]

    def loop_predict(text_list, label):
        preds = np.empty(len(text_list), dtype=np.int64)
        order = sorted(range(len(text_list)), key=lambda i: -len(text_list[i]))
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            seqs = []
            suffix_starts = []
            for i in idx:
                base = tok(text_list[i], add_special_tokens=False)["input_ids"]
                suffix = label_token_ids[label]
                ids = (base + suffix)[-max_length:]
                seqs.append(torch.tensor(ids, dtype=torch.long))
                suffix_starts.append(len(ids) - len(suffix))
            pad = torch.nn.utils.rnn.pad_sequence(
                seqs, batch_first=True, padding_value=tok.pad_token_id)
            attn = (pad != tok.pad_token_id).long()
            pad, attn = pad.to(dev), attn.to(dev)
            with torch.no_grad():
                logits = model(input_ids=pad, attention_mask=attn).logits
            suffix = label_token_ids[label]
            scores = []
            for row, start_pos in zip(range(len(idx)), suffix_starts):
                positions = torch.arange(start_pos - 1,
                                          start_pos - 1 + len(suffix),
                                          device=dev)
                target = pad[row, start_pos:start_pos + len(suffix)]
                scores.append(logits[row, positions].float().log_softmax(-1)
                              [torch.arange(len(suffix), device=dev), target].sum())
            # Caller collects one score vector per label; return this batch.
            done = min(start + batch, len(order))
            if label == 0 and (start == 0 or done == len(order)
                               or start // batch % 50 == 49):
                print(f"  [loop] {done}/{len(order)}", flush=True)
            out = torch.stack(scores).cpu().numpy()
            yield idx, out

    def loop_scores(text_list):
        all_scores = np.zeros((len(text_list), len(LABELS)), dtype=np.float32)
        for label in range(len(LABELS)):
            for idx, scores in loop_predict(text_list, label):
                all_scores[idx, label] = scores
        return all_scores.argmax(1).astype(np.int64)

    loop_out = {}
    t1 = time.time()
    for name in ("dev", "test"):
        pred = loop_scores(texts[name])
        loop_out[name] = metrics(ys[name], pred)
    loop_sec = time.time() - t1

    layer_scores = {int(k.split("_")[-1]): v for k, v in metrics_out.items()}
    selected_layer = max(taps, key=lambda layer: (
        layer_scores[layer]["mlp"]["dev"]["macro_f1"],
        layer_scores[layer]["mlp"]["dev"]["accuracy"]))
    result = {
        "dataset": "ContractNLI",
        "split": {"train_docs": len(train_docs_used),
                  "dev_docs": len(dev_docs_used),
                  "test_docs": len(test_docs_used),
                  "train_rows": len(train), "dev_rows": len(dev_rows),
                  "test_rows": len(test_rows)},
        "model": f"Qwen/Qwen3-{size}", "size": size, "depth": depth,
        "taps": list(taps), "selected_layer_by_dev_mlp_f1": selected_layer,
        "max_length": max_length, "batch": batch,
        "label_order": list(LABELS),
        "train_class_counts": np.bincount(ys["train"], minlength=3).tolist(),
        "dev_class_counts": np.bincount(ys["dev"], minlength=3).tolist(),
        "test_class_counts": np.bincount(ys["test"], minlength=3).tolist(),
        "metrics": metrics_out, "loop_full_label": loop_out,
        "timing_sec": {"residual_extraction": extraction_sec,
                        "loop_full_label": loop_sec},
        "artifacts": {
            "root": str(Path("contractnli") / model_slug(size)),
            "residuals": "residuals/residual_l{layer}.npz",
            "heads": "heads/head_l{layer}.npz",
        },
        "protocol": "Residual heads fit on train contracts only; layer selected on dev; test is held out. Loop scores complete multi-token candidate labels on the same left-truncated prompt.",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir = Path(DATA_MOUNT) / "contractnli" / model_slug(size)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    (out_dir / "metadata.json").write_text(json.dumps({
        "dataset": "ContractNLI", "labels": list(LABELS),
        "max_length": max_length, "taps": list(taps),
        "selected_layer": selected_layer,
    }, indent=2))
    RESULT_VOLUME.commit()
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(image=image, volumes={DATA_MOUNT: RESULT_VOLUME,
                                     WEIGHTS_MOUNT: WEIGHTS_VOLUME},
              gpu="A100-80GB", timeout=12 * 60 * 60, cpu=16.0,
              memory=65536)
def run_contractnli(size: str = "0.6B", train_docs: int = 423,
                    dev_docs: int = 61, test_docs: int = 123,
                    max_length: int = 2048, batch: int = 8):
    return _remote(size, train_docs, dev_docs, test_docs, max_length, batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="0.6B", choices=("0.6B", "4B", "8B"))
    ap.add_argument("--train-docs", type=int, default=423)
    ap.add_argument("--dev-docs", type=int, default=61)
    ap.add_argument("--test-docs", type=int, default=123)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    with app.run():
        out = run_contractnli.remote(
            args.size, args.train_docs, args.dev_docs, args.test_docs,
            args.max_length, args.batch)
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
