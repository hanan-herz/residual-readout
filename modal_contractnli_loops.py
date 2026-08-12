"""Modal ContractNLI loop-only comparison.

Runs two improved closed-set autoregressive baselines without recomputing
residual banks or heads:
  - an explicit task instruction with length-normalized label likelihood;
  - an 8-shot balanced labeled prompt with length-normalized likelihood.

Results are written separately under:
  contractnli/<model>/loops/loop_variants.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
_KEEP = frozenset({"modal_contractnli_loops.py", "common.py"})


def _ignore(path) -> bool:
    parts = Path(path).parts
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

app = modal.App("vector-is-the-answer-contractnli-loops")
DATA_MOUNT = "/data"
RESULT_VOLUME = modal.Volume.from_name("bench-results", create_if_missing=True)
WEIGHTS_MOUNT = "/weights"
WEIGHTS_VOLUME = modal.Volume.from_name("model-weights", create_if_missing=True)
DATA_URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
LABELS = ("Entailment", "Contradiction", "NotMentioned")
INSTRUCTION = (
    "Classify the hypothesis using exactly one of these labels: "
    "Entailment, Contradiction, NotMentioned. Entailment means the contract "
    "supports the hypothesis; Contradiction means the contract conflicts with "
    "it; NotMentioned means the contract does not address it. Output only the "
    "label.\n\n"
)


def _remote(size: str, max_length: int, batch: int, fewshot_k: int):
    os.chdir("/root/contractnli")
    sys.path.insert(0, "/root/contractnli")
    os.environ["HF_HOME"] = os.path.join(DATA_MOUNT, "hf_home")

    import numpy as np
    import torch
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score)
    from common import load_model, model_slug

    work = Path("/tmp/contractnli-loops")
    work.mkdir(parents=True, exist_ok=True)
    archive = work / "contract-nli.zip"
    root = work / "contract-nli"
    if not (root / "train.json").exists():
        if not archive.exists():
            print(f"[data] downloading {DATA_URL}", flush=True)
            urllib.request.urlretrieve(DATA_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(work)

    label_map = {name: i for i, name in enumerate(LABELS)}

    def load_rows(split: str):
        data = json.loads((root / f"{split}.json").read_text())
        rows = []
        for doc in data["documents"]:
            anns = doc["annotation_sets"][0]["annotations"]
            for key, meta in data["labels"].items():
                rows.append({
                    "contract": doc["text"],
                    "hypothesis": meta["hypothesis"],
                    "y": label_map[anns[key]["choice"]],
                    "doc_id": str(doc["id"]),
                    "key": key,
                })
        return rows

    train = load_rows("train")
    dev_rows = load_rows("dev")
    test_rows = load_rows("test")
    print(f"[data] train={len(train)} dev={len(dev_rows)} test={len(test_rows)}",
          flush=True)

    def stem(row):
        return (f"Contract:\n{row['contract']}\n\n"
                f"Hypothesis: {row['hypothesis']}\nDecision:")

    rng = np.random.default_rng(0)
    by_class = {i: [] for i in range(len(LABELS))}
    for row in train:
        by_class[row["y"]].append(row)
    for rows in by_class.values():
        rng.shuffle(rows)
    examples = []
    while len(examples) < fewshot_k:
        progressed = False
        for class_i in range(len(LABELS)):
            if len(examples) >= fewshot_k:
                break
            if by_class[class_i]:
                row = by_class[class_i].pop()
                examples.append(row)
                progressed = True
        if not progressed:
            break
    fewshot_block = "\n\n".join(
        stem(row) + " " + LABELS[row["y"]] for row in examples
    )
    print(f"[fewshot] selected={len(examples)} counts="
          f"{[sum(r['y'] == i for r in examples) for i in range(3)]}",
          flush=True)

    weights_dir = os.path.join(WEIGHTS_MOUNT, model_slug(size))
    print(f"[model] loading Qwen/Qwen3-{size} from {weights_dir}", flush=True)
    model, tok = load_model(size, dtype=torch.float16, device="cuda",
                            weights_dir=weights_dir)
    tok.truncation_side = "left"
    dev = model.device

    label_token_ids = [tok(" " + label, add_special_tokens=False)["input_ids"]
                       for label in LABELS]
    print("[tokens] " + ", ".join(
        f"{label}={ids}" for label, ids in zip(LABELS, label_token_ids)),
          flush=True)

    def score_full(texts, tag):
        """Score all complete labels, normalized by label token count."""
        scores = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        max_label_len = max(map(len, label_token_ids))
        base_limit = max_length - max_label_len
        for label_i, suffix in enumerate(label_token_ids):
            for start in range(0, len(order), batch):
                idx = order[start:start + batch]
                seqs, suffix_starts = [], []
                for i in idx:
                    base = tok(texts[i], add_special_tokens=False,
                               truncation=True, max_length=base_limit)["input_ids"]
                    ids = base + suffix
                    seqs.append(torch.tensor(ids, dtype=torch.long))
                    suffix_starts.append(len(base))
                pad = torch.nn.utils.rnn.pad_sequence(
                    seqs, batch_first=True, padding_value=tok.pad_token_id
                ).to(dev)
                attn = (pad != tok.pad_token_id).long()
                with torch.no_grad():
                    logits = model(input_ids=pad, attention_mask=attn).logits
                vals = []
                for row_i, start_pos in enumerate(suffix_starts):
                    positions = torch.arange(
                        start_pos - 1, start_pos - 1 + len(suffix), device=dev)
                    target = pad[row_i, start_pos:start_pos + len(suffix)]
                    value = torch.log_softmax(logits[row_i, positions].float(), -1)[
                        torch.arange(len(suffix), device=dev), target].sum()
                    vals.append(value / len(suffix))
                scores[idx, label_i] = torch.stack(vals).cpu().numpy()
            print(f"  [{tag}] {LABELS[label_i]} done", flush=True)
        return scores

    def metric_block(y_true, score):
        pred = score.argmax(1)
        return {
            "accuracy": float(accuracy_score(y_true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro")),
            "predicted_counts": np.bincount(pred, minlength=3).tolist(),
            "confusion_matrix_rows_gold": confusion_matrix(
                y_true, pred, labels=list(range(3))).tolist(),
            "score_mean": score.mean(0).round(5).tolist(),
        }

    variants = {
        "instructed_length_normalized": {
            "prompt_prefix": INSTRUCTION,
            "fewshot_k": 0,
        },
        "balanced_fewshot_length_normalized": {
            "prompt_prefix": fewshot_block + "\n\n",
            "fewshot_k": len(examples),
        },
    }
    result_variants = {}
    t0 = time.time()
    for name, spec in variants.items():
        print(f"[variant] {name}", flush=True)
        dev_texts = [spec["prompt_prefix"] + stem(row) for row in dev_rows]
        test_texts = [spec["prompt_prefix"] + stem(row) for row in test_rows]
        dev_y = np.asarray([r["y"] for r in dev_rows], dtype=np.int64)
        test_y = np.asarray([r["y"] for r in test_rows], dtype=np.int64)
        scores_dev = score_full(dev_texts, name + ".dev")
        scores_test = score_full(test_texts, name + ".test")
        result_variants[name] = {
            "dev": metric_block(dev_y, scores_dev),
            "test": metric_block(test_y, scores_test),
            "label_token_lengths": [len(ids) for ids in label_token_ids],
            "length_normalized": True,
            "fewshot_k": spec["fewshot_k"],
        }

    result = {
        "dataset": "ContractNLI",
        "model": f"Qwen/Qwen3-{size}",
        "size": size,
        "split": {
            "train_docs": 423, "dev_docs": 61, "test_docs": 123,
            "train_rows": len(train), "dev_rows": len(dev_rows),
            "test_rows": len(test_rows),
        },
        "labels": list(LABELS),
        "max_length": max_length,
        "batch": batch,
        "fewshot_k_requested": fewshot_k,
        "fewshot_examples": [
            {"doc_id": row["doc_id"], "hypothesis": row["hypothesis"],
             "label": LABELS[row["y"]]}
            for row in examples
        ],
        "variants": result_variants,
        "timing_sec": time.time() - t0,
        "protocol": (
            "Loop-only rerun. Both variants score complete multi-token labels "
            "with mean log probability. The instructed arm adds explicit "
            "ContractNLI label definitions; the few-shot arm uses balanced "
            "train-only demonstrations. Dev is reported for comparison and "
            "test is held out."
        ),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir = Path(DATA_MOUNT) / "contractnli" / model_slug(size) / "loops"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "loop_variants.json").write_text(json.dumps(result, indent=2))
    (out_dir / "metadata.json").write_text(json.dumps({
        "dataset": "ContractNLI", "labels": list(LABELS),
        "variants": list(result_variants), "fewshot_k": len(examples),
        "max_length": max_length,
    }, indent=2))
    RESULT_VOLUME.commit()
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(image=image, volumes={DATA_MOUNT: RESULT_VOLUME,
                                     WEIGHTS_MOUNT: WEIGHTS_VOLUME},
              gpu="A100-80GB", timeout=12 * 60 * 60, cpu=16.0,
              memory=65536)
def run_loops(size: str = "0.6B", max_length: int = 2048,
              batch: int = 8, fewshot_k: int = 8):
    return _remote(size, max_length, batch, fewshot_k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="0.6B", choices=("0.6B", "4B", "8B"))
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fewshot-k", type=int, default=8)
    args = ap.parse_args()
    with app.run():
        out = run_loops.remote(args.size, args.max_length, args.batch,
                               args.fewshot_k)
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
