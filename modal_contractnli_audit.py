"""Modal audit for ContractNLI closed-set autoregressive label scoring.

Runs a small 0.6B dev subset with several label-scoring variants so the
surprisingly weak full-label baseline can be checked before scaling models.
Results are saved under bench-results/contractnli/<model>/audit/.
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
_KEEP = frozenset({"modal_contractnli_audit.py", "common.py"})


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

app = modal.App("vector-is-the-answer-contractnli-audit")
DATA_MOUNT = "/data"
RESULT_VOLUME = modal.Volume.from_name("bench-results", create_if_missing=True)
WEIGHTS_MOUNT = "/weights"
WEIGHTS_VOLUME = modal.Volume.from_name("model-weights", create_if_missing=True)
DATA_URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
LABELS = ("Entailment", "Contradiction", "NotMentioned")


def _remote(size: str, limit_docs: int, max_length: int, batch: int):
    os.chdir("/root/contractnli")
    sys.path.insert(0, "/root/contractnli")
    os.environ["HF_HOME"] = os.path.join(DATA_MOUNT, "hf_home")

    import numpy as np
    import torch
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score)
    from common import load_model, model_slug

    work = Path("/tmp/contractnli-audit")
    work.mkdir(parents=True, exist_ok=True)
    archive = work / "contract-nli.zip"
    root = work / "contract-nli"
    if not (root / "dev.json").exists():
        if not archive.exists():
            print(f"[data] downloading {DATA_URL}", flush=True)
            urllib.request.urlretrieve(DATA_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(work)

    data = json.loads((root / "dev.json").read_text())
    docs = data["documents"][:limit_docs]
    label_map = {name: i for i, name in enumerate(LABELS)}
    rows = []
    for doc in docs:
        anns = doc["annotation_sets"][0]["annotations"]
        for key, meta in data["labels"].items():
            rows.append({
                "contract": doc["text"],
                "hypothesis": meta["hypothesis"],
                "y": label_map[anns[key]["choice"]],
            })

    y = np.asarray([row["y"] for row in rows], dtype=np.int64)
    print(f"[data] docs={len(docs)} rows={len(rows)} "
          f"gold_counts={np.bincount(y, minlength=3).tolist()}", flush=True)

    weights_dir = os.path.join(WEIGHTS_MOUNT, model_slug(size))
    print(f"[model] loading Qwen/Qwen3-{size} from {weights_dir}", flush=True)
    model, tok = load_model(size, dtype=torch.float16, device="cuda",
                            weights_dir=weights_dir)
    tok.truncation_side = "left"
    dev = next(model.parameters()).device
    print(f"[model] device={dev}", flush=True)

    def prompt(row, instructed=False):
        stem = (f"Contract:\n{row['contract']}\n\n"
                f"Hypothesis: {row['hypothesis']}\nDecision:")
        if instructed:
            return ("Classify the hypothesis using exactly one of these labels: "
                    "Entailment, Contradiction, NotMentioned.\n\n" + stem)
        return stem

    label_ids = [tok(" " + label, add_special_tokens=False)["input_ids"]
                 for label in LABELS]
    print("[tokens]", flush=True)
    for label, ids in zip(LABELS, label_ids):
        print(f"  {label} {ids} {tok.convert_ids_to_tokens(ids)}", flush=True)

    def score_full(texts, normalize=False):
        scores = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        for label_i, suffix in enumerate(label_ids):
            for start in range(0, len(order), batch):
                idx = order[start:start + batch]
                seqs, starts = [], []
                for i in idx:
                    base = tok(texts[i], add_special_tokens=False)["input_ids"]
                    ids = (base + suffix)[-max_length:]
                    seqs.append(torch.tensor(ids, dtype=torch.long))
                    starts.append(len(ids) - len(suffix))
                pad = torch.nn.utils.rnn.pad_sequence(
                    seqs, batch_first=True, padding_value=tok.pad_token_id
                ).to(dev)
                attn = (pad != tok.pad_token_id).long()
                with torch.no_grad():
                    logits = model(input_ids=pad, attention_mask=attn).logits
                vals = []
                for row_i, start_pos in enumerate(starts):
                    positions = torch.arange(
                        start_pos - 1, start_pos - 1 + len(suffix), device=dev)
                    target = pad[row_i, start_pos:start_pos + len(suffix)]
                    value = torch.log_softmax(logits[row_i, positions].float(), -1)[
                        torch.arange(len(suffix), device=dev), target].sum()
                    if normalize:
                        value = value / len(suffix)
                    vals.append(value)
                scores[idx, label_i] = torch.stack(vals).cpu().numpy()
            print(f"[full{'_mean' if normalize else '_raw'}] "
                  f"label {LABELS[label_i]} done", flush=True)
        return scores

    def last_real(attn):
        return attn.shape[1] - 1 - attn.flip(1).argmax(1)

    def score_first(texts):
        scores = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            enc = tok([texts[i] for i in idx], return_tensors="pt",
                      padding=True, truncation=True,
                      max_length=max_length).to(dev)
            with torch.no_grad():
                logits = model(**enc).logits
            last = last_real(enc["attention_mask"]).to(dev)
            rows_i = torch.arange(len(idx), device=dev)
            first_logp = torch.log_softmax(logits[rows_i, last].float(), -1)
            for label_i, suffix in enumerate(label_ids):
                scores[idx, label_i] = first_logp[:, suffix[0]].cpu().numpy()
        return scores

    def metrics(pred):
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "predicted_counts": np.bincount(pred, minlength=3).tolist(),
            "confusion_matrix_rows_gold": confusion_matrix(
                y, pred, labels=list(range(len(LABELS)))).tolist(),
        }

    texts = [prompt(row) for row in rows]
    instructed_texts = [prompt(row, instructed=True) for row in rows]
    variants = {}
    t0 = time.time()

    raw = score_full(texts, normalize=False)
    variants["full_raw"] = metrics(raw.argmax(1))
    variants["full_raw_score_summary"] = {
        "mean": raw.mean(0).round(4).tolist(),
        "min": raw.min(0).round(4).tolist(),
        "max": raw.max(0).round(4).tolist(),
    }

    mean = score_full(texts, normalize=True)
    variants["full_mean"] = metrics(mean.argmax(1))
    variants["first_token"] = metrics(score_first(texts).argmax(1))
    variants["first_token_instructed"] = metrics(
        score_first(instructed_texts).argmax(1))
    instructed_mean = score_full(instructed_texts, normalize=True)
    variants["full_mean_instructed"] = metrics(instructed_mean.argmax(1))

    result = {
        "dataset": "ContractNLI",
        "model": f"Qwen/Qwen3-{size}",
        "size": size,
        "split": "dev",
        "docs": len(docs),
        "rows": len(rows),
        "labels": list(LABELS),
        "max_length": max_length,
        "batch": batch,
        "label_token_ids": label_ids,
        "variants": variants,
        "elapsed_sec": time.time() - t0,
        "protocol": "Small dev audit of full multi-token label likelihood, "
                     "length normalization, first-token scoring, and an "
                     "explicitly instructed prompt.",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir = Path(DATA_MOUNT) / "contractnli" / model_slug(size) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "loop_audit_small.json").write_text(json.dumps(result, indent=2))
    RESULT_VOLUME.commit()
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(image=image, volumes={DATA_MOUNT: RESULT_VOLUME,
                                     WEIGHTS_MOUNT: WEIGHTS_VOLUME},
              gpu="A100-80GB", timeout=2 * 60 * 60, cpu=16.0,
              memory=65536)
def run_audit(size: str = "0.6B", limit_docs: int = 8,
              max_length: int = 2048, batch: int = 8):
    return _remote(size, limit_docs, max_length, batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="0.6B", choices=("0.6B", "4B", "8B"))
    ap.add_argument("--limit-docs", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    with app.run():
        out = run_audit.remote(args.size, args.limit_docs,
                               args.max_length, args.batch)
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
