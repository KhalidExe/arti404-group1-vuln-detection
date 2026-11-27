"""Method 2: fine-tuned CodeBERT (microsoft/codebert-base) for vulnerable / safe classification.

Owner: Khalid Waleed Alhilal

Setup (master plan, section 3 and H01 section 5):
- RoBERTa-style encoder (125M parameters) + linear classification head with 2 outputs.
- Input: cleaned C function, tokenized to at most 512 sub-word tokens (longer functions are cut).
- AdamW, linear learning-rate schedule with 10% warm-up, weight decay 0.01, bf16 autocast.
- Effective batch size 32 (micro-batch x gradient accumulation), dynamic padding.
- Dev macro F1 is computed after every epoch; the best epoch is kept (early stopping, patience 2).
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from transformers import set_seed as hf_set_seed

MODEL_NAME = "microsoft/codebert-base"
MAX_LEN = 512
EFFECTIVE_BATCH = 32
MICRO_BATCH = 16          # 16 x 2 accumulation steps = 32; fits in 16 GB with bf16 at 512 tokens
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10


def set_all_seeds(seed: int) -> None:
    """Seed python, numpy, torch (CPU + CUDA) and transformers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def get_device() -> torch.device:
    """CUDA if available, otherwise CPU (CPU is only practical for tiny tests)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(name: str = MODEL_NAME):
    """CodeBERT's byte-level BPE tokenizer."""
    return AutoTokenizer.from_pretrained(name)


def token_lengths(tokenizer, codes: list[str]) -> np.ndarray:
    """Number of CodeBERT tokens per function, including <s> and </s>, without truncation.

    Used to count how many functions are cut at MAX_LEN.
    """
    enc = tokenizer(list(codes), add_special_tokens=True, truncation=False)
    return np.array([len(ids) for ids in enc["input_ids"]])


class _EncodedDataset(torch.utils.data.Dataset):
    """Pre-tokenized functions (no padding; the collator pads each batch to its longest item)."""

    def __init__(self, tokenizer, codes: list[str], labels: list[int] | None = None):
        self.enc = tokenizer(list(codes), truncation=True, max_length=MAX_LEN)
        self.labels = None if labels is None else [int(x) for x in labels]

    def __len__(self) -> int:
        return len(self.enc["input_ids"])

    def __getitem__(self, i: int) -> dict:
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


def _dev_scores(y_true, y_pred) -> dict:
    """Metrics used for model selection on dev (same definitions as evaluation.compute_metrics)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "vuln_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


@torch.no_grad()
def predict_proba(model, tokenizer, codes: list[str], batch_size: int = 32) -> np.ndarray:
    """Return P(vulnerable) for every function, shape (n,).

    Functions are sorted by length so each batch has little padding, then put back in the
    original order.
    """
    codes = list(codes)
    if len(codes) == 0:
        return np.zeros(0, dtype=np.float32)
    device = next(model.parameters()).device
    model.eval()
    ds = _EncodedDataset(tokenizer, codes)
    order = np.argsort([len(x) for x in ds.enc["input_ids"]])
    collator = DataCollatorWithPadding(tokenizer)
    probs = np.zeros(len(codes), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = collator([ds[int(i)] for i in idx])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**batch).logits
        probs[idx] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    return probs


@torch.no_grad()
def predict_proba_head_tail(model, tokenizer, codes: list[str], head: int = 128, tail: int = 382,
                            batch_size: int = 32) -> np.ndarray:
    """P(vulnerable) when a long function keeps its first `head` and last `tail` tokens.

    Evaluation-only variant (no retraining): instead of cutting after the first 510 tokens, the
    middle of the function is dropped, so the end of the function is also seen. Functions that fit
    into 512 tokens are encoded exactly as in predict_proba.
    """
    codes = list(codes)
    if not codes:
        return np.zeros(0, dtype=np.float32)
    device = next(model.parameters()).device
    model.eval()
    enc = tokenizer(codes, add_special_tokens=False, truncation=False)["input_ids"]
    keep = MAX_LEN - 2
    ids = []
    for t in enc:
        if len(t) > keep:
            t = t[:head] + t[-tail:]
        ids.append([tokenizer.cls_token_id] + t + [tokenizer.sep_token_id])
    order = np.argsort([len(x) for x in ids])
    collator = DataCollatorWithPadding(tokenizer)
    probs = np.zeros(len(codes), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = collator([{"input_ids": ids[int(i)], "attention_mask": [1] * len(ids[int(i)])} for i in idx])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**batch).logits
        probs[idx] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    return probs


def train(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    lr: float,
    seed: int,
    out_dir: str,
    max_epochs: int = 8,
    patience: int = 2,
    micro_batch: int = MICRO_BATCH,
    max_steps: int | None = None,
    save_model: bool = True,
    log_path: str | None = None,
) -> dict:
    """Fine-tune CodeBERT on ``train_df`` and keep the epoch with the best dev macro F1.

    Args:
        train_df, dev_df: DataFrames with columns ``code`` and ``label``.
        lr: peak learning rate for AdamW.
        seed: random seed (weights of the new head, data order, dropout).
        out_dir: where the best checkpoint (model + tokenizer) and ``train_log.json`` are written.
        max_epochs, patience: early stopping on dev macro F1.
        micro_batch: per-step batch; gradient accumulation keeps the effective batch at 32.
        max_steps: stop after this many optimizer steps (smoke tests only).
        save_model: write the best checkpoint to ``out_dir``.
        log_path: optional text file that gets one line per epoch / progress update.

    Returns:
        dict with lr, seed, best_epoch, best_dev (metrics), history (per epoch), timing.
    """
    set_all_seeds(seed)
    device = get_device()
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    tokenizer = load_tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, attn_implementation="sdpa"
    ).to(device)

    train_ds = _EncodedDataset(tokenizer, train_df["code"].tolist(), train_df["label"].tolist())
    collator = DataCollatorWithPadding(tokenizer)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=micro_batch, shuffle=True, collate_fn=collator, generator=gen)

    accum = max(1, EFFECTIVE_BATCH // micro_batch)
    steps_per_epoch = math.ceil(len(loader) / accum)
    total_steps = steps_per_epoch * max_epochs if max_steps is None else max_steps

    # No weight decay on biases and LayerNorm weights (standard BERT fine-tuning practice).
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=lr, fused=device.type == "cuda")
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps), total_steps)

    log(f"train n={len(train_ds)} dev n={len(dev_df)} lr={lr} seed={seed} micro_batch={micro_batch} "
        f"accum={accum} steps/epoch={steps_per_epoch} total_steps={total_steps} device={device}")

    history, best, best_epoch, bad_epochs, global_step = [], -1.0, 0, 0, 0
    start_epoch, stop = 1, False
    t_start = time.time()

    # Resume after an interruption: the state of the last finished epoch is kept in last_state.pt.
    state_path = out / "last_state.pt"
    if state_path.exists() and max_steps is None:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        gen.set_state(st["gen_state"])
        torch.set_rng_state(st["torch_rng"])
        if device.type == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(st["cuda_rng"])
        np.random.set_state(st["np_rng"])
        random.setstate(st["py_rng"])
        history, best, best_epoch = st["history"], st["best"], st["best_epoch"]
        bad_epochs, global_step, stop = st["bad_epochs"], st["global_step"], st["stop"]
        start_epoch = st["epoch"] + 1
        log(f"resumed from {state_path} after epoch {st['epoch']} (best epoch {best_epoch})")

    for epoch in range(start_epoch, max_epochs + 1):
        if stop:
            break
        model.train()
        t_epoch, running, n_batches = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(**batch).loss / accum
            loss.backward()
            running += loss.item() * accum
            n_batches += 1
            if (i + 1) % accum == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 50 == 0:
                    log(f"epoch {epoch} step {global_step} loss {running / n_batches:.4f} "
                        f"{(time.time() - t_epoch) / max(1, global_step - (epoch - 1) * steps_per_epoch):.3f}s/step")
                if max_steps is not None and global_step >= max_steps:
                    break
        train_secs = time.time() - t_epoch

        t_dev = time.time()
        dev_prob = predict_proba(model, tokenizer, dev_df["code"].tolist())
        dev_scores = _dev_scores(dev_df["label"].values, (dev_prob >= 0.5).astype(int))
        dev_secs = time.time() - t_dev
        rec = {"epoch": epoch, "train_loss": running / max(1, n_batches), "steps": global_step,
               "train_secs": round(train_secs, 1), "dev_secs": round(dev_secs, 1), **{f"dev_{k}": v for k, v in dev_scores.items()}}
        history.append(rec)
        log(f"epoch {epoch} done: loss {rec['train_loss']:.4f} dev_macro_f1 {dev_scores['macro_f1']:.4f} "
            f"dev_acc {dev_scores['accuracy']:.4f} ({train_secs:.0f}s train, {dev_secs:.0f}s dev)")

        if dev_scores["macro_f1"] > best:
            best, best_epoch, bad_epochs = dev_scores["macro_f1"], epoch, 0
            if save_model:
                model.save_pretrained(out)
                tokenizer.save_pretrained(out)
        else:
            bad_epochs += 1
        if max_steps is not None and global_step >= max_steps:
            break
        if bad_epochs >= patience:
            log(f"early stop after epoch {epoch} (best epoch {best_epoch})")
            stop = True
        if save_model and max_steps is None:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "gen_state": gen.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                        "np_rng": np.random.get_state(), "py_rng": random.getstate(),
                        "history": history, "best": best, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
                        "global_step": global_step, "stop": stop, "epoch": epoch}, state_path)
        if stop:
            break

    result = {
        "model": MODEL_NAME, "lr": lr, "seed": seed, "max_len": MAX_LEN,
        "effective_batch": EFFECTIVE_BATCH, "micro_batch": micro_batch,
        "best_epoch": best_epoch,
        "best_dev": {k.replace("dev_", ""): v for k, v in history[best_epoch - 1].items() if k.startswith("dev_") and k != "dev_secs"},
        "history": history, "total_secs": round(time.time() - t_start, 1),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    with open(out / "train_log.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    if state_path.exists():
        state_path.unlink()          # the run is complete; the resume state is no longer needed
    return result


def load(model_dir: str):
    """Load a saved checkpoint. Returns ``(model, tokenizer)`` on GPU if available, in eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, attn_implementation="sdpa")
    model.to(get_device()).eval()
    return model, tokenizer


if __name__ == "__main__":
    # Self-test (CPU is fine): 4 toy functions, 2 optimizer steps, then predict.
    toy = pd.DataFrame({
        "code": ["void f(char *s) { char b[64]; strcpy(b, s); }",
                 "void f(const char *s) { char b[64]; snprintf(b, sizeof b, \"%s\", s); }",
                 "int g(int *a, int n) { return a[n]; }",
                 "int g(int *a, int n, int i) { if (i < 0 || i >= n) return -1; return a[i]; }"] * 4,
        "label": [1, 0, 1, 0] * 4,
    })
    res = train(toy, toy.iloc[:4], lr=2e-5, seed=42, out_dir="models/_selftest",
                max_epochs=1, micro_batch=8, max_steps=2, save_model=False)
    print("best_dev:", res["best_dev"])
    print("token lengths:", token_lengths(load_tokenizer(), toy["code"].tolist()[:4]))
