"""Command-line entry point for the heavy experiments. Results go to results/.

Usage (from the repo root, inside the venv):
    python scripts/run_experiments.py env        # print GPU and package versions
    python scripts/run_experiments.py download   # download the dataset to data/raw/
    python scripts/run_experiments.py smoke      # short CodeBERT run to measure speed
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RAW_DIR = REPO_ROOT / "data" / "raw"
RESULTS_DIR = REPO_ROOT / "results"
LOG_DIR = REPO_ROOT / "logs"


def cmd_env(_args) -> None:
    """Print versions of Python, CUDA, torch and the main libraries."""
    import importlib.metadata as md

    import torch
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    }
    for pkg in ["transformers", "datasets", "scikit-learn", "pandas", "numpy", "pyarrow",
                "tree-sitter", "tree-sitter-c", "scipy", "statsmodels", "matplotlib", "joblib"]:
        try:
            info[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            info[pkg] = None
    print(json.dumps(info, indent=2))


def cmd_download(_args) -> None:
    """Download the three official splits and save them as parquet in data/raw/."""
    from datasets import load_dataset
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("google/code_x_glue_cc_defect_detection")
    for split in ds:
        path = RAW_DIR / f"{split}.parquet"
        ds[split].to_pandas().to_parquet(path, index=False)
        print(split, len(ds[split]), "->", path)


def _raw_minimal() -> dict:
    """Temporary loader used only for the speed test before data_prep.py is integrated.

    Whitespace is collapsed; no comment removal and no de-duplication.
    """
    import pandas as pd
    out = {}
    for name, file in (("train", "train"), ("dev", "validation"), ("test", "test")):
        df = pd.read_parquet(RAW_DIR / f"{file}.parquet")
        out[name] = pd.DataFrame({"id": df["id"].astype(int),
                                  "code": df["func"].str.split().str.join(" "),
                                  "label": df["target"].astype(int)})
    return out


def cmd_smoke(args) -> None:
    """Run a fixed number of optimizer steps on the raw data and measure the speed."""
    import numpy as np

    from src import codebert_model as cb
    splits = _raw_minimal()
    LOG_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    res = cb.train(splits["train"], splits["dev"], lr=2e-5, seed=42, out_dir="models/_smoke",
                   max_epochs=1, micro_batch=args.micro_batch, max_steps=args.steps,
                   save_model=False, log_path=str(LOG_DIR / "smoke.log"))
    h = res["history"][0]
    sec_per_step = h["train_secs"] / h["steps"]
    steps_per_epoch = int(np.ceil(len(splits["train"]) / cb.EFFECTIVE_BATCH))
    tok = cb.load_tokenizer()
    lens = cb.token_lengths(tok, splits["test"]["code"].tolist())
    out = {
        "steps": h["steps"], "micro_batch": args.micro_batch, "train_secs": h["train_secs"],
        "sec_per_step": round(sec_per_step, 3), "steps_per_epoch": steps_per_epoch,
        "est_train_secs_per_epoch": round(sec_per_step * steps_per_epoch, 1),
        "dev_eval_secs": h["dev_secs"], "dev_macro_f1_after_smoke": h["dev_macro_f1"],
        "raw_test_truncated_512": int((lens > cb.MAX_LEN).sum()),
        "raw_test_truncated_512_pct": round(float((lens > cb.MAX_LEN).mean()), 4),
        "peak_gpu_mem_gb": round(__import__("torch").cuda.max_memory_allocated() / 1e9, 2),
        "wall_secs": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=2))
    with open(LOG_DIR / "smoke_summary.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env").set_defaults(func=cmd_env)
    sub.add_parser("download").set_defaults(func=cmd_download)
    p = sub.add_parser("smoke")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--micro-batch", type=int, default=16)
    p.set_defaults(func=cmd_smoke)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
