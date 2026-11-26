"""Command-line runner for all heavy experiments. Every reported number is written to results/.

Run from the repo root inside the venv, in this order:

    python scripts/run_experiments.py env                 # GPU + package versions
    python scripts/run_experiments.py download            # dataset -> data/raw/
    python scripts/run_experiments.py clean               # data_prep: cleaned splits + data stats
    python scripts/run_experiments.py rename              # renaming: test_renamed + train_augmented
    python scripts/run_experiments.py tfidf-tune          # dev grid (seed 42)
    python scripts/run_experiments.py tfidf-seeds         # 3 seeds with the best params
    python scripts/run_experiments.py codebert-lr-search  # lr in {1e-5, 2e-5, 3e-5}, seed 42
    python scripts/run_experiments.py codebert-seeds      # seeds 123, 2026 with the chosen lr
    python scripts/run_experiments.py augment             # both methods on train_augmented
    python scripts/run_experiments.py predict-missing     # fill predictions for splits added later
    python scripts/run_experiments.py summarize           # metrics_summary.json, tables, figures

Predictions are stored per run in results/preds/<method>_<train>_seed<seed>_<split>.csv
(columns: id, label, prob, pred). summarize builds every table and figure from these files.
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RAW_DIR = REPO_ROOT / "data" / "raw"
PROC_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"
PRED_DIR = RESULTS_DIR / "preds"
FIG_DIR = RESULTS_DIR / "figures"
MODELS_DIR = REPO_ROOT / "models"
LOG_DIR = REPO_ROOT / "logs"

SEEDS = [42, 123, 2026]
PRIMARY_SEED = 42
LR_GRID = [1e-5, 2e-5, 3e-5]
EVAL_SPLITS = ["dev", "test", "test_renamed"]


# ----------------------------------------------------------------------------- helpers

def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def _load_split(name: str):
    """Load one processed split (train, dev, test, test_renamed, train_augmented) or None."""
    import pandas as pd
    path = PROC_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def _pred_path(method: str, train: str, seed: int, split: str) -> Path:
    return PRED_DIR / f"{method}_{train}_seed{seed}_{split}.csv"


def _save_preds(df, probs, method: str, train: str, seed: int, split: str) -> None:
    """Save per-function predictions of one run on one split."""
    import pandas as pd
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": df["id"].values, "label": df["label"].values,
                        "prob": probs, "pred": (probs >= 0.5).astype(int)})
    out.to_csv(_pred_path(method, train, seed, split), index=False)


def _lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("-0", "-")


def _codebert_dir(train: str, lr: float, seed: int) -> Path:
    return MODELS_DIR / "codebert" / f"{train}_lr{_lr_tag(lr)}_seed{seed}"


def _raw_minimal() -> dict:
    """Raw splits with whitespace collapsed only (smoke tests before data_prep is integrated)."""
    import pandas as pd
    out = {}
    for name, file in (("train", "train"), ("dev", "validation"), ("test", "test")):
        df = pd.read_parquet(RAW_DIR / f"{file}.parquet")
        out[name] = pd.DataFrame({"id": df["id"].astype(int),
                                  "code": df["func"].str.split().str.join(" "),
                                  "label": df["target"].astype(int)})
    return out


# ----------------------------------------------------------------------------- env / data

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


def cmd_clean(_args) -> None:
    """Run data_prep: cleaned + de-duplicated splits, data stats, test truncation count."""
    from src import codebert_model as cb
    from src import data_prep
    raw = data_prep.load_raw(str(RAW_DIR))
    splits, stats = data_prep.build_clean_splits(raw)
    data_prep.save_splits(splits, str(PROC_DIR))

    tok = cb.load_tokenizer()
    lens = cb.token_lengths(tok, splits["test"]["code"].tolist())
    n_trunc = int((lens > cb.MAX_LEN).sum())
    stats["test_truncated_512"] = {"count": n_trunc, "pct": round(n_trunc / len(lens), 4)}
    for name in ("train", "dev"):
        l2 = cb.token_lengths(tok, splits[name]["code"].tolist())
        stats.setdefault("truncated_512_other_splits", {})[name] = {
            "count": int((l2 > cb.MAX_LEN).sum()), "pct": round(float((l2 > cb.MAX_LEN).mean()), 4)}
    stats["test_token_length"] = {"median": float(__import__("numpy").median(lens)),
                                  "mean": round(float(lens.mean()), 1), "max": int(lens.max())}
    _write_json(stats, RESULTS_DIR / "data_stats.json")
    print(json.dumps(stats, indent=2))


def cmd_rename(_args) -> None:
    """Run renaming: renamed test set, augmented train set, renaming stats and examples."""
    from src import renaming
    test, train = _load_split("test"), _load_split("train")
    renamed, stats = renaming.build_renamed_split(test)
    renamed.to_parquet(PROC_DIR / "test_renamed.parquet", index=False)
    _write_json(stats, RESULTS_DIR / "renaming_stats.json")
    aug = renaming.build_augmented_train(train)
    aug.to_parquet(PROC_DIR / "train_augmented.parquet", index=False)
    print(json.dumps(stats, indent=2))
    print("train_augmented rows:", len(aug))


# ----------------------------------------------------------------------------- TF-IDF + RF

QUICK_PARAMS = {"ngram_range": (1, 2), "max_features": 20000, "n_estimators": 300,
                "max_depth": None, "min_samples_leaf": 1, "class_weight": None}


def cmd_tfidf_smoke(_args) -> None:
    """One TF-IDF + RF setting on the RAW splits: checks the code runs and measures time."""
    from src import evaluation, tfidf_rf
    splits = _raw_minimal()
    t0 = time.time()
    model = tfidf_rf.train(splits["train"], QUICK_PARAMS, seed=PRIMARY_SEED)
    fit_secs = time.time() - t0
    res = {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in QUICK_PARAMS.items()},
           "fit_secs": round(fit_secs, 1)}
    for name in ("dev", "test"):
        p = tfidf_rf.predict_proba(model, splits[name]["code"].tolist())
        res[name] = evaluation.compute_metrics(splits[name]["label"].values, (p >= 0.5).astype(int))
    res["top_features"] = tfidf_rf.top_features(model, k=15).to_dict(orient="records")
    LOG_DIR.mkdir(exist_ok=True)
    _write_json(res, LOG_DIR / "tfidf_smoke_raw.json")
    print(json.dumps(res, indent=2, default=str))


def cmd_tfidf_tune(args) -> None:
    """Dev-set grid search (seed 42). Writes results/tfidf_grid.csv and tfidf_best_params.json."""
    from src import tfidf_rf
    train, dev = _load_split("train"), _load_split("dev")
    t0 = time.time()
    best, table = tfidf_rf.tune_on_dev(train, dev, tfidf_rf.GRID, seed=PRIMARY_SEED)
    table.to_csv(RESULTS_DIR / "tfidf_grid.csv", index=False)
    best = {k: (list(v) if isinstance(v, tuple) else v) for k, v in best.items()}
    _write_json({"best_params": best, "grid_secs": round(time.time() - t0, 1)},
                RESULTS_DIR / "tfidf_best_params.json")
    print("best params:", best)


def _tfidf_params() -> dict:
    best = _read_json(RESULTS_DIR / "tfidf_best_params.json")["best_params"]
    if isinstance(best.get("ngram_range"), list):
        best["ngram_range"] = tuple(best["ngram_range"])
    return best


def cmd_tfidf_seeds(args) -> None:
    """Train TF-IDF + RF with the best params for every seed; predict dev/test/test_renamed."""
    from src import tfidf_rf
    params, train_name = _tfidf_params(), args.train
    train = _load_split(train_name)
    for seed in args.seeds:
        t0 = time.time()
        model = tfidf_rf.train(train, params, seed=seed)
        path = MODELS_DIR / "tfidf_rf" / f"{train_name}_seed{seed}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        tfidf_rf.save(model, str(path))
        if train_name == "train" and seed == PRIMARY_SEED:
            tfidf_rf.save(model, str(MODELS_DIR / "tfidf_rf_seed42.joblib"))
            tfidf_rf.top_features(model, k=30).to_csv(RESULTS_DIR / "tfidf_top_features.csv", index=False)
        for split in EVAL_SPLITS:
            df = _load_split(split)
            if df is not None:
                _save_preds(df, tfidf_rf.predict_proba(model, df["code"].tolist()), "tfidf_rf", train_name, seed, split)
        print(f"tfidf_rf {train_name} seed {seed}: {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------- CodeBERT

def _codebert_predict_dir(model_dir: Path, train_name: str, seed: int, splits=EVAL_SPLITS, only_missing=False) -> None:
    from src import codebert_model as cb
    todo = [s for s in splits if _load_split(s) is not None
            and not (only_missing and _pred_path("codebert", train_name, seed, s).exists())]
    if not todo:
        return
    model, tok = cb.load(str(model_dir))
    for split in todo:
        df = _load_split(split)
        _save_preds(df, cb.predict_proba(model, tok, df["code"].tolist()), "codebert", train_name, seed, split)
        print(f"  predicted {split} with {model_dir.name}")


def cmd_codebert_lr_search(args) -> None:
    """Seed-42 runs for every learning rate; pick the lr with the best dev macro F1."""
    import pandas as pd

    from src import codebert_model as cb
    train, dev = _load_split("train"), _load_split("dev")
    LOG_DIR.mkdir(exist_ok=True)
    rows = []
    for lr in args.lrs:
        out = _codebert_dir("train", lr, PRIMARY_SEED)
        if (out / "train_log.json").exists() and not args.force:
            res = _read_json(out / "train_log.json")
            print(f"lr {lr}: already done (best epoch {res['best_epoch']})")
        else:
            res = cb.train(train, dev, lr=lr, seed=PRIMARY_SEED, out_dir=str(out),
                           log_path=str(LOG_DIR / "codebert_lr_search.log"))
        rows.append({"lr": lr, "best_epoch": res["best_epoch"], "epochs_run": len(res["history"]),
                     **{f"dev_{k}": v for k, v in res["best_dev"].items()},
                     "total_secs": res["total_secs"],
                     "secs_per_epoch": round(sum(h["train_secs"] for h in res["history"]) / len(res["history"]), 1)})
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "codebert_lr_search.csv", index=False)
    table = pd.DataFrame(rows)
    best = table.sort_values("dev_macro_f1", ascending=False).iloc[0]
    choice = {"learning_rate": float(best["lr"]), "best_epoch": int(best["best_epoch"]),
              "dev_macro_f1": float(best["dev_macro_f1"]), "batch_size": cb.EFFECTIVE_BATCH, "max_len": cb.MAX_LEN}
    _write_json(choice, RESULTS_DIR / "codebert_best_params.json")
    print("chosen:", choice)
    best_dir = _codebert_dir("train", choice["learning_rate"], PRIMARY_SEED)
    _codebert_predict_dir(best_dir, "train", PRIMARY_SEED)
    # Checkpoint used by Try it and published as a release asset.
    target = MODELS_DIR / "codebert_seed42"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(best_dir, target)


def cmd_codebert_seeds(args) -> None:
    """Train the remaining seeds with the chosen learning rate and predict all available splits."""
    from src import codebert_model as cb
    lr = _read_json(RESULTS_DIR / "codebert_best_params.json")["learning_rate"]
    train_name = args.train
    train, dev = _load_split(train_name), _load_split("dev")
    for seed in args.seeds:
        out = _codebert_dir(train_name, lr, seed)
        if not (out / "train_log.json").exists() or args.force:
            cb.train(train, dev, lr=lr, seed=seed, out_dir=str(out),
                     log_path=str(LOG_DIR / f"codebert_{train_name}_seeds.log"))
        _codebert_predict_dir(out, train_name, seed)


def cmd_augment(args) -> None:
    """Retrain both methods on train_augmented (original + renamed copies)."""
    ns = argparse.Namespace(train="train_augmented", seeds=args.seeds, force=args.force)
    if args.method in ("both", "tfidf_rf"):
        cmd_tfidf_seeds(ns)
    if args.method in ("both", "codebert"):
        cmd_codebert_seeds(ns)


def cmd_predict_missing(_args) -> None:
    """Predict splits that did not exist when a model was trained (e.g. test_renamed)."""
    from src import tfidf_rf
    for d in sorted((MODELS_DIR / "codebert").glob("*_seed*")):
        train_name, rest = d.name.split("_lr")
        seed = int(rest.split("_seed")[1])
        lr = float(rest.split("_seed")[0])
        best = _read_json(RESULTS_DIR / "codebert_best_params.json", {}).get("learning_rate")
        if best is not None and abs(lr - best) < 1e-12:
            _codebert_predict_dir(d, train_name, seed, only_missing=True)
    for f in sorted((MODELS_DIR / "tfidf_rf").glob("*_seed*.joblib")):
        train_name, seed = f.stem.rsplit("_seed", 1)
        model = None
        for split in EVAL_SPLITS:
            df = _load_split(split)
            if df is not None and not _pred_path("tfidf_rf", train_name, int(seed), split).exists():
                model = model or tfidf_rf.load(str(f))
                _save_preds(df, tfidf_rf.predict_proba(model, df["code"].tolist()), "tfidf_rf", train_name, int(seed), split)


# ----------------------------------------------------------------------------- summary

def _r4(x: float) -> float:
    return round(float(x), 4)


def _agg(evaluation, runs: list[dict]) -> dict:
    agg = evaluation.aggregate_seeds(runs) if len(runs) > 1 else {k: {"mean": v, "std": 0.0} for k, v in runs[0].items()}
    return {k: {"mean": _r4(v["mean"]), "std": _r4(v["std"])} for k, v in agg.items()}


def _method_block(evaluation, method: str, train_name: str, seeds: list[int]) -> tuple[dict, list[dict]]:
    """dev / original_test / renamed_test / drop for one method and training set, plus per-run rows."""
    import pandas as pd
    block, rows, per_split = {}, [], {}
    for split, key in (("dev", "dev"), ("test", "original_test"), ("test_renamed", "renamed_test")):
        runs = []
        for seed in seeds:
            p = _pred_path(method, train_name, seed, split)
            if not p.exists():
                continue
            df = pd.read_csv(p)
            m = evaluation.compute_metrics(df["label"].values, df["pred"].values)
            runs.append(m)
            rows.append({"method": method, "train": train_name, "seed": seed, "split": split, **m})
        per_split[key] = runs
        if runs:
            block[key] = _agg(evaluation, runs)
    o, r = per_split.get("original_test", []), per_split.get("renamed_test", [])
    if o and r and len(o) == len(r):
        drops = [{"accuracy": a["accuracy"] - b["accuracy"], "macro_f1": a["macro_f1"] - b["macro_f1"]} for a, b in zip(o, r)]
        block["drop"] = _agg(evaluation, drops)
    return block, rows


def cmd_summarize(_args) -> None:
    """Build metrics_summary.json, per_run_metrics.csv, test_predictions.csv, figures and tables."""
    import numpy as np
    import pandas as pd

    from src import codebert_model as cb
    from src import evaluation

    summary = _read_json(RESULTS_DIR / "metrics_summary.json", {}) or {}
    summary.update({"primary_seed": PRIMARY_SEED, "seeds": SEEDS})
    summary["data"] = _read_json(RESULTS_DIR / "data_stats.json", {})
    summary["renaming"] = _read_json(RESULTS_DIR / "renaming_stats.json", {})

    train, test = _load_split("train"), _load_split("test")
    if train is not None and test is not None:
        mb = evaluation.majority_baseline(train["label"].values, test["label"].values)
        summary["majority_baseline"] = {"original_test": {k: _r4(v) for k, v in mb.items()}}

    all_rows = []
    tf_best = _read_json(RESULTS_DIR / "tfidf_best_params.json", {})
    cb_best = _read_json(RESULTS_DIR / "codebert_best_params.json", {})
    for method, best in (("tfidf_rf", tf_best.get("best_params", {})), ("codebert", cb_best)):
        block, rows = _method_block(evaluation, method, "train", SEEDS)
        all_rows += rows
        summary[method] = {"best_params": best, **block}

    aug = {"seeds_used": []}
    for method in ("tfidf_rf", "codebert"):
        used = [s for s in SEEDS if _pred_path(method, "train_augmented", s, "test").exists()]
        if used:
            block, rows = _method_block(evaluation, method, "train_augmented", used)
            all_rows += rows
            aug[method] = {k: v for k, v in block.items() if k != "dev"}
            aug["seeds_used"] = sorted(set(aug["seeds_used"]) | set(used))
    summary["augmentation"] = aug
    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "per_run_metrics.csv", index=False)

    # McNemar on the primary-seed predictions (TF-IDF vs CodeBERT).
    summary["mcnemar"] = {}
    for split, key in (("test", "original_test"), ("test_renamed", "renamed_test")):
        a, b = _pred_path("tfidf_rf", "train", PRIMARY_SEED, split), _pred_path("codebert", "train", PRIMARY_SEED, split)
        if a.exists() and b.exists():
            da, db = pd.read_csv(a), pd.read_csv(b)
            assert (da["id"].values == db["id"].values).all(), "prediction files are not aligned"
            res = evaluation.mcnemar_test(da["label"].values, da["pred"].values, db["pred"].values)
            summary["mcnemar"][key] = {k: (_r4(v) if isinstance(v, float) else v) for k, v in res.items()}

    # Model used by Try it: higher mean dev macro F1 (decided on dev only).
    devs = {m: summary.get(m, {}).get("dev", {}).get("macro_f1", {}).get("mean") for m in ("tfidf_rf", "codebert")}
    if all(v is not None for v in devs.values()):
        summary["selected_model_for_try_it"] = max(devs, key=devs.get)

    # test_predictions.csv: one row per test function, both models, original and renamed.
    if test is not None:
        tok = cb.load_tokenizer()
        tp = pd.DataFrame({"id": test["id"].values, "label": test["label"].values})
        tp["n_tokens"] = cb.token_lengths(tok, test["code"].tolist())
        tp["truncated"] = tp["n_tokens"] > cb.MAX_LEN
        for method in ("tfidf_rf", "codebert"):
            for split, tag in (("test", "orig"), ("test_renamed", "ren")):
                p = _pred_path(method, "train", PRIMARY_SEED, split)
                if p.exists():
                    d = pd.read_csv(p).set_index("id")
                    tp[f"{method}_{tag}_prob"] = tp["id"].map(d["prob"]).round(4)
                    tp[f"{method}_{tag}_pred"] = tp["id"].map(d["pred"])
        tp.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

        # Accuracy on truncated vs non-truncated test functions (primary seed).
        rows = []
        for method in ("tfidf_rf", "codebert"):
            col = f"{method}_orig_pred"
            if col in tp:
                for flag, grp in tp.groupby("truncated"):
                    m = evaluation.compute_metrics(grp["label"].values, grp[col].values)
                    rows.append({"method": method, "truncated": bool(flag), "n": len(grp), **{k: _r4(v) for k, v in m.items()}})
        if rows:
            pd.DataFrame(rows).to_csv(RESULTS_DIR / "truncation_table.csv", index=False)

        # Figures.
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        names = {"tfidf_rf": "TF-IDF + RF", "codebert": "CodeBERT"}
        for method in ("tfidf_rf", "codebert"):
            for tag, label in (("orig", "original test"), ("ren", "renamed test")):
                col = f"{method}_{tag}_pred"
                if col in tp:
                    evaluation.plot_confusion_matrix(tp["label"].values, tp[col].values,
                                                     f"{names[method]} - {label} (seed 42)",
                                                     str(FIG_DIR / f"cm_{method}_{tag}.png"))
        if all("drop" in summary.get(m, {}) for m in ("tfidf_rf", "codebert")):
            evaluation.plot_drop_chart(summary, str(FIG_DIR / "drop_chart.png"))

    _write_json(summary, RESULTS_DIR / "metrics_summary.json")
    print(json.dumps(summary, indent=2))


# ----------------------------------------------------------------------------- smoke

def cmd_smoke(args) -> None:
    """Run a fixed number of CodeBERT optimizer steps on the raw data and measure the speed."""
    import numpy as np

    from src import codebert_model as cb
    splits = _raw_minimal()
    LOG_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    res = cb.train(splits["train"], splits["dev"], lr=2e-5, seed=PRIMARY_SEED, out_dir="models/_smoke",
                   max_epochs=1, micro_batch=args.micro_batch, max_steps=args.steps,
                   save_model=False, log_path=str(LOG_DIR / "smoke.log"))
    h = res["history"][0]
    sec_per_step = h["train_secs"] / h["steps"]
    steps_per_epoch = int(np.ceil(len(splits["train"]) / cb.EFFECTIVE_BATCH))
    lens = cb.token_lengths(cb.load_tokenizer(), splits["test"]["code"].tolist())
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
    _write_json(out, LOG_DIR / "smoke_summary.json")


# ----------------------------------------------------------------------------- CLI

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (("env", cmd_env), ("download", cmd_download), ("clean", cmd_clean),
                     ("rename", cmd_rename), ("tfidf-smoke", cmd_tfidf_smoke), ("tfidf-tune", cmd_tfidf_tune),
                     ("predict-missing", cmd_predict_missing), ("summarize", cmd_summarize)):
        sub.add_parser(name).set_defaults(func=fn)

    p = sub.add_parser("tfidf-seeds")
    p.add_argument("--train", default="train", choices=["train", "train_augmented"])
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.set_defaults(func=cmd_tfidf_seeds)

    p = sub.add_parser("codebert-lr-search")
    p.add_argument("--lrs", type=float, nargs="+", default=LR_GRID)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_codebert_lr_search)

    p = sub.add_parser("codebert-seeds")
    p.add_argument("--train", default="train", choices=["train", "train_augmented"])
    p.add_argument("--seeds", type=int, nargs="+", default=[123, 2026])
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_codebert_seeds)

    p = sub.add_parser("augment")
    p.add_argument("--method", default="both", choices=["both", "tfidf_rf", "codebert"])
    p.add_argument("--seeds", type=int, nargs="+", default=[PRIMARY_SEED])
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_augment)

    p = sub.add_parser("smoke")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--micro-batch", type=int, default=16)
    p.set_defaults(func=cmd_smoke)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
