"""Method 1: TF-IDF over C tokens + Random Forest.

Owner: Mishary Waleed AlAwadh

Input DataFrames have columns id (int), code (str), label (int, 1 = vulnerable).
The test set is never used here for tuning: tune_on_dev only sees train + dev.

Run the self-test from the repo root:   python -m src.tfidf_rf
"""

from __future__ import annotations

import itertools
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline

try:
    from src.c_tokenizer import tokenize_c
except ImportError:  # running from inside src/
    from c_tokenizer import tokenize_c

# Default dev grid (master plan, section 3).
GRID = {
    "ngram_range": [(1, 1), (1, 2)],
    "max_features": [20000, 50000],
    "n_estimators": [300, 500],
    "max_depth": [None, 60],
    "min_samples_leaf": [1, 2],
    "class_weight": [None, "balanced"],
}

_VEC_KEYS = ("ngram_range", "max_features")
_RF_KEYS = ("n_estimators", "max_depth", "min_samples_leaf", "class_weight")


def _identity(x):
    """Used during tuning, where the code is tokenized once in advance."""
    return x


def _vectorizer(params: dict, pretokenized: bool = False) -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=_identity if pretokenized else tokenize_c,
        preprocessor=_identity if pretokenized else None,
        lowercase=False,
        token_pattern=None,
        ngram_range=tuple(params["ngram_range"]),
        max_features=params["max_features"],
        sublinear_tf=True,
    )


def _forest(params: dict, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight=params["class_weight"],
        random_state=seed,
        n_jobs=-1,
    )


def build_pipeline(params: dict, seed: int) -> Pipeline:
    """TF-IDF (tokenize_c) -> Random Forest, built from one parameter dict."""
    return Pipeline([
        ("tfidf", _vectorizer(params)),
        ("rf", _forest(params, seed)),
    ])


def tune_on_dev(train_df: pd.DataFrame, dev_df: pd.DataFrame, grid: dict = GRID,
                seed: int = 42) -> tuple[dict, pd.DataFrame]:
    """Fit on train, score every grid combination on DEV macro F1.

    The vectorizer is fitted once per (ngram_range, max_features) and reused
    for all Random Forest settings. Returns (best_params, results_table),
    with the table sorted by dev macro F1 (best first).
    """
    t0 = time.time()
    train_tok = [tokenize_c(c) for c in train_df["code"]]
    dev_tok = [tokenize_c(c) for c in dev_df["code"]]
    y_train = train_df["label"].to_numpy()
    y_dev = dev_df["label"].to_numpy()
    print(f"[tune] tokenized train/dev in {time.time() - t0:.1f}s")

    vec_combos = list(itertools.product(*(grid[k] for k in _VEC_KEYS)))
    rf_combos = list(itertools.product(*(grid[k] for k in _RF_KEYS)))
    total = len(vec_combos) * len(rf_combos)

    rows, all_params = [], []
    i = 0
    for vec_values in vec_combos:
        vec_params = dict(zip(_VEC_KEYS, vec_values))
        vec = _vectorizer(vec_params, pretokenized=True)
        X_train = vec.fit_transform(train_tok)
        X_dev = vec.transform(dev_tok)
        for rf_values in rf_combos:
            i += 1
            params = {**vec_params, **dict(zip(_RF_KEYS, rf_values))}
            t1 = time.time()
            rf = _forest(params, seed).fit(X_train, y_train)
            pred = rf.predict(X_dev)
            row = {
                **{k: str(v) for k, v in params.items()},
                "dev_macro_f1": f1_score(y_dev, pred, average="macro", labels=[0, 1],
                                         zero_division=0),
                "dev_accuracy": accuracy_score(y_dev, pred),
                "dev_vuln_recall": recall_score(y_dev, pred, pos_label=1, zero_division=0),
                "fit_seconds": round(time.time() - t1, 1),
            }
            rows.append(row)
            all_params.append(params)
            print(f"[tune] {i}/{total} {params} -> dev macro F1 {row['dev_macro_f1']:.4f}")

    results = pd.DataFrame(rows)
    best_idx = int(results["dev_macro_f1"].idxmax())   # first best on ties
    best_params = all_params[best_idx]
    results = results.sort_values("dev_macro_f1", ascending=False).reset_index(drop=True)
    print(f"[tune] best {best_params} (dev macro F1 {rows[best_idx]['dev_macro_f1']:.4f}), "
          f"total {time.time() - t0:.0f}s")
    return best_params, results


def train(train_df: pd.DataFrame, params: dict, seed: int) -> Pipeline:
    """Fit the full pipeline on the training DataFrame."""
    model = build_pipeline(params, seed)
    model.fit(train_df["code"].tolist(), train_df["label"].to_numpy())
    return model


def predict_proba(model: Pipeline, codes: list[str]) -> np.ndarray:
    """Probability of class 1 (vulnerable), shape (n,)."""
    proba = model.predict_proba(list(codes))
    col = list(model.classes_).index(1)
    return proba[:, col]


def predict(model: Pipeline, codes: list[str], threshold: float = 0.5) -> np.ndarray:
    """Hard labels (1 = vulnerable) from predict_proba."""
    return (predict_proba(model, codes) >= threshold).astype(int)


def top_features(model: Pipeline, k: int = 30) -> pd.DataFrame:
    """Top-k TF-IDF features by Random Forest impurity importance."""
    names = model.named_steps["tfidf"].get_feature_names_out()
    importances = model.named_steps["rf"].feature_importances_
    order = np.argsort(importances)[::-1][:k]
    return pd.DataFrame({"feature": names[order], "importance": importances[order]})


def save(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)


def load(path: str) -> Pipeline:
    return joblib.load(path)


if __name__ == "__main__":
    import os
    import tempfile

    rng = np.random.default_rng(0)
    names = ["buf", "dstBuf", "out_ptr", "pkt", "ctx_data", "tmp", "line", "name"]

    def fake_func(vulnerable: bool) -> str:
        a, b = rng.choice(names, 2, replace=False)
        if vulnerable:
            return (f"void f(char *{b}) {{\n  char {a}[64];\n"
                    f"  strcpy({a}, {b});\n  puts({a});\n}}")
        return (f"void f(char *{b}) {{\n  char {a}[64];\n  if (strlen({b}) >= sizeof({a})) return;\n"
                f"  snprintf({a}, sizeof({a}), \"%s\", {b});\n  puts({a});\n}}")

    def fake_df(n: int, start: int) -> pd.DataFrame:
        labels = rng.integers(0, 2, n)
        return pd.DataFrame({"id": np.arange(start, start + n),
                             "code": [fake_func(bool(y)) for y in labels],
                             "label": labels})

    train_df, dev_df, test_df = fake_df(200, 0), fake_df(60, 200), fake_df(60, 260)
    small_grid = {"ngram_range": [(1, 1), (1, 2)], "max_features": [500],
                  "n_estimators": [20], "max_depth": [None], "min_samples_leaf": [1],
                  "class_weight": [None, "balanced"]}

    best, table = tune_on_dev(train_df, dev_df, small_grid, seed=42)
    print(table)
    assert len(table) == 4 and table["dev_macro_f1"].iloc[0] == table["dev_macro_f1"].max()

    model = train(train_df, best, seed=1)
    p = predict_proba(model, test_df["code"].tolist())
    assert p.shape == (len(test_df),) and ((p >= 0) & (p <= 1)).all()
    acc = (predict(model, test_df["code"].tolist()) == test_df["label"].to_numpy()).mean()
    print(f"fake test accuracy {acc:.3f} (should be ~1.0, the fake data is easy)")
    assert acc > 0.95

    feats = top_features(model, k=10)
    print(feats)
    assert list(feats.columns) == ["feature", "importance"] and len(feats) == 10

    path = os.path.join(tempfile.gettempdir(), "tfidf_rf_selftest.joblib")
    save(model, path)
    assert np.allclose(predict_proba(load(path), test_df["code"].tolist()), p)
    os.remove(path)
    print("tfidf_rf self-test passed")
