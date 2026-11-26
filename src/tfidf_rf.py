"""Method 1: TF-IDF (token unigrams/bigrams) + Random Forest.

Owner: Mishary Waleed AlAwadh

Hyper-parameters are chosen on the dev set (macro F1) only. The test set is never used for tuning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Default dev grid (master plan, section 3).
GRID = {
    "ngram_range": [(1, 1), (1, 2)],
    "max_features": [20000, 50000],
    "n_estimators": [300, 500],
    "max_depth": [None, 60],
    "min_samples_leaf": [1, 2],
    "class_weight": [None, "balanced"],
}


def build_pipeline(params: dict, seed: int) -> Pipeline:
    """TfidfVectorizer(tokenizer=tokenize_c, lowercase=False, token_pattern=None, sublinear_tf=True,
    ngram_range, max_features) -> RandomForestClassifier(n_estimators, max_depth, min_samples_leaf,
    class_weight, random_state=seed, n_jobs=-1)."""
    raise NotImplementedError


def tune_on_dev(train_df: pd.DataFrame, dev_df: pd.DataFrame, grid: dict, seed: int = 42) -> tuple[dict, pd.DataFrame]:
    """Fit on train, score dev macro F1 for every grid combination.

    Returns:
        (best_params, results_table) - one row per combination.
    """
    raise NotImplementedError


def train(train_df: pd.DataFrame, params: dict, seed: int) -> Pipeline:
    """Fit the pipeline on ``train_df["code"]`` / ``train_df["label"]``."""
    raise NotImplementedError


def predict_proba(model: Pipeline, codes: list[str]) -> np.ndarray:
    """Probability of class 1 (vulnerable), shape (n,)."""
    raise NotImplementedError


def top_features(model: Pipeline, k: int = 30) -> pd.DataFrame:
    """Top-k features by Random Forest importance. Columns: feature, importance."""
    raise NotImplementedError


def save(model: Pipeline, path: str) -> None:
    """Save with joblib."""
    raise NotImplementedError


def load(path: str) -> Pipeline:
    """Load with joblib."""
    raise NotImplementedError


if __name__ == "__main__":
    # Self-test on a tiny toy dataset.
    raise NotImplementedError
