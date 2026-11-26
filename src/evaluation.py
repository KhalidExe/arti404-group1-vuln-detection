"""Metrics, majority baseline, seed aggregation, McNemar's test and figures.

Owner: Mishary Waleed AlAwadh
"""
from __future__ import annotations


def compute_metrics(y_true, y_pred) -> dict:
    """Return ``{"accuracy", "macro_f1", "vuln_recall"}`` (sklearn; vuln_recall = recall of class 1)."""
    raise NotImplementedError


def majority_baseline(train_labels, test_labels) -> dict:
    """Predict the most common train label for every test row; return compute_metrics of that."""
    raise NotImplementedError


def aggregate_seeds(runs: list[dict]) -> dict:
    """``[{metric: value}, ...]`` -> ``{metric: {"mean", "std"}}`` with sample std (ddof=1)."""
    raise NotImplementedError


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """McNemar's test on paired predictions.

    b = A right & B wrong, c = A wrong & B right.
    If b + c < 25: exact binomial test (``scipy.stats.binomtest(b, b + c, 0.5)``), method "exact";
    else chi2 = (|b - c| - 1)^2 / (b + c) with 1 dof, method "chi2_cc".

    Returns:
        ``{"b", "c", "statistic", "p_value", "method"}``
    """
    raise NotImplementedError


def plot_confusion_matrix(y_true, y_pred, title: str, path: str) -> None:
    """Confusion matrix with labels Safe/Vulnerable and counts in cells; 300 dpi PNG, grayscale-readable."""
    raise NotImplementedError


def plot_drop_chart(summary: dict, path: str) -> None:
    """Grouped bars: accuracy and macro F1 for TF-IDF and CodeBERT, original vs renamed, std error bars."""
    raise NotImplementedError


if __name__ == "__main__":
    # Self-test with small arrays that can be checked by hand.
    raise NotImplementedError
