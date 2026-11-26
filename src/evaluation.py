"""Evaluation shared by both methods: metrics, baseline, seeds, McNemar, figures.

Owner: Mishary Waleed AlAwadh

Labels: 1 = vulnerable, 0 = safe.
Run the self-test from the repo root:   python -m src.evaluation
"""

from __future__ import annotations

import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

METRICS = ("accuracy", "macro_f1", "vuln_recall")
CLASS_NAMES = ("Safe", "Vulnerable")


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy, macro F1 (both classes count equally) and recall of class 1."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1],
                                   zero_division=0)),
        "vuln_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def majority_baseline(train_labels, test_labels) -> dict:
    """Predict the most common TRAIN label for every test row.

    On a tie the smaller label (0 = safe) is used. The returned dict holds the
    metrics plus "majority_label".
    """
    counts = Counter(int(y) for y in train_labels)
    top = max(counts.values())
    majority = min(label for label, n in counts.items() if n == top)
    y_test = np.asarray(test_labels)
    result = compute_metrics(y_test, np.full(len(y_test), majority))
    result["majority_label"] = majority
    return result


def aggregate_seeds(runs: list[dict]) -> dict:
    """[{metric: value}, ...] over seeds -> {metric: {"mean", "std"}}, std with ddof=1."""
    out = {}
    for key in runs[0]:
        values = np.array([run[key] for run in runs], dtype=float)
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        out[key] = {"mean": float(values.mean()), "std": std}
    return out


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """McNemar's test on paired predictions of two models on the same test rows.

    b = A right & B wrong, c = A wrong & B right. Rows where both are right or
    both are wrong do not affect the test.
    b + c < 25  -> exact two-sided binomial test, statistic = b.
    otherwise   -> chi-square with continuity correction, 1 degree of freedom.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    right_a, right_b = pred_a == y_true, pred_b == y_true
    b = int(np.sum(right_a & ~right_b))
    c = int(np.sum(~right_a & right_b))
    n = b + c
    if n == 0:   # the models never disagree on correctness
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "exact"}
    if n < 25:
        p = binomtest(b, n, 0.5).pvalue
        return {"b": b, "c": c, "statistic": float(b), "p_value": float(p), "method": "exact"}
    stat = (abs(b - c) - 1) ** 2 / n
    return {"b": b, "c": c, "statistic": float(stat), "p_value": float(chi2.sf(stat, 1)),
            "method": "chi2_cc"}


def _ensure_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, title: str, path: str) -> None:
    """2x2 confusion matrix, gray scale, counts and row % in every cell, 300 dpi PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=max(cm.max(), 1) * 1.25)
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.6
            ax.text(j, i, f"{cm[i, j]}\n({row_pct[i, j]:.1f}%)", ha="center", va="center",
                    fontsize=11, color="white" if dark else "black")
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    _ensure_dir(path)
    fig.savefig(path, dpi=300)
    plt.close(fig)


_METHOD_NAMES = {"tfidf_rf": "TF-IDF + RF", "codebert": "CodeBERT"}


def _drop_chart_data(summary: dict) -> dict:
    """Normalise both accepted summary formats to {display_name: {"original", "renamed"}}."""
    if any(m in summary for m in _METHOD_NAMES):          # results/metrics_summary.json
        return {_METHOD_NAMES[m]: {"original": summary[m]["original_test"],
                                   "renamed": summary[m]["renamed_test"]}
                for m in _METHOD_NAMES if m in summary}
    return summary


def plot_drop_chart(summary: dict, path: str) -> None:
    """Grouped bars: accuracy and macro F1, per method, original vs renamed test set.

    Accepts either the full results/metrics_summary.json dict, i.e.
        {"tfidf_rf": {"original_test": AGG, "renamed_test": AGG, ...},
         "codebert": {"original_test": AGG, "renamed_test": AGG, ...}, ...other keys}
    or a plain {"TF-IDF + RF": {"original": AGG, "renamed": AGG}, "CodeBERT": {...}},
    where AGG = {"accuracy": {"mean", "std"}, "macro_f1": {"mean", "std"}, ...}
    (the output of aggregate_seeds). Error bars = std over seeds. The number above
    each pair is the drop (original - renamed) in percentage points.
    """
    summary = _drop_chart_data(summary)
    methods = list(summary)
    conditions = [("original", "Original test", "white", ""),
                  ("renamed", "Renamed test", "0.55", "///")]
    panels = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1")]
    width = 0.36
    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
    for ax, (metric, label) in zip(axes, panels):
        tops = np.zeros(len(methods))
        for k, (cond, cond_label, color, hatch) in enumerate(conditions):
            means = np.array([summary[m][cond][metric]["mean"] for m in methods])
            stds = np.nan_to_num([summary[m][cond][metric]["std"] for m in methods])
            pos = x + (k - 0.5) * width
            bars = ax.bar(pos, means, width, yerr=stds, capsize=4, color=color,
                          edgecolor="black", hatch=hatch, label=cond_label)
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02, f"{mean:.3f}",
                        ha="center", va="bottom", fontsize=8, rotation=90,
                        bbox=dict(facecolor="white", edgecolor="none", pad=1))
            tops = np.maximum(tops, means + stds)
        for i, m in enumerate(methods):
            drop = (summary[m]["original"][metric]["mean"]
                    - summary[m]["renamed"][metric]["mean"]) * 100
            ax.text(x[i], tops[i] + 0.02, f"drop {drop:.1f} pp", ha="center", fontsize=9)
        ax.set_xticks(x, labels=methods)
        ax.set_title(label)
        ax.set_xlabel("Method")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="0.85", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Score (mean over 3 seeds, bars = std)")
    axes[0].legend(loc="upper left", fontsize=8, frameon=True)
    fig.tight_layout()
    _ensure_dir(path)
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import tempfile

    # compute_metrics -- hand check:
    # y_true 1 1 1 1 0 0 0 0 0 0 / y_pred 1 1 0 0 0 0 0 0 1 1
    # correct 6/10 -> acc 0.6; vulnerable: TP 2 of 4 -> recall 0.5;
    # F1(1): P=2/4, R=2/4 -> 0.5; F1(0): P=4/6, R=4/6 -> 0.6667; macro 0.5833
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
    m = compute_metrics(y_true, y_pred)
    print("metrics", m)
    assert np.isclose(m["accuracy"], 0.6) and np.isclose(m["vuln_recall"], 0.5)
    assert np.isclose(m["macro_f1"], (0.5 + 2 / 3) / 2)

    # majority_baseline -- train has 3 zeros, 2 ones -> predict 0 everywhere:
    # acc 6/10; recall 0; F1(0) = 2*0.6*1/1.6 = 0.75, F1(1) = 0 -> macro 0.375
    base = majority_baseline([0, 0, 0, 1, 1], y_true)
    print("baseline", base)
    assert base["majority_label"] == 0 and np.isclose(base["accuracy"], 0.6)
    assert np.isclose(base["macro_f1"], 0.375) and base["vuln_recall"] == 0.0

    # aggregate_seeds -- 0.60, 0.62, 0.64 -> mean 0.62, std (ddof=1) 0.02
    agg = aggregate_seeds([{"accuracy": 0.60}, {"accuracy": 0.62}, {"accuracy": 0.64}])
    print("aggregate", agg)
    assert np.isclose(agg["accuracy"]["mean"], 0.62) and np.isclose(agg["accuracy"]["std"], 0.02)

    # mcnemar exact -- b=3, c=1: P(X>=3 | n=4, 0.5) = 5/16, two-sided = 10/16 = 0.625
    yt = np.ones(10, dtype=int)
    a = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 0])
    bb = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0])
    r = mcnemar_test(yt, a, bb)
    print("mcnemar exact", r)
    assert (r["b"], r["c"], r["method"]) == (3, 1, "exact") and np.isclose(r["p_value"], 0.625)

    # mcnemar chi2 -- b=30, c=10: (|30-10|-1)^2 / 40 = 361/40 = 9.025, p ~ 0.00266
    yt = np.ones(50, dtype=int)
    a = np.r_[np.ones(30), np.zeros(10), np.ones(10)].astype(int)
    bb = np.r_[np.zeros(30), np.ones(10), np.ones(10)].astype(int)
    r = mcnemar_test(yt, a, bb)
    print("mcnemar chi2", r)
    assert (r["b"], r["c"], r["method"]) == (30, 10, "chi2_cc")
    assert np.isclose(r["statistic"], 9.025) and abs(r["p_value"] - 0.00266) < 1e-4

    out = tempfile.mkdtemp()
    plot_confusion_matrix(y_true, y_pred, "Self-test confusion matrix",
                          os.path.join(out, "cm.png"))
    s = lambda mu, sd: {"mean": mu, "std": sd}
    fake = {"TF-IDF + RF": {"original": {"accuracy": s(0.60, 0.01), "macro_f1": s(0.58, 0.01)},
                            "renamed": {"accuracy": s(0.55, 0.01), "macro_f1": s(0.52, 0.02)}},
            "CodeBERT": {"original": {"accuracy": s(0.62, 0.01), "macro_f1": s(0.61, 0.01)},
                         "renamed": {"accuracy": s(0.60, 0.01), "macro_f1": s(0.59, 0.01)}}}
    plot_drop_chart(fake, os.path.join(out, "drop.png"))
    # Same chart from the metrics_summary.json format used by scripts/run_experiments.py
    full = {"primary_seed": 42, "mcnemar": {},
            "tfidf_rf": {"best_params": {}, "original_test": fake["TF-IDF + RF"]["original"],
                         "renamed_test": fake["TF-IDF + RF"]["renamed"], "drop": {}},
            "codebert": {"best_params": {}, "original_test": fake["CodeBERT"]["original"],
                         "renamed_test": fake["CodeBERT"]["renamed"], "drop": {}}}
    plot_drop_chart(full, os.path.join(out, "drop_summary.png"))
    print("figures written to", out)
    print("evaluation self-test passed")
