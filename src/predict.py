"""Try it: classify one new C function with the saved models (no retraining).

Owner: Khalid Waleed Alhilal

The input goes through the same cleaning as the training data (data_prep.clean_code).
The selected model (higher mean dev macro F1, see results/metrics_summary.json) is shown
first, the other model is printed for comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODEBERT_DIR = REPO_ROOT / "models" / "codebert_seed42"
TFIDF_PATH = REPO_ROOT / "models" / "tfidf_rf_seed42.joblib"
SUMMARY_PATH = REPO_ROOT / "results" / "metrics_summary.json"
THRESHOLD = 0.5

_cache: dict = {}


def _selected_model() -> str:
    """Name of the model chosen for Try it ("codebert" or "tfidf_rf")."""
    try:
        with open(SUMMARY_PATH, encoding="utf-8") as fh:
            name = json.load(fh).get("selected_model_for_try_it", "codebert")
        return name if name in ("codebert", "tfidf_rf") else "codebert"
    except FileNotFoundError:
        return "codebert"


def _probabilities(clean: str) -> dict[str, float]:
    """P(vulnerable) from each saved model that is available."""
    probs: dict[str, float] = {}
    if CODEBERT_DIR.exists():
        from src import codebert_model
        if "codebert" not in _cache:
            _cache["codebert"] = codebert_model.load(str(CODEBERT_DIR))
        model, tok = _cache["codebert"]
        probs["codebert"] = float(codebert_model.predict_proba(model, tok, [clean])[0])
    if TFIDF_PATH.exists():
        from src import tfidf_rf
        if "tfidf_rf" not in _cache:
            _cache["tfidf_rf"] = tfidf_rf.load(str(TFIDF_PATH))
        probs["tfidf_rf"] = float(tfidf_rf.predict_proba(_cache["tfidf_rf"], [clean])[0])
    return probs


def try_it(code: str) -> None:
    """Print the predicted label (Vulnerable / Safe) and P(vulnerable) for one C function."""
    if code is None or not str(code).strip():
        print("Please paste one C function (the input is empty).")
        return
    from src.data_prep import clean_code
    clean = clean_code(str(code))
    if not clean:
        print("Only comments or whitespace were found. Please paste the code of one C function.")
        return
    probs = _probabilities(clean)
    if not probs:
        print("No saved model found in models/. Run the download cell in the notebook first.")
        return
    names = {"codebert": "CodeBERT (fine-tuned)", "tfidf_rf": "TF-IDF + Random Forest"}
    selected = _selected_model()
    order = [selected] + [m for m in probs if m != selected]
    for i, m in enumerate(order):
        if m not in probs:
            continue
        p = probs[m]
        label = "Vulnerable" if p >= THRESHOLD else "Safe"
        tag = "selected model" if i == 0 else "for comparison"
        print(f"{names[m]:<26} ({tag}): {label:<10}  P(vulnerable) = {p:.3f}")


if __name__ == "__main__":
    try_it("")
    try_it("void copy(char *src) { char buf[64]; strcpy(buf, src); }")
