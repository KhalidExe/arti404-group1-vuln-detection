# Do Vulnerability Detectors Learn the Bug or the Names?

**TF-IDF with Random Forest versus fine-tuned CodeBERT on C functions**
ARTI 404 Natural Language Processing · IAU (CCSIT) · Section 9MA2 · Group 1

| Member | Responsibility |
|---|---|
| Khalid Waleed Alhilal (leader) | CodeBERT fine-tuning, seeds and dev tuning, Try it, repository |
| Ali Khalid Albinali | Data loading, cleaning, duplicate check, splits; identifier renaming, renamed test set, augmented training set |
| Mishary Waleed AlAwadh | C tokenizer, TF-IDF + Random Forest, majority baseline, metrics, McNemar, feature and error analysis |

## Question

Which method detects vulnerable C functions more reliably, and which one loses less accuracy and
macro F1 when identifier names are replaced?

Input: one C function as text. Output: `Vulnerable` or `Safe` and the probability of `Vulnerable`.

## Repository layout

| Path | Content | Owner |
|---|---|---|
| `vuln_detection.ipynb` | Main notebook: Setup, Data, Method 1, Method 2, Results, Try it, Sources | all |
| `src/data_prep.py` | Load, clean, hash, de-duplicate, splits, statistics | Ali |
| `src/renaming.py` | tree-sitter identifier renaming, renamed test set, augmented train set | Ali |
| `src/c_tokenizer.py` | C tokenizer for TF-IDF | Mishary |
| `src/tfidf_rf.py` | TF-IDF + Random Forest: tune, train, save, load, predict | Mishary |
| `src/evaluation.py` | Metrics, majority baseline, McNemar, figures | Mishary |
| `src/codebert_model.py` | CodeBERT fine-tuning, early stopping, save, load, predict | Khalid |
| `src/predict.py` | `try_it(code)` used by the notebook | Khalid |
| `scripts/run_experiments.py` | Command-line runner for the heavy experiments | Khalid |
| `results/` | `metrics_summary.json`, CSV tables, figures (all reported numbers come from here) | |
| `data/processed/` | Cleaned splits, renamed test set, augmented train set | |
| `models/` | Saved models (not in git; downloaded from the GitHub Release) | |

## Setup

Tested with Jupyter (local), Python 3.11, CUDA 12.8, NVIDIA RTX 5080 (Windows 11).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

RTX 50-series GPUs need a CUDA 12.8+ PyTorch build. Without a GPU the notebook still runs with
the default settings (it loads the saved models instead of training).

## Data

CodeXGLUE Defect Detection (Devign): 27,318 C functions from FFmpeg and QEMU, label 1 = vulnerable.
Official split 21,854 / 2,732 / 2,732. Downloaded automatically from Hugging Face
(`google/code_x_glue_cc_defect_detection`):

```bash
python scripts/run_experiments.py download
```

## How to run

_To be completed._

## Results

_To be completed. All numbers come from `results/metrics_summary.json`._

## Credits

- Dataset: S. Lu et al., CodeXGLUE (2021); Y. Zhou et al., Devign (2019).
- Pretrained model: `microsoft/codebert-base` (Z. Feng et al., 2020), via Hugging Face Transformers.
- Parsing: tree-sitter and tree-sitter-c.
- Libraries: PyTorch, Transformers, Datasets, scikit-learn, pandas, NumPy, SciPy, statsmodels, Matplotlib.
