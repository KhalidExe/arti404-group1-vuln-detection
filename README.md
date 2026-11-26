# Do Vulnerability Detectors Learn the Bug or the Names?

**TF-IDF with Random Forest versus fine-tuned CodeBERT on C functions**
ARTI 404 Natural Language Processing · Imam Abdulrahman Bin Faisal University (CCSIT) · Section 9MA2 · Group 1
Instructor: Dr. Mohammed Mosuily

| Member | Responsibility |
|---|---|
| Khalid Waleed Alhilal (leader) | CodeBERT fine-tuning, seeds and dev tuning, Try it, repository |
| Ali Khalid Albinali | Data loading, cleaning, duplicate check, splits; identifier renaming, renamed test set, augmented training set |
| Mishary Waleed AlAwadh | C tokenizer, TF-IDF + Random Forest, majority baseline, metrics, McNemar, feature and error analysis |

## Question

Which method detects vulnerable C functions more reliably, and which one loses less accuracy and
macro F1 when identifier names are replaced?

- **Input:** one C function as text. **Output:** `Vulnerable` or `Safe` and the probability of `Vulnerable`.
- **Method 1:** own C tokenizer → TF-IDF (unigrams + bigrams) → Random Forest.
- **Method 2:** fine-tuned `microsoft/codebert-base` (max 512 tokens).
- **Core test:** every user-defined variable and function name in the test set is replaced with
  `VAR1, VAR2, …` / `FUN1, FUN2, …` (tree-sitter). A model that learned the bug pattern should lose little;
  a model that memorised project names should lose more.

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
| `src/assets.py` | Downloads the saved models from the GitHub Release | Khalid |
| `scripts/run_experiments.py` | Command-line runner for every experiment | Khalid |
| `scripts/make_release.py` | Uploads the saved models to the GitHub Release | Khalid |
| `data/processed/` | Cleaned splits, renamed test set, augmented train set | |
| `data/original_examples.json` | Two original labelled C functions per member | all |
| `results/` | `metrics_summary.json`, CSV tables, figures (every reported number comes from here) | |
| `models/` | Saved models (not in git; downloaded from the GitHub Release) | |

## Setup

Tested on Windows 11 with Jupyter (local), Python 3.11, CUDA 12.8 and an NVIDIA RTX 5080 (16 GB).

```bash
git clone https://github.com/KhalidExe/arti404-group1-vuln-detection.git
cd arti404-group1-vuln-detection

python3.11 -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

- RTX 50-series (Blackwell) GPUs need a CUDA 12.8+ PyTorch build; older cu121/cu124 builds do not run on them.
- Check the GPU: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"`.
- Without a GPU, install the CPU build of PyTorch instead (`pip install torch==2.11.0`). The notebook still runs
  with the default settings; only the CodeBERT check in the Results section is slower.

### Package versions

| Package | Version | Package | Version |
|---|---|---|---|
| Python | 3.11.18 | scikit-learn | 1.9.1 |
| torch (cu128) | 2.11.0 | pandas | 3.1.1 |
| transformers | 5.24.1 | numpy | 2.4.6 |
| datasets | 5.2.0 | scipy | 1.17.1 |
| tree-sitter | 0.27.0 | statsmodels | 0.15.0 |
| tree-sitter-c | 0.24.2 | matplotlib | 3.11.2 |
| joblib | 1.6.0 | jupyterlab | 4.6.6 |

## Data

CodeXGLUE Defect Detection (Devign): 27,318 C functions from FFmpeg and QEMU, label 1 = vulnerable,
official split 21,854 / 2,732 / 2,732. The cleaned splits are already in `data/processed/`. To rebuild them:

```bash
python scripts/run_experiments.py download   # Hugging Face google/code_x_glue_cc_defect_detection -> data/raw/
python scripts/run_experiments.py clean      # comments removed, whitespace collapsed, duplicates resolved
python scripts/run_experiments.py rename     # renamed test set + augmented training set
```

## Saved models

The trained seed-42 models are attached to the GitHub Release
[`v1.0`](https://github.com/KhalidExe/arti404-group1-vuln-detection/releases/tag/v1.0):

| File | Content |
|---|---|
| `codebert_seed42.zip` | fine-tuned CodeBERT checkpoint (unpacked to `models/codebert_seed42/`) |
| `tfidf_rf_seed42.joblib` | TF-IDF + Random Forest pipeline (`models/tfidf_rf_seed42.joblib`) |

The notebook and `try_it` download them automatically (`src/assets.py`). Manual download:
`python -m src.assets`.

## How to run

```bash
jupyter lab        # start it in the repository root
```

Open `vuln_detection.ipynb` and choose **Kernel → Restart Kernel and Run All Cells**.

- `RUN_FULL_TRAINING = False` (default): loads the saved models and results, recomputes the seed-42 test
  metrics from the loaded models and checks them against `results/metrics_summary.json`.
- `RUN_FULL_TRAINING = True`: reruns every experiment (about 4 hours on an RTX 5080). The same steps can be
  run from the command line:

```bash
python scripts/run_experiments.py tfidf-tune
python scripts/run_experiments.py tfidf-seeds
python scripts/run_experiments.py length-baseline
python scripts/run_experiments.py codebert-lr-search
python scripts/run_experiments.py codebert-seeds
python scripts/run_experiments.py augment --method both
python scripts/run_experiments.py predict-missing
python scripts/run_experiments.py summarize
```

**Try it:** in the notebook, put one C function in the `code` variable and run the cell, or from Python:

```python
from src.predict import try_it
try_it("void f(char *s) { char b[64]; strcpy(b, s); }")
```

## Results

_Filled in from `results/metrics_summary.json` after the final runs._

## Credits

- **Dataset:** CodeXGLUE Defect Detection [1], built from the Devign data [2]
  (`google/code_x_glue_cc_defect_detection` on Hugging Face).
- **Pretrained model:** `microsoft/codebert-base` [4], via Hugging Face Transformers.
- **Parsing:** tree-sitter and tree-sitter-c.
- **Libraries:** PyTorch, Transformers, Datasets, scikit-learn, pandas, NumPy, SciPy, statsmodels, Matplotlib, joblib.

### References

1. S. Lu et al., "CodeXGLUE: A machine learning benchmark dataset for code understanding and generation," Proc. NeurIPS Datasets and Benchmarks Track, 2021.
2. Y. Zhou, S. Liu, J. Siow, X. Du, and Y. Liu, "Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks," Proc. NeurIPS, 2019.
3. R. Russell et al., "Automated vulnerability detection in source code using deep representation learning," Proc. IEEE ICMLA, 2018, pp. 757–762.
4. Z. Feng et al., "CodeBERT: A pre-trained model for programming and natural languages," Findings of EMNLP, 2020.
5. Z. Li et al., "VulDeePecker: A deep learning-based system for vulnerability detection," Proc. NDSS, 2018.
6. N. Risse and M. Böhme, "Uncovering the limits of machine learning for automatic vulnerability detection," Proc. USENIX Security, 2024.
7. M. Fu and C. Tantithamthavorn, "LineVul: A transformer-based line-level vulnerability prediction," Proc. MSR, 2022.
8. S. Chakraborty, R. Krishna, Y. Ding, and B. Ray, "Deep learning based vulnerability detection: Are we there yet?" IEEE Trans. Softw. Eng., vol. 48, no. 9, pp. 3280–3296, 2022.
