"""Package the seed-42 models and upload them to a GitHub Release (needs the GitHub CLI, gh).

Usage, from the repo root after training:
    python scripts/make_release.py            # create release v1.0 (or replace its assets)
    python scripts/make_release.py --dry-run  # only build the files in dist/

Assets (names must match src/assets.py):
    codebert_seed42.zip     the fine-tuned CodeBERT checkpoint (models/codebert_seed42/)
    tfidf_rf_seed42.joblib  the TF-IDF + Random Forest pipeline
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.assets import GITHUB_REPO, RELEASE_TAG  # noqa: E402

MODELS_DIR = REPO_ROOT / "models"
DIST_DIR = REPO_ROOT / "dist"
NOTES = ("Trained models for the ARTI 404 Group 1 project (seed 42): fine-tuned CodeBERT checkpoint and "
         "TF-IDF + Random Forest pipeline. The notebook downloads them automatically.")


def build() -> list[Path]:
    """Zip the CodeBERT folder and copy the joblib file into dist/."""
    DIST_DIR.mkdir(exist_ok=True)
    cb_dir = MODELS_DIR / "codebert_seed42"
    tf_file = MODELS_DIR / "tfidf_rf_seed42.joblib"
    for p in (cb_dir, tf_file):
        if not p.exists():
            sys.exit(f"missing {p}; train the models first")
    zip_path = DIST_DIR / "codebert_seed42.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(cb_dir.iterdir()):
            if f.is_file():
                zf.write(f, f.name)
    tf_copy = DIST_DIR / tf_file.name
    shutil.copy2(tf_file, tf_copy)
    for p in (zip_path, tf_copy):
        print(f"{p.name}: {p.stat().st_size / 1e6:.1f} MB")
    return [zip_path, tf_copy]


def find_gh() -> str:
    gh = shutil.which("gh") or r"C:\Program Files\GitHub CLI\gh.exe"
    if not Path(gh).exists() and not shutil.which("gh"):
        sys.exit("GitHub CLI (gh) not found")
    return gh


def upload(files: list[Path]) -> None:
    """Create the release, or replace the assets if the release already exists."""
    gh = find_gh()
    exists = subprocess.run([gh, "release", "view", RELEASE_TAG, "--repo", GITHUB_REPO],
                            capture_output=True).returncode == 0
    if exists:
        cmd = [gh, "release", "upload", RELEASE_TAG, *map(str, files), "--repo", GITHUB_REPO, "--clobber"]
    else:
        cmd = [gh, "release", "create", RELEASE_TAG, *map(str, files), "--repo", GITHUB_REPO,
               "--title", f"Trained models {RELEASE_TAG}", "--notes", NOTES]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"https://github.com/{GITHUB_REPO}/releases/tag/{RELEASE_TAG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    files = build()
    if not args.dry_run:
        upload(files)
