"""Download the saved models from the GitHub Release when they are not on disk.

Owner: Khalid Waleed Alhilal

The trained models are too large for the git repository (CodeBERT is about 500 MB), so they are
published as assets of a GitHub Release. ``ensure_models()`` downloads and unpacks them into
``models/`` the first time the notebook or ``try_it`` needs them.
"""
from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

GITHUB_REPO = "KhalidExe/arti404-group1-vuln-detection"
RELEASE_TAG = "v1.0"
RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

# asset file name in the release -> path it is unpacked to
ASSETS = {
    "codebert_seed42.zip": MODELS_DIR / "codebert_seed42",
    "tfidf_rf_seed42.joblib": MODELS_DIR / "tfidf_rf_seed42.joblib",
}


def _download(url: str, dest: Path) -> None:
    """Stream a file to disk and print the progress in MB."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def ensure_models(verbose: bool = True) -> None:
    """Download every missing model listed in ASSETS from the GitHub Release."""
    for asset, target in ASSETS.items():
        if target.exists():
            if verbose:
                print(f"found   {target.relative_to(REPO_ROOT)}")
            continue
        url = f"{RELEASE_URL}/{asset}"
        print(f"download {url}")
        if asset.endswith(".zip"):
            archive = MODELS_DIR / asset
            _download(url, archive)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(target)
            archive.unlink()
        else:
            _download(url, target)


if __name__ == "__main__":
    ensure_models()
