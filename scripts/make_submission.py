"""Build the course submission ZIP: report, slides, notebook + code + results, GitHub links.

Usage, from the repo root (after the report and slides are final):
    python scripts/make_submission.py                  # builds ../submission/Group1_Project.zip
    python scripts/make_submission.py --group-id G01   # other group id string
    python scripts/make_submission.py --dry-run        # list the files and sizes only

Layout of the ZIP:
    report/            <Group>_Report.pdf (report + appendix) and the .docx sources
    slides/            <Group>_Slides.pptx
    notebook/          vuln_detection.ipynb, src/, scripts/, requirements.txt, README.md,
                       results/ (json, csv, md, figures; preds/ only if the ZIP stays under 50 MB),
                       data/original_examples.json
    GITHUB_LINK.txt    repository and Release URLs
Data parquet files and trained models are not included: the README explains how to download them.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.assets import GITHUB_REPO, RELEASE_TAG, RELEASE_URL  # noqa: E402

PROJECT_ROOT = REPO_ROOT.parent
REPO_URL = f"https://github.com/{GITHUB_REPO}"
RELEASE_PAGE = f"{REPO_URL}/releases/tag/{RELEASE_TAG}"
MAX_MB = 50
NOTEBOOK_FILES = ["vuln_detection.ipynb", "requirements.txt", "README.md", "data/original_examples.json"]
NOTEBOOK_DIRS = ["src", "scripts", "results"]


def tracked_files() -> list[str]:
    """Files under version control (so caches and local helper files are never packed)."""
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def collect(group: str, paper_dir: Path, include_preds: bool) -> list[tuple[Path, str]]:
    """(source path, path inside the ZIP) for every file that goes into the submission."""
    items: list[tuple[Path, str]] = []
    report = [paper_dir / f"{group}_Report.pdf", paper_dir / f"{group}_Report.docx"]
    appendix = paper_dir / f"{group}_Appendix.docx"
    slides = paper_dir / f"{group}_Slides.pptx"
    missing = [p for p in report + [slides] if not p.exists()]
    if missing:
        sys.exit("missing paper files: " + ", ".join(str(p) for p in missing))
    items += [(p, f"report/{p.name}") for p in report]
    if appendix.exists():
        items.append((appendix, f"report/{appendix.name}"))
    items.append((slides, f"slides/{slides.name}"))

    tracked = tracked_files()
    for rel in tracked:
        top = rel.split("/")[0]
        if rel in NOTEBOOK_FILES or top in NOTEBOOK_DIRS:
            if rel.endswith(".gitkeep") or rel.endswith(".parquet"):
                continue
            if rel.startswith("results/preds/") and not include_preds:
                continue
            items.append((REPO_ROOT / rel, f"notebook/{rel}"))
    return items


def build(zip_path: Path, items: list[tuple[Path, str]]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    links = (f"GitHub repository: {REPO_URL}\n"
             f"Trained models (GitHub Release {RELEASE_TAG}): {RELEASE_PAGE}\n"
             f"  {RELEASE_URL}/codebert_seed42.zip\n  {RELEASE_URL}/tfidf_rf_seed42.joblib\n")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc in items:
            zf.write(src, arc)
        zf.writestr("GITHUB_LINK.txt", links)


def check_online() -> dict:
    """Repository must be public and the Release assets must be downloadable."""
    res = {}
    with urllib.request.urlopen(f"https://api.github.com/repos/{GITHUB_REPO}") as r:
        res["repo_public"] = not json.load(r)["private"]
    for asset in ("codebert_seed42.zip", "tfidf_rf_seed42.joblib"):
        req = urllib.request.Request(f"{RELEASE_URL}/{asset}", method="HEAD")
        with urllib.request.urlopen(req) as r:
            res[asset] = {"status": r.status, "MB": round(int(r.headers.get("Content-Length", 0)) / 1e6, 1)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group-id", default="Group1")
    ap.add_argument("--paper-dir", default=str(PROJECT_ROOT / "paper"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "submission"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    zip_path = Path(args.out_dir) / f"{args.group_id}_Project.zip"
    items = collect(args.group_id, Path(args.paper_dir), include_preds=True)
    total = sum(p.stat().st_size for p, _ in items) / 1e6
    print(f"{len(items)} files, {total:.1f} MB before compression")
    if args.dry_run:
        for p, arc in items:
            print(f"  {arc}  ({p.stat().st_size / 1e3:.0f} kB)")
        print(json.dumps(check_online(), indent=2))
        return

    build(zip_path, items)
    if zip_path.stat().st_size / 1e6 > MAX_MB:
        print(f"ZIP above {MAX_MB} MB, rebuilding without results/preds/")
        items = collect(args.group_id, Path(args.paper_dir), include_preds=False)
        build(zip_path, items)

    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        names = zf.namelist()
    assert bad is None, f"corrupt entry in zip: {bad}"
    for n in names:
        print("  " + n)
    print(f"{zip_path}  {zip_path.stat().st_size / 1e6:.1f} MB, {len(names)} entries, zip test OK")
    print(json.dumps(check_online(), indent=2))


if __name__ == "__main__":
    main()
