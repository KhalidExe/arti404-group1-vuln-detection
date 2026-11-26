"""Data loading, cleaning, de-duplication and splits.

Owner: Ali Khalid Albinali

Dataset: CodeXGLUE Defect Detection (Devign), Hugging Face id
``google/code_x_glue_cc_defect_detection``. HF columns: ``id``, ``func``, ``target``.
HF split ``validation`` is called ``dev`` everywhere in this project.

Output format used by every other module: a pandas DataFrame with columns
``id`` (int), ``code`` (str, cleaned), ``label`` (int: 1 = vulnerable, 0 = safe), ``hash`` (str).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

HF_DATASET_ID = "google/code_x_glue_cc_defect_detection"
SPLITS = ("train", "dev", "test")
OFFICIAL_SIZES = {"train": 21854, "dev": 2732, "test": 2732}

# HF split name -> project split name
_HF_TO_PROJECT = {"train": "train", "validation": "dev", "test": "test"}
_RAW_COLUMNS = ["id", "func", "target"]


# ----------------------------------------------------------------------------- loading

def _raw_file(raw_dir: Path, name: str) -> Path | None:
    """Find the parquet file of one split. ``dev`` may be saved as dev.parquet or validation.parquet."""
    candidates = [name] + (["validation"] if name == "dev" else [])
    for stem in candidates:
        path = raw_dir / f"{stem}.parquet"
        if path.exists():
            return path
    return None


def _tidy_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df[_RAW_COLUMNS].copy()
    df["id"] = df["id"].astype(int)
    df["func"] = df["func"].astype(str)
    df["target"] = df["target"].astype(int)
    return df.reset_index(drop=True)


def load_raw(raw_dir: str = "data/raw") -> dict[str, pd.DataFrame]:
    """Load the three official splits.

    Use the parquet files in ``raw_dir`` if they exist, otherwise download them with
    ``datasets.load_dataset(HF_DATASET_ID)`` and save them to ``raw_dir``.

    Returns:
        ``{"train": df, "dev": df, "test": df}`` with columns ``id``, ``func``, ``target``.
    """
    raw_path = Path(raw_dir)
    files = {name: _raw_file(raw_path, name) for name in SPLITS}

    if any(path is None for path in files.values()):
        from datasets import load_dataset

        raw_path.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(HF_DATASET_ID)
        for hf_name in ds:
            # same file names as `scripts/run_experiments.py download` (train / validation / test)
            ds[hf_name].to_pandas().to_parquet(raw_path / f"{hf_name}.parquet", index=False)
        files = {name: _raw_file(raw_path, name) for name in SPLITS}

    return {name: _tidy_raw(pd.read_parquet(path)) for name, path in files.items()}


# ----------------------------------------------------------------------------- cleaning

def _strip_comments(code: str) -> str:
    """Remove C comments with a small state machine that knows about string and char literals.

    States: normal code, inside "..." , inside '...', inside // comment, inside /* */ comment.
    Inside a literal a backslash escapes the next character, so "a\\"// b" stays one string.
    A comment is replaced by one space so that ``a/**/b`` does not become the token ``ab``.
    A literal that is not closed before a newline is ended at the newline (C does not allow
    raw newlines in literals), so one broken quote cannot swallow the rest of the function.
    """
    out: list[str] = []
    i, n = 0, len(code)
    while i < n:
        ch = code[i]
        nxt = code[i + 1] if i + 1 < n else ""

        if ch == "/" and nxt == "/":                      # line comment
            i += 2
            while i < n and code[i] != "\n":
                if code[i] == "\\" and i + 1 < n and code[i + 1] == "\n":
                    i += 2                                # backslash-newline continues the comment
                    continue
                i += 1
            out.append(" ")
            continue

        if ch == "/" and nxt == "*":                      # block comment
            end = code.find("*/", i + 2)
            i = n if end == -1 else end + 2
            out.append(" ")
            continue

        if ch in ('"', "'"):                              # string or char literal: copy as is
            quote = ch
            j = i + 1
            while j < n:
                c = code[j]
                if c == "\\" and j + 1 < n:
                    j += 2
                    continue
                if c == quote:
                    j += 1
                    break
                if c == "\n":                             # unterminated literal
                    break
                j += 1
            out.append(code[i:j])
            i = j
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def _collapse_whitespace(code: str) -> str:
    """Collapse every run of whitespace outside string/char literals to one space and strip.

    Text inside literals is kept byte for byte, so the program keeps its meaning.
    """
    out: list[str] = []
    i, n = 0, len(code)
    pending_space = False
    while i < n:
        ch = code[i]
        if ch.isspace():
            pending_space = True
            i += 1
            continue
        if pending_space and out:
            out.append(" ")
        pending_space = False

        if ch in ('"', "'"):
            quote = ch
            j = i + 1
            while j < n:
                c = code[j]
                if c == "\\" and j + 1 < n:
                    j += 2
                    continue
                if c == quote:
                    j += 1
                    break
                if c == "\n":
                    break
                j += 1
            out.append(code[i:j])
            i = j
            continue

        out.append(ch)
        i += 1
    return "".join(out)


def clean_code(code: str) -> str:
    """Remove ``//`` and ``/* */`` comments without touching string or char literals,
    collapse every run of whitespace to one space, and strip.

    This exact function is used for model input, the renamed test set and the Try it cell.
    """
    if code is None:
        return ""
    return _collapse_whitespace(_strip_comments(str(code))).strip()


def code_hash(code: str) -> str:
    """sha1 of the cleaned code (hex)."""
    return hashlib.sha1(code.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- splits

def build_clean_splits(raw: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    """Clean, hash and de-duplicate the splits.

    Rules (in this order, every count goes into the stats dict):
      1. ``code = clean_code(func)``, ``hash = sha1(code.encode()).hexdigest()``, ``label = target``.
      2. Label conflict: same hash with different labels anywhere -> remove all rows of that hash.
      3. Cross-split duplicate: keep the hash only in the earliest split (train > dev > test).
      4. Within-split duplicate: keep the first row of each hash.
      5. ``assert`` that no hash appears in two splits.

    Returns:
        (splits, stats). ``splits`` has keys train/dev/test with columns id, code, label, hash.
        ``stats`` has the shape of ``metrics_summary["data"]``::

            {"official": {"train": 21854, "dev": 2732, "test": 2732},
             "after_cleaning": {"train": {"n", "vulnerable", "safe"}, "dev": {...}, "test": {...}},
             "duplicates": {"within_split_removed", "cross_split_removed",
                            "label_conflict_groups", "label_conflict_rows_removed"}}
    """
    official = {name: int(len(raw[name])) for name in SPLITS}

    # 1. clean, hash, label
    frames = []
    for name in SPLITS:
        df = raw[name]
        code = df["func"].map(clean_code)
        frames.append(pd.DataFrame({
            "split": name,
            "id": df["id"].astype(int).values,
            "code": code.values,
            "label": df["target"].astype(int).values,
            "hash": code.map(code_hash).values,
        }))
    allrows = pd.concat(frames, ignore_index=True)

    # 2. label conflicts: a hash with more than one label anywhere -> drop every row of it
    n_labels = allrows.groupby("hash")["label"].nunique()
    conflict_hashes = set(n_labels[n_labels > 1].index)
    conflict_mask = allrows["hash"].isin(conflict_hashes)
    label_conflict_groups = len(conflict_hashes)
    label_conflict_rows_removed = int(conflict_mask.sum())
    allrows = allrows[~conflict_mask]

    # 3. cross-split duplicates: keep a hash only in the earliest split it appears in
    order = {name: k for k, name in enumerate(SPLITS)}
    split_rank = allrows["split"].map(order)
    first_rank = split_rank.groupby(allrows["hash"]).transform("min")
    cross_mask = split_rank > first_rank
    cross_split_removed = int(cross_mask.sum())
    allrows = allrows[~cross_mask]

    # 4. within-split duplicates: keep the first row of each hash (original row order)
    within_mask = allrows.duplicated(subset=["split", "hash"], keep="first")
    within_split_removed = int(within_mask.sum())
    allrows = allrows[~within_mask]

    splits: dict[str, pd.DataFrame] = {}
    for name in SPLITS:
        part = allrows[allrows["split"] == name][["id", "code", "label", "hash"]]
        part = part.reset_index(drop=True)
        part["id"] = part["id"].astype(int)
        part["label"] = part["label"].astype(int)
        splits[name] = part

    # 5. final check: no hash in two splits, no hash twice inside one split
    for k, a in enumerate(SPLITS):
        assert splits[a]["hash"].is_unique, f"duplicate hash left inside {a}"
        for b in SPLITS[k + 1:]:
            shared = set(splits[a]["hash"]) & set(splits[b]["hash"])
            assert not shared, f"{len(shared)} hashes appear in both {a} and {b}"

    stats = {
        "official": official,
        "after_cleaning": {
            name: {
                "n": int(len(splits[name])),
                "vulnerable": int((splits[name]["label"] == 1).sum()),
                "safe": int((splits[name]["label"] == 0).sum()),
            }
            for name in SPLITS
        },
        "duplicates": {
            "within_split_removed": within_split_removed,
            "cross_split_removed": cross_split_removed,
            "label_conflict_groups": label_conflict_groups,
            "label_conflict_rows_removed": label_conflict_rows_removed,
        },
    }
    return splits, stats


def save_splits(splits: dict[str, pd.DataFrame], out_dir: str = "data/processed") -> None:
    """Save each split as ``<out_dir>/<name>.parquet``."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_parquet(out_path / f"{name}.parquet", index=False)


def load_splits(in_dir: str = "data/processed", names=("train", "dev", "test")) -> dict[str, pd.DataFrame]:
    """Load ``<in_dir>/<name>.parquet`` for every name in ``names``."""
    in_path = Path(in_dir)
    out = {}
    for name in names:
        path = in_path / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python -m src.data_prep "
                                    f"(or python scripts/run_experiments.py clean)")
        out[name] = pd.read_parquet(path)
    return out


# ----------------------------------------------------------------------------- self-test

_CLEAN_CASES = [
    ("int a; // x", "int a;"),
    ('char *s = "// not a comment";', 'char *s = "// not a comment";'),
    ("a = 1; /* multi\nline */ b = 2;", "a = 1; b = 2;"),
    ("c = '/';", "c = '/';"),
    # extra cases
    ('p = "a /* b */ c"; /* real */', 'p = "a /* b */ c";'),
    ('s = "quote \\" // still string"; // gone', 's = "quote \\" // still string";'),
    ("c = '\\''; // x", "c = '\\'';"),
    ("x = a/**/b;", "x = a b;"),
    ("int\n\n   f(void)\t{\r\n return 0; }", "int f(void) { return 0; }"),
    ("y = a / b; // divide", "y = a / b;"),
    ('s = "two  spaces";', 's = "two  spaces";'),
    ("// only a comment", ""),
    ("a; // line \\\n still comment\nb;", "a; b;"),
]


def _self_test_clean_code() -> None:
    failed = 0
    for src, want in _CLEAN_CASES:
        got = clean_code(src)
        ok = got == want
        failed += not ok
        print(f"[{'ok' if ok else 'FAIL'}] {src!r} -> {got!r}" + ("" if ok else f"   expected {want!r}"))
    assert failed == 0, f"{failed} clean_code case(s) failed"
    print(f"clean_code: all {len(_CLEAN_CASES)} cases passed\n")


if __name__ == "__main__":
    # Run from the repo root:  python -m src.data_prep
    import json

    _self_test_clean_code()

    raw = load_raw("data/raw")
    for name in SPLITS:
        print(f"raw {name}: {len(raw[name])} rows")
    splits, stats = build_clean_splits(raw)
    print(json.dumps(stats, indent=2))

    for name, want in OFFICIAL_SIZES.items():
        if stats["official"][name] != want:
            print(f"WARNING: official {name} size is {stats['official'][name]}, expected {want}")

    save_splits(splits, "data/processed")
    back = load_splits("data/processed")
    for name in SPLITS:
        assert len(back[name]) == len(splits[name])
        assert list(back[name].columns) == ["id", "code", "label", "hash"]
    print("saved and re-loaded data/processed/{train,dev,test}.parquet")
