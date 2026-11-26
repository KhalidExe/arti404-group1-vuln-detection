"""Identifier renaming with tree-sitter (the renamed test set and the augmented train set).

Owner: Ali Khalid Albinali

Every user-defined function name becomes FUN1, FUN2, ... and every other user-defined
identifier becomes VAR1, VAR2, ..., numbered by first appearance inside each function.
Keywords, library calls (LIBRARY_FUNCTIONS), KEEP_IDENTIFIERS, ALL_CAPS names, struct
fields (field_identifier), type names (type_identifier) and goto labels are not changed,
so the program does the same thing and the label stays valid.
"""
from __future__ import annotations

import pandas as pd

# C standard library + common POSIX names that are kept as-is when called.
LIBRARY_FUNCTIONS: frozenset[str] = frozenset()

# Names that are never renamed (NULL, true, false, errno, stdin, stdout, stderr, EOF, ...).
KEEP_IDENTIFIERS: frozenset[str] = frozenset()


def rename_identifiers(code: str) -> tuple[str, dict]:
    """Rename user-defined identifiers in one C function.

    Returns:
        (renamed_code, info) with
        ``info = {"changed": bool, "n_var": int, "n_fun": int, "mapping": dict[str, str],
                  "parse_errors_before": int, "parse_errors_after": int}``
    """
    raise NotImplementedError


def build_renamed_split(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rename every function in ``df`` (columns id, code, label).

    Returns:
        (renamed_df with the same ids and labels, stats) with stats shaped like
        ``metrics_summary["renaming"]``::

            {"test_functions_changed_pct", "mean_identifiers_renamed",
             "mean_var_renamed", "mean_fun_renamed", "new_parse_errors"}
    """
    raise NotImplementedError


def build_augmented_train(train_df: pd.DataFrame) -> pd.DataFrame:
    """Original rows + a renamed copy of every row (same labels), with a bool column ``is_renamed``."""
    raise NotImplementedError


if __name__ == "__main__":
    # Self-test: print before/after for the copy_name example and 2 real test functions,
    # then write results/renaming_examples.md.
    raise NotImplementedError
