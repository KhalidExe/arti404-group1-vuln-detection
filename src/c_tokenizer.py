"""C tokenizer used by the TF-IDF features.

Owner: Mishary Waleed AlAwadh

Splits on operators and punctuation (multi-character operators such as ``->``, ``==``,
``<<=`` stay one token), keeps identifiers, keywords and numbers, maps string literals to
``<STR>`` and char literals to ``<CHR>``, and adds the lower-cased camelCase/snake_case
sub-parts of each identifier (``bufLen`` -> ``bufLen``, ``buf``, ``len``).
"""
from __future__ import annotations


def tokenize_c(code: str) -> list[str]:
    """Tokenize one (cleaned) C function into a list of tokens."""
    raise NotImplementedError


if __name__ == "__main__":
    # Self-test: print tokenize_c('if (len > n) memcpy(dstBuf, src_ptr, len);')
    raise NotImplementedError
