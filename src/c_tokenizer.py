"""C tokenizer for Method 1 (TF-IDF + Random Forest).

Owner: Mishary Waleed AlAwadh

tokenize_c(code) splits one C function into tokens:
  * multi-character operators stay one token (->, ++, <<=, ...),
  * identifiers and keywords are kept; an identifier made of several parts
    (camelCase or snake_case) is followed by its lower-cased sub-parts,
    e.g. bufLen -> bufLen, buf, len   and   buf_len -> buf_len, buf, len,
  * numbers are kept as written,
  * string literals -> <STR>, char literals -> <CHR>,
  * comments are skipped (in case any survive cleaning).
"""

from __future__ import annotations

import re

# Longest operators first so that "<<=" wins over "<<" and "<".
_OPERATORS = [
    "<<=", ">>=", "...",
    "->", "++", "--", "==", "!=", "<=", ">=", "&&", "||", "<<", ">>",
    "+=", "-=", "*=", "/=", "&=", "|=", "^=", "%=", "##",
]
_SINGLE = r"[+\-*/%=<>!&|^~?:;,.()\[\]{}#\\]"

_TOKEN_RE = re.compile(
    r"""
    (?P<comment>//[^\n]*|/\*.*?\*/)
  | (?P<string>(?:L|u8|u|U)?"(?:\\.|[^"\\\n])*")
  | (?P<char>(?:L|u|U)?'(?:\\.|[^'\\\n])*')
  | (?P<number>0[xX][0-9a-fA-F]+[uUlL]*
              |(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[uUlLfF]*)
  | (?P<ident>[A-Za-z_]\w*)
  | (?P<op>"""
    + "|".join(re.escape(op) for op in _OPERATORS)
    + "|" + _SINGLE + r""")
  | (?P<space>\s+)
  | (?P<other>.)
    """,
    re.VERBOSE | re.DOTALL,
)

# Parts of one underscore-free chunk: "HTTPServer2Buf" -> HTTP, Server2, Buf.
_CAMEL_RE = re.compile(r"[A-Z]+\d*(?![a-z])|[A-Z]?[a-z]+\d*|\d+")


def split_identifier(name: str) -> list[str]:
    """Return the lower-cased parts of an identifier (camelCase and snake_case)."""
    parts: list[str] = []
    for chunk in name.split("_"):
        if chunk:
            parts.extend(p.lower() for p in _CAMEL_RE.findall(chunk))
    return parts


def tokenize_c(code: str) -> list[str]:
    """Tokenize the source code of one C function."""
    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(code):
        kind = m.lastgroup
        if kind in ("space", "comment"):
            continue
        text = m.group()
        if kind == "string":
            tokens.append("<STR>")
        elif kind == "char":
            tokens.append("<CHR>")
        elif kind == "ident":
            tokens.append(text)
            parts = split_identifier(text)
            if len(parts) > 1:          # no sub-parts for one-part names
                tokens.extend(parts)
        else:                           # number, op, other
            tokens.append(text)
    return tokens


if __name__ == "__main__":
    sample = 'if (len > n) memcpy(dstBuf, src_ptr, len);'
    print(sample)
    print(tokenize_c(sample))
    assert tokenize_c(sample) == [
        "if", "(", "len", ">", "n", ")", "memcpy", "(",
        "dstBuf", "dst", "buf", ",", "src_ptr", "src", "ptr", ",",
        "len", ")", ";",
    ]

    more = ('p->size <<= 2; x += 0x1Fu; s = "a \\"b\\""; c = \'\\n\'; '
            'f(a, ...); i++ != --j && k || 3.5e-2f; /* gone */ AVCodecContext *VAR1;')
    print(more)
    print(tokenize_c(more))
    t = tokenize_c(more)
    for expected in ["->", "<<=", "+=", "0x1Fu", "<STR>", "<CHR>", "...", "++",
                     "!=", "--", "&&", "||", "3.5e-2f", "AVCodecContext",
                     "av", "codec", "context", "VAR1"]:
        assert expected in t, expected
    assert "gone" not in t and "var" not in t   # comment skipped, VAR1 not split
    print("c_tokenizer self-test passed")
