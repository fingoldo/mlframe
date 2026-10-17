"""Lexical helper for meta-tests that pin a rationale comment marker next to an ``except`` handler.

A comment is not observable at runtime, so those tests cannot call the code. They tokenize the module instead and look
only at real COMMENT tokens, so a marker quoted inside a string literal or docstring does not count.
"""

from __future__ import annotations

import tokenize
from types import ModuleType


def comment_text(*modules: ModuleType) -> str:
    """Return every comment token of *modules*, one per line, in module order."""
    comments: list[str] = []
    for module in modules:
        with tokenize.open(module.__file__) as handle:
            comments.extend(tok.string for tok in tokenize.generate_tokens(handle.readline) if tok.type == tokenize.COMMENT)
    return "\n".join(comments)
