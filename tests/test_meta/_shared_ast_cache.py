"""Process-wide cache for ``path -> parsed AST``, shared across every meta-test that
independently walks ``src/mlframe/**/*.py``.

Each scanner keeps its OWN directory walk and skip logic (they differ: some skip
``_benchmarks/``, some skip ``legacy/``, etc.) -- only the ``read_text()`` +
``ast.parse()`` pair is shared. Under pytest-xdist, ``@cache`` is process-scoped
(one cache per worker), so a file visited by several scanner tests that happen to
land on the same worker is read and parsed once instead of once per scanner.
"""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path


@cache
def source_text(path: Path) -> str | None:
    """Read ``path`` once; returns None on decode/OS errors."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


@cache
def parsed_ast(path: Path) -> ast.Module | None:
    """AST-parse ``path`` once (reusing the cached ``source_text``); returns None on decode/syntax errors."""
    src = source_text(path)
    if src is None:
        return None
    try:
        return ast.parse(src, filename=str(path))
    except SyntaxError:
        return None
