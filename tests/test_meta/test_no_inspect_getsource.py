"""Meta-linter: no test file may use inspect.getsource for assertions.

Memory rule: feedback_behavioral_tests forbids using inspect.getsource()
to assert on prod source-text. AST-walk instead asserts BEHAVIOUR.

This linter scans every tests/**/*.py file and counts violations.
The whitelist allows tests/test_rng_determinism.py to use the related
inspect.getsourcefile pattern (now migrated to mod.__file__).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_TESTS = Path(__file__).resolve().parent.parent  # tests/

# Files that legitimately use inspect.getsourcefile / inspect.getfile (NOT getsource).
# Use forward slashes for cross-platform path comparison.
WHITELIST: set[str] = {
    # AST-only meta-linters; never use literal getsource().
    "test_meta/test_no_inspect_getsource.py",
    "test_meta/test_no_unicode_in_console_output.py",
}


def _find_getsource_calls(path: Path) -> list[tuple[int, str]]:
    """Return list of (line_no, kind) for each inspect.getsource(...) call."""
    try:
        src = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # inspect.getsource(...) or <alias>.getsource(...)
        if isinstance(func, ast.Attribute) and func.attr == "getsource":
            hits.append((node.lineno, "inspect.getsource"))
        # `getsource(...)` after `from inspect import getsource`
        if isinstance(func, ast.Name) and func.id == "getsource":
            hits.append((node.lineno, "getsource (bare)"))
    return hits


def _iter_test_files() -> list[Path]:
    """All `.py` files under tests/ (skips __pycache__ and conftest.py-style fixtures)."""
    files: list[Path] = []
    for p in _REPO_TESTS.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        # Only check test_*.py files; helper modules (conftest, synthetic) are OK
        # if they don't assert behavior but they shouldn't getsource either.
        files.append(p)
    return files


def test_no_inspect_getsource_in_test_files() -> None:
    """Every test_*.py file must avoid inspect.getsource() for assertions.

    A handful of files (whitelist) may use inspect.getsourcefile() to get
    a file path - that's allowed, this lint only targets the source-text
    inspection antipattern.
    """
    offenders: list[str] = []
    for path in _iter_test_files():
        rel = path.relative_to(_REPO_TESTS).as_posix()
        if rel in WHITELIST:
            continue
        hits = _find_getsource_calls(path)
        if hits:
            for ln, kind in hits:
                offenders.append(f"{rel}:{ln} ({kind})")
    assert not offenders, (
        "inspect.getsource() in test files violates feedback_behavioral_tests rule. "
        "Use behavioural tests (monkeypatch, raises, identity checks) instead. "
        f"Offenders:\n  " + "\n  ".join(offenders[:50])
    )
