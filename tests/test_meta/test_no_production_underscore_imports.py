"""Meta-test: no production code (non-test, non-bench, non-profile) imports an underscore-prefixed module from ``mlframe.training.core`` outside its own sibling cluster.

Underscore modules under ``src/mlframe/training/core`` are internal-only -- their signatures and names can change at any time without a deprecation cycle. The public surface is the names re-exported from ``mlframe.training.core/__init__.py``. Tests and ``_benchmarks/``/``_profile_*`` harnesses MAY import them directly for white-box coverage; production callers must NOT.

This sensor walks the AST of every file under ``src/mlframe`` (production code) and asserts no ``from mlframe.training.core._<name> import ...`` / ``import mlframe.training.core._<name>`` statement appears outside of:

* the sibling cluster itself (``src/mlframe/training/core/_*.py`` files importing each other)
* ``_benchmarks/`` / ``_profile_*`` files (white-box instrumentation, treated as test-adjacent)
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "mlframe"
INTERNAL_PREFIX = "mlframe.training.core."


def _is_test_adjacent(path: Path) -> bool:
    parts = path.parts
    rel = path.relative_to(SRC).as_posix() if path.is_relative_to(SRC) else path.as_posix()
    if rel.startswith("training/core/"):
        return True
    if "_benchmarks" in parts:
        return True
    name = path.name
    if name.startswith("_profile_") or name.startswith("_bench_"):
        return True
    return False


def _iter_imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def test_no_production_underscore_imports_into_training_core_internals():
    offenders: list[tuple[str, str]] = []
    for py_path in SRC.rglob("*.py"):
        if _is_test_adjacent(py_path):
            continue
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for mod in _iter_imports(tree):
            if not mod.startswith(INTERNAL_PREFIX):
                continue
            suffix = mod[len(INTERNAL_PREFIX) :]
            head = suffix.split(".", 1)[0]
            if head.startswith("_"):
                offenders.append((py_path.relative_to(REPO_ROOT).as_posix(), mod))
    if offenders:
        formatted = "\n".join(f"  {p} -> {m}" for p, m in offenders)
        pytest.fail(
            "Production code must not import underscore-prefixed modules from mlframe.training.core "
            "outside the sibling cluster. Use the public re-exports from mlframe.training.core "
            "instead. Offenders:\n" + formatted
        )
