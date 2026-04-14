"""Meta-tests enforcing repo conventions.

- No stdlib ``json`` imports in test files (MEMORY.md: always orjson).
- ``mlframe.postcalibration`` exposes a precompiled ``_INCLUDE_RE`` sentinel.
- No ``ensure_installed(...)`` calls in test files (use ``pytest.importorskip``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

_JSON_IMPORT_RE = re.compile(r"^\s*(?:import\s+json\b|from\s+json\s+import\b)", re.MULTILINE)
_ENSURE_INSTALLED_RE = re.compile(r"\bensure_installed\s*\(")


def _iter_test_files() -> list[Path]:
    return [p for p in TESTS_DIR.rglob("*.py") if p.name != "test_conventions.py"]


def test_no_stdlib_json_in_tests() -> None:
    offenders: list[str] = []
    for path in _iter_test_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if _JSON_IMPORT_RE.search(text):
            offenders.append(str(path.relative_to(TESTS_DIR)))
    assert not offenders, (
        "Test files must use orjson instead of stdlib json; offenders: " + ", ".join(offenders)
    )


def test_no_ensure_installed_in_tests() -> None:
    offenders: list[str] = []
    for path in _iter_test_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if _ENSURE_INSTALLED_RE.search(text):
            offenders.append(str(path.relative_to(TESTS_DIR)))
    assert not offenders, (
        "Test files must use pytest.importorskip(...) instead of ensure_installed(...); "
        "offenders: " + ", ".join(offenders)
    )


def test_postcalibration_include_re_is_compiled() -> None:
    pytest.importorskip("sklearn")
    postcalibration = pytest.importorskip("mlframe.postcalibration")
    include_re = getattr(postcalibration, "_INCLUDE_RE", None)
    assert isinstance(include_re, re.Pattern), (
        "mlframe.postcalibration._INCLUDE_RE must be a module-level compiled re.Pattern"
    )
    # Also validate the lru_cache-wrapped compiler is present and returns a Pattern.
    compile_pattern = getattr(postcalibration, "_compile_pattern", None)
    assert callable(compile_pattern), "_compile_pattern helper must exist"
    assert isinstance(compile_pattern("foo"), re.Pattern)
