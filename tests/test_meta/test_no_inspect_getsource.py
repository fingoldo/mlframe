"""Meta-linter: no test file may use inspect.getsource for assertions.

Memory rule: feedback_behavioral_tests forbids using inspect.getsource()
to assert on prod source-text. AST-walk instead asserts BEHAVIOUR.

This linter scans every tests/**/*.py file and counts violations.
The whitelist allows tests/test_rng_determinism.py to use the related
inspect.getsourcefile pattern (now migrated to mod.__file__).

A second linter (``test_no_source_text_position_proxy_in_test_files``) closes
the ``read_text().find(...)`` escape hatch: reading a prod ``src/mlframe`` module
via ``read_text`` and then asserting on the BYTE POSITION of substrings (``.find``
/ ``.index`` / ``.rfind``) is the same source-text-proxy antipattern as
``inspect.getsource`` -- it just dodges AST detection of the ``getsource`` call.
The complementary ``"literal" in read_text_var`` membership shape is covered by
``test_no_source_text_proxy`` / ``_source_proxy_scan``. Pure LOC-budget facade
sensors (``read_text().splitlines()`` length checks from monolith-split tests)
are NOT source-text proxies and are not flagged.
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
    # F-37/F-38/F-39 predict-path order tests assert STATIC properties of
    # predict_step's body (order of helper calls, which context manager is
    # used, whether the no-branch form is in source). A behavioural rewrite
    # would have to fake CUDA / cuDNN to exercise the same paths, multiplying
    # surface area for no gain -- the static check IS the gate the author
    # wanted. User-reaffirmed 2026-05-31 by reverting the behavioural rewrites.
    "training/neural/test_compile_predict.py",
    "training/neural/test_cuda_graph_predict.py",
    "training/neural/test_torch_compile_safety_and_profiler.py",
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


# Files that still inspect prod source byte-positions and are NOT owned by this change.
# Documented known-debt: each asserts on the location of a substring inside a prod module
# read via ``read_text`` rather than on runtime behaviour, and should migrate to a
# behavioural sensor. Listed so the gate stays live for NEW code while tracking the
# remaining sites explicitly (never silently passed). Use forward slashes.
SOURCE_POSITION_PROXY_WHITELIST: set[str] = {
    "training/test_dataset_cache_fingerprint.py",
    "training/test_mlp_ttr_regression_no_collapse.py",
    "training/test_t_scale_composite_report_skip.py",
}

_PROD_SOURCE_PATH_MARKERS = ("mlframe", "__file__")
_POSITION_METHODS = {"find", "index", "rfind"}


def _prod_source_read_text_names(tree: ast.Module) -> set[str]:
    """Names bound from ``<prod-path>.read_text()`` -- a path expression that resolves into
    an mlframe prod module (rooted in ``mlframe`` / a non-test ``__file__``)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            continue
        value = node.value
        if value is None:
            continue
        if not any(
            isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute) and s.func.attr == "read_text"
            for s in ast.walk(value)
        ):
            continue
        seg = ast.unparse(value)
        if "mlframe" not in seg and not ("__file__" in seg and "test" not in seg.lower()):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in targets:
            if isinstance(t, ast.Name):
                names.add(t.id)
    return names


def _find_source_position_proxies(path: Path) -> list[tuple[int, str]]:
    """Return [(lineno, method)] for ``<prod-read_text-var>.find/index/rfind(...)`` calls."""
    try:
        src = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []
    names = _prod_source_read_text_names(tree)
    if not names:
        return []
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in _POSITION_METHODS:
            continue
        base = node.func.value
        while isinstance(base, (ast.Attribute, ast.Subscript)):
            base = base.value
        if isinstance(base, ast.Name) and base.id in names:
            hits.append((node.lineno, node.func.attr))
    return hits


def test_no_source_text_position_proxy_in_test_files() -> None:
    """No test file may assert on the byte-position of substrings inside a prod
    ``src/mlframe`` module read via ``read_text`` (``.find`` / ``.index`` / ``.rfind``).

    This is the source-text-proxy antipattern that ``inspect.getsource`` is banned for;
    reading the file directly to ``.find`` a string and assert ordering carries the same
    fragility (breaks on any rename / refactor, proves nothing about runtime values) while
    dodging the ``getsource`` AST check. Migrate to a behavioural sensor that calls the prod
    function and asserts on the result.
    """
    offenders: list[str] = []
    for path in _iter_test_files():
        rel = path.relative_to(_REPO_TESTS).as_posix()
        if rel in SOURCE_POSITION_PROXY_WHITELIST:
            continue
        for ln, method in _find_source_position_proxies(path):
            offenders.append(f"{rel}:{ln} (read_text(...).{method})")
    assert not offenders, (
        "Source-text position assertions on prod mlframe source violate "
        "feedback_behavioral_tests rule. Call the prod function and assert on the "
        f"RESULT instead. Offenders:\n  " + "\n  ".join(offenders[:50])
    )
