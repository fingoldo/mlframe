"""Meta-test: a test function must be able to FAIL.

This institutionalises the scan that produced the 2026-09-01 audit's `xcut_nondiscriminating_asserts` cluster --
27 findings from 24,528 test functions -- so the shapes it found cannot reappear unnoticed. Every one of them
was a test that passed for a reason unrelated to the property it claimed to check, and several were provably
incapable of failing at all:

* **Zero assertions.** A negative contract ("bf16 must NOT be auto-set on CPU") whose whole body is a `fit()`
  call, closing with "if we got here without crashing, the plumbing worked" -- false, because the library it
  tests accepts the wrong value with a warning rather than an error.
* **Every assertion swallowed.** A Hypothesis property test wrapping all four of its assertions in
  `except AssertionError: pass`, so across 50 generated examples none could fail the suite.
* **Every assertion behind an `if`.** Five prediction-sanity tests guarded by `hasattr` + not-None + non-empty,
  so a suite that stopped producing predictions -- the regression they are the last line of defence against --
  passed all five by satisfying none of the guards. And a late-binding regression test whose one assertion sat
  behind `if result is not None`, which is how it passed despite checking for two attributes the result type
  does not have.
* **An `if` whose body is `pass`.** The comparison named in a test's own title, written as a no-op; a
  documented expectation ("default_beta must NOT trigger at n=2000") written as a no-op directly beneath the
  comment saying it would be wrong.
* **An imperative `pytest.xfail(...)`** as the last statement, discarding the measurement just taken -- one of
  which was concealing a gap that had already CLOSED.

Detector, per test function, flagging when ANY holds:
  1. no `assert` and no `pytest.raises` / `pytest.warns` / `pytest.approx` / `assert*` helper call anywhere;
  2. an `except` handler that catches `AssertionError` (or bare/`Exception`) with a body that does not re-raise;
  3. at least one assertion and EVERY assertion nested inside an `if`;
  4. an `if` statement whose body is exactly `pass`;
  5. an imperative `pytest.xfail(...)` call.

Fixture-only helpers, `conftest.py` and parametrised-skip scaffolding are excluded by the `test_*` name gate.
Baseline-diffed -- 24k functions cannot be fixed at once, and the value is in stopping NEW ones. Refresh with
``--refresh-nondiscriminating-assert-baseline`` after reviewing a finding.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import orjson
import pytest

_TESTS_DIR = Path(__file__).resolve().parent.parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_nondiscriminating_assert_baseline.json"

_FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
# Calls that carry a real check even though they are not `assert` statements.
_ASSERTING_CALLS = frozenset({"raises", "warns", "approx", "fail", "assert_array_equal", "assert_allclose", "assert_frame_equal", "assert_series_equal"})


def _refresh_requested() -> bool:
    """True if ``--refresh-nondiscriminating-assert-baseline`` was passed on the pytest command line."""
    return "--refresh-nondiscriminating-assert-baseline" in sys.argv


def _own_nodes(func: ast.AST):
    """Walk ``func`` without descending into nested function definitions (a helper's asserts are its own)."""
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _FUNC_NODES):
            stack.extend(ast.iter_child_nodes(node))


def _has_any_check(func: ast.AST) -> bool:
    """True when the function contains an assert statement or a call that performs a check."""
    for node in _own_nodes(func):
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
            if name in _ASSERTING_CALLS or name.startswith("assert"):
                return True
    return False


def _swallows_assertion_error(func: ast.AST) -> bool:
    """True when a handler catches AssertionError (directly or via Exception/bare) and does not re-raise."""
    for node in _own_nodes(func):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            caught = {"BaseException"}
        else:
            caught = {n.id for n in ast.walk(node.type) if isinstance(n, ast.Name)}
        if not (caught & {"AssertionError", "Exception", "BaseException"}):
            continue
        if any(isinstance(inner, ast.Raise) for inner in ast.walk(node)):
            continue
        # A handler containing its own assertion is checking the failure, not swallowing it.
        if any(isinstance(inner, ast.Assert) for inner in ast.walk(node)):
            continue
        return True
    return False


def _every_assert_is_conditional(func: ast.AST) -> bool:
    """True when the function has assertions and every one of them is nested inside an `if`."""
    asserts = [n for n in _own_nodes(func) if isinstance(n, ast.Assert)]
    if not asserts:
        return False
    conditional: set = set()
    for node in _own_nodes(func):
        if isinstance(node, ast.If):
            conditional.update(id(inner) for inner in ast.walk(node) if isinstance(inner, ast.Assert))
    return all(id(a) in conditional for a in asserts)


def _has_pass_only_if(func: ast.AST) -> bool:
    """True when an `if` body is exactly `pass` -- a documented expectation written as a no-op."""
    for node in _own_nodes(func):
        if isinstance(node, ast.If) and len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            return True
    return False


def _has_imperative_xfail(func: ast.AST) -> bool:
    """True when the body calls ``pytest.xfail(...)``, which discards whatever was measured."""
    for node in _own_nodes(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "xfail":
            return True
    return False


def _reasons(func: ast.AST) -> list:
    """Every nondiscriminating shape this function exhibits, as short slugs."""
    out: list = []
    if not _has_any_check(func):
        out.append("no-assert")
    if _swallows_assertion_error(func):
        out.append("swallows-assertionerror")
    if _every_assert_is_conditional(func):
        out.append("all-asserts-conditional")
    if _has_pass_only_if(func):
        out.append("pass-body-if")
    if _has_imperative_xfail(func):
        out.append("imperative-xfail")
    return out


def _build_offending_set() -> set:
    """``{"relpath:lineno:func:reasons", ...}`` for every nondiscriminating test function under ``tests/``."""
    out: set = set()
    for py in _TESTS_DIR.rglob("test_*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        rel = py.relative_to(_TESTS_DIR).as_posix()
        for func in ast.walk(tree):
            if not isinstance(func, _FUNC_NODES) or not func.name.startswith("test_"):
                continue
            reasons = _reasons(func)
            if reasons:
                out.add(f"{rel}:{func.lineno}:{func.name}:{','.join(reasons)}")
    return out


def test_no_new_nondiscriminating_assert():
    """No test function is added that cannot fail for the reason it claims to check."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"nondiscriminating-assert baseline written with {len(current)} entry/entries")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    added = sorted(current - baseline)
    assert not added, (
        "New test function(s) that cannot fail for the reason they claim to check:\n  "
        + "\n  ".join(added)
        + "\n\n  no-assert                 the body runs code and checks nothing; 'it did not crash' is not the\n"
        "                            contract, and a library that accepts a wrong value with a warning makes\n"
        "                            that reasoning false outright.\n"
        "  swallows-assertionerror   `except AssertionError: pass` (or a broad except with no re-raise) means no\n"
        "                            assertion inside can ever fail the suite.\n"
        "  all-asserts-conditional   every assertion sits behind an `if`, so the STRONGER failure -- the object\n"
        "                            not being produced at all -- skips the check instead of failing it. Assert\n"
        "                            the precondition too.\n"
        "  pass-body-if              a documented expectation written as a no-op; if the comment says a state is\n"
        "                            wrong, assert that it does not occur.\n"
        "  imperative-xfail          `pytest.xfail(...)` discards the measurement just taken. Measure first and\n"
        "                            xfail only when the gap is confirmed still open, so a gap that CLOSES is\n"
        "                            reported rather than concealed.\n\n"
        "If a flag is a genuine false positive, refresh with --refresh-nondiscriminating-assert-baseline."
    )
