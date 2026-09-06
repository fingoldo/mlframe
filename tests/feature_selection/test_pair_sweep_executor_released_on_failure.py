"""The pair-sweep executor must be released on the failure path, not only at the end of a clean sweep.

`_chunk_state["pipeline_ex"] = ThreadPoolExecutor(max_workers=1)` was created about two hundred lines
before its `shutdown(wait=True)`, with nothing covering the span. Any exception in the pair loop -- a cupy
OutOfMemoryError, a kernel launch failure, a KeyboardInterrupt -- skipped the shutdown.
`ThreadPoolExecutor` registers its worker through `threading._register_atexit`, so the thread survived to
interpreter exit, and a still-pending future held the shared double buffer with it: at a 2M-row chunk over
40 operands that is 640 MB per buffer, 1.28 GB for the pair, retained for the rest of the process.

Driving the real failure needs the GPU-strict chunk pipeline, which needs cupy and a device, so this checks
the containment structurally instead -- but on the parse tree rather than on the source text, and asking
the question that matters: is the shutdown inside a `finally` that ENCLOSES the construction. The shared
`py_ci_shared.resource_release_paths` check does not cover this: it accepts any `try/finally` anywhere in
the module that mentions the release, and this module already had `_ex0.shutdown(wait=False)` inside an
`except`, which satisfied it. Verified against the pre-fix revision: it reported nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "filters" / "_feature_engineering_pairs" / "_pairs_core.py"


def _tree() -> ast.Module:
    """Parse the module under test."""
    return ast.parse(MODULE.read_bytes().decode("utf-8"))


def _enclosing_function(tree: ast.Module, lineno: int) -> ast.FunctionDef:
    """The innermost function containing `lineno`."""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno <= lineno <= node.end_lineno:
            if best is None or node.lineno > best.lineno:
                best = node
    assert best is not None, f"no function encloses line {lineno}"
    return best


def _construction_line(tree: ast.Module) -> int:
    """Where the pipeline executor is built."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ThreadPoolExecutor":
            return node.lineno
    pytest.fail("no ThreadPoolExecutor construction found; this test has lost its subject")


def _blocking_shutdowns(tree: ast.Module) -> list[ast.Call]:
    """Every `.shutdown(wait=True)` call -- the release that must be guaranteed."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "shutdown":
            for kw in node.keywords:
                if kw.arg == "wait" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    out.append(node)
    return out


def test_the_blocking_shutdown_sits_in_a_finally_that_covers_the_construction():
    """Presence of a try/finally somewhere is not enough; it has to enclose the site that creates the thread."""
    tree = _tree()
    built_at = _construction_line(tree)
    shutdowns = _blocking_shutdowns(tree)
    assert shutdowns, "no blocking shutdown found; the executor would outlive the sweep"

    covered = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try) or not node.finalbody:
            continue
        finally_span = range(node.finalbody[0].lineno, node.finalbody[-1].end_lineno + 1)
        try_span = range(node.body[0].lineno, node.body[-1].end_lineno + 1)
        if built_at in try_span or built_at < try_span.start:
            if any(call.lineno in finally_span for call in shutdowns):
                covered = True
                break
    assert covered, (
        f"the executor is built at line {built_at} but no `finally` covering that point calls "
        f"shutdown(wait=True) (found at lines {[c.lineno for c in shutdowns]}); an exception in the pair "
        "loop leaks the worker thread and the double chunk buffer it holds"
    )


def test_the_shared_buffers_are_dropped_alongside_the_executor():
    """The buffers are what the leaked thread holds, so the same `finally` has to release the reference."""
    tree = _tree()
    finally_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and node.finalbody:
            finally_lines.update(range(node.finalbody[0].lineno, node.finalbody[-1].end_lineno + 1))

    dropped = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "pop"
            and node.lineno in finally_lines
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            dropped.add(node.args[0].value)
    assert "pipeline_buffers" in dropped, "the double chunk buffer is not released with the executor"


def test_the_construction_is_still_guarded_by_its_own_setup_except():
    """The setup's own `except` -- which shuts the half-built executor down -- must survive the rewrite."""
    tree = _tree()
    non_blocking = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "shutdown"
        and any(kw.arg == "wait" and isinstance(kw.value, ast.Constant) and kw.value.value is False for kw in node.keywords)
    ]
    assert non_blocking, "the setup-failure path no longer shuts down a partially built executor"
