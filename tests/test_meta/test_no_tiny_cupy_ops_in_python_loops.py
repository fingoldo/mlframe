"""A Python ``for`` loop that issues several small cupy calls per iteration is launch-bound and is on the reviewed baseline or batched.

The old pair-floor code looped ``for p in range(n_pairs)`` and called ``cp.bincount`` / ``cp.sum`` / ``cp.where`` / ``cp.log`` for every pair: tens of thousands of kernel launches, each
doing microseconds of work, so the wall time is the launch latency and not the arithmetic. The fix is one batched kernel over all iterations (a grid dimension instead of the Python
loop) or a vectorised expression over a stacked array.

Detector (AST): in modules that mention cupy, a ``for`` loop (not ``while``) whose iterable is not a small literal ``range`` (bound <= 16) and whose body holds at least 3 calls through
``cp.`` / ``cupy.`` / ``xp.`` (or at least 2 when one of them is a data-size-dependent routine such as bincount, unique, argsort, sort, where, searchsorted, cumsum) is reported as
``<module path>::<function>``. Loops whose bound is small by construction (``range(nd)``, degrees, streams) are skipped only when the bound is a literal; others are reviewed by hand.

Benchmark scripts under ``_benchmarks`` are skipped: their loops time launches on purpose.

The baseline is a worklist of launch-bound loops for future optimisation, not a verdict: some loops run a handful of iterations in practice. A new loop outside the baseline fails the
test, and a baseline entry whose loop is gone fails too, so the list only shrinks.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "mlframe"
BASELINE = Path(__file__).with_name("_tiny_cupy_loops_baseline.json")

_XP_NAMES = {"cp", "cupy", "xp"}
_HEAVY = {"bincount", "unique", "argsort", "sort", "where", "searchsorted", "cumsum"}
_SMALL_RANGE = 16


def _cupy_call_names(loop: ast.For) -> "list[str]":
    """Attribute names of every ``cp.<f>(...)`` / ``cupy.<f>(...)`` / ``xp.<f>(...)`` call inside the loop body (nested loops included)."""
    names: list[str] = []
    for stmt in loop.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id in _XP_NAMES:
                names.append(node.func.attr)
    return names


def _is_small_literal_loop(loop: ast.For) -> bool:
    """True when the iterable is ``range(<int literal <= 16>)`` or a literal tuple/list of at most 16 items."""
    it = loop.iter
    if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range" and it.args:
        last = it.args[0] if len(it.args) == 1 else it.args[1]
        return isinstance(last, ast.Constant) and isinstance(last.value, int) and last.value <= _SMALL_RANGE
    if isinstance(it, (ast.Tuple, ast.List)):
        return len(it.elts) <= _SMALL_RANGE
    return False


def tiny_cupy_loops(source: str) -> "list[tuple[str, int]]":
    """``[(function name, line)]`` of launch-bound cupy loops in ``source``."""
    if not any(tok in source for tok in ("cupy", "cp.", "xp.")):
        return []
    tree = ast.parse(source)
    hits: list[tuple[str, int]] = []

    def visit(node: ast.AST, func: str) -> None:
        """Walk ``node`` carrying the name of the enclosing function."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name if func == "<module>" else f"{func}.{child.name}")
                continue
            if isinstance(child, ast.For) and not _is_small_literal_loop(child):
                calls = _cupy_call_names(child)
                if len(calls) >= 3 or (len(calls) >= 2 and _HEAVY & set(calls)):
                    hits.append((func, child.lineno))
            visit(child, func)

    visit(tree, "<module>")
    return hits


def _current() -> "set[str]":
    """``{"<module path>::<function>"}`` over the package source."""
    found: set[str] = set()
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "_benchmarks" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            hits = tiny_cupy_loops(text)
        except SyntaxError:
            continue
        rel = str(path.relative_to(SRC_ROOT.parent.parent)).replace("\\", "/")
        found.update(f"{rel}::{func}" for func, _ in hits)
    return found


def _baseline() -> "set[str]":
    """The reviewed ``{"<module path>::<function>"}`` worklist."""
    return set(json.loads(BASELINE.read_text(encoding="utf-8")))


def test_no_new_launch_bound_cupy_loops():
    """A launch-bound cupy loop outside the reviewed baseline fails the test; batch it into one kernel or a vectorised expression."""
    new = sorted(_current() - _baseline())
    assert not new, f"Python loops of small cupy calls outside the baseline (batch them into one kernel / vectorised op, or review and baseline): {new}"


def test_baseline_has_no_stale_entries():
    """A baseline entry whose loop is gone or batched is removed, so the baseline only shrinks."""
    stale = sorted(_baseline() - _current())
    assert not stale, f"baseline entries no longer needed (remove them): {stale}"


@pytest.mark.parametrize("rel", sorted({e.split("::")[0] for e in _baseline()}))
def test_baselined_module_exists(rel):
    """Every baseline path points at a real module, so a rename cannot hide a loop."""
    assert (SRC_ROOT.parent.parent / rel).is_file(), rel


def test_checker_flags_per_pair_loop_and_accepts_batched_form():
    """The old per-pair bincount/sum/where/log loop is flagged; a vectorised expression and a tiny literal loop are accepted."""
    bad = (
        "import cupy as cp\n"
        "def pair_floor(codes, n_pairs):\n"
        "    for p in range(n_pairs):\n"
        "        h = cp.bincount(codes[p])\n"
        "        s = cp.sum(h)\n"
        "        out[p] = cp.log(s)\n"
    )
    good = (
        "import cupy as cp\n"
        "def pair_floor(codes):\n"
        "    h = cp.bincount(codes.ravel())\n"
        "    return cp.log(cp.sum(h))\n"
        "def small(a):\n"
        "    for d in range(4):\n"
        "        a = cp.sum(cp.where(a, 1, 0))\n"
    )
    assert tiny_cupy_loops(bad) == [("pair_floor", 3)]
    assert not tiny_cupy_loops(good)
