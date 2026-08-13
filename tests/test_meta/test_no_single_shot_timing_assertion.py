"""Meta-test: a test function must not assert on a single-shot wall-clock timing ratio.

A lone ``t0 = time.perf_counter(); ...; t1 = time.perf_counter()`` measurement is CI-contention
noise, not a signal: a shared 2-vCPU runner under ``pytest-xdist`` with up to ~20 concurrent shards
routinely perturbs a single measurement by 2x-3x for reasons unrelated to the code under test. This
session fixed the same class of flake four separate times (``test_bootstrap_auc_presort.py``,
``test_biz_val_polars_dynamic_window``, ``test_biz_val_training_baseline_diagnostics``,
``test_y_clip_bounds_quantile_bit_identity``) by switching each to best-of-N: ``min(_fn() for _ in
range(3))`` per side before comparing. That is now the established, sanctioned pattern -- new timing
assertions should use it from the start rather than reintroducing the flake and needing the same fix
again later.

Heuristic detector (baseline-diffed, not zero-tolerance, precisely because it IS a heuristic): flags
a function that (a) calls ``time.time`` / ``time.perf_counter`` / ``timeit.default_timer``, (b)
contains an ``assert`` whose condition is a division compared via ``<``/``<=``/``>``/``>=`` against a
constant (the "ratio/speedup floor" shape), and (c) has neither a ``min(...)`` call anywhere in its
body nor an enclosing ``for``/``while`` loop -- the two ways this repo's fixes wrap a timing call to
make it best-of-N. False positives (an unrelated division assert that happens to share a function
with an unrelated timing call) are expected and get grandfathered into the baseline on first sight;
only NEW occurrences fail the gate. Refresh with ``--refresh-single-shot-timing-baseline`` after
reviewing a new finding.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import orjson
import pytest

_TESTS_DIR = Path(__file__).resolve().parent.parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_single_shot_timing_baseline.json"

_TIMER_FUNCS = frozenset({"time", "perf_counter", "perf_counter_ns", "process_time", "default_timer"})
_COMPARE_OPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)


def _refresh_requested() -> bool:
    """True if ``--refresh-single-shot-timing-baseline`` was passed on the pytest command line."""
    return "--refresh-single-shot-timing-baseline" in sys.argv


def _is_timer_call(node: ast.AST) -> bool:
    """True for ``time.perf_counter()``, ``timeit.default_timer()``, or a bare imported form of either."""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr in _TIMER_FUNCS
    return isinstance(f, ast.Name) and f.id in _TIMER_FUNCS


def _is_ratio_floor_assert(node: ast.Assert) -> bool:
    """True if ``node.test`` is ``<division> <cmp> <constant>`` (or the reverse), a speedup-floor shape."""
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], _COMPARE_OPS):
        return False
    left, right = test.left, test.comparators[0]

    def _is_div(n: ast.AST) -> bool:
        """True for a ``a / b`` division expression."""
        return isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)

    def _is_const(n: ast.AST) -> bool:
        """True for a literal int/float constant."""
        return isinstance(n, ast.Constant) and isinstance(n.value, (int, float))

    return (_is_div(left) and _is_const(right)) or (_is_const(left) and _is_div(right))


_FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _own_body_nodes(func: ast.AST):
    """Yield every node in ``func``'s body, but never descend into a NESTED function/class def -- those
    are scanned as their own separate units so an inner helper's ``min(...)`` doesn't spuriously
    launder an unrelated single-shot measurement elsewhere in the outer function, and vice versa.
    """
    stack: list[ast.AST] = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        if isinstance(node, _FUNC_NODES) and node is not func:
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _single_shot_timing_functions(tree: ast.Module) -> list[tuple[int, str]]:
    """``[(lineno, func_name), ...]`` for every function matching the single-shot-timing-assert shape."""
    out: list[tuple[int, str]] = []
    for func in ast.walk(tree):
        if not isinstance(func, _FUNC_NODES):
            continue
        has_timer = has_ratio_assert = has_min_call = has_loop = False
        for node in _own_body_nodes(func):
            if _is_timer_call(node):
                has_timer = True
            elif isinstance(node, ast.Assert) and _is_ratio_floor_assert(node):
                has_ratio_assert = True
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "min":
                has_min_call = True
            elif isinstance(node, (ast.For, ast.While)):
                has_loop = True
        if has_timer and has_ratio_assert and not (has_min_call or has_loop):
            out.append((func.lineno, func.name))
    return out


def _build_offending_set() -> set[str]:
    """``{"relpath:lineno:func_name", ...}`` for every single-shot-timing-assert function under ``tests/``."""
    out: set[str] = set()
    for py in _TESTS_DIR.rglob("test_*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        rel = py.relative_to(_TESTS_DIR).as_posix()
        for lineno, name in _single_shot_timing_functions(tree):
            out.add(f"{rel}:{lineno}:{name}")
    return out


def test_no_new_single_shot_timing_assertion():
    """No test function gains a new single-shot (non-best-of-N) wall-clock timing-ratio assertion."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"single-shot-timing baseline written with {len(current)} entry/entries")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    added = sorted(current - baseline)
    drained = sorted(baseline - current)

    if drained:
        sys.stderr.write(
            f"\n[test_no_new_single_shot_timing_assertion] {len(drained)} site(s) DRAINED:\n  "
            + "\n  ".join(drained)
            + "\n  Refresh: pytest ... --refresh-single-shot-timing-baseline\n"
        )

    assert not added, (
        f"{len(added)} new function(s) assert on a single-shot wall-clock timing ratio. A shared CI runner "
        "under xdist contention routinely perturbs one measurement by 2x-3x for reasons unrelated to the "
        "code under test -- use best-of-N instead: t = min(_fn() for _ in range(3)) per side before "
        "comparing (this session's established, repeatedly-applied fix pattern). If this is a genuine false "
        "positive (the division/timer calls are unrelated to each other), re-run with "
        "--refresh-single-shot-timing-baseline after review.\n  " + "\n  ".join(added)
    )


_DETECTOR_SAMPLE = '''
import time

def test_single_shot_flaky():
    t0 = time.perf_counter()
    serial()
    t_serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    parallel()
    t_parallel = time.perf_counter() - t0
    assert t_serial / t_parallel >= 1.15

def test_best_of_n_via_min():
    def _serial():
        t0 = time.perf_counter()
        serial()
        return time.perf_counter() - t0
    def _parallel():
        t0 = time.perf_counter()
        parallel()
        return time.perf_counter() - t0
    t_serial = min(_serial() for _ in range(3))
    t_parallel = min(_parallel() for _ in range(3))
    assert t_serial / t_parallel >= 1.15

def test_best_of_n_via_loop():
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    best = min(times)
    other = 1.0
    assert best / other >= 1.15

def test_unrelated_ratio_assert():
    t0 = time.perf_counter()
    run()
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0
    ratio = 4 / 2
    assert ratio >= 1.0
'''


def test_detector_flags_only_the_single_shot_ratio_case():
    """Only the genuinely single-shot ratio-assert function is flagged; both best-of-N shapes and the
    non-ratio-shaped unrelated-division function are not."""
    found = _single_shot_timing_functions(ast.parse(_DETECTOR_SAMPLE))
    assert [name for _, name in found] == ["test_single_shot_flaky"], found
