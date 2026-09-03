"""Meta-test: no ``isinstance(x, Tick)`` fixed-duration-offset check anywhere in the repo.

``compute_ml_perf_by_time`` (``training/evaluation.py``) used to gate its day-divisor fast path on
``isinstance(_off, Tick)`` -- pandas<3 classified every fixed-duration offset (``Day``, ``Hour``,
...) under ``pandas.tseries.offsets.Tick``. pandas 3.0 (PDEP-14-adjacent offset refactor) dropped
``Day`` from ``Tick``'s subclass hierarchy while ``Day().nanos`` still works identically -- the
isinstance check silently stopped matching the extremely common ``"D"`` freq on pandas>=3, falling
through to a slower/different code path with no error, no warning, just a quiet behavior change.
``.nanos`` (which raises ``ValueError`` for genuinely non-fixed offsets like ``Week``/``MonthEnd``
in both pandas 2.x and 3.x) is the version-stable fixed-duration test; see the fix and its
docstring in ``training/evaluation.py::compute_ml_perf_by_time``.

Baseline-diff (not zero-tolerance) so a legitimate future ``isinstance(..., Tick)`` use -- e.g. an
error-message branch that doesn't drive a behavioral fork -- can be explicitly grandfathered rather
than blocking unrelated work; run with ``--refresh-tick-isinstance-baseline`` to accept new findings
after review.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import orjson
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCAN_DIRS = (_REPO_ROOT / "src" / "mlframe", _REPO_ROOT / "tests")
_BASELINE_PATH = Path(__file__).resolve().parent / "_tick_isinstance_baseline.json"


def _refresh_requested() -> bool:
    """True if ``--refresh-tick-isinstance-baseline`` was passed on the pytest command line."""
    return "--refresh-tick-isinstance-baseline" in sys.argv


def _names_a_tick_class(node: ast.AST) -> bool:
    """True for a bare ``Tick`` name or a ``....offsets.Tick`` / ``....Tick`` attribute access."""
    if isinstance(node, ast.Name):
        return node.id == "Tick"
    return isinstance(node, ast.Attribute) and node.attr == "Tick"


def _isinstance_tick_calls(tree: ast.Module) -> list[tuple[int, str]]:
    """``[(lineno, "isinstance(..., Tick)" or "isinstance(..., (..., Tick, ...))"), ...]``."""
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "isinstance"):
            continue
        if len(node.args) != 2:
            continue
        cls_arg = node.args[1]
        if _names_a_tick_class(cls_arg):
            out.append((node.lineno, "isinstance(..., Tick)"))
        elif isinstance(cls_arg, ast.Tuple) and any(_names_a_tick_class(elt) for elt in cls_arg.elts):
            out.append((node.lineno, "isinstance(..., (..., Tick, ...))"))
    return out


def _build_offending_set() -> set[str]:
    """``{"relpath:lineno:what", ...}`` for every ``isinstance(..., Tick)``-shaped call under the scanned dirs."""
    out: set[str] = set()
    for scan_dir in _SCAN_DIRS:
        for py in scan_dir.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
            except (SyntaxError, OSError):
                continue
            rel = py.relative_to(_REPO_ROOT).as_posix()
            for lineno, what in _isinstance_tick_calls(tree):
                out.add(f"{rel}:{lineno}:{what}")
    return out


def test_no_new_tick_isinstance_check():
    """No file gains a new ``isinstance(x, Tick)`` fixed-duration-offset check beyond the baseline."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"Tick-isinstance baseline written with {len(current)} entry/entries")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    added = sorted(current - baseline)
    drained = sorted(baseline - current)

    if drained:
        sys.stderr.write(
            f"\n[test_no_new_tick_isinstance_check] {len(drained)} site(s) DRAINED:\n  "
            + "\n  ".join(drained)
            + "\n  Refresh: pytest ... --refresh-tick-isinstance-baseline\n"
        )

    assert not added, (
        f"{len(added)} new isinstance(x, Tick) fixed-duration-offset check(s). pandas 3.0 dropped Day from "
        "Tick's subclass hierarchy while Day().nanos still works identically -- isinstance(x, Tick) silently "
        "stops matching the common 'D' freq there with no error. Use a try/except ValueError around x.nanos "
        "instead (see training/evaluation.py::compute_ml_perf_by_time for the pattern). If this really is a "
        "safe use (e.g. an error message, not a behavioral fork), re-run with "
        "--refresh-tick-isinstance-baseline after review.\n  " + "\n  ".join(added)
    )


_DETECTOR_SAMPLE = '''
from pandas.tseries.offsets import Tick
import pandas.tseries.offsets as offsets

def a(off):
    if isinstance(off, Tick):
        return True

def b(off):
    if isinstance(off, offsets.Tick):
        return True

def c(off):
    if isinstance(off, (offsets.Week, Tick, offsets.MonthEnd)):
        return True

def d(off):
    try:
        return off.nanos
    except ValueError:
        return None

def e(off):
    return isinstance(off, offsets.Week)
'''


def test_detector_sees_tick_isinstance_and_ignores_nanos_and_other_offsets():
    """The scan flags the three Tick-isinstance shapes and neither the .nanos check nor an unrelated isinstance."""
    found = _isinstance_tick_calls(ast.parse(_DETECTOR_SAMPLE))
    assert sorted(ln for ln, _ in found) == [6, 10, 14], found
