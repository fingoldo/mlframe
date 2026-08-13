"""Meta-test: a test module must not call ``logging.disable(...)`` at import time without a
matching, properly-scoped restore.

``logging.disable(level)`` is a process-wide, ``Logger.manager``-level override -- unlike a
per-logger ``setLevel()`` or an attached handler's own level, it short-circuits
``Logger.isEnabledFor()`` for EVERY logger in the process before any handler ever sees the
record. Under pytest-xdist, one worker imports many test modules into a single process and runs
tests from all of them; a bare module-level ``logging.disable(logging.CRITICAL)`` with no
matching ``logging.disable(logging.NOTSET)`` fires at import time and stays in effect for the
rest of that worker's lifetime, silently swallowing every LATER test's ``logger.warning(...)`` /
``logger.debug(...)`` calls -- regardless of that later test's own handler/level setup, and
regardless of ``caplog``, which relies on the same manager-level gate.

Found live: 13 test files across the suite carried this exact bare, unrestored call
(tests/test_x_ml_correctness_meta_fixes.py and 12 siblings). Confirmed as the root cause of THREE
separate CI failures in unrelated files -- tests/training/test_schema_drift_perf.py's two
StreamHandler-based assertions and tests/training/test_xgb_dmatrix_reuse_shim.py's DEBUG-level
caplog assertion -- all failed with an EMPTY captured log stream whenever one of the polluting
modules happened to be imported into the same pytest-xdist worker first.

The fix is always a module-scoped ``autouse`` fixture: ``logging.disable(logging.CRITICAL)``
before ``yield``, ``logging.disable(logging.NOTSET)`` after -- this preserves the original intent
(suppress noisy logging FOR THIS MODULE's own tests) while guaranteeing restoration once the
module's tests finish, instead of leaking into every later module in the same worker.

Baseline-diff (not zero-tolerance), matching this directory's other bare-mutation-pattern
meta-tests (test_no_module_level_env_mutation_in_tests.py). Refresh with
``--refresh-logging-disable-baseline`` after reviewing a new finding.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import orjson
import pytest

_TESTS_DIR = Path(__file__).resolve().parent.parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_module_level_logging_disable_baseline.json"

_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _refresh_requested() -> bool:
    """True if ``--refresh-logging-disable-baseline`` was passed on the pytest command line."""
    return "--refresh-logging-disable-baseline" in sys.argv


def _is_logging_disable_call(node: ast.AST) -> bool:
    """True for ``logging.disable(...)`` or a bare ``disable(...)`` (``from logging import disable``)."""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr == "disable"
    return isinstance(f, ast.Name) and f.id == "disable"


def _import_time_nodes(tree: ast.Module):
    """Yield every node evaluated at import time (module body + if/try/with/for bodies), never
    descending into a function/class/lambda -- mirrors test_no_module_level_env_mutation_in_tests.py's
    identical traversal, since the same "runs once at import, not scoped to a call" hazard applies.
    """
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, _SCOPE_NODES):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _unrestored_disable_calls(tree: ast.Module) -> list[int]:
    """Line numbers of import-time ``logging.disable(...)`` calls in a module that has NO
    import-time ``logging.disable(logging.NOTSET)`` (or equivalent falsy-level) restore call
    anywhere at module scope. A module that disables-then-immediately-restores at import time
    (rare, but not the hazard this guards against) is not flagged.
    """
    disables: list[int] = []
    has_notset_restore = False
    for node in _import_time_nodes(tree):
        if not _is_logging_disable_call(node) or not node.args:
            continue
        arg = node.args[0]
        # logging.NOTSET == 0; also treat a literal 0 the same way.
        is_notset = (isinstance(arg, ast.Attribute) and arg.attr == "NOTSET") or (isinstance(arg, ast.Constant) and arg.value == 0)
        if is_notset:
            has_notset_restore = True
        else:
            disables.append(node.lineno)
    return [] if has_notset_restore else disables


def _build_offending_set() -> set[str]:
    """``{"relpath:lineno", ...}`` for every unrestored import-time ``logging.disable(...)`` call."""
    out: set[str] = set()
    for py in _TESTS_DIR.rglob("test_*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        rel = py.relative_to(_TESTS_DIR).as_posix()
        for lineno in _unrestored_disable_calls(tree):
            out.add(f"{rel}:{lineno}")
    return out


def test_no_new_module_level_logging_disable():
    """No test module gains a new unrestored import-time ``logging.disable(...)`` call."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"module-level-logging-disable baseline written with {len(current)} entry/entries")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    added = sorted(current - baseline)
    drained = sorted(baseline - current)

    if drained:
        sys.stderr.write(
            f"\n[test_no_new_module_level_logging_disable] {len(drained)} site(s) DRAINED:\n  "
            + "\n  ".join(drained)
            + "\n  Refresh: pytest ... --refresh-logging-disable-baseline\n"
        )

    assert not added, (
        f"{len(added)} test module(s) call logging.disable(...) at IMPORT time with no matching "
        "logging.disable(logging.NOTSET) restore. logging.disable is process-wide (Logger.manager-level, "
        "not per-logger), so under xdist this silently swallows every LATER test's logger.warning/debug "
        "calls (and caplog assertions) in the same worker, regardless of that test's own handler setup -- "
        "confirmed as the root cause of 3 separate CI failures in unrelated files. Wrap the call in a "
        "module-scoped autouse fixture instead: logging.disable(logging.CRITICAL) before yield, "
        "logging.disable(logging.NOTSET) after.\n  " + "\n  ".join(added)
    )


_DETECTOR_SAMPLE = '''
import logging

logging.disable(logging.CRITICAL)

def test_something():
    pass
'''

_DETECTOR_SAMPLE_RESTORED = '''
import logging
import pytest

@pytest.fixture(autouse=True, scope="module")
def _suppress():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)
'''

_DETECTOR_SAMPLE_SAFE_INSIDE_FUNCTION = '''
import logging

def test_something():
    logging.disable(logging.CRITICAL)
    logging.disable(logging.NOTSET)
'''


def test_detector_flags_unrestored_and_ignores_restored_and_function_scoped():
    """The scan flags a bare unrestored module-level disable, and ignores both a properly
    restored module-level pair and a disable call safely scoped inside a function body."""
    assert _unrestored_disable_calls(ast.parse(_DETECTOR_SAMPLE)) == [4]
    assert _unrestored_disable_calls(ast.parse(_DETECTOR_SAMPLE_RESTORED)) == []
    assert _unrestored_disable_calls(ast.parse(_DETECTOR_SAMPLE_SAFE_INSIDE_FUNCTION)) == []
