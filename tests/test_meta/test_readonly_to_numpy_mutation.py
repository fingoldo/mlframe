"""Meta-test: flag ``np.fill_diagonal(X, ...)``-style in-place mutation of an array that came
straight from ``<pandas obj>.to_numpy()`` with no ``copy=True``.

Root cause (2026-07-31, real bug): ``pandas.DataFrame.to_numpy()`` can return a read-only view
of the frame's backing storage under Copy-on-Write (default from pandas 3.0, opt-in in 2.x) --
``_shap_proxy_preflight.py``'s ``dataset_diagnostics`` fed exactly such a view straight into
``np.fill_diagonal(C, 0.0)`` and crashed with ``ValueError: underlying array is read-only`` on a
CoW-enabled pandas install, while passing silently on this dev machine's non-CoW pandas -- an
environment-dependent failure invisible without the exact same pandas config.

This walks each function body and flags any local name assigned from a ``.to_numpy()`` call with
no ``copy=True`` keyword that is later passed as the first positional argument to
``np.fill_diagonal`` (or any other in-place-mutating numpy call using the same first-arg-is-target
convention: ``np.fill_diagonal``/``np.copyto``) within the SAME function -- the array must be
copied first (``.to_numpy(copy=True)`` or ``np.array(..., copy=True)``).

Snapshot-style baseline like the sibling meta-tests in this directory: new violations fail unless
added to the baseline.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import orjson
import pytest

import mlframe

from tests.test_meta._shared_ast_cache import parsed_ast

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_readonly_to_numpy_mutation_baseline.json"

_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "tests", "legacy", "profiling", "explore", "_benchmarks")

_INPLACE_MUTATORS = {"fill_diagonal", "copyto"}


def _refresh_requested() -> bool:
    """True if ``--refresh-readonly-to-numpy-mutation-baseline`` was passed on the pytest command line."""
    return "--refresh-readonly-to-numpy-mutation-baseline" in sys.argv


def _is_to_numpy_call_without_copy(call: ast.Call) -> bool:
    """True for ``<anything>.to_numpy(...)`` with no ``copy=True`` keyword argument."""
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr == "to_numpy"):
        return False
    for kw in call.keywords:
        if kw.arg == "copy":
            return not (isinstance(kw.value, ast.Constant) and kw.value.value is True)
    return True


def _is_np_inplace_mutator_call(call: ast.Call) -> str | None:
    """Return the mutated arg's Name id if ``call`` is ``np.fill_diagonal(X, ...)`` / ``np.copyto(X, ...)``, else None."""
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr in _INPLACE_MUTATORS):
        return None
    if not (isinstance(func.value, ast.Name) and func.value.id in ("np", "numpy")):
        return None
    if not call.args:
        return None
    first = call.args[0]
    return first.id if isinstance(first, ast.Name) else None


def _build_offending_set() -> set[str]:
    """``{relpath:lineno}`` for every in-place mutator call whose target traces to an uncopied ``.to_numpy()``."""
    out: set[str] = set()
    for py in MLFRAME_DIR.rglob("*.py"):
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
            continue
        tree = parsed_ast(py)
        if tree is None:
            continue
        rel = py.relative_to(MLFRAME_DIR).as_posix()
        for func_node in ast.walk(tree):
            if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            risky_names: set[str] = set()
            for node in ast.walk(func_node):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and _is_to_numpy_call_without_copy(node.value):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            risky_names.add(target.id)
            for node in ast.walk(func_node):
                if not isinstance(node, ast.Call):
                    continue
                target_name = _is_np_inplace_mutator_call(node)
                if target_name is not None and target_name in risky_names:
                    out.add(f"{rel}:{node.lineno}")
    return out


def test_no_new_readonly_to_numpy_mutation():
    """No new in-place mutation of an uncopied ``.to_numpy()`` result beyond the frozen baseline."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"readonly-to_numpy-mutation baseline refreshed at {_BASELINE_PATH.name} ({len(current)} site(s))")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_readonly_to_numpy_mutation] "
            f"{len(fixed)} site(s) DRAINED:\n  "
            + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh baseline to lock in.\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} new in-place mutation(s) of an uncopied ``.to_numpy()`` result -- under "
            f"pandas Copy-on-Write this raises ``ValueError: underlying array is read-only`` at "
            f"runtime. Use ``.to_numpy(copy=True)``, OR refresh the baseline if this is a confirmed "
            f"false positive:\n  " + "\n  ".join(new[:30]) + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )
