"""Meta-test: no FE-family module reintroduces the ``return X.copy(), ...`` no-op-copy idiom on the
short-circuit "nothing to do" path.

mrmr_audit_2026-07-22 meta-test proposal #3 (X_EFFICIENCY_ARCHITECTURE-2): 39 FE-family constructor
functions under ``feature_selection/filters/`` shared the exact same pattern -- an early-return branch
(no candidate columns / disabled family / degenerate input) that did ``return X.copy(), ...`` instead of
``return X, ...``, defeating the "memory / RAM discipline" rule (CLAUDE.md) by copying a frame that is
never mutated on that path, for every single FE call regardless of whether the family fired.

Detection: AST-walk every ``Return`` statement under ``feature_selection/filters/**`` whose value is a
tuple/expression containing a call ``<name>.copy()`` where ``<name>`` is a bare ``Name`` node matching
one of the function's own OWN parameter names (i.e. the exact caller-supplied frame, not some local
derived/filtered copy the function legitimately built and needs to hand back). This is narrower than a
general "any .copy() is suspicious" scan -- returning a *local* mutated copy is correct and common;
returning the *untouched input parameter*'s ``.copy()`` is the specific no-op waste this bug class is.
Snapshot-style (baseline-diff), matching the established ``test_no_bare_except.py`` idiom.
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
_FILTERS_DIR = MLFRAME_DIR / "feature_selection" / "filters"
_BASELINE_PATH = Path(__file__).resolve().parent / "_fe_noop_copy_baseline.json"

_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "_benchmarks", "_vendored")


def _refresh_requested() -> bool:
    """True if ``--refresh-fe-noop-copy-baseline`` was passed on the pytest command line."""
    return "--refresh-fe-noop-copy-baseline" in sys.argv


def _param_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Every positional/keyword-or-positional/keyword-only parameter name ``fn`` declares."""
    args = fn.args
    return {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}


def _is_noop_copy_of_param(node: ast.AST, param_names: set[str]) -> bool:
    """True if ``node`` is a call ``<name>.copy()`` where ``<name>`` is one of the function's own
    unmodified input parameters -- the exact no-op-copy shape this bug class recurred as."""
    if not (isinstance(node, ast.Call) and not node.args and not node.keywords):
        return False
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "copy"):
        return False
    base = func.value
    return isinstance(base, ast.Name) and base.id in param_names


def _return_contains_noop_copy(ret: ast.Return, param_names: set[str]) -> bool:
    """True if any top-level element of ``ret``'s returned value (bare or tuple) is a no-op-copy call."""
    value = ret.value
    if value is None:
        return False
    candidates = value.elts if isinstance(value, ast.Tuple) else [value]
    return any(_is_noop_copy_of_param(c, param_names) for c in candidates)


def _build_offending_set() -> set[str]:
    """``{relpath:lineno}`` for every ``return <param>.copy(), ...`` no-op-copy return under ``filters/``."""
    out: set[str] = set()
    for py in _FILTERS_DIR.rglob("*.py"):
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
            continue
        tree = parsed_ast(py)
        if tree is None:
            continue
        rel = py.relative_to(MLFRAME_DIR).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            param_names = _param_names(node)
            if not param_names:
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and _return_contains_noop_copy(sub, param_names):
                    out.add(f"{rel}:{sub.lineno}")
    return out


def test_no_new_fe_noop_copy_returns():
    """No new FE-family function returns its own untouched input parameter via ``.copy()``, beyond the
    frozen baseline."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"fe-noop-copy baseline refreshed at {_BASELINE_PATH.name} ({len(current)} site(s))")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_fe_noop_copy_returns] {len(fixed)} site(s) "
            f"DRAINED:\n  " + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh: pytest ... --refresh-fe-noop-copy-baseline\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} new ``return <param>.copy(), ...`` no-op-copy return(s) -- returning an "
            f"unmutated input frame via .copy() wastes a full-frame copy on every call regardless of "
            f"whether the family fired. Return the parameter directly:\n  "
            + "\n  ".join(new[:30]) + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )
