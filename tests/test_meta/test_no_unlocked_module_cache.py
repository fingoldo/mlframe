"""Meta-test: no new module-level mutable cache dict under ``src/mlframe/**`` is defined in a file that
never takes a ``threading.Lock``/``RLock`` anywhere -- the exact shape of a recurring concurrency bug.

mrmr_audit_2026-07-22 meta-test proposal #1: this session alone fixed the identical bug class 3+
times independently (USABILITY_A-11/12 ``_fit_constant_memmap`` cache in ``_joblib_safe.py``,
X_EDGE_CASES_BEST_PRACTICES-1 ``resident_operand`` cache in ``_fe_resident_operands.py``,
X_EDGE_CASES_BEST_PRACTICES-4 the GPU-strict audit monkeypatch state in ``_gpu_strict_fe/_audit.py``) --
a module-level ``dict``/``OrderedDict`` cache mutated by a function that can run under joblib
``backend="threading"`` (a genuinely-supported concurrent-``.fit()`` pattern in this codebase) with no
lock at all, or a lock covering only part of the get-or-compute-or-evict sequence.

Detection (necessarily a coarse proxy, not a full race-detector): flag any module that (a) defines a
module-level dict/OrderedDict-typed name ending in ``_CACHE`` (the codebase's own consistent naming
convention for these), AND (b) never constructs a ``threading.Lock()``/``threading.RLock()`` anywhere in
the same file. A module with a real lock is exempt even if the lock's coverage is imperfect (that's a
finer-grained bug a human still has to judge) -- this only catches the "zero locking machinery exists at
all" case, which is exactly the shape every one of the 3 recurrences above started as. Snapshot-style
(baseline-diff), matching the established ``test_no_bare_except.py`` idiom.
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
_BASELINE_PATH = Path(__file__).resolve().parent / "_unlocked_module_cache_baseline.json"

_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "tests", "legacy", "profiling", "explore", "_benchmarks")


def _refresh_requested() -> bool:
    """True if ``--refresh-unlocked-module-cache-baseline`` was passed on the pytest command line."""
    return "--refresh-unlocked-module-cache-baseline" in sys.argv


def _is_dict_like_cache_value(value: ast.AST) -> bool:
    """True if ``value`` looks like an empty/fresh dict-like cache container: ``{}``, ``dict()``,
    ``OrderedDict()``, or a call to a ``dict``/``OrderedDict``-named constructor."""
    if isinstance(value, ast.Dict):
        return True
    if isinstance(value, ast.Call):
        func = value.func
        name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
        return name in ("dict", "OrderedDict", "defaultdict")
    return False


def _module_has_lock_construction(tree: ast.Module) -> bool:
    """True if the module constructs a ``threading.Lock()``/``threading.RLock()`` (bare or attribute
    form) anywhere -- the coarse "some locking machinery exists" signal."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
        if name in ("Lock", "RLock"):
            return True
    return False


def _module_level_cache_names(tree: ast.Module) -> list[tuple[str, int]]:
    """``[(name, lineno), ...]`` for every module-level (top of ``Module.body``, not nested in a
    function/class) assignment to a ``*_CACHE`` name holding a dict-like value."""
    out: list[tuple[str, int]] = []
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        if value is None or not _is_dict_like_cache_value(value):
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id.endswith("_CACHE"):
                out.append((t.id, node.lineno))
    return out


def _build_offending_set() -> set[str]:
    """``{relpath:lineno}`` for every module-level ``*_CACHE`` dict defined in a file with no
    ``Lock()``/``RLock()`` construction anywhere."""
    out: set[str] = set()
    for py in MLFRAME_DIR.rglob("*.py"):
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
            continue
        if py.name.endswith(".py.old"):
            continue
        tree = parsed_ast(py)
        if tree is None:
            continue
        caches = _module_level_cache_names(tree)
        if not caches:
            continue
        if _module_has_lock_construction(tree):
            continue
        rel = py.relative_to(MLFRAME_DIR).as_posix()
        for _name, lineno in caches:
            out.add(f"{rel}:{lineno}")
    return out


def test_no_new_unlocked_module_level_cache():
    """No new module-level ``*_CACHE`` dict in a file with zero ``threading.Lock``/``RLock``
    construction, beyond the frozen baseline.

    A dict-cache mutated inside a function reachable from joblib ``backend='threading'`` concurrent
    ``.fit()`` calls with NO lock anywhere in the module is a real race (lost updates, or -- worse on
    Windows -- a double-unlink/double-close on a cached filesystem/device resource). This is a coarse
    proxy (it cannot prove the cache is actually reachable concurrently, nor that a lock elsewhere in
    the call chain doesn't already cover it) -- new hits should be reviewed, not blindly "fixed" by
    slapping a lock on a cache that's provably single-threaded.
    """
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"unlocked-module-cache baseline refreshed at {_BASELINE_PATH.name} ({len(current)} site(s))")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_unlocked_module_level_cache] {len(fixed)} site(s) "
            f"DRAINED:\n  " + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh: pytest ... --refresh-unlocked-module-cache-baseline\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} new module-level *_CACHE dict(s) in a file with no threading.Lock/RLock "
            f"anywhere -- review for a real concurrent-.fit() race (add a lock covering the whole "
            f"get-or-compute-or-evict sequence) or confirm single-threaded-only and note why:\n  "
            + "\n  ".join(new[:30]) + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )
