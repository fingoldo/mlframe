"""Meta-test: a ``*_CACHE`` in a module that HAS a lock must not be mutated by a function that never takes it.

``test_no_unlocked_module_cache`` deliberately exempts any module that constructs a ``Lock``/``RLock``
anywhere, and says so: "a module with a real lock is exempt even if the lock's coverage is imperfect (that's
a finer-grained bug a human still has to judge)". That exemption is exactly where the bug class survives.
Once a module grows a lock, the next cache added to it looks protected and reviews as protected, while the
new mutation site takes no lock at all - the "locked elsewhere, unlocked here" shape. It is strictly harder
to spot by eye than the zero-locking case, because the file greps as lock-aware.

Detection is a per-FUNCTION version of the same coarse proxy: for each module-level dict-like ``*_CACHE`` in
a module that constructs a lock, flag every function that MUTATES that cache (``C[k] = v``, ``del C[k]``,
``C.pop/clear/setdefault/update/popitem``) without a ``with`` statement anywhere in its body.

Deliberately coarse in the safe direction, so it stays a reviewable signal rather than noise:
  * any ``with`` block counts, not specifically the right lock - proving the *right* lock is held over the
    whole get-or-compute-or-evict sequence needs judgement, and over-claiming would make this unfixable;
  * a helper that mutates under a lock held by its CALLER is a false positive - annotate and baseline it.

Scope: every module-level dict that some function mutates, whatever its name, and mutations through a local alias
(``c = _REGISTRY; c[k] = v``). It used to see only ``*_CACHE`` names and direct mutations, so ``_DEFERRED_HOST_FILL``
in ``_gpu_resident_fe.py`` (mutated from several threads through ``c = ...``, no lock) was never examined.

Baseline-diff, matching the sibling test's idiom. Refresh with::

    pytest tests/test_meta/test_module_cache_mutated_without_its_lock.py --refresh-cache-mutation-lock-baseline
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import orjson
import pytest

import mlframe

from tests.test_meta._module_mutable_state import build_dict_index, build_reexport_index, function_defs, imported_module_dicts, mutable_module_dicts, mutations_of
from tests.test_meta._shared_ast_cache import parsed_ast

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_cache_mutation_lock_baseline.json"

_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "tests", "legacy", "profiling", "explore", "_benchmarks")

_MUTATING_METHODS = frozenset({"pop", "clear", "setdefault", "update", "popitem"})


def _refresh_requested() -> bool:
    """True if ``--refresh-cache-mutation-lock-baseline`` was passed on the pytest command line."""
    return "--refresh-cache-mutation-lock-baseline" in sys.argv


def _is_dict_like_cache_value(value: ast.AST) -> bool:
    """True if ``value`` is a fresh dict-like container literal or constructor call."""
    if isinstance(value, ast.Dict):
        return True
    if isinstance(value, ast.Call):
        func = value.func
        name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
        return name in ("dict", "OrderedDict", "defaultdict")
    return False


def _module_has_lock_construction(tree: ast.Module) -> bool:
    """True if the module constructs a ``Lock()``/``RLock()`` anywhere.

    This test is the complement of the sibling one, so it looks at exactly the modules the sibling skips.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
            if name in ("Lock", "RLock"):
                return True
    return False


def _module_level_cache_names(tree: ast.Module) -> set[str]:
    """Names of module-level dicts that some function mutates: shared mutable state, whatever it is called."""
    return set(mutable_module_dicts(tree))


def _mutations_of(func: ast.AST, cache_names: set[str]) -> set[str]:
    """Alias-aware mutation finder; see ``_module_mutable_state.mutations_of``."""
    return mutations_of(func, cache_names)


def _has_with_block(func: ast.AST) -> bool:
    """True if the function body contains any ``with``/``async with`` - the coarse "takes a lock" proxy."""
    for node in ast.walk(func):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            return True
    return False


def _offending_in_tree(tree: ast.Module) -> set[tuple[int, str, str]]:
    """``{(lineno, func_name, cache_name), ...}`` for one parsed module; empty unless it constructs a lock."""
    caches = _module_level_cache_names(tree)
    if not caches or not _module_has_lock_construction(tree):
        return set()
    out: set[tuple[int, str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or _has_with_block(node):
            continue
        for name in sorted(_mutations_of(node, caches)):
            out.add((node.lineno, node.name, name))
    return out


def _build_offending_set() -> set[str]:
    """``{"relpath:lineno:func:CACHE", ...}`` for each unlocked mutation site in a lock-aware module."""
    out: set[str] = set()
    for py in MLFRAME_DIR.rglob("*.py"):
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS) or py.name.endswith(".py.old"):
            continue
        tree = parsed_ast(py)
        if tree is None:
            continue
        rel = py.relative_to(MLFRAME_DIR).as_posix()
        for lineno, func, cache in _offending_in_tree(tree):
            out.add(f"{rel}:{lineno}:{func}:{cache}")
    return out


_BLIND_CHECK_LOCKED = """
import threading
_L = threading.Lock()
_X_CACHE = {}

def reader(k):
    return _X_CACHE.get(k)

def guarded(k, v):
    with _L:
        _X_CACHE[k] = v

def unguarded(k, v):
    _X_CACHE[k] = v

def unguarded_pop(k):
    _X_CACHE.pop(k, None)

_PENDING = {}

def aliased(k, v):
    c = _PENDING
    c[k] = v
"""

_BLIND_CHECK_NO_LOCK = """
_X_CACHE = {}

def unguarded(k, v):
    _X_CACHE[k] = v
"""


def test_detector_is_not_blind():
    """The predicate flags exactly the unguarded mutators - not the reader, not the ``with``-guarded one.

    Without this a refactor that quietly stopped matching anything would still "pass" forever, since a
    baseline-diff test is green precisely when it finds nothing new.
    """
    found = {(f, c) for _ln, f, c in _offending_in_tree(ast.parse(_BLIND_CHECK_LOCKED))}
    # ``aliased`` is the shape that hid _DEFERRED_HOST_FILL: not a *_CACHE name, and mutated through a local alias.
    assert found == {("unguarded", "_X_CACHE"), ("unguarded_pop", "_X_CACHE"), ("aliased", "_PENDING")}, found

    # A module with no lock at all belongs to the SIBLING test; this one must stay silent on it, so the
    # two baselines never double-report the same site.
    assert _offending_in_tree(ast.parse(_BLIND_CHECK_NO_LOCK)) == set()


def test_no_new_cache_mutation_outside_a_lock():
    """No new function mutates a module-level ``*_CACHE`` without any ``with`` block, in a lock-aware module."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"cache-mutation-lock baseline written with {len(current)} site(s)")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_cache_mutation_outside_a_lock] {len(fixed)} site(s) DRAINED:\n  "
            + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh: pytest ... --refresh-cache-mutation-lock-baseline\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} function(s) mutate a module-level dict with no lock held, in a module that "
            "DOES construct one - the 'locked elsewhere, unlocked here' race. The module greps as "
            "lock-aware, so this will not be caught by reading it. Either take the existing lock across "
            "the whole get-or-compute-or-evict sequence, or confirm the caller already holds it and "
            "baseline the site with a note saying which caller:\n  " + "\n  ".join(new[:30]) + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )


_XMOD_BASELINE_PATH = Path(__file__).resolve().parent / "_cross_module_dict_mutation_baseline.json"


def _cross_module_offending(modules: dict) -> set[str]:
    """``{"relpath:lineno:func:origin.module:NAME"}`` for functions mutating ANOTHER module's dict with no ``with`` block.

    A dict defined in one module and written from another is shared state whatever either module's own locking looks
    like; the per-module scan above never saw it (``_CB_VAL_POOL_CACHE`` lives in ``_predict_guards`` and was evicted
    and inserted from ``cb/_cb_pool`` with no lock). Whether the defining module has a lock is irrelevant here.
    """
    index = build_dict_index(modules)
    reexports = build_reexport_index(modules, index)
    out: set[str] = set()
    for rel, tree in modules.items():
        imported = imported_module_dicts(tree, rel, index, reexports)
        if not imported:
            continue
        for fn in function_defs(tree):
            if _has_with_block(fn):
                continue
            for local in sorted(mutations_of(fn, set(imported))):
                out.add(f"{rel}:{fn.lineno}:{fn.name}:{imported[local]}")
    return out


def _all_modules() -> dict:
    """``{rel_posix: tree}`` for every scanned module."""
    out: dict = {}
    for py in MLFRAME_DIR.rglob("*.py"):
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS) or py.name.endswith(".py.old"):
            continue
        tree = parsed_ast(py)
        if tree is not None:
            out[py.relative_to(MLFRAME_DIR).as_posix()] = tree
    return out


def test_cross_module_detector_is_not_blind():
    """An import-then-mutate of another module's dict is flagged; the same code under a ``with`` is not."""
    owner = ast.parse("_SHARED = {}\n")
    user = ast.parse("from ._owner import _SHARED\n\ndef put(k, v):\n    _SHARED[k] = v\n\ndef safe(k, v, lock):\n    with lock:\n        _SHARED[k] = v\n")
    found = _cross_module_offending({"pkg/_owner.py": owner, "pkg/user.py": user})
    assert found == {"pkg/user.py:3:put:mlframe.pkg._owner:_SHARED"}, found


def test_no_new_cross_module_dict_mutation_outside_a_lock():
    """No new function mutates another module's module-level dict without any ``with`` block."""
    current = _cross_module_offending(_all_modules())
    if _refresh_requested() or not _XMOD_BASELINE_PATH.exists():
        _XMOD_BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"cross-module dict-mutation baseline written with {len(current)} site(s)")
    baseline = set(orjson.loads(_XMOD_BASELINE_PATH.read_bytes()))
    new = sorted(current - baseline)
    if new:
        pytest.fail(
            f"{len(new)} function(s) mutate another module's dict with no lock held. The dict is shared state across modules; "
            "take the lock that guards it (add one in its home module if there is none), or baseline the site saying why it "
            "is single-threaded:\n  " + "\n  ".join(new[:30])
        )
