"""Meta-test: a bounded eviction may drop a cache entry, never a pending obligation.

``_DEFERRED_HOST_FILL`` in ``_gpu_resident_fe.py`` recorded host buffers that still had to be filled from the GPU. It
was capped at 8 with a FIFO ``popitem(last=False)``, and its reader, ``ensure_host_codes_filled``, did
``if h is None: return``. The FE pair scoring runs more than 8 threads, each registering a buffer and then waiting
on the numba kernel lock, so the oldest registrations were evicted and never filled. The CPU kernel read the
uninitialised buffers as bin codes: access violations on Windows, or MI computed on garbage. The lock gates did not
look at it (the name does not end in ``_CACHE``), and nothing distinguished "evicting a cache entry" from
"cancelling work".

Three checks:

1. Every eviction that throws the evicted entry away carries ``# evict-ok: <reason>`` saying why a miss is
   harmless (it recomputes, re-uploads, falls back). The reason is where a reviewer decides "cache or obligation".
2. An obligation registry (a module dict whose reader, on a miss, just returns) must not discard evicted entries at
   all, marker or not: the eviction has to do the pending work first.
3. An obligation registry that evicts in any form is bounded by design and has an overflow test, listed in
   ``OVERFLOW_TESTS``, that registers more entries than the cap from several threads.
"""

from __future__ import annotations

from tests.test_meta._scan_guard import assert_scanned_enough

import ast
from pathlib import Path

import pytest

import mlframe

from tests.test_meta._eviction_scan import discarded_evictions, marked_evict_ok, silent_miss_readers
from tests.test_meta._module_mutable_state import (
    aliases_in,
    base_name,
    build_dict_index,
    build_reexport_index,
    function_defs,
    imported_module_dicts,
    module_level_dicts,
    module_qualname,
)
from tests.test_meta._shared_ast_cache import parsed_ast

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent
REPO_ROOT = MLFRAME_DIR.parent.parent
_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "legacy", "profiling", "explore", "_benchmarks")

# "relpath:DICT_NAME" -> the test file that overflows that registry past its cap, concurrently.
OVERFLOW_TESTS = {
    "feature_selection/filters/_gpu_resident_fe.py:_DEFERRED_HOST_FILL": "tests/feature_selection/gpu/test_deferred_host_fill_never_dropped_unfilled.py",
}


def _sources():
    """Yield ``(relpath, tree, lines)`` for every scanned module under ``src/mlframe``."""
    _files = sorted(MLFRAME_DIR.rglob("*.py"))
    assert_scanned_enough(len(_files), "src/mlframe")
    for py in _files:
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
            continue
        tree = parsed_ast(py)
        assert tree is not None, f"{py} did not parse; this gate would silently skip it"
        yield py.relative_to(MLFRAME_DIR).as_posix(), tree, py.read_bytes().decode("utf-8", "replace").splitlines()


def _evicts_at_all(tree: ast.Module, name: str) -> bool:
    """True when some function evicts from ``name`` in any form, its result used or not."""
    names = set(module_level_dicts(tree))
    for fn in function_defs(tree):
        alias = aliases_in(fn, names)
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "popitem":
                b = base_name(node.func.value)
                if alias.get(b, b) == name:
                    return True
    return any(n == name for _ln, _fn, n in discarded_evictions(tree))


def _violations(tree: ast.Module, lines: list[str], rel: str, imported: dict | None = None, foreign_obligations: dict | None = None) -> list[str]:
    """Problems in one module, as readable lines.

    ``imported`` maps local names to another module's dict (``"origin:NAME"``); ``foreign_obligations`` maps such labels
    to that module's silent-miss readers, so evicting an obligation registry from outside its own module is caught too.
    """
    out: list[str] = []
    obligations = dict(silent_miss_readers(tree))
    obligations.update(foreign_obligations or {})
    for ln, fn, name in discarded_evictions(tree, imported):
        if name in obligations:
            out.append(
                f"{rel}:{ln} {fn} evicts from {name} and throws the entry away, but {obligations[name]} treat a missing "
                f"entry as nothing to do: evicting cancels pending work. Do the work on eviction instead."
            )
        elif not marked_evict_ok(lines, ln):
            out.append(f"{rel}:{ln} {fn} evicts from {name} without '# evict-ok: <why a miss is harmless>'")
    for name in obligations:
        if _evicts_at_all(tree, name) and f"{rel}:{name}" not in OVERFLOW_TESTS:
            out.append(f"{rel}: {name} is a bounded obligation registry with no overflow test in OVERFLOW_TESTS")
    return out


def test_no_eviction_drops_pending_work():
    """Every discarded eviction is a marked cache eviction, and no obligation registry discards entries, in any module."""
    modules = list(_sources())
    index = build_dict_index({rel: tree for rel, tree, _ in modules})
    reexports = build_reexport_index({rel: tree for rel, tree, _ in modules}, index)
    foreign: dict = {}
    for rel, tree, _ in modules:
        for name, readers in silent_miss_readers(tree).items():
            foreign[f"{module_qualname(rel)}:{name}"] = readers
    problems: list[str] = []
    for rel, tree, lines in modules:
        problems.extend(_violations(tree, lines, rel, imported_module_dicts(tree, rel, index, reexports), foreign))
    if problems:
        pytest.fail(f"{len(problems)} eviction problem(s):\n  " + "\n  ".join(problems))


def test_overflow_tests_exist_and_name_their_registry():
    """Each listed overflow test exists, names its registry, and the registry is still an obligation registry."""
    for key, test_rel in OVERFLOW_TESTS.items():
        rel, name = key.rsplit(":", 1)
        test_path = REPO_ROOT / test_rel
        assert test_path.exists(), f"overflow test {test_rel} for {key} is missing"
        assert name in test_path.read_text(encoding="utf-8"), f"{test_rel} never mentions {name}"
        tree = parsed_ast(MLFRAME_DIR / rel)
        assert tree is not None and name in silent_miss_readers(tree), f"{key} is no longer an obligation registry; drop it from OVERFLOW_TESTS"


_BUGGY_REGISTRY = """
from collections import OrderedDict
_PENDING = OrderedDict()

def stash(k, v):
    c = _PENDING
    c[k] = v
    while len(c) > 8:
        c.popitem(last=False)

def settle(k):
    h = _PENDING.get(k)
    if h is None:
        return
    h.run()
"""

_FIXED_REGISTRY = """
from collections import OrderedDict
_PENDING = OrderedDict()

def stash(k, v):
    _PENDING[k] = v
    while len(_PENDING) > 8:
        _PENDING.popitem(last=False)[1].run()

def settle(k):
    h = _PENDING.get(k)
    if h is None:
        return
    h.run()
"""

_CACHE_MARKED = """
_MEMO = {}

def get(k):
    v = _MEMO.get(k)
    if v is None:
        v = _MEMO[k] = k * 2
        if len(_MEMO) > 8:
            # evict-ok: memo; a miss recomputes the value
            _MEMO.pop(next(iter(_MEMO)))
    return v
"""


_FOREIGN_EVICTOR = """
from ._reg import _PENDING

def trim():
    while len(_PENDING) > 8:
        _PENDING.popitem(last=False)
"""


def test_detector_is_not_blind():
    """The shape of the real bug is flagged through a local alias; the fixed and the marked-cache shapes are not."""
    buggy = _violations(ast.parse(_BUGGY_REGISTRY), _BUGGY_REGISTRY.splitlines(), "m.py")
    assert any("cancels pending work" in v and "_PENDING" in v for v in buggy), buggy

    # Filling on eviction clears check 2, but the registry is still bounded, so check 3 wants an overflow test.
    fixed = _violations(ast.parse(_FIXED_REGISTRY), _FIXED_REGISTRY.splitlines(), "m.py")
    assert fixed == ["m.py: _PENDING is a bounded obligation registry with no overflow test in OVERFLOW_TESTS"], fixed

    assert _violations(ast.parse(_CACHE_MARKED), _CACHE_MARKED.splitlines(), "m.py") == []

    # Evicting another module's obligation registry, through an import, is the same bug.
    tree = ast.parse(_FOREIGN_EVICTOR)
    imported = imported_module_dicts(tree, "pkg/user.py", {"mlframe.pkg._reg": {"_PENDING"}})
    assert imported == {"_PENDING": "mlframe.pkg._reg:_PENDING"}, imported
    found = _violations(tree, _FOREIGN_EVICTOR.splitlines(), "pkg/user.py", imported, {"mlframe.pkg._reg:_PENDING": ["settle"]})
    assert len(found) == 1 and "cancels pending work" in found[0], found
    unmarked = _CACHE_MARKED.replace("# evict-ok: memo; a miss recomputes the value", "# trimmed")
    assert len(_violations(ast.parse(unmarked), unmarked.splitlines(), "m.py")) == 1


def test_a_dict_imported_through_a_reexport_resolves_to_its_origin():
    """A consumer that imports another module's dict through a re-export module (a package ``shared.py``) mutates the
    same shared state; the alias must resolve to the dict's origin or the mutation is invisible to the gates."""
    shared = ast.parse("from ._reg import _PENDING as PENDING  # noqa: F401\n")
    user = ast.parse("from mlframe.pkg.shared import PENDING as _pending\n")
    index = {"mlframe.pkg._reg": {"_PENDING"}}
    reexports = build_reexport_index({"pkg/shared.py": shared}, index)
    assert imported_module_dicts(user, "other/user.py", index, reexports) == {"_pending": "mlframe.pkg._reg:_PENDING"}
    assert imported_module_dicts(user, "other/user.py", index) == {}  # what the gates saw before: nothing
