"""Meta-test: every ``importlib.reload(...)`` / ``del sys.modules[...]`` /
``sys.modules.pop(...)`` in tests/ must be paired with a snapshot+restore
mechanism IN THE SAME SCOPE so module rebinding does not split class
identity for sibling tests.

Background (canonical pollution pattern documented in CLAUDE.md "Test
pollution: never rebind module objects without snapshot/restore"):
reloading an mlframe module rebinds its top-level symbols. Every test
file that did ``from mlframe.X import Y`` at file-load keeps a reference
to the OLD Y; lazy ``from .X import Y`` inside function bodies resolves
to the NEW Y. The two ends disagree on class identity, breaking
class-attribute caches, isinstance checks, and idempotent install
markers (2026-05-22 MRMR fit-cache incident).

Sensor logic (AST-scoped, not whole-file string match):
- AST-scan every test file under tests/ for the three primitives.
- For each hit, locate its ENCLOSING function/fixture and require a
  paired restore mechanism reachable from that same scope:
  a) a restore in the enclosing function's own body
     (``finally``/teardown restoring ``sys.modules`` or a module
     ``__dict__``, or an ``addfinalizer(restore_callable)``), OR
  b) the enclosing function requests (by parameter name) a fixture
     defined in the same file whose body performs the restore, OR
  c) an autouse fixture in the same file performs the restore, OR
  d) a ``subprocess.run(...)`` in the enclosing function (full
     process isolation -- no in-process module mutation to restore).

Whole-file string matching was the prior (too coarse) heuristic: a file
could carry a restore marker in an unrelated function and pass even
though the reload site itself was unpaired. The AST scoping closes that.

Additionally, reloads/deletes of mlframe modules that own module-level
mutable singletons (caches/registries/locks) are flagged with an
escalated message even when a ``__dict__`` swap is present, since a
``__dict__`` restore does not rebuild a singleton that other modules
already captured by reference.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


TESTS_DIR = Path(__file__).resolve().parent.parent

# Files known to use the primitives ONLY against non-mlframe stub modules
# (e.g. ``polars_ds`` mock injection) where rebinding does not split
# mlframe class identity. Listed by repo-relative posix path.
_KNOWN_STUB_ONLY_FILES: frozenset[str] = frozenset({
    "training/test_pipeline_json_roundtrip_cache.py",
})

# mlframe modules that own module-level mutable singletons (caches / registries / locks).
# Reloading these splits the singleton even with a ``__dict__`` snapshot+restore, because
# importers captured the OLD singleton object by reference. Used to escalate the failure message.
_SINGLETON_OWNING_MODULES: frozenset[str] = frozenset({
    "mlframe.feature_selection.filters.mrmr",
    "mlframe.training.phases",
    "mlframe.training.composite.cache",
    "mlframe.training.suite_artefact_cache",
    "mlframe.system.kernel_tuning_cache",
})

_RELOAD_PRIMITIVES = ("importlib.reload", "del sys.modules", "sys.modules.pop")


def _is_reload_call(node: ast.AST) -> str | None:
    """Return the primitive name if ``node`` is a reload primitive call/delete, else None."""
    if isinstance(node, ast.Call):
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr == "reload"
                and isinstance(func.value, ast.Name) and func.value.id == "importlib"):
            return "importlib.reload"
        if (isinstance(func, ast.Attribute) and func.attr == "pop"
                and isinstance(func.value, ast.Attribute) and func.value.attr == "modules"
                and isinstance(func.value.value, ast.Name) and func.value.value.id == "sys"):
            return "sys.modules.pop"
    if isinstance(node, ast.Delete):
        for tgt in node.targets:
            if (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Attribute)
                    and tgt.value.attr == "modules"
                    and isinstance(tgt.value.value, ast.Name) and tgt.value.value.id == "sys"):
                return "del sys.modules"
    return None


def _reload_targets_singleton_module(src: str) -> bool:
    """True if any reload primitive in ``src`` names a known singleton-owning mlframe module literal."""
    return any(mod in src for mod in _SINGLETON_OWNING_MODULES)


def _node_has_restore(node: ast.AST) -> bool:
    """Detect a restore mechanism anywhere within ``node`` (a function/fixture body subtree).

    Accepted: ``sys.modules[...] = <name>`` re-assignment (snapshot restore), ``sys.modules.update(...)``,
    a module ``__dict__.clear()`` / ``__dict__.update(...)`` pair, ``addfinalizer(<callable>)``, or
    ``subprocess.run(...)`` (process isolation)."""
    for sub in ast.walk(node):
        # sys.modules[...] = ...  (restore from a saved reference)
        if isinstance(sub, ast.Assign):
            for tgt in sub.targets:
                if (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Attribute)
                        and tgt.value.attr == "modules"
                        and isinstance(tgt.value.value, ast.Name) and tgt.value.value.id == "sys"):
                    return True
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Attribute):
                # sys.modules.update(...)
                if (f.attr == "update" and isinstance(f.value, ast.Attribute)
                        and f.value.attr == "modules"
                        and isinstance(f.value.value, ast.Name) and f.value.value.id == "sys"):
                    return True
                # <mod>.__dict__.update(...) / .clear()  -- module __dict__ swap
                if f.attr in ("update", "clear") and isinstance(f.value, ast.Attribute) and f.value.attr == "__dict__":
                    return True
                # request.addfinalizer(<restore>)
                if f.attr == "addfinalizer":
                    return True
                # subprocess.run(...)
                if (f.attr == "run" and isinstance(f.value, ast.Name) and f.value.id == "subprocess"):
                    return True
    return False


def _iter_func_defs(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _is_fixture(func: ast.AST) -> tuple[bool, bool]:
    """Return (is_fixture, is_autouse) for a function def by inspecting its decorators."""
    is_fixture = False
    is_autouse = False
    for dec in getattr(func, "decorator_list", []):
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute) and target.attr == "fixture":
            is_fixture = True
        elif isinstance(target, ast.Name) and target.id == "fixture":
            is_fixture = True
        if isinstance(dec, ast.Call):
            for kw in dec.keywords:
                if kw.arg == "autouse" and isinstance(kw.value, ast.Constant) and kw.value.value:
                    is_autouse = True
    return is_fixture, is_autouse


def _enclosing_func(tree: ast.AST, target: ast.AST):
    """Return the innermost FunctionDef enclosing ``target`` (by line span), else None."""
    best = None
    for func in _iter_func_defs(tree):
        start = func.lineno
        end = getattr(func, "end_lineno", start)
        if start <= target.lineno <= end:
            if best is None or func.lineno > best.lineno:
                best = func
    return best


def _collect_unsafe_sites() -> list[tuple[str, int, str, bool]]:
    """Return (rel_path, lineno, primitive, hits_singleton) tuples for AST-scoped-unsafe sites."""
    out: list[tuple[str, int, str, bool]] = []
    for py in TESTS_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not any(tok in src for tok in _RELOAD_PRIMITIVES):
            continue
        rel = py.relative_to(TESTS_DIR).as_posix()
        if rel in _KNOWN_STUB_ONLY_FILES:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue

        # Restore-bearing fixtures defined in this file (name -> has_restore); plus any autouse restore fixture.
        fixture_restore: dict[str, bool] = {}
        autouse_restore = False
        for func in _iter_func_defs(tree):
            is_fixture, is_autouse = _is_fixture(func)
            if not is_fixture:
                continue
            has = _node_has_restore(func)
            fixture_restore[func.name] = has
            if is_autouse and has:
                autouse_restore = True

        hits_singleton = _reload_targets_singleton_module(src)

        for node in ast.walk(tree):
            prim = _is_reload_call(node)
            if prim is None:
                continue
            enclosing = _enclosing_func(tree, node)
            safe = False
            if autouse_restore:
                safe = True
            elif enclosing is not None:
                if _node_has_restore(enclosing):
                    safe = True
                else:
                    # The enclosing function may delegate the restore to a requested fixture.
                    arg_names = {a.arg for a in enclosing.args.args}
                    if any(fixture_restore.get(name, False) for name in arg_names):
                        safe = True
            if not safe:
                out.append((rel, node.lineno, prim, hits_singleton))
    return out


def test_no_unpaired_module_reload_in_tests():
    sites = _collect_unsafe_sites()
    if sites:
        formatted = "\n  ".join(
            f"{p}:{l} ({prim})" + ("  [!! reloads a singleton-owning mlframe module]" if sing else "")
            for p, l, prim, sing in sites[:30]
        )
        more = f"\n  ... and {len(sites) - 30} more" if len(sites) > 30 else ""
        pytest.fail(
            f"{len(sites)} test reload site(s) lack a snapshot+restore mechanism reachable from the "
            f"SAME function/fixture scope. Per CLAUDE.md 'Test pollution: never rebind module objects "
            f"without snapshot/restore' this is a cross-test pollution risk. Restore in the same "
            f"function (finally / addfinalizer), a requested fixture, an autouse fixture, or isolate "
            f"the reload in a subprocess. Reloads of singleton-owning modules need subprocess "
            f"isolation (a __dict__ swap does not rebuild the captured singleton).\n  "
            + formatted + more
        )


def test_known_stub_only_files_actually_exist():
    """Guard against the allowlist going stale (file renames / removals)."""
    for rel in _KNOWN_STUB_ONLY_FILES:
        assert (TESTS_DIR / rel).exists(), (
            f"_KNOWN_STUB_ONLY_FILES entry {rel!r} no longer exists; prune the "
            f"allowlist in tests/test_meta/test_no_unsafe_module_reload.py"
        )
