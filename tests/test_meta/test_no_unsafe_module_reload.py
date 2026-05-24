"""Meta-test: every ``importlib.reload(...)`` / ``del sys.modules[...]`` /
``sys.modules.pop(...)`` in tests/ must be paired with a snapshot+restore
mechanism so module rebinding does not split class identity for sibling
tests.

Background (canonical pollution pattern documented in CLAUDE.md "Test
pollution: never rebind module objects without snapshot/restore"):
reloading an mlframe module rebinds its top-level symbols. Every test
file that did ``from mlframe.X import Y`` at file-load keeps a reference
to the OLD Y; lazy ``from .X import Y`` inside function bodies resolves
to the NEW Y. The two ends disagree on class identity, breaking
class-attribute caches, isinstance checks, and idempotent install
markers (2026-05-22 MRMR fit-cache incident).

Sensor logic:
- AST-scan every test file under tests/ for the three primitives.
- For each hit, check the SAME file contains a paired restore mechanism:
  a) An ``addfinalizer`` registered with a function that touches
     ``sys.modules`` or a module ``__dict__``, OR
  b) A snapshot of ``sys.modules`` (``dict(sys.modules.items())`` /
     ``{n: m for n, m in sys.modules.items()...}``) restored after
     ``yield`` / in a ``finally`` block, OR
  c) A module ``__dict__`` snapshot+restore pattern, OR
  d) A subprocess invocation that isolates the reload entirely.

A baseline file tracks legitimate violations (subprocess probes whose
restore is implicit, dynamic stub modules that aren't mlframe ones).
New unpaired uses fail the test.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest


TESTS_DIR = Path(__file__).resolve().parent.parent

# Files known to use the primitives ONLY against non-mlframe stub modules
# (e.g. ``polars_ds`` mock injection) where rebinding does not split
# mlframe class identity. Listed by repo-relative posix path.
_KNOWN_STUB_ONLY_FILES: frozenset[str] = frozenset({
    "training/test_pipeline_json_roundtrip_cache.py",
})


def _file_has_restore_mechanism(src: str) -> bool:
    """Heuristic: detect any of the four accepted restore mechanisms.

    String-level checks intentionally (the patterns are stable enough
    and a full data-flow analysis would be overkill for a sensor).
    """
    markers = (
        # snapshot+restore of sys.modules
        "sys.modules[name] = snapshot",
        "sys.modules.update(snapshot",
        "sys.modules[_key] = _saved",
        # module __dict__ snapshot+restore (see test_automl.py fixture)
        "_mod_ref.__dict__.clear()",
        "_mod_ref.__dict__.update(",
        "__dict__.update(_saved_dict",
        # explicit addfinalizer paired with restore callable
        "addfinalizer(_restore",
        # subprocess isolation
        "subprocess.run(",
    )
    return any(m in src for m in markers)


def _collect_unsafe_sites() -> list[tuple[str, int, str]]:
    """Return (rel_path, lineno, primitive) tuples for unsafe sites.

    A site is unsafe iff the file contains one of the three primitives
    AND does NOT contain any of the accepted restore markers.
    """
    out: list[tuple[str, int, str]] = []
    for py in TESTS_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        # Cheap pre-filter; skip files with no reload primitive at all.
        if not any(tok in src for tok in ("importlib.reload", "del sys.modules", "sys.modules.pop")):
            continue
        rel = py.relative_to(TESTS_DIR).as_posix()
        if rel in _KNOWN_STUB_ONLY_FILES:
            continue
        if _file_has_restore_mechanism(src):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # importlib.reload(X)
                if (isinstance(func, ast.Attribute) and func.attr == "reload"
                        and isinstance(func.value, ast.Name) and func.value.id == "importlib"):
                    out.append((rel, node.lineno, "importlib.reload"))
                # sys.modules.pop(...)
                if (isinstance(func, ast.Attribute) and func.attr == "pop"
                        and isinstance(func.value, ast.Attribute)
                        and func.value.attr == "modules"
                        and isinstance(func.value.value, ast.Name)
                        and func.value.value.id == "sys"):
                    out.append((rel, node.lineno, "sys.modules.pop"))
            # del sys.modules[<expr>]
            if isinstance(node, ast.Delete):
                for tgt in node.targets:
                    if (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Attribute)
                            and tgt.value.attr == "modules"
                            and isinstance(tgt.value.value, ast.Name)
                            and tgt.value.value.id == "sys"):
                        out.append((rel, node.lineno, "del sys.modules"))
    return out


def test_no_unpaired_module_reload_in_tests():
    sites = _collect_unsafe_sites()
    if sites:
        formatted = "\n  ".join(f"{p}:{l} ({prim})" for p, l, prim in sites[:30])
        more = f"\n  ... and {len(sites) - 30} more" if len(sites) > 30 else ""
        pytest.fail(
            f"{len(sites)} test file(s) call importlib.reload / del sys.modules / "
            f"sys.modules.pop without a snapshot+restore mechanism in the same "
            f"file. Per CLAUDE.md 'Test pollution: never rebind module objects "
            f"without snapshot/restore' this is a cross-test pollution risk. "
            f"Wrap the reload in a fixture that snapshots either ``sys.modules`` "
            f"or the target module's ``__dict__`` and restores it at teardown.\n  "
            + formatted + more
        )


def test_known_stub_only_files_actually_exist():
    """Guard against the allowlist going stale (file renames / removals)."""
    for rel in _KNOWN_STUB_ONLY_FILES:
        assert (TESTS_DIR / rel).exists(), (
            f"_KNOWN_STUB_ONLY_FILES entry {rel!r} no longer exists; prune the "
            f"allowlist in tests/test_meta/test_no_unsafe_module_reload.py"
        )
