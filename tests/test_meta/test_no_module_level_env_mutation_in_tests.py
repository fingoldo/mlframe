"""Meta-test: a test module must not mutate ``os.environ`` at import time.

Under ``pytest-xdist`` one worker process imports many test modules and then runs tests from all of them.
A module-level ``os.environ[...] = ...`` therefore applies to every test that worker runs afterwards, not
just to the file that wrote it - and which files share a worker depends on the distribution, so the damage
lands on a different, innocent test on each run. The symptom is a cross-file failure that disappears the
moment the victim is run alone, which is the most expensive kind of failure to chase.

``os.environ.setdefault`` is the worst form: it looks defensive, is silent when the variable is already
set, and is never restored. ``tests/feature_selection/biz_val/test_biz_val_hybrid_cooccur_clusterrep.py``
blanked ``CUDA_VISIBLE_DEVICES`` this way at import, which disabled the GPU for every later test in the
same worker. That stayed invisible while no production code read the variable live; once the GPU opt-out
became a hard contract, it turned into ~70 spurious GPU-test failures.

The fix is always ``monkeypatch.setenv`` inside the test or a fixture, which pytest restores. If a variable
genuinely must be set before an import that happens at module scope, do it in ``conftest.py`` with an
autouse fixture, or move the import inside the test.

Refresh with::

    pytest tests/test_meta/test_no_module_level_env_mutation_in_tests.py --refresh-module-env-mutation-baseline
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import orjson
import pytest

_TESTS_DIR = Path(__file__).resolve().parent.parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_module_env_mutation_baseline.json"

_MUTATING_METHODS = frozenset({"setdefault", "update", "pop", "clear", "popitem"})


def _refresh_requested() -> bool:
    """True if ``--refresh-module-env-mutation-baseline`` was passed on the pytest command line."""
    return "--refresh-module-env-mutation-baseline" in sys.argv


def _is_environ(node: ast.AST) -> bool:
    """True for ``os.environ`` / a bare ``environ`` reference (both are the same process-wide mapping)."""
    if isinstance(node, ast.Attribute) and node.attr == "environ":
        return True
    return isinstance(node, ast.Name) and node.id == "environ"


_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _import_time_nodes(tree: ast.Module):
    """Yield every node that is evaluated at import time.

    Descends into module-level ``if`` / ``try`` / ``with`` / ``for`` bodies, since those run at import too,
    but never into a function/class/lambda body - code there runs when called, which is the whole point of
    the recommended fix. The pruning has to be done by hand: ``ast.walk`` has no way to skip a subtree, so
    filtering its output only drops the ``def`` node itself while still yielding everything inside it.
    """
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, _SCOPE_NODES):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _env_mutations(tree: ast.Module) -> list[tuple[int, str]]:
    """``[(lineno, what), ...]`` for each import-time ``os.environ`` mutation or ``os.putenv`` call."""
    out: list[tuple[int, str]] = []
    for node in _import_time_nodes(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and _is_environ(t.value):
                    out.append((node.lineno, "os.environ[...] = ..."))
        elif isinstance(node, ast.Delete):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and _is_environ(t.value):
                    out.append((node.lineno, "del os.environ[...]"))
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in _MUTATING_METHODS and _is_environ(f.value):
                out.append((node.lineno, f"os.environ.{f.attr}(...)"))
            elif isinstance(f, ast.Attribute) and f.attr in ("putenv", "unsetenv"):
                out.append((node.lineno, f"os.{f.attr}(...)"))
    return out


def _build_offending_set() -> set[str]:
    """``{"relpath:lineno:what", ...}`` for every import-time environment mutation under ``tests/``."""
    out: set[str] = set()
    for py in _TESTS_DIR.rglob("test_*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        rel = py.relative_to(_TESTS_DIR).as_posix()
        for lineno, what in _env_mutations(tree):
            out.add(f"{rel}:{lineno}:{what}")
    return out


def test_no_new_module_level_env_mutation():
    """No test module gains an import-time ``os.environ`` mutation beyond the baseline."""
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(orjson.dumps(sorted(current), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8")
        pytest.skip(f"module-env-mutation baseline written with {len(current)} entry/entries")

    baseline = set(orjson.loads(_BASELINE_PATH.read_bytes()))
    added = sorted(current - baseline)
    drained = sorted(baseline - current)

    if drained:
        sys.stderr.write(
            f"\n[test_no_new_module_level_env_mutation] {len(drained)} site(s) DRAINED:\n  "
            + "\n  ".join(drained)
            + "\n  Refresh: pytest ... --refresh-module-env-mutation-baseline\n"
        )

    assert not added, (
        f"{len(added)} test module(s) mutate os.environ at IMPORT time. Under xdist one worker imports many "
        "test modules into a single process, so this leaks into every later test that worker runs - and which "
        "tests those are changes with the distribution, so it fails a different innocent file each run. Use "
        "monkeypatch.setenv in the test or a fixture (pytest restores it); if a variable must be set before a "
        "module-scope import, put it in conftest.py or move the import into the test.\n  " + "\n  ".join(added)
    )


_DETECTOR_SAMPLE = '''
import os

os.environ["A"] = "1"
os.environ.setdefault("B", "")

if True:
    os.environ["C"] = "3"

def f():
    os.environ["SAFE"] = "ok"

class K:
    def m(self):
        os.environ.setdefault("ALSO_SAFE", "ok")
'''


def test_detector_sees_import_time_writes_and_ignores_function_bodies():
    """The scan flags the three import-time writes and neither of the two inside callables.

    A baseline-diff test passes precisely when it finds nothing new, so a detector that quietly stopped
    matching would look permanently healthy.
    """
    found = _env_mutations(ast.parse(_DETECTOR_SAMPLE))
    assert sorted(ln for ln, _ in found) == [4, 5, 8], found
