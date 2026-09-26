"""A test module that trains the full suite from three or more tests shares one training through a fixture.

``train_mlframe_models_suite`` is the most expensive call a test can make. test_composite_integration.py ran it from eight
test functions, six of them on the same frame with nearly the same config, each to inspect one metadata slot - ten minutes
for assertions that needed one run. A module-scoped fixture that trains once and lets each test assert against the result
did the same job in 45 seconds. Modules where three or more test functions call the suite directly are recorded in
``_heavy_training_baseline.json`` with their count, which may only shrink.
"""

from __future__ import annotations

from tests.test_meta._scan_guard import assert_scanned_enough

import ast
import orjson
from pathlib import Path

from tests.test_meta._shared_ast_cache import parsed_ast

_TESTS_DIR = Path(__file__).resolve().parents[1]
_BASELINE = Path(__file__).resolve().parent / "_heavy_training_baseline.json"
_SUITE_CALLS = frozenset({"train_mlframe_models_suite"})
_THRESHOLD = 3


def direct_suite_trainings(tree: ast.Module) -> list[str]:
    """Names of the test functions that call the suite themselves (a fixture that does it is the fix, not a finding)."""
    out = []
    for func in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")):
        if any(isinstance(c, ast.Call) and getattr(c.func, "id", getattr(c.func, "attr", None)) in _SUITE_CALLS for c in ast.walk(func)):
            out.append(func.name)
    return out


def _scan() -> dict[str, int]:
    """``{module: n direct suite trainings}`` for every test module at or over the threshold."""
    found = {}
    _files = sorted(_TESTS_DIR.rglob("test_*.py"))
    assert_scanned_enough(len(_files), str(_TESTS_DIR), minimum=20)
    for path in _files:
        if "__pycache__" in path.parts:
            continue
        tree = parsed_ast(path)
        if tree is None:
            continue
        n = len(direct_suite_trainings(tree))
        if n >= _THRESHOLD:
            found[path.relative_to(_TESTS_DIR).as_posix()] = n
    return found


def test_heavy_trainings_are_shared_through_a_fixture():
    """No module gains direct suite trainings beyond its recorded count; a module that drains must lower its record."""
    got = _scan()
    recorded = orjson.loads(_BASELINE.read_bytes())
    grew = {m: (recorded.get(m, 0), n) for m, n in got.items() if n > recorded.get(m, 0)}
    shrank = {m: (n, got.get(m, 0)) for m, n in recorded.items() if got.get(m, 0) < n}
    assert not grew, ("these modules train the full suite from more test functions than recorded; train once in a module-scoped "
                      f"fixture and assert against it (recorded, now): {grew}")
    assert not shrank, f"these modules train less often now; lower their record in {_BASELINE.name} (recorded, now): {shrank}"


def test_the_scan_counts_direct_calls_and_not_the_fixture():
    """Canary: three test functions calling the suite are counted; a fixture that calls it is not."""
    src = (
        "import pytest\n"
        "@pytest.fixture(scope='module')\n"
        "def run():\n"
        "    return train_mlframe_models_suite(df=1)\n"
        "def test_a():\n"
        "    train_mlframe_models_suite(df=1)\n"
        "def test_b(run):\n"
        "    assert run\n"
        "def test_c():\n"
        "    core.train_mlframe_models_suite(df=2)\n"
    )
    assert direct_suite_trainings(ast.parse(src)) == ["test_a", "test_c"]
