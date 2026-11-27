"""An MRMR instance built without ``__init__`` must run the optimizer ``MRMR()`` runs (audit 2026-11-20, D_api_tests_perf F04).

The readers of ``fe_optimizer`` take ``getattr(self, "fe_optimizer", <fallback>)``; a fallback that differs from the constructor default silently runs
another optimizer on an instance created through ``object.__new__`` or a subclass that bypasses the constructor.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from mlframe.feature_selection.filters.mrmr import MRMR
from tests.test_meta._scan_guard import assert_scanned_enough

_SRC = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection"


def _fallbacks_of(attribute: str) -> "list[tuple[str, int, object]]":
    """``(file, line, fallback)`` of every ``getattr(self, "<attribute>", <literal>)`` under the feature-selection package."""
    found: list[tuple[str, int, object]] = []
    files = [path for path in sorted(_SRC.rglob("*.py")) if "_benchmarks" not in path.parts]
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) == 3):
                continue
            obj, name, fallback = node.args
            if isinstance(obj, ast.Name) and obj.id == "self" and isinstance(name, ast.Constant) and name.value == attribute and isinstance(fallback, ast.Constant):
                found.append((path.relative_to(_SRC).as_posix(), node.lineno, fallback.value))
    assert_scanned_enough(len(files), "src/mlframe/feature_selection")
    return found


def test_fe_optimizer_fallback_equals_the_constructor_default():
    """Every literal fallback of ``fe_optimizer`` in the feature-selection package equals the default of ``MRMR.__init__``."""
    default = inspect.signature(MRMR.__init__).parameters["fe_optimizer"].default
    fallbacks = _fallbacks_of("fe_optimizer")
    assert fallbacks, "no getattr(self, 'fe_optimizer', ...) reader found: the scan no longer matches the code, or the readers moved"
    wrong = [(file, line, value) for file, line, value in fallbacks if value != default]
    assert not wrong, f"fallbacks that differ from the constructor default {default!r}: {wrong}"
