"""Audit 2026-11-20 finding 34-F22: module-level logger and unary sentinel are each defined exactly once in _fit_steps.

A duplicated assignment hides an edit: changing the second copy looks like it changes behaviour but the first one wins.
"""

import ast
from pathlib import Path

import mlframe.training.composite.discovery._fit_steps as fit_steps


def _module_assignments(name):
    """Count top-level assignments to ``name`` in the module source."""
    tree = ast.parse(Path(fit_steps.__file__).read_text(encoding="utf-8"))
    count = 0
    for node in tree.body:
        if isinstance(node, ast.Assign):
            count += sum(1 for t in node.targets if isinstance(t, ast.Name) and t.id == name)
    return count


def test_logger_is_defined_once():
    """A single top-level logger assignment."""
    assert _module_assignments("logger") == 1


def test_unary_sentinel_is_defined_once():
    """A single top-level _UNARY_BASE_SENTINEL assignment."""
    assert _module_assignments("_UNARY_BASE_SENTINEL") == 1
