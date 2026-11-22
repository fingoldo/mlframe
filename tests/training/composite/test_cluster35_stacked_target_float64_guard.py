"""Audit 2026-11-20 finding 35-F6: both stacked-fit entry points pull the target as float64.

A float32 target makes the residual ``y - pass1_pred`` inherit float32 rounding. A full stacked fit is too heavy for a unit test, so this
guard reads the two call sites and fails if either loses its float64 cast.
"""

import ast
from pathlib import Path

import mlframe.training.composite.discovery._stacked as stacked


def _target_pulls(func_name):
    """The ``y_full = ...`` assignments inside ``func_name`` and whether each one casts to float64."""
    tree = ast.parse(Path(stacked.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            out = []
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "y_full" for t in sub.targets):
                    src = ast.unparse(sub.value)
                    out.append("float64" in src)
            return out
    raise AssertionError(f"{func_name} not found")


def test_fit_stacked_pulls_float64_target():
    """fit_stacked casts its target to float64."""
    assert _target_pulls("fit_stacked") and all(_target_pulls("fit_stacked"))


def test_fit_stacked_on_residual_pulls_float64_target():
    """fit_stacked_on_residual casts its target to float64."""
    assert _target_pulls("fit_stacked_on_residual") and all(_target_pulls("fit_stacked_on_residual"))
