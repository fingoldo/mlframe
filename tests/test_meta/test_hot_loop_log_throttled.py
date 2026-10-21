"""Regression guard: a warning or error logged once per iteration of a loop must go through ``log_throttle``.

The ``unthrottled_hot_loop_log`` scanner flagged bare ``logger.warning``/``logger.error`` calls inside loops in the
files below. Each now routes through ``mlframe.utils.log_throttle.log_throttle`` so a failing item logs a bounded
number of times per call site instead of once per iteration. This test fails if a bare level call reappears inside a
``for``/``while`` body in any of these files.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import mlframe

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent

_FILES = [
    "evaluation/_bootstrap_fused_binary_bundle.py",
    "feature_selection/filters/_mrmr_fit_impl/_finalise.py",
    "feature_selection/filters/_mrmr_fit_impl/_fit_impl_stages/_cat_fe.py",
    "preprocessing/gaussian_power_transform_search.py",
    "training/_commit_headroom.py",
    "training/_inert_fields.py",
    "training/cb/_cb_eval_weights.py",
    "training/composite/discovery/_tiny_rerank.py",
    "training/composite/transforms/linear.py",
    "training/core/_phase_composite_post_moe.py",
    "training/core/_target_slots.py",
    "training/core/predict.py",
    "training/pipeline/_per_target_supervised_fe.py",
]

_BARE_LEVELS = {"warning", "error", "exception", "critical", "warn"}


def _bare_level_calls_in_loops(tree: ast.AST) -> list[int]:
    """Line numbers of bare ``logger``-style level calls nested inside a loop body."""
    found: list[int] = []
    for loop in ast.walk(tree):
        if not isinstance(loop, (ast.For, ast.While, ast.AsyncFor)):
            continue
        for node in ast.walk(loop):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _BARE_LEVELS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"logger", "log", "_logger", "_log"}
            ):
                found.append(node.lineno)
    return sorted(set(found))


@pytest.mark.parametrize("rel", _FILES)
def test_no_bare_level_log_inside_loop(rel):
    """No file in the list logs at warning/error level from inside a loop."""
    tree = ast.parse((MLFRAME_DIR / rel).read_text(encoding="utf-8"))
    assert _bare_level_calls_in_loops(tree) == [], f"{rel}: unthrottled warning/error inside a loop at lines {_bare_level_calls_in_loops(tree)}"
