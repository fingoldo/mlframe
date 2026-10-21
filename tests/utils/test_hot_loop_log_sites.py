"""Regression guard: warnings raised inside a loop body in the audited modules go through ``log_throttle``, never a bare ``logger.warning``.

A bare ``logger.warning`` inside a per-item loop emits one line per failing item, which compounds into log spam under load. Each audited site
now routes through ``mlframe.utils.log_throttle.log_throttle``, which logs the first few occurrences per key and then a suppression notice.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from mlframe.utils.log_throttle import log_throttle, reset_throttle_counts

_REPO = Path(__file__).resolve().parents[2]

_AUDITED_FILES = [
    "src/mlframe/calibration/_post_train_calibrators.py",
    "src/mlframe/feature_selection/_benchmarks/bench_poly_fe_ml_metrics.py",
    "src/mlframe/feature_selection/_benchmarks/fs_hybrid/_arms_rank_aggregation.py",
    "src/mlframe/feature_selection/_benchmarks/fs_hybrid/_cell_store.py",
    "src/mlframe/feature_selection/_benchmarks/fs_hybrid/_panel.py",
    "src/mlframe/feature_selection/_benchmarks/fs_hybrid/_realdata_cache.py",
    "src/mlframe/feature_selection/_benchmarks/fs_hybrid/ablation_rfecv_aggregation.py",
    "src/mlframe/feature_selection/_benchmarks/fs_hybrid/sweep_recovery_vs_n.py",
    "src/mlframe/training/core/_misc_helpers_feature_types.py",
    "src/mlframe/training/core/_phase_composite_post_xt_ensemble/_xt_ensemble_helpers.py",
    "src/mlframe/training/core/_phase_fit_pipeline_blocks.py",
    "src/mlframe/training/core/_phase_helpers_fit_split.py",
    "src/mlframe/training/io.py",
    "src/mlframe/training/pipeline/__init__.py",
    "src/mlframe/training/strategies/_cat_levels_shared.py",
    "src/mlframe/training/strategies/hgb.py",
]


def _bare_warning_calls_in_loops(source: str) -> list[int]:
    """Return the line numbers of bare logger/log .warning calls nested inside a loop body."""
    tree = ast.parse(source)
    hits: list[int] = []
    for loop in ast.walk(tree):
        if not isinstance(loop, (ast.For, ast.While, ast.AsyncFor)):
            continue
        for node in ast.walk(loop):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "warning"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("logger", "log")
            ):
                hits.append(node.lineno)
    return sorted(set(hits))


@pytest.mark.parametrize("rel_path", _AUDITED_FILES)
def test_no_bare_warning_inside_loop(rel_path: str) -> None:
    """Audited modules route in-loop warnings through log_throttle."""
    source = (_REPO / rel_path).read_text(encoding="utf-8")
    assert _bare_warning_calls_in_loops(source) == [], f"{rel_path}: bare logger.warning inside a loop, route it through log_throttle"


def test_throttle_caps_repeated_hot_loop_warning(caplog: pytest.LogCaptureFixture) -> None:
    """log_throttle emits only a handful of warnings for one key repeated many times."""
    key = "test_hot_loop_cap"
    reset_throttle_counts(key)
    lg = logging.getLogger(key)
    with caplog.at_level(logging.WARNING, logger=key):
        for i in range(200):
            log_throttle(lg, key, logging.WARNING, "item %d failed", i)
    emitted = [r for r in caplog.records if r.name == key and "item" in r.getMessage()]
    assert 1 <= len(emitted) <= 10
    reset_throttle_counts(key)
