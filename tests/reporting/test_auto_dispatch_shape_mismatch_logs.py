"""REPORTING_A-7 regression test: when target_type authoritatively selects the multiclass or binary
branch of render_multi_target_panels but the actual probs shape doesn't satisfy that branch's guard, the
function must log a warning before returning None -- matching the multilabel branch's existing behavior
on its analogous shape mismatch, instead of silently falling through with no log line.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.reporting import auto_dispatch, render_multi_target_panels

pytestmark = pytest.mark.fast


def test_multiclass_target_type_with_wrong_shape_logs_warning(tmp_path, caplog):
    """target_type='multiclass_classification' with 2-column probs (< K=3) must warn, not silently return None."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 50)
    proba = rng.dirichlet([1, 1], size=50)  # only 2 columns, fails the K>=3 multiclass gate

    with caplog.at_level(logging.WARNING, logger=auto_dispatch.logger.name):
        result = render_multi_target_panels(
            targets=y,
            probs=proba,
            target_type="multiclass_classification",
            plot_outputs="matplotlib[png]",
            multiclass_panels="CONFUSION",
            base_path=str(tmp_path / "x"),
        )
    assert result is None
    assert any("multiclass_classification target_type" in r.getMessage() for r in caplog.records)


def test_binary_target_type_with_2d_targets_logs_warning(tmp_path, caplog):
    """target_type='binary_classification' with 2-D targets must warn, not silently return None."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=(50, 2))  # 2-D targets, fails the binary ndim==1 gate
    proba = rng.uniform(size=50)

    with caplog.at_level(logging.WARNING, logger=auto_dispatch.logger.name):
        result = render_multi_target_panels(
            targets=y,
            probs=proba,
            target_type="binary_classification",
            plot_outputs="matplotlib[png]",
            binary_panels="ROC",
            base_path=str(tmp_path / "x"),
        )
    assert result is None
    assert any("binary_classification target_type" in r.getMessage() for r in caplog.records)


def test_binary_target_type_with_unusable_probs_shape_logs_warning(tmp_path, caplog):
    """target_type='binary_classification' with 1-D targets but a probs shape that resolves to no usable
    score column (e.g. 3 columns) must warn, not silently return None."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 50)
    proba = rng.dirichlet([1, 1, 1], size=50)  # 3 columns, no y_score resolution for binary

    with caplog.at_level(logging.WARNING, logger=auto_dispatch.logger.name):
        result = render_multi_target_panels(
            targets=y,
            probs=proba,
            target_type="binary_classification",
            plot_outputs="matplotlib[png]",
            binary_panels="ROC",
            base_path=str(tmp_path / "x"),
        )
    assert result is None
    assert any("binary_classification target_type" in r.getMessage() for r in caplog.records)


def test_heuristic_mode_still_falls_through_without_warning(tmp_path, caplog):
    """Sanity: with target_type='' (heuristic mode), a multiclass-shaped-but-too-few-columns probs array
    must still silently fall through to try other branches -- the new guard must not fire for tt=''."""
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 50)
    proba = rng.dirichlet([1, 1], size=50)  # 2 columns: resolves as binary, not multiclass

    with caplog.at_level(logging.WARNING, logger=auto_dispatch.logger.name):
        result = render_multi_target_panels(
            targets=y,
            probs=proba,
            target_type="",
            plot_outputs="matplotlib[png]",
            multiclass_panels="CONFUSION",
            binary_panels="ROC",
            base_path=str(tmp_path / "x"),
        )
    # Heuristic mode should fall through to binary and succeed, not hit the new authoritative-mismatch warning.
    assert result == "binary"
    assert not any("multiclass_classification target_type" in r.getMessage() for r in caplog.records)
