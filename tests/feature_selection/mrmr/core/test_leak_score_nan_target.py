"""Regression tests: NaN target rows do not hide or fake a leak, and a failing check is logged."""

import logging

import numpy as np

from mlframe.feature_selection.filters import _leak_suspects
from mlframe.utils.log_throttle import reset_throttle_counts


def test_leak_score_ignores_nan_target_rows_binary():
    """A perfectly separating column must still be flagged when a few target rows are NaN."""
    rng = np.random.default_rng(0)
    y = (rng.random(400) > 0.5).astype(float)
    col = y + 0.01 * rng.random(400)
    y[:5] = np.nan
    assert _leak_suspects.leak_score(col, y) >= _leak_suspects.SUSPECT_SCORE


def test_leak_score_ignores_nan_target_rows_continuous():
    """A monotone column must keep a Spearman score near 1 when a few target rows are NaN."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=400)
    col = y**3
    y[:5] = np.nan
    assert _leak_suspects.leak_score(col, y) >= _leak_suspects.SUSPECT_SCORE


def test_record_leak_suspects_failure_is_logged(caplog):
    """A broken estimator must produce a warning, not a debug line."""
    reset_throttle_counts()

    class _Est:
        """An estimator without the fitted attributes."""

    with caplog.at_level(logging.WARNING, logger=_leak_suspects.logger.name):
        out = _leak_suspects.record_leak_suspects(_Est(), None, np.zeros(3))
    assert out == []
    assert any("leak suspect check skipped" in r.getMessage() for r in caplog.records)
