"""Rows no OOF fold validates on stay NaN, not a real-looking 0.0."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from mlframe.feature_engineering.transformer._oof import _allocate_oof_outputs


def test_fresh_output_buffers_are_nan():
    """All per-head buffers start as NaN, scalar and vector valued alike."""
    out = _allocate_oof_outputs(10, 2, 3, ("y_mean", "y_std", "x_mean"), np.float32)
    assert out and all(np.isnan(v).all() for v in out.values())


def test_timeseries_split_leaves_its_first_block_nan_after_scatter():
    """Scatter fold values at the val_idx of TimeSeriesSplit(5) over 60 rows: the first 10 rows (never validated) must remain NaN."""
    out = _allocate_oof_outputs(60, 1, 2, ("y_mean",), np.float32)
    for _tr, va in TimeSeriesSplit(5).split(np.arange(60)):
        out["y_mean_h0"][va] = 1.0
    assert np.isnan(out["y_mean_h0"][:10]).all() and (out["y_mean_h0"][10:] == 1.0).all()
