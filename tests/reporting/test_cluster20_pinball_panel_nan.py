"""The pinball-by-alpha panel drops non-finite rows instead of printing a NaN mean (audit 2026-11-20, finding 20-F17)."""

import numpy as np

from mlframe.reporting.charts.quantile import _pinball_by_alpha_panel


def test_one_nan_prediction_does_not_turn_the_pinball_title_into_nan():
    """The panel must not report 'mean=nan' because a single row is non-finite."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=60)
    preds = np.column_stack([np.quantile(y, 0.1) * np.ones(60), np.quantile(y, 0.9) * np.ones(60)])
    preds[3, 0] = np.nan
    spec = _pinball_by_alpha_panel(y, preds, [0.1, 0.9])
    assert "nan" not in str(spec).lower()
