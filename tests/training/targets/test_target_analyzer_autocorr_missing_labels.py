"""The target analyzer measures lag autocorrelation only when enough rows are labelled for lags to be lags."""

from __future__ import annotations

import numpy as np

from mlframe.training.targets._target_distribution_analyzer_target_fn import analyze_target_distribution


def _ar_series(n=3000, phi=0.9, seed=0):
    """AR(1) series with coefficient phi."""
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = 0.0
    for i in range(1, n):
        y[i] = phi * y[i - 1] + rng.normal()
    return y


def _diagnostics(report):
    """Diagnostics from a report that may be a dict or an object."""
    return report["diagnostics"] if isinstance(report, dict) else report.diagnostics


def test_a_sparsely_labelled_target_does_not_get_an_autocorrelation_from_compacted_rows():
    """A sparsely labelled target does not get an autocorrelation from compacted rows."""
    y = _ar_series()
    y[np.random.default_rng(1).random(len(y)) < 0.5] = np.nan
    diag = _diagnostics(analyze_target_distribution(y, target_type="regression", has_time_axis=True))
    assert "max_abs_autocorr" not in diag and diag["autocorr_skipped_labelled_share"] < 0.9


def test_a_fully_labelled_target_still_gets_its_autocorrelation():
    """A fully labelled target still gets its autocorrelation."""
    diag = _diagnostics(analyze_target_distribution(_ar_series(), target_type="regression", has_time_axis=True))
    assert diag["max_abs_autocorr"] > 0.8
