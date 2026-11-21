"""Cluster 23 regressions: the leakage p-value is reported with an explicit tolerance, and an event an hour before midnight is a fraction of a day earlier (audit 2026-11-20, findings 23-F5, 23-F8)."""

import numpy as np

from mlframe.evaluation.expanding_window_leakage import _leak_verdict


def test_pvalue_is_reported_when_an_explicit_tolerance_is_given():
    """A caller's leak_tolerance used to leave the p-value NaN (23-F5)."""
    leaky = np.array([0.90, 0.92, 0.95, 0.91, 0.93])
    honest = np.array([0.80, 0.81, 0.82, 0.79, 0.80])
    _tol, pval, _det = _leak_verdict(leaky, honest, float(honest.mean()), float((leaky - honest).mean()), 0.01)
    assert np.isfinite(pval)


def test_event_an_hour_before_midnight_is_closer_than_a_full_day_before():
    """An hour before a midnight event is nearer than a full day before it; the floored day count made them equal (23-F8)."""
    import pandas as pd

    from mlframe.feature_engineering.event_proximity_decay import event_proximity_decay_features

    event = [pd.Timestamp("2026-01-02")]
    near = event_proximity_decay_features(pd.Series(pd.to_datetime(["2026-01-01 23:00:00"])), event, cap=30)
    far = event_proximity_decay_features(pd.Series(pd.to_datetime(["2026-01-01 00:00:00"])), event, cap=30)
    assert float(near.to_numpy(dtype=float).sum()) > float(far.to_numpy(dtype=float).sum())
