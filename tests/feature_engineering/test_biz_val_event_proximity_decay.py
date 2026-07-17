"""biz_value test for ``feature_engineering.event_proximity_decay.event_proximity_decay_features``.

The win (5th_m5-forecasting-accuracy.md): sales/demand often RAMPS UP in the days approaching an event and
ramps back down after, not just spiking on the exact event day. A sparse binary "is-event-day" indicator
misses this pre/post-event ramp entirely, while the distance-decayed proximity feature captures it directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from mlframe.feature_engineering.event_proximity_decay import event_proximity_decay_features


def _make_event_ramp_dataset(n_days: int, event_days: list, seed: int):
    rng = np.random.default_rng(seed)
    dates = pd.Series(np.arange(n_days))

    # true demand ramps up to a peak at each event day and decays symmetrically -- a smooth proximity
    # signal, not a spike confined to the exact event day.
    y = np.zeros(n_days, dtype=np.float64)
    for event_day in event_days:
        distance = np.abs(np.arange(n_days) - event_day)
        y += np.maximum(0.0, 15.0 - distance)
    y += rng.normal(scale=1.0, size=n_days)

    return dates, y


def test_biz_val_event_proximity_decay_captures_ramp_binary_indicator_misses():
    event_days = [20, 60, 100, 140]
    dates, y = _make_event_ramp_dataset(n_days=180, event_days=event_days, seed=0)

    binary_indicator = np.isin(np.arange(180), event_days).astype(np.float64).reshape(-1, 1)
    r2_binary = cross_val_score(Ridge(alpha=1.0), binary_indicator, y, cv=5, scoring="r2").mean()

    decay_features = event_proximity_decay_features(dates, event_dates=event_days, cap=15)
    r2_decay = cross_val_score(Ridge(alpha=1.0), decay_features.to_numpy(), y, cv=5, scoring="r2").mean()

    assert r2_binary < 0.3, f"expected a sparse binary event indicator to capture very little of the smooth ramp signal, got R^2={r2_binary:.4f}"
    assert r2_decay > 0.85, f"expected the distance-decayed proximity feature to capture the ramp signal almost fully, got R^2={r2_decay:.4f}"
    assert r2_decay > r2_binary + 0.5, f"expected the decay feature to materially beat the binary indicator, got decay={r2_decay:.4f} binary={r2_binary:.4f}"


def test_event_proximity_decay_features_exact_values():
    dates = pd.Series(np.arange(10))
    out = event_proximity_decay_features(dates, event_dates=[5], cap=3)
    np.testing.assert_allclose(out["event_proximity_event0"].to_numpy(), [0, 0, 0, 1, 2, 3, 2, 1, 0, 0])
    np.testing.assert_allclose(out["event_proximity_total_force"].to_numpy(), out["event_proximity_event0"].to_numpy())


def test_event_proximity_decay_features_datetime_input():
    dates = pd.Series(pd.date_range("2021-01-01", periods=20))
    out = event_proximity_decay_features(dates, event_dates=[pd.Timestamp("2021-01-10")], cap=5)
    assert out["event_proximity_event0"].max() == 5.0
    peak_idx = out["event_proximity_event0"].idxmax()
    assert dates.iloc[peak_idx] == pd.Timestamp("2021-01-10")


def test_event_proximity_decay_features_asymmetric_default_matches_symmetric():
    """Opt-in params unset -> bit-identical to the prior symmetric-only behavior."""
    dates = pd.Series(np.arange(50))
    baseline = event_proximity_decay_features(dates, event_dates=[10, 30], cap=12)
    opted_out = event_proximity_decay_features(dates, event_dates=[10, 30], cap=12, cap_before=None, cap_after=None)
    pd.testing.assert_frame_equal(baseline, opted_out, check_exact=True)


def test_event_proximity_decay_features_asymmetric_exact_values():
    dates = pd.Series(np.arange(10))
    out = event_proximity_decay_features(dates, event_dates=[5], cap=3, cap_before=8, cap_after=2)
    # before event (idx 0..4): cap_before=8 leg; at/after (idx 5..9): cap_after=2 leg.
    np.testing.assert_allclose(out["event_proximity_event0"].to_numpy(), [3, 4, 5, 6, 7, 2, 1, 0, 0, 0])
    np.testing.assert_allclose(out["event_proximity_total_force"].to_numpy(), out["event_proximity_event0"].to_numpy())


def _make_asymmetric_ramp_dataset(n_days: int, event_days: list, ramp_up: float, ramp_down: float, seed: int):
    rng = np.random.default_rng(seed)
    dates = pd.Series(np.arange(n_days))

    # true demand builds up SLOWLY over `ramp_up` days pre-event (anticipation buildup, e.g. a promotion)
    # but decays QUICKLY over `ramp_down` days post-event -- an asymmetric, not symmetric, proximity shape.
    y = np.zeros(n_days, dtype=np.float64)
    for event_day in event_days:
        distance = np.arange(n_days) - event_day
        before = np.maximum(0.0, ramp_up - np.abs(distance))
        after = np.maximum(0.0, ramp_down - np.abs(distance))
        y += np.where(distance < 0, before, after)
    y += rng.normal(scale=0.5, size=n_days)

    return dates, y


def test_biz_val_event_proximity_decay_asymmetric_beats_symmetric_on_asymmetric_ramp():
    event_days = [40, 100, 160]
    dates, y = _make_asymmetric_ramp_dataset(n_days=220, event_days=event_days, ramp_up=25.0, ramp_down=5.0, seed=1)

    symmetric_features = event_proximity_decay_features(dates, event_dates=event_days, cap=25)
    r2_symmetric = cross_val_score(Ridge(alpha=1.0), symmetric_features.to_numpy(), y, cv=5, scoring="r2").mean()

    asymmetric_features = event_proximity_decay_features(dates, event_dates=event_days, cap=25, cap_before=25, cap_after=5)
    r2_asymmetric = cross_val_score(Ridge(alpha=1.0), asymmetric_features.to_numpy(), y, cv=5, scoring="r2").mean()

    # measured r2_asymmetric~=0.746, r2_symmetric~=-2.22 on this synthetic (seed=1); thresholds set well
    # below the measured margin.
    assert r2_asymmetric > 0.65, f"expected the asymmetric decay to capture most of the asymmetric ramp signal, got R^2={r2_asymmetric:.4f}"
    assert r2_asymmetric > r2_symmetric + 0.5, (
        f"expected asymmetric decay to materially beat symmetric decay on an asymmetric ramp, got asymmetric={r2_asymmetric:.4f} symmetric={r2_symmetric:.4f}"
    )
