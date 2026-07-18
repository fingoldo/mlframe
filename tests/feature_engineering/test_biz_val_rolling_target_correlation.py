"""biz_value test for ``feature_engineering.rolling_target_correlation_tracker``.

The win: when the true best predictive feature SWITCHES partway through a time series (a regime change --
feature "a" drives the target in the first half, feature "b" in the second), a STATIC one-time correlation
computed once (over an initial window) picks feature "a" and sticks with it forever, going badly wrong once
the regime switches. The rolling tracker recomputes the best feature on a trailing window and adapts,
recovering the target far better across the full series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from mlframe.feature_engineering import rolling_target_correlation_tracker


def _make_regime_switch_dataset(n: int, seed: int):
    """Helper: Make regime switch dataset."""
    rng = np.random.default_rng(seed)
    switch = n // 2
    feat_a = rng.normal(size=n)
    feat_b = rng.normal(size=n)
    noise_cols = rng.normal(size=(n, 5))
    y = np.where(np.arange(n) < switch, feat_a, feat_b) + rng.normal(scale=0.2, size=n)
    X = pd.DataFrame(np.column_stack([feat_a, feat_b, noise_cols]), columns=["a", "b"] + [f"n{i}" for i in range(5)])
    return X, y


def test_biz_val_rolling_target_correlation_tracker_beats_static_selection_mse():
    """Biz val rolling target correlation tracker beats static selection mse."""
    X, y = _make_regime_switch_dataset(n=2000, seed=0)
    tracker = rolling_target_correlation_tracker(X, y, window=200, min_periods=50)
    valid = tracker["dyn_target_corr_value"].notna().to_numpy()

    mse_dynamic = mean_squared_error(y[valid], tracker.loc[valid, "dyn_target_corr_value"])

    # Static baseline: pick the single globally-best-correlated feature over an initial window and use it
    # for the whole series (the "compute once, freeze forever" approach this idea improves on).
    static_corrs = {c: abs(np.corrcoef(X[c][:300], y[:300])[0, 1]) for c in X.columns}
    best_static = max(static_corrs, key=static_corrs.get)
    mse_static = mean_squared_error(y[valid], X.loc[valid, best_static])

    improvement = 1.0 - mse_dynamic / mse_static
    assert (
        improvement > 0.7
    ), f"expected >70% MSE reduction vs. a static one-time feature selection, got {improvement:.4f} (static={mse_static:.4f}, dynamic={mse_dynamic:.4f})"

    # The tracker should have actually detected the regime switch (both features get selected at some point).
    feature_counts = tracker.loc[valid, "dyn_target_corr_feature"].value_counts()
    assert "a" in feature_counts.index and "b" in feature_counts.index


def test_rolling_target_correlation_tracker_only_uses_past_rows():
    """The correlation driving row i's selection must be computable from rows strictly before i -- verify
    by checking early rows (before min_periods trailing observations exist) are NaN, not leaking a
    whole-series correlation into rows that shouldn't have one yet."""
    X, y = _make_regime_switch_dataset(n=500, seed=1)
    tracker = rolling_target_correlation_tracker(X, y, window=100, min_periods=50)
    assert tracker["dyn_target_corr_value"].iloc[:50].isna().all()
    assert tracker["dyn_target_corr_value"].iloc[100:].notna().any()


def test_rolling_target_correlation_tracker_requires_numeric_columns():
    """Rolling target correlation tracker requires numeric columns."""
    import pytest

    X = pd.DataFrame({"a": ["x", "y", "z"]})
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        rolling_target_correlation_tracker(X, y, window=2, feature_columns=[])
