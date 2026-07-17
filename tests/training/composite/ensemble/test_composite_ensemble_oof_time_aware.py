"""Regression: ``compute_oof_holdout_predictions`` must use a time-aware
split (past train / future holdout) when row ordering is monotone in time,
instead of a random shuffle that leaks the future into the past.

Pre-fix: always ``rng.permutation`` regardless of ordering.
Post-fix: if ``time_ordering`` is monotone (or the base column is), the
holdout slice is the trailing rows; train_idx + holdout_idx are
contiguous, non-overlapping, and the holdout's min index strictly
exceeds the train's max index.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from mlframe.training.composite.ensemble import (
    _is_monotone_nondecreasing,
    compute_oof_holdout_predictions,
)


class TestMonotoneDetection:
    """Groups tests covering monotone detection."""
    def test_monotone_int_array(self) -> None:
        """Monotone int array."""
        assert _is_monotone_nondecreasing(np.arange(100))
        assert _is_monotone_nondecreasing(np.array([1.0, 1.0, 2.0, 3.0]))

    def test_non_monotone(self) -> None:
        """Non monotone."""
        assert not _is_monotone_nondecreasing(np.array([1.0, 3.0, 2.0]))

    def test_with_nan(self) -> None:
        """With nan."""
        assert not _is_monotone_nondecreasing(np.array([1.0, np.nan, 3.0]))

    def test_short(self) -> None:
        """Short."""
        assert not _is_monotone_nondecreasing(np.array([1.0]))


class _DummyRegressor:
    """Always predicts the mean of y_train; honours fit/predict so the OOF
    helper's clone path works."""

    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(self, X, y):
        """Fit."""
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        """Predict."""
        n = len(X)
        return np.full(n, self._mean if self._mean is not None else 0.0)

    def get_params(self, deep: bool = True) -> dict:
        """Get params."""
        return {}

    def set_params(self, **kw):
        """Set params."""
        return self


def test_time_ordering_signal_produces_contiguous_future_holdout() -> None:
    """Time ordering signal produces contiguous future holdout."""
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "base": np.arange(n, dtype=np.float64),  # monotone -> looks like time
        }
    )
    y = rng.normal(size=n)
    timestamps = np.arange(n, dtype=np.float64)
    component_models = [_DummyRegressor()]
    holdout_preds, y_holdout, surviving = compute_oof_holdout_predictions(
        component_models=component_models,
        component_names=["c0"],
        component_specs=[None],
        train_X=df,
        y_train_full=y,
        base_train_full_per_spec={},
        holdout_frac=0.2,
        random_state=42,
        time_ordering=timestamps,
    )
    expected_holdout_n = round(n * 0.2)
    assert holdout_preds.shape == (expected_holdout_n, 1)
    # y_holdout must match the LAST 20% of y exactly (time-aware tail slice).
    np.testing.assert_array_equal(y_holdout, y[-expected_holdout_n:])
    assert surviving == ["c0"]


def test_monotone_base_column_no_longer_auto_switches() -> None:
    """A monotone base column with no explicit ``time_ordering`` must NOT flip to a trailing-slice split.

    The base-column auto-probe was removed: only an explicit ``time_ordering`` signal turns on the time-aware
    holdout, so a sorted-but-non-temporal base no longer silently changes the OOF leakage profile.
    """
    rng = np.random.default_rng(1)
    n = 400
    base = np.arange(n, dtype=np.float64)  # monotone "timestamp-like" base
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "base": base,
        }
    )
    y = rng.normal(size=n)
    component_models = [_DummyRegressor()]
    holdout_preds, y_holdout, _surviving = compute_oof_holdout_predictions(
        component_models=component_models,
        component_names=["c0"],
        component_specs=[None],
        train_X=df,
        y_train_full=y,
        base_train_full_per_spec={"base": base},
        holdout_frac=0.25,
        random_state=42,
        time_ordering=None,
    )
    expected = round(n * 0.25)
    assert holdout_preds.shape == (expected, 1)
    # Random shuffle: the holdout is NOT the contiguous tail.
    tail = y[-expected:]
    assert not np.array_equal(np.sort(y_holdout), np.sort(tail))


def test_random_split_fallback_when_not_monotone() -> None:
    """Non-monotone base + no explicit time_ordering -> legacy random
    shuffle behaviour (holdout y is NOT just the tail slice)."""
    rng = np.random.default_rng(2)
    n = 400
    base = rng.normal(size=n)  # NOT monotone
    df = pd.DataFrame({"f1": rng.normal(size=n), "base": base})
    y = rng.normal(size=n)
    component_models = [_DummyRegressor()]
    _, y_holdout, _ = compute_oof_holdout_predictions(
        component_models=component_models,
        component_names=["c0"],
        component_specs=[None],
        train_X=df,
        y_train_full=y,
        base_train_full_per_spec={"base": base},
        holdout_frac=0.25,
        random_state=42,
        time_ordering=None,
    )
    expected = round(n * 0.25)
    # With overwhelming probability, the random shuffle does NOT pick the
    # contiguous tail; assert at least one tail row was NOT chosen.
    tail = y[-expected:]
    assert not np.array_equal(np.sort(y_holdout), np.sort(tail))
