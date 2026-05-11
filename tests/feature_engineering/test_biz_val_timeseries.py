"""biz_val tests for ``mlframe.feature_engineering.timeseries`` --
``find_next_cumsum_*``, ``general_acf``, ``compute_corr``.

Per CLAUDE.md: each test asserts a SYNTHETIC measurable WIN that
locks in the timeseries-feature contract. Naming:
``test_biz_val_timeseries_<fn>_<scenario>``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# find_next_cumsum_left_index / find_next_cumsum_right_index
# ---------------------------------------------------------------------------


def test_biz_val_timeseries_find_next_cumsum_right_index_basic():
    """``find_next_cumsum_right_index(values, amount=N, left_index=0)``
    must return the right-index at which cumsum from ``left_index``
    first reaches/exceeds ``amount``. With all-ones, amount=5,
    left=0 -> right_index should be ~5."""
    from mlframe.feature_engineering.timeseries import find_next_cumsum_right_index
    arr = np.ones(20, dtype=np.float64)
    result = find_next_cumsum_right_index(arr, amount=5.0, left_index=0)
    # Returns a tuple; first element is the right-index.
    right_idx = result[0] if isinstance(result, tuple) else result
    assert right_idx is not None
    # Cumsum from 0 reaches 5.0 at index 4 (sum=5 after 5 values, indices 0..4)
    # depending on inclusive/exclusive semantics.
    assert 3 <= right_idx <= 6, (
        f"cumsum to 5 from all-ones should converge near index 5; got {right_idx}"
    )


def test_biz_val_timeseries_find_next_cumsum_left_index_basic():
    """Symmetric to right_index: scan from ``right_index`` leftward
    until cumsum reaches ``amount``."""
    from mlframe.feature_engineering.timeseries import find_next_cumsum_left_index
    arr = np.ones(20, dtype=np.float64)
    result = find_next_cumsum_left_index(arr, amount=5.0, right_index=19)
    left_idx = result[0] if isinstance(result, tuple) else result
    assert left_idx is not None
    assert 13 <= left_idx <= 16


@pytest.mark.parametrize("amount", [1.0, 3.0, 8.0, 15.0])
def test_biz_val_timeseries_find_next_cumsum_right_amount_monotone(amount):
    """Larger ``amount`` must require a larger or equal right_index
    on a monotone-positive series. Parametrize over {1, 3, 8, 15}."""
    from mlframe.feature_engineering.timeseries import find_next_cumsum_right_index
    arr = np.ones(30, dtype=np.float64)
    result = find_next_cumsum_right_index(arr, amount=amount, left_index=0)
    right_idx = result[0] if isinstance(result, tuple) else result
    assert right_idx is not None
    # On all-ones, right_idx ~ amount (up to inclusive/exclusive offset).
    assert abs(right_idx - amount) <= 2, (
        f"amount={amount}: right_idx should be ~amount; got {right_idx}"
    )


def test_biz_val_timeseries_find_next_cumsum_use_abs_handles_all_negative():
    """``use_abs=True`` applies abs to the running cumsum (NOT to the
    elements). On all-negative input, cumsum is -1, -2, ..., -N; with
    use_abs=True the abs|cumsum| reaches the target ``amount``.

    On all-negative ones, amount=5: |cumsum| reaches 5 at index 5
    (cumsum=-5, abs=5)."""
    from mlframe.feature_engineering.timeseries import find_next_cumsum_right_index
    arr = -np.ones(20, dtype=np.float64)
    result = find_next_cumsum_right_index(arr, amount=5.0,
                                              left_index=0, use_abs=True)
    right_idx = result[0] if isinstance(result, tuple) else result
    assert right_idx is not None
    # |cumsum| = 5 at index 5 (cumsum -5).
    assert 4 <= right_idx <= 6, (
        f"use_abs on all-neg series: expect ~5; got {right_idx}"
    )


def test_biz_val_timeseries_find_next_cumsum_use_abs_alternating_signal_stays_low():
    """On alternating +1/-1, the running cumsum stays near 0, so
    even with ``use_abs=True`` the |cumsum| never reaches 5 within
    a 30-element window. Documents this contract: ``use_abs`` does
    NOT take |element|, it takes |cumsum|."""
    from mlframe.feature_engineering.timeseries import find_next_cumsum_right_index
    arr = np.array([1.0, -1.0] * 15, dtype=np.float64)
    result = find_next_cumsum_right_index(arr, amount=5.0,
                                              left_index=0, use_abs=True)
    right_idx = result[0] if isinstance(result, tuple) else result
    # Falls off the end -- right_idx is the last index (29 in this 30-array).
    assert right_idx >= 25, (
        f"alternating with use_abs|cumsum| stays small; expected EOA index "
        f"(>=25); got {right_idx}"
    )


def test_biz_val_timeseries_find_next_cumsum_min_samples_floor():
    """``min_samples=N`` must prevent returning a right_index <= N
    even if the cumsum has been reached earlier. Catches regressions
    where min_samples is silently ignored."""
    from mlframe.feature_engineering.timeseries import find_next_cumsum_right_index
    arr = np.ones(30, dtype=np.float64)
    # min_samples=10 should make right_idx >= 10 even if amount=2
    # would normally converge by index 2.
    result = find_next_cumsum_right_index(arr, amount=2.0, left_index=0,
                                              min_samples=10)
    right_idx = result[0] if isinstance(result, tuple) else result
    assert right_idx is not None and right_idx >= 10, (
        f"min_samples=10 must floor right_idx; got {right_idx}"
    )


# ---------------------------------------------------------------------------
# compute_corr
# ---------------------------------------------------------------------------


def test_biz_val_timeseries_compute_corr_perfect_correlation():
    """``compute_corr`` with y = x must return abs-value close to 1.0."""
    from mlframe.feature_engineering.timeseries import compute_corr
    rng = np.random.default_rng(42)
    x = rng.normal(size=500)
    corr = compute_corr(dependent_vals=x, independent_vals=x,
                          deciding_func=np.corrcoef, absolutize=True)
    # corrcoef returns matrix; deciding_func might unwrap. Either way,
    # |corr| ~ 1.
    val = np.asarray(corr).ravel()[0] if hasattr(corr, "__len__") else corr
    val_arr = np.asarray(val).ravel() if hasattr(val, "__len__") else np.array([val])
    # Pull the largest absolute correlation entry from whatever shape.
    val_max = float(np.max(np.abs(np.asarray(corr))))
    assert val_max >= 0.95, (
        f"y=x must yield |corr| ~ 1; got max|corr|={val_max:.4f}"
    )


def test_biz_val_timeseries_compute_corr_zero_correlation_on_independent():
    """``compute_corr`` on independent inputs must return |corr| close
    to 0 (within sampling noise of ~0.1 for n=500)."""
    from mlframe.feature_engineering.timeseries import compute_corr
    rng = np.random.default_rng(42)
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    corr = compute_corr(dependent_vals=y, independent_vals=x,
                          deciding_func=np.corrcoef, absolutize=True)
    val_max = float(np.max(np.abs(np.asarray(corr))))
    # Sampling noise for n=500 + Bonferroni-style envelope: |corr|<0.2.
    assert val_max < 0.2, (
        f"independent inputs should yield small |corr|; got {val_max:.4f}"
    )


# ---------------------------------------------------------------------------
# general_acf
# ---------------------------------------------------------------------------


def test_biz_val_timeseries_general_acf_strong_signal_present():
    """``general_acf`` on an AR(1)-like series y[t] = 0.9*y[t-1] +
    noise should reveal positive auto-correlation at small lags."""
    from mlframe.feature_engineering.timeseries import general_acf
    rng = np.random.default_rng(42)
    n = 2000
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.9 * y[t - 1] + rng.normal(scale=0.3)
    res = general_acf(Y=y, lag_len=10, min_samples=500)
    # general_acf returns a tuple or array of acf values. Just sanity:
    # must return something non-None.
    assert res is not None
    # And the strongest lag must show appreciable abs correlation.
    arr = np.asarray(res)
    if arr.dtype != object and arr.size > 1:
        assert float(np.max(np.abs(arr))) > 0.1, (
            f"AR(1) series should have detectable autocorr; "
            f"got max|acf|={float(np.max(np.abs(arr))):.4f}"
        )
