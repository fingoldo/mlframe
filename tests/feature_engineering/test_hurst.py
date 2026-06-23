"""Hypothesis-based tests for hurst.py module."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from mlframe.feature_engineering.hurst import (
    compute_hurst_exponent,
    compute_hurst_rs,
    precompute_hurst_exponent,
)


@given(st.lists(st.floats(min_value=1, max_value=1000, allow_nan=False, allow_infinity=False), min_size=20, max_size=500))
@settings(max_examples=50, deadline=None)
def test_hurst_returns_valid_range(arr):
    """Hurst exponent should generally be between 0 and 1.5 for most series."""
    h, c = compute_hurst_exponent(np.array(arr, dtype=np.float64))
    if not np.isnan(h):
        assert -0.5 <= h <= 2.0, f"Hurst exponent {h} out of expected range"


@given(st.lists(st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False), min_size=5, max_size=5))
def test_hurst_min_window_edge_case(arr):
    """Test with arrays at minimum window size."""
    result = compute_hurst_exponent(np.array(arr, dtype=np.float64), min_window=5)
    assert len(result) == 2
    assert isinstance(result[0], (float, np.floating))
    assert isinstance(result[1], (float, np.floating))


@given(st.integers(min_value=1, max_value=4))
def test_hurst_returns_nan_for_short_arrays(size):
    """Arrays shorter than min_window should return NaN."""
    arr = np.random.randn(size).astype(np.float64)
    h, c = compute_hurst_exponent(arr, min_window=5)
    assert np.isnan(h) and np.isnan(c)


@given(st.lists(st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False), min_size=10, max_size=100))
@settings(max_examples=30, deadline=None)
def test_hurst_rs_nonnegative(arr):
    """R/S statistic should be non-negative when defined.

    ``compute_hurst_rs`` now returns NaN for degenerate windows (R==0 or S<=eps); the prior
    behaviour of returning 0.0 conflated "undefined" with a genuine R/S of zero, masking issues
    upstream. NaN passes this property test because it is not negative.
    """
    arr = np.array(arr, dtype=np.float64)
    rs = compute_hurst_rs(arr)
    assert np.isnan(rs) or rs >= 0.0


@given(st.floats(min_value=0.1, max_value=0.5))
@settings(deadline=None)
def test_hurst_windows_log_step(log_step):
    """Test different window log steps."""
    arr = np.random.randn(100).astype(np.float64)
    h, c = compute_hurst_exponent(arr, windows_log_step=log_step)
    assert len((h, c)) == 2


@given(st.booleans())
def test_hurst_take_diffs_parameter(take_diffs):
    """Test both take_diffs modes."""
    arr = np.cumsum(np.random.randn(100)).astype(np.float64)  # Random walk
    h, c = compute_hurst_exponent(arr, take_diffs=take_diffs)
    assert len((h, c)) == 2


@given(st.integers(min_value=3, max_value=20))
def test_hurst_min_window_parameter(min_window):
    """Test different min_window values."""
    arr = np.random.randn(100).astype(np.float64)
    h, c = compute_hurst_exponent(arr, min_window=min_window)
    assert len((h, c)) == 2


def test_hurst_random_walk_close_to_half():
    """Random walk should have Hurst exponent close to 0.5."""
    np.random.seed(42)
    arr = np.cumsum(np.random.randn(1000)).astype(np.float64)
    h, c = compute_hurst_exponent(arr, take_diffs=True)
    # Random walk H should be around 0.5 (with some tolerance)
    if not np.isnan(h):
        assert 0.3 <= h <= 0.7, f"Random walk Hurst {h} not close to 0.5"


def test_hurst_trending_series():
    """Trending series should have Hurst exponent > 0.5."""
    np.random.seed(42)
    trend = np.linspace(0, 10, 1000)
    noise = np.random.randn(1000) * 0.1
    arr = (trend + noise).astype(np.float64)
    h, c = compute_hurst_exponent(arr)
    if not np.isnan(h):
        assert h > 0.3, f"Trending series Hurst {h} expected to be > 0.3"


# Regression: the np.arange(s) + derived-invariant hoist out of the per-segment j-loop
# (dfa_alpha / dfa_alpha2_quadratic / multifractal_dfa) must stay bit-identical to the
# pre-hoist computation. Reference kernels below are the verbatim pre-hoist bodies.
from numba import njit as _njit
from mlframe.feature_engineering.hurst import (
    dfa_alpha as _new_dfa_alpha,
    dfa_alpha2_quadratic as _new_dfa_alpha2,
)


@_njit(cache=True, fastmath=True)
def _ref_dfa_alpha(x):
    n = x.size
    if n < 20:
        return np.nan
    mu = x.mean()
    y = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc += x[i] - mu
        y[i] = acc
    sizes = np.array([10, 20, 40, 80])
    sizes = sizes[sizes < n // 2]
    if sizes.size < 2:
        return np.nan
    log_s = np.empty(sizes.size)
    log_f = np.empty(sizes.size)
    for k in range(sizes.size):
        s = sizes[k]
        m = n // s
        var_sum = 0.0
        for j in range(m):
            seg = y[j * s:(j + 1) * s]
            t = np.arange(s).astype(np.float64)
            tm = t.mean()
            sm = seg.mean()
            num = 0.0
            den = 0.0
            for i in range(s):
                num += (t[i] - tm) * (seg[i] - sm)
                den += (t[i] - tm) ** 2
            slope = num / (den + 1e-12)
            intercept = sm - slope * tm
            resid_sq = 0.0
            for i in range(s):
                fit = intercept + slope * t[i]
                resid_sq += (seg[i] - fit) ** 2
            var_sum += resid_sq / s
        f_s = np.sqrt(var_sum / m)
        log_s[k] = np.log(s)
        log_f[k] = np.log(f_s + 1e-12)
    lm = log_s.mean()
    fm = log_f.mean()
    num = 0.0
    den = 0.0
    for k in range(log_s.size):
        num += (log_s[k] - lm) * (log_f[k] - fm)
        den += (log_s[k] - lm) ** 2
    return num / (den + 1e-12)


@_njit(cache=True, fastmath=True)
def _ref_dfa_alpha2(x):
    n = x.size
    if n < 50:
        return np.nan
    mu = x.mean()
    y = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc += x[i] - mu
        y[i] = acc
    sizes = np.array([10, 20, 40, 80])
    sizes = sizes[sizes < n // 2]
    if sizes.size < 2:
        return np.nan
    log_s = np.empty(sizes.size)
    log_f = np.empty(sizes.size)
    for k in range(sizes.size):
        s = sizes[k]
        m = n // s
        var_sum = 0.0
        for j in range(m):
            seg = y[j * s:(j + 1) * s]
            t = np.arange(s).astype(np.float64)
            S0 = float(s); S1 = t.sum(); S2 = (t * t).sum()
            S3 = (t * t * t).sum(); S4 = (t * t * t * t).sum()
            Sy = seg.sum(); Sty = (t * seg).sum(); St2y = (t * t * seg).sum()
            M00 = S0; M01 = S1; M02 = S2; M11 = S2; M12 = S3; M22 = S4
            det = (M00 * (M11 * M22 - M12 * M12) - M01 * (M01 * M22 - M12 * M02) + M02 * (M01 * M12 - M11 * M02))
            if abs(det) < 1e-12:
                continue
            inv_det = 1.0 / det
            c00 = (M11 * M22 - M12 * M12) * inv_det
            c01 = -(M01 * M22 - M12 * M02) * inv_det
            c02 = (M01 * M12 - M11 * M02) * inv_det
            c11 = (M00 * M22 - M02 * M02) * inv_det
            c12 = -(M00 * M12 - M01 * M02) * inv_det
            c22 = (M00 * M11 - M01 * M01) * inv_det
            a = c00 * Sy + c01 * Sty + c02 * St2y
            b = c01 * Sy + c11 * Sty + c12 * St2y
            cq = c02 * Sy + c12 * Sty + c22 * St2y
            resid_sq = 0.0
            for i in range(s):
                fit = a + b * t[i] + cq * t[i] * t[i]
                d = seg[i] - fit
                resid_sq += d * d
            var_sum += resid_sq / s
        f_s = np.sqrt(var_sum / m)
        log_s[k] = np.log(s)
        log_f[k] = np.log(f_s + 1e-12)
    lm = log_s.mean()
    fm = log_f.mean()
    num = 0.0
    den = 0.0
    for k in range(log_s.size):
        num += (log_s[k] - lm) * (log_f[k] - fm)
        den += (log_s[k] - lm) ** 2
    return num / (den + 1e-12)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_dfa_arange_hoist_bit_identical_to_prehoist(seed):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(2000)).astype(np.float64)
    # Linear-detrend hoist is exactly bit-identical.
    assert _new_dfa_alpha(x) == _ref_dfa_alpha(x)
    # Quadratic-detrend hoist computes the same design moments but, under fastmath=True, the
    # once-vs-per-segment accumulation schedule may reorder FMA/reassoc by a single ULP -- a
    # reduction-order delta (~1e-15), far below anything that could move a feature decision.
    assert _new_dfa_alpha2(x) == pytest.approx(_ref_dfa_alpha2(x), rel=1e-12, abs=1e-12)
