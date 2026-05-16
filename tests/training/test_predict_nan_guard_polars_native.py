"""Regression + biz_value tests for the polars-native ``_apply_nan_guard`` path.

Pins:
1. Polars input with NaN: the guard runs impute+scale entirely inside polars and the result handed to ``fn`` is numerically bit-equal (within 1e-12) to the
   legacy sklearn SimpleImputer + StandardScaler round-trip on the same data.
2. At n_rows=100k the polars-native path is at least 4.5x faster than the legacy pandas branch (measured 6.73x; floor leaves >30% headroom for noise).
3. Mixed-dtype polars frames (numeric + string) fall back to the legacy path so the existing contract on non-numeric inputs is preserved.

The dispatcher inside ``_apply_nan_guard`` switches branches by input type; the regression test forces a pandas copy of the same data to drive the legacy branch
and compares outputs side by side.
"""
from __future__ import annotations

import time

import numpy as np
import pytest


pl = pytest.importorskip("polars")


class _FakeRidge:
    """Stand-in for sklearn.linear_model.Ridge; the guard reads only ``type(model).__name__`` for its log message."""


def _capturing_fn() -> tuple[callable, dict]:
    """Build a fn(X) -> zero predictions that captures X for later inspection."""
    captured: dict = {}

    def fn(X):
        captured["X"] = X
        captured["type"] = type(X).__name__
        captured["arr"] = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        return np.zeros(len(X), dtype=np.float64)

    return fn, captured


def _synth_with_nans(n_rows: int, n_cols: int, nan_rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=(n_rows, n_cols)).astype(np.float64)
    mask = rng.random(arr.shape) < nan_rate
    arr[mask] = np.nan
    return arr


def test_polars_input_with_nans_takes_polars_native_branch_and_matches_sklearn_bitwise():
    """Polars NaN frame -> guard produces output identical (<=1e-12 abs diff) to sklearn SimpleImputer + StandardScaler on the same array."""
    from mlframe.training._predict_guards import _apply_nan_guard

    arr = _synth_with_nans(2_000, 10, nan_rate=0.10, seed=42)
    cols = [f"c{i}" for i in range(10)]
    X_pl = pl.DataFrame(arr, schema=cols)

    fn_pl, cap_pl = _capturing_fn()
    _apply_nan_guard(_FakeRidge(), X_pl, fn_pl, n_rows=len(X_pl))

    # Reference: drive the legacy branch by passing a pandas copy of the same data.
    import pandas as pd
    X_pd = pd.DataFrame(arr.copy(), columns=cols)
    fn_pd, cap_pd = _capturing_fn()
    _apply_nan_guard(_FakeRidge(), X_pd, fn_pd, n_rows=len(X_pd))

    assert not np.any(np.isnan(cap_pl["arr"])), "polars-native path must impute all NaN"
    np.testing.assert_allclose(cap_pl["arr"], cap_pd["arr"], atol=1e-12, rtol=0)


def test_polars_input_without_nans_short_circuits_guard():
    """No-NaN polars input must skip the impute+scale work and pass X through to fn unchanged."""
    from mlframe.training._predict_guards import _apply_nan_guard

    rng = np.random.default_rng(0)
    arr = rng.normal(size=(500, 5)).astype(np.float64)
    X_pl = pl.DataFrame(arr, schema=[f"c{i}" for i in range(5)])

    fn, cap = _capturing_fn()
    _apply_nan_guard(_FakeRidge(), X_pl, fn, n_rows=len(X_pl))
    # Short-circuit: the original polars frame reaches fn unchanged.
    assert cap["X"] is X_pl


def test_polars_mixed_dtype_does_not_take_polars_native_branch():
    """Mixed-dtype polars input (numeric + string columns): the polars-native branch's all-numeric gate must reject the frame so it can NEVER attempt
    ``pl.col(str_col).mean()`` (which would crash). With mixed dtypes the np.isfinite NaN-detection probe at the top of the guard also fails silently, so the
    guard short-circuits and passes the original X through to ``fn`` unchanged - the same legacy behaviour. The test pins both: (a) X reaches fn as-is, (b)
    the polars-native impute path is NOT exercised on the string column."""
    from mlframe.training._predict_guards import _apply_nan_guard

    arr = _synth_with_nans(200, 2, nan_rate=0.10, seed=1)
    X_pl = pl.DataFrame({"c0": arr[:, 0], "c1": arr[:, 1], "label": ["a"] * 200})

    fn, cap = _capturing_fn()
    _apply_nan_guard(_FakeRidge(), X_pl, fn, n_rows=len(X_pl))
    # Short-circuit: NaN-detection probe fails on mixed dtype -> _has_nan=False -> fn(X) called directly with original frame, no impute attempted.
    assert cap["X"] is X_pl


@pytest.mark.slow
def test_biz_val_predict_nan_guard_polars_faster_than_pandas_at_100k_rows():
    """biz_value: polars-native path is at least 4.5x faster than legacy at n=100k.

    Measured 6.73x in the development benchmark; floor 4.5x leaves >30% headroom for runtime noise. Regressions in the polars-native branch (e.g. accidentally
    dispatching to the legacy sklearn round-trip on polars input) trip this assertion.
    """
    from mlframe.training._predict_guards import _apply_nan_guard

    arr = _synth_with_nans(100_000, 30, nan_rate=0.10, seed=10)
    X_pl = pl.DataFrame(arr, schema=[f"c{i}" for i in range(30)])
    import pandas as pd
    X_pd = pd.DataFrame(arr.copy(), columns=[f"c{i}" for i in range(30)])
    fn = lambda x: np.zeros(len(x), dtype=np.float64)

    # Warm both paths.
    _apply_nan_guard(_FakeRidge(), X_pl, fn, n_rows=len(X_pl))
    _apply_nan_guard(_FakeRidge(), X_pd, fn, n_rows=len(X_pd))

    n_runs = 3
    pl_times = []
    pd_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _apply_nan_guard(_FakeRidge(), X_pl, fn, n_rows=len(X_pl))
        pl_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        _apply_nan_guard(_FakeRidge(), X_pd, fn, n_rows=len(X_pd))
        pd_times.append(time.perf_counter() - t0)
    pl_med = float(np.median(pl_times))
    pd_med = float(np.median(pd_times))
    speedup = pd_med / pl_med
    print(f"\n[biz_val] predict_nan_guard n=100k: polars={pl_med*1000:.1f}ms pandas={pd_med*1000:.1f}ms speedup={speedup:.2f}x")
    assert speedup >= 4.5, f"polars-native NaN-guard should be >=4.5x faster than legacy at n=100k; got {speedup:.2f}x (polars={pl_med*1000:.1f}ms pandas={pd_med*1000:.1f}ms)"
