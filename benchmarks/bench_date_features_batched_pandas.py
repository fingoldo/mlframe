"""Benchmark: pandas branch of create_date_features / add_cyclical_date_features.

Compares the OLD one-column-at-a-time insertion (double loop, block-manager
fragmentation) against the NEW single-concat batched insertion, on a wide
date-feature set (6 date cols x 8 methods + cyclical sin/cos ~= 100 new cols).

Also asserts the batched output is column-for-column, dtype-for-dtype,
value-for-value identical to the loop output (assert_frame_equal).

Run:  python benchmarks/bench_date_features_batched_pandas.py
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from mlframe.feature_engineering.basic import (  # noqa: E402
    create_date_features,
    _DEFAULT_DATE_METHODS,
    _DEFAULT_CYCLICAL_PERIODS,
    _resolve_pandas_method,
    _cyclical_sincos_njit,
)

def _make_wide_df(n: int, n_date_cols: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2020-01-01")
    data = {}
    for i in range(n_date_cols):
        secs = rng.integers(0, 5 * 365 * 24 * 3600, size=n)
        data[f"dt{i}"] = base + pd.to_timedelta(secs, unit="s")
    # a couple of non-date columns to make concat non-trivial
    data["x"] = rng.standard_normal(n)
    data["y"] = rng.integers(0, 100, size=n)
    return pd.DataFrame(data)


def _old_impl(df: pd.DataFrame, cols, methods, periods):
    """Replica of the pre-fix double-loop pandas branch (column-at-a-time)."""
    two_pi = float(2.0 * np.pi)
    warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
    df = df.copy(deep=False)
    precomputed = {}
    for col in cols:
        obj = df[col].dt
        for method, dtype in methods.items():
            field = _resolve_pandas_method(obj, method, dtype)
            df[col + "_" + method] = field
            if method != "is_weekend":
                precomputed[(col, method)] = field.to_numpy()
    for col in cols:
        obj = df[col].dt
        for period_name, period_value in periods:
            pc = precomputed.get((col, period_name))
            if pc is not None:
                base = np.ascontiguousarray(pc, dtype=np.float64)
            else:
                base = np.ascontiguousarray(
                    _resolve_pandas_method(obj, period_name, np.float64).to_numpy(),
                    dtype=np.float64,
                )
            s, c = _cyclical_sincos_njit(base, two_pi / float(period_value))
            df[f"{col}_{period_name}_sin"] = s
            df[f"{col}_{period_name}_cos"] = c
    return df


def _time(fn, *a, repeat=9, **kw):
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    cols = [f"dt{i}" for i in range(6)]
    methods = dict(_DEFAULT_DATE_METHODS)
    periods = _DEFAULT_CYCLICAL_PERIODS
    # warmup njit + import caches
    _old_impl(_make_wide_df(2000), cols, methods, periods)
    create_date_features(_make_wide_df(2000), cols=cols, methods=methods,
                         add_cyclical=True, cyclical_periods=periods, delete_original_cols=False)
    for n in (10_000, 200_000):
        df = _make_wide_df(n)
        # OLD path (suppress the PerformanceWarning it legitimately raises)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            t_old, out_old = _time(_old_impl, df, cols, methods, periods)
        # NEW path (library function, must NOT raise PerformanceWarning)
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=pd.errors.PerformanceWarning)
            create_date_features(df, cols=cols, methods=methods, add_cyclical=True,
                                 cyclical_periods=periods, delete_original_cols=False)
        t_new, out_new = _time(
            create_date_features, df, cols=cols, methods=methods,
            add_cyclical=True, cyclical_periods=periods,
            delete_original_cols=False,
        )
        pd.testing.assert_frame_equal(out_old, out_new)
        n_new = out_new.shape[1] - df.shape[1]
        speedup = t_old / t_new
        print(f"n={n:>7}  new_cols={n_new:>3}  old={t_old*1e3:8.2f}ms  "
              f"new={t_new*1e3:8.2f}ms  speedup={speedup:5.2f}x  identical=OK")


if __name__ == "__main__":
    main()
