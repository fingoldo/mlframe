"""Regression tests for the batched (single pd.concat) pandas insertion path in
``create_date_features`` / ``add_cyclical_date_features``.

Locks in FINDING #3: the pandas branch must accumulate derived columns and insert
them in ONE pd.concat instead of column-at-a-time assignment. Asserts:

* no ``pandas.errors.PerformanceWarning`` (block-manager fragmentation) is raised, and
* the batched output is column-for-column, dtype-for-dtype, value-for-value identical
  to the reference column-at-a-time implementation (captured expected).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mlframe.feature_engineering.basic import (
    create_date_features,
    add_cyclical_date_features,
    _DEFAULT_DATE_METHODS,
    _DEFAULT_CYCLICAL_PERIODS,
    _resolve_pandas_method,
    _cyclical_sincos_njit,
)


def _make_wide_df(n: int = 500, n_date_cols: int = 6, seed: int = 0) -> pd.DataFrame:
    """Helper: Make wide df."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2020-01-01")
    data = {}
    for i in range(n_date_cols):
        secs = rng.integers(0, 5 * 365 * 24 * 3600, size=n)
        data[f"dt{i}"] = base + pd.to_timedelta(secs, unit="s")
    data["x"] = rng.standard_normal(n)
    data["y"] = rng.integers(0, 100, size=n)
    return pd.DataFrame(data)


def _reference_loop(df, cols, methods, periods, add_cyclical):
    """Column-at-a-time reference (the pre-fix behaviour) = captured expected."""
    two_pi = float(2.0 * np.pi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        df = df.copy(deep=False)
        precomputed = {}
        for col in cols:
            obj = df[col].dt
            for method, dtype in methods.items():
                field = _resolve_pandas_method(obj, method, dtype)
                df[col + "_" + method] = field
                if add_cyclical and method != "is_weekend":
                    precomputed[(col, method)] = field.to_numpy()
        if add_cyclical:
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


def test_create_date_features_batched_identical_and_no_fragmentation():
    """Create date features batched identical and no fragmentation."""
    df = _make_wide_df()
    cols = [c for c in df.columns if c.startswith("dt")]
    methods = dict(_DEFAULT_DATE_METHODS)
    periods = _DEFAULT_CYCLICAL_PERIODS

    expected = _reference_loop(df, cols, methods, periods, add_cyclical=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=pd.errors.PerformanceWarning)
        got = create_date_features(
            df,
            cols=cols,
            methods=methods,
            add_cyclical=True,
            cyclical_periods=periods,
            delete_original_cols=False,
        )

    # column order, dtypes and values must all match the reference exactly
    pd.testing.assert_frame_equal(got, expected)


def test_add_cyclical_date_features_batched_identical_and_no_fragmentation():
    """Add cyclical date features batched identical and no fragmentation."""
    df = _make_wide_df()
    cols = [c for c in df.columns if c.startswith("dt")]
    periods = _DEFAULT_CYCLICAL_PERIODS

    expected = _reference_loop(df, cols, {}, periods, add_cyclical=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=pd.errors.PerformanceWarning)
        got = add_cyclical_date_features(df, cols=cols, periods=periods)

    pd.testing.assert_frame_equal(got, expected)


def test_batched_preserves_exact_column_order():
    """Batched preserves exact column order."""
    df = _make_wide_df()
    cols = [c for c in df.columns if c.startswith("dt")]
    got = create_date_features(
        df,
        cols=cols,
        add_cyclical=True,
        delete_original_cols=False,
    )
    # originals first (in input order), then each col's methods, then cyclical pairs
    assert list(got.columns[: df.shape[1]]) == list(df.columns)
    assert got.shape[1] > df.shape[1]
