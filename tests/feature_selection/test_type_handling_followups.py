"""Follow-up type-handling fixes: (1) an all-categorical polars frame must not crash categorize_dataset; (2) abs_pearson
passes f32 arrays to the kernel WITHOUT a full-length f64 copy (bit-identical -- the kernel promotes to f64 internally).
"""

from __future__ import annotations

import numpy as np
import pytest


def test_polars_all_categorical_frame_does_not_crash():
    """Regression: a polars frame with ONLY categorical columns (0 numeric) made df.select([]).to_numpy() collapse to
    (0, 0) rows, so the numeric<->categorical np.append raised a row-count mismatch. Must now categorize cleanly, with
    the same shape pandas yields."""
    pl = pytest.importorskip("polars")
    from mlframe.feature_selection.filters.discretization import categorize_dataset

    df = pl.DataFrame({"c1": ["a", "b", "a", "c"], "c2": ["x", "y", "x", "y"]})
    data, cols, _nbins = categorize_dataset(df=df, method="quantile", n_bins=10)
    assert data.shape == (4, 2), "n rows x both categorical columns"
    assert set(cols) == {"c1", "c2"}
    data_pd, cols_pd, _ = categorize_dataset(df=df.to_pandas(), method="quantile", n_bins=10)
    assert data.shape == data_pd.shape and set(cols) == set(cols_pd), "polars must match pandas here"


def test_abs_pearson_f32_native_is_bit_identical_to_f64_copy():
    """abs_pearson now passes f32 arrays in their native dtype (skipping the old full-length f64 copy). The kernel
    promotes each value to f64 before the squares, so the result is BIT-IDENTICAL to the f64-copy path -- even on a
    large-mean column where an f32 sum-of-squares would catastrophically cancel."""
    from mlframe.feature_selection.filters._fe_usability_signal import _abs_pearson_njit, abs_pearson

    rng = np.random.default_rng(0)
    for scale, mean in [(1.0, 0.0), (1e3, 1e8), (1e5, 1e6)]:
        a = (rng.standard_t(3, size=50000) * scale + mean).astype(np.float32)
        b = (a * 0.5 + rng.normal(0, scale * 0.3, 50000)).astype(np.float32)
        a[::777] = np.nan  # mixed NaN
        got = abs_pearson(a, b)  # native f32 path (no f64 array copy)
        ref = float(_abs_pearson_njit(np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)))
        assert got == ref, f"f32-native must equal the f64-copy result (scale={scale} mean={mean}): {got} vs {ref}"
