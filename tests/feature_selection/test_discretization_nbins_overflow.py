"""Regression: discretiser ordinal codes must not wrap negative when n_bins > int8 max.

Pre-fix, the default ``dtype=np.int8`` discretiser paths cast bin codes to int8 BEFORE
(or without) clipping, so any ``n_bins > 128`` wrapped the top bins to negative via int8
modulo. ``discretize_uniform`` then clip-clamped those negatives to bin 0 (collapsing the
high-value region); the quantile ``searchsorted`` path had no clip at all (raw negatives).
Both silently mis-binned high-magnitude data, poisoning every downstream MI / SU / MRMR score.

Fix: ``_safe_code_dtype`` widens the requested int dtype to fit ``n_bins-1`` at every public
entry, and ``discretize_uniform`` clips in the float domain before the cast.
"""
import numpy as np
import pytest

from mlframe.feature_selection.filters.discretization import (
    discretize_array,
    discretize_2d_array,
    discretize_2d_quantile_batch,
)


def _monotonic(codes):
    return bool(np.all(np.diff(codes.astype(np.int64)) >= 0))


@pytest.mark.parametrize("method", ["quantile", "uniform"])
def test_discretize_array_nbins_over_int8_no_wrap(method):
    # Sorted high-magnitude ramp: codes MUST be non-decreasing and span the full range.
    arr = np.linspace(1000.0, 2000.0, 1000)
    out = discretize_array(arr, n_bins=200, method=method)
    assert out.min() >= 0, f"{method}: negative code -> int8 wraparound regressed"
    assert out.max() == 199, f"{method}: top bin lost (collapsed by overflow); got max {out.max()}"
    assert _monotonic(out), f"{method}: non-monotonic codes from overflow wraparound"
    assert np.iinfo(out.dtype).max >= 199, "dtype not auto-widened to hold n_bins-1"


def test_discretize_2d_array_nbins_over_int8_no_wrap():
    arr = np.linspace(0.0, 1000.0, 800)
    a2 = np.column_stack([arr, arr])
    out = discretize_2d_array(a2, n_bins=200, method="quantile", prefer_gpu=False)
    assert out.min() >= 0
    assert out.max() == 199
    assert _monotonic(out[:, 0])


def test_discretize_2d_quantile_batch_nbins_over_int8_no_wrap():
    arr = np.linspace(0.0, 1000.0, 800).astype(np.float32)
    a2 = np.column_stack([arr, arr])
    out = discretize_2d_quantile_batch(a2, n_bins=200)
    assert out.min() >= 0
    assert out.max() == 199
    assert _monotonic(out[:, 0])


def test_default_nbins_path_unchanged_int8():
    # The common n_bins=10 path must stay int8 and bit-identical to the numpy reference.
    arr = np.linspace(0.0, 1000.0, 1000)
    ref = np.searchsorted(
        np.nanpercentile(arr, np.linspace(0, 100, 11))[1:-1], arr, side="right"
    ).astype(np.int8)
    out = discretize_array(arr, n_bins=10, method="quantile")
    assert out.dtype == np.int8
    assert np.array_equal(out, ref)
