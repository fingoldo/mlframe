"""Regression: cusum_features njit walk is bit-identical to the prior Python loop.

Pins that ``_cusum_walk_njit`` produces exactly the same float64 output as the
original scalar Python Page-Hinkley walk (branch-dependent pos/neg resets) across
scalar, NaN-containing, grouped, explicit-threshold and drift paths. ``seg_mean``
is precomputed via ``np.nanmean`` in the wrapper so the reduction order matches,
giving exact (max_abs == 0) identity, not just ~1e-9.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.grouped import iter_group_segments
from mlframe.feature_engineering.stationarity import cusum_features

KEYS = ("cusum_pos", "cusum_neg", "rows_since_reset", "n_resets_in_window")


def _old_cusum(values, threshold=None, *, group_ids=None, drift=0.0):
    """Verbatim PRE-optimization pure-Python cusum_features."""
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if threshold is None:
        finite = arr[np.isfinite(arr)]
        if finite.size < 2:
            threshold = 1.0
        else:
            mad = float(np.median(np.abs(finite - np.median(finite))))
            threshold = 5.0 * mad * 1.4826 if mad > 0 else 1.0

    out_pos = np.zeros(n)
    out_neg = np.zeros(n)
    out_since = np.zeros(n)
    out_count = np.zeros(n)

    def _walk(idx_seg):
        """Helper: Walk."""
        m = idx_seg.size
        if m == 0:
            return
        seg = arr[idx_seg]
        seg_mean = float(np.nanmean(seg)) if np.isfinite(seg).any() else 0.0
        pos = neg = rows_since = 0.0
        n_resets = 0
        for i in range(m):
            x = seg[i]
            if not np.isfinite(x):
                out_pos[idx_seg[i]] = pos
                out_neg[idx_seg[i]] = neg
                out_since[idx_seg[i]] = rows_since
                out_count[idx_seg[i]] = n_resets
                rows_since += 1
                continue
            dev = x - seg_mean
            pos = max(0.0, pos + dev - drift)
            neg = min(0.0, neg + dev + drift)
            if (pos > threshold) or (neg < -threshold):
                pos = neg = rows_since = 0.0
                n_resets += 1
            else:
                rows_since += 1
            out_pos[idx_seg[i]] = pos
            out_neg[idx_seg[i]] = neg
            out_since[idx_seg[i]] = rows_since
            out_count[idx_seg[i]] = n_resets

    if group_ids is None:
        _walk(np.arange(n))
    else:
        sort_idx, starts, ends = iter_group_segments(group_ids)
        for s, e in zip(starts, ends):
            _walk(sort_idx[s:e])

    return {"cusum_pos": out_pos, "cusum_neg": out_neg, "rows_since_reset": out_since, "n_resets_in_window": out_count}


def _assert_identical(new, old):
    """Helper: Assert identical."""
    for k in KEYS:
        assert np.array_equal(new[k], old[k], equal_nan=True), k


@pytest.mark.parametrize("n", [1, 3, 100, 5000])
def test_cusum_scalar_bit_identical(n):
    """Cusum scalar bit identical."""
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n).cumsum()
    _assert_identical(cusum_features(x), _old_cusum(x))


def test_cusum_with_nans_bit_identical():
    """Cusum with nans bit identical."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(800).cumsum()
    x[::13] = np.nan
    _assert_identical(cusum_features(x), _old_cusum(x))


def test_cusum_explicit_threshold_and_drift_bit_identical():
    """Cusum explicit threshold and drift bit identical."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(1500).cumsum()
    _assert_identical(
        cusum_features(x, threshold=2.5, drift=0.3),
        _old_cusum(x, threshold=2.5, drift=0.3),
    )


def test_cusum_all_nan_segment_bit_identical():
    """Cusum all nan segment bit identical."""
    x = np.full(50, np.nan)
    _assert_identical(cusum_features(x), _old_cusum(x))


def test_cusum_grouped_bit_identical():
    """Cusum grouped bit identical."""
    rng = np.random.default_rng(4)
    n = 900
    x = rng.standard_normal(n).cumsum()
    x[::29] = np.nan
    groups = rng.integers(0, 4, size=n)
    _assert_identical(
        cusum_features(x, group_ids=groups),
        _old_cusum(x, group_ids=groups),
    )
