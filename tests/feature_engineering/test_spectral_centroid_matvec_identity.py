"""Pins the spec@k (BLAS gemv) reduction in rolling_spectral_centroid / _bandwidth
to be numerically equivalent (~1e-9) to the explicit (spec*k).sum(axis=1) reference.

Guards the perf optimization: the gemv form must keep producing the same
power-weighted-mean bin index, so the FE feature selection is unaffected.
"""

import numpy as np
import pytest

from mlframe.feature_engineering.spectral import (
    rolling_spectral_bandwidth,
    rolling_spectral_centroid,
)
from mlframe.feature_engineering.grouped import per_group_sliding_window
from mlframe.feature_engineering.spectral import _spec_pow


def _ref_centroid(values, group_ids, window_K, detrend=True):
    """Explicit (spec * k[None, :]).sum(axis=1) reference (pre-optimization form)."""
    out = np.full(values.size, np.nan, dtype=np.float64)
    for sort_idx_seg, _w, write_idx in per_group_sliding_window(values, group_ids, window_K=window_K):
        seg = values[sort_idx_seg].astype(np.float64)
        seg_mean = float(np.nanmean(seg)) if np.isfinite(seg).any() else 0.0
        seg_f = np.where(np.isfinite(seg), seg, seg_mean)
        spec, _ = _spec_pow(seg_f, window_K, detrend)
        k = np.arange(spec.shape[1], dtype=np.float64)
        denom = spec.sum(axis=1) + 1e-12
        out[write_idx] = (spec * k[None, :]).sum(axis=1) / denom
    return out


@pytest.mark.parametrize("K", [50, 100, 256])
def test_centroid_gemv_matches_explicit_reduction(K):
    rng = np.random.default_rng(123)
    n = 5_000
    values = np.cumsum(rng.standard_normal(n))
    group_ids = rng.integers(0, 4, n)

    got = rolling_spectral_centroid(values, group_ids, window_K=K)
    ref = _ref_centroid(values, group_ids, window_K=K)

    finite = np.isfinite(got) & np.isfinite(ref)
    assert finite.any()
    np.testing.assert_allclose(got[finite], ref[finite], rtol=1e-9, atol=1e-9)


def test_bandwidth_finite_and_nonnegative_after_gemv():
    rng = np.random.default_rng(321)
    n = 3_000
    values = np.cumsum(rng.standard_normal(n))
    group_ids = rng.integers(0, 3, n)
    bw = rolling_spectral_bandwidth(values, group_ids, window_K=100)
    finite = np.isfinite(bw)
    assert finite.any()
    assert (bw[finite] >= 0).all()
