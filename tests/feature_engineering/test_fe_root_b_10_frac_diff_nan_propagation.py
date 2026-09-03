"""FE_ROOT_B-10: _frac_diff_single zero-filled any NaN in the input before convolving, so a NaN past
the first K rows silently corrupted every downstream window that touched it (substituting 0, not
propagating NaN)."""

from __future__ import annotations

import numpy as np

from mlframe.feature_engineering.stationarity import frac_diff


def test_mid_series_nan_propagates_through_the_next_k_minus_1_rows():
    """A single NaN at position p (p >= K) must NaN out rows [p, p+K-1] of the output, not silently
    substitute 0.0 for that value in every window that touches it."""
    rng = np.random.default_rng(0)
    n, K, p = 200, 20, 100
    values = np.cumsum(rng.normal(size=n))
    values[p] = np.nan

    out = frac_diff(values, d=0.5, K=K)

    # Rows before the NaN's window (unaffected) must stay finite.
    assert np.isfinite(out[K : p - K]).all(), "rows unaffected by the NaN window must remain finite"
    # Every row whose K-length causal window touches position p must be NaN.
    assert np.isnan(out[p : p + K]).all(), "every output row whose window touches the NaN input must be NaN"
    # Rows well past the NaN's influence must be finite again.
    assert np.isfinite(out[p + K + 5 :]).all(), "rows past the NaN's window influence must recover to finite"


def test_clean_series_still_all_finite_past_the_warmup():
    """Sanity: a NaN-free series is unaffected by the propagation fix (only the leading K rows are NaN)."""
    rng = np.random.default_rng(1)
    n, K = 100, 20
    values = np.cumsum(rng.normal(size=n))
    out = frac_diff(values, d=0.5, K=K)
    assert np.isnan(out[:K]).all()
    assert np.isfinite(out[K:]).all()
