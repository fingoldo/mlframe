"""Regression test: spectral rolling features must emit FINITE output for an all-NaN window.

The bug (fixed): six functions imputed non-finite window values via ``float(np.nanmean(seg) or 0)``. For an all-NaN window ``np.nanmean`` returns NaN, and ``NaN or 0``
evaluates to NaN (NaN is truthy in Python), so the impute fill was NaN, the FFT input was all-NaN, and the entire window's spectral output came out NaN. The fix uses an
explicit guard that falls back to 0.0 when the window has no finite values.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.spectral import (
    rolling_spectral_bandwidth,
    rolling_spectral_centroid,
    rolling_spectral_flatness,
    rolling_spectral_flux,
    rolling_spectral_rolloff,
    rolling_periodicity_score,
)


pytestmark = pytest.mark.fast

_FUNCS = [
    rolling_spectral_centroid,
    rolling_spectral_bandwidth,
    rolling_spectral_rolloff,
    rolling_spectral_flatness,
    rolling_spectral_flux,
    rolling_periodicity_score,
]


@pytest.mark.parametrize("func", _FUNCS, ids=lambda f: f.__name__)
def test_spectral_all_nan_group_yields_finite(func):
    # The impute (``seg_mean``) is computed over the WHOLE group; the bug triggers when the entire group is all-NaN so ``nanmean`` returns NaN and the pre-fix
    # ``NaN or 0`` kept NaN as the fill, making every window's FFT input all-NaN. Group 0 is fully NaN; group 1 is finite (so the function does real work too).
    """Spectral all nan group yields finite."""
    n, K = 40, 8
    group_ids = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    values = np.concatenate([np.full(n, np.nan), np.arange(n, dtype=np.float64)])

    out = np.asarray(func(values, group_ids, window_K=K))

    g0 = out[:n]
    # The all-NaN group must yield finite values at its computed (full-window) anchors, never NaN/inf propagated from the all-NaN FFT. fill_value=NaN warmup
    # positions (first K-1) are exempt, as is the first full-window anchor (index K-1) which `rolling_spectral_flux` legitimately leaves NaN (no previous window);
    # from index K onward every function anchors a fully-defined window and must be finite.
    anchors = g0[K:]
    assert not np.isinf(g0).any(), f"{func.__name__} produced inf on an all-NaN group."
    assert np.isfinite(anchors).all(), (
        f"{func.__name__} produced NaN for an all-NaN group: `NaN or 0` kept NaN as the impute fill, so the FFT input was all-NaN."
    )
