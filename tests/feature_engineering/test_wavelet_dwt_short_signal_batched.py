"""wavedec_batched on signals shorter than the filter agrees with the per-signal path (tracker 2026-11-02 wave 11, NJ11-1)."""

import numpy as np
import pytest

from mlframe.feature_engineering import wavelet_dwt as w


@pytest.mark.skipif(not w._HAS_NUMBA, reason="needs numba")
@pytest.mark.parametrize("wavelet,T", [("db4", 4), ("db4", 3), ("db4", 6), ("coif3", 10)])
def test_batched_matches_single_for_signals_shorter_than_the_filter(wavelet, T):
    """T < filter_len - 1 made a level write past the scratch buffer; results were garbage or the process crashed."""
    rng = np.random.default_rng(0)
    sig = rng.normal(size=(8, T))
    approx, details, lengths = w.wavedec_batched(sig, wavelet, 3)
    for i in range(sig.shape[0]):
        coeffs = w.wavedec(sig[i], wavelet, 3)
        np.testing.assert_allclose(approx[i], coeffs[0], atol=1e-12)
        off = 0
        for lvl in range(3, 0, -1):
            n = int(lengths[4 - lvl])
            np.testing.assert_allclose(details[i, off : off + n], coeffs[lvl], atol=1e-12)
            off += n
