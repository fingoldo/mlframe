"""The feature-major byte-code pair-MI kernel equals the row-major kernel and declines shapes outside the byte layout."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._batch_pair_mi_cuda_fm import batch_pair_mi_cuda_feature_major, codes_fit_bytes
from mlframe.feature_selection.filters._batch_pair_mi_cuda_shared_fused import batch_pair_mi_cuda_shared_fused


def _problem(n=2003, k=9, seed=0, n_classes=3, nb=7):
    """Random codes (odd row count so the tail path runs), a class target correlated with two columns, and the all-pairs index arrays."""
    r = np.random.default_rng(seed)
    data = r.integers(0, nb, size=(n, k + 1)).astype(np.int32)
    y = np.clip(((data[:, 0] + data[:, 1]) // 3 + (r.random(n) < 0.2)).astype(np.int64), 0, n_classes - 1)
    data[:, k] = y
    nbins = np.full(k + 1, nb, dtype=np.int64)
    nbins[k] = n_classes
    freqs = np.bincount(y, minlength=n_classes).astype(np.float64) / n
    ia, ib = np.triu_indices(k, 1)
    return data, ia.astype(np.int64), ib.astype(np.int64), nbins, y, freqs


@pytest.mark.parametrize("n", [2003, 2000, 1])
def test_feature_major_equals_row_major(n):
    """Same MI per pair as the shared-fused row-major kernel (counts are exact; the reduction is the same code), for row counts with and without a word tail."""
    data, ia, ib, nbins, y, freqs = _problem(n=max(n, 4) if n != 1 else 1003)
    if n == 1:
        data, y = data[:3], y[:3]
        freqs = np.bincount(y, minlength=3).astype(np.float64) / 3
    fm = batch_pair_mi_cuda_feature_major(data, ia, ib, nbins, y, freqs)
    ref = batch_pair_mi_cuda_shared_fused(data, ia, ib, nbins, y, freqs)
    np.testing.assert_allclose(fm, ref, rtol=1e-12, atol=1e-14)


def test_declines_codes_beyond_a_byte():
    """A code above 255 does not fit the byte layout: the kernel raises so the caller keeps the row-major kernels."""
    data, ia, ib, nbins, y, freqs = _problem()
    data = data.copy()
    data[0, 0] = 300
    nbins = nbins.copy()
    nbins[0] = 301
    assert not codes_fit_bytes(data, y, nbins, freqs)
    with pytest.raises(ValueError):
        batch_pair_mi_cuda_feature_major(data, ia, ib, nbins, y, freqs)
