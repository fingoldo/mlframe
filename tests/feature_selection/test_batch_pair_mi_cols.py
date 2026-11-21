"""The contiguous-column CPU pair MI equals the strided reference kernel bit for bit."""

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory import batch_pair_mi_prange, batch_pair_mi_prange_cols


def _case(n, p, n_classes, seed, dtype):
    """Codes of mixed cardinality, a target, and random pairs (some repeated columns)."""
    rng = np.random.default_rng(seed)
    nb = rng.integers(2, 12, size=p).astype(np.int64)
    fd = np.stack([rng.integers(0, nb[j], size=n) for j in range(p)], axis=1).astype(dtype)
    y = rng.integers(0, n_classes, size=n).astype(np.int32)
    freqs = np.bincount(y, minlength=n_classes) / n
    pa = rng.integers(0, p, size=300).astype(np.int64)
    pb = rng.integers(0, p, size=300).astype(np.int64)
    return fd, pa, pb, nb, y, freqs


@pytest.mark.parametrize("dtype", [np.int8, np.int32])
@pytest.mark.parametrize("n_classes", [2, 5])
def test_cols_kernel_equals_reference(dtype, n_classes):
    """Same floats for int8 and int32 codes, binary and multiclass targets."""
    args = _case(3000, 40, n_classes, 0, dtype)
    assert np.array_equal(batch_pair_mi_prange(*args), batch_pair_mi_prange_cols(*args))


def test_cols_kernel_handles_no_pairs_and_no_rows():
    """Empty inputs go to the reference and give its (empty / zero) answers."""
    fd, pa, pb, nb, y, freqs = _case(500, 10, 2, 1, np.int8)
    assert batch_pair_mi_prange_cols(fd, pa[:0], pb[:0], nb, y, freqs).shape == (0,)
    z = batch_pair_mi_prange_cols(fd[:0], pa, pb, nb, y[:0], freqs)
    assert np.array_equal(z, batch_pair_mi_prange(fd[:0], pa, pb, nb, y[:0], freqs))
