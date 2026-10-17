"""The bit-set family null returns the maxima of the histogram reference, and observed MIs share its arithmetic."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _family_maxt_null_v2_njit as V
from mlframe.feature_selection.filters._family_maxt_gate import family_null_max, family_null_max_njit, mi_all_columns_njit


def _family(n, k, bins, classes, seed):
    """Random integer-coded family with uneven bin counts and a label vector."""
    rng = np.random.default_rng(seed)
    n_x = rng.integers(2, bins + 1, k).astype(np.int64)
    cols = np.stack([rng.integers(0, int(b), n) for b in n_x]).astype(np.int16)
    y = rng.integers(0, classes, n).astype(np.int64)
    return cols, n_x, y


@pytest.mark.parametrize("classes", [2, 3, 5])
@pytest.mark.parametrize("n", [999, 4096])
def test_bitset_null_matches_the_histogram_reference(n, classes):
    """Same per-permutation maxima as ``family_null_max_njit`` up to the summation order of the two formulas."""
    cols, n_x, y = _family(n, 17, 12, classes, seed=n + classes)
    rng = np.random.default_rng(1)
    perms = y[np.stack([rng.permutation(n) for _ in range(40)])]
    ref = family_null_max_njit(cols, n_x, perms, classes)
    got = V.null_max_bitsets(V.bitset_groups(cols, n_x), n_x, perms, classes)
    assert np.allclose(ref, got, rtol=0, atol=1e-12)


def test_family_null_max_uses_the_bitset_path_and_keeps_the_permutation_stream():
    """``family_null_max`` draws the same permutations whichever kernel counts them, and is invariant to the column grouping budget."""
    cols, n_x, y = _family(2000, 30, 10, 2, seed=3)
    a = family_null_max(cols, n_x, y, 2, 70, np.random.default_rng(5), block=32)
    old = V.BITSET_BUDGET_BYTES
    try:
        V.BITSET_BUDGET_BYTES = 10 * 32 * 8  # a few columns per group
        b = family_null_max(cols, n_x, y, 2, 70, np.random.default_rng(5), block=32)
    finally:
        V.BITSET_BUDGET_BYTES = old
    assert np.array_equal(a, b)
    rng = np.random.default_rng(5)
    ref = np.concatenate([family_null_max_njit(cols, n_x, y[np.stack([rng.permutation(len(y)) for _ in range(m)])], 2) for m in (32, 32, 6)])
    assert np.allclose(a, ref, rtol=0, atol=1e-12)


def test_observed_and_null_values_of_the_same_labels_are_bit_equal():
    """A permutation that returns the observed labels gives exactly the observed MI, so exact ties are decided consistently."""
    cols, n_x, y = _family(1500, 12, 9, 3, seed=8)
    observed = mi_all_columns_njit(cols, n_x, y, 3)
    null = V.null_max_bitsets(V.bitset_groups(cols, n_x), n_x, y[None, :], 3)
    assert null[0] == observed.max()


def test_device_bitset_null_matches_the_host_bitset_null():
    """The CUDA bit-set kernel returns the host maxima (skipped without a device)."""
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device")
    except Exception:
        pytest.skip("no CUDA device")
    from mlframe.feature_selection.filters._family_maxt_cuda_bits import DeviceBitsets, null_max_bits_device
    from mlframe.feature_selection.filters._family_maxt_gate import _label_const
    from mlframe.feature_selection.filters._family_maxt_null_v2_njit import label_bitsets_batch

    for classes in (2, 5, 10):
        cols, n_x, y = _family(3000, 25, 11, classes, seed=classes)
        rng = np.random.default_rng(2)
        perms = y[np.stack([rng.permutation(len(y)) for _ in range(20)])]
        prep = V.bitset_groups(cols, n_x)
        host = V.null_max_bitsets(prep, n_x, perms, classes)
        dev = null_max_bits_device(DeviceBitsets(cp, prep, n_x, len(y)), label_bitsets_batch(perms, classes, (len(y) + 63) // 64), _label_const(prep[0], y, classes))
        assert np.allclose(host, dev, rtol=0, atol=1e-12)


def test_continuous_target_is_binned_before_the_family_test():
    """A float target with a distinct value per row is quantile-binned: a candidate that tracks it passes, noise does not."""
    from mlframe.feature_selection.filters._family_maxt_gate import family_maxt_pass_codes

    rng = np.random.default_rng(0)
    n = 4000
    signal = rng.normal(size=n)
    y = signal + 0.5 * rng.normal(size=n)
    cols = np.stack([np.digitize(signal, np.quantile(signal, np.linspace(0, 1, 11)[1:-1])), rng.integers(0, 10, n)]).astype(np.int16)
    flags = family_maxt_pass_codes(cols, np.array([10, 10], dtype=np.int64), y, n_perm=999)
    assert flags.tolist() == [True, False]
