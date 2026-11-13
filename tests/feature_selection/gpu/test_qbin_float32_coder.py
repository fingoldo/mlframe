"""A float32 column is quantile-binned without widening: the 32-bit key coder gives the float64 route's codes."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from mlframe.feature_selection.filters import _fe_batched_qbin as Q

pytestmark = pytest.mark.gpu


def _frame(n=6000, k=40, seed=0):
    """float32 columns with heavy ties, binary and constant columns and signed zeros."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k)).astype(np.float32)
    x[:, :10] = np.round(x[:, :10] * 2) / 2
    x[:, 10:14] = (x[:, 10:14] > 0).astype(np.float32)
    x[:, 14:17] = 1.0
    x[:, k - 2] = -0.0
    x[:, k - 1] = np.where(x[:, k - 1] > 0, 0.0, -0.0)
    return x


def _bin(x, nbins, double):
    """Bin ``x``; with ``double`` the float32 values are widened first, which is the float64 route."""
    if double:
        x = cp.asarray(x).astype(cp.float64)
    return Q.batched_quantile_bin_gpu(x, nbins)


@pytest.mark.parametrize("nbins", [4, 10, 16])
def test_float32_codes_equal_the_float64_route(nbins):
    """Same equi-frequency codes, column for column, whether the float32 values are binned as float32 or widened to float64."""
    x = cp.asarray(_frame())
    single = _bin(x, nbins, double=False)
    double = _bin(x, nbins, double=True)
    assert single.dtype == double.dtype == cp.int64
    assert bool((single == double).all())


def test_host_float32_input_takes_the_single_route_and_matches():
    """A host float32 array is uploaded as float32 and binned to the same codes."""
    x = _frame(seed=3)
    assert bool((_bin(x, 10, double=False) == _bin(x, 10, double=True)).all())


def test_float64_input_keeps_the_float64_route():
    """A float64 column is binned in float64 (its codes can differ from its float32 narrowing, which must not be applied)."""
    rng = np.random.default_rng(5)
    x = cp.asarray(rng.standard_normal((4000, 6)))
    x[:, 0] = cp.round(x[:, 0] * 1e9) / 1e9
    reference = Q.batched_quantile_bin_gpu(x, 10)
    assert reference.dtype == cp.int64 and reference.shape == x.shape
    assert bool((reference == Q.batched_quantile_bin_gpu(x.copy(), 10)).all())


def test_edge_rounding_keeps_values_between_a_double_edge_and_its_float32_neighbour():
    """An edge that is not a float32 (an interpolated quantile) is rounded up, so a value equal to the float32 just below it stays on the lower side as in float64."""
    x = cp.asarray(np.array([[0.0], [1.0], [1.0000001], [2.0], [3.0], [4.0]], dtype=np.float32))
    assert bool((_bin(x, 3, double=False) == _bin(x, 3, double=True)).all())


def test_both_coder_variants_compile():
    """The shared source compiles for the double and the float column type."""
    for real in ("double", "float"):
        assert Q._get_qbin_coder_kernel(real) is not None


@pytest.mark.parametrize("nbins", [4, 10, 16])
def test_byte_codes_equal_the_int64_codes_and_feed_the_cmi_unchanged(nbins):
    """``byte_codes=True`` returns uint8 codes equal to the int64 ones, and the batched CMI over them (marginal and conditional) gives the same values."""
    from mlframe.feature_selection.filters._fe_batched_mi import batched_cmi_gpu

    x = cp.asarray(_frame(seed=7))
    wide = Q.batched_quantile_bin_gpu(x, nbins)
    byte = Q.batched_quantile_bin_gpu(x, nbins, byte_codes=True)
    assert byte.dtype == cp.uint8 and wide.dtype == cp.int64
    assert bool((byte.astype(cp.int64) == wide).all())
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, x.shape[0]).astype(np.int64)
    z = cp.asarray(rng.integers(0, 4, x.shape[0]).astype(np.int64))
    for cond in (None, z):
        assert np.array_equal(np.asarray(batched_cmi_gpu(byte, y, cond, kx=nbins, ky=3)), np.asarray(batched_cmi_gpu(wide, y, cond, kx=nbins, ky=3)))


def test_byte_codes_are_ignored_beyond_255_bins():
    """More than 255 bins cannot be held in a byte: the request is ignored and int64 codes come back."""
    x = cp.asarray(np.random.default_rng(0).standard_normal((3000, 3)).astype(np.float32))
    assert Q.batched_quantile_bin_gpu(x, 300, byte_codes=True).dtype == cp.int64
