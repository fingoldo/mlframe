"""The quantile binning streams the columns in blocks when the whole working set does not fit, and has a host twin with the same codes."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from mlframe.feature_selection.filters import _fe_batched_qbin as Q
from mlframe.feature_selection.filters import _fe_gpu_vram

pytestmark = pytest.mark.gpu


def _frame(n=4000, k=30, seed=0):
    """float32 columns with ties, binary and constant columns."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k)).astype(np.float32)
    x[:, :6] = np.round(x[:, :6] * 2) / 2
    x[:, 6:9] = (x[:, 6:9] > 0).astype(np.float32)
    x[:, 9] = 1.0
    return x


def test_block_width_halves_until_the_cushion_admits(monkeypatch):
    """The width is the largest halving of K whose working set the cushion accepts, and 1 when nothing but a single column fits."""
    n, itemsize = 1000, 4
    per_col = n * (2 * itemsize + 8)
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda need, **k: need <= 5 * per_col)
    assert Q._column_block_width(cp, n, 32, itemsize) == 4
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda need, **k: False)
    assert Q._column_block_width(cp, n, 32, itemsize) == 1


@pytest.mark.parametrize("nbins", [4, 10])
def test_blocked_codes_equal_the_single_pass(monkeypatch, nbins):
    """With a cushion that admits 5 columns at a time the codes equal the unblocked result."""
    x = cp.asarray(_frame())
    whole = Q.batched_quantile_bin_gpu(x, nbins)
    per_col = x.shape[0] * (2 * 4 + 8)
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda need, **k: need <= 5 * per_col)
    blocked = Q.batched_quantile_bin_gpu(x, nbins)
    assert bool((whole == blocked).all())


@pytest.mark.parametrize("nbins", [4, 10, 16])
def test_host_twin_equals_the_device_codes(nbins):
    """The ``@njit`` host kernel gives the device coder's codes, column for column."""
    x = _frame(seed=2)
    assert np.array_equal(Q.quantile_bin_host(x, nbins), cp.asnumpy(Q.batched_quantile_bin_gpu(cp.asarray(x), nbins)))
