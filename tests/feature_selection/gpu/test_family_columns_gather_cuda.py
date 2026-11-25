"""The one-launch gathers of a family's code columns (``_family_columns_gather_cuda``) equal numpy on every layout the family batches come in, around the tile boundaries."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _family_maxt_gate_device as device
from mlframe.feature_selection.filters._family_columns_gather_cuda import gathered_codes, transposed_codes

if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)


def _batches(n: int, widths: "list[int]", dtype, seed: int = 0, strided: bool = False):
    """``(device batches, host family)``: batches of the given widths; with ``strided`` each is a column slice of a wider matrix (a chunk view with a row stride larger than its width)."""
    rng = np.random.default_rng(seed)
    host, dev = [], []
    for w in widths:
        wide = rng.integers(0, 100, size=(n, w + (5 if strided else 0))).astype(dtype)
        piece = wide[:, 2 : 2 + w] if strided else wide
        host.append(np.ascontiguousarray(piece))
        dev.append(cp.asarray(wide)[:, 2 : 2 + w] if strided else cp.asarray(wide))
    return dev, np.concatenate(host, axis=1)


@pytest.mark.parametrize("n", [1, 31, 32, 33, 1009, 4097])
@pytest.mark.parametrize("widths", [[1], [1] * 31, [1] * 32, [1] * 33, [3, 1, 5, 2], [40, 1]])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32])
def test_the_transposed_family_equals_numpy(n, widths, dtype):
    """Every cell of the ``(columns, rows)`` matrix is the family cell, for batch widths below, at and above the 32-wide tile and row counts one below, at and above a tile."""
    dev, host = _batches(n, widths, dtype)
    assert device._multi_layout(cp, dev, n) is not None
    out = transposed_codes(cp, dev, n)
    assert out.shape == (host.shape[1], n) and out.flags.c_contiguous
    assert np.array_equal(cp.asnumpy(out), host.T)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("n", [31, 1009])
def test_the_gathered_columns_equal_numpy_in_the_order_asked(n, strided):
    """The chosen columns come out in the order given (repeats and a reversed order included), from batches that are slices of wider matrices too."""
    dev, host = _batches(n, [1] * 40 + [7, 3], np.int8, seed=3, strided=strided)
    assert device._multi_layout(cp, dev, n) is not None
    chosen = np.array([41, 0, 17, 17, 33, 40, 5], dtype=np.int64)
    out = gathered_codes(cp, dev, n, chosen)
    assert out.shape == (n, chosen.shape[0]) and out.flags.c_contiguous
    assert np.array_equal(cp.asnumpy(out), host[:, chosen])


def test_a_strided_family_transposes_like_a_contiguous_one():
    """Column slices of a wider matrix (row stride larger than the width) are read in place with their stride."""
    dev, host = _batches(1009, [1] * 36, np.int8, seed=1, strided=True)
    assert device._multi_layout(cp, dev, 1009) is not None
    assert np.array_equal(cp.asnumpy(transposed_codes(cp, dev, 1009)), host.T)
