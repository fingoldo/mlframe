"""The tiled transpose kernels (32 x 8 threads over a 32 x 32 tile) return exactly the bytes of the plain transpose, for every dtype and for shapes that leave partial tiles."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _gpu_resident_select_kernels as K

_SHAPES = [(1, 1), (1, 70), (70, 1), (31, 33), (32, 32), (33, 31), (65, 129), (1000, 37), (37, 1000), (2049, 65)]


def _data(rng, shape, dtype):
    """Random values of ``dtype`` that include negatives and the type's extremes."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        a = rng.integers(info.min, int(info.max) + 1, shape, dtype=dtype)
        a.flat[0] = info.min
        a.flat[-1] = info.max
        return a
    return rng.standard_normal(shape).astype(dtype)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_rows_to_columns_equals_the_plain_transpose(shape, dtype):
    """(n, K) float matrix -> (K, n) copy, byte for byte."""
    x = _data(np.random.default_rng(sum(shape)), shape, dtype)
    got = cp.asnumpy(K._transpose_to_cm(cp.asarray(x)))
    assert got.shape == (shape[1], shape[0])
    np.testing.assert_array_equal(got, x.T)


@pytest.mark.parametrize("shape", _SHAPES)
def test_columns_back_to_rows_equals_the_plain_transpose(shape):
    """(K, n) float32 matrix -> (n, K) copy, byte for byte (the reverse kernel of the same tile)."""
    x = _data(np.random.default_rng(sum(shape) + 1), shape, np.float32)
    got = cp.asnumpy(K._transpose_cm_to_rm(cp.asarray(x)))
    np.testing.assert_array_equal(got, x.T)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", [np.int8, np.int16])
def test_narrow_integer_codes_equal_the_plain_transpose(shape, dtype):
    """int8 and int16 code matrices, including the type's extreme values."""
    x = _data(np.random.default_rng(sum(shape) + 2), shape, dtype)
    got = cp.asnumpy(K.transpose_codes_to_cm(cp.asarray(x)))
    np.testing.assert_array_equal(got, x.T)


def test_the_block_shape_launched_is_32_by_8():
    """The kernels are written for 8 thread rows per block; a launch with another shape would leave rows of the tile unwritten."""
    import inspect

    src = inspect.getsource(K)
    assert src.count("(32, 8), (") == 3 and "(32, 32), (" not in src
    assert "for (int j = 0; j < 32; j += 8)" in K._TRANSPOSE_F32_SRC
