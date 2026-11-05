"""The batched plain-unary kernel computes each operand group in the group's own element type and agrees with numpy applied in that type.

The kernel used to declare ``const double* G`` and ran for the float64 group only (a float32 group took the per-column path); it is now compiled per element type, so a float32 group is
batched too and never meets a double pointer.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._gpu_resident_operand_build import _BATCH_UNARY_OPS, _get_batch_unary_kernel

_REFERENCE = {
    "identity": lambda x: x,
    "neg": lambda x: -x,
    "abs": np.abs,
    "sqr": lambda x: x * x,
    "reciproc": lambda x: x ** x.dtype.type(-1),
    "sqrt": lambda x: np.sqrt(np.abs(x)),
    "sin": np.sin,
    "rint": np.rint,
    "qubed": lambda x: x * x * x,
    "cbrt": np.cbrt,
    "exp": np.exp,
    "cos": np.cos,
    "tan": np.tan,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "arcsinh": np.arcsinh,
}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_each_group_type_matches_numpy_in_that_type(dtype):
    """Every op of the group, on a block of values in [-2, 2] (zero included), equals numpy in the group's type to the float32 table's precision."""
    rng = np.random.default_rng(0)
    n, m = 4000, 3
    host = rng.uniform(-2.0, 2.0, size=(n, m)).astype(dtype)
    host[::50, 0] = 0.0
    names = [k for k in _REFERENCE if k in _BATCH_UNARY_OPS]
    slots = np.array([i % m for i in range(len(names))], dtype=np.int64)
    opcs = np.array([_BATCH_UNARY_OPS[k] for k in names], dtype=np.int32)
    out_cols = np.arange(len(names), dtype=np.int64)
    out = cp.zeros((n, len(names)), dtype=cp.float32)
    G = cp.ascontiguousarray(cp.asarray(host))
    kernel = _get_batch_unary_kernel(dtype)
    total = n * len(names)
    kernel(((total + 255) // 256,), (256,), (G, cp.asarray(slots), cp.asarray(opcs), cp.asarray(out_cols), np.int64(n), np.int32(m), np.int32(len(names)), np.int32(len(names)), out))
    got = cp.asnumpy(out)
    with np.errstate(all="ignore"):
        for j, name in enumerate(names):
            want = _REFERENCE[name](host[:, slots[j]]).astype(np.float32)
            finite = np.isfinite(want)
            assert np.array_equal(np.isfinite(got[:, j]), finite), name
            np.testing.assert_allclose(got[finite, j], want[finite], rtol=2e-6, atol=1e-6, err_msg=f"{name} ({np.dtype(dtype)})")


def test_the_kernel_is_compiled_once_per_element_type():
    """The same type returns the cached kernel; the two types are different kernels."""
    assert _get_batch_unary_kernel(np.float32) is _get_batch_unary_kernel(np.float32)
    assert _get_batch_unary_kernel(np.float32) is not _get_batch_unary_kernel(np.float64)
