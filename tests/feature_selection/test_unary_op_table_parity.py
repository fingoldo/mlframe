"""The shared unary op-code table covers codes 0 to 16 once each, and the numba ladder gives each high op its named function."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._numba_utils_usability import _apply_unary
from mlframe.feature_selection.filters._unary_op_table import UNARY_OP_CODES

_HIGH_REFERENCE = {
    "rint": np.rint,
    "qubed": lambda v: v * v * v,
    "invsquared": lambda v: 1.0 / (v * v),
    "invqubed": lambda v: 1.0 / (v * v * v),
    "cbrt": np.cbrt,
    "invcbrt": lambda v: np.power(v, -1.0 / 3.0),
    "invsqrt": lambda v: np.power(v, -0.5),
    "exp": np.exp,
}


def test_codes_are_the_contiguous_range_zero_to_sixteen():
    """Each code is used once, and the table covers every code the ladders dispatch on."""
    assert sorted(UNARY_OP_CODES.values()) == list(range(17))


@pytest.mark.parametrize("name", sorted(_HIGH_REFERENCE))
def test_numba_ladder_matches_the_named_function(name):
    """The numba ladder computes the function its table name says, on positive and negative inputs."""
    v = np.array([0.5, 1.7, 3.2, 9.0, -2.5, -0.75], dtype=np.float64)
    want = _HIGH_REFERENCE[name](v)
    got = np.array([_apply_unary(float(x), UNARY_OP_CODES[name], 0.0) for x in v])
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)


def test_cuda_stack_kernel_matches_the_numba_ladder_for_every_code():
    """The CUDA stack kernel and the numba ladder give the same value for each of the 17 table codes, including the smart log."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._unary_stack_cuda import unary_stack_cuda

    # no 0.0: the numba ladder raises ZeroDivisionError on 1/0 (Python error model) where the CUDA kernel returns inf, and the callers never pass a zero to it
    x = np.array([0.5, 1.7, 3.2, 9.0, -2.5, -0.75, 6.25], dtype=np.float64)
    xmin = float(x.min())
    codes = np.array(sorted(UNARY_OP_CODES.values()), dtype=np.int32)
    with np.errstate(all="ignore"):
        want = np.array([[_apply_unary(float(v), int(c), xmin) for v in x] for c in codes])
    try:
        got = unary_stack_cuda(cp.asarray(x), cp.asarray(codes), xmin)
    except Exception as exc:  # no usable device
        pytest.skip(f"no usable CUDA device: {exc!r}")
    if got is None:
        pytest.skip("unary_stack_cuda declined (device or memory not available)")
    np.testing.assert_allclose(cp.asnumpy(got), want, rtol=1e-12, atol=0.0, equal_nan=True)


def _operand_build_reference():
    """numpy reference for every op of the resident operand build's own table, by name."""
    return {
        "identity": lambda v: v, "neg": lambda v: -v, "abs": np.abs, "sqr": lambda v: v * v, "reciproc": lambda v: 1.0 / v,
        "sqrt": lambda v: np.sqrt(np.abs(v)), "sin": np.sin, "sign": np.sign, "rint": np.rint, "qubed": lambda v: v * v * v,
        "invsquared": lambda v: 1.0 / (v * v), "invqubed": lambda v: 1.0 / (v * v * v), "cbrt": np.cbrt,
        "invcbrt": lambda v: np.power(v, -1.0 / 3.0), "invsqrt": lambda v: np.power(v, -0.5), "exp": np.exp,
        "cos": np.cos, "tan": np.tan, "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan, "sinh": np.sinh, "cosh": np.cosh,
        "tanh": np.tanh, "arcsinh": np.arcsinh, "arccosh": np.arccosh, "arctanh": np.arctanh,
    }


@pytest.mark.parametrize("sample", [[-0.85, -0.5, -0.1, 0.2, 0.6, 0.9], [1.5, 2.0, 3.25, 4.0]])
def test_operand_build_kernel_matches_a_numpy_reference_for_every_op_of_its_table(sample):
    """Each op of ``_BATCH_UNARY_OPS`` runs through the real ``batch_unary`` kernel and equals its named numpy function (nan where the function is undefined)."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_resident_operand_build import _BATCH_UNARY_OPS, _get_batch_unary_kernel

    ref = _operand_build_reference()
    assert set(ref) == set(_BATCH_UNARY_OPS), "the table and the reference must name the same ops"
    x = np.array(sample, dtype=np.float64)
    n = x.size
    names = sorted(_BATCH_UNARY_OPS)
    ncols = len(names)
    try:
        kernel = _get_batch_unary_kernel(np.float64)
        G = cp.asarray(x.reshape(n, 1))
        slot = cp.zeros(ncols, dtype=cp.int64)
        opc = cp.asarray(np.array([_BATCH_UNARY_OPS[nm] for nm in names], dtype=np.int32))
        out_col = cp.arange(ncols, dtype=cp.int64)
        out = cp.zeros((n, ncols), dtype=cp.float32)
        kernel(((n * ncols + 127) // 128,), (128,), (G, slot, opc, out_col, np.int64(n), np.int32(1), np.int32(ncols), np.int32(ncols), out))
        got = cp.asnumpy(out)
    except Exception as exc:  # no usable device
        pytest.skip(f"no usable CUDA device: {exc!r}")
    with np.errstate(all="ignore"):
        for j, nm in enumerate(names):
            want = np.asarray(ref[nm](x), dtype=np.float64).astype(np.float32)
            np.testing.assert_allclose(got[:, j], want, rtol=2e-5, atol=1e-6, equal_nan=True, err_msg=f"op {nm!r} (code {_BATCH_UNARY_OPS[nm]})")
