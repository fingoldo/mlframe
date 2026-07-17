"""Bit-identity of the fused njit integer-lattice block vs the numpy per-op form.

``_lattice_columns_for_pair`` fast-paths the default (gcd, lcm, bitwise_and) op set through a fused njit kernel;
it must be byte-identical (incl NaN on non-finite replay-drift rows) to the numpy loop, and any other op subset
must still take the exact numpy path.
"""

import numpy as np

from mlframe.feature_selection.filters._integer_lattice_fe import (
    INTEGER_LATTICE_OPS,
    _lattice_columns_for_pair,
    _lattice_gcd_lcm_and_njit,
)


def _numpy_ref(a, b, ops):
    from mlframe.feature_selection.filters._integer_lattice_fe import _nonfinite_mask, _to_int

    bad = _nonfinite_mask(a, b)
    ai, bi = _to_int(a), _to_int(b)
    g = np.gcd(ai, bi) if ("gcd" in ops or "lcm" in ops) else None
    mat = np.empty((a.shape[0], len(ops)), dtype=np.float64)
    for j, op in enumerate(ops):
        if op == "gcd":
            mat[:, j] = g.astype(np.float64)
        elif op == "lcm":
            safe_g = np.where(g == 0, 1, g)
            mat[:, j] = (np.abs(ai.astype(np.float64)) * np.abs(bi.astype(np.float64))) / safe_g
        elif op == "bitwise_and":
            mat[:, j] = np.bitwise_and(ai, bi).astype(np.float64)
    if bad.any():
        mat[bad, :] = np.nan
    return mat


def test_njit_default_ops_bit_identical_incl_nan():
    rng = np.random.default_rng(0)
    for n in (500, 2000, 20000):
        a = rng.integers(0, 5000, n).astype(np.float64)
        b = rng.integers(1, 5000, n).astype(np.float64)
        a[::37] = np.nan  # replay-drift non-finite rows
        b[::53] = np.inf
        got = _lattice_columns_for_pair(a, b, INTEGER_LATTICE_OPS)
        ref = _numpy_ref(a, b, INTEGER_LATTICE_OPS)
        assert np.array_equal(got, ref, equal_nan=True), n


def test_non_default_ops_take_numpy_path():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 100, 1000).astype(np.float64)
    b = rng.integers(1, 100, 1000).astype(np.float64)
    for ops in (("gcd",), ("bitwise_and", "gcd"), ("lcm",)):
        got = _lattice_columns_for_pair(a, b, ops)
        ref = _numpy_ref(a, b, ops)
        assert np.array_equal(got, ref, equal_nan=True), ops


def test_kernel_gcd_lcm_correct():
    a = np.array([12.0, 0.0, 7.0], dtype=np.float64)
    b = np.array([18.0, 5.0, 1.0], dtype=np.float64)
    out = _lattice_gcd_lcm_and_njit(a, b)
    assert out[0, 0] == 6.0 and out[0, 1] == 36.0 and out[0, 2] == float(12 & 18)  # gcd(12,18)=6, lcm=36
    assert out[1, 0] == 5.0 and out[1, 1] == 0.0  # gcd(0,5)=5, lcm(0,5)=0
