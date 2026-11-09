"""The float32 usability-correlation default keeps the double kernel for inputs float32 cannot represent.

An offset of 1e8 next to a spread of 0.1 leaves float32 with a step of 8, so a linear pair reads as less than perfectly correlated and a
near-constant column reads as constant. Such inputs must not take the float32 path.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import batch_pair_usability_corr_gpu as bp


def test_offset_heavy_columns_are_not_float32_safe():
    """A column near 1e8 with a spread of 0.1 is flagged."""
    n = 1000
    y = np.arange(n, dtype=np.float64)
    x0 = 1e8 + 0.1 * np.arange(n)
    assert bp._float32_unsafe(y, x0.reshape(-1, 1)) is True


def test_well_scaled_columns_stay_on_float32():
    """Ordinary standard-normal columns keep the float32 path."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    X = rng.normal(size=(500, 3))
    assert bp._float32_unsafe(y, X) is False


def test_non_finite_values_take_the_double_path():
    """A NaN or inf in any input routes the call to the double kernel."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=100)
    X = rng.normal(size=(100, 2))
    X[3, 1] = np.nan
    assert bp._float32_unsafe(y, X) is True


def test_one_nan_operand_moves_only_its_own_pairs_to_the_double_kernel():
    """A NaN in one operand used to send every pair to the double kernel; now only the pairs on that operand run in double, the rest in float32, and every value matches the double reference."""
    pytest = __import__("pytest")
    if not bp._CUDA_AVAIL:
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(3)
    n, k = 4000, 12
    ops = rng.standard_normal((k, n)).astype(np.float32)
    ops[4, 7] = np.nan
    ops[9] = ops[9] * 0.01 + 5e4  # offset 5e4 against a spread of 0.01: unsafe for float32
    y = (ops[0] * ops[1] + rng.standard_normal(n)).astype(np.float32)
    pa, pb = np.triu_indices(k, 1)
    pa, pb = pa.astype(np.int64), pb.astype(np.int64)
    forms = bp.ALL_FORM_IDS
    y_unsafe, op_unsafe = bp._float32_unsafe_columns(y, ops)
    assert not y_unsafe and op_unsafe.tolist() == [False] * 4 + [True] + [False] * 4 + [True] + [False] * 2
    got = bp.batch_pair_usability_corr_cuda_warp(y, ops, pa, pb, forms)
    ref = bp.batch_pair_usability_corr_njit_parallel(y.astype(np.float64), ops.astype(np.float64), pa, pb, forms)
    risky = op_unsafe[pa] | op_unsafe[pb]
    np.testing.assert_allclose(got[risky], ref[risky], rtol=0, atol=1e-6)  # double kernel (summation order only)
    np.testing.assert_allclose(got[~risky], ref[~risky], rtol=0, atol=5e-4)  # float32 kernel
