"""GPU_INFRA_B-7 fix (mrmr_audit_2026-07-22): the coalesced tiled-transpose kernel family
(_transpose_to_cm/_transpose_cm_to_rm/transpose_codes_to_cm) had no isolated/direct unit test anywhere in
the suite -- coverage was indirect only, via whatever K/n shapes the canonical GPU fit / parity tests happen
to exercise end-to-end. This pins non-multiple-of-32 K/n shapes (plus K=1, n=1) directly against
cp.ascontiguousarray(x.T) for both directions and f32/f64/int8/int16 dtypes.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    cp.cuda.runtime.getDeviceCount()
    _HAVE_CUDA_DEVICE = True
except Exception:
    _HAVE_CUDA_DEVICE = False

pytestmark = pytest.mark.skipif(not _HAVE_CUDA_DEVICE, reason="no CUDA device")

from mlframe.feature_selection.filters._gpu_resident_select_kernels import (
    _transpose_cm_to_rm,
    _transpose_to_cm,
    transpose_codes_to_cm,
)

_SHAPES = [(1, 1), (33, 1), (1, 33), (37, 65), (1000, 7)]


@pytest.mark.parametrize("n,K", _SHAPES)
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_transpose_to_cm_matches_reference(n, K, dtype):
    """(n, K) -> (K, n) forward transpose matches cp.ascontiguousarray(x.T) for non-multiple-of-32 shapes."""
    rng = np.random.default_rng(0)
    x = cp.asarray(rng.normal(size=(n, K)).astype(dtype))
    got = _transpose_to_cm(x)
    expected = cp.ascontiguousarray(x.T)
    assert got.shape == (K, n)
    cp.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("K,n", _SHAPES)
def test_transpose_cm_to_rm_matches_reference(K, n):
    """(K, n) -> (n, K) inverse transpose matches cp.ascontiguousarray(x.T) for non-multiple-of-32 shapes."""
    rng = np.random.default_rng(0)
    x = cp.asarray(rng.normal(size=(K, n)).astype(cp.float32))
    got = _transpose_cm_to_rm(x)
    expected = cp.ascontiguousarray(x.T)
    assert got.shape == (n, K)
    cp.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("n,K", _SHAPES)
@pytest.mark.parametrize("dtype", [cp.int8, cp.int16])
def test_transpose_codes_to_cm_matches_reference(n, K, dtype):
    """Int-codes (n, K) -> (K, n) transpose matches cp.ascontiguousarray(x.T) for non-multiple-of-32 shapes."""
    rng = np.random.default_rng(0)
    iinfo = np.iinfo(dtype)
    x = cp.asarray(rng.integers(0, min(iinfo.max, 20), size=(n, K)).astype(dtype))
    got = transpose_codes_to_cm(x)
    expected = cp.ascontiguousarray(x.T)
    assert got.shape == (K, n)
    cp.testing.assert_array_equal(got, expected)
