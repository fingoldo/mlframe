"""The one-launch external-validation candidate kernel reproduces the host njit materialisation cell for cell, NaN, inf and zeros included."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_materialise import _materialise_extval_njit
from mlframe.feature_selection.filters._gpu_resident_extval_fused import fused_extval_candidates


def _device_available():
    """True when a CUDA device answers."""
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _device_available(), reason="needs a CUDA device")
def test_fused_candidates_match_the_host_materialisation():
    """Every (external factor, operand) column, with zeros, NaN and infinities in both operands, equals ``_materialise_extval_njit``."""
    rng = np.random.default_rng(0)
    n, n_ext = 3000, 4
    pa = rng.normal(size=n)
    pa[::97] = 0.0
    pa[5], pa[7] = np.nan, np.inf
    ext = rng.normal(size=(n, n_ext))
    ext[::89, 0] = 0.0
    ext[11, 0], ext[3, n_ext - 1] = -np.inf, np.nan
    ops = np.arange(9, dtype=np.int64)
    host = np.empty((n, n_ext * ops.size), dtype=np.float64)
    _materialise_extval_njit(pa, ext, ops, host)
    dev = fused_extval_candidates(cp, cp.asarray(pa), cp.asarray(np.ascontiguousarray(ext.T)), cp.asarray(ops, dtype=cp.int32)).get()
    assert np.array_equal(dev, host, equal_nan=True)


@pytest.mark.skipif(not _device_available(), reason="needs a CUDA device")
def test_float32_candidates_follow_the_operand_type_and_match_the_host_to_float32_rounding():
    """float32 operands run the float kernel (no read past the buffer): the output is float32 and every cell equals the double host cell rounded to float32, NaN and inf in the same cells."""
    rng = np.random.default_rng(1)
    n, n_ext = 2000, 3
    pa = rng.normal(size=n).astype(np.float32)
    pa[::97] = 0.0
    pa[5], pa[7] = np.nan, np.inf
    ext = rng.normal(size=(n, n_ext)).astype(np.float32)
    ext[::89, 0] = 0.0
    ext[11, 0], ext[3, n_ext - 1] = -np.inf, np.nan
    ops = np.arange(9, dtype=np.int64)
    host = np.empty((n, n_ext * ops.size), dtype=np.float64)
    _materialise_extval_njit(pa.astype(np.float64), ext.astype(np.float64), ops, host)
    dev = fused_extval_candidates(cp, cp.asarray(pa), cp.asarray(np.ascontiguousarray(ext.T)), cp.asarray(ops, dtype=cp.int32))
    assert dev.dtype == cp.float32
    got = dev.get()
    want = host.astype(np.float32)
    assert np.array_equal(np.isnan(got), np.isnan(want)) and np.array_equal(np.isinf(got), np.isinf(want))
    finite = np.isfinite(want)
    np.testing.assert_allclose(got[finite], want[finite], rtol=1e-6, atol=0)
