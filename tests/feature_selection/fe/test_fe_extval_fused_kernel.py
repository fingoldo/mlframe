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
