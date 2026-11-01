"""The device CMI scorer bins and scores candidates in column blocks sized from free VRAM; a small block gives the same values as one block."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _device_ok() -> bool:
    """Whether a CUDA device can allocate."""
    try:
        cp.zeros(1)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _device_ok(), reason="no usable CUDA device")
def test_blocked_scoring_equals_single_block(monkeypatch):
    """Forcing a tiny free-memory reading makes the scorer use several blocks; the CMI values are unchanged."""
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as M
    from mlframe.feature_selection.filters._fe_batched_mi import batched_cmi_gpu, batched_quantile_bin_gpu

    rng = np.random.default_rng(0)
    n, k = 4000, 60
    X = rng.normal(size=(n, k)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=n) > 0).astype(np.int64)
    z = (X[:, 1] > 0).astype(np.int64)
    whole = M._binned_cmi_in_column_blocks(cp, X, 10, y, z, batched_cmi_gpu, batched_quantile_bin_gpu)
    real = cp.cuda.runtime.memGetInfo
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (n * 36 * 3 * 7, real()[1]))  # about 7 columns per block
    blocked = M._binned_cmi_in_column_blocks(cp, X, 10, y, z, batched_cmi_gpu, batched_quantile_bin_gpu)
    np.testing.assert_allclose(blocked, whole, rtol=0, atol=1e-12)
    assert blocked.shape == (k,)
