"""The GPU batch MI executor redoes a chunk that does not fit in half-size pieces instead of failing the slice."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._fe_gpu_batch import _executor
from mlframe.feature_selection.filters.hermite_fe import shared


def _device_available():
    """True when a CUDA device answers."""
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _device_available(), reason="needs a CUDA device")
def test_out_of_memory_chunk_is_halved_and_gives_the_unchunked_result(monkeypatch):
    """A kernel that runs out of memory above 6 columns is retried on halves; the per-column MI equals the run without the limit."""
    rng = np.random.default_rng(0)
    n, k = 4000, 24
    X = rng.normal(size=(n, k))
    y = (X[:, 3] + 0.5 * rng.normal(size=n) > 0).astype(np.int64)
    ref = _executor.gpu_fe_batch_mi(X, y, 10)
    real = shared.plugin_mi_classif_batch_cuda_resident
    calls = []

    def limited(Xg, *a, **kw):
        """Fail like a full device for wide blocks."""
        calls.append(int(Xg.shape[1]))
        if Xg.shape[1] > 6:
            raise cp.cuda.memory.OutOfMemoryError(1 << 30, 1 << 20)
        return real(Xg, *a, **kw)

    monkeypatch.setattr(shared, "plugin_mi_classif_batch_cuda_resident", limited)
    got = _executor.gpu_fe_batch_mi(X, y, 10)
    assert np.allclose(got, ref, rtol=0, atol=1e-12)
    assert max(c for c in calls if c <= 6) <= 6 and any(c > 6 for c in calls)
