"""When the int32 row-major upload does not fit VRAM, byte codes still go to the feature-major kernel (about 20x the row-chunked one), with the CPU kernel's values."""

import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("numba.cuda")

from mlframe.feature_selection.filters import batch_pair_mi_gpu as g


@pytest.mark.parametrize("force", ["cuda", None])
def test_rejected_int32_upload_uses_feature_major(monkeypatch, force):
    """The upload estimate says no; the backend tag stays ``cuda`` (feature-major), not ``cuda_row_chunked``, and the MI equals the CPU kernel's."""
    if not g._CUDA_AVAIL:
        pytest.skip("numba.cuda unavailable")
    monkeypatch.setattr(g, "_gpu_upload_fits", lambda *a, **k: False)
    monkeypatch.setattr(g, "_batch_pair_mi_backend_choice", lambda *a, **k: "cuda")
    rng = np.random.default_rng(1)
    n, p = 20000, 12
    X = rng.integers(0, 8, size=(n, p)).astype(np.int32)
    nb = np.full(p, 8, np.int32)
    y = rng.integers(0, 3, size=n).astype(np.int32)
    fy = np.bincount(y, minlength=3) / n
    ii, jj = np.triu_indices(p, 1)
    got, tag = g.dispatch_batch_pair_mi(X, ii.astype(np.int64), jj.astype(np.int64), nb, y, fy, force_backend=force)
    assert tag == "cuda"
    ref = g.batch_pair_mi_njit_prange(X, ii.astype(np.int64), jj.astype(np.int64), nb, y, fy)
    np.testing.assert_allclose(got, ref, atol=1e-9)
