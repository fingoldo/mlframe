"""The streaming GPU random-Fourier-feature matmul sizes its device and pinned staging buffers by the rows it actually has, not by the default batch size."""
from __future__ import annotations

import numpy as np
import pytest


def _cuda_ok() -> bool:
    """True when cupy sees a device with at least ~300 MB free."""
    try:
        import cupy as cp

        free, _total = cp.cuda.runtime.memGetInfo()
        return bool(free > 300 * 1024**2)
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _cuda_ok(), reason="no CUDA device with free VRAM")]


def _reference(X, W, b, scale):
    """NumPy RFF: ``scale * [cos(XW + b), sin(XW + b)]``."""
    ang = X @ W + b
    return scale * np.hstack([np.cos(ang), np.sin(ang)])


def test_staging_buffers_are_clamped_to_the_row_count(monkeypatch):
    """With 10 rows and the default 100_000-row batch, no staging buffer is allocated for more than 10 rows, and the result matches NumPy."""
    from mlframe.feature_engineering.transformer import _kernels_cupy as k

    shapes = []
    real = k._get_pinned_buffer

    def spy(name, shape, dtype):
        """Record the requested pinned-buffer shape."""
        shapes.append(tuple(shape))
        return real(name, shape, dtype)

    monkeypatch.setattr(k, "_get_pinned_buffer", spy)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4)).astype(np.float32)
    W = rng.normal(size=(4, 3)).astype(np.float32)
    b = rng.uniform(0, 6, 3).astype(np.float32)
    out = np.empty((10, 6), dtype=np.float32)
    k.rff_matmul_cupy(X, W, b, out, 0.5)
    assert shapes and all(s[0] <= 10 for s in shapes)
    np.testing.assert_allclose(out, _reference(X, W, b, 0.5), atol=1e-5)


@pytest.mark.parametrize("n,batch", [(7, 3), (8, 4), (1, 100)])
def test_multi_batch_result_matches_numpy(n, batch):
    """Row counts that are and are not a multiple of the batch size, and a single row, all reproduce the NumPy result."""
    from mlframe.feature_engineering.transformer import _kernels_cupy as k

    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    W = rng.normal(size=(5, 2)).astype(np.float32)
    b = rng.uniform(0, 6, 2).astype(np.float32)
    out = np.empty((n, 4), dtype=np.float32)
    k.rff_matmul_cupy(X, W, b, out, 1.0, batch_rows=batch)
    np.testing.assert_allclose(out, _reference(X, W, b, 1.0), atol=1e-5)


def test_empty_input_returns_without_launching():
    """Zero rows is a no-op instead of an IndexError on the empty batch list."""
    from mlframe.feature_engineering.transformer import _kernels_cupy as k

    out = np.empty((0, 4), dtype=np.float32)
    k.rff_matmul_cupy(np.empty((0, 5), dtype=np.float32), np.zeros((5, 2), dtype=np.float32), np.zeros(2, dtype=np.float32), out, 1.0)
    assert out.shape == (0, 4)


def test_failure_mid_loop_leaves_the_pipeline_usable():
    """An exception while the copy streams are busy is raised after they are drained, and the next call works."""
    from mlframe.feature_engineering.transformer import _kernels_cupy as k

    rng = np.random.default_rng(2)
    X = rng.normal(size=(8, 3)).astype(np.float32)
    W = rng.normal(size=(3, 2)).astype(np.float32)
    b = rng.uniform(0, 6, 2).astype(np.float32)
    read_only = np.empty((8, 4), dtype=np.float32)
    read_only.flags.writeable = False
    with pytest.raises(ValueError):
        k.rff_matmul_cupy(X, W, b, read_only, 1.0, batch_rows=4)
    out = np.empty((8, 4), dtype=np.float32)
    k.rff_matmul_cupy(X, W, b, out, 1.0, batch_rows=4)
    np.testing.assert_allclose(out, _reference(X, W, b, 1.0), atol=1e-5)
