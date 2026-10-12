"""``compute_batch_aucs`` / ``compute_batch_rmse`` had NO fallback at all on a genuine GPU dispatch failure
(cupy OOM, driver fault): the auto-dispatch path called the GPU primitive directly inside the ``if
use_gpu:`` branch with no try/except, so any device-side exception propagated straight up and aborted the
whole caller (found live via a 2M-row fuzz-optimization profiling run: cupy's thrust-backed ``argsort``
raised a plain ``MemoryError`` under VRAM pressure and crashed the entire training suite).

Mirrors the sibling fix already shipped for ``_mi_classif_batch_numba``'s GPU dispatch (same bug class,
same project): a genuine device fault degrades to the CPU reference with a visible warning; an explicit
``force_backend='gpu'`` still raises, since that caller asked for GPU-or-error, not a silent downgrade.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import mlframe.metrics._gpu_metrics as gm
from mlframe.metrics.core import compute_batch_aucs, compute_batch_rmse


def _label_matrix(n: int = 200, k: int = 3, seed: int = 0):
    """A random (N, K) binary label matrix plus an (N, K) score matrix for the batch-AUC dispatchers."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=(n, k))
    scores = rng.random((n, k))
    return y, scores


@pytest.mark.parametrize("error", [MemoryError("bad allocation"), RuntimeError("device fault")], ids=["memory_error", "generic_device_error"])
def test_compute_batch_aucs_falls_back_to_cpu_on_gpu_dispatch_failure(error, monkeypatch):
    """A simulated GPU dispatch failure degrades to the CPU reference instead of propagating."""
    yt, ys = _label_matrix()
    # Ground truth computed BEFORE any patching: `_resolve_backend` is replaced wholesale below (ignoring
    # its `force` argument), so computing this afterwards would route even `force_backend="cpu"` into the
    # (also-mocked) GPU branch and assert nothing.
    roc_cpu, pr_cpu = compute_batch_aucs(yt, ys, force_backend="cpu")
    monkeypatch.setattr(gm, "_resolve_backend", lambda *a, **k: True)
    monkeypatch.setattr(gm, "_device_error_classes", lambda: (type(error),))
    with patch.object(gm, "gpu_multiple_roc_auc_scores", side_effect=error):
        roc_gpu_fallback, pr_gpu_fallback = gm.compute_batch_aucs(yt, ys)
    np.testing.assert_allclose(roc_gpu_fallback, roc_cpu, rtol=1e-12)
    np.testing.assert_allclose(pr_gpu_fallback, pr_cpu, rtol=1e-12)


def test_compute_batch_rmse_falls_back_to_cpu_on_gpu_dispatch_failure(monkeypatch):
    """A simulated GPU dispatch failure degrades RMSE to the CPU reference instead of propagating."""
    rng = np.random.default_rng(1)
    yt = rng.standard_normal((200, 3))
    yp = rng.standard_normal((200, 3))
    out_cpu = compute_batch_rmse(yt, yp, force_backend="cpu")
    monkeypatch.setattr(gm, "_resolve_backend", lambda *a, **k: True)
    monkeypatch.setattr(gm, "_device_error_classes", lambda: (MemoryError,))
    with patch.object(gm, "gpu_multiple_rmse_scores", side_effect=MemoryError("bad allocation")):
        out_fallback = gm.compute_batch_rmse(yt, yp)
    np.testing.assert_allclose(out_fallback, out_cpu, rtol=1e-12)


def test_force_backend_gpu_still_raises_on_dispatch_failure(monkeypatch):
    """An explicit force_backend='gpu' means GPU-or-error, not a silent downgrade -- the fallback only
    applies to auto-dispatch (the common case)."""
    yt, ys = _label_matrix()
    monkeypatch.setattr(gm, "is_gpu_metrics_available", lambda: True)
    monkeypatch.setattr(gm, "_device_error_classes", lambda: (MemoryError,))
    with patch.object(gm, "gpu_multiple_roc_auc_scores", side_effect=MemoryError("bad allocation")):
        with pytest.raises(MemoryError):
            gm.compute_batch_aucs(yt, ys, force_backend="gpu")

    yt_r = np.random.default_rng(2).standard_normal((200, 3))
    yp_r = np.random.default_rng(3).standard_normal((200, 3))
    with patch.object(gm, "gpu_multiple_rmse_scores", side_effect=MemoryError("bad allocation")):
        with pytest.raises(MemoryError):
            gm.compute_batch_rmse(yt_r, yp_r, force_backend="gpu")
