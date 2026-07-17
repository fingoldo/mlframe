"""Regression: the FE noise-gate dispatcher honours the GPU opt-out BEFORE the KTC lookup.

Under ``CUDA_VISIBLE_DEVICES=""`` (the documented mlframe "no GPU on this run" convention)
cupy still enumerates the device on some builds, so a cached GPU ``backend_choice`` routed
the noise-gate to the cupy path whose device->host copy hangs indefinitely -- not an
exception, so the GPU try/except never caught it. The dispatcher must force the CPU kernel
when the user opted out of CUDA.

Contract probe: under opt-out the dispatcher returns a valid ``fe_mi[K]`` WITHOUT ever
entering the GPU twin (``_batch_mi_with_noise_gate_gpu``). We spy on the GPU twin and assert
it is never called (it is the device path the opt-out must skip), rather than relying on the
caller-passed CPU kernel being invoked verbatim -- the dispatcher legitimately swaps in the
per-host KTC-selected njit kernel via ``select_batch_mi_kernel``, so a sentinel passed as
``batch_mi_kernel`` is not the right liveness probe.
"""

from __future__ import annotations

import numpy as np

import mlframe.feature_selection.filters._feature_engineering_pairs._pairs_dispatch as _disp
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_dispatch import (
    _dispatch_batch_mi_with_noise_gate,
)
from mlframe.feature_selection.filters.info_theory._batch_kernels import (
    batch_mi_with_noise_gate,
)


def _make_args():
    rng = np.random.default_rng(0)
    n, K, nb = 200, 4, 5
    disc_2d = rng.integers(0, nb, size=(n, K)).astype(np.int32)
    y = rng.integers(0, 2, size=n)
    # ``classes_y`` is the PER-ROW dense class-code vector (length n), indexed ``classes_y[r]`` inside the
    # kernel -- NOT the unique-class list. ``freqs_y`` is the per-class proportion table (length n_classes).
    classes_y = y.astype(np.int64)
    classes_y_safe = classes_y
    freqs_y = (np.bincount(y) / n).astype(np.float64)
    return disc_2d, classes_y, classes_y_safe, freqs_y


def _run_under_optout(monkeypatch, env_setup):
    """Dispatch once under an opt-out env, spying the GPU twin. Returns (out, gpu_called)."""
    env_setup(monkeypatch)
    disc_2d, classes_y, classes_y_safe, freqs_y = _make_args()

    gpu_called = {"hit": False}
    _orig_gpu = _disp._batch_mi_with_noise_gate_gpu

    def _spy_gpu(*args, **kwargs):
        gpu_called["hit"] = True
        return _orig_gpu(*args, **kwargs)

    monkeypatch.setattr(_disp, "_batch_mi_with_noise_gate_gpu", _spy_gpu)

    out = _dispatch_batch_mi_with_noise_gate(
        disc_2d=disc_2d,
        quantization_nbins=5,
        classes_y=classes_y,
        classes_y_safe=classes_y_safe,
        freqs_y=freqs_y,
        npermutations=4,
        min_nonzero_confidence=0.0,
        use_su=False,
        batch_mi_kernel=batch_mi_with_noise_gate,
    )
    return out, gpu_called["hit"], disc_2d


def test_cuda_visible_devices_empty_forces_cpu_kernel(monkeypatch):
    out, gpu_called, disc_2d = _run_under_optout(
        monkeypatch,
        lambda mp: mp.setenv("CUDA_VISIBLE_DEVICES", ""),
    )
    assert not gpu_called, "GPU twin was entered despite CUDA_VISIBLE_DEVICES='' opt-out"
    assert out.shape[0] == disc_2d.shape[1]
    assert np.all(np.isfinite(out))


def test_mlframe_disable_gpu_forces_cpu_kernel(monkeypatch):
    def _env(mp):
        mp.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        mp.setenv("MLFRAME_DISABLE_GPU", "1")

    out, gpu_called, disc_2d = _run_under_optout(monkeypatch, _env)
    assert not gpu_called, "GPU twin was entered despite MLFRAME_DISABLE_GPU=1 opt-out"
    assert out.shape[0] == disc_2d.shape[1]
    assert np.all(np.isfinite(out))
