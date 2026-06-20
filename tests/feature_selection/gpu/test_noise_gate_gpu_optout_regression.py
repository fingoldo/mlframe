"""Regression: the FE noise-gate dispatcher honours the GPU opt-out BEFORE the KTC lookup.

Under ``CUDA_VISIBLE_DEVICES=""`` (the documented mlframe "no GPU on this run" convention)
cupy still enumerates the device on some builds, so a cached GPU ``backend_choice`` routed
the noise-gate to the cupy path whose device->host copy hangs indefinitely -- not an
exception, so the GPU try/except never caught it. The dispatcher must force the CPU kernel
when the user opted out of CUDA.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_dispatch import (
    _dispatch_batch_mi_with_noise_gate,
)


def _make_args():
    rng = np.random.default_rng(0)
    n, K, nb = 200, 4, 5
    disc_2d = rng.integers(0, nb, size=(n, K)).astype(np.int32)
    y = rng.integers(0, 2, size=n)
    classes_y = np.unique(y).astype(np.int64)
    classes_y_safe = classes_y
    freqs_y = (np.bincount(y) / n).astype(np.float64)
    return disc_2d, classes_y, classes_y_safe, freqs_y


def test_cuda_visible_devices_empty_forces_cpu_kernel(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    disc_2d, classes_y, classes_y_safe, freqs_y = _make_args()

    called = {"cpu": False}

    def _sentinel_cpu_kernel(**kwargs):
        called["cpu"] = True
        return np.zeros(disc_2d.shape[1], dtype=np.float64)

    out = _dispatch_batch_mi_with_noise_gate(
        disc_2d=disc_2d, quantization_nbins=5, classes_y=classes_y,
        classes_y_safe=classes_y_safe, freqs_y=freqs_y, npermutations=4,
        min_nonzero_confidence=0.0, use_su=False, batch_mi_kernel=_sentinel_cpu_kernel,
    )
    assert called["cpu"], "GPU opt-out did not force the CPU kernel under CUDA_VISIBLE_DEVICES=''"
    assert out.shape[0] == disc_2d.shape[1]


def test_mlframe_disable_gpu_forces_cpu_kernel(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    disc_2d, classes_y, classes_y_safe, freqs_y = _make_args()
    called = {"cpu": False}

    def _sentinel_cpu_kernel(**kwargs):
        called["cpu"] = True
        return np.zeros(disc_2d.shape[1], dtype=np.float64)

    _dispatch_batch_mi_with_noise_gate(
        disc_2d=disc_2d, quantization_nbins=5, classes_y=classes_y,
        classes_y_safe=classes_y_safe, freqs_y=freqs_y, npermutations=4,
        min_nonzero_confidence=0.0, use_su=False, batch_mi_kernel=_sentinel_cpu_kernel,
    )
    assert called["cpu"], "MLFRAME_DISABLE_GPU=1 did not force the CPU kernel"
