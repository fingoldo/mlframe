"""Backend dispatcher for the FE batcher -- picks the CPU or GPU scoring path by flags + live hardware.

The two paths (``_fe_cpu_batch.cpu_fe_batch_mi`` and ``_fe_gpu_batch.gpu_fe_batch_mi``) are
selection-IDENTICAL (same edge binning + plug-in MI; ``test_fe_batch_parity``), so this only chooses the
FASTEST path for the host -- it can never change which features are selected. Selection order:

  1. ``MLFRAME_FE_VRAM_BACKEND = cpu | gpu`` -- explicit force (tests / diagnostics / a known-good host).
  2. ``MLFRAME_FE_GPU_STRICT`` -- the diagnostic full-residency flag forces GPU.
  3. AUTO -- KTC per-host crossover (``fe_batch_backend`` cache key) when a tuned entry exists; else the
     conservative default CPU. The dev GTX 1050 Ti measured 0.5x for the resident pack-and-batch pattern,
     so AUTO must NOT route to GPU on an unprofiled weak card -- GPU is opt-in (force / STRICT) or a tuned
     crossover until the Phase-3 sweep populates the cache. A GPU choice degrades to CPU when CUDA is absent.

This mirrors the established ``batch_pair_mi_gpu.dispatch_batch_pair_mi`` / ``plugin_mi_classif_batch_dispatch``
shape (force-env override + KTC + CPU-default fallback), not a default-off opt-in flag.
"""
from __future__ import annotations

import os

import numpy as np


def _cuda_available() -> bool:
    try:
        from ._fe_gpu_strict import _cuda_usable
        return bool(_cuda_usable())
    except Exception:
        return False


def _ktc_backend_choice(n_rows: int, n_cands: int) -> str | None:
    """Per-host tuned CPU/GPU crossover for the FE batch, or None on miss. The Phase-3 sweep populates the
    ``fe_batch_backend`` cache key; until then this returns None (-> conservative CPU default)."""
    try:
        from ._kernel_tuning import get_kernel_tuning_cache
        cache = get_kernel_tuning_cache()
        if cache is None:
            return None
        tuned = cache.lookup("fe_batch_backend")
        if not tuned:
            return None
        crossover = int(tuned.get("gpu_min_work", 0) or 0)
        if crossover > 0 and int(n_rows) * int(n_cands) >= crossover:
            return "gpu"
        return "cpu"
    except Exception:
        return None


def choose_fe_batch_backend(n_rows: int, n_cands: int) -> str:
    """Return "cpu" or "gpu" for the given workload + host. Honours the force env, then STRICT, then a
    KTC crossover, else the conservative CPU default. A "gpu" choice is downgraded to "cpu" without CUDA."""
    forced = os.environ.get("MLFRAME_FE_VRAM_BACKEND", "").strip().lower()
    if forced in ("cpu", "gpu"):
        choice = forced
    elif os.environ.get("MLFRAME_FE_GPU_STRICT", "").strip().lower() in ("1", "true", "on", "yes"):
        choice = "gpu"
    else:
        choice = _ktc_backend_choice(n_rows, n_cands) or "cpu"
    if choice == "gpu" and not _cuda_available():
        return "cpu"
    return choice


def fe_batch_mi(
    X_cands: np.ndarray,
    y_codes: np.ndarray,
    nbins: int = 10,
    *,
    backend: str | None = None,
    n_workers: int = 1,
) -> np.ndarray:
    """Score candidate matrix ``X_cands`` (n, K) vs discrete ``y_codes`` by edge-binned plug-in MI on the
    dispatched backend. ``backend`` overrides the dispatch ("cpu"|"gpu"). A GPU run that raises falls back
    to the (selection-identical) CPU path so a transient device error is never a correctness regression.
    Returns a host (K,) float64 MI array."""
    X = np.asarray(X_cands)
    n = X.shape[0] if X.ndim >= 1 else 0
    k = X.shape[1] if X.ndim == 2 else 1
    chosen = (backend or choose_fe_batch_backend(n, k)).lower()

    if chosen == "gpu":
        try:
            import numpy as _np
            from ._fe_gpu_batch import multi_gpu_fe_batch_mi
            from ._fe_gpu_batch._devices import fe_gpu_f32_enabled
            _dt = _np.float32 if fe_gpu_f32_enabled() else _np.float64  # f32 opt-in: 2.2x, selection-equiv
            return multi_gpu_fe_batch_mi(X, y_codes, nbins, dtype=_dt)  # spreads across GPUs; single-GPU = 1 device
        except Exception:
            pass  # fall through to the selection-identical CPU path
    from ._fe_cpu_batch import cpu_fe_batch_mi
    return cpu_fe_batch_mi(X, y_codes, nbins, n_workers=n_workers)
