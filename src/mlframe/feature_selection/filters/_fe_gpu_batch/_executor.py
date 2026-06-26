"""Single-GPU resident FE-batch MI executor (2026-06-26).

Scores a candidate matrix on ONE device by the resident edge-binned plain plug-in MI
(``_hermite_fe_mi._plugin_mi_classif_batch_cuda_resident``), VRAM-budget column-chunked so peak device
memory stays inside a fraction of free VRAM (``_gpu_resident_fe._gpu_k_chunk``). y is uploaded ONCE and
its class span (y_min / n_classes) computed ONCE, then reused across chunks. Math is identical to the CPU
twin ``_fe_cpu_batch.cpu_fe_batch_mi`` (same percentile-edge binning + plug-in MI) so the two backends
select the same forms.

NaN policy: candidate columns are scrubbed to 0 on upload (the FE nan_to_num convention) -- a no-op on the
all-finite production candidate matrix, on which it matches the CPU dense path bit-for-bit. (The CPU path
additionally finite-MASKS partial-NaN columns; that edge case is not reproduced here because the FE
candidate matrices fed to the batcher are nan-filled upstream.)
"""
from __future__ import annotations

import numpy as np


def gpu_fe_batch_mi(
    X_cands: np.ndarray,
    y_codes: np.ndarray,
    nbins: int = 10,
    *,
    device: int | None = None,
    free_blocks: bool = True,
) -> np.ndarray:
    """Edge-binned plain plug-in MI of every column of ``X_cands`` (n, K) vs discrete ``y_codes`` (n,),
    computed on a single GPU. Returns a host (K,) float64 MI array. ``device`` selects the CUDA device
    (default: current). Raises if cupy / CUDA is unavailable -- the dispatcher gates that upstream.
    """
    import cupy as cp

    X = np.ascontiguousarray(X_cands, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, k = X.shape
    out = np.zeros(k, dtype=np.float64)
    if n == 0 or k == 0:
        return out

    # Import via the ``hermite_fe`` PACKAGE, not ``_hermite_fe_mi`` directly: the two are mutually circular
    # (``_hermite_fe_mi`` imports ``hermite_fe`` at its top; ``hermite_fe`` re-imports ``_hermite_fe_mi`` at
    # its bottom). A cold direct ``from .._hermite_fe_mi import`` partially-initialises it and raises; going
    # through the package runs its full __init__ first, so the re-exported name is bound. See _fe_edge_mi.
    from ..hermite_fe import _plugin_mi_classif_batch_cuda_resident
    from .._gpu_resident_fe import _gpu_k_chunk

    dev_ctx = cp.cuda.Device(device) if device is not None else cp.cuda.Device()
    with dev_ctx:
        y_gpu = cp.asarray(np.ascontiguousarray(y_codes, dtype=np.int64))
        y_min = int(y_gpu.min().item())
        n_classes = int(y_gpu.max().item()) - y_min + 1
        # VRAM-budget column chunk: f64 candidate matrix is the dominant resident footprint.
        chunk = _gpu_k_chunk(n, bytes_per_elem=8, max_cols=k)
        for s in range(0, k, chunk):
            sl = slice(s, min(s + chunk, k))
            block = np.ascontiguousarray(X[:, sl])
            Xg = cp.asarray(block)
            cp.nan_to_num(Xg, copy=False)  # FE scrub: no-op on finite, matches CPU dense path
            out[sl] = np.asarray(
                _plugin_mi_classif_batch_cuda_resident(Xg, y_gpu, nbins, y_min=y_min, n_classes=n_classes)
            )
            del Xg
            if free_blocks:
                cp.get_default_memory_pool().free_all_blocks()
    return out
