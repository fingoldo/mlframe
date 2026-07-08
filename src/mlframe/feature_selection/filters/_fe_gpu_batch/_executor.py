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

import os
from typing import Any

import numpy as np

# Column blocks per device for the packer's load-balancing granularity (more blocks -> finer makespan
# balance across heterogeneous devices; the per-device executor re-chunks each block for VRAM anyway).
_BLOCKS_PER_DEVICE = int(os.environ.get("MLFRAME_FE_VRAM_BLOCKS_PER_DEVICE", "4") or 4)


def gpu_fe_batch_mi(
    X_cands: np.ndarray,
    y_codes: np.ndarray,
    nbins: int = 10,
    *,
    device: int | None = None,
    free_blocks: bool = True,
    scrub: bool = True,
    dtype: "np.dtype | type" = np.float64,
) -> np.ndarray:
    """Edge-binned plain plug-in MI of every column of ``X_cands`` (n, K) vs discrete ``y_codes`` (n,),
    computed on a single GPU. Returns a host (K,) float64 MI array. ``device`` selects the CUDA device
    (default: current). Raises if cupy / CUDA is unavailable -- the dispatcher gates that upstream.
    """
    import cupy as cp

    _f32 = np.dtype(dtype) == np.float32  # f32: half the H2D + f32 radix-select (selection-equivalent, not 1e-9)
    X = np.ascontiguousarray(X_cands, dtype=(np.float32 if _f32 else np.float64))
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
        # VRAM-budget column chunk: the candidate matrix is the dominant resident footprint (4B f32 / 8B f64).
        chunk = _gpu_k_chunk(n, bytes_per_elem=(4 if _f32 else 8), max_cols=k)
        for s in range(0, k, chunk):
            sl = slice(s, min(s + chunk, k))
            block = np.ascontiguousarray(X[:, sl])
            Xg = cp.asarray(block)
            # FE scrub (inf/-inf/nan -> 0, the FE convention -- NOT cupy's default inf->float-max). cupy's
            # nan_to_num always runs a full array scan (_check_nan_inf, ~12% of the GPU wall here), so callers
            # that already guarantee finite columns (e.g. the orth path's dense finite-filtered subset) pass
            # scrub=False to skip it entirely. Default True keeps the generic path safe on non-finite input.
            if scrub:
                # In-place where(isfinite) not cp.nan_to_num(nan=0,...): nan_to_num runs cupy.isnan() on each
                # scalar fill arg (a blocking D2H sync); Xg[...]= assigns elementwise in place, identical result.
                Xg[...] = cp.where(cp.isfinite(Xg), Xg, cp.asarray(0.0, dtype=Xg.dtype))
            out[sl] = np.asarray(_plugin_mi_classif_batch_cuda_resident(Xg, y_gpu, nbins, y_min=y_min, n_classes=n_classes, keep_dtype=_f32))
            del Xg
            if free_blocks:
                cp.get_default_memory_pool().free_all_blocks()
    return out


def multi_gpu_fe_batch_mi(
    X_cands: np.ndarray,
    y_codes: np.ndarray,
    nbins: int = 10,
    *,
    profiles: Any = None,
    scrub: bool = True,
    dtype: "np.dtype | type" = np.float64,
) -> np.ndarray:
    """Edge-binned plug-in MI of (n, K) ``X_cands`` spread across HETEROGENEOUS GPUs to minimise wall time.

    Columns are partitioned into blocks, the blocks are assigned to devices by the speed-weighted CP-SAT
    packer (faster GPU gets more), and each device scores its columns resident IN PARALLEL (one thread per
    device; cupy device contexts are thread-local). Per-column MI is assignment-invariant, so the result is
    identical to the single-GPU path regardless of how columns are spread (``test_fe_multi_gpu``). Collapses
    to the single-device executor when 0 or 1 device is visible -- the multi-GPU code runs unchanged on a
    1-GPU host. ``profiles`` (a list of ``DeviceProfile``) overrides enumeration (used by tests).
    """
    from concurrent.futures import ThreadPoolExecutor

    from ._devices import enumerate_device_profiles
    from ._packer import pack_blocks_to_devices

    X = np.ascontiguousarray(X_cands, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, k = X.shape
    out = np.zeros(k, dtype=np.float64)
    if n == 0 or k == 0:
        return out

    profs = list(profiles) if profiles is not None else enumerate_device_profiles()
    if len(profs) <= 1:
        dev = profs[0].device if profs else None
        return gpu_fe_batch_mi(X, y_codes, nbins, device=dev, scrub=scrub, dtype=dtype)

    g = len(profs)
    # Column blocks: enough for the packer to balance across devices; the per-device executor re-chunks
    # each device's columns for VRAM, so block size here is purely load-balancing granularity.
    n_blocks = min(k, max(g, g * _BLOCKS_PER_DEVICE))
    bounds = np.linspace(0, k, n_blocks + 1).astype(int)
    blocks = [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_blocks) if bounds[i + 1] > bounds[i]]
    works = [e - s for (s, e) in blocks]
    speeds = [p.speed for p in profs]
    dev_of_block = pack_blocks_to_devices(works, speeds)

    cols_per_dev: dict[int, list[int]] = {d: [] for d in range(g)}
    for bi, (s, e) in enumerate(blocks):
        cols_per_dev[dev_of_block[bi]].extend(range(s, e))

    def _score_on(dev_slot: int) -> None:
        """Score this device's assigned column block via ``gpu_fe_batch_mi`` on its own GPU, writing results into ``out`` at the original column indices; run from the ThreadPoolExecutor so devices execute concurrently."""
        cols = cols_per_dev[dev_slot]
        if not cols:
            return
        idx = np.asarray(cols, dtype=np.int64)
        sub = np.ascontiguousarray(X[:, idx])
        out[idx] = gpu_fe_batch_mi(sub, y_codes, nbins, device=profs[dev_slot].device, scrub=scrub, dtype=dtype)

    with ThreadPoolExecutor(max_workers=g) as pool:
        list(pool.map(_score_on, [d for d in range(g) if cols_per_dev[d]]))
    return out
