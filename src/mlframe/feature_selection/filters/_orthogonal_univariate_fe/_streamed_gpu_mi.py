"""Streamed, multi-GPU plug-in MI over column blocks: bounded pinned staging and VRAM-sized blocks.

Columns arrive as blocks (any iterable of ``(n_rows, width)`` arrays of the caller's dtype), so the full ``n x p`` matrix is never resident.
Each block is copied into one of two pinned staging buffers per device, uploaded, scored with the CUDA batch MI, and dropped.
Blocks go round-robin over the visible CUDA devices; each device's budget is its own free VRAM, so the block width is the
smallest that fits every device.

Safety margins (measured, pinned against pageable on this host: 5.8 vs 3.8 GB/s H2D):
  * pinned staging is capped at ``PIN_CAP_FRAC`` of free host RAM and at ``PIN_CAP_MAX`` bytes in total; the two buffers per device
    are allocated once and reused, because the cupy pinned pool exhausts under many separate allocations (cudaErrorAlreadyMapped);
  * a block uses at most ``VRAM_FRAC`` of a device's free VRAM minus ``VRAM_RESERVE``, sized with a ``WORK_FACTOR`` for the MI
    intermediates (the batch MI holds the block, its ranks and the joint histograms).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

PIN_CAP_FRAC = 0.20  # share of free host RAM that pinned staging may take
PIN_CAP_MAX = 1 << 30  # never more than 1 GiB pinned in total
VRAM_FRAC = 0.50  # share of a device's free VRAM one block may use
VRAM_RESERVE = 256 << 20  # kept free on every device
WORK_FACTOR = 3  # block bytes x this = VRAM the MI intermediates need
BLOCK_BYTES_MAX = 512 << 20  # no block larger than this, whatever the VRAM

HOST_BLOCK_FRAC = 0.10  # share of free host RAM the CPU batch MI may take for one whole matrix (same budget _mi_chunk_cols_for blocks to)
# Sized for the wider of the two working widths (8 bytes per value), so the budget also covers the float32 default that the CPU
# batch MI uses through _crit_np_dtype(); the budget only over-estimates when the narrower width is in effect.
CPU_WORK_ITEMSIZE = 8

__all__ = ["plan_streamed_mi", "mi_classif_streamed", "choose_mi_path"]


def _devices(max_devices: Optional[int] = None) -> list[tuple[int, int]]:
    """``[(device_id, free_vram_bytes)]`` of the usable CUDA devices, or [] when CUDA is off or globally disabled."""
    try:
        from .._gpu_policy import gpu_globally_disabled
        import cupy as cp
        if gpu_globally_disabled():
            return []
        out = []
        for d in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(d):
                free, _total = cp.cuda.runtime.memGetInfo()
                # bytes the cupy pool holds but has not handed out are reusable by this process, so they count as free
                # (the same correction the FE VRAM gate applies); without it the block width shrinks after every run
                free += int(cp.get_default_memory_pool().free_bytes())
            out.append((d, int(free)))
        return out[:max_devices] if max_devices is not None else out
    except Exception:
        # fallback: no usable CUDA device is the expected CPU-only case; no streamed GPU devices are returned.
        logger.debug("streamed GPU MI: no CUDA device", exc_info=True)
        return []


_HOST_FREE_TTL_SECONDS = 1.0  # the probe costs ~2 ms on a memory-starved Windows host and runs once per MI path choice; free RAM does not move that fast
_host_free_cache: "tuple[float, int]" = (float("-inf"), 0)


def _host_free_bytes() -> int:
    """Bytes of host RAM available now, read at most once per ``_HOST_FREE_TTL_SECONDS`` (2 GiB when psutil is missing, the conservative default)."""
    global _host_free_cache
    now = time.monotonic()
    stamp, cached = _host_free_cache
    if now - stamp < _HOST_FREE_TTL_SECONDS:
        return cached
    try:
        import psutil
        value = int(psutil.virtual_memory().available)
    except Exception as _exc:
        logger.debug("free-memory probe failed, assuming 2 GiB: %s", _exc)
        value = 2 * 1024**3
    _host_free_cache = (now, value)
    return value


def plan_streamed_mi(n_rows: int, devices: list[tuple[int, int]], host_free: int, dtype: Any) -> dict:
    """Block width (columns), staging bytes and the pinned budget for a run; ``width`` is 0 when no device fits one column."""
    if not devices:
        return {"width": 0, "block_bytes": 0, "pinned_total": 0, "devices": []}
    itemsize = int(np.dtype(dtype).itemsize)
    col_bytes = max(1, int(n_rows) * itemsize)
    widths = []
    for _d, free in devices:
        budget = max(0, int(free * VRAM_FRAC) - VRAM_RESERVE)
        widths.append(budget // (col_bytes * WORK_FACTOR))
    width = int(min(widths))
    width = min(width, BLOCK_BYTES_MAX // col_bytes)
    pinned_total = min(int(host_free * PIN_CAP_FRAC), PIN_CAP_MAX)
    # two buffers per device; if the cap cannot hold them at this width, shrink the width to fit
    while width > 0 and 2 * len(devices) * width * col_bytes > pinned_total:
        width -= 1
    return {
        "width": int(width),
        "block_bytes": int(width * col_bytes),
        "pinned_total": int(pinned_total),
        "devices": [d for d, _ in devices],
    }


def choose_mi_path(n_rows: int, n_cols: int, dtype: Any, devices: list[tuple[int, int]], host_free: int) -> str:
    """Which MI path a ``(n_rows, n_cols)`` batch takes, decided from free memory alone.

    * ``"resident"``: the whole matrix is scored in one batch. On a GPU when the planner's block width (the widest block every
      device can hold, see :func:`plan_streamed_mi`) covers all ``n_cols``; on the CPU when the whole batch's 8-byte-per-value working set
      (``n_rows * n_cols * 8 * WORK_FACTOR``) fits ``HOST_BLOCK_FRAC`` of free host RAM.
    * ``"streamed"``: GPU present, but the matrix must go through VRAM in blocks (:func:`mi_classif_streamed`).
    * ``"cpu_blocks"``: no usable GPU (or it cannot hold one column) and the whole batch does not fit host RAM; the CPU scores
      column blocks sized to free RAM.

    No row or column constant is involved: every threshold is a planner fraction applied to the free bytes passed in."""
    if int(n_cols) <= 0:
        return "resident"
    if devices:
        plan = plan_streamed_mi(n_rows, devices, host_free, dtype)
        if plan["width"] >= int(n_cols):
            return "resident"
        if plan["width"] > 0:
            return "streamed"
    whole = int(n_rows) * int(n_cols) * CPU_WORK_ITEMSIZE * WORK_FACTOR
    if whole <= int(host_free * HOST_BLOCK_FRAC):
        return "resident"
    return "cpu_blocks"


def mi_classif_streamed(blocks: Iterable[np.ndarray], y: np.ndarray, *, nbins: int, n_rows: int, dtype: Any, max_devices: Optional[int] = None) -> np.ndarray:
    """Plug-in MI of every column of every block against ``y``, concatenated in block order.

    ``blocks`` yields ``(n_rows, w)`` arrays of ``dtype`` (each block may have its own width up to the planned one). Returns the
    MI per column. Raises when no CUDA device is usable: the caller decides the fallback.

    This is the production streamed path (``_mi_classif_batch_numba`` routes here when :func:`choose_mi_path` says ``"streamed"``).
    :func:`mi_classif_streamed_fill` is its zero-copy twin; the parity test pins the two to identical output."""
    import cupy as cp
    from .._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident

    devices = _devices(max_devices)
    plan = plan_streamed_mi(n_rows, devices, _host_free_bytes(), dtype)
    if plan["width"] <= 0:
        raise RuntimeError("streamed GPU MI: no device has VRAM for a column block")
    stage: dict[int, list] = {}
    for d in plan["devices"]:
        with cp.cuda.Device(d):
            # two pinned staging buffers per device, allocated once and reused for the whole run
            stage[d] = [cp.cuda.alloc_pinned_memory(plan["block_bytes"]) for _ in range(2)]
    y64 = np.ascontiguousarray(y, dtype=np.int64)
    y_min = int(y64.min()) if y64.size else 0
    n_classes = (int(y64.max()) - y_min + 1) if y64.size else 1
    dev_y: dict[int, object] = {}
    for d in plan["devices"]:
        with cp.cuda.Device(d):
            dev_y[d] = cp.asarray(y64)  # one copy of the labels per device, reused for every block
    out: list[np.ndarray] = []
    for k, block in enumerate(blocks):
        block = np.asarray(block, dtype=dtype)
        w = block.shape[1]
        if w > plan["width"]:
            raise ValueError(f"block of {w} columns exceeds the planned width {plan['width']}")
        d = plan["devices"][k % len(plan["devices"])]
        buf = stage[d][k // len(plan["devices"]) % 2]
        with cp.cuda.Device(d):
            host = np.frombuffer(buf, dtype=dtype, count=n_rows * w).reshape(n_rows, w)
            np.copyto(host, block)
            dev_block = cp.asarray(host)
            mi = _plugin_mi_classif_batch_cuda_resident(
                dev_block, dev_y[d], int(nbins), y_min=y_min, n_classes=n_classes, keep_dtype=True,
            )
            out.append(cp.asnumpy(mi))
            del dev_block
    if not out:
        return np.zeros(0, dtype=dtype)
    return np.concatenate(out)


def mi_classif_streamed_fill(fill: Any, n_blocks: int, widths: list, y: np.ndarray, *, nbins: int, n_rows: int, dtype: Any, max_devices: Optional[int] = None) -> np.ndarray:
    """Like :func:`mi_classif_streamed`, but the source writes each block straight into the pinned staging buffer.

    ``fill(k, out)`` receives a writable ``(n_rows, widths[k])`` view of ``dtype`` of the pinned buffer and must fill it in place; this
    removes the host-side copy that :func:`mi_classif_streamed` makes from a numpy block into the staging buffer. ``widths`` is the
    column count of each block (known up front, so the staging buffers can be planned)."""
    import cupy as cp
    from .._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident

    devices = _devices(max_devices)
    plan = plan_streamed_mi(n_rows, devices, _host_free_bytes(), dtype)
    if plan["width"] <= 0:
        raise RuntimeError("streamed GPU MI: no device has VRAM for a column block")
    if max(widths, default=0) > plan["width"]:
        raise ValueError(f"a block of {max(widths)} columns exceeds the planned width {plan['width']}")
    stage: dict[int, list] = {}
    for d in plan["devices"]:
        with cp.cuda.Device(d):
            stage[d] = [cp.cuda.alloc_pinned_memory(plan["block_bytes"]) for _ in range(2)]
    y64 = np.ascontiguousarray(y, dtype=np.int64)
    y_min = int(y64.min()) if y64.size else 0
    n_classes = (int(y64.max()) - y_min + 1) if y64.size else 1
    dev_y: dict[int, object] = {}
    for d in plan["devices"]:
        with cp.cuda.Device(d):
            dev_y[d] = cp.asarray(y64)
    out: list[np.ndarray] = []
    for k in range(n_blocks):
        w = int(widths[k])
        d = plan["devices"][k % len(plan["devices"])]
        buf = stage[d][k // len(plan["devices"]) % 2]
        with cp.cuda.Device(d):
            host = np.frombuffer(buf, dtype=dtype, count=n_rows * w).reshape(n_rows, w)
            fill(k, host)
            dev_block = cp.empty((n_rows, w), dtype=dtype)
            cp.cuda.runtime.memcpy(dev_block.data.ptr, buf.ptr, n_rows * w * host.itemsize, cp.cuda.runtime.memcpyHostToDevice)
            mi = _plugin_mi_classif_batch_cuda_resident(dev_block, dev_y[d], int(nbins), y_min=y_min, n_classes=n_classes, keep_dtype=True)
            out.append(cp.asnumpy(mi))
            del dev_block
    if not out:
        return np.zeros(0, dtype=dtype)
    return np.concatenate(out)
