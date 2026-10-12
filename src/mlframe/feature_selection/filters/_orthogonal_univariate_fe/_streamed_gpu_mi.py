"""Streamed, multi-GPU plug-in MI over column blocks: bounded pinned staging and VRAM-sized blocks.

Columns arrive as blocks (any iterable of ``(n_rows, width)`` float32 arrays), so the full ``n x p`` matrix is never resident.
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
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

PIN_CAP_FRAC = 0.20  # share of free host RAM that pinned staging may take
PIN_CAP_MAX = 1 << 30  # never more than 1 GiB pinned in total
VRAM_FRAC = 0.50  # share of a device's free VRAM one block may use
VRAM_RESERVE = 256 << 20  # kept free on every device
WORK_FACTOR = 3  # block bytes x this = VRAM the MI intermediates need
BLOCK_BYTES_MAX = 512 << 20  # no block larger than this, whatever the VRAM

__all__ = ["plan_streamed_mi", "mi_classif_streamed"]


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
            out.append((d, int(free)))
        return out[:max_devices] if max_devices else out
    except Exception:
        logger.debug("streamed GPU MI: no CUDA device", exc_info=True)
        return []


def _host_free_bytes() -> int:
    """Bytes of host RAM available now (2 GiB when psutil is missing, the conservative default)."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 2 * 1024**3


def plan_streamed_mi(n_rows: int, devices: list[tuple[int, int]], host_free: int) -> dict:
    """Block width (columns), staging bytes and the pinned budget for a run; ``width`` is 0 when no device fits one column."""
    if not devices:
        return {"width": 0, "block_bytes": 0, "pinned_total": 0, "devices": []}
    col_bytes = max(1, int(n_rows) * 4)
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


def mi_classif_streamed(blocks: Iterable[np.ndarray], y: np.ndarray, *, nbins: int = 10, n_rows: int, max_devices: Optional[int] = None) -> np.ndarray:
    """Plug-in MI of every column of every block against ``y``, concatenated in block order.

    ``blocks`` yields ``(n_rows, w)`` float32 arrays (each block may have its own width up to the planned one). Returns float64
    MI per column. Raises when no CUDA device is usable: the caller decides the fallback."""
    import cupy as cp
    from .._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident

    devices = _devices(max_devices)
    plan = plan_streamed_mi(n_rows, devices, _host_free_bytes())
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
        block = np.asarray(block, dtype=np.float32)
        w = block.shape[1]
        if w > plan["width"]:
            raise ValueError(f"block of {w} columns exceeds the planned width {plan['width']}")
        d = plan["devices"][k % len(plan["devices"])]
        buf = stage[d][k // len(plan["devices"]) % 2]
        with cp.cuda.Device(d):
            host = np.frombuffer(buf, dtype=np.float32, count=n_rows * w).reshape(n_rows, w)
            np.copyto(host, block)
            dev_block = cp.asarray(host)
            mi = _plugin_mi_classif_batch_cuda_resident(
                dev_block, dev_y[d], int(nbins), y_min=y_min, n_classes=n_classes, keep_dtype=True,
            )
            out.append(cp.asnumpy(cp.asarray(mi, dtype=cp.float64)))
            del dev_block
    if not out:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(out)
