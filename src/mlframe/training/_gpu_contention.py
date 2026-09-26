"""Train a target's boosters on CPU while the GPU is busy with somebody else's work, and back on GPU once it is free.

The GPU decision used to look only at the card's TOTAL memory, once per target. A production run shared the card with
an embedding job: CatBoost started on a GPU at 100% utilisation with 8.1 of 8.2 GB taken, ran at a sixth of its early
rate for two hours and was cut off by the time budget. The same fit on CPU takes minutes.

:func:`gpu_busy_reason` looks at the card as it is right now, just before a target's models are configured:

* the memory other processes hold (our own process's cupy pools and previous CatBoost buffers are reusable, so they
  do not count when the driver reports per-process memory; when it does not, as under Windows WDDM, all used memory
  counts), against what the data needs;
* the utilisation, which at this point is not ours (no fit of ours is running), against ``MLFRAME_GPU_BUSY_UTIL_PCT``.

The decision is taken per target, so it goes both ways: a target configured while the card is busy trains on CPU, the
next one configured after the other job finished trains on GPU again. ``MLFRAME_GPU_CONTENTION_CPU_FALLBACK=0``
turns the check off.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

from mlframe.utils.env_flags import env_flag, env_float

logger = logging.getLogger(__name__)

DEFAULT_BUSY_UTIL_PCT = 70.0
"""Utilisation, in percent, above which the card counts as busy with other work. Sampled when no fit of ours runs."""

_LAST_BUSY: dict[str, Optional[str]] = {"reason": None}
_LAST_BUSY_LOCK = threading.Lock()  # parallel targets configure at once; each switch is logged once


def contention_check_enabled() -> bool:
    """False when ``MLFRAME_GPU_CONTENTION_CPU_FALLBACK`` is 0 / false / no / off."""
    return env_flag("MLFRAME_GPU_CONTENTION_CPU_FALLBACK", default=True)


def busy_util_pct() -> float:
    """``MLFRAME_GPU_BUSY_UTIL_PCT``, default :data:`DEFAULT_BUSY_UTIL_PCT`."""
    return env_float("MLFRAME_GPU_BUSY_UTIL_PCT", DEFAULT_BUSY_UTIL_PCT)


def gpu_busy_reason(snapshot: Optional[dict], required_gb: float, *, device_index: int = 0, own_pid: Optional[int] = None,
                    util_threshold_pct: Optional[float] = None) -> Optional[str]:
    """Why the GPU should not be used for the next fit, or None when it is free enough (or cannot be read).

    ``required_gb`` is the memory the fit needs on the card. An unreadable snapshot is not evidence of contention: the
    existing decision stands.
    """
    if not snapshot:
        return None
    gpu = next((g for g in snapshot.get("gpus", []) if g.get("index") == device_index), None)
    if gpu is None:
        return None
    total_mb, used_mb, util = gpu.get("mem_total_mb"), gpu.get("mem_used_mb"), gpu.get("util_pct")
    own_pid = os.getpid() if own_pid is None else own_pid
    procs = [p for p in snapshot.get("processes", []) if p.get("gpu") in (None, device_index)]
    per_process_known = bool(procs) and all(p.get("mem_mb") is not None for p in procs)
    if total_mb:
        if per_process_known:
            held_mb = sum(p["mem_mb"] for p in procs if p.get("pid") != own_pid)
        else:
            held_mb = used_mb or 0.0
        free_gb = (total_mb - held_mb) / 1024.0
        if free_gb < required_gb:
            return (f"gpu{device_index} has {free_gb:.1f} GB free of {total_mb / 1024.0:.1f} GB "
                    f"({'held by other processes' if per_process_known else 'in use'}), the fit needs {required_gb:.1f} GB")
    threshold = busy_util_pct() if util_threshold_pct is None else util_threshold_pct
    if util is not None and util >= threshold:
        return f"gpu{device_index} is {util:.0f}% busy with other work (threshold {threshold:.0f}%, MLFRAME_GPU_BUSY_UTIL_PCT)"
    return None


def cb_device_index(cb_devices: Any) -> int:
    """The first device CatBoost would use (``"0"``, ``"1:2"``, ``"0-3"``, ``[1]``); 0 when unspecified or unparsable."""
    if cb_devices is None:
        return 0
    head = str(cb_devices[0] if isinstance(cb_devices, (list, tuple)) and cb_devices else cb_devices)
    digits = ""
    for ch in head:
        if not ch.isdigit():
            break
        digits += ch
    return int(digits) if digits else 0


def gpu_is_free_now(required_gb: float, *, device_index: int = 0) -> bool:
    """False when :func:`gpu_busy_reason` finds the card busy right now; logs every switch between GPU and CPU once."""
    if not contention_check_enabled():
        return True
    try:
        from ._gpu_state_probe import gpu_snapshot

        reason = gpu_busy_reason(gpu_snapshot(), required_gb, device_index=device_index)
    except Exception as e:
        logger.warning("GPU contention check failed (%s); keeping the configured device.", e)
        return True
    with _LAST_BUSY_LOCK:
        previous = _LAST_BUSY["reason"]
        _LAST_BUSY["reason"] = reason
    if reason is not None:
        logger.warning("[gpu-contention] %s: this target's boosters train on CPU. MLFRAME_GPU_CONTENTION_CPU_FALLBACK=0 keeps the GPU.", reason)
        return False
    if previous is not None:
        logger.info("[gpu-contention] gpu%d is free again (was: %s); this target's boosters train on GPU.", device_index, previous)
    return True


def gpu_fits_now(fits_gpu: bool, fits_cb_gpu: bool, data_size_gb: float, cb_devices: Any) -> "tuple[bool, bool]":
    """``(fits_gpu, fits_cb_gpu)`` of ``configure_training_params``, both False when the card is busy with other work.

    The memory the fit needs is the same budget the total-memory check uses (``GPU_VRAM_SAFE_SATURATION_LIMIT`` of the
    data plus ``GPU_VRAM_SAFE_FREE_LIMIT_GB``)."""
    if not (fits_gpu or fits_cb_gpu):
        return fits_gpu, fits_cb_gpu
    from ._model_factories import GPU_VRAM_SAFE_FREE_LIMIT_GB, GPU_VRAM_SAFE_SATURATION_LIMIT

    if gpu_is_free_now(GPU_VRAM_SAFE_SATURATION_LIMIT * data_size_gb + GPU_VRAM_SAFE_FREE_LIMIT_GB, device_index=cb_device_index(cb_devices)):
        return fits_gpu, fits_cb_gpu
    return False, False
