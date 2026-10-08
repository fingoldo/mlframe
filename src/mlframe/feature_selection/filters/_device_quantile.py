"""Exact ``np.quantile`` (linear method) from a device sort.

``np.quantile`` of a 1M-row column partitions on the host (~115 ms per call at 9 quantiles); the FE stages call it per column on data that is going to the
device anyway. One device sort plus a read of the handful of order statistics the linear method needs, interpolated with numpy's own ``_lerp`` and index
rules, returns the SAME bits as ``np.quantile``: the host version also sorts exactly those elements and applies the same arithmetic. The result is
self-checked against ``np.quantile`` on a small array the first time it is used; if numpy's internals ever drift, the device path switches itself off and
callers keep the host call.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_STATE = {"verified": None}  # None = not checked yet, True / False after the one-time self-check
_LOCK = threading.Lock()


def _interpolate(sorted_lo: np.ndarray, sorted_hi: np.ndarray, virtual: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """numpy's linear-method interpolation of the two bracketing order statistics."""
    from numpy.lib.function_base import _lerp  # type: ignore[attr-defined]

    gamma = np.asanyarray(virtual - prev)
    return np.asarray(_lerp(sorted_lo, sorted_hi, gamma))


def _indices(n: int, qs: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Bracketing indices and virtual positions exactly as ``numpy.lib.function_base._get_indexes`` derives them for the linear method."""
    virtual = np.asanyarray((n - 1) * np.asarray(qs, dtype=np.float64))
    prev = np.asanyarray(np.floor(virtual)).astype(np.intp)
    nxt = prev + 1
    above = virtual >= n - 1
    if above.any():
        prev[above] = n - 1
        nxt[above] = n - 1
    below = virtual < 0
    if below.any():
        prev[below] = 0
        nxt[below] = 0
    return virtual, prev, nxt


def _device_quantile_raw(x, qs: np.ndarray) -> Optional[np.ndarray]:
    """The device computation itself (``x`` host or device); ``None`` for NaN-bearing or empty input."""
    import cupy as cp

    n = int(x.size)
    if n == 0:
        return None
    xs = cp.sort(x.astype(cp.float64, copy=False) if isinstance(x, cp.ndarray) else cp.asarray(np.ascontiguousarray(x, dtype=np.float64)))
    virtual, prev, nxt = _indices(n, qs)
    idx = np.concatenate([prev, nxt, [n - 1]])
    vals = xs[cp.asarray(idx)].get()
    if np.isnan(vals[-1]):
        return None  # a NaN sorts last: np.quantile propagates it, so leave that case to the host
    lo, hi = vals[: len(prev)], vals[len(prev) : 2 * len(prev)]
    return _interpolate(lo, hi, virtual, prev)


def _self_check() -> bool:
    """Compare the device result with ``np.quantile`` on arrays with ties, negatives and heavy tails; any difference disables the device path."""
    rng = np.random.default_rng(12345)
    qs = np.linspace(0.0, 1.0, 11)[1:-1]
    for arr in (rng.standard_normal(1001), np.round(rng.standard_normal(2000), 1), rng.standard_normal(777) ** 3, rng.integers(0, 3, 500).astype(np.float64)):
        got = _device_quantile_raw(arr, qs)
        if got is None or not np.array_equal(got, np.quantile(arr, qs)):
            return False
    return True


def device_quantile(x, qs: np.ndarray) -> Optional[np.ndarray]:
    """``np.quantile(x, qs)`` (linear) computed from a device sort, bit-identical, or ``None`` when the device path does not apply (no cupy, NaNs, a failed
    self-check) so the caller uses the host call."""
    with _LOCK:
        if _STATE["verified"] is None:
            try:
                _STATE["verified"] = bool(_self_check())
            except Exception as e:
                logger.debug("device quantile self-check failed, staying on the host: %s", e)
                _STATE["verified"] = False
            if not _STATE["verified"]:
                logger.warning("device quantile did not reproduce np.quantile exactly on the self-check; using the host np.quantile")
        if not _STATE["verified"]:
            return None
    try:
        return _device_quantile_raw(x if hasattr(x, "__cuda_array_interface__") else np.asarray(x), np.asarray(qs, dtype=np.float64))
    except Exception as e:
        logger.debug("device quantile failed, using the host: %s", e)
        return None
