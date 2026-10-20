"""One device launch for the whole ``(nu, n)`` stack of unary transforms of an operand column.

The pair-combo table builds, for every operand of every pair, ``nu`` unary transforms (identity, neg, abs, sqr, ... exp) and stacks them. Doing that with one cupy call per unary is ``nu`` kernel
launches plus a ``stack`` copy per operand; this kernel evaluates all of them in a single elementwise pass whose output is already the stacked, contiguous block. The arithmetic is the one of
``_usability_njit_pool._gpu_apply_unary`` operation by operation, in the element type of the operand column (single or double precision), so the result is bit-identical to that reference.

A small per-call cache keyed by the operand's device buffer lets an operand that recurs across pairs (A with B, A with C, ...) be transformed once.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_BODY = r"""
    long long c = i / n;
    long long j = i - c * n;
    T v = x[j];
    int code = codes[c];
    T r;
    switch (code) {
        case 0: r = v; break;
        case 1: r = -v; break;
        case 2: r = fabs(v); break;
        case 3: r = v * v; break;
        case 4: r = T(1.0) / v; break;
        case 5: r = sqrt(fabs(v)); break;
        case 6: r = pos ? log(v) : log(v + shift); break;
        case 7: r = sin(v); break;
        case 8: r = (T)((v > T(0)) - (v < T(0))); break;
        case 9: r = rint(v); break;
        case 10: r = v * v * v; break;
        case 11: r = T(1.0) / (v * v); break;
        case 12: r = T(1.0) / (v * v * v); break;
        case 13: r = cbrt(v); break;
        case 14: r = pow(v, T(-1.0 / 3.0)); break;
        case 15: r = pow(v, T(-0.5)); break;
        default: r = exp(v); break;
    }
    out = r;
"""

_KERNEL: Optional[Any] = None
_KERNEL_LOCK = threading.Lock()


def _kernel(cp: Any) -> Any:
    """The compiled elementwise kernel, built once per process."""
    global _KERNEL
    if _KERNEL is None:
        with _KERNEL_LOCK:
            if _KERNEL is None:
                _KERNEL = cp.ElementwiseKernel("raw T x, raw int32 codes, int64 n, T shift, bool pos", "T out", _BODY, "unary_stack")
    return _KERNEL


def unary_stack_cuda(d_x: Any, codes_d: Any, xmin: float) -> Optional[Any]:
    """``(len(codes_d), n)`` stack of the unary transforms of the device column ``d_x``, or ``None`` when the kernel cannot be built or launched.

    ``codes_d`` is an int32 device array of unary op-codes (the preset order); ``xmin`` is the column's nanmin, used only by the smart log (code 6).
    """
    try:
        import cupy as cp

        if d_x.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or d_x.ndim != 1:  # Intentional float64: the two element types the kernel is templated for
            return None
        n = int(d_x.shape[0])
        nu = int(codes_d.shape[0])
        out = cp.empty((nu, n), dtype=d_x.dtype)
        shift = float(1e-5 - xmin) if xmin <= 0.0 else 0.0
        _kernel(cp)(d_x, codes_d, np.int64(n), d_x.dtype.type(shift), bool(xmin > 0.0), out)
        return out
    except Exception as exc:  # nosec B110 - any device or compile failure falls back to the per-unary cupy path
        logger.debug("unary_stack_cuda unavailable (%s: %s); using the per-unary path", type(exc).__name__, exc)
        return None


class UnaryStackCache:
    """Per-call cache of unary stacks keyed by the operand's device buffer and the op-code array; bounded by a byte budget, oldest entries evicted first.

    The cache holds references to the operand buffer and the code array so that their ``id`` cannot be reused while the entry lives.
    """

    def __init__(self, budget_bytes: int):
        self._budget = int(budget_bytes)
        self._used = 0
        self._entries: "dict[tuple[int, int], tuple[Any, Any, Any]]" = {}
        self._lock = threading.Lock()

    def get(self, d_x: Any, codes_d: Any, xmin: float) -> Optional[Any]:
        """The stack for ``d_x``, from the cache when present; ``None`` when the kernel is unavailable."""
        key = (id(d_x), id(codes_d))
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None:
                return hit[2]
            stack = unary_stack_cuda(d_x, codes_d, xmin)
            if stack is None:
                return None
            size = int(stack.nbytes)
            if size > self._budget:
                return stack
            while self._entries and self._used + size > self._budget:
                old = next(iter(self._entries))
                self._used -= int(self._entries.pop(old)[2].nbytes)
            self._entries[key] = (d_x, codes_d, stack)
            self._used += size
            return stack

    def clear(self) -> None:
        """Drop every cached stack."""
        self._entries.clear()
        self._used = 0
