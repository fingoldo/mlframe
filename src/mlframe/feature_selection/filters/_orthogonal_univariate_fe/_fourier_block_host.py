"""Numba CPU twin of the Fourier / chirp block kernel (``_extra_basis_fourier_cuda.fourier_block_cuda``).

For a ``(names, rows)`` block, name ``j`` reads source column ``src[j]`` of the stacked operands, forms the axis ``z`` (the linear power axis ``(x**p - lo) / span`` or the chirp axis
``(sign(zc) * zc**2 - lo) / span`` with ``zc = (x - mean) / std``) and writes ``sin|cos((2 pi freq) * z)``. The device kernel evaluates the same expressions in the element type of the operand column;
this loop does too (the parameters are rounded once to that type), one pass per name instead of the roughly eight whole-array passes of the numpy chain, parallel over the rows of every name.
``fourier_block_host`` is the reference the device kernel is checked against and the host side of the kernel-tuning entry of ``_extra_basis_fourier_cuda``.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - numba is a hard dependency in practice
    from mlframe._numba_fallback import njit, prange

__all__ = ["fourier_block_host", "sincos_axis_host", "sincos_legs"]


@njit(parallel=True, cache=True, nogil=True)
def _fourier_block_njit(xs, src, cosine, power, quad, lo, span, mean, std, w):
    """``out[j, i]`` for every name ``j`` and row ``i``; ``xs`` is ``(sources, n)`` and the parameter vectors have the dtype of ``xs``."""
    k = src.shape[0]
    n = xs.shape[1]
    out = np.empty((k, n), dtype=xs.dtype)
    for j in range(k):
        x = xs[src[j]]
        lo_j = lo[j]
        span_j = span[j]
        w_j = w[j]
        if quad[j] != 0:
            mean_j = mean[j]
            std_j = std[j]
            for i in prange(n):
                zc = (x[i] - mean_j) / std_j
                sgn = (1 if zc > 0 else 0) - (1 if zc < 0 else 0)
                z = (sgn * (zc * zc) - lo_j) / span_j
                ang = w_j * z
                out[j, i] = np.cos(ang) if cosine[j] != 0 else np.sin(ang)
        else:
            p = power[j]
            for i in prange(n):
                xp = x[i] if p == 1 else x[i] ** p
                z = (xp - lo_j) / span_j
                ang = w_j * z
                out[j, i] = np.cos(ang) if cosine[j] != 0 else np.sin(ang)
    return out


def fourier_block_host(xs: np.ndarray, src: np.ndarray, cosine: np.ndarray, power: np.ndarray, quad: np.ndarray, lo: np.ndarray, span: np.ndarray, mean: np.ndarray, std: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The ``(names, rows)`` block on the host. ``xs`` is ``(sources, rows)`` float32 or float64; ``lo``, ``span``, ``mean``, ``std`` and ``w`` are rounded once to its dtype, as the device launcher rounds them."""
    dt = xs.dtype
    return np.asarray(
        _fourier_block_njit(
            np.ascontiguousarray(xs),
            np.ascontiguousarray(src, dtype=np.int64),
            np.ascontiguousarray(cosine, dtype=np.int64),
            np.ascontiguousarray(power, dtype=np.int64),
            np.ascontiguousarray(quad, dtype=np.int64),
            np.ascontiguousarray(lo, dtype=dt),
            np.ascontiguousarray(span, dtype=dt),
            np.ascontiguousarray(mean, dtype=dt),
            np.ascontiguousarray(std, dtype=dt),
            np.ascontiguousarray(w, dtype=dt),
        )
    )


@njit(parallel=True, cache=True, nogil=True)
def _sincos_axis_njit(z, w):
    """``(len(w), 2, n)``: ``sin(w[f] * z)`` and ``cos(w[f] * z)`` of every angle weight in one pass over ``z``; ``w`` has the dtype of ``z``."""
    f = w.shape[0]
    n = z.shape[0]
    out = np.empty((f, 2, n), dtype=z.dtype)
    for i in prange(n):
        zi = z[i]
        for q in range(f):
            ang = w[q] * zi
            out[q, 0, i] = np.sin(ang)
            out[q, 1, i] = np.cos(ang)
    return out


def sincos_axis_host(z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``(len(weights), 2, n)`` sine and cosine legs for the angle weights ``2 pi freq`` on the axis ``z``: one pass instead of an angle array and two ufunc passes per frequency. The weights are rounded once to the dtype of ``z``, as a Python float combined with an array of that dtype is."""
    zc = np.ascontiguousarray(z)
    return np.asarray(_sincos_axis_njit(zc, np.ascontiguousarray(weights, dtype=zc.dtype)))


#: Rows from which the fused parallel pass beats two numpy ufunc passes for double-precision axes (measured at 100000 rows; the thread launch dominates below).
_SINCOS_MIN_ROWS = 20_000


def sincos_legs(z: np.ndarray, freq: float) -> "tuple[np.ndarray, np.ndarray]":
    """``(sin(2 pi freq z), cos(2 pi freq z))``. A double-precision axis of at least ``_SINCOS_MIN_ROWS`` rows takes the fused parallel pass (equal to numpy's values, 3.9x faster at 100000 rows, six frequencies); a single-precision axis keeps the numpy ufuncs, whose SIMD single-precision sine and cosine are 2.2x faster than the numba loop."""
    if z.dtype == np.float64 and z.shape[0] >= _SINCOS_MIN_ROWS:
        both = sincos_axis_host(z, np.asarray([2.0 * np.pi * float(freq)]))
        return both[0, 0], both[0, 1]
    ang = 2.0 * np.pi * freq * z
    return np.sin(ang), np.cos(ang)
