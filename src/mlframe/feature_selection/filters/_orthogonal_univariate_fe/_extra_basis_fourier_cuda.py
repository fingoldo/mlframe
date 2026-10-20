"""One device launch for a whole block of Fourier / chirp basis columns.

``_extra_basis_resident._fourier_col_gpu`` builds one column per call from about eight elementwise cupy operations, and the caller stacks the columns with one copy launch each; a pool of
several hundred engineered Fourier names is then a few thousand tiny launches. This kernel evaluates every name in one pass over a ``(names, rows)`` block: ``name j`` reads source column
``src[j]`` of the stacked resident operands, forms the axis ``z`` (the linear power axis ``(x**p - lo) / span`` or the chirp axis ``(sign(zc) * zc**2 - lo) / span`` with ``zc = (x - mean) / std``)
and writes ``sin|cos((2 pi freq) * z)``.

The arithmetic is that of the per-column function operation by operation, in the element type of the operand column, with the scalar parameters rounded to that type as cupy rounds a Python float
combined with an array of that type; the one product followed by a subtraction (the chirp axis) uses a rounded multiply so it is not contracted into a fused multiply-add, as two separate elementwise
kernels would not. (Compiling the whole kernel without fused multiply-add is not equivalent: it changes the device ``sin`` of double precision.) The columns are bit-identical to the per-column build.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_SRC = r"""
#define MUL_RN {MUL_RN}
extern "C" __global__ void fourier_block(const {REAL}* __restrict__ xs, const int* __restrict__ src, const int* __restrict__ cosine, const int* __restrict__ power,
                                         const int* __restrict__ quad, const {REAL}* __restrict__ lo, const {REAL}* __restrict__ span, const {REAL}* __restrict__ mean,
                                         const {REAL}* __restrict__ std, const {REAL}* __restrict__ w, const long long n, const int K, {REAL}* __restrict__ out) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)K * n) return;
    const int j = (int)(idx / n);
    const long long i = idx - (long long)j * n;
    const {REAL} x = xs[(long long)src[j] * n + i];
    {REAL} z;
    if (quad[j]) {
        const {REAL} zc = (x - mean[j]) / std[j];
        const {REAL} sgn = (zc > ({REAL})0) - (zc < ({REAL})0);
        const {REAL} u = MUL_RN(sgn, zc * zc);  // rounded product: not contracted with the subtraction below, as in two separate kernels
        z = (u - lo[j]) / span[j];
    } else {
        const int p = power[j];
        {REAL} xp = x;
        if (p != 1) xp = pow(x, ({REAL})p);
        z = (xp - lo[j]) / span[j];
    }
    const {REAL} ang = w[j] * z;
    out[idx] = cosine[j] ? cos(ang) : sin(ang);
}
"""

_kernels: dict = {}
_LOCK = threading.Lock()


def _kernel(cp: Any, real: str) -> Any:
    """The compiled kernel for one element type, built once per process."""
    with _LOCK:
        kern = _kernels.get(real)
        if kern is None:
            source = _SRC.replace("{MUL_RN}", "__fmul_rn" if real == "float" else "__dmul_rn").replace("{REAL}", real)
            kern = cp.RawModule(code=source).get_function("fourier_block")
            _kernels[real] = kern
        return kern


def _axis_params(m: dict) -> "tuple[float, float, float, float, int, int]":
    """``(lo, span, mean, std, power, quad)`` of one name's axis with the guards of ``_fourier_col_gpu`` (span and std floored at 1e-12)."""
    lo = float(m["lo"])
    span = float(m["span"])
    span = span if span > 1e-12 else 1e-12
    if m.get("arg") == "quadratic":
        std = float(m["std"])
        std = std if std > 1e-12 else 1e-12
        return lo, span, float(m["mean"]), std, 1, 1
    return lo, span, 0.0, 1.0, int(m.get("power", 1)), 0


def fourier_block_cuda(cp: Any, columns: "list[Any]", metas: "list[dict]") -> Optional[Any]:
    """``(len(metas), n)`` block of the Fourier / chirp columns, row ``j`` built from the device column ``columns[j]`` with ``metas[j]``, or ``None`` when the kernel is unavailable.

    ``columns`` are resident ``(n,)`` operands of one floating dtype (single or double precision); the block is in that dtype.
    """
    try:
        if not columns:
            return None
        dtype = columns[0].dtype
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or any(c.dtype != dtype for c in columns):  # Intentional float64: the two element types the kernel is templated for
            return None
        n = int(columns[0].shape[0])
        if any(int(c.shape[0]) != n for c in columns):
            return None
        index: dict = {}
        stacked: list = []
        src = np.empty(len(metas), dtype=np.int32)
        for j, col in enumerate(columns):
            slot = index.get(id(col))
            if slot is None:
                slot = len(stacked)
                index[id(col)] = slot
                stacked.append(col)
            src[j] = slot
        params = np.array([_axis_params(m) for m in metas], dtype=np.float64)  # Intentional float64: axis parameters are formed in double precision and rounded once to the element type
        cosine = np.array([1 if m.get("kind") != "sin" else 0 for m in metas], dtype=np.int32)
        # Intentional float64: the angle weight 2 pi freq is formed in double precision (as in the per-column function) and rounded once to the element type
        weight = np.array([(2.0 * math.pi) * float(m["freq"]) for m in metas], dtype=np.float64)
        real = "float" if dtype == np.dtype(np.float32) else "double"
        k = len(metas)
        xs = cp.stack(stacked)
        out = cp.empty((k, n), dtype=dtype)

        def up(a: np.ndarray) -> Any:
            """The array on the device in the element type of the columns."""
            return cp.asarray(np.ascontiguousarray(a.astype(dtype)))

        threads = 256
        _kernel(cp, real)(
            (((k * n) + threads - 1) // threads,),
            (threads,),
            (xs, cp.asarray(src), cp.asarray(cosine), cp.asarray(params[:, 4].astype(np.int32)), cp.asarray(params[:, 5].astype(np.int32)), up(params[:, 0]), up(params[:, 1]), up(params[:, 2]), up(params[:, 3]), up(weight), np.int64(n), np.int32(k), out),
        )
        return out
    except Exception as exc:  # nosec B110 - any device or compile failure falls back to the per-column build
        logger.debug("fourier_block_cuda unavailable (%s: %s); using the per-column build", type(exc).__name__, exc)
        return None
