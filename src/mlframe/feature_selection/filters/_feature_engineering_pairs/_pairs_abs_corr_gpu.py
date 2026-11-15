"""Device twin of ``_abs_corr_finite_njit`` for the deferred-float GPU pair path.

The pair emit step scores each leader column's |corr(y)| for the linear-usability tie-break. On the deferred-float path the candidate column only exists on
the device, so the host formulation copied the whole ``(n,)`` column back (25 columns x 120 KB per small fit, growing with n) to reduce it to ONE scalar.
This computes the same statistic on the device and returns just that scalar.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, cast

import numpy as np

from .._kernel_types import render
from .._raw_module_cache import RawModuleCache
from ._pairs_common import _DEGENERATE_REL_TOL

logger = logging.getLogger(__name__)

# One block per call: pass 1 reduces the joint-finite row count, the two sums (for the means) and the two maxima of |value|; pass 2 reduces the centred sums of squares and the cross product. Both
# passes use the fixed tree below, so the value is the same on every run. It replaces about twenty small cupy operations (3.0 ms a call at 40000 rows, 5792 calls in a child fit of the strict fit
# profile) by one launch and one read of six doubles; the formulas are those of the cupy chain it keeps as ``_abs_corr_finite_cupy`` (the reference of the equivalence test and the fallback).
_SRC = render(
    r"""
extern "C" __global__ void abs_corr_finite_stats(const {REAL}* __restrict__ a, const {ACC}* __restrict__ y, const unsigned char* __restrict__ fin, const long long n, {ACC}* __restrict__ out) {
    __shared__ {ACC} red[3][256];
    const int t = threadIdx.x;
    {ACC} cnt = 0.0, sa = 0.0, sy = 0.0, amax = 0.0, ymax = 0.0;
    for (long long i = t; i < n; i += blockDim.x) {
        const {ACC} av = ({ACC})a[i];
        if (fin[i] != 0 && isfinite(av)) {
            const {ACC} yv = y[i];
            cnt += 1.0; sa += av; sy += yv;
            amax = fmax(amax, fabs(av)); ymax = fmax(ymax, fabs(yv));
        }
    }
    red[0][t] = sa; red[1][t] = sy; red[2][t] = cnt;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) { red[0][t] += red[0][t + s]; red[1][t] += red[1][t + s]; red[2][t] += red[2][t + s]; }
        __syncthreads();
    }
    const {ACC} n_f = red[2][0];
    const {ACC} ma = n_f > 0.0 ? red[0][0] / n_f : 0.0, my = n_f > 0.0 ? red[1][0] / n_f : 0.0;
    __syncthreads();
    // the two maxima ride in the same shared array after the sums have been consumed
    red[0][t] = amax; red[1][t] = ymax;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) { red[0][t] = fmax(red[0][t], red[0][t + s]); red[1][t] = fmax(red[1][t], red[1][t + s]); }
        __syncthreads();
    }
    const {ACC} amax_all = red[0][0], ymax_all = red[1][0];
    __syncthreads();
    {ACC} saa = 0.0, syy = 0.0, say = 0.0;
    for (long long i = t; i < n; i += blockDim.x) {
        const {ACC} av = ({ACC})a[i];
        if (fin[i] != 0 && isfinite(av)) {
            const {ACC} da = av - ma, dy = y[i] - my;
            saa += da * da; syy += dy * dy; say += da * dy;
        }
    }
    red[0][t] = saa; red[1][t] = syy; red[2][t] = say;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) { red[0][t] += red[0][t + s]; red[1][t] += red[1][t + s]; red[2][t] += red[2][t + s]; }
        __syncthreads();
    }
    if (t == 0) { out[0] = n_f; out[1] = red[0][0]; out[2] = red[1][0]; out[3] = red[2][0]; out[4] = amax_all; out[5] = ymax_all; }
}
""",
)
_MODULES = {"float": RawModuleCache(_SRC.replace("{REAL}", "float")), "double": RawModuleCache(_SRC.replace("{REAL}", "double"))}


def kernel(cp: Any, real: Any) -> Any:
    """``abs_corr_finite_stats`` for columns of dtype ``real`` (float32 or float64); compile point of :func:`abs_corr_finite_gpu` and of the register-pressure registry."""
    return _MODULES["float" if np.dtype(real) == np.float32 else "double"].get(cp).get_function("abs_corr_finite_stats")


def _finish(stats: Any, min_n: int) -> float:
    """The degenerate handling and the ratio, from ``(n, sum da^2, sum dy^2, sum da dy, max |a|, max |y|)``."""
    cnt, va, vy, cay, amax, ymax = (float(v) for v in stats)
    if cnt < min_n:
        return 0.0
    n_i = int(cnt)
    if va <= n_i * (_DEGENERATE_REL_TOL * amax) ** 2 or vy <= n_i * (_DEGENERATE_REL_TOL * ymax) ** 2:
        return 0.0
    denom = (va * vy) ** 0.5
    if denom <= 0.0:
        return 0.0
    return float(abs(cay / denom))


def abs_corr_finite_gpu(a_dev: Any, y: np.ndarray, y_finite: np.ndarray, min_n: int = 8) -> float:
    """|Pearson corr| of the device column ``a_dev`` against the host target ``y`` over rows where both are finite; same two-pass arithmetic and degenerate
    handling as ``_abs_corr_finite_njit`` (``0.0`` below ``min_n`` joint-finite rows or for a near-constant side). Only scalars cross to the host.

    One kernel launch and one read of six doubles; any fault of the kernel path runs the cupy chain (:func:`_abs_corr_finite_cupy`) instead."""
    import cupy as cp

    from mlframe.feature_selection.filters._fe_resident_operands import resident_operand

    try:
        # Intentional float64: the target is reduced in float64 (a target at offset ~1e7 keeps no variance in float32).
        y_dev = resident_operand(y, ("abs_corr_y",), dtype=np.float64).ravel()
        fin_dev = resident_operand(np.asarray(y_finite, dtype=np.uint8), ("abs_corr_yfin",), dtype=np.uint8).ravel()
        a = a_dev.ravel() if a_dev.dtype in (cp.float32, cp.float64) else a_dev.astype(np.float64).ravel()  # Intentional float64: any other dtype widens to double
        a = cp.ascontiguousarray(a)
        out = cp.empty(6, dtype=cp.float64)  # Intentional float64: the six reduced sums
        kern = kernel(cp, a.dtype)
        kern((1,), (256,), (a, y_dev, fin_dev, np.int64(a.shape[0]), out))  # launch-ok: one block per call, 256 threads reduce both passes in a fixed tree
        return _finish(out.get(), min_n)
    except Exception as exc:
        logger.debug("abs_corr_finite kernel failed (%s: %s); cupy chain", type(exc).__name__, exc)
        return _abs_corr_finite_cupy(a_dev, y, y_finite, min_n)


def _abs_corr_finite_cupy(a_dev: Any, y: np.ndarray, y_finite: np.ndarray, min_n: int = 8) -> float:
    """The cupy-operation form of :func:`abs_corr_finite_gpu` (about twenty small launches): the reference of the equivalence test and the fallback."""
    import cupy as cp

    from mlframe.feature_selection.filters._fe_resident_operands import resident_operand

    # Intentional float64: the host twin reduces in float64; a target at offset ~1e7 keeps no variance in float32.
    y_dev = resident_operand(y, ("abs_corr_y",), dtype=np.float64).ravel()
    fin_dev = resident_operand(np.asarray(y_finite, dtype=np.uint8), ("abs_corr_yfin",), dtype=np.uint8).ravel() != 0
    a = a_dev.astype(np.float64, copy=False).ravel()  # Intentional float64: offset data (~1e7) keeps no variance in float32 (test offset=1e7 case off by 1.2e-2)
    # Masked sums instead of boolean-index compaction: ``a[mask]`` and ``int(count_nonzero)`` each block the stream on a device read, and this runs per
    # candidate. Everything stays on the device and ONE small vector comes back at the end.
    mask = fin_dev & cp.isfinite(a)
    # Intentional float64: a float32 row count stops being exact above 2**24 rows.
    m = mask.astype(np.float64)
    n = m.sum()
    a0 = cp.where(mask, a, 0.0)
    y0 = cp.where(mask, y_dev, 0.0)
    da = (a0 - (a0 * m).sum() / n) * m
    dy = (y0 - (y0 * m).sum() / n) * m
    stats = cp.stack([n, (da * da).sum(), (dy * dy).sum(), (da * dy).sum(), cp.abs(a0).max(), cp.abs(y0).max()]).get()
    cnt, va, vy, cay, amax, ymax = (float(v) for v in stats)
    if cnt < min_n:
        return 0.0
    n_i = int(cnt)
    if va <= n_i * (_DEGENERATE_REL_TOL * amax) ** 2 or vy <= n_i * (_DEGENERATE_REL_TOL * ymax) ** 2:
        return 0.0
    denom = (va * vy) ** 0.5
    if denom <= 0.0:
        return 0.0
    return float(abs(cay / denom))


def abs_corr_or_none(a_dev: Any, y: Optional[np.ndarray], y_finite: Optional[np.ndarray]) -> Optional[float]:
    """``abs_corr_finite_gpu`` with the host helper's contract: ``0.0`` when there is no target or the lengths differ, and ``None`` on any device fault so
    the caller falls back to the host column."""
    if y is None or y_finite is None:
        return 0.0
    if int(a_dev.shape[0]) != int(y.shape[0]):
        return 0.0
    try:
        return abs_corr_finite_gpu(a_dev, y, y_finite, 8)
    except Exception as e:
        logger.debug("device |corr| failed, caller falls back to the host column: %s", e)
        return None


def candidate_abs_corr(resolve_col: Any, safe_abs_corr: Any, buf_col: int) -> Optional[float]:
    """|corr(y)| of candidate column ``buf_col`` via the device capability of a deferred-float ``resolve_col`` and the target held by ``safe_abs_corr``,
    or ``None`` when either is absent (host buffers, no target) or the device path faulted - the caller then reads the host column."""
    fn = getattr(resolve_col, "abs_corr", None)
    target = getattr(safe_abs_corr, "target", None)
    if fn is None or target is None or target[0] is None:
        return None
    return cast("Optional[float]", fn(buf_col, target[0], target[1]))


def abs_corr_zerofill_gpu(a_dev: Any, b: np.ndarray) -> float:
    """Device twin of ``_abs_corr_zerofill_njit``: |Pearson corr| of the device column ``a_dev`` against the host column ``b`` with every non-finite entry
    read as 0 (the zero-fill statistic the degenerate-pair veto was written against, NOT the masked one). Only scalars cross to the host."""
    import cupy as cp

    from mlframe.feature_selection.filters._fe_resident_operands import resident_operand

    # Intentional float64: operands at offset ~1e7 keep no variance in float32 (the host twin reduces in float64).
    a = a_dev.astype(np.float64, copy=False).ravel()
    # Intentional float64: the second operand is reduced in float64 for the same offset reason as the first.
    bd = resident_operand(np.ascontiguousarray(np.asarray(b, dtype=np.float64)), ("zerofill_operand_f64",), dtype=np.float64).ravel()
    n = int(a.shape[0])
    if n == 0:
        return 0.0
    a = cp.where(cp.isfinite(a), a, 0.0)
    bd = cp.where(cp.isfinite(bd), bd, 0.0)
    da = a - a.sum() / n
    db = bd - bd.sum() / n
    stats = cp.stack([(da * da).sum(), (db * db).sum(), (da * db).sum(), cp.abs(a).max(), cp.abs(bd).max()]).get()
    va, vb, cab, amax, bmax = (float(v) for v in stats)
    if va <= n * (_DEGENERATE_REL_TOL * amax) ** 2 or vb <= n * (_DEGENERATE_REL_TOL * bmax) ** 2:
        return 0.0
    denom = (va * vb) ** 0.5
    if denom <= 0.0:
        return 0.0
    return float(abs(cab / denom))
