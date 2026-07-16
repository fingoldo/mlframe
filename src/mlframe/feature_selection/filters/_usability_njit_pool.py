"""Fused njit(parallel) per-pair MI enumeration for the usability-aware retention pool.

The retention path of :func:`build_usability_candidate_pool` (``rank_pairs_by_joint_mi=True``)
enumerates, for each of the top joint-MI pairs, ``|unary|^2 * |binary|`` forms, computing for each
``MI(binary(unary_a(x1), unary_b(x2)); y)``. The original loop Python-dispatches numpy for every
combo's value + quantile-bin + plug-in MI -- ~3.5s/pair at n=10000, ~62s of a structured fit.

This module fuses that triple into ONE njit kernel: for each ``(ua_code, ub_code, bn_code)`` combo,
compute the value in nopython (float32 semantics, matching the numpy bin_funcs bit-for-bit), skip if
``std<=1e-9`` (sentinel ``-1.0``), else quantile-bin (MATCHING ``_quantile_bin``'s numpy edges +
searchsorted) and compute the marginal Miller-Madow MI vs ``y_codes`` (MATCHING
``marginal_mi_binned_fixed_y``). The Python caller then ranks by MI, applies the unchanged diversity
filter, and rebuilds recipes only for the kept few -- so what is RECOVERED is identical to the
Python loop (verified: per-combo MI matches to ~6e-15, value to 0.0).

CORRECTNESS (the pool replay-VERIFIES every kept recipe with ``np.allclose atol=1e-4`` against the
Python unary): all 17 ``medium`` unaries (identity,neg,abs,sqr,reciproc,sqrt,log,sin,sign,rint,
qubed,invsquared,invqubed,cbrt,invcbrt,invsqrt,exp) and all 6 ``minimal`` binaries (mul,add,sub,div,
max,min) reproduce the numpy path to 0.0 abs diff in float32 (measured over normal/exponential/
heavy-tail/integer-tie operands). ``smart_log``'s data-dependent additive shift is matched exactly by
passing each operand's ``nanmin`` into the kernel (the shift is ``1e-5 - x_min`` when ``x_min<=0``).
If a future preset adds a unary/binary that is NOT in the code tables, the caller falls back to the
Python loop for the WHOLE pool (``njit_op_codes_or_none`` returns None) -- correctness first.

THREE kernel versions are kept (``feedback_keep_all_kernel_versions``): a SERIAL njit, a
``parallel=True`` (prange over combos) njit twin, and a cupy GPU twin (``_pair_combo_mi_cupy``, which
vectorises the value+bin+MI across a chunk of combos on device). They are dispatched per-host by
(n_rows, n_combos) via the canonical ``kernel_tuning_cache`` (NO hardcoded threshold --
``feedback_use_kernel_tuning_cache_for_gpu``). CPU njit is the DEFAULT and the FALLBACK: any missing
cupy / global-GPU-off / device error routes back to CPU and the fit is never broken. Force a backend
for testing via ``MLFRAME_USABILITY_POOL_BACKEND=njit|njit_parallel|gpu`` (mirrors ``MLFRAME_MI_BACKEND``).
The GPU per-combo MI is bit-faithful to the CPU kernels to fp64 round-off (~1e-15) -- same lerp
quantile edges, same Miller-Madow plug-in MI -- so the recovered forms are unchanged.
"""
from __future__ import annotations

import os

import numpy as np
from numba import njit, prange

from pyutilz.performance.kernel_tuning.registry import kernel_tuner

# Optional GPU dep. The dispatcher gracefully falls back to the CPU njit kernels when cupy is
# missing / the device errors -- the fit is NEVER broken by a GPU problem (correctness first).
try:
    import cupy as _cp
    _CUPY_AVAIL = True
except Exception:
    _cp = None
    _CUPY_AVAIL = False


# Unary op-code table for the ``medium`` preset (registry order is irrelevant -- the caller maps each
# preset name to its code and falls back to Python if any name is absent). Every entry below is
# bit-exact (float32) to ``create_unary_transformations(preset='medium')[name]``; see ``_apply_unary``.
_NJIT_UNARY_OP_CODES: dict = {
    "identity": 0, "neg": 1, "abs": 2, "sqr": 3, "reciproc": 4, "sqrt": 5,
    "log": 6, "sin": 7, "sign": 8, "rint": 9, "qubed": 10, "invsquared": 11,
    "invqubed": 12, "cbrt": 13, "invcbrt": 14, "invsqrt": 15, "exp": 16,
}

# Binary op-code table for the ``minimal`` preset (the pool's default binary set). Bit-exact to
# ``create_binary_transformations(preset='minimal')`` -- see ``_apply_binary``; ``div`` mirrors
# ``_safe_div`` (exact x/y for y!=0, 1e-9 floor only on an exact-zero denominator).
_NJIT_USABILITY_BINARY_OP_CODES: dict = {
    "mul": 0, "add": 1, "sub": 2, "div": 3, "max": 4, "min": 5,
}


def njit_unary_codes_or_none(unary_names) -> "np.ndarray | None":
    """Map each unary name (preset iteration order) to its op-code, or None if ANY name is not
    njit-coded (the caller then uses the per-candidate numpy path -- correctness first)."""
    codes = []
    for name in unary_names:
        c = _NJIT_UNARY_OP_CODES.get(name)
        if c is None:
            return None
        codes.append(c)
    return np.asarray(codes, dtype=np.int64)


def njit_binary_codes_or_none(binary_names) -> "np.ndarray | None":
    """Map each binary name to its op-code, or None if ANY name is not njit-coded."""
    codes = []
    for name in binary_names:
        c = _NJIT_USABILITY_BINARY_OP_CODES.get(name)
        if c is None:
            return None
        codes.append(c)
    return np.asarray(codes, dtype=np.int64)


@njit(cache=True, inline="always")
def _apply_unary(v, code, xmin):
    """One element of the medium-preset unary, in float64 (the Python path applies the numpy unary to
    the float64 operand). ``xmin`` is the operand column's ``nanmin`` -- only ``log`` (smart_log) uses
    it (additive shift ``1e-5 - xmin`` when ``xmin<=0``, exact ``log`` when ``xmin>0``)."""
    if code == 0:  # identity
        return v
    elif code == 1:  # neg
        return -v
    elif code == 2:  # abs
        return abs(v)
    elif code == 3:  # sqr = np.power(x, 2)
        return v * v
    elif code == 4:  # reciproc = np.power(x, -1)
        return 1.0 / v
    elif code == 5:  # sqrt = np.sqrt(np.abs(x))
        return np.sqrt(abs(v))
    elif code == 6:  # log = smart_log
        if xmin > 0.0:
            return np.log(v)
        return np.log(v + (1e-5 - xmin))
    elif code == 7:  # sin
        return np.sin(v)
    elif code == 8:  # sign = np.sign
        return 0.0 if v == 0.0 else (1.0 if v > 0.0 else -1.0)
    elif code == 9:  # rint
        return np.rint(v)
    elif code == 10:  # qubed = np.power(x, 3)
        return v * v * v
    elif code == 11:  # invsquared = np.power(x, -2)
        return 1.0 / (v * v)
    elif code == 12:  # invqubed = np.power(x, -3)
        return 1.0 / (v * v * v)
    elif code == 13:  # cbrt
        return np.cbrt(v)
    elif code == 14:  # invcbrt = np.power(x, -1/3)
        return v ** (-1.0 / 3.0)
    elif code == 15:  # invsqrt = np.power(x, -1/2)
        return v ** (-0.5)
    else:  # code == 16: exp
        return np.exp(v)


@njit(cache=True, inline="always")
def _apply_binary(a, b, bn):
    """One element of the minimal-preset binary, returning the float32-scrubbed value cast back to
    float64 (the Python path does ``_scrub(bf(ta, tb), float32)`` then upcasts to float64 for binning).
    Matches numpy's nan-propagating max/min and ``_safe_div`` exactly."""
    if bn == 0:  # mul
        v = a * b
    elif bn == 1:  # add
        v = a + b
    elif bn == 2:  # sub
        v = a - b
    elif bn == 3:  # div = _safe_div: exact a/b for b!=0, 1e-9 floor on exact-zero denominator
        v = a / (1e-9 if b == 0.0 else b)
    elif bn == 4:  # max = np.maximum (nan-propagating)
        if a != a or b != b:
            v = a + b
        else:
            v = a if a > b else b
    else:  # bn == 5: min = np.minimum (nan-propagating)
        if a != a or b != b:
            v = a + b
        else:
            v = a if a < b else b
    # np.nan_to_num(nan=0, posinf=0, neginf=0) at float32 precision (feature_dtype)
    f = np.float32(v)
    if not (f == f and f != np.inf and f != -np.inf):
        f = np.float32(0.0)
    return np.float64(f)


@njit(cache=True, inline="always")
def _qbin_into(val, nbins, qs, codes_out):
    """Equi-frequency quantile-bin ``val`` (finite float64) into ``codes_out``, returning the bin
    cardinality. BIT-IDENTICAL to ``_quantile_bin``'s all-finite fast path: the same ``np.quantile``
    linear-interpolation edges (numpy ``_lerp`` form ``lo + (hi-lo)*frac`` at virtual index
    ``q*(n-1)``), the same ``np.unique`` edge dedup, the same ``searchsorted(edges[1:-1], side='right')``.
    The kernel value is already scrubbed finite, so only the finite fast path is needed."""
    n = val.shape[0]
    nq = qs.shape[0]
    # Only the order statistics at the lerp anchors (lo, lo+1 of each quantile) are needed, so np.partition
    # at exactly those ~2*nq positions (O(n) introselect) replaces the O(n log n) full np.sort. part[lo] /
    # part[hi] are the identical order statistics np.sort gives, so q/edges/codes are BIT-IDENTICAL.
    kths = np.empty(2 * nq, dtype=np.int64)
    los = np.empty(nq, dtype=np.int64)
    fracs = np.empty(nq, dtype=np.float64)
    m2 = 0
    for k in range(nq):
        pos = qs[k] * (n - 1)
        lo = int(np.floor(pos))
        hi = lo + 1 if lo < n - 1 else lo
        los[k] = lo
        fracs[k] = pos - lo
        kths[m2] = lo
        kths[m2 + 1] = hi
        m2 += 2
    part = np.partition(val, kths[:m2])
    q = np.empty(nq, dtype=np.float64)
    for k in range(nq):
        lo = los[k]
        hi = lo + 1 if lo < n - 1 else lo
        q[k] = part[lo] + (part[hi] - part[lo]) * fracs[k]
    # np.unique(q) -- q is ascending, so dedup adjacent equals in one pass.
    edges = np.empty(nq, dtype=np.float64)
    m = 0
    for k in range(nq):
        if m == 0 or q[k] != edges[m - 1]:
            edges[m] = q[k]
            m += 1
    if m <= 2:
        if m == 2:
            for i in range(n):
                codes_out[i] = 1 if val[i] >= edges[1] else 0
            return 2
        for i in range(n):
            codes_out[i] = 0
        return 1
    # inner edges = edges[1:m-1]; searchsorted right via bisect.
    ni = m - 2
    kx = 0
    for i in range(n):
        v = val[i]
        lo = 0
        hi = ni
        while lo < hi:
            mid = (lo + hi) // 2
            if v < edges[1 + mid]:
                hi = mid
            else:
                lo = mid + 1
        codes_out[i] = lo
        if lo > kx:
            kx = lo
    return kx + 1


@njit(cache=True, inline="always")
def _marginal_mi_njit(xb, kx, y_codes, h_y, k_y, n):
    """Plug-in Miller-Madow ``MI(X;Y) = H(X)+H(Y)-H(X,Y)`` minus ``(k_x+k_y-k_xy-1)/(2n)``, clamped
    to >=0 -- bit-faithful to ``marginal_mi_binned_fixed_y`` (same plug-in entropies, same bias, same
    natural log). ``xb`` are dense 0..kx-1 bin codes, ``y_codes`` dense 0..k_y-1; the joint id is
    ``xb*k_y + y`` (a bijection onto the occupied cells)."""
    invn = 1.0 / n
    cx = np.zeros(kx, dtype=np.int64)
    for i in range(n):
        cx[xb[i]] += 1
    hx = 0.0
    kx_occ = 0
    for c in cx:
        if c > 0:
            p = c * invn
            hx -= p * np.log(p)
            kx_occ += 1
    cj = np.zeros(kx * k_y, dtype=np.int64)
    for i in range(n):
        cj[xb[i] * k_y + y_codes[i]] += 1
    hxy = 0.0
    kxy = 0
    for c in cj:
        if c > 0:
            p = c * invn
            hxy -= p * np.log(p)
            kxy += 1
    mi = hx + h_y - hxy
    bias = (kx_occ + k_y - kxy - 1) / (2.0 * n)
    r = mi - bias
    return r if r > 0.0 else 0.0


@njit(cache=True)
def _pair_combo_mi_njit(x1, x2, y_codes, h_y, k_y, qs, ua_arr, ub_arr, bn_arr, xmin_a, xmin_b):
    """SERIAL: for each combo ``(ua_arr[j], ub_arr[j], bn_arr[j])`` compute the value column, the
    ``std<=1e-9`` sentinel, and the marginal MI vs ``y_codes``. Returns the per-combo MI (or -1.0
    sentinel). The ``parallel=True`` twin below is byte-identical (each combo writes only ``out[j]``)."""
    nc = ua_arr.shape[0]
    n = x1.shape[0]
    nbins = qs.shape[0] - 1
    out = np.empty(nc, dtype=np.float64)
    for j in range(nc):
        ua = ua_arr[j]
        ub = ub_arr[j]
        bn = bn_arr[j]
        val = np.empty(n, dtype=np.float64)
        s = 0.0
        ss = 0.0
        for i in range(n):
            a = _apply_unary(x1[i], ua, xmin_a)
            b = _apply_unary(x2[i], ub, xmin_b)
            v = _apply_binary(a, b, bn)
            val[i] = v
            s += v
            ss += v * v
        mean = s / n
        var = ss / n - mean * mean
        if var <= 1e-18:  # std <= 1e-9
            out[j] = -1.0
            continue
        codes = np.empty(n, dtype=np.int64)
        kx = _qbin_into(val, nbins, qs, codes)
        out[j] = _marginal_mi_njit(codes, kx, y_codes, h_y, k_y, n)
    return out


@njit(parallel=True, cache=True)
def _pair_combo_mi_njit_parallel(x1, x2, y_codes, h_y, k_y, qs, ua_arr, ub_arr, bn_arr, xmin_a, xmin_b):
    """``parallel=True`` (prange over COMBOS) twin of :func:`_pair_combo_mi_njit` -- BYTE-IDENTICAL
    output (each combo ``j`` reads only ``x1``/``x2`` and writes only ``out[j]``; zero cross-combo
    dependence -> result independent of thread count). Kept separate per ``feedback_keep_all_kernel_versions``."""
    nc = ua_arr.shape[0]
    n = x1.shape[0]
    nbins = qs.shape[0] - 1
    out = np.empty(nc, dtype=np.float64)
    for j in prange(nc):
        ua = ua_arr[j]
        ub = ub_arr[j]
        bn = bn_arr[j]
        val = np.empty(n, dtype=np.float64)
        s = 0.0
        ss = 0.0
        for i in range(n):
            a = _apply_unary(x1[i], ua, xmin_a)
            b = _apply_unary(x2[i], ub, xmin_b)
            v = _apply_binary(a, b, bn)
            val[i] = v
            s += v
            ss += v * v
        mean = s / n
        var = ss / n - mean * mean
        if var <= 1e-18:
            out[j] = -1.0
            continue
        codes = np.empty(n, dtype=np.int64)
        kx = _qbin_into(val, nbins, qs, codes)
        out[j] = _marginal_mi_njit(codes, kx, y_codes, h_y, k_y, n)
    return out


# UNARY-TABLE kernels (2026-06-21, ``feedback_keep_all_kernel_versions``). The per-combo loop above
# recomputes ``_apply_unary(x1[i], ua)`` and ``_apply_unary(x2[i], ub)`` for EVERY combo -- so each
# distinct unary (sin/exp/log/cbrt/... -- the transcendental ones are NOT free) is re-evaluated once per
# binary that pairs with it (|binary| times, 6x in the minimal preset). Precomputing the ``nu_tab``
# distinct unary transforms of x1 and x2 ONCE into a ``(nu_tab, n)`` table, then indexing it in the combo
# loop, drops the value step to a single binary op + two table reads per element. The qbin (sort) + MI
# are unchanged, so the per-combo MI is BIT-IDENTICAL (verified max abs diff 0.0 over the medium/minimal
# preset). Measured n=3000, 1734 combos: parallel 107ms -> 76ms (~1.4x). ``nu_tab`` = max op-code+1 so the
# table is indexed directly by the op-code stored in ``ua_arr``/``ub_arr`` (the kernel reads U1[ua_arr[j]]).
@njit(cache=True)
def _pair_combo_mi_njit_table(x1, x2, y_codes, h_y, k_y, qs, ua_arr, ub_arr, bn_arr, xmin_a, xmin_b, nu_tab):
    """SERIAL unary-table twin of :func:`_pair_combo_mi_njit` -- BIT-IDENTICAL output, precomputes the
    ``nu_tab`` distinct unary transforms of each operand once into a (nu_tab, n) table."""
    nc = ua_arr.shape[0]
    n = x1.shape[0]
    nbins = qs.shape[0] - 1
    U1 = np.empty((nu_tab, n), dtype=np.float64)
    U2 = np.empty((nu_tab, n), dtype=np.float64)
    for u in range(nu_tab):
        for i in range(n):
            U1[u, i] = _apply_unary(x1[i], u, xmin_a)
            U2[u, i] = _apply_unary(x2[i], u, xmin_b)
    out = np.empty(nc, dtype=np.float64)
    for j in range(nc):
        ua = ua_arr[j]
        ub = ub_arr[j]
        bn = bn_arr[j]
        val = np.empty(n, dtype=np.float64)
        s = 0.0
        ss = 0.0
        for i in range(n):
            v = _apply_binary(U1[ua, i], U2[ub, i], bn)
            val[i] = v
            s += v
            ss += v * v
        mean = s / n
        var = ss / n - mean * mean
        if var <= 1e-18:
            out[j] = -1.0
            continue
        codes = np.empty(n, dtype=np.int64)
        kx = _qbin_into(val, nbins, qs, codes)
        out[j] = _marginal_mi_njit(codes, kx, y_codes, h_y, k_y, n)
    return out


@njit(parallel=True, cache=True)
def _pair_combo_mi_njit_table_parallel(x1, x2, y_codes, h_y, k_y, qs, ua_arr, ub_arr, bn_arr, xmin_a, xmin_b, nu_tab):
    """``parallel=True`` unary-table twin -- BIT-IDENTICAL to :func:`_pair_combo_mi_njit_table` (the table
    precompute is parallel over the ``nu_tab`` unaries, then prange over combos; each combo writes only
    ``out[j]``, zero cross-combo dependence). The DEFAULT CPU path on the retention enumeration."""
    nc = ua_arr.shape[0]
    n = x1.shape[0]
    nbins = qs.shape[0] - 1
    U1 = np.empty((nu_tab, n), dtype=np.float64)
    U2 = np.empty((nu_tab, n), dtype=np.float64)
    for u in prange(nu_tab):
        for i in range(n):
            U1[u, i] = _apply_unary(x1[i], u, xmin_a)
            U2[u, i] = _apply_unary(x2[i], u, xmin_b)
    out = np.empty(nc, dtype=np.float64)
    for j in prange(nc):
        ua = ua_arr[j]
        ub = ub_arr[j]
        bn = bn_arr[j]
        val = np.empty(n, dtype=np.float64)
        s = 0.0
        ss = 0.0
        for i in range(n):
            v = _apply_binary(U1[ua, i], U2[ub, i], bn)
            val[i] = v
            s += v
            ss += v * v
        mean = s / n
        var = ss / n - mean * mean
        if var <= 1e-18:
            out[j] = -1.0
            continue
        codes = np.empty(n, dtype=np.int64)
        kx = _qbin_into(val, nbins, qs, codes)
        out[j] = _marginal_mi_njit(codes, kx, y_codes, h_y, k_y, n)
    return out


# ----------------------------------------------------------------------------------------------
# GPU (cupy) fused kernel -- BIT-FAITHFUL to the CPU njit twins.
# ----------------------------------------------------------------------------------------------
# The combo loop is embarrassingly parallel (nc combos x n rows). The cupy path vectorises the
# value computation, the std-sentinel, the quantile-bin and the Miller-Madow marginal MI across a
# CHUNK of combos at once (so a GTX-1050-Ti-class 4GB card never OOMs at n=50000). Bit-faithfulness:
#
#   * value: ``_apply_unary``/``_apply_binary`` are re-expressed in cupy float64 element ops; the
#     binary scrub is the SAME ``np.float32`` round-trip + nan/inf->0 (cupy float32 cast matches
#     numpy's round-half-to-even). smart_log's ``1e-5 - xmin`` shift is passed in identically.
#   * quantile-bin: the SAME np.quantile lerp edges (virtual index ``q*(n-1)``, ``lo+(hi-lo)*frac``),
#     the SAME adjacent-dedup of the sorted-quantile vector, the SAME ``searchsorted(inner, 'right')``.
#     cupy.sort/cupy.searchsorted are bit-identical to numpy for finite float64 (the kernel value is
#     already scrubbed finite). The per-combo bin cardinality ``kx`` is recovered as ``max(code)+1``.
#   * MI: the SAME plug-in entropies (natural log), the SAME Miller-Madow bias
#     ``(kx_occ + k_y - kxy - 1)/(2n)`` and the same clamp-to-0. Joint id = ``xb*k_y + y`` via bincount.
#
# Because every per-combo MI is computed from the identical formula on the identical edges, the
# returned MI array matches the CPU kernel to fp64 round-off (~1e-15), so the RANKING -- and hence
# the recovered forms -- are unchanged. cupy is NEVER put on ``self`` (the pool is transient); only
# module-level singletons (the tuner spec) persist.

# Per-chunk combo budget so the (chunk x n) float64 working set + sort/bincount temporaries stay well
# inside a 4GB card. ~ chunk*n*8 * (a few buffers); 256*50000*8 = ~100MB per buffer.
_GPU_COMBO_CHUNK = 256


def _gpu_apply_unary(x, code, xmin):
    """cupy float64 vectorised twin of :func:`_apply_unary` over the whole column ``x`` (1-D device
    array). ``code`` is a Python int; ``xmin`` the operand's nanmin (only smart_log uses it)."""
    cp = _cp
    if code == 0:  # identity
        return x
    elif code == 1:  # neg
        return -x
    elif code == 2:  # abs
        return cp.abs(x)
    elif code == 3:  # sqr
        return x * x
    elif code == 4:  # reciproc
        return 1.0 / x
    elif code == 5:  # sqrt = sqrt(abs)
        return cp.sqrt(cp.abs(x))
    elif code == 6:  # log = smart_log
        if xmin > 0.0:
            return cp.log(x)
        return cp.log(x + (1e-5 - xmin))
    elif code == 7:  # sin
        return cp.sin(x)
    elif code == 8:  # sign
        return cp.sign(x)
    elif code == 9:  # rint
        return cp.rint(x)
    elif code == 10:  # qubed
        return x * x * x
    elif code == 11:  # invsquared
        return 1.0 / (x * x)
    elif code == 12:  # invqubed
        return 1.0 / (x * x * x)
    elif code == 13:  # cbrt
        return cp.cbrt(x)
    elif code == 14:  # invcbrt
        return x ** (-1.0 / 3.0)
    elif code == 15:  # invsqrt
        return x ** (-0.5)
    else:  # exp
        return cp.exp(x)


def _gpu_apply_binary(a, b, bn):
    """cupy float64 vectorised twin of :func:`_apply_binary` -- including the ``np.float32`` scrub +
    nan/inf->0 at float32 precision -- returning a float64 device array of the scrubbed values."""
    cp = _cp
    if bn == 0:  # mul
        v = a * b
    elif bn == 1:  # add
        v = a + b
    elif bn == 2:  # sub
        v = a - b
    elif bn == 3:  # div = _safe_div (1e-9 floor only on exact-zero denom)
        denom = cp.where(b == 0.0, cp.float64(1e-9), b)
        v = a / denom
    elif bn == 4:  # max = np.maximum (nan-propagating)
        v = cp.maximum(a, b)
    else:  # min = np.minimum (nan-propagating)
        v = cp.minimum(a, b)
    # np.nan_to_num(nan=0, posinf=0, neginf=0) at float32 precision (feature_dtype).
    f = v.astype(cp.float32)
    f = cp.where(cp.isfinite(f), f, cp.float32(0.0))
    return f.astype(cp.float64)


def _gpu_quantile_bin_codes(V, qs):
    """Equi-frequency quantile-bin each ROW of ``V`` (shape ``(m, n)``, finite float64) -- BIT-IDENTICAL
    to :func:`_qbin_into` per row. Returns ``(codes (m,n) int64, kx (m,) int64)``. Vectorised over the
    ``m`` combos in the chunk; the searchsorted is done per row (each row has its own edge vector)."""
    cp = _cp
    m, n = V.shape
    srt = cp.sort(V, axis=1)  # (m, n) ascending -- matches np.sort
    pos = qs * (n - 1)  # virtual indices, (nq,)
    lo = cp.floor(pos).astype(cp.int64)  # (nq,)
    hi = cp.where(lo < n - 1, lo + 1, lo)
    frac = pos - lo
    q = srt[:, lo] + (srt[:, hi] - srt[:, lo]) * frac[None, :]  # (m, nq) lerp edges
    codes = cp.zeros((m, n), dtype=cp.int64)
    kx = cp.ones(m, dtype=cp.int64)
    # Per-combo dedup + searchsorted. nq is tiny (nbins+1, ~11), so the Python loop over rows is the
    # m axis; each row's searchsorted is a fully vectorised cupy call over n.
    for r in range(m):
        qr = q[r]
        # np.unique on an ascending vector == adjacent dedup.
        edges = cp.unique(qr)  # ascending, deduped
        me = int(edges.shape[0])
        if me <= 1:
            kx[r] = 1
            continue
        if me == 2:
            codes[r] = (V[r] >= edges[1]).astype(cp.int64)
            kx[r] = 2
            continue
        inner = edges[1 : me - 1]
        cr = cp.searchsorted(inner, V[r], side="right")  # 0..ni
        codes[r] = cr
        kx[r] = int(cr.max()) + 1
    return codes, kx


def _gpu_marginal_mi(codes, kx, y_codes, h_y, k_y, n):
    """Vectorised Miller-Madow marginal MI for each ROW of ``codes`` (shape ``(m, n)`` dense 0..kx[r]-1)
    vs the shared ``y_codes`` (dense 0..k_y-1). BIT-FAITHFUL to :func:`_marginal_mi_njit`: same plug-in
    entropies (natural log), same bias ``(kx_occ + k_y - kxy - 1)/(2n)``, same clamp. Returns (m,) MI."""
    cp = _cp
    m = codes.shape[0]
    invn = 1.0 / n
    out = cp.empty(m, dtype=cp.float64)
    for r in range(m):
        xb = codes[r]
        kxr = int(kx[r])
        # marginal H(X)
        cx = cp.bincount(xb, minlength=kxr).astype(cp.float64)
        nzx = cx[cx > 0]
        px = nzx * invn
        hx = float(-(px * cp.log(px)).sum())
        kx_occ = int((cx > 0).sum())
        # joint H(X,Y) via flat id xb*k_y + y
        joint = xb * k_y + y_codes
        cj = cp.bincount(joint, minlength=kxr * k_y).astype(cp.float64)
        nzj = cj[cj > 0]
        pj = nzj * invn
        hxy = float(-(pj * cp.log(pj)).sum())
        kxy = int((cj > 0).sum())
        mi = hx + h_y - hxy
        bias = (kx_occ + k_y - kxy - 1) / (2.0 * n)
        r_mi = mi - bias
        out[r] = r_mi if r_mi > 0.0 else 0.0
    return out


def _pair_combo_mi_cupy(x1, x2, y_codes, h_y, k_y, qs, ua_arr, ub_arr, bn_arr, xmin_a, xmin_b):
    """cupy GPU twin of :func:`_pair_combo_mi_njit` -- per-combo MI (or -1.0 std-sentinel), BIT-FAITHFUL
    to the CPU kernels (see the section docstring). Precomputes the ``nu`` distinct unary transforms of
    each operand ONCE on device, then processes combos in chunks. Raises on any cupy/device error so the
    dispatcher can fall back to CPU (the fit is never broken by a GPU problem)."""
    cp = _cp
    nc = int(ua_arr.shape[0])
    n = int(x1.shape[0])
    out = np.empty(nc, dtype=np.float64)
    if nc == 0:
        return out

    d_x1 = cp.asarray(x1, dtype=cp.float64)
    d_x2 = cp.asarray(x2, dtype=cp.float64)
    d_y = cp.asarray(y_codes, dtype=cp.int64)
    d_qs = cp.asarray(qs, dtype=cp.float64)

    # Precompute the DISTINCT unary transforms appearing in ua_arr / ub_arr (reused across combos).
    ua_codes = ua_arr.tolist()
    ub_codes = ub_arr.tolist()
    bn_codes = bn_arr.tolist()
    ua_cache: dict = {}
    ub_cache: dict = {}
    for c in set(ua_codes):
        ua_cache[c] = _gpu_apply_unary(d_x1, c, xmin_a)
    for c in set(ub_codes):
        ub_cache[c] = _gpu_apply_unary(d_x2, c, xmin_b)

    for start in range(0, nc, _GPU_COMBO_CHUNK):
        stop = min(start + _GPU_COMBO_CHUNK, nc)
        m = stop - start
        V = cp.empty((m, n), dtype=cp.float64)
        for r in range(m):
            j = start + r
            a = ua_cache[ua_codes[j]]
            b = ub_cache[ub_codes[j]]
            V[r] = _gpu_apply_binary(a, b, bn_codes[j])
        # std<=1e-9 sentinel: var = E[v^2] - E[v]^2 <= 1e-18.
        mean = V.mean(axis=1)
        var = (V * V).mean(axis=1) - mean * mean
        live = var > 1e-18
        codes, kx = _gpu_quantile_bin_codes(V, d_qs)
        mi = _gpu_marginal_mi(codes, kx, d_y, h_y, k_y, n)
        mi = cp.where(live, mi, cp.float64(-1.0))
        out[start:stop] = cp.asnumpy(mi)
    return out


# ----------------------------------------------------------------------------------------------
# Per-host backend crossover (serial njit / parallel njit / GPU cupy), resolved via the canonical
# kernel_tuning_cache -- NO hardcoded threshold (feedback_use_kernel_tuning_cache_for_gpu /
# feedback_fastest_default_with_dispatch). All three kernels are bit-faithful so the sweep just
# picks the FASTEST EQUIVALENT per (n_rows, n_combos) cell. CPU njit is the DEFAULT + FALLBACK: any
# cupy import / device error routes back to CPU, the fit is never broken.
#
# On a GTX 1050 Ti (cc 6.1, the dev box) prior fused MI kernels were HW-BOUND (no win vs CPU); the
# tuner is expected to keep this kernel on CPU here. The GPU path is the deliverable that wins on
# large-n / stronger cards -- the tuner measures and routes, we hardcode nothing.
# ----------------------------------------------------------------------------------------------
_USABILITY_SWEEP_COMBOS = [64, 289, 578, 1156, 1734]  # ~ |unary|^2*|binary| neighbourhood (17^2*{...})
_USABILITY_SWEEP_ROWS = [2_000, 10_000, 50_000]  # n_rows axis: GPU H2D/launch overhead amortises with n
_USABILITY_SALT = 2  # bumped: added n_rows axis + gpu (cupy) backend


def _make_usability_inputs(dims: dict):
    """An (n_rows x 2) operand pair + (n_combos) op-code arrays mirroring the kernel's call shape so
    the sweep measures the serial/parallel/gpu crossover on the real signature."""
    rng = np.random.default_rng(0)
    n_rows = int(dims.get("n_rows", 10_000))
    nc = int(dims["n_combos"])
    x1 = np.ascontiguousarray(rng.standard_normal(n_rows))
    x2 = np.ascontiguousarray(rng.exponential(1.2, n_rows))
    y = x1 * x1 / (x2 + 0.5) + 0.1 * rng.standard_normal(n_rows)
    # crude 10-bin codes for y
    edges = np.quantile(y, np.linspace(0.0, 1.0, 11))
    y_codes = np.searchsorted(np.unique(edges)[1:-1], y, side="right").astype(np.int64)
    k_y = int(y_codes.max()) + 1
    cy = np.bincount(y_codes).astype(np.float64) / n_rows
    h_y = float(-(cy[cy > 0] * np.log(cy[cy > 0])).sum())
    qs = np.linspace(0.0, 1.0, 11)
    ua = (rng.integers(0, 17, size=nc)).astype(np.int64)
    ub = (rng.integers(0, 17, size=nc)).astype(np.int64)
    bn = (rng.integers(0, 6, size=nc)).astype(np.int64)
    return (x1, x2, y_codes, h_y, k_y, qs, ua, ub, bn, float(x1.min()), float(x2.min()))


def _run_usability_sweep() -> list:
    """Runs the serial/parallel/gpu crossover sweep over the configured (n_rows, n_combos) grid, checking each
    variant's output against the serial reference within the loosened GPU-reassociation tolerance."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {
        "serial": lambda *a: _pair_combo_mi_njit(*a),
        "parallel": lambda *a: _pair_combo_mi_njit_parallel(*a),
    }
    if _CUPY_AVAIL:
        variants["gpu"] = lambda *a: _pair_combo_mi_cupy(*a)
    # GPU sort/bincount/log reassociate at the last bit -> the per-combo MI agrees with the CPU njit
    # to fp64 round-off, not bit-for-bit. The retention ranking only needs the ORDER preserved, so a
    # loosened equiv tol on the GPU cell is the documented deviation (see the GPU section docstring).
    return sweep_backend_grid(
        variants,
        {"n_rows": list(_USABILITY_SWEEP_ROWS), "n_combos": list(_USABILITY_SWEEP_COMBOS)},
        _make_usability_inputs,
        reference="serial", repeats=3, equiv_rtol=1e-6, equiv_atol=1e-9,
    )


def _usability_fallback_choice(n_rows: int, n_combos: int) -> str:
    """Pre-sweep heuristic (the spec's dynamic fallback callable): parallel njit above a conservative
    combo count where the prange fork-join overhead is amortised by the per-combo work. NEVER defaults
    to GPU -- the GPU path only fires once the per-host tuner has MEASURED it faster (on the dev GTX
    1050 Ti it is HW-bound, so the measured choice stays CPU)."""
    return "parallel" if int(n_combos) >= 64 else "serial"


_USABILITY_PARALLELISM_SPEC = kernel_tuner(
    kernel_name="usability_pool_kernel_parallelism",
    variant_fns=(_pair_combo_mi_njit, _pair_combo_mi_njit_parallel),  # CPU bodies; GPU covered by salt
    tuner=_run_usability_sweep,
    axes={"n_rows": list(_USABILITY_SWEEP_ROWS), "n_combos": list(_USABILITY_SWEEP_COMBOS)},
    fallback=_usability_fallback_choice,
    gpu_capable=True,
    salt=_USABILITY_SALT,
    cli_label="usability_pool_kernel_parallelism",
)


def _usability_backend_choice(n_rows: int, n_combos: int) -> str:
    """Per-host backend ('serial'/'parallel'/'gpu') for this (n_rows, n_combos) via the spec's
    choose(). Env override ``MLFRAME_USABILITY_POOL_BACKEND`` (njit|njit_parallel|gpu) forces a
    backend for testing (mirrors ``MLFRAME_MI_BACKEND``). The GPU choice is gated downstream on live
    cupy + the global GPU off-switch; on any failure the caller falls back to CPU."""
    forced = os.environ.get("MLFRAME_USABILITY_POOL_BACKEND", "").strip().lower()
    if forced == "njit":
        return "serial"
    if forced == "njit_parallel":
        return "parallel"
    if forced == "gpu":
        return "gpu"
    try:
        return str(_USABILITY_PARALLELISM_SPEC.choose(n_rows=int(n_rows), n_combos=int(n_combos)))
    except Exception:
        return _usability_fallback_choice(int(n_rows), int(n_combos))


def score_pair_combos(x1, x2, y_codes, y_terms, nbins, ua_codes, ub_codes, bn_codes):
    """Score every ``(ua, ub, bn)`` combo for a pair, returning the per-combo MI array (sentinel -1.0
    for an ``std<=1e-9`` value). ``ua_codes``/``ub_codes`` are the per-unary op-codes (in preset order),
    ``bn_codes`` the per-binary op-codes; this enumerates the full ``ua x ub x bn`` product in the
    SAME iteration order as the Python loop (``for ua: for ub: for bn``) so the returned index maps
    1:1 to the Python enumeration. ``y_terms`` is ``(y_i, h_y, k_y)`` from ``precompute_marginal_y_terms``."""
    _, h_y, k_y = y_terms
    nu = len(ua_codes)
    nb = len(bn_codes)
    nc = nu * nu * nb
    ua_arr = np.empty(nc, dtype=np.int64)
    ub_arr = np.empty(nc, dtype=np.int64)
    bn_arr = np.empty(nc, dtype=np.int64)
    j = 0
    for ia in range(nu):
        for ib in range(nu):
            for ibn in range(nb):
                ua_arr[j] = ua_codes[ia]
                ub_arr[j] = ub_codes[ib]
                bn_arr[j] = bn_codes[ibn]
                j += 1
    x1 = np.ascontiguousarray(x1, dtype=np.float64)
    x2 = np.ascontiguousarray(x2, dtype=np.float64)
    y_codes = np.ascontiguousarray(y_codes, dtype=np.int64)
    qs = np.linspace(0.0, 1.0, int(nbins) + 1)
    xmin_a = float(np.nanmin(x1)) if x1.size else 0.0
    xmin_b = float(np.nanmin(x2)) if x2.size else 0.0
    n_rows = int(x1.shape[0])
    args = (x1, x2, y_codes, float(h_y), int(k_y), qs, ua_arr, ub_arr, bn_arr, xmin_a, xmin_b)

    # bench-attempt-rejected (2026-06-23, GTX 1050 Ti, F2 100k MRMR wall /loop): forcing this pair-combo
    # MI table to the GPU twin (MLFRAME_USABILITY_POOL_BACKEND=gpu) is a 3x LOSS end-to-end -- F2 100k wall
    # 34.8s -> 102.5s (selection byte-identical, recipe hash 962a4c7b). The per-PAIR invocation enumerates a
    # small ua x ub x bn combo grid, so the GPU sees many tiny launches + per-pair H2D that swamp the
    # _pair_combo_mi_njit_table_parallel CPU kernel (cProfile tottime 0.97s, 10 calls). KTC correctly routes
    # this shape to "parallel" (CPU); the GPU path only pays off at large n_combos batched across pairs.
    # iter16 (2026-06-23) re-confirmed: the genuine resident fix is to batch EVERY pair's candidate columns
    # into ONE resident (n, K) matrix scored by the resident plug-in MI (the iter15 best_existing_op_mi_resident
    # pattern), NOT per-pair. That requires restructuring the per-pair retention loop + its lazy-recompute
    # diversity filter (_usability_aware_selection.py:230) which is tightly coupled to per-pair iteration -- a
    # large refactor whose payoff is sub-crossover on this card (small per-pair k). Deferred to a quiet machine
    # / capable GPU; the resident-GPU win this iter went to the maxT permutation-null floor (a single batched
    # workload, no caller refactor) -- see _permutation_null_resident.py (1.17-1.64x, bit-identical floor).
    choice = _usability_backend_choice(n_rows, nc)
    if choice == "gpu":
        # GPU path -- gated on live cupy + the global GPU off-switch (MLFRAME_DISABLE_GPU /
        # CUDA_VISIBLE_DEVICES=""). Any cupy/device error falls back to the CPU kernel: the fit is
        # NEVER broken by a GPU problem (correctness first).
        from ._gpu_policy import gpu_globally_disabled
        if _CUPY_AVAIL and not gpu_globally_disabled():
            try:
                return _pair_combo_mi_cupy(*args)
            except Exception:  # nosec B110 - optional/best-effort path, rationale documented
                pass  # fall through to CPU
        # cupy missing / disabled / device error -> CPU. Re-resolve serial-vs-parallel for the CPU twin.
        choice = _usability_fallback_choice(n_rows, nc)

    # CPU path: use the UNARY-TABLE kernels (precompute each operand's distinct unary transforms once;
    # bit-identical to the recompute-per-combo twins, ~1.4x faster -- 2026-06-21). ``nu_tab`` = highest
    # op-code present + 1 so the table is indexed directly by the op-codes in ua_arr/ub_arr. The
    # recompute-per-combo originals (_pair_combo_mi_njit[_parallel]) are kept as the tuner's measured
    # CPU bodies + an instant rollback (feedback_keep_all_kernel_versions).
    nu_tab = int(max(int(ua_arr.max()), int(ub_arr.max())) + 1) if nc else 1
    table_kernel = _pair_combo_mi_njit_table_parallel if choice == "parallel" else _pair_combo_mi_njit_table
    return table_kernel(*args, nu_tab)
