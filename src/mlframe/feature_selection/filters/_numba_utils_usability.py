"""Njit kernels of the usability pool, hosted under ``_numba_utils`` so the cross-module @njit invariant holds. Re-exported by ``_numba_utils``."""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from .discretization.shared import multiselect_inplace
from collections.abc import Sequence

_NJIT_UNARY_OP_CODES: dict = {
    "identity": 0,
    "neg": 1,
    "abs": 2,
    "sqr": 3,
    "reciproc": 4,
    "sqrt": 5,
    "log": 6,
    "sin": 7,
    "sign": 8,
    "rint": 9,
    "qubed": 10,
    "invsquared": 11,
    "invqubed": 12,
    "cbrt": 13,
    "invcbrt": 14,
    "invsqrt": 15,
    "exp": 16,
}


_NJIT_USABILITY_BINARY_OP_CODES: dict = {
    "mul": 0,
    "add": 1,
    "sub": 2,
    "div": 3,
    "max": 4,
    "min": 5,
}


def njit_unary_codes_or_none(unary_names: Sequence[str]) -> "np.ndarray | None":
    """Map each unary name (preset iteration order) to its op-code, or None if ANY name is not
    njit-coded (the caller then uses the per-candidate numpy path - correctness first)."""
    codes = []
    for name in unary_names:
        c = _NJIT_UNARY_OP_CODES.get(name)
        if c is None:
            return None
        codes.append(c)
    return np.asarray(codes, dtype=np.int64)


def njit_binary_codes_or_none(binary_names: Sequence[str]) -> "np.ndarray | None":
    """Map each binary name to its op-code, or None if ANY name is not njit-coded."""
    codes = []
    for name in binary_names:
        c = _NJIT_USABILITY_BINARY_OP_CODES.get(name)
        if c is None:
            return None
        codes.append(c)
    return np.asarray(codes, dtype=np.int64)


# Intentional float64: relative tolerance on the float64 statistics the degenerate-column test compares, so its scale is float64 epsilon.
_USABILITY_DEGENERATE_REL_TOL = 32.0 * np.finfo(np.float64).eps


@njit(cache=True, inline="always")
def _apply_unary(v, code, xmin):
    """One element of the medium-preset unary, in double precision (the Python path applies the numpy unary to
    the double precision operand). ``xmin`` is the operand column's ``nanmin`` - only ``log`` (smart_log) uses
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
        # NaN input used to fall through to the else branch
        # (IEEE comparisons with NaN are always False) and get coerced to -1.0, disagreeing with both
        # np.sign(nan)==nan and this module's own cupy twin cp.sign(nan)==nan. Unreachable today via the
        # only wired caller (_usability_aware_selection.py's _scrub replaces NaN/Inf with 0.0 first), but
        # this is an exported, reusable function - fixed for any future direct caller.
        if np.isnan(v):
            return v
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
    double precision (the Python path does ``_scrub(bf(ta, tb), float32)`` then upcasts to double precision for binning).
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
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    return np.float64(f)


@njit(cache=True, inline="always")
def _qbin_into(val, nbins, qs, codes_out):
    """Equi-frequency quantile-bin ``val`` (finite double precision) into ``codes_out``, returning the bin
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
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
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
    # One multiselect pass over the sorted unique anchors (the lo/hi sequence is non-decreasing), not a separate selection per index.
    part = val.copy()
    multiselect_inplace(part, np.unique(kths[:m2]))
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    q = np.empty(nq, dtype=np.float64)
    for k in range(nq):
        lo = los[k]
        hi = lo + 1 if lo < n - 1 else lo
        q[k] = part[lo] + (part[hi] - part[lo]) * fracs[k]
    # np.unique(q) - q is ascending, so dedup adjacent equals in one pass.
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
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
    # searchsorted(inner, v, side='right') on sorted inner edges is the number of edges <= v. Counting them is branch-free and identical,
    # and beats the bisection's data-dependent branches on the few (<= nbins - 1) edges.
    for i in range(n):
        v = val[i]
        c = 0
        for e in range(ni):
            c += edges[1 + e] <= v
        codes_out[i] = c
        if c > kx:
            kx = c
    return kx + 1


@njit(cache=True, inline="always")
def _marginal_mi_njit(xb, kx, y_codes, h_y, k_y, n):
    """Plug-in Miller-Madow ``MI(X;Y) = H(X)+H(Y)-H(X,Y)`` minus ``(k_x+k_y-k_xy-1)/(2n)``, clamped
    to >=0 - bit-faithful to ``marginal_mi_binned_fixed_y`` (same plug-in entropies, same bias, same
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
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    out = np.empty(nc, dtype=np.float64)
    for j in range(nc):
        ua = ua_arr[j]
        ub = ub_arr[j]
        bn = bn_arr[j]
        # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
        val = np.empty(n, dtype=np.float64)
        s = 0.0
        vmin = np.inf
        vmax = -np.inf
        for i in range(n):
            a = _apply_unary(x1[i], ua, xmin_a)
            b = _apply_unary(x2[i], ub, xmin_b)
            v = _apply_binary(a, b, bn)
            val[i] = v
            s += v
            if v < vmin:
                vmin = v
            if v > vmax:
                vmax = v
        # Centred two-pass variance, plus an EXACT constant short-circuit. The previous form was
        # ``var = ss / n - mean * mean`` from raw power sums, whose cancellation noise floor is ~eps * mean^2
        # -- far ABOVE the ``1e-18`` constant gate for any |mean| > 0.067, so the gate was reading noise. On two
        # epoch-second columns added together (v ~ 3.4e9) the floor is ~2.6e3: any combo whose true variance was
        # under that computed a random-signed value, landed negative about half the time, and was written out as
        # the -1.0 sentinel -- a SILENT false-negative feature rejection, indistinguishable downstream from a
        # legitimately constant combo. The converse also fired: a truly constant combo whose noise landed
        # positive passed the gate and was quantile-binned as a degenerate single-bin column, which vmin/vmax
        # now catches exactly. ``val`` is already materialised in the first pass, so the second costs one more
        # sequential read of an array that is still in cache.
        if vmax == vmin:
            out[j] = -1.0
            continue
        mean = s / n
        ss = 0.0
        for i in range(n):
            d = val[i] - mean
            ss += d * d
        var = ss / n
        # Relative to the combo's own magnitude (vmin/vmax are known): an absolute 1e-18 floor rejected every genuinely tiny-scale combo.
        if var <= (_USABILITY_DEGENERATE_REL_TOL * max(abs(vmin), abs(vmax))) ** 2:
            out[j] = -1.0
            continue
        codes = np.empty(n, dtype=np.int64)
        kx = _qbin_into(val, nbins, qs, codes)
        out[j] = _marginal_mi_njit(codes, kx, y_codes, h_y, k_y, n)
    return out


@njit(parallel=True, cache=True)
def _pair_combo_mi_njit_parallel(x1, x2, y_codes, h_y, k_y, qs, ua_arr, ub_arr, bn_arr, xmin_a, xmin_b):
    """``parallel=True`` (prange over COMBOS) twin of :func:`_pair_combo_mi_njit` - BYTE-IDENTICAL
    output (each combo ``j`` reads only ``x1``/``x2`` and writes only ``out[j]``; zero cross-combo
    dependence -> result independent of thread count). Kept separate per ``feedback_keep_all_kernel_versions``."""
    nc = ua_arr.shape[0]
    n = x1.shape[0]
    nbins = qs.shape[0] - 1
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    out = np.empty(nc, dtype=np.float64)
    for j in prange(nc):
        ua = ua_arr[j]
        ub = ub_arr[j]
        bn = bn_arr[j]
        # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
        val = np.empty(n, dtype=np.float64)
        s = 0.0
        vmin = np.inf
        vmax = -np.inf
        for i in range(n):
            a = _apply_unary(x1[i], ua, xmin_a)
            b = _apply_unary(x2[i], ub, xmin_b)
            v = _apply_binary(a, b, bn)
            val[i] = v
            s += v
            if v < vmin:
                vmin = v
            if v > vmax:
                vmax = v
        # Centred two-pass variance, plus an EXACT constant short-circuit. The previous form was
        # ``var = ss / n - mean * mean`` from raw power sums, whose cancellation noise floor is ~eps * mean^2
        # -- far ABOVE the ``1e-18`` constant gate for any |mean| > 0.067, so the gate was reading noise. On two
        # epoch-second columns added together (v ~ 3.4e9) the floor is ~2.6e3: any combo whose true variance was
        # under that computed a random-signed value, landed negative about half the time, and was written out as
        # the -1.0 sentinel -- a SILENT false-negative feature rejection, indistinguishable downstream from a
        # legitimately constant combo. The converse also fired: a truly constant combo whose noise landed
        # positive passed the gate and was quantile-binned as a degenerate single-bin column, which vmin/vmax
        # now catches exactly. ``val`` is already materialised in the first pass, so the second costs one more
        # sequential read of an array that is still in cache.
        if vmax == vmin:
            out[j] = -1.0
            continue
        mean = s / n
        ss = 0.0
        for i in range(n):
            d = val[i] - mean
            ss += d * d
        var = ss / n
        if var <= (_USABILITY_DEGENERATE_REL_TOL * max(abs(vmin), abs(vmax))) ** 2:
            out[j] = -1.0
            continue
        codes = np.empty(n, dtype=np.int64)
        kx = _qbin_into(val, nbins, qs, codes)
        out[j] = _marginal_mi_njit(codes, kx, y_codes, h_y, k_y, n)
    return out


@njit(cache=True)
def _pair_combo_mi_njit_table(x1, x2, y_codes, h_y, k_y, qs, ua_arr, ub_arr, bn_arr, xmin_a, xmin_b, nu_tab):
    """SERIAL unary-table twin of :func:`_pair_combo_mi_njit` - BIT-IDENTICAL output, precomputes the
    ``nu_tab`` distinct unary transforms of each operand once into a (nu_tab, n) table."""
    nc = ua_arr.shape[0]
    n = x1.shape[0]
    nbins = qs.shape[0] - 1
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    U1 = np.empty((nu_tab, n), dtype=np.float64)
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    U2 = np.empty((nu_tab, n), dtype=np.float64)
    for u in range(nu_tab):
        for i in range(n):
            U1[u, i] = _apply_unary(x1[i], u, xmin_a)
            U2[u, i] = _apply_unary(x2[i], u, xmin_b)
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    out = np.empty(nc, dtype=np.float64)
    for j in range(nc):
        ua = ua_arr[j]
        ub = ub_arr[j]
        bn = bn_arr[j]
        # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
        val = np.empty(n, dtype=np.float64)
        s = 0.0
        vmin = np.inf
        vmax = -np.inf
        for i in range(n):
            v = _apply_binary(U1[ua, i], U2[ub, i], bn)
            val[i] = v
            s += v
            if v < vmin:
                vmin = v
            if v > vmax:
                vmax = v
        # Centred two-pass variance, plus an EXACT constant short-circuit. The previous form was
        # ``var = ss / n - mean * mean`` from raw power sums, whose cancellation noise floor is ~eps * mean^2
        # -- far ABOVE the ``1e-18`` constant gate for any |mean| > 0.067, so the gate was reading noise. On two
        # epoch-second columns added together (v ~ 3.4e9) the floor is ~2.6e3: any combo whose true variance was
        # under that computed a random-signed value, landed negative about half the time, and was written out as
        # the -1.0 sentinel -- a SILENT false-negative feature rejection, indistinguishable downstream from a
        # legitimately constant combo. The converse also fired: a truly constant combo whose noise landed
        # positive passed the gate and was quantile-binned as a degenerate single-bin column, which vmin/vmax
        # now catches exactly. ``val`` is already materialised in the first pass, so the second costs one more
        # sequential read of an array that is still in cache.
        if vmax == vmin:
            out[j] = -1.0
            continue
        mean = s / n
        ss = 0.0
        for i in range(n):
            d = val[i] - mean
            ss += d * d
        var = ss / n
        if var <= (_USABILITY_DEGENERATE_REL_TOL * max(abs(vmin), abs(vmax))) ** 2:
            out[j] = -1.0
            continue
        codes = np.empty(n, dtype=np.int64)
        kx = _qbin_into(val, nbins, qs, codes)
        out[j] = _marginal_mi_njit(codes, kx, y_codes, h_y, k_y, n)
    return out


@njit(parallel=True, cache=True)
def _pair_combo_mi_njit_table_parallel(x1, x2, y_codes, h_y, k_y, qs, ua_arr, ub_arr, bn_arr, xmin_a, xmin_b, nu_tab):
    """``parallel=True`` unary-table twin - BIT-IDENTICAL to :func:`_pair_combo_mi_njit_table` (the table
    precompute is parallel over the ``nu_tab`` unaries, then prange over combos; each combo writes only
    ``out[j]``, zero cross-combo dependence). The DEFAULT CPU path on the retention enumeration."""
    nc = ua_arr.shape[0]
    n = x1.shape[0]
    nbins = qs.shape[0] - 1
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    U1 = np.empty((nu_tab, n), dtype=np.float64)
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    U2 = np.empty((nu_tab, n), dtype=np.float64)
    for u in prange(nu_tab):
        for i in range(n):
            U1[u, i] = _apply_unary(x1[i], u, xmin_a)
            U2[u, i] = _apply_unary(x2[i], u, xmin_b)
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    out = np.empty(nc, dtype=np.float64)
    for j in prange(nc):
        ua = ua_arr[j]
        ub = ub_arr[j]
        bn = bn_arr[j]
        # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
        val = np.empty(n, dtype=np.float64)
        s = 0.0
        vmin = np.inf
        vmax = -np.inf
        for i in range(n):
            v = _apply_binary(U1[ua, i], U2[ub, i], bn)
            val[i] = v
            s += v
            if v < vmin:
                vmin = v
            if v > vmax:
                vmax = v
        # Centred two-pass variance, plus an EXACT constant short-circuit. The previous form was
        # ``var = ss / n - mean * mean`` from raw power sums, whose cancellation noise floor is ~eps * mean^2
        # -- far ABOVE the ``1e-18`` constant gate for any |mean| > 0.067, so the gate was reading noise. On two
        # epoch-second columns added together (v ~ 3.4e9) the floor is ~2.6e3: any combo whose true variance was
        # under that computed a random-signed value, landed negative about half the time, and was written out as
        # the -1.0 sentinel -- a SILENT false-negative feature rejection, indistinguishable downstream from a
        # legitimately constant combo. The converse also fired: a truly constant combo whose noise landed
        # positive passed the gate and was quantile-binned as a degenerate single-bin column, which vmin/vmax
        # now catches exactly. ``val`` is already materialised in the first pass, so the second costs one more
        # sequential read of an array that is still in cache.
        if vmax == vmin:
            out[j] = -1.0
            continue
        mean = s / n
        ss = 0.0
        for i in range(n):
            d = val[i] - mean
            ss += d * d
        var = ss / n
        if var <= (_USABILITY_DEGENERATE_REL_TOL * max(abs(vmin), abs(vmax))) ** 2:
            out[j] = -1.0
            continue
        codes = np.empty(n, dtype=np.int64)
        kx = _qbin_into(val, nbins, qs, codes)
        out[j] = _marginal_mi_njit(codes, kx, y_codes, h_y, k_y, n)
    return out
