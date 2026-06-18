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

Two kernel twins are kept (``feedback_keep_all_kernel_versions``): a SERIAL njit and a
``parallel=True`` (prange over combos) twin, dispatched per-host by combo count via the canonical
``kernel_tuning_cache`` (NO hardcoded threshold -- ``feedback_use_kernel_tuning_cache_for_gpu``).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from pyutilz.performance.kernel_tuning.registry import kernel_tuner


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
    if code == 0:        # identity
        return v
    elif code == 1:      # neg
        return -v
    elif code == 2:      # abs
        return abs(v)
    elif code == 3:      # sqr = np.power(x, 2)
        return v * v
    elif code == 4:      # reciproc = np.power(x, -1)
        return 1.0 / v
    elif code == 5:      # sqrt = np.sqrt(np.abs(x))
        return np.sqrt(abs(v))
    elif code == 6:      # log = smart_log
        if xmin > 0.0:
            return np.log(v)
        return np.log(v + (1e-5 - xmin))
    elif code == 7:      # sin
        return np.sin(v)
    elif code == 8:      # sign = np.sign
        return 0.0 if v == 0.0 else (1.0 if v > 0.0 else -1.0)
    elif code == 9:      # rint
        return np.rint(v)
    elif code == 10:     # qubed = np.power(x, 3)
        return v * v * v
    elif code == 11:     # invsquared = np.power(x, -2)
        return 1.0 / (v * v)
    elif code == 12:     # invqubed = np.power(x, -3)
        return 1.0 / (v * v * v)
    elif code == 13:     # cbrt
        return np.cbrt(v)
    elif code == 14:     # invcbrt = np.power(x, -1/3)
        return v ** (-1.0 / 3.0)
    elif code == 15:     # invsqrt = np.power(x, -1/2)
        return v ** (-0.5)
    else:                # code == 16: exp
        return np.exp(v)


@njit(cache=True, inline="always")
def _apply_binary(a, b, bn):
    """One element of the minimal-preset binary, returning the float32-scrubbed value cast back to
    float64 (the Python path does ``_scrub(bf(ta, tb), float32)`` then upcasts to float64 for binning).
    Matches numpy's nan-propagating max/min and ``_safe_div`` exactly."""
    if bn == 0:          # mul
        v = a * b
    elif bn == 1:        # add
        v = a + b
    elif bn == 2:        # sub
        v = a - b
    elif bn == 3:        # div = _safe_div: exact a/b for b!=0, 1e-9 floor on exact-zero denominator
        v = a / (1e-9 if b == 0.0 else b)
    elif bn == 4:        # max = np.maximum (nan-propagating)
        if a != a or b != b:
            v = a + b
        else:
            v = a if a > b else b
    else:                # bn == 5: min = np.minimum (nan-propagating)
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
    srt = np.sort(val)
    nq = qs.shape[0]
    q = np.empty(nq, dtype=np.float64)
    for k in range(nq):
        pos = qs[k] * (n - 1)
        lo = int(np.floor(pos))
        hi = lo + 1 if lo < n - 1 else lo
        frac = pos - lo
        q[k] = srt[lo] + (srt[hi] - srt[lo]) * frac
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


# ----------------------------------------------------------------------------------------------
# Per-host serial-vs-parallel crossover (combo count axis), resolved via the canonical
# kernel_tuning_cache -- NO hardcoded threshold (feedback_use_kernel_tuning_cache_for_gpu).
# The two kernels are byte-identical so the sweep just times them at representative combo counts.
# ----------------------------------------------------------------------------------------------
_USABILITY_SWEEP_COMBOS = [64, 289, 578, 1156, 1734]  # ~ |unary|^2*|binary| neighbourhood (17^2*{...})
_USABILITY_SALT = 1


def _make_usability_inputs(dims: dict):
    """An (n_rows x 2) operand pair + (n_combos) op-code arrays mirroring the kernel's call shape so
    the sweep measures the serial-vs-parallel crossover on the real signature (n=10000-ish rows)."""
    rng = np.random.default_rng(0)
    n_rows = 10000
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
    from pyutilz.dev.benchmarking import sweep_backend_grid

    def _serial(*a):
        return _pair_combo_mi_njit(*a)

    def _parallel(*a):
        return _pair_combo_mi_njit_parallel(*a)

    return sweep_backend_grid(
        {"serial": _serial, "parallel": _parallel},
        {"n_combos": list(_USABILITY_SWEEP_COMBOS)},
        _make_usability_inputs,
        reference="serial", repeats=3, equiv_rtol=0.0, equiv_atol=0.0,
    )


def _usability_fallback_choice(n_combos: int) -> str:
    """Pre-sweep heuristic: parallel above a conservative combo count where the prange fork-join
    overhead is amortised by the per-combo work (each combo does n_rows unary+binary+sort+2 hist
    passes). The retention pair enumeration carries ~289-1734 combos, comfortably above the floor."""
    return "parallel" if int(n_combos) >= 64 else "serial"


_USABILITY_PARALLELISM_SPEC = kernel_tuner(
    kernel_name="usability_pool_kernel_parallelism",
    variant_fns=(_pair_combo_mi_njit, _pair_combo_mi_njit_parallel),
    tuner=_run_usability_sweep,
    axes={"n_combos": list(_USABILITY_SWEEP_COMBOS)},
    fallback=_usability_fallback_choice,
    gpu_capable=False,
    salt=_USABILITY_SALT,
    cli_label="usability_pool_kernel_parallelism",
)


def _use_parallel_kernel(n_combos: int) -> bool:
    """Dispatch predicate: use the ``parallel=True`` combo-prange twin iff the per-host
    kernel_tuning_cache says the combo count is above the serial-vs-parallel crossover. The usability
    pool runs on the SERIAL MAIN THREAD (no joblib nesting), so a numba prange is always safe here."""
    try:
        return _USABILITY_PARALLELISM_SPEC.choose(n_combos=int(n_combos)) == "parallel"
    except Exception:
        return _usability_fallback_choice(int(n_combos)) == "parallel"


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
    kernel = _pair_combo_mi_njit_parallel if _use_parallel_kernel(nc) else _pair_combo_mi_njit
    return kernel(x1, x2, y_codes, float(h_y), int(k_y), qs, ua_arr, ub_arr, bn_arr, xmin_a, xmin_b)
