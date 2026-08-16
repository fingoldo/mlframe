"""GPU-resident FE: prewarp/orth-basis device build (Tier E carve).

Carved VERBATIM out of ``_gpu_resident_fe.py`` (sibling re-export pattern) to bring the parent under the
1k-LOC ceiling: the GPU port of the per-operand PRE-WARP apply + the matrix-native orth-FE basis-column
evaluation (the ``_*_clenshaw_gpu`` recurrences, ``_gpu_evaluate_basis_*``, ``_gpu_route_bases_batched``,
``_gpu_apply_prewarp`` - all numpy/cupy, no parent dependency).

The analytic pair-MI / grand-fusion / dispatch block that used to live here (``gpu_pairs_fe_mi``,
``grand_fused_pair_mi[_fused]``, ``gpu_resident_pair_recipes``, ``pair_candidate_mi_dispatch``) was carved
again into the sibling ``_gpu_resident_pair_mi.py`` (a carve of this carve) - it had zero references into
this file's basis-evaluation block. The gate helpers and the candidate-grid primitives stay in the PARENT
and are imported below; the parent re-exports every public/used name moved here so all
``from .._gpu_resident_fe import X`` paths still resolve byte-for-byte. No kernel-source,
dispatch-threshold, residency, or selection behavior changed.
"""
from __future__ import annotations

import numpy as np

# --- GPU port of the per-operand PRE-WARP apply (phase R1, 2026-06-21) ----------------------------------
# Mirrors hermite_fe.apply_operand_prewarp so the operand-table mirror can BUILD a prewarp operand column on
# the device (from the resident raw input + the tiny stored spec) instead of COPYING the host-computed column
# (the 1.68 MB non-plain H2D floor). The preprocess (z-score / min-max / shift, all elementwise + a clip) and
# the Clenshaw polynomial recurrences below replicate NUMPY's chebval/legval/hermeval(He)/lagval to fp
# round-off (same Clenshaw algorithm + float64 op order). SELECTION-EQUIVALENCE NOTE (P2-2): the production
# HOST path (polyeval_dispatch -> njit) evaluates these three (cheb/leg/herme) by a FORWARD recurrence, which
# differs from numpy/GPU-Clenshaw by ~1e-12 at degree>=3 (laguerre is forward on both, so it is bit-consistent
# across device+host). So a candidate MI-RANKED on GPU-Clenshaw values and later REPLAYED via host-forward
# differs by ~1e-12 - far below any selection threshold (selection is decided on the consistent GPU values
# within the resident path; test_gpu_basis_column_parity pins the host<->GPU bound). Unifying both onto one
# recurrence is a FUTURE kernel change, unneeded at the default max_degree. Any unsupported basis / failure
# RAISES so the caller falls back to the host copy (never a correctness regression). fourier_adaptive
# (escalation only - not in the main operand table) is ported too for completeness.

def _cheb_clenshaw_gpu(cp, x, c):
    """Clenshaw-recurrence evaluation of a Chebyshev series with coefficients ``c`` at cupy array ``x``, matching the host ``_chebval_njit``."""
    if len(c) == 1:
        c0 = c[0]; c1 = 0.0
    elif len(c) == 2:
        c0 = c[0]; c1 = c[1]
    else:
        x2 = 2.0 * x
        c0 = c[-2]; c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1 * x2
    return c0 + c1 * x


def _leg_clenshaw_gpu(cp, x, c):
    """Clenshaw-recurrence evaluation of a Legendre series with coefficients ``c`` at cupy array ``x``, matching the host ``_legval_njit``."""
    if len(c) == 1:
        return c[0] + 0.0 * x
    if len(c) == 2:
        c0 = c[0]; c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]; c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * x * (2 * nd - 1)) / nd
    return c0 + c1 * x


def _herme_clenshaw_gpu(cp, x, c):
    """Clenshaw-recurrence evaluation of a (probabilists') Hermite series with coefficients ``c`` at cupy array ``x``, matching the host ``_hermeval_njit``."""
    if len(c) == 1:
        return c[0] + 0.0 * x
    if len(c) == 2:
        c0 = c[0]; c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]; c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - c1 * (nd - 1)
            c1 = tmp + c1 * x
    return c0 + c1 * x


def _lag_clenshaw_gpu(cp, x, c):
    """Forward-recurrence evaluation of a Laguerre series with coefficients ``c`` at cupy array ``x``. Uses the forward form (not Clenshaw) since Laguerre's Clenshaw variant was found wrong here - see the comment below."""
    # FORWARD recurrence ``out = sum_k c[k] L_k`` matching the host _lagval_njit EXACTLY
    # (L_0=1, L_1=1-x, L_k = ((2k-1-x)L_{k-1} - (k-1)L_{k-2})/k). The prior Clenshaw-style
    # recurrence here was WRONG for Laguerre (verified: L_2(0) gave -0.5 vs the correct 1) -
    # it was never exercised because the canonical prewarp uses the chebyshev basis, so no pin
    # caught it; the matrix-native basis parity test (test_gpu_basis_column_parity) surfaced it.
    nc = len(c)
    if nc == 0:
        return cp.zeros_like(x)
    out = cp.full(x.shape, c[0], dtype=x.dtype)
    if nc == 1:
        return out
    p_prev = cp.ones_like(x)  # L_0
    p_curr = 1.0 - x  # L_1
    out = out + c[1] * p_curr
    for k in range(2, nc):
        p_next = ((2 * k - 1 - x) * p_curr - (k - 1) * p_prev) / k
        out = out + c[k] * p_next
        p_prev = p_curr
        p_curr = p_next
    return out


_PREWARP_CLENSHAW_GPU = {
    "chebyshev": _cheb_clenshaw_gpu,
    "legendre": _leg_clenshaw_gpu,
    "hermite": _herme_clenshaw_gpu,
    "laguerre": _lag_clenshaw_gpu,
}

_BASIS_ID = {"chebyshev": 0, "legendre": 1, "hermite": 2, "laguerre": 3}

# FUSED CLENSHAW BASIS RawKernel (launch-reduction, 2026-06-25): one launch evaluates the one-hot
# orthonormal basis polynomial B_d(Z) for EVERY (column, degree) of a preprocessed (n,m) Z block, output
# (n, m*nd) in the concatenate layout [deg0: all m cols][deg1: ...]. Replaces the per-degree clen() cupy
# chains (~nd*degree elementwise ops -> the measured #1 launch source _gpu_evaluate_basis_matrix). Each
# in-kernel recurrence reproduces _{cheb,leg,herme}_clenshaw_gpu (backward Clenshaw, one-hot coef) and
# _lag_clenshaw_gpu (forward) op-for-op in float64 -> bit-identical basis values.
_CLENSHAW_SRC = r"""
__device__ double eval_basis(int basis, int d, double x) {
    if (d == 0) return 1.0;
    if (d == 1) return basis == 3 ? (1.0 - x) : x;
    if (basis == 3) {                                  // laguerre forward L_d
        double pp = 1.0, pc = 1.0 - x, pn;
        for (int k = 2; k <= d; ++k) { pn = ((2.0*k - 1.0 - x) * pc - (k - 1) * pp) / k; pp = pc; pc = pn; }
        return pc;
    }
    if (basis == 0) {                                  // chebyshev Clenshaw one-hot
        double x2 = 2.0 * x, c0 = 0.0, c1 = 1.0;
        for (int i = 3; i <= d + 1; ++i) { double tmp = c0; c0 = -c1; c1 = tmp + c1 * x2; }
        return c0 + c1 * x;
    }
    if (basis == 1) {                                  // legendre Clenshaw one-hot
        int nd = d + 1; double c0 = 0.0, c1 = 1.0;
        for (int i = 3; i <= d + 1; ++i) { double tmp = c0; nd -= 1; c0 = -(c1 * (nd - 1)) / nd; c1 = tmp + (c1 * x * (2*nd - 1)) / nd; }
        return c0 + c1 * x;
    }
    // basis == 2: hermite(He) Clenshaw one-hot
    int nd = d + 1; double c0 = 0.0, c1 = 1.0;
    for (int i = 3; i <= d + 1; ++i) { double tmp = c0; nd -= 1; c0 = -c1 * (nd - 1); c1 = tmp + c1 * x; }
    return c0 + c1 * x;
}
extern "C" __global__
void clenshaw_basis(const double* __restrict__ Z, const int* __restrict__ degs, const long long n,
                    const int m, const int nd, const int basis, double* __restrict__ out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)m * nd;
    if (t >= total) return;
    int j = (int)(t % (long long)m);
    long long r = t / (long long)m;
    int k = (int)(r % (long long)nd);
    long long i = r / (long long)nd;
    double z = Z[i * (long long)m + j];
    out[i * ((long long)m * nd) + (long long)k * m + j] = eval_basis(basis, degs[k], z);
}
"""
_CLENSHAW_KERNEL = None


def _get_clenshaw_kernel():
    """Lazily compile-and-cache the ``clenshaw_basis`` RawKernel on first call (CUDA compile is ~30ms; subsequent calls reuse the cached kernel handle)."""
    global _CLENSHAW_KERNEL
    if _CLENSHAW_KERNEL is None:
        import cupy as cp
        _CLENSHAW_KERNEL = cp.RawKernel(_CLENSHAW_SRC, "clenshaw_basis")
    return _CLENSHAW_KERNEL


def _clenshaw_basis_block_gpu(cp, Zg, basis, degrees):
    """Fused (n, m*nd) one-hot basis block for ``basis`` over (n,m) ``Zg``, columns ordered
    [degrees[0]: all m][degrees[1]: ...] (== the per-degree clen()+concatenate layout). Returns None for an
    unsupported basis (caller per-degree cupy fallback). Bit-identical to clen() per degree."""
    bid = _BASIS_ID.get(basis)
    if bid is None:
        return None
    Z = cp.ascontiguousarray(Zg.astype(cp.float64, copy=False))
    n, m = int(Z.shape[0]), int(Z.shape[1])
    nd = len(degrees)
    degs = cp.asarray(np.asarray([int(d) for d in degrees], dtype=np.int32))
    out = cp.empty((n, m * nd), dtype=cp.float64)
    total = n * m * nd
    threads = 256
    _get_clenshaw_kernel()(((total + threads - 1) // threads,), (threads,), (Z, degs, np.int64(n), np.int32(m), np.int32(nd), np.int32(bid), out))
    return out

# --- GPU port of the orth-FE basis-column evaluation (matrix-native, Piece 2, 2026-06-21) -------------
# Faithful cupy mirror of _orthogonal_univariate_fe._evaluate_basis_column (no-aux, no-replay path):
# the robust heavy-tail axis detection (_hermite_robust._detect_heavy_tail_numpy/_robust_scale/
# _robust_lo_hi) + the per-basis preprocess (z-score / min-max / shift, robust + plain branches) + the
# one-hot Clenshaw eval (the _*_clenshaw_gpu above). Lets the orth-FE candidate matrix be built ON the
# device (operands resident) so it feeds _plugin_mi_classif_batch_cuda_resident with NO H2D - removing
# the np.median (robust axis) + argsort/reduce (plug-in MI) of the CPU tail. Constants mirror
# _hermite_robust (K=3, OUTER_K=10, GAP=3, MAX_FRAC=0.20). cp.median/percentile match np to fp round-off;
# parity is asserted by test_gpu_basis_column_parity (uniform/gaussian/heavytail/skewed x 4 bases).
# These MIRROR hermite_fe._hermite_robust._ROBUST_AXIS_* (NOT imported: the GPU module avoids a top-level
# hermite_fe import - the cycle the in-function lazy imports work around). Keep in sync; the drift is
# guarded by tests/feature_selection/gpu/test_gpu_cpu_robust_constants_in_sync.py.
_GPU_ROBUST_AXIS_K = 3.0
_GPU_ROBUST_AXIS_OUTER_K = 10.0
_GPU_ROBUST_AXIS_GAP = 3.0
_GPU_ROBUST_AXIS_MAX_FRAC = 0.20


def _batched_quantiles(cp, M, q_fracs):
    """Per-column quantiles ``(len(q_fracs), K)`` over finite (n, K) ``M``, via the sort-free radix-select
    machinery (``_gpu_resident_select._radix_quantiles``) - BIT-IDENTICAL to ``cp.percentile(M,
    [q*100 ...], axis=0)`` and to ``cp.median`` at q=0.5. Replaces the per-call full sort of cp.median /
    cp.percentile in the robust-scale / heavy-tail batched path (the measured #1 GPU-launch source, 672)
    with one select+interp launch pair. Falls back to cp.percentile on any radix inapplicability / error."""
    try:
        from ._gpu_resident_select import _radix_quantiles

        out = _radix_quantiles(M, q_fracs)
        if out is not None:
            return out
    except Exception as e:  # nosec B110 - best-effort/optional path, no module logger
        import logging
        logging.getLogger(__name__).debug("_radix_quantiles failed, falling back to cp.percentile: %s", e)
    return cp.percentile(M, cp.asarray([float(q) * 100.0 for q in q_fracs]), axis=0)


def _gpu_robust_scale(cp, xf, med):
    """cupy mirror of _hermite_robust._robust_scale: 1.4826*MAD, IQR/1.349 fallback, 0.0 if degenerate."""
    mad = float(cp.median(cp.abs(xf - med)))
    scale = 1.4826 * mad
    if scale > 1e-12:
        return scale
    q25, q75 = (float(v) for v in cp.percentile(xf, cp.asarray([25.0, 75.0])))
    iqr_scale = (q75 - q25) / 1.349
    return iqr_scale if iqr_scale > 1e-12 else 0.0


def _gpu_detect_heavy_tail(cp, xf):
    """cupy mirror of _hermite_robust._detect_heavy_tail_numpy (the n>=3000 oracle path)."""
    if xf.size < 8:
        return False
    med = float(cp.median(xf))
    scale = _gpu_robust_scale(cp, xf, med)
    if scale <= 1e-12:
        return False
    dev = cp.abs(xf - med)
    thr = _GPU_ROBUST_AXIS_OUTER_K * scale
    outer_mask = dev > thr
    n_outer = int(cp.count_nonzero(outer_mask))
    if n_outer == 0 or n_outer > _GPU_ROBUST_AXIS_MAX_FRAC * xf.size:
        return False
    bulk_edge = float(dev[~outer_mask].max())
    outer_min = float(dev[outer_mask].min())
    return (outer_min / max(bulk_edge, 1e-12)) >= _GPU_ROBUST_AXIS_GAP


def _gpu_robust_lo_hi(cp, x, xf, med):
    """Robust ``[lo, hi]`` bounds around the median for basis-domain rescaling; falls back to ``[min, max]`` when the robust scale collapses to near-zero."""
    scale = _gpu_robust_scale(cp, xf, med)
    if scale <= 1e-12:
        return float(cp.min(xf)), float(cp.max(xf))
    return med - _GPU_ROBUST_AXIS_K * scale, med + _GPU_ROBUST_AXIS_K * scale


def _gpu_basis_preprocess(cp, x, basis, *, robust: bool):
    """cupy mirror of the per-basis preprocess fit (_preprocess_zscore/minmax/shift). Returns z (device).
    ``robust`` is the resolved (_robust_axis_enabled() AND _gpu_detect_heavy_tail) decision from the caller."""
    xf = x[cp.isfinite(x)]
    if basis == "hermite":  # z-score
        if robust:
            center = float(cp.median(xf))
            lo, hi = _gpu_robust_lo_hi(cp, x, xf, center)
            std = (hi - lo) / 6.0
            std = std if std > 1e-12 else (float(cp.std(xf)) + 1e-12)
            return cp.clip((x - center) / std, -6.0, 6.0)
        mean = float(cp.mean(x)); std = float(cp.std(x)) + 1e-12
        return (x - mean) / std
    if basis in ("legendre", "chebyshev"):  # min-max -> [-1, 1]
        if robust:
            med = float(cp.median(xf))
            lo, hi = _gpu_robust_lo_hi(cp, x, xf, med)
            span = hi - lo + 1e-12
            return cp.clip(2.0 * (x - lo) / span - 1.0, -1.0, 1.0)
        lo = float(cp.min(x)); hi = float(cp.max(x)); span = hi - lo + 1e-12
        return 2.0 * (x - lo) / span - 1.0
    if basis == "laguerre":  # shift to >= 0
        if robust:
            med = float(cp.median(xf))
            lo, hi = _gpu_robust_lo_hi(cp, x, xf, med)
            upper = hi - lo
            return cp.clip(x - lo + 1e-9, 0.0, upper + 1e-9)
        lo = float(cp.min(x))
        return x - lo + 1e-9
    raise ValueError(f"basis {basis!r} not GPU-ported")


def _gpu_evaluate_basis_column(cp, x, basis, degree, *, robust_axis: bool):
    """Device port of _evaluate_basis_column (no-aux / no-replay path). ``x`` is an (n,) cupy float64
    operand; returns the (n,) cupy float64 basis-column values. ``robust_axis`` = _robust_axis_enabled()
    (host env, passed in to avoid a per-call import). Heavy-tail is detected on-device. Raises for an
    unported basis so the caller falls back to the host _evaluate_basis_column (never a correctness loss)."""
    xf64 = x.astype(cp.float64)
    use_robust = bool(robust_axis) and _gpu_detect_heavy_tail(cp, xf64[cp.isfinite(xf64)])
    z = cp.ascontiguousarray(_gpu_basis_preprocess(cp, xf64, basis, robust=use_robust), dtype=cp.float64)
    clen = _PREWARP_CLENSHAW_GPU.get(basis)
    if clen is None:
        raise ValueError(f"basis {basis!r} not GPU-ported")
    coef = [0.0] * (int(degree) + 1)
    coef[int(degree)] = 1.0
    return clen(cp, z, coef)


# --- BATCHED device basis build (matrix-native Piece 3b, 2026-06-21) ---------------------------------
# Vectorized port of the per-column _gpu_evaluate_basis_column: process ALL columns of a (basis, robust)
# group in ONE preprocess + ONE Clenshaw call per degree (axis=0 stats over the (n, g) submatrix),
# killing the per-column cupy launch overhead that made the per-column loop perf-lose at high feature
# count (p200). Bit-equivalent to the per-column path (same math, just vectorised); guarded by
# test_gpu_basis_column_parity's batched leg.

_ROBUST_SCALE_COMBINE_SRC = r"""
extern "C" __global__
void robust_scale_combine(const double* __restrict__ mad, const double* __restrict__ q25,
                          const double* __restrict__ q75, const int K, double* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;
    double sc = 1.4826 * mad[i];
    double iqr = (q75[i] - q25[i]) / 1.349;
    out[i] = sc > 1e-12 ? sc : (iqr > 1e-12 ? iqr : 0.0);
}
"""
_ROBUST_SCALE_COMBINE_KERNEL = None


def _get_robust_scale_combine_kernel(cp):
    """Lazily compile-and-cache the ``robust_scale_combine`` RawKernel (fuses the MAD/IQR-fallback/degenerate combine into one launch)."""
    global _ROBUST_SCALE_COMBINE_KERNEL
    if _ROBUST_SCALE_COMBINE_KERNEL is None:
        _ROBUST_SCALE_COMBINE_KERNEL = cp.RawKernel(_ROBUST_SCALE_COMBINE_SRC, "robust_scale_combine")
    return _ROBUST_SCALE_COMBINE_KERNEL


def _gpu_robust_scale_batched(cp, M, med, *, q25=None, q75=None):
    """Per-column (axis=0) robust scale over finite (n, g) M: 1.4826*MAD, IQR/1.349 fallback, 0 degenerate.

    Quantiles routed through ``_batched_quantiles`` (sort-free radix-select, bit-identical to cp.median /
    cp.percentile) to drop the per-call full sort - the measured #1 GPU-launch source. ``q25`` / ``q75``
    may be passed pre-computed by a caller that already extracted M's quartiles (e.g. heavy-tail folds
    them into one [0.25,0.5,0.75] radix call) to skip the redundant q25/q75 select+interp here."""
    mad = _batched_quantiles(cp, cp.abs(M - med), [0.5])[0]  # median(|M-med|) per column
    if q25 is None or q75 is None:
        q = _batched_quantiles(cp, M, [0.25, 0.75])  # (2, g) == cp.percentile(M,[25,75],axis=0)
        q25, q75 = q[0], q[1]
    # FUSED combine (launch-reduction): 1.4826*MAD, IQR/1.349 fallback, 0 degenerate - one (K,) kernel vs
    # the scale + iqr + where + where chain. Same f64 ops -> bit-identical; cupy chain fallback on error.
    try:
        K = int(mad.shape[0])
        out = cp.empty(K, dtype=cp.float64)
        threads = 128
        _get_robust_scale_combine_kernel(cp)(((K + threads - 1) // threads,), (threads,),
            (cp.ascontiguousarray(mad.astype(cp.float64, copy=False)),
             cp.ascontiguousarray(q25.astype(cp.float64, copy=False)),
             cp.ascontiguousarray(q75.astype(cp.float64, copy=False)), np.int32(K), out))
        return out
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("robust-scale RawKernel failed, falling back to the cupy chain: %s", e)
        scale = 1.4826 * mad
        iqr = (q75 - q25) / 1.349
        return cp.where(scale > 1e-12, scale, cp.where(iqr > 1e-12, iqr, 0.0))


def _gpu_robust_lo_hi_batched(cp, M, med, scale):
    """Per-column robust ``[lo, hi]`` bounds around ``med``; columns with a degenerate (near-zero) scale fall back to their raw ``[min, max]``."""
    deg = scale <= 1e-12
    lo = cp.where(deg, M.min(axis=0), med - _GPU_ROBUST_AXIS_K * scale)
    hi = cp.where(deg, M.max(axis=0), med + _GPU_ROBUST_AXIS_K * scale)
    return lo, hi


_HEAVY_TAIL_STATS_SRC = r"""
extern "C" __global__
void heavy_tail_stats(const double* __restrict__ M, const double* __restrict__ med,
                      const double* __restrict__ scale, const double outerK, const double gapthr,
                      const double maxfrac, const long long n, const int K,
                      signed char* __restrict__ out_v) {
    int col = blockIdx.x;
    if (col >= K) return;
    double mc = med[col];
    double sc = scale[col];
    double thr = outerK * sc;
    int tid = threadIdx.x;
    long long nloc = 0;
    double bulk = -1e308, omin = 1e308;
    for (long long i = tid; i < n; i += blockDim.x) {
        double dev = fabs(M[i * (long long)K + col] - mc);
        if (dev > thr) { nloc += 1; if (dev < omin) omin = dev; }
        else if (dev > bulk) bulk = dev;
    }
    __shared__ long long sh_n[256];
    __shared__ double sh_b[256];
    __shared__ double sh_o[256];
    sh_n[tid] = nloc; sh_b[tid] = bulk; sh_o[tid] = omin;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sh_n[tid] += sh_n[tid + s];
            if (sh_b[tid + s] > sh_b[tid]) sh_b[tid] = sh_b[tid + s];
            if (sh_o[tid + s] < sh_o[tid]) sh_o[tid] = sh_o[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        long long no = sh_n[0];
        double bk = sh_b[0] > 1e-12 ? sh_b[0] : 1e-12;
        double gap = sh_o[0] / bk;
        out_v[col] = ((sc > 1e-12) && (no > 0) && ((double)no <= maxfrac * (double)n) && (gap >= gapthr)) ? 1 : 0;
    }
}
"""
_HEAVY_TAIL_STATS_KERNEL = None


def _get_heavy_tail_stats_kernel(cp):
    """Lazily compile-and-cache the ``heavy_tail_stats`` RawKernel (per-column outlier-fraction/gap heavy-tail detector)."""
    global _HEAVY_TAIL_STATS_KERNEL
    if _HEAVY_TAIL_STATS_KERNEL is None:
        _HEAVY_TAIL_STATS_KERNEL = cp.RawKernel(_HEAVY_TAIL_STATS_SRC, "heavy_tail_stats")
    return _HEAVY_TAIL_STATS_KERNEL


def _gpu_detect_heavy_tail_batched(cp, M):
    """Vectorized per-column heavy-tail over a FINITE (n, K) matrix (caller skips non-finite cols).
    Returns (K,) cupy bool, matching _gpu_detect_heavy_tail per column.

    The post-quantile outlier stats + verdict (dev=|M-med|, outer mask, n_outer, bulk_edge max, outer_min,
    gap, the bool AND-chain - ~10 cuLaunchKernel) fuse into ONE block-per-column RawKernel: max/min reduce
    order-independently (bit-identical to cp.max/min), n_outer is an exact int, and thread 0 emits the final
    bool. Falls back to the cupy reduction chain on any kernel error."""
    n = int(M.shape[0])
    # med + q25 + q75 of M fold into ONE radix-select call (same M); MAD's median(|M-med|) stays separate
    # (needs med first). Bit-identical to cp.median(M)/cp.percentile(M,[25,75]).
    q = _batched_quantiles(cp, M, [0.25, 0.5, 0.75])
    med = q[1]
    scale = _gpu_robust_scale_batched(cp, M, med, q25=q[0], q75=q[2])
    try:
        Mc = cp.ascontiguousarray(M.astype(cp.float64, copy=False))
        K = int(Mc.shape[1])
        out_v = cp.empty(K, dtype=cp.int8)
        _get_heavy_tail_stats_kernel(cp)((K,), (256,),
            (Mc, cp.ascontiguousarray(med.astype(cp.float64, copy=False)),
             cp.ascontiguousarray(scale.astype(cp.float64, copy=False)),
             float(_GPU_ROBUST_AXIS_OUTER_K), float(_GPU_ROBUST_AXIS_GAP),
             float(_GPU_ROBUST_AXIS_MAX_FRAC), np.int64(n), np.int32(K), out_v))
        return out_v.astype(cp.bool_)
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("robust-axis-outlier RawKernel failed, falling back to the cupy chain: %s", e)
        dev = cp.abs(M - med)
        outer = dev > (_GPU_ROBUST_AXIS_OUTER_K * scale)
        n_outer = outer.sum(axis=0)
        bulk_edge = cp.where(outer, -cp.inf, dev).max(axis=0)
        outer_min = cp.where(outer, dev, cp.inf).min(axis=0)
        gap_ok = (outer_min / cp.maximum(bulk_edge, 1e-12)) >= _GPU_ROBUST_AXIS_GAP
        return (scale > 1e-12) & (n_outer > 0) & (n_outer <= _GPU_ROBUST_AXIS_MAX_FRAC * n) & gap_ok


_BASIS_PREPROCESS_SRC = r"""
extern "C" __global__
void basis_preprocess(const double* __restrict__ M, const double* __restrict__ med,
                      const double* __restrict__ scale, const double* __restrict__ mn,
                      const double* __restrict__ mx, const double* __restrict__ stdcol,
                      const int basis_code, const double Kc, const long long n, const int K,
                      double* __restrict__ out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (t >= total) return;
    int col = (int)(t % (long long)K);
    long long row = t / (long long)K;
    double x = M[row * (long long)K + col];
    double mc = med[col], sc = scale[col];
    bool deg = sc <= 1e-12;
    double lo = deg ? mn[col] : mc - Kc * sc;
    double hi = deg ? mx[col] : mc + Kc * sc;
    double z;
    if (basis_code == 0) {                                   // hermite robust z-score
        double std = (hi - lo) / 6.0;
        if (!(std > 1e-12)) std = stdcol[col] + 1e-12;       // cp.where(std>1e-12, std, M.std+1e-12)
        z = (x - mc) / std;
        z = z < -6.0 ? -6.0 : (z > 6.0 ? 6.0 : z);
    } else if (basis_code == 1) {                            // legendre/chebyshev min-max -> [-1,1]
        double span = hi - lo + 1e-12;
        z = 2.0 * (x - lo) / span - 1.0;
        z = z < -1.0 ? -1.0 : (z > 1.0 ? 1.0 : z);
    } else {                                                 // laguerre shift -> >= 0
        double hic = (hi - lo) + 1e-9;
        z = x - lo + 1e-9;
        z = z < 0.0 ? 0.0 : (z > hic ? hic : z);
    }
    out[row * (long long)K + col] = z;
}
"""
_BASIS_PREPROCESS_KERNEL = None


def _get_basis_preprocess_kernel(cp):
    """Lazily compile-and-cache the robust ``basis_preprocess`` RawKernel (per-basis clipped rescale: z-score/min-max/shift, robust lo/hi already computed)."""
    global _BASIS_PREPROCESS_KERNEL
    if _BASIS_PREPROCESS_KERNEL is None:
        _BASIS_PREPROCESS_KERNEL = cp.RawKernel(_BASIS_PREPROCESS_SRC, "basis_preprocess")
    return _BASIS_PREPROCESS_KERNEL


_BASIS_PREPROCESS_CODE = {"hermite": 0, "legendre": 1, "chebyshev": 1, "laguerre": 2}

# Non-robust preprocess fused: hermite z = (x-mean)/std (no clip); legendre/cheb z = 2(x-lo)/span - 1 (no
# clip); laguerre z = x-lo+1e-9. p0/p1 are the per-column params; ONE launch vs the cupy reduction+arith chain.
_BASIS_PREPROCESS_NR_SRC = r"""
extern "C" __global__
void basis_preprocess_nr(const double* __restrict__ M, const double* __restrict__ p0,
                         const double* __restrict__ p1, const int basis_code,
                         const long long n, const int K, double* __restrict__ out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (t >= total) return;
    int col = (int)(t % (long long)K);
    long long row = t / (long long)K;
    double x = M[row * (long long)K + col];
    double a = p0[col], b = p1[col];
    double z;
    if (basis_code == 0)       z = (x - a) / b;            // hermite z-score (b = std + 1e-12)
    else if (basis_code == 1)  z = 2.0 * (x - a) / b - 1.0; // legendre/cheb min-max (a = lo, b = span)
    else                       z = x - a + 1e-9;            // laguerre shift (a = lo)
    out[row * (long long)K + col] = z;
}
"""
_BASIS_PREPROCESS_NR_KERNEL = None


def _get_basis_preprocess_nr_kernel(cp):
    """Lazily compile-and-cache the non-robust ``basis_preprocess_nr`` RawKernel (unclipped per-basis rescale from precomputed p0/p1 params)."""
    global _BASIS_PREPROCESS_NR_KERNEL
    if _BASIS_PREPROCESS_NR_KERNEL is None:
        _BASIS_PREPROCESS_NR_KERNEL = cp.RawKernel(_BASIS_PREPROCESS_NR_SRC, "basis_preprocess_nr")
    return _BASIS_PREPROCESS_NR_KERNEL


_COL_MEANSTD_SRC = r"""
extern "C" __global__
void col_meanstd(const double* __restrict__ M, const long long n, const int K,
                 double* __restrict__ out_mean, double* __restrict__ out_std) {
    int col = blockIdx.x;
    if (col >= K) return;
    int tid = threadIdx.x, nt = blockDim.x;
    __shared__ double sh[256];
    double s = 0.0;
    for (long long i = tid; i < n; i += nt) s += M[i * (long long)K + col];
    sh[tid] = s; __syncthreads();
    for (int j = nt >> 1; j > 0; j >>= 1) { if (tid < j) sh[tid] += sh[tid + j]; __syncthreads(); }
    double mean = sh[0] / (double)n;
    __syncthreads();
    double d2 = 0.0;
    for (long long i = tid; i < n; i += nt) { double dv = M[i * (long long)K + col] - mean; d2 += dv * dv; }
    sh[tid] = d2; __syncthreads();
    for (int j = nt >> 1; j > 0; j >>= 1) { if (tid < j) sh[tid] += sh[tid + j]; __syncthreads(); }
    if (tid == 0) { out_mean[col] = mean; out_std[col] = sqrt(sh[0] / (double)n); }
}
"""
_COL_MEANSTD_KERNEL = None


def _col_meanstd(cp, Mc):
    """Per-column mean AND std (ddof=0, two-pass) of a C-contiguous (n, K) f64 matrix in ONE launch (vs
    Mc.mean(axis=0) + Mc.std(axis=0)). Same two-pass formula as cupy std -> rel ~1e-13 (basis path tol 1e-6)."""
    global _COL_MEANSTD_KERNEL
    if _COL_MEANSTD_KERNEL is None:
        _COL_MEANSTD_KERNEL = cp.RawKernel(_COL_MEANSTD_SRC, "col_meanstd")
    n, K = int(Mc.shape[0]), int(Mc.shape[1])
    mean = cp.empty(K, dtype=cp.float64)
    std = cp.empty(K, dtype=cp.float64)
    _COL_MEANSTD_KERNEL((K,), (256,), (Mc, np.int64(n), np.int32(K), mean, std))
    return mean, std


def _gpu_basis_preprocess_nonrobust_fused(cp, M, basis):
    """Non-robust preprocess for a (n, g) M sharing one basis, FUSED into a single kernel (the per-column
    mean/std or min/max reductions + the per-element transform) - bit-identical to the cupy chain."""
    code = _BASIS_PREPROCESS_CODE[basis]
    n, g = int(M.shape[0]), int(M.shape[1])
    Mc = cp.ascontiguousarray(M.astype(cp.float64, copy=False))
    if code == 0:                                  # hermite: (M-mean)/std
        _mean, _std = _col_meanstd(cp, Mc)         # mean + std in ONE launch
        p0 = _mean; p1 = _std + 1e-12
    elif code == 1:                                # legendre/cheb: 2(M-lo)/span - 1
        lo, hi = _col_minmax(cp, Mc)               # min + max in ONE launch (bit-identical)
        p0 = lo; p1 = (hi - lo) + 1e-12
    else:                                          # laguerre: M-lo+1e-9 (p1 unused)
        p0 = Mc.min(axis=0); p1 = p0
    out = cp.empty((n, g), dtype=cp.float64)
    total = n * g
    threads = 256
    _get_basis_preprocess_nr_kernel(cp)(((total + threads - 1) // threads,), (threads,),
        (Mc, cp.ascontiguousarray(p0.astype(cp.float64, copy=False)),
         cp.ascontiguousarray(p1.astype(cp.float64, copy=False)),
         np.int32(code), np.int64(n), np.int32(g), out))
    return out


_COL_MINMAX_SRC = r"""
extern "C" __global__
void col_minmax(const double* __restrict__ M, const long long n, const int K,
                double* __restrict__ out_min, double* __restrict__ out_max) {
    int col = blockIdx.x;
    if (col >= K) return;
    int tid = threadIdx.x, nt = blockDim.x;
    double lmin = 1e308, lmax = -1e308;
    for (long long i = tid; i < n; i += nt) { double v = M[i * (long long)K + col]; if (v < lmin) lmin = v; if (v > lmax) lmax = v; }
    __shared__ double sh_min[256];
    __shared__ double sh_max[256];
    sh_min[tid] = lmin; sh_max[tid] = lmax;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) { if (sh_min[tid + s] < sh_min[tid]) sh_min[tid] = sh_min[tid + s]; if (sh_max[tid + s] > sh_max[tid]) sh_max[tid] = sh_max[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { out_min[col] = sh_min[0]; out_max[col] = sh_max[0]; }
}
"""
_COL_MINMAX_KERNEL = None


def _col_minmax(cp, Mc):
    """Per-column min AND max of a C-contiguous (n, K) f64 matrix in ONE launch (vs Mc.min(axis=0) +
    Mc.max(axis=0)). min/max reduce order-independently -> bit-identical. Returns (mn (K,), mx (K,))."""
    global _COL_MINMAX_KERNEL
    if _COL_MINMAX_KERNEL is None:
        _COL_MINMAX_KERNEL = cp.RawKernel(_COL_MINMAX_SRC, "col_minmax")
    n, K = int(Mc.shape[0]), int(Mc.shape[1])
    mn = cp.empty(K, dtype=cp.float64)
    mx = cp.empty(K, dtype=cp.float64)
    _COL_MINMAX_KERNEL((K,), (256,), (Mc, np.int64(n), np.int32(K), mn, mx))
    return mn, mx


def _gpu_basis_preprocess_robust_fused(cp, M, basis):
    """Robust preprocess for a (n, g) M sharing one basis, FUSED into a single kernel (lo/hi from the
    median+scale, the std-fallback, and the per-basis clip/scale done per-element in ONE launch) instead of
    the cupy where/clip/arith chain. med + scale come from the radix-quantile path; mn/mx (and std for the
    hermite fallback) are per-column reductions. Bit-identical to the cupy chain (same f64 ops)."""
    med = _batched_quantiles(cp, M, [0.5])[0]
    scale = _gpu_robust_scale_batched(cp, M, med)
    code = _BASIS_PREPROCESS_CODE[basis]
    n, g = int(M.shape[0]), int(M.shape[1])
    Mc = cp.ascontiguousarray(M.astype(cp.float64, copy=False))
    mn, mx = _col_minmax(cp, Mc)  # min + max in ONE launch (bit-identical)
    stdcol = Mc.std(axis=0) if code == 0 else scale  # std only used by the hermite fallback; else unused
    out = cp.empty((n, g), dtype=cp.float64)
    total = n * g
    threads = 256
    _get_basis_preprocess_kernel(cp)(((total + threads - 1) // threads,), (threads,),
        (Mc, cp.ascontiguousarray(med.astype(cp.float64, copy=False)),
         cp.ascontiguousarray(scale.astype(cp.float64, copy=False)),
         cp.ascontiguousarray(mn.astype(cp.float64, copy=False)),
         cp.ascontiguousarray(mx.astype(cp.float64, copy=False)),
         cp.ascontiguousarray(stdcol.astype(cp.float64, copy=False)),
         np.int32(code), float(_GPU_ROBUST_AXIS_K), np.int64(n), np.int32(g), out))
    return out


def _gpu_basis_preprocess_batched(cp, M, basis, *, robust):
    """Vectorized per-basis preprocess over (n, g) M (all g columns share basis + robust). Returns Z (n, g)."""
    if basis in _BASIS_PREPROCESS_CODE:
        try:
            return _gpu_basis_preprocess_robust_fused(cp, M, basis) if robust else _gpu_basis_preprocess_nonrobust_fused(cp, M, basis)
        except Exception as e:  # nosec B110 - non-trivial body; best-effort/optional path, no module logger
            import logging
            logging.getLogger(__name__).debug("fused basis-preprocess kernel failed, falling back to the exact cupy chain: %s", e)
    if basis == "hermite":  # z-score
        if robust:
            center = _batched_quantiles(cp, M, [0.5])[0]
            scale = _gpu_robust_scale_batched(cp, M, center)
            lo, hi = _gpu_robust_lo_hi_batched(cp, M, center, scale)
            std = (hi - lo) / 6.0
            std = cp.where(std > 1e-12, std, M.std(axis=0) + 1e-12)
            return cp.clip((M - center) / std, -6.0, 6.0)
        mean = M.mean(axis=0); std = M.std(axis=0) + 1e-12
        return (M - mean) / std
    if basis in ("legendre", "chebyshev"):  # min-max -> [-1, 1]
        if robust:
            med = _batched_quantiles(cp, M, [0.5])[0]
            scale = _gpu_robust_scale_batched(cp, M, med)
            lo, hi = _gpu_robust_lo_hi_batched(cp, M, med, scale)
            span = hi - lo + 1e-12
            return cp.clip(2.0 * (M - lo) / span - 1.0, -1.0, 1.0)
        lo = M.min(axis=0); hi = M.max(axis=0); span = hi - lo + 1e-12
        return 2.0 * (M - lo) / span - 1.0
    if basis == "laguerre":  # shift -> >= 0
        if robust:
            med = _batched_quantiles(cp, M, [0.5])[0]
            scale = _gpu_robust_scale_batched(cp, M, med)
            lo, hi = _gpu_robust_lo_hi_batched(cp, M, med, scale)
            upper = hi - lo
            return cp.clip(M - lo + 1e-9, 0.0, upper + 1e-9)
        lo = M.min(axis=0)
        return M - lo + 1e-9
    raise ValueError(f"basis {basis!r} not GPU-ported")


def _gpu_evaluate_basis_matrix(cp, M, bases, degrees, *, robust_axis, heavy_host=None):
    """BATCHED device build. ``M`` is a finite (n, K) cupy operand matrix; ``bases`` a per-column basis
    list; ``degrees`` the degree sequence. Groups columns by (basis, robust-decision) and runs the
    preprocess + one-hot Clenshaw VECTORISED per group/degree. Returns ``(cand_matrix (n, total), meta)``
    where ``meta`` is a list of ``(col_idx, basis, degree)`` aligned with the candidate columns (any column
    whose basis is not GPU-ported is dropped - the caller host-fallbacks those). ``(None, [])`` if empty.

    ``heavy_host`` (optional (K,) bool): the per-column heavy-tail verdict. It is BASIS-INDEPENDENT (a
    function of M's values only), so a caller that evaluates the SAME M under several candidate bases (the
    routing sweep) computes it ONCE and passes it in - avoiding the 4x redundant cp.median/percentile
    reduction _gpu_detect_heavy_tail_batched would otherwise run per basis. Bit-identical (same M -> same
    verdict)."""
    _n, K = M.shape
    if heavy_host is None:
        if robust_axis:
            heavy_host = cp.asnumpy(_gpu_detect_heavy_tail_batched(cp, M))
        else:
            heavy_host = np.zeros(K, dtype=bool)
    groups: dict = {}
    for ci in range(K):
        groups.setdefault((bases[ci], bool(heavy_host[ci])), []).append(ci)
    cand_blocks: list = []
    meta: list = []
    for (basis, robust), idx in groups.items():
        clen = _PREWARP_CLENSHAW_GPU.get(basis)
        if clen is None:
            continue
        Mg = M[:, idx]
        Zg = _gpu_basis_preprocess_batched(cp, Mg, basis, robust=robust)
        # FUSED: one RawKernel builds B_d(Zg) for ALL degrees x columns (layout [d0: all cols][d1: ...]),
        # bit-identical to the per-degree clen() chain it replaces. Fallback to clen() per degree on miss.
        block = _clenshaw_basis_block_gpu(cp, Zg, basis, list(degrees))
        if block is not None:
            cand_blocks.append(block)
            for d in degrees:
                meta.extend((ci, basis, int(d)) for ci in idx)
        else:
            for d in degrees:
                coef = [0.0] * (int(d) + 1)
                coef[int(d)] = 1.0
                cand_blocks.append(clen(cp, Zg, coef))  # (n, len(idx))
                meta.extend((ci, basis, int(d)) for ci in idx)
    if not cand_blocks:
        return None, []
    return cp.ascontiguousarray(cp.concatenate(cand_blocks, axis=1)), meta


_ABS_CORR_SRC = r"""
extern "C" __global__
void abs_corr(const double* __restrict__ cand, const double* __restrict__ y, const long long n,
              const int m, const double Sy, const double Syy, double* __restrict__ out) {
    int c = blockIdx.x;
    if (c >= m) return;
    __shared__ double sh[3];                       // sum_x, sum_xx, sum_xy
    int tid = threadIdx.x, nt = blockDim.x;
    if (tid < 3) sh[tid] = 0.0;
    __syncthreads();
    double psx = 0.0, psxx = 0.0, psxy = 0.0;
    for (long long i = tid; i < n; i += nt) {
        double v = cand[i * (long long)m + c];
        double yy = y[i];
        psx += v; psxx += v * v; psxy += v * yy;
    }
    atomicAdd(&sh[0], psx); atomicAdd(&sh[1], psxx); atomicAdd(&sh[2], psxy);
    __syncthreads();
    if (tid == 0) {
        double nf = (double)n, sx = sh[0], sxx = sh[1], sxy = sh[2];
        double cov = nf * sxy - sx * Sy;
        double vx = nf * sxx - sx * sx;            // = n * sum((v-mean)^2)
        double vy = nf * Syy - Sy * Sy;
        double den = sqrt(vx * vy);
        double corr = den > 1e-300 ? fabs(cov / den) : 0.0;
        bool ok = isfinite(corr) && (sqrt(vx / nf) > 1e-12);   // cn = sqrt(sum cc^2) = sqrt(vx/n) > 1e-12
        out[c] = ok ? corr : -1.0;
    }
}
"""
_ABS_CORR_KERNEL = None


def _get_abs_corr_kernel():
    """Lazily compile-and-cache the ``abs_corr`` RawKernel (per-column |Pearson correlation| against a fixed target vector)."""
    global _ABS_CORR_KERNEL
    if _ABS_CORR_KERNEL is None:
        import cupy as cp
        _ABS_CORR_KERNEL = cp.RawKernel(_ABS_CORR_SRC, "abs_corr")
    return _ABS_CORR_KERNEL


def _gpu_batched_abs_corr(cp, cand, y_cont):
    """|Pearson corr| of every column of ``cand`` (n, m) with the (n,) continuous ``y_cont``, on device.
    Degenerate columns (std <= 1e-12) or non-finite results -> -1.0 (so the host argmax skips them, mirroring
    the host router's ``if std(v) < 1e-12: continue``). Same Pearson definition as np.corrcoef(v, y)[0,1].

    FUSED (launch-reduction): one RawKernel (one block per column, one-pass sums) computes corr =
    cov/sqrt(vx*vy) - algebraically identical to the centered cupy form - instead of the
    subtract/multiply/sum(axis) chain (~10 cuLaunchKernel). Parity <1e-6 vs the centered form (one-pass
    re-association), within the router's documented <1e-6 corr tolerance (the argmax/tie logic is unchanged).
    Falls back to the cupy chain on any kernel error."""
    try:
        cand2 = cand if cand.ndim == 2 else cand[:, None]
        n, m = int(cand2.shape[0]), int(cand2.shape[1])
        cand2 = cp.ascontiguousarray(cand2.astype(cp.float64, copy=False))
        yv = y_cont.astype(cp.float64, copy=False).ravel()
        Sy, Syy = (float(_s) for _s in cp.asnumpy(cp.stack([yv.sum(), (yv * yv).sum()])))  # one D2H for the pair
        out = cp.empty(m, dtype=cp.float64)
        _get_abs_corr_kernel()((m,), (256,), (cand2, yv, np.int64(n), np.int32(m), np.float64(Sy), np.float64(Syy), out), shared_mem=3 * 8)
        return out
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("abs-corr RawKernel failed, falling back to the cupy chain: %s", e)
        yc = y_cont - y_cont.mean()
        yn = cp.sqrt((yc * yc).sum())
        cc = cand - cand.mean(axis=0)
        cn = cp.sqrt((cc * cc).sum(axis=0))
        num = (cc * yc[:, None]).sum(axis=0)
        denom = cn * yn
        corr = cp.abs(cp.where(denom > 1e-300, num / denom, 0.0))
        return cp.where(cp.isfinite(corr) & (cn > 1e-12), corr, -1.0)


def _gpu_route_bases_batched(cp, M, y_cont, candidate_bases, degrees, *, robust_axis):
    """Device port of the no-aux ``basis_route_by_signal`` for ALL columns of finite (n, K) ``M`` at once.
    For each candidate basis it evaluates every column x degree on the device (reusing
    ``_gpu_evaluate_basis_matrix``), computes the |corr| vs ``y_cont``, then runs the EXACT host argmax
    (per-column ``bcorr = max over degrees`` with degenerate-skip, then first-basis-wins argmax over
    ``candidate_bases``). Returns a length-K list of chosen basis names, or ``None`` at a column index where
    no basis produced a usable expansion (caller host-fallbacks that column to basis_route_by_moments).

    Only the corr VALUES come from the GPU (parity-<1e-6); the argmax/tie logic is byte-identical to the host
    router (same loop order, same strict ``>``, same ``bcorr`` init 0.0), so a routing divergence can only
    arise from a genuine <1e-6 near-tie between two bases - exactly the case the opt-in default guards."""
    K = int(M.shape[1])
    # Heavy-tail verdict is BASIS-INDEPENDENT -> compute ONCE here and reuse for all candidate bases instead
    # of _gpu_evaluate_basis_matrix recomputing the cp.median/percentile reduction per basis (4x redundant).
    if robust_axis:
        _heavy = cp.asnumpy(_gpu_detect_heavy_tail_batched(cp, M))
    else:
        _heavy = np.zeros(K, dtype=bool)
    # corr_by_basis[basis] = (K,) host array of bcorr (max over degrees, degenerate-skipped), init 0.0
    corr_by_basis: dict = {}
    for basis in candidate_bases:
        bcorr = np.zeros(K, dtype=np.float64)
        cand, meta = _gpu_evaluate_basis_matrix(cp, M, [basis] * K, list(degrees), robust_axis=robust_axis, heavy_host=_heavy)
        if cand is not None:
            ac = cp.asnumpy(_gpu_batched_abs_corr(cp, cand, y_cont))  # (len(meta),)
            for j, (ci, _b, _d) in enumerate(meta):
                if ac[j] > bcorr[ci]:
                    bcorr[ci] = ac[j]
        corr_by_basis[basis] = bcorr
    chosen: list = []
    for ci in range(K):
        best_corr = -1.0
        best_basis = None
        for basis in candidate_bases:  # candidate_bases order -> first basis wins a tie (host parity)
            bc = float(corr_by_basis[basis][ci])
            if bc > best_corr:
                best_corr = bc
                best_basis = basis
        chosen.append(best_basis)
    return chosen
