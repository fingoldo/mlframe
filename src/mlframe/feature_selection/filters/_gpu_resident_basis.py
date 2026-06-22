"""GPU-resident FE: prewarp/orth-basis device build + grand-fusion pair-MI (Tier E carve).

Carved VERBATIM out of ``_gpu_resident_fe.py`` (sibling re-export pattern) to bring the parent under the
1k-LOC ceiling. Two self-contained blocks live here:

  * the GPU port of the per-operand PRE-WARP apply + the matrix-native orth-FE basis-column evaluation
    (the ``_*_clenshaw_gpu`` recurrences, ``_gpu_evaluate_basis_*``, ``_gpu_route_bases_batched``,
    ``_gpu_apply_prewarp`` -- all numpy/cupy, no parent dependency); and
  * the analytic ``gpu_pairs_fe_mi`` + its per-host KTC sweep, the grand-fusion pair-MI path
    (``grand_fused_pair_mi`` / ``grand_fused_pair_mi_fused``), the structured-recipe emitter
    (``gpu_resident_pair_recipes``), and the backend dispatcher (``pair_candidate_mi_dispatch``).

The gate helpers and the candidate-grid primitives stay in the PARENT and are imported below; the parent
re-exports every public/used name moved here so all ``from .._gpu_resident_fe import X`` paths still
resolve byte-for-byte. The few cross-sibling references (``_gpu_resident_discretize_codes`` /
``gpu_discretize_codes_host`` in ``_gpu_resident_select``) are LAZY-imported inside the function bodies to
avoid an import cycle. No kernel-source, dispatch-threshold, residency, or selection behavior changed.
"""
from __future__ import annotations

import os

import numpy as np

# Parent-defined names these blocks consume. Imported at module top: the PARENT does
# ``from ._gpu_resident_basis import *`` at its BOTTOM (after all these names are defined), so re-entering
# the partially-initialised parent here always finds them -- no circular-import hazard.
from ._gpu_resident_fe import (
    _BINOP_CODE,
    _COMBOS,
    _COMBO_IDX_CACHE,
    _GPU_RESIDENT_MIN_N,
    _UNARY_IDX,
    _binary_apply,
    _build_candidate_matrix,
    _candidate_names,
    _fused_generate_block,
    _get_fused_gen_bin_hist_kernel,
    _gpu_k_chunk,
    _quantile_levels_dev,
    _unary_apply,
    _unary_stack_cm,
    cpu_pair_candidate_mi,
    fe_gpu_grand_fusion_enabled,
)


# --- GPU port of the per-operand PRE-WARP apply (phase R1, 2026-06-21) ----------------------------------
# Mirrors hermite_fe.apply_operand_prewarp so the operand-table mirror can BUILD a prewarp operand column on
# the device (from the resident raw input + the tiny stored spec) instead of COPYING the host-computed column
# (the 1.68 MB non-plain H2D floor). The preprocess (z-score / min-max / shift, all elementwise + a clip) and
# the Clenshaw polynomial recurrences below replicate NUMPY's chebval/legval/hermeval(He)/lagval to fp
# round-off (same Clenshaw algorithm + float64 op order). SELECTION-EQUIVALENCE NOTE (P2-2): the production
# HOST path (polyeval_dispatch -> njit) evaluates these three (cheb/leg/herme) by a FORWARD recurrence, which
# differs from numpy/GPU-Clenshaw by ~1e-12 at degree>=3 (laguerre is forward on both, so it is bit-consistent
# across device+host). So a candidate MI-RANKED on GPU-Clenshaw values and later REPLAYED via host-forward
# differs by ~1e-12 -- far below any selection threshold (selection is decided on the consistent GPU values
# within the resident path; test_gpu_basis_column_parity pins the host<->GPU bound). Unifying both onto one
# recurrence is a FUTURE kernel change, unneeded at the default max_degree. Any unsupported basis / failure
# RAISES so the caller falls back to the host copy (never a correctness regression). fourier_adaptive
# (escalation only -- not in the main operand table) is ported too for completeness.

def _cheb_clenshaw_gpu(cp, x, c):
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
    # FORWARD recurrence ``out = sum_k c[k] L_k`` matching the host _lagval_njit EXACTLY
    # (L_0=1, L_1=1-x, L_k = ((2k-1-x)L_{k-1} - (k-1)L_{k-2})/k). The prior Clenshaw-style
    # recurrence here was WRONG for Laguerre (verified: L_2(0) gave -0.5 vs the correct 1) --
    # it was never exercised because the canonical prewarp uses the chebyshev basis, so no pin
    # caught it; the matrix-native basis parity test (test_gpu_basis_column_parity) surfaced it.
    nc = len(c)
    if nc == 0:
        return cp.zeros_like(x)
    out = cp.full(x.shape, c[0], dtype=x.dtype)
    if nc == 1:
        return out
    p_prev = cp.ones_like(x)        # L_0
    p_curr = 1.0 - x               # L_1
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

# --- GPU port of the orth-FE basis-column evaluation (matrix-native, Piece 2, 2026-06-21) -------------
# Faithful cupy mirror of _orthogonal_univariate_fe._evaluate_basis_column (no-aux, no-replay path):
# the robust heavy-tail axis detection (_hermite_robust._detect_heavy_tail_numpy/_robust_scale/
# _robust_lo_hi) + the per-basis preprocess (z-score / min-max / shift, robust + plain branches) + the
# one-hot Clenshaw eval (the _*_clenshaw_gpu above). Lets the orth-FE candidate matrix be built ON the
# device (operands resident) so it feeds _plugin_mi_classif_batch_cuda_resident with NO H2D -- removing
# the np.median (robust axis) + argsort/reduce (plug-in MI) of the CPU tail. Constants mirror
# _hermite_robust (K=3, OUTER_K=10, GAP=3, MAX_FRAC=0.20). cp.median/percentile match np to fp round-off;
# parity is asserted by test_gpu_basis_column_parity (uniform/gaussian/heavytail/skewed x 4 bases).
# These MIRROR hermite_fe._hermite_robust._ROBUST_AXIS_* (NOT imported: the GPU module avoids a top-level
# hermite_fe import -- the cycle the in-function lazy imports work around). Keep in sync; the drift is
# guarded by tests/feature_selection/gpu/test_gpu_cpu_robust_constants_in_sync.py.
_GPU_ROBUST_AXIS_K = 3.0
_GPU_ROBUST_AXIS_OUTER_K = 10.0
_GPU_ROBUST_AXIS_GAP = 3.0
_GPU_ROBUST_AXIS_MAX_FRAC = 0.20


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

def _gpu_robust_scale_batched(cp, M, med):
    """Per-column (axis=0) robust scale over finite (n, g) M: 1.4826*MAD, IQR/1.349 fallback, 0 degenerate."""
    mad = cp.median(cp.abs(M - med), axis=0)
    scale = 1.4826 * mad
    q = cp.percentile(M, cp.asarray([25.0, 75.0]), axis=0)  # (2, g)
    iqr = (q[1] - q[0]) / 1.349
    return cp.where(scale > 1e-12, scale, cp.where(iqr > 1e-12, iqr, 0.0))


def _gpu_robust_lo_hi_batched(cp, M, med, scale):
    deg = scale <= 1e-12
    lo = cp.where(deg, M.min(axis=0), med - _GPU_ROBUST_AXIS_K * scale)
    hi = cp.where(deg, M.max(axis=0), med + _GPU_ROBUST_AXIS_K * scale)
    return lo, hi


def _gpu_detect_heavy_tail_batched(cp, M):
    """Vectorized per-column heavy-tail over a FINITE (n, K) matrix (caller skips non-finite cols).
    Returns (K,) cupy bool, matching _gpu_detect_heavy_tail per column."""
    n = M.shape[0]
    med = cp.median(M, axis=0)
    scale = _gpu_robust_scale_batched(cp, M, med)
    dev = cp.abs(M - med)
    outer = dev > (_GPU_ROBUST_AXIS_OUTER_K * scale)
    n_outer = outer.sum(axis=0)
    bulk_edge = cp.where(outer, -cp.inf, dev).max(axis=0)
    outer_min = cp.where(outer, dev, cp.inf).min(axis=0)
    gap_ok = (outer_min / cp.maximum(bulk_edge, 1e-12)) >= _GPU_ROBUST_AXIS_GAP
    return (
        (scale > 1e-12) & (n_outer > 0)
        & (n_outer <= _GPU_ROBUST_AXIS_MAX_FRAC * n) & gap_ok
    )


def _gpu_basis_preprocess_batched(cp, M, basis, *, robust):
    """Vectorized per-basis preprocess over (n, g) M (all g columns share basis + robust). Returns Z (n, g)."""
    if basis == "hermite":  # z-score
        if robust:
            center = cp.median(M, axis=0)
            scale = _gpu_robust_scale_batched(cp, M, center)
            lo, hi = _gpu_robust_lo_hi_batched(cp, M, center, scale)
            std = (hi - lo) / 6.0
            std = cp.where(std > 1e-12, std, M.std(axis=0) + 1e-12)
            return cp.clip((M - center) / std, -6.0, 6.0)
        mean = M.mean(axis=0); std = M.std(axis=0) + 1e-12
        return (M - mean) / std
    if basis in ("legendre", "chebyshev"):  # min-max -> [-1, 1]
        if robust:
            med = cp.median(M, axis=0)
            scale = _gpu_robust_scale_batched(cp, M, med)
            lo, hi = _gpu_robust_lo_hi_batched(cp, M, med, scale)
            span = hi - lo + 1e-12
            return cp.clip(2.0 * (M - lo) / span - 1.0, -1.0, 1.0)
        lo = M.min(axis=0); hi = M.max(axis=0); span = hi - lo + 1e-12
        return 2.0 * (M - lo) / span - 1.0
    if basis == "laguerre":  # shift -> >= 0
        if robust:
            med = cp.median(M, axis=0)
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
    whose basis is not GPU-ported is dropped -- the caller host-fallbacks those). ``(None, [])`` if empty.

    ``heavy_host`` (optional (K,) bool): the per-column heavy-tail verdict. It is BASIS-INDEPENDENT (a
    function of M's values only), so a caller that evaluates the SAME M under several candidate bases (the
    routing sweep) computes it ONCE and passes it in -- avoiding the 4x redundant cp.median/percentile
    reduction _gpu_detect_heavy_tail_batched would otherwise run per basis. Bit-identical (same M -> same
    verdict)."""
    n, K = M.shape
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
        for d in degrees:
            coef = [0.0] * (int(d) + 1)
            coef[int(d)] = 1.0
            cand_blocks.append(clen(cp, Zg, coef))   # (n, len(idx))
            meta.extend((ci, basis, int(d)) for ci in idx)
    if not cand_blocks:
        return None, []
    return cp.ascontiguousarray(cp.concatenate(cand_blocks, axis=1)), meta


def _gpu_batched_abs_corr(cp, cand, y_cont):
    """|Pearson corr| of every column of ``cand`` (n, m) with the (n,) continuous ``y_cont``, on device.
    Degenerate columns (std <= 1e-12) or non-finite results -> -1.0 (so the host argmax skips them, mirroring
    the host router's ``if std(v) < 1e-12: continue``). Same Pearson definition as np.corrcoef(v, y)[0,1]."""
    yc = y_cont - y_cont.mean()
    yn = cp.sqrt((yc * yc).sum())
    cc = cand - cand.mean(axis=0)
    cn = cp.sqrt((cc * cc).sum(axis=0))            # (m,)
    num = (cc * yc[:, None]).sum(axis=0)           # (m,)
    denom = cn * yn
    corr = cp.where(denom > 1e-300, num / denom, 0.0)
    corr = cp.abs(corr)
    finite_ok = cp.isfinite(corr) & (cn > 1e-12)
    return cp.where(finite_ok, corr, -1.0)


def _gpu_route_bases_batched(cp, M, y_cont, candidate_bases, degrees, *, robust_axis):
    """Device port of the no-aux ``basis_route_by_signal`` for ALL columns of finite (n, K) ``M`` at once.
    For each candidate basis it evaluates every column x degree on the device (reusing
    ``_gpu_evaluate_basis_matrix``), computes the |corr| vs ``y_cont``, then runs the EXACT host argmax
    (per-column ``bcorr = max over degrees`` with degenerate-skip, then first-basis-wins argmax over
    ``candidate_bases``). Returns a length-K list of chosen basis names, or ``None`` at a column index where
    no basis produced a usable expansion (caller host-fallbacks that column to basis_route_by_moments).

    Only the corr VALUES come from the GPU (parity-<1e-6); the argmax/tie logic is byte-identical to the host
    router (same loop order, same strict ``>``, same ``bcorr`` init 0.0), so a routing divergence can only
    arise from a genuine <1e-6 near-tie between two bases -- exactly the case the opt-in default guards."""
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
            ac = cp.asnumpy(_gpu_batched_abs_corr(cp, cand, y_cont))   # (len(meta),)
            for j, (ci, _b, _d) in enumerate(meta):
                if ac[j] > bcorr[ci]:
                    bcorr[ci] = ac[j]
        corr_by_basis[basis] = bcorr
    chosen: list = []
    for ci in range(K):
        best_corr = -1.0
        best_basis = None
        for basis in candidate_bases:        # candidate_bases order -> first basis wins a tie (host parity)
            bc = float(corr_by_basis[basis][ci])
            if bc > best_corr:
                best_corr = bc
                best_basis = basis
        chosen.append(best_basis)
    return chosen


def gpu_pairs_fe_mi(cand: np.ndarray, quantization_nbins: int, classes_y: np.ndarray,
                    classes_y_safe: np.ndarray, freqs_y: np.ndarray, npermutations: int,
                    min_nonzero_confidence: float, use_su: bool):
    """Full GPU path for the FE pair-search candidate MI, for the ANALYTIC large-n branch only.

    Returns ``fe_mi[K]`` BIT-IDENTICAL to the production ``_dispatch_batch_mi_with_noise_gate`` analytic
    path, or ``None`` when that branch does not apply (SU-normalised, npermutations<=0, analytic
    disabled / inapplicable) so the caller falls back to the CPU dispatcher. Selection is preserved by
    construction:
      * GPU quantile binning == CPU ``discretize_2d_quantile_batch`` (verified maxdiff 0), and
      * the GPU observed-MI (npermutations=0) == the CPU kernel's observed MI (verified maxdiff 0;
        the GPU twin does only integer counting, entropy stays on the bit-exact CPU path),
    so feeding them through the SAME ``analytic_batch_noise_gate`` (chi2 keep/reject on the observed MI
    + per-column occupied-bin df) yields identical gated MI. Moves BOTH the binning and the observed-MI
    counting -- the dominant large-n per-pair cost -- onto the GPU. Any failure returns None (-> CPU)."""
    n, K = int(cand.shape[0]), int(cand.shape[1])
    if bool(use_su) or int(npermutations) <= 0:
        return None  # SU has no chi2 analytic form; npermutations<=0 is already the cheap CPU path
    try:
        from ._gpu_resident_select import gpu_discretize_codes_host  # lazy: cross-sibling, avoids cycle
        from ._analytic_mi_null import (
            analytic_batch_noise_gate, analytic_null_applicable, analytic_null_enabled,
        )
        by = int(np.unique(np.asarray(classes_y)).size)
        if not (analytic_null_enabled() and analytic_null_applicable(n, int(quantization_nbins), by)):
            return None  # sparse / small-n -> the asymptotic is unreliable; CPU permutation path
        from .batch_mi_noise_gate_gpu import dispatch_batch_mi_with_noise_gate_gpu

        codes = gpu_discretize_codes_host(cand, int(quantization_nbins), dtype=np.int8)  # bit-identical binning
        # (gpu_discretize_codes_host returns FILLED host codes -- this binning-only leg does not defer the
        # D2H, so the analytic dispatch below reads them directly without an ensure_host_codes_filled call.)
        fnb = np.full(K, int(quantization_nbins), dtype=np.int64)
        yc = np.ascontiguousarray(classes_y, dtype=np.int64)
        observed = None
        for _fb in ("cupy", "cuda"):
            observed = dispatch_batch_mi_with_noise_gate_gpu(
                disc_2d=codes, factors_nbins=fnb, classes_y=yc,
                classes_y_safe=np.ascontiguousarray(classes_y_safe), freqs_y=np.ascontiguousarray(freqs_y, dtype=np.float64),
                npermutations=0, base_seed=np.uint64(0), min_nonzero_confidence=float(min_nonzero_confidence),
                use_su=False, dtype=np.int32, force_backend=_fb,
            )
            if observed is not None:
                break
        if observed is None:
            return None  # no GPU backend -> CPU dispatcher
        observed = observed[0] if isinstance(observed, tuple) else observed
        # The analytic keep/reject is cheap CPU post-processing on the K-length observed MI (+ per-column
        # occupied-bin df from the codes), identical to what _dispatch_batch_mi_with_noise_gate runs.
        return analytic_batch_noise_gate(codes, np.asarray(observed, dtype=np.float64), yc, n,
                                         float(min_nonzero_confidence))
    except Exception:
        # Surface the cause (don't silently degrade to CPU forever): a real logic/shape/numeric bug in
        # the GPU path would otherwise be invisible -- the exact "GPU never helped" failure mode.
        import logging
        logging.getLogger(__name__).debug("gpu_pairs_fe_mi failed; CPU fallback", exc_info=True)
        return None


def _fe_gpu_pairs_mi_fallback_choice(n_rows: int, n_cols: int) -> str:
    """Pre-sweep crossover for the FE pair-MI GPU path: GPU only when the work size ``n_rows * n_cols``
    is large enough to amortise the per-pair H2D of the candidate matrix; CPU otherwise. Conservative so
    a small/mid fit stays on the CPU (never a regression) until the per-host sweep refines it. Env
    override ``MLFRAME_FE_GPU_DISCRETIZE_MIN_NK`` (default 2e6 ~= n=100k x K=20)."""
    try:
        min_nk = int(os.environ.get("MLFRAME_FE_GPU_DISCRETIZE_MIN_NK", "2000000"))
    except ValueError:
        min_nk = 2_000_000
    return "gpu" if int(n_rows) * int(n_cols) >= min_nk else "cpu"


def _make_fe_gpu_pairs_inputs(dims: dict) -> tuple:
    """Synthetic (cand_matrix, nbins, classes_y, freqs_y) for the crossover sweep -- an a**2/b pair so
    the analytic branch engages (n >= analytic_null_min_n)."""
    n = int(dims["n_rows"])
    rng = np.random.default_rng(0)
    a = rng.uniform(1.0, 5.0, n); b = rng.uniform(1.0, 5.0, n)
    cand = np.ascontiguousarray(_build_candidate_matrix(np, a, b)).astype(np.float32)
    np.nan_to_num(cand, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y = a ** 2 / b
    edges = np.quantile(y, np.linspace(0, 1, 21)[1:-1])
    yc = np.searchsorted(edges, y).astype(np.int64)
    fy = np.bincount(yc, minlength=int(yc.max()) + 1).astype(np.float64) / n
    return (cand, 20, yc, fy)


def _run_fe_gpu_pairs_mi_sweep() -> list:
    """Per-host CPU-vs-GPU crossover sweep for the FE pair-MI path -> backend_choice regions keyed on
    n_rows. Both variants take the SAME (cand, nbins, yc, fy) and pay their own discretize (+ the GPU
    H2D), so the timing is realistic; the GPU variant is bit-identical (verified) so equivalence holds at
    a tight tol. Skips silently (-> []) when CUDA is unavailable."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        if not is_cuda_available():
            return []
    except Exception:
        return []
    from pyutilz.dev.benchmarking import sweep_backend_grid
    from .discretization import discretize_2d_quantile_batch
    from .info_theory import batch_mi_with_noise_gate
    from ._feature_engineering_pairs._pairs_dispatch import _dispatch_batch_mi_with_noise_gate

    def _cpu(cand, nbins, yc, fy):
        disc = discretize_2d_quantile_batch(cand, n_bins=nbins, dtype=np.int8, assume_finite=True)
        return _dispatch_batch_mi_with_noise_gate(
            disc_2d=disc, quantization_nbins=nbins, classes_y=yc, classes_y_safe=yc, freqs_y=fy,
            npermutations=3, min_nonzero_confidence=0.0, use_su=False, batch_mi_kernel=batch_mi_with_noise_gate,
        )

    def _gpu(cand, nbins, yc, fy):
        return gpu_pairs_fe_mi(cand, nbins, yc, yc, fy, 3, 0.0, False)

    return sweep_backend_grid(
        {"cpu": _cpu, "gpu": _gpu},
        {"n_rows": [50_000, 100_000, 300_000]},  # GPU path engages only at n >= analytic_null_min_n
        _make_fe_gpu_pairs_inputs,
        reference="cpu", repeats=3, equiv_rtol=1e-9, equiv_atol=1e-12,
    )


def _fe_gpu_pairs_mi_code_version():
    try:
        from ._gpu_resident_select import _gpu_resident_discretize_codes, gpu_discretize_codes_host  # lazy: cross-sibling
        from pyutilz.performance.kernel_tuning.code_versioning import compute_code_version
        return compute_code_version(gpu_pairs_fe_mi, gpu_discretize_codes_host, _gpu_resident_discretize_codes)
    except Exception:
        return None


def fe_gpu_pairs_mi_backend_choice(n_rows: int, n_cols: int) -> str:
    """Per-host 'gpu' or 'cpu' for the FE pair-MI path via the shared get_or_tune orchestrator
    (per-host cache, code-version checked, background sweep, measurement-backed fallback). Never blocks
    the fit: async_sweep tunes off the hot path; the conservative fallback routes meanwhile."""
    try:
        # Under an explicit max_runtime_mins budget, skip the (blocking-on-first-use, CUDA-detected-regardless-of-
        # CUDA_VISIBLE_DEVICES) CPU-vs-GPU crossover sweep -- it runs the CPU+GPU variants at n up to 300k (tens of
        # seconds) and would blow a tiny budget. Route via the measurement-backed fallback instead; the sweep still runs
        # on a normal no-budget fit, so per-host tuning is unaffected.
        from ._fe_deadline import fe_budget_active
        if fe_budget_active():
            return _fe_gpu_pairs_mi_fallback_choice(n_rows, n_cols)
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
        res = KernelTuningCache.load_or_create().get_or_tune(
            "fe_gpu_pairs_mi",
            dims={"n_rows": int(n_rows)},
            tuner=_run_fe_gpu_pairs_mi_sweep,
            axes=["n_rows"],
            fallback={"backend_choice": _fe_gpu_pairs_mi_fallback_choice(n_rows, n_cols)},
            code_version=_fe_gpu_pairs_mi_code_version(),
            async_sweep=True,
        )
        bc = res if isinstance(res, str) else str((res or {}).get("backend_choice", "cpu"))
        return bc if bc in ("cpu", "gpu") else "cpu"
    except Exception:
        return _fe_gpu_pairs_mi_fallback_choice(n_rows, n_cols)


def ensure_fe_gpu_pairs_mi_tuning(force: bool = False):
    """Force-run + persist the FE pair-MI CPU-vs-GPU crossover sweep for this host (CLI refresh hook)."""
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
        cache = KernelTuningCache.load_or_create()
        if not force:
            existing = cache.get_regions("fe_gpu_pairs_mi")
            if existing:
                return existing
        regions = _run_fe_gpu_pairs_mi_sweep()
        if regions:
            cache.update("fe_gpu_pairs_mi", axes=["n_rows"], regions=regions,
                         code_version=_fe_gpu_pairs_mi_code_version())
        return regions
    except Exception:
        return None


def grand_fused_pair_mi(
    a, b, y_codes, classes_y_safe, freqs_y, *,
    nbins: int = 20, npermutations: int = 25, min_nonzero_confidence: float = 0.0, use_su: bool = False,
):
    """GRAND FUSION: GPU fused-generate candidates -> RESIDENT GPU discretize -> the EXISTING bit-identical
    GPU noise-gate (``batch_mi_noise_gate_gpu``). Returns the SAME noise-gated fe_mi[K] the production
    pair-search computes, but with generation+discretization+noise-gate all on the GPU. Only the small
    int8 disc crosses to host for the existing noise-gate (which does its own resident permutation
    counting). Bit-identical: GPU discretize == CPU discretize (verified maxdiff 0) and the GPU noise-gate
    is the production twin. VRAM-chunked. Returns ``(names, fe_mi)``.

    MEASURED (GTX 1050 Ti, K=384, nperm=25, BIT-IDENTICAL to the production ``_dispatch_batch_mi_with_
    noise_gate`` -- bit=True, argmax match):
      * vs the PRODUCTION dispatch (its analytic large-n gate): n=50k 3169->717ms 4.42x; n=200k
        13589->2948ms 4.61x. This is the honest, fair speedup.
      * vs a forced CPU PERMUTATION gate (which production AVOIDS at large n via the analytic shortcut):
        n=200k 53902->2753ms ~19.6x -- do NOT quote this as the production win; it is the permutation-path
        ceiling, shown only to locate where the time goes (the noise-gate dominates at K=384).
    The default chooser routes the gate to CPU on this host (a tuner mis-calibration: the GPU gate is
    ~15x faster on the permutation path); grand_fused forces cupy/cuda so the gate runs on the GPU.

    GRAND FUSION (2026-06-20, default ON via ``MLFRAME_FE_GPU_GRAND_FUSION``): when enabled this delegates
    to :func:`grand_fused_pair_mi_fused`, which NEVER materialises the (n,K) float candidate matrix, the
    (n,K) int codes, the (n,K) D2H disc, nor the noise-gate's (n,K) ``d_base`` / (rows*n*K) flat index --
    it fuses gen+bin+joint-histogram into ONE shared-mem-atomic RawKernel per chunk (recompute-not-store,
    Option F1 + roadmap #3). MEASURED (GTX 1050 Ti, K=384, nperm=25, BIT-IDENTICAL -- maxdiff 0, argmax
    match, vs this non-fused path): n=100k 2.16x + 3.0x less peak GPU mem; n=300k 2.15x + 2.75x; n=1M
    3.39x + 2.26x. Selection is EXACT (same percentile edges; only the data movement changes). Falls back
    to this exact non-fused body if the shared histogram (P1*nbins*K_y int32) exceeds the device's per-block
    shared-mem limit or any GPU error occurs."""
    if fe_gpu_grand_fusion_enabled():
        try:
            return grand_fused_pair_mi_fused(
                a, b, y_codes, classes_y_safe, freqs_y, nbins=nbins, npermutations=npermutations,
                min_nonzero_confidence=min_nonzero_confidence, use_su=use_su,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).info("grand-fusion fused path unavailable (%s); non-fused fallback", e)

    import cupy as cp

    from ._gpu_resident_select import _gpu_resident_discretize_codes  # lazy: cross-sibling, avoids cycle
    from . import hermite_fe as _hf  # noqa: F401 -- full-init parent before the GPU MI import cycle
    from .batch_mi_noise_gate_gpu import dispatch_batch_mi_with_noise_gate_gpu

    a_gpu = cp.asarray(a, dtype=cp.float64)
    b_gpu = cp.asarray(b, dtype=cp.float64)
    n = int(a_gpu.shape[0])
    ua_cm = _unary_stack_cm(cp, a_gpu)
    ub_cm = _unary_stack_cm(cp, b_gpu)
    y_i64 = np.ascontiguousarray(y_codes, dtype=np.int64)
    csafe = np.ascontiguousarray(classes_y_safe)
    fy = np.ascontiguousarray(freqs_y, dtype=np.float64)
    k_chunk = _gpu_k_chunk(n)
    parts: list[np.ndarray] = []
    for start in range(0, len(_COMBOS), k_chunk):
        block = _COMBOS[start:start + k_chunk]
        cand = _fused_generate_block(ua_cm, ub_cm, block)        # GPU gen (resident)
        disc_host = cp.asnumpy(_gpu_resident_discretize_codes(cand, nbins).astype(cp.int8))  # GPU disc -> small D2H
        del cand
        fnb = np.full(len(block), int(nbins), dtype=np.int64)
        # FORCE the GPU noise-gate: measured 15x faster than CPU here (K=384 n=200k: cupy 3093ms vs
        # CPU njit 46176ms -- the noise-gate is the REAL bottleneck, not gen/discretize). The default
        # chooser (_batch_mi_noise_gate_backend_choice) picks CPU on this host -- a tuner mis-calibration
        # (it under-rates the GPU gate, same class as the MI-dispatch issue); force cupy/cuda so the
        # grand-fused path actually runs the gate on the GPU. Falls back to CPU only if no GPU backend.
        out = None
        for _fb in ("cupy", "cuda"):
            out = dispatch_batch_mi_with_noise_gate_gpu(
                disc_2d=disc_host, factors_nbins=fnb, classes_y=y_i64, classes_y_safe=csafe, freqs_y=fy,
                npermutations=int(npermutations), base_seed=np.uint64(0),
                min_nonzero_confidence=float(min_nonzero_confidence), use_su=bool(use_su),
                dtype=np.int32, force_backend=_fb,
            )
            if out is not None:
                break
        if out is None:  # GPU noise-gate unavailable -> the always-correct CPU kernel on the same disc
            from .info_theory import batch_mi_with_noise_gate as _cpu_gate
            fe_mi = _cpu_gate(
                disc_2d=disc_host, factors_nbins=fnb, classes_y=y_i64, classes_y_safe=csafe, freqs_y=fy,
                npermutations=int(npermutations), base_seed=np.uint64(0),
                min_nonzero_confidence=float(min_nonzero_confidence), use_su=bool(use_su),
                dtype=np.int32, classes_dtype=np.int16,
            )
        else:
            fe_mi = out[0] if isinstance(out, tuple) else out
        parts.append(np.asarray(fe_mi, dtype=np.float64))
    return _candidate_names(), np.concatenate(parts) if parts else np.empty(0)


def _grand_fusion_block_counts(ua_cm, ub_cm, block, edges_int, y_all_dev, nbins, K_y, total_size):
    """Run the fused gen+bin+histogram kernel for one candidate ``block``, returning the (P1, total_size)
    int64 joint-count matrix on HOST. ``edges_int`` is the (blk, nbins-1) interior-edge matrix (the exact
    ``cp.percentile`` edges for this block), ``y_all_dev`` is the (P1, n) int32 device y-vectors. The
    candidate float matrix is NEVER stored: each cell is regenerated + binned + atomic-histogrammed inline.

    ``col_off[c] = c * nbins * K_y`` (uniform nbins per FE candidate); ``total_size = blk * nbins * K_y``."""
    import cupy as cp

    n = int(ua_cm.shape[1])
    K = int(len(block))
    # Same fit-invariant index trio as _fused_generate_block (block is a slice of the module constant
    # _COMBOS); reuse the shared cache to drop the per-chunk-per-pair list-comps + tiny H2D.
    _ck = tuple(block)
    _cc = _COMBO_IDX_CACHE.get(_ck)
    if _cc is None:
        ua_idx = cp.asarray(np.asarray([_UNARY_IDX[ua] for ua, _, _ in block], dtype=np.int32))
        ub_idx = cp.asarray(np.asarray([_UNARY_IDX[ub] for _, ub, _ in block], dtype=np.int32))
        bop = cp.asarray(np.asarray([_BINOP_CODE[bp] for _, _, bp in block], dtype=np.int32))
        _COMBO_IDX_CACHE[_ck] = (ua_idx, ub_idx, bop)
    else:
        ua_idx, ub_idx, bop = _cc
    col_off = cp.arange(K, dtype=cp.int64) * (int(nbins) * int(K_y))
    P1 = int(y_all_dev.shape[0])
    counts = cp.zeros((P1, int(total_size)), dtype=cp.int64)
    # ONE BLOCK PER CANDIDATE: shared-mem histogram is (P1, nbins, K_y) int32. Check it fits this device's
    # per-block shared-memory limit; if not, the caller must fall back (the host gates on this).
    hist_bytes = P1 * int(nbins) * int(K_y) * 4
    threads = 256
    _get_fused_gen_bin_hist_kernel()(
        (K,), (threads,),
        (ua_cm, ub_cm, ua_idx, ub_idx, bop, edges_int, col_off, y_all_dev,
         np.int64(n), np.int32(K), np.int32(int(nbins)), np.int32(int(K_y)),
         np.int32(P1), np.int64(int(total_size)), counts),
        shared_mem=hist_bytes,
    )
    out = cp.asnumpy(counts)
    del counts, ua_idx, ub_idx, bop, col_off
    return out


def grand_fused_pair_mi_fused(
    a, b, y_codes, classes_y_safe, freqs_y, *,
    nbins: int = 20, npermutations: int = 25, min_nonzero_confidence: float = 0.0, use_su: bool = False,
):
    """GRAND-FUSION (never materialise (n,K)): the fully-fused twin of :func:`grand_fused_pair_mi`.

    Collapses gen -> discretize -> noise-gate-counting into ONE histogram kernel per chunk. Pass 1 (per
    chunk, VRAM-governed) generates the (n, blk) candidate floats ONLY long enough to take the EXACT
    ``cp.percentile`` interior edges, then DISCARDS them (the edges -- (blk, nbins-1) -- are the only
    survivor). Pass 2 launches :func:`_grand_fusion_block_counts`: each (row, candidate) thread RE-generates
    its value, bins it against those exact edges (identical math -> identical codes -> SAME selection), and
    atomic-adds into the (P1, total_size) joint histogram for the original-y + every shuffled-y at once.
    The MI/SU + the noise-gate rejection are reduced from those integer counts on the bit-exact CPU path
    (``_mi_from_counts_cpu`` + ``_gate_from_mi``) -- so the returned fe_mi is BIT-IDENTICAL to
    ``grand_fused_pair_mi`` / the production gate, while the (n,K) float matrix, the (n,K) int codes, the
    (n,K) D2H disc, and the noise-gate's (n,K) ``d_base`` + (rows*n*K) flat index are ALL eliminated.
    Returns ``(names, fe_mi)``. Raises if cupy is unavailable (caller gates)."""
    import cupy as cp

    from .batch_mi_noise_gate_gpu import (
        _build_shuffle_matrix, _gate_from_mi, _mi_columns_from_counts_cpu,
    )

    a_gpu = cp.asarray(a, dtype=cp.float64)
    b_gpu = cp.asarray(b, dtype=cp.float64)
    n = int(a_gpu.shape[0])
    ua_cm = _unary_stack_cm(cp, a_gpu)
    ub_cm = _unary_stack_cm(cp, b_gpu)

    # y-vectors: row 0 = original y, rows 1.. = the Fisher-Yates shuffles (SAME host LCG the noise-gate
    # uses -> bit-identical permutation stream). Uploaded ONCE as (P1, n) int32 and shared by every chunk.
    y_orig = np.ascontiguousarray(y_codes, dtype=np.int64).reshape(1, n)
    K_y = int(np.asarray(freqs_y).shape[0])
    fy = np.ascontiguousarray(freqs_y, dtype=np.float64)
    nperm = int(npermutations) if npermutations and npermutations > 0 else 0
    if nperm > 0:
        shuf = _build_shuffle_matrix(np.asarray(classes_y_safe), np.uint64(0), nperm)
        y_all_host = np.empty((nperm + 1, n), dtype=np.int64)
        y_all_host[0, :] = y_orig[0, :]
        y_all_host[1:, :] = shuf.astype(np.int64)
    else:
        y_all_host = y_orig
    P1 = int(y_all_host.shape[0])
    y_all_dev = cp.asarray(np.ascontiguousarray(y_all_host, dtype=np.int32))

    # The shared-mem histogram is (P1, nbins, K_y) int32 per block; it must fit this device's per-block
    # shared-memory limit. If not (very high nperm / many y-classes / large nbins), raise so the caller
    # falls back to the non-fused exact path -- correctness over fusion. Reserve a little headroom.
    hist_bytes = P1 * int(nbins) * K_y * 4
    try:
        sm_limit = int(cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["sharedMemPerBlock"])
    except Exception:
        sm_limit = 48 * 1024
    if hist_bytes > sm_limit - 256:
        raise RuntimeError(
            f"grand-fusion shared histogram {hist_bytes}B exceeds device limit {sm_limit}B "
            f"(P1={P1}, nbins={nbins}, K_y={K_y}); caller falls back to the non-fused path"
        )

    # Binning working dtype: mirror _gpu_resident_discretize_codes (native f64 here -> bit-identical edges
    # to the non-fused grand-fusion path; MLFRAME_FE_GPU_BINNING_DTYPE=float32 forces f32 percentile).
    forced = os.environ.get("MLFRAME_FE_GPU_BINNING_DTYPE", "").strip().lower()
    work = cp.float32 if forced in ("float32", "f32", "single") else cp.float64
    qs = _quantile_levels_dev(cp, nbins, work)

    k_chunk = _gpu_k_chunk(n)
    original_mi_parts: list[np.ndarray] = []
    perm_mi_parts: list[list[np.ndarray]] = [[] for _ in range(nperm)]
    for start in range(0, len(_COMBOS), k_chunk):
        block = _COMBOS[start:start + k_chunk]
        blk = len(block)
        # PASS 1: transient generate -> exact percentile edges -> discard the float matrix.
        cand = _fused_generate_block(ua_cm, ub_cm, block)            # (n, blk) f64, transient
        if cand.dtype != work:
            cand = cand.astype(work, copy=False)
        if blk == 1:
            # cupy single-column percentile bug guard (mirror _gpu_resident_discretize_codes).
            bin_edges = cp.percentile(cand.ravel(), qs).reshape(-1, 1)  # (nbins+1, 1)
        else:
            bin_edges = cp.percentile(cand, qs, axis=0)             # (nbins+1, blk)
        del cand                                                    # (n,blk) float GONE before the hist pass
        # interior edges, transposed to (blk, nbins-1) row-major f64 for the kernel's per-candidate scan.
        edges_int = cp.ascontiguousarray(bin_edges[1:-1, :].T.astype(cp.float64))
        del bin_edges
        total_size = blk * int(nbins) * K_y
        counts = _grand_fusion_block_counts(ua_cm, ub_cm, block, edges_int, y_all_dev, nbins, K_y, total_size)
        del edges_int
        # CPU bit-exact reduction (identical to batch_mi_with_noise_gate_cupy). Use the BATCHED njit
        # _mi_columns_from_counts_cpu (one compiled call over all blk columns) instead of the per-(perm,k)
        # Python->njit dispatch -- bit-identical (same _mi_from_counts_cpu body per column) but removes the
        # blk*(nperm+1) Python-call overhead. ref_mi reproduces the perm-skip: all-positive for the original
        # row (compute every column), then `om` for each perm row (compute only where om>0; else stays 0).
        nb_ky = int(nbins) * K_y
        _col_off = np.arange(blk, dtype=np.int64) * nb_ky
        _nbins_arr = np.full(blk, int(nbins), dtype=np.int64)
        _all_pos = np.ones(blk, dtype=np.float64)
        om = _mi_columns_from_counts_cpu(
            np.ascontiguousarray(counts[0]), _col_off, _nbins_arr, K_y, fy, n, bool(use_su), _all_pos,
        )
        original_mi_parts.append(om)
        for p in range(nperm):
            mp = _mi_columns_from_counts_cpu(
                np.ascontiguousarray(counts[p + 1]), _col_off, _nbins_arr, K_y, fy, n, bool(use_su), om,
            )
            perm_mi_parts[p].append(mp)
        del counts
    original_mi = np.concatenate(original_mi_parts) if original_mi_parts else np.empty(0)
    perm_mis = [np.concatenate(perm_mi_parts[p]) for p in range(nperm)] if original_mi_parts else []
    fe_mi = _gate_from_mi(original_mi, perm_mis, nperm, float(min_nonzero_confidence))
    return _candidate_names(), fe_mi


def _log_shift_anchor(operand_vals: np.ndarray, unary_name: str):
    """Frozen smart_log shift for a ``log`` side -- ``(1e-5 - nanmin)`` if the FULL column reaches <=0,
    else 0.0 (mirrors _step_core._ls_anchor exactly so replay is byte-identical). None for non-log."""
    if unary_name != "log":
        return None
    mn = float(np.nanmin(np.asarray(operand_vals, dtype=np.float64)))
    return (1e-5 - mn) if mn <= 0 else 0.0


def gpu_resident_pair_recipes(
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    y_codes: np.ndarray,
    *,
    src_a_name: str,
    src_b_name: str,
    cols_names,
    unary_preset: str = "minimal",
    binary_preset: str = "minimal",
    quantization_nbins=None,
    quantization_method=None,
    quantization_dtype=np.float32,
    top_k: int = 1,
    nbins: int = 20,
):
    """Score a pair's candidate grid on the GPU and return the top-``top_k`` as STRUCTURED, replayable
    ``EngineeredRecipe`` objects -- the bridge from this path's flat (name, MI) to what production FE
    consumes. For each winner it emits, via the SAME builders the CPU path uses
    (``get_new_feature_name`` + ``build_unary_binary_recipe``): the canonical name, the structured
    (src column names, unary names, binary name), the active presets (frozen for replay-stable semantics),
    the quantization params with fit-time edges PINNED (``fit_values_for_edges`` -> leak-free transform),
    and the frozen ``log_shift`` anchor for any ``log`` side. So the GPU result is a first-class recipe
    that ``transform()`` replays bit-identically on raw inputs -- not a string to be re-parsed.

    Returns a list of ``(name, EngineeredRecipe, mi)`` sorted by descending MI. The MI uses the exact
    GPU-resident path (``gpu_resident_pair_candidate_mi``); the recipe fields are built on CPU (cheap for
    top_k winners). Combo order is ``_COMBOS`` (kept in sync with the minimal preset)."""
    from .engineered_recipes import build_unary_binary_recipe
    from .feature_engineering import get_new_feature_name

    # Route through the dispatcher so this works on ANY backend (GPU-resident in the sweet spot, CPU
    # otherwise) -- recipe emission is backend-agnostic.
    names, mi = pair_candidate_mi_dispatch(a_vals, b_vals, y_codes, nbins=nbins)
    a64 = np.ascontiguousarray(a_vals, dtype=np.float64)
    b64 = np.ascontiguousarray(b_vals, dtype=np.float64)
    cols = list(cols_names)
    idx_a = cols.index(src_a_name)
    idx_b = cols.index(src_b_name)
    out = []
    for ci in np.argsort(mi)[::-1][: int(top_k)]:
        ua, ub, bop = _COMBOS[int(ci)]
        fe_tuple = (((idx_a, ua), (idx_b, ub)), bop, 0)
        name = get_new_feature_name(fe_tuple, cols)
        # Continuous fit-time engineered column (for edge pinning) -- identical op chain as the GPU path.
        fit_vals = _binary_apply(np, bop, _unary_apply(np, ua, a64), _unary_apply(np, ub, b64))
        fit_vals = np.nan_to_num(np.asarray(fit_vals, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        recipe = build_unary_binary_recipe(
            name=name,
            src_a_name=src_a_name, src_b_name=src_b_name,
            unary_a_name=ua, unary_b_name=ub, binary_name=bop,
            unary_preset=unary_preset, binary_preset=binary_preset,
            quantization_nbins=quantization_nbins,
            quantization_method=quantization_method,
            quantization_dtype=quantization_dtype,
            fit_values_for_edges=fit_vals,
            log_shift_a=_log_shift_anchor(a64, ua),
            log_shift_b=_log_shift_anchor(b64, ub),
        )
        out.append((name, recipe, float(mi[int(ci)])))
    return out


def pair_candidate_mi_dispatch(a: np.ndarray, b: np.ndarray, y_codes: np.ndarray, *, nbins: int = 20):
    """Route a pair's candidate-MI to the measured-fastest backend: GPU-resident in the sweet spot
    (cupy present, n >= the crossover, VRAM-chunked so it can't thrash), CPU njit otherwise. Returns
    ``(names, mi)`` identical in shape/order to both paths. The default FE pipeline does NOT call this
    yet (gated prototype); it is the dispatcher the production wiring will use."""
    n = int(np.asarray(a).shape[0])
    # Per-host crossover via the shared kernel_tuning_cache (mirrors the FE pair-MI path) rather than the
    # hardcoded 50k: the per-host CPU-vs-GPU sweep decides, and _GPU_RESIDENT_MIN_N is only the source-code
    # FALLBACK (inside _fe_gpu_pairs_mi_fallback_choice) when the cache is cold / lookup fails. Honours the
    # project rule against hardcoded GPU thresholds; the 50k stays as the conservative cold-start default.
    try:
        _use_gpu = fe_gpu_pairs_mi_backend_choice(n, len(_COMBOS)) == "gpu"
    except Exception:
        _use_gpu = n >= _GPU_RESIDENT_MIN_N
    if _use_gpu:
        try:
            import cupy  # noqa: F401

            # Resolve via the parent module (where this function is re-exported) so a monkeypatch of
            # ``gpu_resident_pair_candidate_mi`` on the parent -- the canonical patch target -- is honoured
            # after the Tier E carve (the name is defined in the parent; this call lives in the sibling).
            from . import _gpu_resident_fe as _parent
            return _parent.gpu_resident_pair_candidate_mi(a, b, y_codes, nbins=nbins)
        except Exception as e:
            # Log (don't silently swallow) -- a GPU OOM/driver error degrading to a slow CPU fallback
            # would otherwise look like "GPU never helped". A chunk-shrink-retry before CPU fallback is
            # a future refinement (the VRAM governor already bounds chunks, so OOM should be rare).
            import logging
            logging.getLogger(__name__).warning("GPU-resident pair MI failed (%s); CPU fallback.", e)
    return cpu_pair_candidate_mi(a, b, y_codes, nbins=nbins)
