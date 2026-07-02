"""Extra-basis univariate FE: B-spline + (adaptive / chirp) Fourier columns.

Complementary to the orthogonal-polynomial univariate path: each source column
can emit cubic B-spline basis columns (sharp local threshold rules) and Fourier
sin/cos columns (periodic / chirp patterns), with an optional held-out-validated
adaptive-frequency + quadratic-argument-chirp detector. The shared univariate
scaffolding (``_dedup_collinear_source_cols``, ``score_features_by_mi_uplift``)
lives in the parent module ``_orthogonal_univariate_fe`` and is lazy-imported
in-body to avoid a cycle.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import numba
import numpy as np
from numba import prange

from ..hermite_fe import _detect_heavy_tail, _robust_axis_enabled, _robust_lo_hi

logger = logging.getLogger(__name__)

__all__ = [
    "generate_extra_basis_features",
    "hybrid_orth_extra_basis_fe_with_recipes",
]


# Backlog #13 (2026-06-09): ``"wavelet"`` adds the Haar / localized
# multiresolution basis alongside the global Fourier + fixed-knot spline. Its
# legs are held-out-scale-selected in ``_wavelet_basis_fe`` and emitted here so
# the extra-basis path (``fe_hybrid_orth_enable`` / ``extra_bases``) can route
# them through the same MI-uplift gate. The standalone default-on stage
# (``fe_wavelet_enable``) reuses the same generator + recipe builder.
_EXTRA_BASIS_KINDS = ("spline", "fourier", "wavelet")


def _fit_spline_for_col(x: np.ndarray, n_inner_knots: int):
    """Returns (knots, lo, hi, num_basis_cols). Lazy delegate to recipes
    module so the knot-vector layout stays in one place.

    Knots are placed at QUANTILES of x (unsupervised). bench-rejected
    (2026-06-03): a TARGET-SUPERVISED knot strategy (knots at a shallow x->y
    tree's splits / conditional-mean curvature) was benchmarked and REJECTED.
    (1) In the real MRMR pipeline NO spline column -- quantile OR supervised --
    survives the MI-uplift gate, so the support is byte-identical with either
    strategy (the gate, not knot placement, is the binding constraint; and
    supervised knots score LOWER at the gate -- narrower, individually-lower-MI
    basis columns). (2) Even at the raw block level the win reverses by shape:
    supervised wins a narrow bump (|corr| 0.884 vs 0.614) but LOSES a sharp step
    (0.793 vs 0.931) and kink (0.719 vs 0.933). (3) The one shape it helps is
    already recovered by the default-on Fourier multitone. Leak-safety would have
    held (knots baked into the recipe, replay reads only knots/lo/hi), but moot.
    Don't add fe_spline_knot_strategy="supervised". (D:/Temp/item7_supervised_knots_findings.md)
    """
    from ..engineered_recipes import _bspline_basis_values, _fit_spline_knots  # noqa: F401
    knots, lo, hi = _fit_spline_knots(x, n_inner_knots, degree=3)
    # Number of cubic B-spline basis functions = len(knots) - degree - 1.
    n_basis = len(knots) - 3 - 1
    return knots, lo, hi, n_basis


def _fit_fourier_for_col(x: np.ndarray):
    """Returns (lo, span). Min-max normalise x so one period covers data range.

    OUTLIER-ROBUST (gated): on a heavy-tailed column the raw min/max span blows up ~1000x, so 99% of the data collapses
    into a sliver of one period near z=0 -- the sin/cos legs go flat there and carry an outlier-inflated MI while a single
    new extreme value in production shifts the whole axis. When the shared heavy-tail detector trips, fit lo/span from the
    inner-quantile core instead so the bulk of the data spans a full period; the few clipped extremes fall outside [0, 1]
    and sin/cos simply wrap them (bounded, harmless), no longer stretching the scale for everyone. Byte-identical to the
    legacy raw min/max path on clean columns (gate OFF). The returned (lo, span) are baked into the recipe and replayed
    verbatim at transform time, so fit and replay stay consistent automatically."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return 0.0, 1.0
    if _robust_axis_enabled() and _detect_heavy_tail(x):
        lo, hi = _robust_lo_hi(x)
        span = max(hi - lo, 1e-12)
        return lo, span
    lo = float(np.min(x[finite]))
    hi = float(np.max(x[finite]))
    span = max(hi - lo, 1e-12)
    return lo, span


def _fit_chirp_warp_for_col(x: np.ndarray):
    """Fit the QUADRATIC-ARGUMENT ("chirp") warp params on ``x`` (2026-06-03).

    The chirp axis is ``u = sign(z) * z**2`` where ``z = (x - mean) / std``.
    Squaring the STANDARDISED z (signed, so the map stays monotone and one-to-one
    across the whole real line rather than folding ``x`` and ``-x`` together)
    turns an oscillation whose frequency GROWS with the argument
    (``y ~ sin(2*pi*f*z**2)``) into a STATIONARY-frequency sinusoid in ``u`` --
    which the existing periodogram peak-search then locks onto. A Fourier on the
    LINEAR argument cannot represent a frequency that grows with z (Phase-0 bench:
    linear multitone R^2 0.07-0.53 vs chirp warp 0.88 on a fast chirp f=2.5).

    Returns ``(mean, std, lo, span)``:
    * ``mean`` / ``std`` -- standardisation of x into z (fit on the finite subset).
    * ``lo`` / ``span``  -- min / range of ``u`` so ``(u - lo) / span`` lands the
      warped axis in [0, 1], matching the linear emitter's z normalisation so the
      same coarse frequency grid applies.

    All four are baked into the recipe at fit time and replayed verbatim at
    transform time (no y, so leakage-free)."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return 0.0, 1.0, 0.0, 1.0
    xf = x[finite]
    mean = float(np.mean(xf))
    std = float(np.std(xf))
    std = std if std > 1e-12 else 1.0
    z = (xf - mean) / std
    u = np.sign(z) * (z * z)
    lo = float(np.min(u))
    hi = float(np.max(u))
    span = max(hi - lo, 1e-12)
    return mean, std, lo, span


@numba.njit(cache=True)
def _chirp_axis_njit(x: np.ndarray, mean: float, std: float, lo: float, span: float) -> np.ndarray:
    """Fused single-pass core of :func:`_chirp_axis`. NO fastmath (the ops -- subtract,
    divide, sign, square, divide -- run in the SAME IEEE order as the numpy expression), so
    the result is BIT-IDENTICAL to the numpy path (verified 0.0 max-abs-diff over 1e5 random
    rows); the win is dropping the three length-n numpy temporaries (z, u, and the two
    intermediate arrays) for one fused C loop."""
    ds = std if std > 1e-12 else 1e-12
    sp = span if span > 1e-12 else 1e-12
    out = np.empty(x.shape[0], dtype=np.float64)
    for i in range(x.shape[0]):
        z = (x[i] - mean) / ds
        u = np.sign(z) * (z * z)
        out[i] = (u - lo) / sp
    return out


def _chirp_axis(x: np.ndarray, mean: float, std: float, lo: float, span: float) -> np.ndarray:
    """Apply the stored chirp warp: x -> z=(x-mean)/std -> u=sign(z)*z**2 ->
    (u-lo)/span. Pure function of the fit-time params -- the single source of
    truth shared by fit-time detection (``generate_extra_basis_features``) and
    transform-time replay (``_apply_orth_fourier``) so both produce a
    bit-identical axis. The per-element math is a fused njit loop
    (:func:`_chirp_axis_njit`, no fastmath) -- bit-identical to the numpy
    expression but with no length-n intermediate temporaries."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    return _chirp_axis_njit(x, float(mean), float(std), float(lo), float(span))


@numba.njit(parallel=True, fastmath=True, cache=True)
def _coarse_basis_njit(z: np.ndarray, freqs: np.ndarray) -> tuple:
    """Build the per-frequency centered sin/cos coarse basis in ONE fused prange-over-freqs pass.

    Returns ``(sin_centered (nf,n), cos_centered (nf,n), sin_ss (nf,), cos_ss (nf,))`` -- the same four quantities the
    numpy build loop produces per grid frequency, but with the sin/cos transcendentals + the mean / sum-of-squares
    reductions fused into a single njit kernel parallelised across the frequencies. The sequential per-element reduction
    differs from numpy's pairwise summation by ~1e-13 (single-ULP class), so this is dispatched only on the held-out
    Fourier detector's coarse sweep, whose ``best_f`` argmax is re-localised by ``_refine_peak_freq`` and whose
    end-to-end MRMR selection is verified byte-identical; the exact numpy path stays as the fallback."""
    nf = freqs.shape[0]
    n = z.shape[0]
    sc = np.empty((nf, n))
    cc = np.empty((nf, n))
    sss = np.empty(nf)
    css = np.empty(nf)
    for fi in prange(nf):
        f = freqs[fi]
        smean = 0.0
        cmean = 0.0
        sbuf = np.empty(n)
        cbuf = np.empty(n)
        for i in range(n):
            ang = 2.0 * np.pi * f * z[i]
            s = np.sin(ang)
            c = np.cos(ang)
            sbuf[i] = s
            cbuf[i] = c
            smean += s
            cmean += c
        smean /= n
        cmean /= n
        s_ss = 0.0
        c_ss = 0.0
        for i in range(n):
            sv = sbuf[i] - smean
            cv = cbuf[i] - cmean
            sc[fi, i] = sv
            cc[fi, i] = cv
            s_ss += sv * sv
            c_ss += cv * cv
        sss[fi] = s_ss
        css[fi] = c_ss
    return sc, cc, sss, css


@numba.njit(fastmath=True, cache=True)
def _corr_sq_reductions_njit(v: np.ndarray, y_centered: np.ndarray) -> tuple:
    """Fuse the three ``_corr_sq_centered`` reductions -- ``sum(v)``, ``v@v``,
    ``v@y_centered`` -- into ONE sequential pass over ``v``/``y_centered``.

    The numpy form ran three separate reductions, and the 1-D ``v @ v`` / ``v @ y``
    dispatch to threaded BLAS whose per-call thread spin-up dominates at the periodogram
    call volume (a ~50x cliff at n~20k). One njit walk is 3-54x faster across n=1.6k..100k,
    bit-close to ~1e-14 (reduction-order single-ULP, far below any frequency-rank scale)."""
    n = v.shape[0]
    sv = 0.0
    vv = 0.0
    vy = 0.0
    for i in range(n):
        x = v[i]
        sv += x
        vv += x * x
        vy += x * y_centered[i]
    return sv, vv, vy


def _corr_sq_centered(v: np.ndarray, y_centered: np.ndarray, y_ss: float) -> float:
    """Squared Pearson correlation of ``v`` with a pre-centered ``y`` whose
    sum-of-squares is ``y_ss``. Avoids ``np.corrcoef`` (2x2-matrix build + two
    std passes) -- a direct centered dot product. Returns 0.0 on a degenerate
    ``v``.

    Computes the centered SS / numerator from RAW ``v`` dot products so no
    length-n ``v - v.mean()`` temporary is allocated: ``v_ss = v@v - sum(v)^2/n``,
    and ``num = v @ y_centered`` is IDENTITY-equal to the centered ``vc @ y_centered``
    because ``y_centered`` sums to zero (the ``v.mean()*sum(y_centered)`` cross
    term vanishes). The three reductions are fused into one njit pass
    (:func:`_corr_sq_reductions_njit`); the reduction-order shift is ~1e-14 (single
    ULP), far below any selection-altering scale."""
    n = v.shape[0]
    sv, vv, vy = _corr_sq_reductions_njit(np.ascontiguousarray(v, dtype=np.float64), y_centered)
    v_ss = vv - sv * sv / n
    # RELATIVE degeneracy guard (P1-4): the raw-moment form ``vv - sv^2/n`` catastrophically cancels for a
    # near-constant ``v`` -- it can land at a tiny positive residual (e.g. 1e-23) that clears an absolute
    # 1e-24 floor yet makes ``(vy^2)/(v_ss*y_ss)`` explode past 1.0, letting a degenerate column win the
    # periodogram. A genuinely varying v has ``v_ss`` an O(1) fraction of ``vv``; cancellation gives
    # ``v_ss << vv``. Reject when the centered SS is a negligible fraction of the raw SS.
    if v_ss <= 1e-12 * vv or v_ss < 1e-24 or y_ss < 1e-24:
        return 0.0
    return (vy * vy) / (v_ss * y_ss)


def _periodogram_power(z01: np.ndarray, y: np.ndarray, freq: float) -> float:
    """Phase-invariant periodogram power of ``y`` at z-space frequency ``freq``.

    ``corr(sin(2*pi*freq*z), y)^2 + corr(cos(2*pi*freq*z), y)^2`` -- the sum of
    the squared linear correlations of the sin and cos projections. Phase-
    invariant because a pure ``sin(2*pi*freq*z + phi)`` decomposes into a
    sin + cos mix whose combined power is independent of phi. Returns 0.0 when
    either projection degenerates (constant), so a frequency whose sin/cos
    collapse over the slice never wins.

    Convenience wrapper that centers ``y`` once; the hot per-column loops call
    :func:`_corr_sq_centered` directly with a pre-centered ``y`` to skip the
    redundant centering on every frequency.
    """
    yc = y - y.mean()
    y_ss = float(yc @ yc)
    if y_ss < 1e-24:
        return 0.0
    ang = 2.0 * np.pi * float(freq) * z01
    return (
        _corr_sq_centered(np.sin(ang), yc, y_ss)
        + _corr_sq_centered(np.cos(ang), yc, y_ss)
    )


# Parallel path wins from ~n>=4k (thread spawn amortised by the per-element transcendental work); below it the
# serial numpy ufunc path is faster (see the bench-rejection note in _power_centered).
_POWER_CENTERED_PAR_MIN_N = 4000


# Fixed contiguous-block count for the parallel periodogram reduction. Process- AND thread-count-STABLE:
# the row range is split into a constant ``_POWER_CENTERED_PAR_NBLOCKS`` contiguous blocks regardless of how
# many numba threads run, each block sums its six accumulators SERIALLY (so a block's partial is exact for that
# block), the prange parallelises ACROSS blocks, and the per-block partials are combined in a FIXED 0..NB-1
# order afterwards. Because both the within-block order and the cross-block combine order are independent of the
# thread schedule, the float result is bit-identical across thread counts / process starts -- unlike a numba
# auto-reduction over ``prange(n)``, whose per-thread partial-combine order drifts in the low ULPs with the
# thread count and silently flips razor-tie frequency-rank selections downstream (_refine_peak_freq argmax).
_POWER_CENTERED_PAR_NBLOCKS = 64


@numba.njit(fastmath=True, parallel=True, cache=True)
def _power_centered_fused_par_njit(z: np.ndarray, yc: np.ndarray, y_ss: float, freq: float) -> float:
    """Periodogram power = corr(sin)^2 + corr(cos)^2 with sin/cos + both centered-SS reductions fused, no
    length-n temporaries. Parallel over a FIXED number of contiguous row-blocks (not a numba auto-reduction
    over ``prange(n)``): each block sums serially into a private partial row, then the partials combine in a
    fixed block order. The result is bit-IDENTICAL across thread counts / process starts (deterministic float
    reduction order), so the downstream razor-tie frequency argmax (_refine_peak_freq) is process-stable.
    Bit-close to ~1e-15 (reduction-order) of the numpy-sin/cos + _corr_sq_centered path -- far below any
    frequency-rank scale."""
    n = z.shape[0]
    tp = 2.0 * np.pi * freq
    nblocks = _POWER_CENTERED_PAR_NBLOCKS
    if nblocks > n:
        nblocks = n
    if nblocks < 1:
        nblocks = 1
    # 6 partial accumulators per block: sums, ss_s, sy, sumc, ss_c, cy.
    partials = np.zeros((nblocks, 6))
    base = n // nblocks
    rem = n % nblocks
    for b in prange(nblocks):
        # Contiguous, thread-independent block bounds: the first ``rem`` blocks take one extra row.
        if b < rem:
            start = b * (base + 1)
            stop = start + (base + 1)
        else:
            start = rem * (base + 1) + (b - rem) * base
            stop = start + base
        psums = 0.0; pss_s = 0.0; psy = 0.0
        psumc = 0.0; pss_c = 0.0; pcy = 0.0
        for i in range(start, stop):
            a = tp * z[i]
            s = np.sin(a); c = np.cos(a); yv = yc[i]
            psums += s; pss_s += s * s; psy += s * yv
            psumc += c; pss_c += c * c; pcy += c * yv
        partials[b, 0] = psums; partials[b, 1] = pss_s; partials[b, 2] = psy
        partials[b, 3] = psumc; partials[b, 4] = pss_c; partials[b, 5] = pcy
    # Fixed-order serial combine across blocks -- thread-count-independent reduction order.
    sums = 0.0; ss_s = 0.0; sy = 0.0
    sumc = 0.0; ss_c = 0.0; cy = 0.0
    for b in range(nblocks):
        sums += partials[b, 0]; ss_s += partials[b, 1]; sy += partials[b, 2]
        sumc += partials[b, 3]; ss_c += partials[b, 4]; cy += partials[b, 5]
    out = 0.0
    # RELATIVE degeneracy guard (P1-4): reject when the centered SS is a negligible fraction of the raw
    # SS (catastrophic-cancellation residual for a near-constant sin/cos projection) -- an absolute 1e-24
    # floor lets a ~1e-23 residual through and explodes the ratio, letting a degenerate frequency win.
    v_ss = ss_s - sums * sums / n
    if v_ss > 1e-12 * ss_s and v_ss >= 1e-24 and y_ss >= 1e-24:
        out += (sy * sy) / (v_ss * y_ss)
    v_cc = ss_c - sumc * sumc / n
    if v_cc > 1e-12 * ss_c and v_cc >= 1e-24 and y_ss >= 1e-24:
        out += (cy * cy) / (v_cc * y_ss)
    return out


def _power_centered(z: np.ndarray, yc: np.ndarray, y_ss: float, freq: float) -> float:
    """Periodogram power at ``freq`` against a pre-centered ``y`` (``yc``,
    sum-of-squares ``y_ss``). Hot-loop variant that skips re-centering y."""
    # bench-attempt-rejected (2026-06-13): a SERIAL fused njit kernel measured 1.06x@n800 / 0.89x@n1667 /
    # 1.26x@n5000 / 0.80x@n20000 -- numba scalar sin/cos loses to numpy's vectorised transcendental ufunc.
    # The PARALLEL fused twin (prange) instead WINS from ~n>=4k (1.45x@5k / 2.48x@20k / 2.71x@50k / 3.11x@100k,
    # rel ~1e-15) -- the per-element sin/cos work amortises thread spawn. Gated below; serial numpy path stays
    # for small n. bench: _benchmarks/bench_power_centered_njit.py.
    if z.shape[0] >= _POWER_CENTERED_PAR_MIN_N:
        return _power_centered_fused_par_njit(
            np.ascontiguousarray(z, dtype=np.float64), np.ascontiguousarray(yc, dtype=np.float64),
            float(y_ss), float(freq),
        )
    ang = 2.0 * np.pi * float(freq) * z
    return (
        _corr_sq_centered(np.sin(ang), yc, y_ss)
        + _corr_sq_centered(np.cos(ang), yc, y_ss)
    )


def _refine_peak_freq(
    z_tr: np.ndarray, yc: np.ndarray, y_ss: float, coarse_f: float,
) -> float:
    """Two-stage local-refine of ``coarse_f`` on the TRAIN rows (pre-centered
    ``yc`` / ``y_ss``), maximising periodogram power.

    Stage 1 scans +-0.25 at 0.05 step (the coarse-grid spacing); stage 2 then
    scans +-0.05 at 0.0125 step around the stage-1 winner. The finer second
    pass tightens secondary-peak localisation after deflation -- which widens
    the downstream Ridge recovery margin on multitone signals (a 0.05-only
    refine left secondary tones mis-located by up to ~0.3, costing R^2)."""
    def _scan(center: float, half_width: float, step: float) -> tuple[float, float]:
        lo_r = max(0.05, center - half_width)
        hi_r = center + half_width
        n_steps = int(round((hi_r - lo_r) / step)) + 1
        best_f = center
        best_p = _power_centered(z_tr, yc, y_ss, center)
        for k in range(n_steps):
            f = lo_r + step * k
            p = _power_centered(z_tr, yc, y_ss, f)
            if p > best_p:
                best_p = p
                best_f = f
        return best_f, best_p
    f1, _ = _scan(coarse_f, 0.25, 0.05)
    f2, _ = _scan(f1, 0.05, 0.0125)
    return float(f2)


def _deflate_sincos(z: np.ndarray, y: np.ndarray, freq: float) -> np.ndarray:
    """Residual of ``y`` after least-squares projection onto
    ``[1, sin(2*pi*freq*z), cos(2*pi*freq*z)]``. Removes the contribution of
    one detected frequency so the next peak-pick sees the remaining tones."""
    ang = 2.0 * np.pi * float(freq) * z
    A = np.column_stack([np.ones_like(z), np.sin(ang), np.cos(ang)])
    try:
        # Normal-equations solve on the well-conditioned 3-column [1, sin, cos] design (faster than the SVD
        # lstsq); fall back to SVD lstsq if A^T A is singular (a degenerate freq collapsing the sin column ->
        # rank-deficient, where lstsq's min-norm solution is the robust choice). Same projection residual.
        AtA = A.T @ A
        coef = np.linalg.solve(AtA, A.T @ y)
        return y - A @ coef
    except np.linalg.LinAlgError:
        try:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            return y - A @ coef
        except Exception:
            return y
    except Exception:
        return y


def _detect_fourier_freqs_for_col(
    z01: np.ndarray,
    y: np.ndarray,
    *,
    f_grid: Sequence[float],
    min_val_corr: float = 0.15,
    min_rows: int = 800,
    max_freqs: int = 4,
) -> list[float]:
    """MULTI-FREQUENCY adaptive detector (2026-06-03).

    Returns the list of held-out-validated dominant z-space frequencies of
    ``y`` as a function of ``z01`` -- the multitone generalisation of
    :func:`_detect_fourier_freq_for_col`. Real signals routinely superpose
    several arbitrary-period oscillations (``sin(3.7x) + sin(5.3x) +
    sin(6.8x)``); detecting only the single dominant frequency leaves a Ridge
    on the support unable to recover the sum.

    Before any frequency search the target is POLYNOMIAL-DETRENDED: y is
    regressed on ``[1, z, z^2, z^3]`` (cubic coefficients fit on TRAIN, applied
    to VAL) and detection runs on the RESIDUAL. A monotone / smooth trend (the
    linear-additive ``y = sign(x1 + 0.7*x2)``) has high LOW-frequency periodogram
    power because a sub-1-cycle sinusoid mimics a ramp; the cubic absorbs it so
    its z-frequency power collapses to ~0, while a genuine oscillation (which a
    cubic cannot express) is left intact. This is the discriminator that
    separates "arbitrary-period oscillation" from "trend the poly basis covers".

    Each iteration then:

    * picks the coarse peak by periodogram power on TRAIN, local-refines +-0.25
      at 0.05 step,
    * confirms ``sqrt(val-slice power) >= max(min_val_corr, 0.30)`` on the
      held-out stride slice -- the 0.30 robust floor rejects finite-sample
      chance peaks (40-seed linear-fixture max spurious 0.232 vs genuine >= 0.96
      at n=800); ``min_val_corr`` is the user-raisable lower bound,
    * DEFLATES both the train and val targets by least-squares-projecting out
      that frequency's ``[1, sin, cos]`` so the next iteration sees the
      remaining tones,

    stopping at ``max_freqs`` or the first frequency that fails the held-out
    gate. N-gated at ``n >= min_rows`` (default 800) so a small-n chance
    frequency never fires. Frequencies already in the running list (within a
    coarse-grid spacing) are skipped to avoid re-locking the same peak.

    GPU-RESIDENCY bench-note (iter17, 2026-06-23): this detector and its njit sub-kernels
    (_coarse_basis_njit, _corr_sq_reductions_njit, _power_centered_fused_par_njit) STAY CPU.
    F2 100k cProfile: _detect_fourier_freqs_for_col 0.357s tottime / 0.938s cum over 42 calls;
    _coarse_basis_njit 0.201s/42; _power_centered_fused_par_njit 0.233s/924; _corr_sq_reductions_njit
    small -- the whole family is ~3-4% of the 31.6s WALL. NOT resident-routable favourably on this HW
    for three independent reasons: (1) the detector caps + row-subsamples its working set at
    MLFRAME_FOURIER_DETECT_MAX_N (200k) and runs on a <=66k train slice, so the operand is small;
    (2) the body is a SEQUENTIAL deflation loop -- pick peak -> _refine_peak_freq local scan -> held-out
    confirm -> least-squares deflate y_tr/y_va -- with host-side argmax/corr control flow each iteration,
    so there is no single batched workload; (3) the only batchable piece (the coarse grid x n sin/cos
    plane build) was ALREADY tried in batched-matrix form TWICE and bench-rejected (np.outer + matrix
    sin/cos LOSES 0.5-0.7x at n>=1100; raw-SS variant NEUTRAL) -- the (<=48 x <=66k) transcendental plane's
    alloc + bandwidth dominates, and the GPU twin would re-pay that as H2D + a tiny launch 42*max_freqs times.
    _power_centered_fused_par_njit returns a single scalar per call (924 calls) -- a GPU twin would transfer a
    66k vector to compute ONE float, pure overhead. All three are already at the CPU optimum (fused
    njit(parallel) machine code, 2.45-9.2x over the numpy loop), gated below the parallel-crossover for small n.
    """
    # GPU-RESIDENT dispatch (residency contract, not a wall win): under the resident flag
    # (MLFRAME_FE_GPU_STRICT + MLFRAME_FE_GPU_STRICT_RESIDENT) keep the column operand + target resident on the
    # device and run the deflation loop on cupy. Selection-equivalent to (NOT byte-identical with) this CPU path
    # within the coarse-grid tolerance. Any cupy/device error falls through to the CPU njit path below, so the
    # default (flag-off) path is byte-identical and a GPU fault never breaks a fit. See the bench-note: this twin
    # is EXPECTED slower on small-n / sequential-loop HW and that is a PASS by the residency contract.
    try:
        from .._gpu_strict_fe._entry import fe_gpu_strict_resident_enabled as _fourier_resident_flag_on  # type: ignore
    except Exception:
        _fourier_resident_flag_on = None  # type: ignore
    if _fourier_resident_flag_on is not None and _fourier_resident_flag_on():
        # Import stays broad-guarded (cupy/twin may be absent); the CALL is narrowed to genuine
        # device/linalg faults so a real twin logic/shape bug (ValueError/KeyError/IndexError)
        # propagates to tests instead of silently degrading to CPU as a "device fallback".
        try:
            from ._fourier_detect_gpu_resident import detect_fourier_freqs_for_col_gpu
            from .._fourier_detect_cap import get_fourier_detect_max_n
            _twin_ready = True
        except Exception:
            _twin_ready = False
        if _twin_ready:
            _dev_errs = []
            try:
                _dev_errs.append(np.linalg.LinAlgError)
                import cupy as _cp  # type: ignore
                _dev_errs.append(_cp.cuda.runtime.CUDARuntimeError)
                _dev_errs.append(_cp.cuda.memory.OutOfMemoryError)
                # FIX4 (2026-06-28): cuSOLVER/cuBLAS faults from cp.linalg.solve/lstsq subclass plain
                # RuntimeError, NOT CUDARuntimeError -> omitting them would crash instead of falling
                # back. getattr so an absent symbol can't break the tuple builder.
                from cupy_backends.cuda.libs import cusolver as _cusolver  # type: ignore
                _dev_errs.append(getattr(_cusolver, "CUSOLVERError", None))
                from cupy_backends.cuda.libs import cublas as _cublas  # type: ignore
                _dev_errs.append(getattr(_cublas, "CUBLASError", None))
            except Exception:
                pass
            _dev_errs = [e for e in _dev_errs if isinstance(e, type) and issubclass(e, BaseException)]
            try:
                return detect_fourier_freqs_for_col_gpu(
                    z01, y, f_grid=f_grid, min_val_corr=min_val_corr, min_rows=min_rows,
                    max_freqs=max_freqs, fourier_detect_max_n=get_fourier_detect_max_n(),
                )
            except tuple(_dev_errs):
                pass  # genuine cupy/device/linalg fault -> CPU njit path (byte-identical default); logic bugs propagate
    z01 = np.asarray(z01, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = z01.size
    if n != y.size or n < int(min_rows):
        return []
    if not np.all(np.isfinite(z01)) or not np.all(np.isfinite(y)):
        return []
    if float(np.std(z01)) < 1e-12 or float(np.std(y)) < 1e-12:
        return []
    grid = [float(f) for f in f_grid if float(f) > 0.0]
    if not grid:
        return []
    # SEEDED held-out split (P1-5): a plain ``idx % 3`` stride leaks under sorted / periodic row order
    # (every 3rd row is then highly correlated with its train neighbours -> the "held-out" gate is not
    # honest and a chance frequency clears the floor). A fixed-seed permutation picks a row-order-
    # INDEPENDENT 1/3, so the split is robust however the caller ordered the rows; deterministic (seed 0).
    val_mask = np.zeros(n, dtype=bool)
    val_mask[np.random.default_rng(0).permutation(n)[: n // 3]] = True
    train_mask = ~val_mask
    z_tr, z_va = z01[train_mask], z01[val_mask]
    y_tr = y[train_mask].copy()
    y_va = y[val_mask].copy()
    if z_tr.size < 16 or z_va.size < 8:
        return []
    # Frequency DETECTION is a proposal heuristic; at very large n the coarse-basis (grid x n) sin/cos
    # planes dominate memory (a 1M-row fit OOMs building (25, ~667k) planes per column). A uniform
    # random ROW-subsample preserves the (z, y) joint distribution the detector fits sin(2*pi*f*z) to,
    # so it recovers the same dominant frequencies at a fraction of the memory. Cap via
    # MLFRAME_FOURIER_DETECT_MAX_N (0 = no cap; default 200k is ample for dominant-frequency detection).
    # Seeded local Generator -> deterministic + NO global-RNG contamination. Below the cap: untouched.
    from .._fourier_detect_cap import get_fourier_detect_max_n
    _fdet_cap = get_fourier_detect_max_n()
    if _fdet_cap > 0 and z_tr.size > _fdet_cap:
        _fdet_rng = np.random.default_rng(0xF0F0_1234)
        _sub_tr = _fdet_rng.choice(z_tr.size, size=_fdet_cap, replace=False)
        z_tr = np.ascontiguousarray(z_tr[_sub_tr]); y_tr = np.ascontiguousarray(y_tr[_sub_tr])
        _va_cap = max(8, _fdet_cap // 2)
        if z_va.size > _va_cap:
            _sub_va = _fdet_rng.choice(z_va.size, size=_va_cap, replace=False)
            z_va = np.ascontiguousarray(z_va[_sub_va]); y_va = np.ascontiguousarray(y_va[_sub_va])
    if float(np.std(y_tr)) < 1e-12 or float(np.std(y_va)) < 1e-12:
        return []
    # POLYNOMIAL DETREND (2026-06-03): regress y on [1, z, z^2, z^3] and run
    # detection on the RESIDUAL. A monotone / smooth trend (the linear-additive
    # target ``y = sign(x1 + 0.7*x2)``) has HIGH periodogram power at LOW
    # frequencies because a sub-1-cycle sinusoid mimics a monotone ramp -- so
    # the raw periodogram would FALSE-POSITIVE a "frequency" on a non-periodic
    # target (the linear fixture emitted a spurious ``x1__sin0.75``). A low-
    # degree polynomial in z ABSORBS any such trend, so its z-frequency power
    # collapses to ~0 after detrending (measured 0.689 -> 0.002), while a
    # genuine oscillation -- which a cubic CANNOT express -- is left intact
    # (sin(5.3x) power 0.94 retained). The cubic coefficients are fit on TRAIN
    # and APPLIED to VAL (no val leakage); this is the discriminator that
    # separates "arbitrary-period oscillation" from "smooth trend the poly
    # basis already covers".
    _V_tr = np.vander(z_tr, 4)  # [z^3, z^2, z, 1]
    try:
        # normal-eq solve on the 4-col cubic Vandermonde (faster than SVD lstsq); lstsq fallback keeps
        # the rank-robustness for a degenerate z (constant/near-constant column).
        try:
            _poly_coef = np.linalg.solve(_V_tr.T @ _V_tr, _V_tr.T @ y_tr)
        except np.linalg.LinAlgError:
            _poly_coef, *_ = np.linalg.lstsq(_V_tr, y_tr, rcond=None)
        y_tr = y_tr - _V_tr @ _poly_coef
        y_va = y_va - np.vander(z_va, 4) @ _poly_coef
    except Exception:
        pass
    if float(np.std(y_tr)) < 1e-9 or float(np.std(y_va)) < 1e-9:
        return []
    # Effective held-out floor. Even after the polynomial detrend, a FINITE-
    # SAMPLE chance frequency can clear a lenient floor: across 40 linear-
    # additive fixtures (``y = sign(x1 + 0.7*x2)``, n=1200) the max spurious
    # held-out sqrt-power was 0.232, while a genuine oscillation sits at >= 0.96
    # even at n=800 -- a wide gap. A robust 0.30 floor rejects the chance peaks
    # without touching genuine recovery (gate-A multitone tones clear 0.6+).
    # ``min_val_corr`` is honoured as a LOWER bound a caller can RAISE; the
    # built-in 0.30 is the anti-false-positive guard the small-n regime needs.
    _eff_min_val_corr = max(float(min_val_corr), 0.30)
    # Precompute the coarse-grid sin/cos bases on TRAIN once: they depend only
    # on z, not y, so deflation iterations reuse them (cProfile: the per-freq
    # np.sin/np.cos + np.corrcoef was the dominant cost at p=200; this drops
    # the coarse sweep to a centered dot product per cached basis).
    # bench-attempt-rejected (2026-06-13): batching the coarse-basis build (and the
    # refine-peak scan) into one ``np.outer`` + matrix ``np.sin``/``np.cos`` eval LOSES
    # at the scene train-slice sizes -- the m*n temporary's allocation + memory-bandwidth
    # cost dominates the saved per-call overhead (0.5-0.7x at n>=1100, only winning at
    # n~533), and the ``axis=1`` reduction shifts power by ~1e-12. benches:
    # profiling/bench_coarse_basis_batched.py, profiling/bench_refine_peak_batched.py.
    # The shipped win is the per-call no-alloc rewrite of ``_corr_sq_centered`` instead.
    # bench-attempt-rejected (2026-06-13): storing RAW sin/cos + centered SS (``raw@raw - sum(raw)^2/n``) to skip the two ``s - s.mean()`` / ``c - c.mean()``
    # temporaries per freq (the same no-alloc identity shipped in ``_corr_sq_centered``) is bit-identical to ~1e-15 BUT NEUTRAL at the detector level: same-process
    # A/B of the full detector at n=1667 chirp48 was 3.645 -> 3.644ms (1.000x). The build is only ~37% of the detector and the alloc-savings are swamped by the
    # dominant sin/cos+dot cost. Isolated build-loop bench is pure noise (0.47x-2.35x scatter). bench: profiling/bench_coarse_basis_nocenter.py.
    # Coarse-basis build (the detector's dominant own-frame cost: scene n=12000 cProfile 1.050s/68 calls). The per-freq
    # numpy sin/cos + center + SS loop fuses into ONE njit(parallel=True) prange-over-freqs kernel (iter52): the
    # transcendentals + reductions run in machine code, parallelised across the 16/48 grid frequencies. Measured 2.45-9.2x
    # warm over the numpy loop across n=533..8000 (bench_coarse_basis_njit_parallel). The fused reduction shifts the
    # sin/cos SS by ~1e-13 (single-ULP, sequential-vs-pairwise sum) which only perturbs the coarse-sweep ``best_f``
    # argmax that ``_refine_peak_freq`` re-localises -- end-to-end MRMR scene selection is byte-identical. Set
    # ``MLFRAME_FOURIER_COARSE_BASIS_EXACT=1`` to force the exact numpy build.
    _coarse_basis = []  # (sin_centered, sin_ss, cos_centered, cos_ss) per grid freq
    _use_exact_basis = os.environ.get("MLFRAME_FOURIER_COARSE_BASIS_EXACT", "") == "1"
    if not _use_exact_basis and len(grid) > 0 and z_tr.size > 0:
        _sc_m, _cc_m, _sss, _css = _coarse_basis_njit(np.ascontiguousarray(z_tr), np.asarray(grid, dtype=np.float64))
        for gi in range(len(grid)):
            _coarse_basis.append((_sc_m[gi], float(_sss[gi]), _cc_m[gi], float(_css[gi])))
    else:
        for f in grid:
            ang = 2.0 * np.pi * f * z_tr
            s = np.sin(ang); c = np.cos(ang)
            sc = s - s.mean(); cc = c - c.mean()
            _coarse_basis.append((sc, float(sc @ sc), cc, float(cc @ cc)))
    out: list[float] = []
    for _ in range(max(1, int(max_freqs))):
        if float(np.std(y_tr)) < 1e-9 or float(np.std(y_va)) < 1e-9:
            break
        yc = y_tr - y_tr.mean()
        y_ss = float(yc @ yc)
        if y_ss < 1e-24:
            break
        best_f = None
        best_power = -1.0
        for gi, f in enumerate(grid):
            sc, s_ss, cc, c_ss = _coarse_basis[gi]
            num_s = float(sc @ yc)
            num_c = float(cc @ yc)
            p = 0.0
            if s_ss >= 1e-24:
                p += (num_s * num_s) / (s_ss * y_ss)
            if c_ss >= 1e-24:
                p += (num_c * num_c) / (c_ss * y_ss)
            if p > best_power:
                best_power = p
                best_f = f
        if best_f is None:
            break
        refined_f = _refine_peak_freq(z_tr, yc, y_ss, best_f)
        # Skip a frequency we've already locked (within half a coarse step).
        if any(abs(refined_f - g) < 0.25 for g in out):
            # Deflate at the coarse peak anyway so the loop can advance, then
            # continue searching the remaining spectrum.
            y_tr = _deflate_sincos(z_tr, y_tr, refined_f)
            y_va = _deflate_sincos(z_va, y_va, refined_f)
            continue
        val_power = _periodogram_power(z_va, y_va, refined_f)
        if val_power <= 0.0 or np.sqrt(val_power) < _eff_min_val_corr:
            break
        out.append(float(refined_f))
        # Deflate both slices so the next peak-pick sees the residual tones.
        y_tr = _deflate_sincos(z_tr, y_tr, refined_f)
        y_va = _deflate_sincos(z_va, y_va, refined_f)
    return out


# Auto-gate for the adaptive Fourier / chirp operators (2026-06-04): a Fourier or chirp leg only helps where the RAW column is NOT already a strong smooth predictor of y. On a near-step / leak column (``leaky ~ y``) the cubic detrend leaves Gibbs ringing the periodogram mistakes for an oscillation; on a genuinely linear / monotone / heavy-tailed-monotone signal the raw column already carries the usability. In both cases a Fourier/chirp leg adds no generalisable signal -- it only manufactures engineered columns that then evict the raw signal from ``support_``. So when the raw column's held-out cubic R^2 clears this cap, skip the adaptive operators for it; genuine oscillatory / chirp targets (raw cubic R^2 ~ 0) stay below the cap and keep firing.
_ADAPTIVE_FE_RAW_USABILITY_CAP: float = 0.5

# Cardinality ceiling for treating an integer-valued column as a categorical group key (NOT a continuous axis). A column whose
# distinct integer values number <= this is an arbitrary-label categorical -- sin/cos of the label code is spurious periodicity
# (the "frequency" the adaptive detector finds is just fitting the arbitrary label->target mapping, not a real oscillation). Mirrors
# the int-as-cat group-key band used by the grouped-agg auto-detector + recommender (min 3); kept low (50) so an ordinal integer
# axis with many levels (counts, ages) still gets Fourier.
_FOURIER_INT_AS_CAT_MAX_CARD: int = 50


@numba.njit(cache=True)
def _is_int_as_cat_njit(x: np.ndarray, max_card: int) -> bool:
    """Fused int-as-cat test: ONE early-exiting pass replacing the numpy
    ``isfinite`` mask + full ``mod(.,1)==0`` pass + full ``np.unique`` sort.

    Walks ``x`` once: skips non-finite, returns False on the FIRST non-integer
    element (``v != floor(v)``), and tracks distinct finite values in a small
    linear-probe buffer that early-exits the moment cardinality exceeds
    ``max_card`` (so a high-card ordinal integer column -- counts/ages -- bails
    after seeing its 51st distinct value instead of sorting all n rows). The
    boolean verdict is IDENTICAL to the numpy form (verified across low/high-card
    int, continuous, and NaN-mixed columns); ``np.unique``'s O(n log n) full sort
    over a 1M column was the dominant cost."""
    n = x.shape[0]
    uniq = np.empty(max_card + 1, dtype=np.float64)
    ucount = 0
    finite_count = 0
    for i in range(n):
        v = x[i]
        if not np.isfinite(v):
            continue
        finite_count += 1
        if v != np.floor(v):
            return False
        found = False
        for j in range(ucount):
            if uniq[j] == v:
                found = True
                break
        if not found:
            uniq[ucount] = v
            ucount += 1
            if ucount > max_card:
                return False
    if finite_count < 8:
        return False
    return 3 <= ucount <= max_card


def _is_int_as_cat_axis(x: np.ndarray, *, max_card: int = _FOURIER_INT_AS_CAT_MAX_CARD) -> bool:
    """True iff ``x`` is an integer-valued low-cardinality column that reads as a categorical group key rather than a continuous
    axis. Fourier sin/cos of such a column's arbitrary integer labels is spurious (region code 0..9 has no periodicity), so the
    adaptive-Fourier basis must skip it -- otherwise it floods the support with label-fitting sin/cos pairs that crowd out the
    genuinely useful grouped aggregates of that key. Continuous columns (floats, high-card ints) return False and keep Fourier.

    The finite/integer/cardinality checks are fused into ONE early-exiting njit
    pass (:func:`_is_int_as_cat_njit`) -- bit-identical verdict, but a continuous
    column bails at the first fractional value and a high-card integer column bails
    at its ``max_card+1``-th distinct value, both without the full ``np.unique`` sort
    the numpy form paid on every 1M-row column."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    return bool(_is_int_as_cat_njit(x, int(max_card)))


def _heldout_smooth_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Held-out R^2 of y on a cubic in x ([1, x, x^2, x^3]); val = every 3rd row (the same stride the frequency detector uses). Scale/shift-invariant, so it reads identically on raw x or its normalised z. Returns 0.0 on degenerate input."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    if n != y.size or n < 32:
        return 0.0
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return 0.0
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    # SEEDED held-out split (P1-5): row-order-independent 1/3, robust to sorted/periodic input (see the
    # detector above). Deterministic (seed 0).
    va = np.zeros(n, dtype=bool)
    va[np.random.default_rng(0).permutation(n)[: n // 3]] = True
    tr = ~va
    if int(tr.sum()) < 16 or int(va.sum()) < 8:
        return 0.0
    # Rank-normalise x to [-1, 1] so heavy tails (Cauchy / lognormal outliers) cannot dominate the least-squares fit and understate the raw column's usability. Ranks are a monotone reparametrisation, so a genuine oscillation stays non-cubic (gate keeps letting Fourier fire) while a monotone heavy-tailed signal reads as smooth (gate blocks Fourier on it).
    ranks = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
    zx = (ranks / max(n - 1, 1)) * 2.0 - 1.0
    try:
        _Vtr = np.vander(zx[tr], 4)
        try:
            coef = np.linalg.solve(_Vtr.T @ _Vtr, _Vtr.T @ y[tr])  # normal-eq (lstsq fallback below)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(_Vtr, y[tr], rcond=None)
    except Exception:
        return 0.0
    pred = np.vander(zx[va], 4) @ coef
    yv = y[va]
    sse = float(np.sum((yv - pred) ** 2))
    sst = float(np.sum((yv - yv.mean()) ** 2))
    if sst < 1e-24:
        return 0.0
    return 1.0 - sse / sst


def _detect_fourier_freq_for_col(
    z01: np.ndarray,
    y: np.ndarray,
    *,
    f_grid: Sequence[float],
    min_val_corr: float = 0.15,
    min_rows: int = 800,
) -> Optional[float]:
    """ADAPTIVE-FREQUENCY Fourier detector (2026-06-03).

    The fixed Fourier univariate grid only covers z-space frequencies {1, 2}.
    An ARBITRARY-period oscillation (e.g. ``y = sin(3.7*x)``, ``sin(5.3*x)``)
    lands at a non-integer z-space frequency and is missed by the fixed grid
    (recovered at |corr| 0.02-0.23). This detector sweeps a coarse z-space
    frequency grid, locally refines around the peak, and returns the dominant
    frequency ONLY when a held-out validation slice confirms it -- otherwise
    None (no adaptive column emitted).

    Method
    ------
    * Deterministic stride train/val split: ``val = arange(n) % 3 == 0`` (a
      third held out, no RNG so the recipe replays identically). The frequency
      is RANKED on train rows and CONFIRMED on the held-out val rows -- a
      chance frequency that fits a train slice but not the held-out slice is
      rejected. This is the n-gated false-positive guard: a naive default-on
      version regressed 9 tests because at small n a chance frequency clears
      the gate. We require ``n >= min_rows`` (default 800) AND val-slice
      confirmation.
    * Rank ``f_grid`` by PERIODOGRAM POWER ``corr(sin)^2 + corr(cos)^2`` on the
      TRAIN rows (phase-invariant: a single sin or cos alone has low |corr| for
      a phase-shifted signal, so we must score the sin+cos pair jointly).
    * Local-refine ``+-0.25`` at ``0.05`` step around the coarse peak (still on
      train).
    * KEEP the refined freq only if ``sqrt(val-slice periodogram power) >=
      max(min_val_corr, 0.30)`` (the held-out effective |corr| of the sin+cos
      support clears the floor). Otherwise return None.

    Before the search, y is POLYNOMIAL-DETRENDED (cubic in z, train-fit /
    val-applied) so a monotone / smooth trend cannot masquerade as a low
    frequency; the 0.30 robust floor then rejects finite-sample chance peaks.
    See :func:`_detect_fourier_freqs_for_col` for the full rationale.

    ``z01`` is the SAME ``z = (x - lo) / span`` in [0, 1] that the Fourier
    emitter uses, so the detected frequency drops straight into the emitter's
    ``fourier_freqs`` for that column. ``y`` may be discrete or continuous;
    Pearson on y is fine because we only need a phase-invariant linear-usability
    score, not MI.

    Returns the SINGLE dominant validated frequency (or None). The multitone
    superposition case is handled by :func:`_detect_fourier_freqs_for_col`,
    which this delegates to (taking the first detected peak) -- so the coarse-
    sweep + local-refine + held-out-gate contract is shared verbatim.
    """
    freqs = _detect_fourier_freqs_for_col(
        z01, y, f_grid=f_grid, min_val_corr=min_val_corr,
        min_rows=min_rows, max_freqs=1,
    )
    return float(freqs[0]) if freqs else None



# generate/recipe entry points carved to _orth_extra_basis_fe_generate.py (1k-LOC ceiling).
from ._orth_extra_basis_fe_generate import (  # noqa: E402, F401
    _build_recipe_from_meta,
    generate_extra_basis_features,
    hybrid_orth_extra_basis_fe_with_recipes,
)
