"""Helpers carved out of ``_orth_extra_basis_fe`` to keep that module under its size budget."""
from __future__ import annotations

import logging
import os
import threading
from typing import cast

from mlframe.utils.disk_cache import hash_array_content
from mlframe.utils.log_throttle import log_throttle

import numba
import numpy as np
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
from mlframe._numerical_floors import VARIANCE_FLOOR
from numba import prange

from ._fourier_core_cycles import freq_is_tail_aliased
from .._numba_utils_shared import radix_order_f32, radix_order_f64

logger = logging.getLogger(__name__)


# Backlog #13: ``"wavelet"`` adds the Haar / localized
# multiresolution basis alongside the global Fourier + fixed-knot spline. Its
# legs are held-out-scale-selected in ``_wavelet_basis_fe`` and emitted here so
# the extra-basis path (``fe_hybrid_orth_enable`` / ``extra_bases``) can route
# them through the same MI-uplift gate. The standalone default-on stage
# (``fe_wavelet_enable``) reuses the same generator + recipe builder.
_EXTRA_BASIS_KINDS = ("spline", "fourier", "wavelet")


logger = logging.getLogger(__name__)


@numba.njit(fastmath=True, cache=True)
def _corr_sq_reductions_njit(v: np.ndarray, y_centered: np.ndarray) -> tuple:
    """Fuse the three ``_corr_sq_centered`` reductions - ``sum(v)``, ``v@v``,
    ``v@y_centered`` - into ONE sequential pass over ``v``/``y_centered``.

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
    std passes) - a direct centered dot product. Returns 0.0 on a degenerate
    ``v``.

    Computes the centered SS / numerator from RAW ``v`` dot products so no
    length-n ``v - v.mean()`` temporary is allocated: ``v_ss = v@v - sum(v)^2/n``,
    and ``num = v @ y_centered`` is IDENTITY-equal to the centered ``vc @ y_centered``
    because ``y_centered`` sums to zero (the ``v.mean()*sum(y_centered)`` cross
    term vanishes). The three reductions are fused into one njit pass
    (:func:`_corr_sq_reductions_njit`); the reduction-order shift is ~1e-14 (single
    ULP), far below any selection-altering scale."""
    n = v.shape[0]
    sv, vv, vy = _corr_sq_reductions_njit(np.ascontiguousarray(v, dtype=_crit_np_dtype()), y_centered)
    v_ss = vv - sv * sv / n
    # RELATIVE degeneracy guard (P1-4): the raw-moment form ``vv - sv^2/n`` catastrophically cancels for a
    # near-constant ``v`` - it can land at a tiny positive residual (e.g. 1e-23) that clears an absolute
    # 1e-24 floor yet makes ``(vy^2)/(v_ss*y_ss)`` explode past 1.0, letting a degenerate column win the
    # periodogram. A genuinely varying v has ``v_ss`` an O(1) fraction of ``vv``; cancellation gives
    # ``v_ss << vv``. Reject when the centered SS is a negligible fraction of the raw SS.
    if v_ss <= 1e-12 * vv or v_ss < VARIANCE_FLOOR or y_ss < VARIANCE_FLOOR:
        return 0.0
    return float((vy * vy) / (v_ss * y_ss))


def _periodogram_power(z01: np.ndarray, y: np.ndarray, freq: float) -> float:
    """Phase-invariant periodogram power of ``y`` at z-space frequency ``freq``.

    ``corr(sin(2*pi*freq*z), y)^2 + corr(cos(2*pi*freq*z), y)^2`` - the sum of
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
    if y_ss < VARIANCE_FLOOR:
        return 0.0
    ang = 2.0 * np.pi * float(freq) * z01
    return _corr_sq_centered(np.sin(ang), yc, y_ss) + _corr_sq_centered(np.cos(ang), yc, y_ss)


_POWER_CENTERED_PAR_MIN_N = 4000


_POWER_CENTERED_PAR_NBLOCKS = 64


@numba.njit(fastmath=True, parallel=True, cache=True)
def _power_centered_fused_par_njit(z: np.ndarray, yc: np.ndarray, y_ss: float, freq: float) -> float:
    """Periodogram power = corr(sin)^2 + corr(cos)^2 with sin/cos + both centered-SS reductions fused, no
    length-n temporaries. Parallel over a FIXED number of contiguous row-blocks (not a numba auto-reduction
    over ``prange(n)``): each block sums serially into a private partial row, then the partials combine in a
    fixed block order. The result is bit-IDENTICAL across thread counts / process starts (deterministic float
    reduction order), so the downstream razor-tie frequency argmax (_refine_peak_freq) is process-stable.
    Bit-close to ~1e-15 (reduction-order) of the numpy-sin/cos + _corr_sq_centered path - far below any
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
    # Fixed-order serial combine across blocks - thread-count-independent reduction order.
    sums = 0.0; ss_s = 0.0; sy = 0.0
    sumc = 0.0; ss_c = 0.0; cy = 0.0
    for b in range(nblocks):
        sums += partials[b, 0]; ss_s += partials[b, 1]; sy += partials[b, 2]
        sumc += partials[b, 3]; ss_c += partials[b, 4]; cy += partials[b, 5]
    out = 0.0
    # RELATIVE degeneracy guard (P1-4): reject when the centered SS is a negligible fraction of the raw
    # SS (catastrophic-cancellation residual for a near-constant sin/cos projection) - an absolute 1e-24
    # floor lets a ~1e-23 residual through and explodes the ratio, letting a degenerate frequency win.
    v_ss = ss_s - sums * sums / n
    if v_ss > 1e-12 * ss_s and v_ss >= VARIANCE_FLOOR and y_ss >= VARIANCE_FLOOR:
        out += (sy * sy) / (v_ss * y_ss)
    v_cc = ss_c - sumc * sumc / n
    if v_cc > 1e-12 * ss_c and v_cc >= VARIANCE_FLOOR and y_ss >= VARIANCE_FLOOR:
        out += (cy * cy) / (v_cc * y_ss)
    return out


@numba.njit(fastmath=True, parallel=True, cache=True)
def _power_centered_batch_njit(z: np.ndarray, yc: np.ndarray, y_ss: float, freqs: np.ndarray) -> np.ndarray:
    """Batch of :func:`_power_centered_fused_par_njit` over every ``freqs[i]``, ONE parallel dispatch.

    ``_refine_peak_freq``'s ``_scan`` helper grid-searches a frequency band by calling
    ``_power_centered`` once per candidate point in a serial Python ``for`` loop (~9-21 points per
    scan, 2 scans per refine call) -- each call independently re-dispatches the parallel njit kernel
    (its own thread-launch overhead) even though every candidate in one scan shares the SAME
    ``z``/``yc``/``y_ss``, only ``freq`` varies. 2M-row cProfile (combo `c0016_c3f401e4`,
    master-seed 2026_04_29): `_power_centered` 30.98s tottime / 2134 calls. Flattens the work into
    ``n_freqs * nblocks`` independent (frequency, block) partial-sum tasks scheduled across ONE
    ``prange``, then reduces each frequency's own ``nblocks`` partials in the SAME fixed 0..NB-1
    order the single-frequency kernel uses -- bit-identical per-frequency result (the reduction order
    within a frequency is unchanged; only which OTHER frequencies' blocks happen to run concurrently
    on other threads differs, which cannot affect a given frequency's own float accumulation)."""
    n = z.shape[0]
    nf = freqs.shape[0]
    nblocks = _POWER_CENTERED_PAR_NBLOCKS
    if nblocks > n:
        nblocks = n
    if nblocks < 1:
        nblocks = 1
    base = n // nblocks
    rem = n % nblocks
    partials = np.zeros((nf, nblocks, 6))
    total_work = nf * nblocks
    for w in prange(total_work):
        fi = w // nblocks
        b = w % nblocks
        tp = 2.0 * np.pi * freqs[fi]
        if b < rem:
            start = b * (base + 1)
            stop = start + (base + 1)
        else:
            start = rem * (base + 1) + (b - rem) * base
            stop = start + base
        psums = 0.0
        pss_s = 0.0
        psy = 0.0
        psumc = 0.0
        pss_c = 0.0
        pcy = 0.0
        for i in range(start, stop):
            a = tp * z[i]
            s = np.sin(a)
            c = np.cos(a)
            yv = yc[i]
            psums += s
            pss_s += s * s
            psy += s * yv
            psumc += c
            pss_c += c * c
            pcy += c * yv
        partials[fi, b, 0] = psums
        partials[fi, b, 1] = pss_s
        partials[fi, b, 2] = psy
        partials[fi, b, 3] = psumc
        partials[fi, b, 4] = pss_c
        partials[fi, b, 5] = pcy
    out = np.zeros(nf, dtype=np.float64)  # Intentional float64: accumulator or output buffer; statistic at float64
    for fi in range(nf):
        sums = 0.0
        ss_s = 0.0
        sy = 0.0
        sumc = 0.0
        ss_c = 0.0
        cy = 0.0
        for b in range(nblocks):
            sums += partials[fi, b, 0]
            ss_s += partials[fi, b, 1]
            sy += partials[fi, b, 2]
            sumc += partials[fi, b, 3]
            ss_c += partials[fi, b, 4]
            cy += partials[fi, b, 5]
        p = 0.0
        v_ss = ss_s - sums * sums / n
        if v_ss > 1e-12 * ss_s and v_ss >= VARIANCE_FLOOR and y_ss >= VARIANCE_FLOOR:
            p += (sy * sy) / (v_ss * y_ss)
        v_cc = ss_c - sumc * sumc / n
        if v_cc > 1e-12 * ss_c and v_cc >= VARIANCE_FLOOR and y_ss >= VARIANCE_FLOOR:
            p += (cy * cy) / (v_cc * y_ss)
        out[fi] = p
    return out


def _power_centered(z: np.ndarray, yc: np.ndarray, y_ss: float, freq: float) -> float:
    """Periodogram power at ``freq`` against a pre-centered ``y`` (``yc``,
    sum-of-squares ``y_ss``). Hot-loop variant that skips re-centering y."""
    # bench-attempt-rejected (2026-06-13): a SERIAL fused njit kernel measured 1.06x@n800 / 0.89x@n1667 /
    # 1.26x@n5000 / 0.80x@n20000 - numba scalar sin/cos loses to numpy's vectorised transcendental ufunc.
    # The PARALLEL fused twin (prange) instead WINS from ~n>=4k (1.45x@5k / 2.48x@20k / 2.71x@50k / 3.11x@100k,
    # rel ~1e-15) - the per-element sin/cos work amortises thread spawn. Gated below; serial numpy path stays
    # for small n. bench: _benchmarks/bench_power_centered_njit.py.
    if z.shape[0] >= _POWER_CENTERED_PAR_MIN_N:
        return float(_power_centered_fused_par_njit(
            np.ascontiguousarray(z, dtype=_crit_np_dtype()), np.ascontiguousarray(yc, dtype=_crit_np_dtype()),
            float(y_ss), float(freq),
        ))
    ang = 2.0 * np.pi * float(freq) * z
    return float(_corr_sq_centered(np.sin(ang), yc, y_ss) + _corr_sq_centered(np.cos(ang), yc, y_ss))


def _refine_peak_freq(
    z_tr: np.ndarray, yc: np.ndarray, y_ss: float, coarse_f: float,
) -> float:
    """Two-stage local-refine of ``coarse_f`` on the TRAIN rows (pre-centered
    ``yc`` / ``y_ss``), maximising periodogram power.

    Stage 1 scans +-0.25 at 0.05 step (the coarse-grid spacing); stage 2 then
    scans +-0.05 at 0.0125 step around the stage-1 winner. The finer second
    pass tightens secondary-peak localisation after deflation - which widens
    the downstream Ridge recovery margin on multitone signals (a 0.05-only
    refine left secondary tones mis-located by up to ~0.3, costing R^2)."""
    def _scan(center: float, half_width: float, step: float) -> tuple[float, float]:
        """Grid-scan candidate frequencies in ``[center-half_width, center+half_width]`` and return the best ``(freq, score)`` pair."""
        lo_r = max(0.05, center - half_width)
        hi_r = center + half_width
        n_steps = round((hi_r - lo_r) / step) + 1
        cand_freqs = [center] + [lo_r + step * k for k in range(n_steps)]
        # Batched fused-njit dispatch (see _power_centered_batch_njit's docstring): every candidate in
        # one scan shares the SAME z_tr/yc/y_ss, only freq varies, so this fuses the whole grid into ONE
        # parallel-njit call instead of one dispatch per candidate. Bit-identical to the per-call
        # _power_centered path; gated on the SAME n threshold that path uses to pick the parallel
        # kernel, so the small-n numpy fallback (never worth batching) is untouched.
        if z_tr.shape[0] >= _POWER_CENTERED_PAR_MIN_N:
            powers = _power_centered_batch_njit(
                np.ascontiguousarray(z_tr, dtype=_crit_np_dtype()), np.ascontiguousarray(yc, dtype=_crit_np_dtype()),
                float(y_ss), np.asarray(cand_freqs, dtype=np.float64),  # Intentional float64: frequency vector for the Fourier replay
            )
        else:
            powers = [_power_centered(z_tr, yc, y_ss, f) for f in cand_freqs]
        best_idx = 0
        best_p = float(powers[0])
        for idx in range(1, len(cand_freqs)):
            p = float(powers[idx])
            if p > best_p:
                best_p = p
                best_idx = idx
        return cand_freqs[best_idx], best_p
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
        return np.asarray(y - A @ coef)
    except np.linalg.LinAlgError:
        try:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            return np.asarray(y - A @ coef)
        except Exception as e:
            # Returning `y` UNCHANGED breaks this function's whole contract -- "removes the contribution of one
            # detected frequency so the next peak-pick sees the remaining tones" -- because the caller's
            # iterative loop then re-detects the same tone and reports it as several distinct frequencies. The
            # value is not neutral, so the failure is announced rather than logged at debug; both solves failing
            # means the design is genuinely unusable at this frequency and there is nothing better to return.
            log_throttle(
                logger,
                "deflate_sincos_both_solves_failed",
                logging.WARNING,
                "_deflate_sincos: both the normal-equations solve and the SVD lstsq fallback failed at freq=%s "
                "(%s: %s); returning y UNDEFLATED, so the caller's next peak-pick will re-detect this tone.",
                freq,
                type(e).__name__,
                e,
            )
            return y
    except Exception as e:
        log_throttle(
            logger,
            "deflate_sincos_normal_equations_failed",
            logging.WARNING,
            "_deflate_sincos: the normal-equations projection failed at freq=%s with a non-LinAlgError (%s: %s), so "
            "the SVD fallback was not attempted; returning y UNDEFLATED and this tone will be re-detected.",
            freq,
            type(e).__name__,
            e,
        )
        return y


# Stable sort orders of the columns this detector ranks twice (the adaptive-column relevance score and the held-out smoothness gate sort the same column):
# a content-keyed cache returns the first order for the second request. Bounded by bytes; an evicted entry is simply sorted again.
# bench-attempt-rejected (2026-11-03): OFF by default. Captured 30000 x 108 fit call, warm, alternating: 2.61 s off vs 2.64 / 2.65 s on, same output; hashing the
# column content costs about what the second argsort saves. ``MLFRAME_EXTRA_BASIS_ORDER_CACHE=1`` turns it on.
_ORDER_CACHE: "dict[tuple, np.ndarray]" = {}
_ORDER_CACHE_BYTES = [0]
_ORDER_CACHE_MAX_BYTES = 64 << 20
_ORDER_CACHE_LOCK = threading.Lock()


def _stable_order(x: np.ndarray) -> np.ndarray:
    """``np.argsort(x, kind="stable")``: for a finite 1-D float32 / float64 vector the radix order, the same permutation about 7 times faster; anything else (NaN, other types) the comparison sort."""
    if x.ndim == 1 and x.size and x.dtype in (np.float32, np.float64) and bool(np.isfinite(x).all()):  # Intentional float64: dtype identity check
        return cast("np.ndarray", radix_order_f32(np.ascontiguousarray(x)) if x.dtype == np.float32 else radix_order_f64(np.ascontiguousarray(x)))
    return cast("np.ndarray", np.argsort(x, kind="stable"))


def cached_stable_argsort(x: np.ndarray) -> np.ndarray:
    """``np.argsort(x, kind="stable")`` of a 1-D array, shared between callers that sort the same content when ``MLFRAME_EXTRA_BASIS_ORDER_CACHE=1`` (off by default)."""
    if os.environ.get("MLFRAME_EXTRA_BASIS_ORDER_CACHE", "0").strip() != "1" or x.ndim != 1:
        return _stable_order(x)
    key = (x.size, str(x.dtype), hash_array_content(np.ascontiguousarray(x)))
    with _ORDER_CACHE_LOCK:
        hit = _ORDER_CACHE.get(key)
    if hit is not None:
        return hit
    order = np.argsort(x, kind="stable")
    order.setflags(write=False)
    with _ORDER_CACHE_LOCK:
        while _ORDER_CACHE and _ORDER_CACHE_BYTES[0] + order.nbytes > _ORDER_CACHE_MAX_BYTES:
            _ORDER_CACHE_BYTES[0] -= _ORDER_CACHE.pop(next(iter(_ORDER_CACHE))).nbytes  # evict-ok: oldest first; an evicted order is sorted again
        if order.nbytes <= _ORDER_CACHE_MAX_BYTES and _ORDER_CACHE.setdefault(key, order) is order:
            _ORDER_CACHE_BYTES[0] += order.nbytes
    return order


# The coarse sweep scores all grid frequencies with one matrix-vector product per sin / cos block (BLAS, GIL released) instead of two dot products per frequency in
# Python; the sums differ from the per-frequency dots in the last bits, which can only move the coarse argmax that ``_refine_peak_freq`` re-localises.
# ``MLFRAME_FOURIER_COARSE_GEMV=0`` keeps the per-frequency loop.
#: Test override for the coarse-GEMV switch; ``None`` reads ``MLFRAME_FOURIER_COARSE_GEMV`` at call time.
_COARSE_GEMV: "bool | None" = None


def _coarse_gemv_enabled() -> bool:
    """Whether the coarse GEMV path is on; the environment is read per call so it can change after import."""
    if _COARSE_GEMV is not None:
        return bool(_COARSE_GEMV)
    return os.environ.get("MLFRAME_FOURIER_COARSE_GEMV", "1").strip() != "0"


def _best_coarse_freq(mats, yc, y_ss, grid):
    """The grid frequency of the largest coarse sin + cos power (the first one on a tie, NaN powers never win), or ``None`` when none exceeds -1."""
    sc_m, sss, cc_m, css = mats
    yd = np.asarray(yc, dtype=sc_m.dtype)
    num_s = sc_m @ yd
    num_c = cc_m @ yd
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(sss >= VARIANCE_FLOOR, (num_s * num_s) / (sss * y_ss), 0.0) + np.where(css >= VARIANCE_FLOOR, (num_c * num_c) / (css * y_ss), 0.0)
    p = np.where(np.isnan(p), -np.inf, p)
    gi = int(np.argmax(p))
    return grid[gi] if p[gi] > -1.0 else None


def _greedy_fourier_frequency_search(max_freqs, y_tr, y_va, grid, _coarse_basis, z_tr, out, z_va, _core_span, _eff_min_val_corr):
    """Greedily extract Fourier frequencies from the residual of the training split."""
    for _ in range(max(1, int(max_freqs))):
        if float(np.std(y_tr)) < 1e-9 or float(np.std(y_va)) < 1e-9:
            break
        yc = y_tr - y_tr.mean()
        y_ss = float(yc @ yc)
        if y_ss < VARIANCE_FLOOR:
            break
        best_f = None
        best_power = -1.0
        _mats = getattr(_coarse_basis, "matrices", None)
        if _mats is not None and _coarse_gemv_enabled():
            best_f = _best_coarse_freq(_mats, yc, y_ss, grid)
        for gi, f in enumerate(grid if _mats is None or not _coarse_gemv_enabled() else ()):
            sc, s_ss, cc, c_ss = _coarse_basis[gi]
            num_s = float(sc @ yc)
            num_c = float(cc @ yc)
            p = 0.0
            if s_ss >= VARIANCE_FLOOR:
                p += (num_s * num_s) / (s_ss * y_ss)
            if c_ss >= VARIANCE_FLOOR:
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
        if freq_is_tail_aliased(refined_f, _core_span):
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
