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
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..hermite_fe import _detect_heavy_tail, _robust_axis_enabled, _robust_lo_hi

logger = logging.getLogger(__name__)

__all__ = [
    "generate_extra_basis_features",
    "hybrid_orth_extra_basis_fe_with_recipes",
]


_EXTRA_BASIS_KINDS = ("spline", "fourier")


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
    from ..engineered_recipes import _fit_spline_knots, _bspline_basis_values  # noqa: F401
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


def _chirp_axis(x: np.ndarray, mean: float, std: float, lo: float, span: float) -> np.ndarray:
    """Apply the stored chirp warp: x -> z=(x-mean)/std -> u=sign(z)*z**2 ->
    (u-lo)/span. Pure function of the fit-time params -- the single source of
    truth shared by fit-time detection (``generate_extra_basis_features``) and
    transform-time replay (``_apply_orth_fourier``) so both produce a
    bit-identical axis."""
    x = np.asarray(x, dtype=np.float64)
    z = (x - float(mean)) / max(float(std), 1e-12)
    u = np.sign(z) * (z * z)
    return (u - float(lo)) / max(float(span), 1e-12)


def _corr_sq_centered(v: np.ndarray, y_centered: np.ndarray, y_ss: float) -> float:
    """Squared Pearson correlation of ``v`` with a pre-centered ``y`` whose
    sum-of-squares is ``y_ss``. Avoids ``np.corrcoef`` (2x2-matrix build + two
    std passes) -- a direct centered dot product. Returns 0.0 on a degenerate
    ``v``."""
    vc = v - v.mean()
    v_ss = float(vc @ vc)
    if v_ss < 1e-24 or y_ss < 1e-24:
        return 0.0
    num = float(vc @ y_centered)
    return (num * num) / (v_ss * y_ss)


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


def _power_centered(z: np.ndarray, yc: np.ndarray, y_ss: float, freq: float) -> float:
    """Periodogram power at ``freq`` against a pre-centered ``y`` (``yc``,
    sum-of-squares ``y_ss``). Hot-loop variant that skips re-centering y."""
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
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ coef
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
    """
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
    idx = np.arange(n)
    val_mask = (idx % 3) == 0
    train_mask = ~val_mask
    z_tr, z_va = z01[train_mask], z01[val_mask]
    y_tr = y[train_mask].copy()
    y_va = y[val_mask].copy()
    if z_tr.size < 16 or z_va.size < 8:
        return []
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
    _coarse_basis = []  # (sin_centered, sin_ss, cos_centered, cos_ss) per grid freq
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


def _is_int_as_cat_axis(x: np.ndarray, *, max_card: int = _FOURIER_INT_AS_CAT_MAX_CARD) -> bool:
    """True iff ``x`` is an integer-valued low-cardinality column that reads as a categorical group key rather than a continuous
    axis. Fourier sin/cos of such a column's arbitrary integer labels is spurious (region code 0..9 has no periodicity), so the
    adaptive-Fourier basis must skip it -- otherwise it floods the support with label-fitting sin/cos pairs that crowd out the
    genuinely useful grouped aggregates of that key. Continuous columns (floats, high-card ints) return False and keep Fourier."""
    x = np.asarray(x, dtype=np.float64).ravel()
    finite = x[np.isfinite(x)]
    if finite.size < 8:
        return False
    if not np.all(np.equal(np.mod(finite, 1.0), 0.0)):
        return False
    card = int(np.unique(finite).size)
    return 3 <= card <= max_card


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
    idx = np.arange(n)
    va = (idx % 3) == 0
    tr = ~va
    if int(tr.sum()) < 16 or int(va.sum()) < 8:
        return 0.0
    # Rank-normalise x to [-1, 1] so heavy tails (Cauchy / lognormal outliers) cannot dominate the least-squares fit and understate the raw column's usability. Ranks are a monotone reparametrisation, so a genuine oscillation stays non-cubic (gate keeps letting Fourier fire) while a monotone heavy-tailed signal reads as smooth (gate blocks Fourier on it).
    ranks = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
    zx = (ranks / max(n - 1, 1)) * 2.0 - 1.0
    try:
        coef, *_ = np.linalg.lstsq(np.vander(zx[tr], 4), y[tr], rcond=None)
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


def generate_extra_basis_features(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    extra_bases: Sequence[str] = ("spline", "fourier"),
    fourier_freqs: Sequence[float] = (1.0, 2.0),
    fourier_powers: Sequence[int] = (1, 2),
    spline_knots: int = 5,
    dedup_collinear_sources: bool = True,
    dedup_corr_threshold: float = 0.999,
    y: Optional[np.ndarray] = None,
    fourier_adaptive: bool = False,
    fourier_adaptive_min_val_corr: float = 0.15,
    fourier_chirp: bool = False,
    fourier_chirp_min_val_corr: float = 0.15,
) -> tuple[pd.DataFrame, dict]:
    """For each column in cols and each requested extra basis, emit the basis
    columns and return them alongside the per-column fit metadata (knot
    vectors, lo/hi, fourier (lo, span)) needed to build recipes.

    Parameters
    ----------
    X : DataFrame
        Source frame. Only numeric columns are processed; non-numeric are
        silently skipped.
    cols : sequence of column names, optional
        Columns to expand. None = all numeric columns.
    extra_bases : tuple of {'spline', 'fourier'}
        Which extra bases to emit. Empty tuple => returns empty frame.
    fourier_freqs : sequence of float
        Frequencies for the Fourier basis. One sin and one cos column per
        frequency per source column.
    spline_knots : int
        Number of inner quantile knots for the cubic B-spline basis.
        Emits ``spline_knots + 3`` basis columns per source column (cubic
        B-spline has K+degree basis functions on K inner knots).
    dedup_collinear_sources : bool, default True
        Drop near-duplicate source columns before basis enumeration
        (mirrors the polynomial univariate path).
    y : array-like, optional
        Target. Only consulted when ``fourier_adaptive`` is True (and only by
        the ADAPTIVE-FREQUENCY detector). Never read for the fixed-grid
        emission, so the legacy path stays leakage-free / y-independent.
    fourier_adaptive : bool, default False
        When True and ``y`` is given, run :func:`_detect_fourier_freqs_for_col`
        on each source column's z (power==1 only) and -- for each held-out-
        validated dominant frequency found (multitone: several peaks via
        residual deflation) -- ADD it to that column's Fourier frequency set.
        The emitted sin/cos meta entries for adaptive frequencies are tagged
        ``"adaptive": True`` so MRMR can protect them past screening. Covers
        arbitrary-period oscillations and their superpositions
        (``sin(3.7*x) + sin(5.3*x) + sin(6.8*x)``) the fixed grid {1, 2} misses.
    fourier_adaptive_min_val_corr : float, default 0.15
        Held-out validation effective-|corr| floor for the adaptive detector.
    fourier_chirp : bool, default False
        ADAPTIVE-CHIRP path (2026-06-03). When True and ``y`` is given, run the
        SAME held-out-validated detector on the QUADRATIC-ARGUMENT warp
        ``u = sign(z) * z**2`` (z standardised on the column) for each source
        column. A chirp ``y ~ sin(2*pi*f*z**2)`` -- whose frequency GROWS with z
        -- is STATIONARY in u, so the detector locks its frequency and the
        emitted ``sin(2*pi*f*u)`` / ``cos(2*pi*f*u)`` reconstruct it; a Fourier on
        the LINEAR argument cannot express a frequency that grows with z. The
        emitted sin/cos meta entries carry ``"arg": "quadratic"`` (the warp the
        recipe replays) AND ``"adaptive": True`` (so MRMR protects them past the
        screen, identical to the linear adaptive legs). This is an ADDITIVE
        second path alongside the linear adaptive one -- both fire; on a plain
        linear target the chirp legs are harmless (Ridge regularises them to ~0,
        Phase-0 bench: combined R^2 == linear-only on linear targets, +0.3-0.5
        R^2 on fast chirps). N-gated at >= 800 MI rows like the linear path.
    fourier_chirp_min_val_corr : float, default 0.15
        Held-out validation effective-|corr| floor for the chirp detector.

    Returns
    -------
    (engineered_X, meta)
        engineered_X : DataFrame of new columns with naming
            ``"{col}__sp{i}"`` (spline) and ``"{col}__sin{f}"`` /
            ``"{col}__cos{f}"`` (fourier).
        meta : dict mapping each emitted column name to a dict with the
            metadata required to build the matching recipe. Keys depend
            on basis kind: spline -> {"basis": "spline", "src": ..., "knots":
            ndarray, "idx": int, "lo": float, "hi": float}; fourier ->
            {"basis": "fourier", "src": ..., "kind": "sin"/"cos", "freq":
            float, "lo": float, "span": float, "power": int[, "adaptive": True]}.

    Notes
    -----
    bench-rejected (2026-06-03): a per-column "poly-vs-Fourier COMPETITION gate"
    -- emit only the better of {orth-poly basis, this Fourier path} per column to
    cut the redundant cross-family features that co-occur in the support -- was
    benchmarked and REJECTED. The co-occurrence is genuine COMPLEMENTARITY, not
    redundancy: on kink/step/bump targets (e.g. y=|x|) the Fourier legs carry
    independent residual R^2 0.16-0.60 that a degree<=4 poly under-fits, so a
    winner-takes-all gate HURT the |x| target OOS by -0.06; no OOS win on a tree
    downstream (deltas +/-0.008, inconsistent sign); on a mixed frame the gate
    declines to fire (different columns have different winners). The existing
    Fleuret redundancy + Spearman cross-stage dedup already remove the only real
    redundancy. Don't add a competition gate. (D:/Temp/item3_poly_fourier_findings.md)
    """
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    extra_bases = tuple(b for b in extra_bases if b in _EXTRA_BASIS_KINDS)
    if not extra_bases:
        return pd.DataFrame(index=X.index), {}
    if dedup_collinear_sources:
        from . import _dedup_collinear_source_cols
        cols = _dedup_collinear_source_cols(
            X, list(cols), corr_threshold=dedup_corr_threshold,
        )
    from ..engineered_recipes import _bspline_basis_values  # local import
    out_cols: dict = {}
    meta: dict = {}
    fourier_freqs = tuple(float(f) for f in fourier_freqs)
    spline_knots = max(2, int(spline_knots))
    # Adaptive-frequency detection runs only when requested AND y is supplied.
    # The y array is coerced to float once here (Pearson on y is all the
    # phase-invariant periodogram needs); detection is gated per-column below.
    _y_adapt = None
    if (fourier_adaptive or fourier_chirp) and y is not None:
        _y_adapt = np.asarray(y, dtype=np.float64).ravel()
        if _y_adapt.size != len(X) or not np.all(np.isfinite(_y_adapt)):
            _y_adapt = None
    # Coarse z-space frequency sweep grid for the adaptive detector. Covers a
    # wide period range (0.5 .. 8.0) at 0.5 stride; local refinement then
    # snaps to the true non-integer frequency. The set of frequencies the
    # detector may ADD is disjoint from the fixed grid by construction (a
    # fixed freq that already recovers the signal needs no adaptive twin).
    _adaptive_f_grid = tuple(0.5 * k for k in range(1, 17))  # 0.5 .. 8.0
    # The CHIRP warp (u = sign(z)*z**2) concentrates a growing-frequency signal
    # at a HIGHER z-space frequency than the linear axis, so the chirp detector
    # sweeps a WIDER grid (0.5 .. 24.0). Phase-0: fast chirps land at u-space
    # peaks up to ~12 and the multitone deflation needs headroom above them.
    _chirp_f_grid = tuple(0.5 * k for k in range(1, 49))  # 0.5 .. 24.0
    for col in cols:
        if col not in X.columns or not pd.api.types.is_numeric_dtype(X[col]):
            continue
        x = np.asarray(X[col].to_numpy(), dtype=np.float64)
        finite_mask = np.isfinite(x)
        if not finite_mask.any():
            continue
        if not finite_mask.all():
            x = np.where(finite_mask, x, float(np.nanmean(x[finite_mask])))
        # Auto-gate: only let the adaptive Fourier/chirp operators fire where the raw column is NOT already a strong smooth predictor of y (see _ADAPTIVE_FE_RAW_USABILITY_CAP).
        _adaptive_fe_ok = True
        if _y_adapt is not None and (fourier_adaptive or fourier_chirp):
            _adaptive_fe_ok = _heldout_smooth_r2(x, _y_adapt) < _ADAPTIVE_FE_RAW_USABILITY_CAP
        if "spline" in extra_bases:
            try:
                knots, lo, hi, n_basis = _fit_spline_for_col(x, spline_knots)
                span = max(hi - lo, 1e-12)
                z = np.clip((x - lo) / span, 0.0, 1.0)
                for i in range(n_basis):
                    vals = _bspline_basis_values(z, knots, i, degree=3)
                    # Skip near-constant columns -- the boundary cubic
                    # B-splines occasionally collapse to ~0 on quantile-
                    # placed knots when ties pile at the edge.
                    if float(np.std(vals)) <= 1e-12:
                        continue
                    name = f"{col}__sp{i}"
                    out_cols[name] = vals
                    meta[name] = {
                        "basis": "spline", "src": col,
                        "knots": knots, "idx": i,
                        "lo": float(lo), "hi": float(hi),
                    }
            except Exception as exc:
                logger.warning(
                    "generate_extra_basis_features: spline on col=%r raised "
                    "%r; skipping spline for that column.",
                    col, exc,
                )
        # Skip Fourier on integer-valued low-cardinality categorical group keys: sin/cos of an arbitrary label code (region 0..9)
        # is spurious periodicity that floods the support and displaces the genuinely useful grouped aggregates of that key.
        if "fourier" in extra_bases and not _is_int_as_cat_axis(x):
            try:
                # POWER-ARGUMENT Fourier (2026-06-03): build the Fourier on x**p for
                # p in fourier_powers, as a SELF-CONTAINED replayable recipe (raw x ->
                # x**p -> Fourier; 1-deep, no nesting). p=2 captures even-argument
                # CHIRPS like ``sin(a**2)`` (freq~1 on the a**2 argument reproduces it
                # exactly) that a Fourier on the linear argument cannot. p=1 keeps the
                # original ``{col}__sin{freq}`` name (back-compat with prior recipes).
                for pwr in fourier_powers:
                    _p = int(pwr)
                    _xp = x if _p == 1 else np.power(x, _p)
                    if not np.all(np.isfinite(_xp)) or float(np.std(_xp)) <= 1e-12:
                        continue
                    lo_f, span_f = _fit_fourier_for_col(_xp)
                    z = (_xp - lo_f) / max(span_f, 1e-12)
                    _pfx = "" if _p == 1 else f"p{_p}"
                    # ADAPTIVE-FREQUENCY (2026-06-03): for the linear argument
                    # (power==1) detect the column's dominant z-space frequency
                    # from a coarse sweep + local refine, held-out validated.
                    # The detected freq is ADDED to this column's freq set and
                    # its sin/cos meta is tagged adaptive=True so MRMR protects
                    # it past screening. Disjoint-by-detection from the fixed
                    # grid: a fixed freq that already recovers the signal makes
                    # the periodogram peak land near it, so the detector's
                    # held-out gate is satisfied by the fixed twin too -- but
                    # we still tag/add the refined freq because the fixed grid
                    # cannot express a non-integer period.
                    _adaptive_freqs: list[float] = []
                    if _p == 1 and _y_adapt is not None and _adaptive_fe_ok:
                        # max_freqs=6: a multitone superposition (3-4 genuine
                        # tones) needs enough sin/cos pairs to SPAN the signal
                        # subspace after the per-iteration deflation leaves a
                        # residual -- 4 pairs recovered the 3-tone gate-A signal
                        # at OOS R^2 ~0.95 but 6 pairs lift it to ~0.985, a far
                        # safer margin above the 0.9 bar. Each extra freq still
                        # passes the held-out 0.30 floor, so noise never inflates
                        # the count (a pure-noise column stops at the first peak).
                        _adaptive_freqs = _detect_fourier_freqs_for_col(
                            z, _y_adapt,
                            f_grid=_adaptive_f_grid,
                            min_val_corr=float(fourier_adaptive_min_val_corr),
                            min_rows=800,
                            max_freqs=6,
                        )
                    _freqs_for_col = list(fourier_freqs)
                    _adaptive_set: set[float] = set()
                    for _af in _adaptive_freqs:
                        if not any(abs(_af - f) < 1e-9 for f in _freqs_for_col):
                            _freqs_for_col.append(_af)
                            _adaptive_set.add(_af)
                    for freq in _freqs_for_col:
                        _is_adaptive = freq in _adaptive_set
                        ang = 2.0 * np.pi * freq * z
                        s_vals = np.sin(ang)
                        c_vals = np.cos(ang)
                        if float(np.std(s_vals)) > 1e-12:
                            name_s = f"{col}__{_pfx}sin{freq:g}"
                            out_cols[name_s] = s_vals
                            meta[name_s] = {
                                "basis": "fourier", "src": col,
                                "kind": "sin", "freq": float(freq),
                                "lo": float(lo_f), "span": float(span_f),
                                "power": _p, "adaptive": _is_adaptive,
                            }
                        if float(np.std(c_vals)) > 1e-12:
                            name_c = f"{col}__{_pfx}cos{freq:g}"
                            out_cols[name_c] = c_vals
                            meta[name_c] = {
                                "basis": "fourier", "src": col,
                                "kind": "cos", "freq": float(freq),
                                "lo": float(lo_f), "span": float(span_f),
                                "power": _p, "adaptive": _is_adaptive,
                            }
                # ADAPTIVE-CHIRP (2026-06-03): a SECOND argument-warp alongside
                # the linear-adaptive path above. The chirp axis u = sign(z)*z**2
                # (z standardised on the column) makes a growing-frequency
                # oscillation ``y ~ sin(2*pi*f*z**2)`` STATIONARY in u, so the
                # SAME held-out-validated multitone detector locks its frequency
                # and the emitted sin/cos on u reconstruct it -- which a Fourier
                # on the linear argument cannot (Phase-0: linear R^2 0.07-0.53 vs
                # chirp 0.88 on a fast chirp). Emitted legs carry arg="quadratic"
                # (the warp the recipe replays) + adaptive=True (so MRMR protects
                # them past the screen, exactly like the linear adaptive legs).
                # Disjoint by name (``__qsin``/``__qcos``) from the linear legs;
                # additive (on a plain linear target the chirp legs are harmless,
                # Ridge regularises them to ~0). N-gated identically (>= 800 rows
                # inside the detector); a pure-noise column admits none.
                if fourier_chirp and _y_adapt is not None and _adaptive_fe_ok:
                    _c_mean, _c_std, _c_lo, _c_span = _fit_chirp_warp_for_col(x)
                    if _c_span > 1e-12 and _c_std > 1e-12:
                        u_axis = _chirp_axis(x, _c_mean, _c_std, _c_lo, _c_span)
                        if np.all(np.isfinite(u_axis)) and float(np.std(u_axis)) > 1e-12:
                            _chirp_freqs = _detect_fourier_freqs_for_col(
                                u_axis, _y_adapt,
                                f_grid=_chirp_f_grid,
                                min_val_corr=float(fourier_chirp_min_val_corr),
                                min_rows=800,
                                max_freqs=6,
                            )
                            for _cf in _chirp_freqs:
                                ang_c = 2.0 * np.pi * _cf * u_axis
                                sc_vals = np.sin(ang_c)
                                cc_vals = np.cos(ang_c)
                                if float(np.std(sc_vals)) > 1e-12:
                                    name_qs = f"{col}__qsin{_cf:g}"
                                    out_cols[name_qs] = sc_vals
                                    meta[name_qs] = {
                                        "basis": "fourier", "src": col,
                                        "kind": "sin", "freq": float(_cf),
                                        "arg": "quadratic",
                                        "mean": float(_c_mean), "std": float(_c_std),
                                        "lo": float(_c_lo), "span": float(_c_span),
                                        "power": 1, "adaptive": True,
                                    }
                                if float(np.std(cc_vals)) > 1e-12:
                                    name_qc = f"{col}__qcos{_cf:g}"
                                    out_cols[name_qc] = cc_vals
                                    meta[name_qc] = {
                                        "basis": "fourier", "src": col,
                                        "kind": "cos", "freq": float(_cf),
                                        "arg": "quadratic",
                                        "mean": float(_c_mean), "std": float(_c_std),
                                        "lo": float(_c_lo), "span": float(_c_span),
                                        "power": 1, "adaptive": True,
                                    }
            except Exception as exc:
                logger.warning(
                    "generate_extra_basis_features: fourier on col=%r raised "
                    "%r; skipping fourier for that column.",
                    col, exc,
                )
    return pd.DataFrame(out_cols, index=X.index), meta


def _build_recipe_from_meta(name: str, meta_entry: dict):
    """Materialise an ``EngineeredRecipe`` from one ``generate_extra_basis_features``
    meta entry. Returns None for unknown basis kinds (defensive)."""
    from ..engineered_recipes import (
        build_orth_spline_recipe, build_orth_fourier_recipe,
    )
    basis = meta_entry["basis"]
    if basis == "spline":
        return build_orth_spline_recipe(
            name=name, src_name=str(meta_entry["src"]),
            knots=np.asarray(meta_entry["knots"], dtype=np.float64),
            idx=int(meta_entry["idx"]),
            lo=float(meta_entry["lo"]), hi=float(meta_entry["hi"]),
        )
    if basis == "fourier":
        return build_orth_fourier_recipe(
            name=name, src_name=str(meta_entry["src"]),
            kind=str(meta_entry["kind"]),
            freq=float(meta_entry["freq"]),
            lo=float(meta_entry["lo"]),
            span=float(meta_entry["span"]),
            power=int(meta_entry.get("power", 1)),
            adaptive=bool(meta_entry.get("adaptive", False)),
            arg=str(meta_entry.get("arg", "linear")),
            mean=(None if meta_entry.get("mean") is None else float(meta_entry["mean"])),
            std=(None if meta_entry.get("std") is None else float(meta_entry["std"])),
        )
    return None


def hybrid_orth_extra_basis_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    extra_bases: Sequence[str] = ("spline", "fourier"),
    fourier_freqs: Sequence[float] = (1.0, 2.0),
    fourier_powers: Sequence[int] = (1, 2),
    spline_knots: int = 5,
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
    fourier_adaptive: bool = False,
    fourier_adaptive_min_val_corr: float = 0.15,
    fourier_chirp: bool = False,
    fourier_chirp_min_val_corr: float = 0.15,
):
    """Layer 32 hybrid: spline + Fourier univariate basis FE + MI-greedy
    selection. Mirrors :func:`hybrid_orth_mi_fe_with_recipes` for the
    polynomial path but emits extra-basis columns (B-spline, Fourier)
    instead. Returns (X_augmented, scores, recipes).

    The selection rule is the same TWO-GATE chain as the polynomial path:
    relative uplift >= min_uplift AND engineered_mi >= max(legacy floor,
    noise-aware floor). See :func:`hybrid_orth_mi_fe` for the rationale.

    ``fourier_adaptive`` (default False) forwards to
    :func:`generate_extra_basis_features` -- when True, each source column's
    dominant z-space frequency is detected (held-out validated) and added to
    its Fourier set, with the emitted sin/cos recipes tagged ``adaptive=True``.

    ``fourier_chirp`` (default False) likewise forwards the ADAPTIVE-CHIRP path:
    the same held-out detector run on the quadratic-argument warp
    ``u = sign(z)*z**2``, emitting ``__qsin``/``__qcos`` legs tagged
    ``arg="quadratic"`` + ``adaptive=True`` (force-admitted past the uplift gate
    and MRMR-protected identically to the linear adaptive legs).
    """
    engineered, meta = generate_extra_basis_features(
        X, cols=cols, extra_bases=extra_bases,
        fourier_freqs=fourier_freqs, fourier_powers=fourier_powers,
        spline_knots=spline_knots,
        y=y, fourier_adaptive=fourier_adaptive,
        fourier_adaptive_min_val_corr=fourier_adaptive_min_val_corr,
        fourier_chirp=fourier_chirp,
        fourier_chirp_min_val_corr=fourier_chirp_min_val_corr,
    )
    if engineered.empty:
        empty_scores = pd.DataFrame(columns=[
            "engineered_col", "source_col", "baseline_mi", "engineered_mi", "uplift",
        ])
        return X.copy(), empty_scores, []
    from . import score_features_by_mi_uplift
    raw_X = X[[c for c in (cols or X.columns) if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]]
    scores = score_features_by_mi_uplift(raw_X, engineered, y, nbins=nbins)
    raw_baselines = scores["baseline_mi"].to_numpy()
    max_raw_baseline = float(raw_baselines.max()) if raw_baselines.size else 0.0
    legacy_floor = float(min_abs_mi_frac) * max_raw_baseline
    n_cands = int(raw_baselines.size)
    sigma_thresh = max(
        5.0,
        float(np.sqrt(2.0 * np.log(max(2.0, 2.0 * n_cands))) + 1.5),
    )
    if raw_baselines.size >= 4:
        med = float(np.median(raw_baselines))
        mad = float(np.median(np.abs(raw_baselines - med)))
        noise_floor = med + sigma_thresh * 1.4826 * mad
    else:
        noise_floor = 0.0
    eng_mis = scores["engineered_mi"].to_numpy()
    if eng_mis.size >= 4:
        med_e = float(np.median(eng_mis))
        mad_e = float(np.median(np.abs(eng_mis - med_e)))
        eng_noise_floor = med_e + sigma_thresh * 1.4826 * mad_e
    else:
        eng_noise_floor = 0.0
    abs_floor = max(legacy_floor, noise_floor, eng_noise_floor)
    qualified = scores[
        (scores["uplift"] >= float(min_uplift))
        & (scores["engineered_mi"] >= abs_floor)
    ]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    # FORCE-ADMIT adaptive Fourier columns: a single adaptive sin OR cos has a
    # LOW marginal |corr| / MI for a phase-shifted oscillation (the phase is
    # split between the two), so the per-column MI-uplift gate would drop them
    # even though the sin+cos PAIR recovers the signal. The adaptive detector
    # already validated the frequency on a held-out slice, so both legs are
    # admitted unconditionally here; the downstream MRMR adaptive-protection
    # block then keeps them past screening. Append in deterministic name order.
    _adaptive_names = [
        nm for nm, m in meta.items()
        if m.get("basis") == "fourier" and m.get("adaptive", False)
    ]
    _keep_set = set(keep)
    for nm in _adaptive_names:
        if nm not in _keep_set and nm in engineered.columns:
            keep.append(nm)
            _keep_set.add(nm)
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    recipes = []
    for name in keep:
        if name not in meta:
            continue
        r = _build_recipe_from_meta(name, meta[name])
        if r is not None:
            recipes.append(r)
    return X_aug, scores, recipes
