"""Haar wavelet / localized multiresolution basis FE.

A NEW univariate operator for a signal shape the catalog cannot capture: a
**localized bump / multiscale piecewise structure** - ``y`` jumps only inside a
narrow sub-interval of x, or has step/contrast structure at SEVERAL scales at
once. The closest existing operators all have the WRONG form:

* Fourier is **GLOBAL** - a localized bump forces a long tone sum and the
  truncated series RINGS (Gibbs) around the discontinuity;
* cubic B-spline knots are placed at **FIXED quantiles** of x (unsupervised), so
  a bump that falls between knots is SMOOTHED AWAY;
* ``numeric_rounding`` is a global flat-step quantiser, blind to location.

Wavelets are simultaneously localized in x AND multiscale, so a small dyadic
Haar set captures a sharp local contrast at the right scale with a couple of legs
where a global basis needs many (and still rings).

Mechanism
---------
On x's support, normalise ``z = clip((x - lo) / span, 0, 1)`` and emit a SMALL
dyadic set of **Haar wavelet indicators** ``psi_{j,k}(z)``: ``+1`` on the LEFT
half / ``-1`` on the RIGHT half of the dyadic interval
``[k/2^j, (k+1)/2^j)``, ``0`` outside. Scales ``j = 0 .. max_scale`` (default 3),
positions ``k = 0 .. 2^j - 1``. Each ``psi_{j,k}`` is a localized step/contrast
detector at scale ``2^{-j}`` centred on a dyadic position - a multiresolution
edge dictionary.

Candidate explosion control (the load-bearing risk)
---------------------------------------------------
Emitting all ``sum_j 2^j`` legs (15 for ``max_scale=3``) per column would flood
the candidate pool. Two self-limiting bounds keep it small:

1. **Held-out scale-selection** (:func:`_select_wavelet_legs`): each leg is RANKED
   by its TRAIN-side (``idx % 3 != 0``) marginal MI vs y, and a leg is kept only
   if its HELD-OUT (``idx % 3 == 0``) marginal MI clears a noise-aware MAD floor
   computed over the candidate legs' held-out MIs. A leg that fits a train slice
   by chance fails the held-out floor -> dropped. Pure noise -> EVERY leg fails
   the held-out floor -> 0 legs emitted (verified). Only the top ``max_legs``
   (default 6) survivors per column are emitted.
2. The downstream :func:`hybrid_wavelet_fe_with_recipes` then re-applies the same
   MI-uplift gate + noise-aware MAD floor the spline / Fourier extra-basis path
   uses (:func:`score_features_by_mi_uplift`), so a surviving leg must ALSO beat
   its raw source's MI - a second, pool-level self-limit.

Why MI-gateable (unlike the hinge)
----------------------------------
A Haar leg ``psi_{j,k}`` is NON-monotone in x (it is +1 then -1 then 0), so it is
NOT MI-invariant by the data-processing inequality - a leg in the RIGHT window
carries genuine MARGINAL MI about a localized target (unlike a single Fourier
phase-leg, whose MI is split across sin/cos, or the monotone hinge/isotonic legs
that MI cannot see). So the wavelet routes through the NORMAL MI-uplift gate, no
deferred-materialisation / re-add dance is needed (contrast the hinge, backlog
#11, which is MI-invariant and needs the protection roster).

Leak-safe replay
----------------
The recipe (kind ``"orth_wavelet"``) stores only ``(lo, span)`` + the dyadic
``(j, k)`` - NO y - so ``transform`` replay is the closed-form indicator
``_dyadic_haar_leg(clip((x - lo) / span, 0, 1), j, k)``. The scale SELECTION
consumes y at FIT time (like every supervised FE here - spline knot placement,
Fourier frequency detection, hinge breakpoint search) but the emitted COLUMN
VALUE does not depend on y, so the replayed feature is leakage-free by
construction. Structurally identical to ``orth_spline`` (store basis params +
``lo``/``span``, replay a closed-form basis function of the source column alone).

Mirrors the spline / Fourier extra-basis FE module
(``_orthogonal_univariate_fe._orth_extra_basis_fe``):
``generate_wavelet_features`` emits columns + per-column fit meta,
``hybrid_wavelet_fe_with_recipes`` scores by MI uplift, applies the two-gate
(uplift + noise-aware MAD floor) chain, and returns ``EngineeredRecipe`` objects
for leak-safe transform-time replay.
"""
from __future__ import annotations

import logging
from typing import Optional, cast

import numpy as np
from numba import njit, prange
from pyutilz.performance.kernel_tuning.registry import kernel_tuner

logger = logging.getLogger(__name__)

# The recipe-application layer (generate_wavelet_features, hybrid_wavelet_fe_with_recipes,
# build_orth_wavelet_recipe, _apply_orth_wavelet) lives in the sibling ``_wavelet_basis_fe_recipes``
# (carved out to keep this module under the 1k LOC ceiling); consumers import from there directly.
__all__ = [
    "_dyadic_haar_leg",
    "_select_wavelet_legs",
]


# Coarsest..finest dyadic scales. j=0 is the root contrast (left vs right half of
# the whole support, = psi_{0,0}); j=3 resolves features at 1/8 of the support.
# Beyond j=3 a leg spans < ~1/16 of the range and on n<=4000 its half-cells hold
# too few rows for a trustworthy held-out MI - so cap at 3 (the backlog's
# j=0..3). The TOTAL leg count before selection is sum_{j=0}^{3} 2^j = 15.
_WAVELET_MAX_SCALE: int = 3
# Max legs EMITTED per column after held-out scale-selection. The held-out MAD
# floor already drops chance legs; this is a hard cap so even a richly-structured
# column adds at most a handful of legs to the pool (candidate-count control).
_WAVELET_MAX_LEGS: int = 6
# N-gate: below this row count a held-out slice cannot validate a fine-scale leg
# reliably (a j=3 half-cell would hold < ~80 rows). Mirrors the hinge >=200.
_WAVELET_MIN_ROWS: int = 200
# Min rows in EACH non-zero half-cell of a candidate leg for its MI to be
# trustworthy. A j=3 leg over uniform x on n=4000 has ~250 rows per half; this
# floor rejects a fine leg whose window happens to be near-empty (sparse x).
_WAVELET_MIN_HALF_ROWS: int = 30
# Sigma multiplier for the held-out noise-aware MAD floor in scale-selection.
# A leg's held-out MI must exceed ``median + SIGMA * 1.4826 * MAD`` of the
# candidate legs' held-out MIs. 3.5 mirrors the orthogonal-cluster abs-floor
# default and is conservative: a chance leg's held-out MI sits in the band, a
# genuine localized leg is a multi-sigma outlier above it. On pure noise every
# leg sits in the band -> none clears -> 0 legs (verified).
_WAVELET_SCALE_SIGMA: float = 3.5
# Absolute held-out MI floor (nats) a surviving leg must also clear, so that on a
# DEGENERATE near-constant candidate set (all held-out MIs ~0, MAD ~0) the MAD
# floor doesn't collapse to ~0 and admit noise. 1e-3 nats is well below a genuine
# localized leg (measured ~0.05-0.2) and above pure-noise held-out MI (~1e-3 or
# less on n=4000, 10 bins binary y).
_WAVELET_MIN_HELDOUT_MI: float = 1e-3
# POOL-LEVEL ADMISSION FLOOR: minimum held-out INCREMENTAL MI of a leg OVER the
# binned raw source column (nats) for the leg to enter support. This is the gate
# that makes the operator self-limiting + complementary. Measured (n=4000, 10
# bins, %3 split): genuine localized BUMP/STEP legs lift the joint MI by
# +0.02..+0.04; a leg over a SMOOTH (linear-usable) column adds <= 0 (raw x
# already carries it, the global Fourier basis owns that regime); a chance leg
# over pure NOISE adds ~+3e-4. 0.005 sits cleanly in the gap: it admits the
# localized winners and rejects smooth (negative) + noise (~3e-4). The naive
# leg-MI-vs-raw-MI uplift gate MIS-FIRES here (a localized y is a function of x,
# so binned raw x already scores high MI and a single leg's marginal MI sits
# below it -> uplift<1 -> wrongly dropped); the incremental gate is the correct,
# more honest statistic (see :func:`_heldout_incremental_mi`).
_WAVELET_MIN_INCR_MI: float = 0.005
# COMPLEMENTARITY GUARD: a leg's held-out incremental MI must also exceed this
# fraction of the SMOOTH-refinement gain (what finer location-only binning of raw
# x adds over the same coarse baseline). This is what makes the operator
# complementary to the global Fourier / spline basis rather than a redundant twin:
# on a SMOOTH (sin / monotone) column, finer location-binning captures the signal
# and ``smooth_gain`` dominates ``leg_incr`` -> the leg is rejected (Fourier owns
# that regime); on a LOCALIZED step / contrast, the leg nails the sharp
# discontinuity that uniform finer binning only resolves slowly, so ``leg_incr``
# dominates -> admitted. Measured (n=4000): STEP leg_incr/smooth_gain ~3.0, BUMP
# ~0.85, SMOOTH ~0.20. 0.5 sits in the gap: it admits step/bump and rejects the
# occasional smooth false-positive that the bare ``min_incr_mi`` floor let through
# (smooth FP rate 2/10 seeds -> 0/10 with this guard; step/bump unaffected).
_WAVELET_SMOOTH_COMPLEMENT_RATIO: float = 0.5
# PERMUTATION-NULL DEBIAS for the held-out incremental MI. Plug-in joint MI over the
# (binned-x x 3-cell-leg) contingency is finite-sample POSITIVELY biased: the joint has
# up to ~3*nbins cells, so on a small held-out slice (~n/3 rows) even a pure-noise leg
# scores a chance incremental MI in the ~2e-3..7e-3 band (n=1500), which overruns the bare
# ``_WAVELET_MIN_INCR_MI`` floor and emits spurious noise legs. We subtract the max
# incremental MI under K y-shuffles (same binning, same cells) so the statistic measures the
# leg's value ABOVE its own finite-sample bias: a localized step/bump keeps a large positive
# debiased incr; a pure-noise leg centers at ~0 and fails the floor. K small (the null mean is
# a stable estimate - its variance shrinks as 1/K and the floor has margin).
_WAVELET_INCR_NULL_PERMS: int = 8


@njit(cache=True, parallel=True)
def _dyadic_haar_leg_njit(z: np.ndarray, left: float, mid: float, right: float, out: np.ndarray) -> None:
    """Single-pass fused build of the Haar leg into ``out`` (caller-owned, right dtype). Replaces the
    zeros-alloc + two separate boolean-mask + fancy-index numpy passes (4 full array traversals) with one
    prange loop doing the 2 comparisons and the write per element in one visit - the whole op is
    memory-bandwidth-bound, so collapsing traversals is the entire win. Bit-identical (same {-1,0,+1}
    membership test, evaluated per-element in the same order).

    Only wins over the serial twin (:func:`_dyadic_haar_leg_njit_serial`) at extreme n (>~20-50M) --
    see :func:`_dyadic_haar_leg`'s dispatch threshold for the measured crossover; this variant stays for
    that regime, but is NOT the default entry point any more."""
    n = z.shape[0]
    for i in prange(n):
        v = z[i]
        if left <= v < mid:
            out[i] = 1.0
        elif mid <= v < right:
            out[i] = -1.0
        else:
            out[i] = 0.0


@njit(cache=True, parallel=False)
def _dyadic_haar_leg_njit_serial(z: np.ndarray, left: float, mid: float, right: float, out: np.ndarray) -> None:
    """Serial twin of :func:`_dyadic_haar_leg_njit` -- identical body, ``range`` instead of ``prange``.

    ``parallel=True`` pays a fixed per-call thread-pool dispatch cost that a 3-way compare-and-write
    (trivial per-element work) cannot amortize at any n this codebase's FE search realistically reaches
    (test/production data: hundreds to a few million rows). Measured (same-process A/B, warm,
    best-of-30-200, ``bench_dyadic_haar_leg_parallel_vs_serial.py``): serial is 16x-12681x FASTER at
    n<=1,000,000, still 4.55x faster at n=5,000,000; the crossover only appears between n=20M (parallel
    1.18x) and n=50M (parallel 1.33x) -- see :func:`_dyadic_haar_leg`'s threshold for where this is used."""
    n = z.shape[0]
    for i in range(n):
        v = z[i]
        if left <= v < mid:
            out[i] = 1.0
        elif mid <= v < right:
            out[i] = -1.0
        else:
            out[i] = 0.0


# Per-host serial/parallel crossover via the canonical kernel_tuning_cache (NO hardcoded threshold --
# feedback_use_kernel_tuning_cache_for_gpu / feedback_fastest_default_with_dispatch). Dev-box measurement
# found the crossover between n=20M (parallel 1.18x) and n=50M (parallel 1.33x) -- the fallback threshold
# below (used only pre-sweep / on tuner failure) is deliberately close to the loss side for that reason,
# but the REAL per-host decision comes from the tuner's measured sweep, not this constant.
_HAAR_LEG_SWEEP_N = [1_000, 1_000_000, 30_000_000]
_HAAR_LEG_SALT = 1


def _make_haar_leg_inputs(dims: dict) -> tuple:
    """(z, left, mid, right) at the sweep's n cell -- a fixed (j=1, k=0) scale/offset is enough since
    the kernel's per-element cost doesn't depend on which dyadic cell is tested."""
    n = int(dims["n"])
    rng = np.random.default_rng(0)
    z = np.ascontiguousarray(rng.random(n), dtype=np.float64)
    return (z, 0.0, 0.25, 0.5)


def _run_haar_leg_sweep() -> list:
    """Serial-vs-parallel wall-clock sweep over the n grid -> kernel_tuning_cache regions."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    def _call(fn, z, left, mid, right):
        """Allocate the output buffer and run the in-place kernel ``fn``, returning ``out`` so
        ``sweep_backend_grid`` has a value to compare for equivalence."""
        out = np.empty(z.shape[0], dtype=np.float32)
        fn(z, left, mid, right, out)
        return out

    variants = {
        "serial": lambda *a: _call(_dyadic_haar_leg_njit_serial, *a),
        "parallel": lambda *a: _call(_dyadic_haar_leg_njit, *a),
    }
    return cast(list, sweep_backend_grid(
        variants,
        {"n": _HAAR_LEG_SWEEP_N},
        _make_haar_leg_inputs,
        reference="serial", repeats=5, equiv_atol=0.0, equiv_rtol=0.0,
    ))


def _haar_leg_fallback_choice(n: int) -> str:
    """Pre-sweep / tuner-failure fallback: parallel above the dev-box-measured n crossover (see the
    module comment above for the confirmed-loss/confirmed-win bracket)."""
    return "parallel" if int(n) >= 20_000_000 else "serial"


_HAAR_LEG_PARALLELISM_SPEC = kernel_tuner(
    kernel_name="haar_leg_kernel_parallelism",
    variant_fns=(_dyadic_haar_leg_njit_serial, _dyadic_haar_leg_njit),
    tuner=_run_haar_leg_sweep,
    axes={"n": _HAAR_LEG_SWEEP_N},
    fallback=_haar_leg_fallback_choice,
    gpu_capable=False,
    salt=_HAAR_LEG_SALT,
    cli_label="haar_leg_kernel_parallelism",
)


def _dyadic_haar_leg(z: np.ndarray, j: int, k: int, dtype=np.float32) -> np.ndarray:
    """Closed-form Haar wavelet indicator ``psi_{j,k}(z)`` for ``z`` in [0, 1].

    ``+1`` on the LEFT half ``[k/2^j, (k+0.5)/2^j)``, ``-1`` on the RIGHT half
    ``[(k+0.5)/2^j, (k+1)/2^j)``, ``0`` outside. Pure function of ``z`` - no y,
    no fitted state beyond the (j, k) integers, so it replays leak-free.

    The output is allocated in ``dtype`` (float32 by default, the large-n working
    dtype). The leg holds only the exact values {-1, 0, +1}, which are bit-exact in
    float32 -> every downstream consumer (binned MI via np.unique/searchsorted, the
    engineered column) is bit-identical to float64 while halving the (n_scales, n)
    working-array footprint (e.g. (10, 1M) = 76 MiB -> 38 MiB). The dyadic-cell
    boolean masks are computed against the float64 ``z`` axis, so the cell
    membership (and hence the leg) does not depend on the output dtype.

    PERF (2026-08-03, incidental to a profiling cycle): fused into one njit pass (originally
    :func:`_dyadic_haar_leg_njit`) - the prior form allocated a zeros array then wrote it via two
    separate boolean-mask + fancy-index passes (4 full traversals of a memory-bandwidth-bound op).

    PERF (2026-08-23): that fused pass was ``parallel=True``, which turned out to cost 16x-12681x
    MORE than a serial pass at every n this codebase's FE search realistically reaches (a 3-way
    compare-and-write is too trivial per-element to amortize numba's fixed per-call thread-pool
    dispatch cost) -- see :func:`_dyadic_haar_leg_njit_serial` and ``_HAAR_LEG_PARALLEL_MIN_N`` for
    the measured crossover. Dispatches by n now; bit-identical either way (same per-element test,
    independent of which thread -- if any -- runs it)."""
    width = 1.0 / (2 ** int(j))
    left = int(k) * width
    mid = left + width / 2.0
    right = left + width
    leg = np.empty_like(z, dtype=dtype)
    zc = np.ascontiguousarray(z, dtype=np.float64)
    try:
        choice = _HAAR_LEG_PARALLELISM_SPEC.choose(n=int(zc.shape[0]))
    except Exception as e:
        logger.debug("haar_leg_kernel_parallelism choose() failed, using the size-based fallback: %s", e)
        choice = _haar_leg_fallback_choice(int(zc.shape[0]))
    fn = _dyadic_haar_leg_njit if choice == "parallel" else _dyadic_haar_leg_njit_serial
    fn(zc, left, mid, right, leg)
    return leg


def _bin_y_codes(y: np.ndarray, nbins: int = 10) -> np.ndarray:
    """Bin the target into integer codes exactly as :func:`_binned_mi` does
    internally. Hoisted so the (per-leg-invariant) y-subset binning can be
    computed once per ``_select_wavelet_legs`` call and reused across all legs.
    Byte-identical to the inline path in ``_binned_mi``."""
    y = np.asarray(y).ravel()
    if np.issubdtype(y.dtype, np.integer) and np.unique(y).size <= 20:
        return y.astype(np.int64)
    if np.unique(y).size <= 20:
        uy = np.unique(y)
        return np.searchsorted(uy, y)
    edges_y = np.quantile(y, np.linspace(0.0, 1.0, nbins + 1)[1:-1])
    return np.asarray(np.digitize(y, edges_y))


def _binnedmi_gpu_enabled(*, n: int | None = None, p: int | None = None) -> bool:
    """Route wavelet leg-rank binned-MI to the GPU when STRICT-GPU / MLFRAME_CMI_GPU is on. Default OFF.

    ``n``/``p`` (optional): the calling dispatch's own shape, forwarded to ``fe_gpu_strict_enabled`` so the
    STRICT/AUTO decision is size-aware for THIS call. Omit to preserve the shape-blind default."""
    import os as _os
    if _os.environ.get("MLFRAME_CMI_GPU", "") == "1":
        return True
    try:
        from ._fe_gpu_strict import fe_gpu_strict_enabled
        return bool(fe_gpu_strict_enabled(n=n, p=p))
    except Exception as e:
        logger.debug("_binnedmi_gpu_enabled: fe_gpu_strict_enabled() check failed, staying on the host binned-MI path: %s", e)
        return False


def _binned_mi_cupy(feat, y, nbins: int, y_codes, discrete: bool = False) -> float:
    """Device twin of :func:`_binned_mi`: bin feat/y + MI = H(feat)+H(y)-H(feat,y) via cp.unique partition
    counts. Same partition -> same MI -> same leg ranking (selection-identical, fp-order ~1e-15).

    ``discrete``: the caller guarantees ``feat`` (and, when ``y_codes`` is None, ``y``) are ALREADY small
    integer bin codes (the held-out incremental-MI path passes ``xc`` / ``xc*3+legcode`` / class codes).
    Then the ``cp.unique`` densify - a FULL device sort just to dedupe already-discrete values, the single
    largest MergeSort source in the F2 STRICT profile - is skipped: the codes are used directly (offset by
    min, a label relabel that leaves the partition, hence the plug-in MI, identical) and scored by the
    fused MI-from-codes RawKernel. Selection-EXACT (no approximation), no sort."""
    import cupy as cp

    # y / y_codes are the FIT-CONSTANT target (invariant across every leg of a source column, AND across
    # every source column in the fit, since the %3 train/held-out mask is the same for all of them) -
    # resident-cache them so repeated calls within one fit (up to 6 legs/column via _heldout_incremental_mi)
    # upload ONCE instead of every call. feat/df is the candidate leg/joint code, which genuinely varies per
    # call, and stays a raw upload.
    from ._fe_resident_operands import resident_operand

    df = cp.asarray(np.asarray(feat, dtype=np.float64).ravel())
    if discrete:
        from ._fe_batched_mi import binned_mi_from_codes_gpu
        fb = (df - df.min()).astype(cp.int64)
        if y_codes is not None:
            yb = resident_operand(np.asarray(y_codes).ravel(), "wavelet_y_codes", dtype=np.int64)
        else:
            dy = resident_operand(np.asarray(y).ravel(), "wavelet_y", dtype=np.int64)
            yb = dy - dy.min()
        return float(max(float(binned_mi_from_codes_gpu(fb[:, None], yb)[0]), 0.0))
    # Sort-free EXACT quantile edges via radix-select (launch-reduction): cp.quantile bins via a comparison
    # MERGE-sort; _radix_select_interior_edges returns the SAME interior order-statistic edges WITHOUT a sort
    # (bit-identical codes through cp.searchsorted/digitize, maxdiff 0). The cp.unique low-card check stays
    # (it picks searchsorted-vs-quantile to MATCH the CPU _binned_mi decision). Falls back to cp.quantile
    # when the radix path is inapplicable/disabled.
    def _interior_edges(v):
        """Compute equi-frequency bin-interior edges for ``v`` via the sort-free radix-select path when enabled (bit-identical to, but faster than, a comparison-sort quantile), falling back to ``cp.quantile`` when the radix path is disabled or unavailable."""
        try:
            from ._gpu_resident_select import _radix_select_interior_edges, fe_gpu_radix_edges_enabled
            if fe_gpu_radix_edges_enabled():
                e = _radix_select_interior_edges(v.reshape(-1, 1), nbins)
                if e is not None:
                    return e.ravel()
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed: %s", e)
            pass
        return cp.quantile(v, cp.linspace(0.0, 1.0, nbins + 1)[1:-1])

    uf = cp.unique(df)
    if int(uf.size) <= nbins:
        fb = cp.searchsorted(uf, df)
    else:
        fb = cp.digitize(df, _interior_edges(df))
    if y_codes is not None:
        yb = resident_operand(np.asarray(y_codes).ravel(), "wavelet_y_codes")
    else:
        dy = resident_operand(np.asarray(y).ravel(), "wavelet_y") if not isinstance(y, cp.ndarray) else y.ravel()
        uy = cp.unique(dy)
        if int(uy.size) <= 20:
            yb = cp.searchsorted(uy, dy)
        else:
            yb = cp.digitize(dy.astype(cp.float64), _interior_edges(dy.astype(cp.float64)))

    # Fused one-launch MI-from-codes (launch-reduction): the binned feat/y codes ``fb``/``yb`` are already
    # valid bin indices, so MI(feat; y) = H(feat)+H(y)-H(feat,y) goes through the single RawKernel
    # (codes -> shared-mem joint histogram -> plug-in MI) instead of three cp.unique-sort + entropy passes.
    # Same plain plug-in MI partition -> selection-equivalent. Falls back internally to the cupy path if the
    # (Kx*Ky) shared tile would not fit.
    from ._fe_batched_mi import binned_mi_from_codes_gpu
    fb = fb.astype(cp.int64, copy=False)
    yb = yb.astype(cp.int64, copy=False)
    mi = float(binned_mi_from_codes_gpu(fb[:, None], yb)[0])
    return float(max(mi, 0.0))


def _binned_mi(feat: np.ndarray, y: np.ndarray, nbins: int = 10, y_codes: Optional[np.ndarray] = None, discrete: bool = False) -> float:
    """Plug-in binned MI(feat; y) in nats. y is treated as discrete classes if it
    has <= 20 unique values, else quantile-binned into ``nbins``. Used only for
    the held-out scale-selection ranking (the pool-level admission reuses the
    project's ``_mi_classif_batch``).

    ``y_codes``: optional precomputed integer y-binning (from :func:`_bin_y_codes`).
    When given, the per-call y re-binning is skipped (byte-identical result). The
    default ``None`` preserves the original behavior for every other caller."""
    feat = np.asarray(feat, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    n = feat.size
    if n == 0 or n != y.size:
        return 0.0
    # GPU route: MI(feat;y) = H(feat)+H(y)-H(feat,y) on device via cp.unique partition counts
    # (the binning's np.unique/digitize over n rows is the wavelet leg-rank hot cost). Same partition ->
    # same MI -> same leg RANKING (selection-identical). Gated (STRICT / MLFRAME_CMI_GPU), default CPU.
    if _binnedmi_gpu_enabled(n=int(n), p=1):
        try:
            return _binned_mi_cupy(feat, y, int(nbins), y_codes, discrete=discrete)
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed: %s", e)
            pass
    # Feature is a Haar leg taking values in {-1, 0, +1} -> use those as classes
    # directly (3 cells); avoids quantile-binning a ternary column.
    uniq_f = np.unique(feat)
    if uniq_f.size <= nbins:
        fb = np.searchsorted(uniq_f, feat)
    else:
        edges = np.quantile(feat, np.linspace(0.0, 1.0, nbins + 1)[1:-1])
        fb = np.digitize(feat, edges)
    if y_codes is not None:
        yb = y_codes
    elif np.issubdtype(y.dtype, np.integer) and np.unique(y).size <= 20:
        yb = y.astype(np.int64)
    elif np.unique(y).size <= 20:
        uy = np.unique(y)
        yb = np.searchsorted(uy, y)
    else:
        edges_y = np.quantile(y, np.linspace(0.0, 1.0, nbins + 1)[1:-1])
        yb = np.digitize(y, edges_y)
    # Joint-histogram MI: the prior O(|fa|*|yb|*n) double loop recomputed an O(n) boolean
    # mask per contingency cell; a single bincount over the dense joint code yields the same
    # plug-in counts. Bit-identical by construction - pa/pb/pab are still count/n float64 and
    # the (a ascending, b ascending) over-nonzero-pab summation order is preserved row-major.
    fa_vals, fa_inv = np.unique(fb, return_inverse=True)
    yb_vals, yb_inv = np.unique(yb, return_inverse=True)
    n_b = yb_vals.size
    joint_code = fa_inv.astype(np.int64) * n_b + yb_inv.astype(np.int64)
    joint_counts = np.bincount(joint_code, minlength=fa_vals.size * n_b).reshape(fa_vals.size, n_b)
    ca = joint_counts.sum(axis=1)
    cb = joint_counts.sum(axis=0)
    nf = float(n)
    pa_row = ca / nf
    pb_col = cb / nf
    mi = 0.0
    for ai in range(fa_vals.size):
        pa = pa_row[ai]
        if pa <= 0:
            continue
        for bi in range(n_b):
            cab = joint_counts[ai, bi]
            if cab > 0:
                pab = cab / nf
                mi += pab * np.log(pab / (pa * pb_col[bi]))
    return float(max(mi, 0.0))


def _x_codes(v: np.ndarray, nbins: int = 10) -> np.ndarray:
    """Quantile-bin a continuous column into <= nbins integer codes (or use the
    distinct values directly if low-cardinality). Helper for the joint-MI
    admission gate."""
    v = np.asarray(v, dtype=np.float64).ravel()
    u = np.unique(v)
    if u.size <= nbins:
        return np.searchsorted(u, v)
    edges = np.quantile(v, np.linspace(0.0, 1.0, nbins + 1)[1:-1])
    return np.digitize(v, edges)


def _heldout_incremental_mi_prep(x: np.ndarray, y: np.ndarray, *, nbins: int = 10) -> Optional[dict]:
    """Per-SOURCE-COLUMN prep for :func:`_heldout_incremental_mi`: everything depending only on ``(x, y,
    nbins)`` - NOT on any particular leg - computed ONCE per source column, including the expensive 8-shuffle
    permutation-null baseline (``Yp`` / ``bm``). A column can emit up to ``_WAVELET_MAX_LEGS`` legs, each of
    which used to redo this identical work; :func:`hybrid_wavelet_fe_with_recipes` now groups legs by source
    column and calls this once per group, reusing the result via :func:`_heldout_incremental_mi_from_prep`.
    Returns ``None`` on the same guards the original function used to return ``(0.0, 0.0)`` for - the caller
    then scores every leg of this column as ``(0.0, 0.0)``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    n = x.size
    if n != y.size or n < _WAVELET_MIN_ROWS:
        return None
    idx = np.arange(n)
    va = (idx % 3) == 0
    if int(va.sum()) < 32:
        return None
    xc = _x_codes(x[va], nbins=nbins)
    y_va = np.asarray(y[va])
    base_nb = max(nbins, int(xc.max()) + 1)
    base_mi = _binned_mi(xc.astype(np.float64), y_va, nbins=base_nb, discrete=True)

    n_perm = int(_WAVELET_INCR_NULL_PERMS)
    Yp: Optional[np.ndarray] = None
    bm: Optional[np.ndarray] = None
    if n_perm > 0:
        _rng = np.random.default_rng(0)
        xc_f = xc.astype(np.float64)
        Yp = np.empty((int(y_va.size), n_perm), dtype=np.int64)
        for _p in range(n_perm):
            Yp[:, _p] = y_va[_rng.permutation(y_va.size)]
        # bm (the null's xc-side term) depends only on (xc, y_va, Yp) - leg-independent, unlike jm (the
        # joint-side term, computed per leg in _heldout_incremental_mi_from_prep). BATCHED (launch-reduction):
        # stack the n_perm permuted-y columns and score them all in ONE binned_mi_from_codes_gpu workload -
        # the SAME plain plug-in-MI kernel _binned_mi(discrete) uses, so selection-equivalent.
        # cupy's own presence check ignores the global GPU opt-out, so without this the batched null would
        # route to the device on a run that declared no GPU. bm=None falls through to the exact host kernel.
        from ._gpu_policy import gpu_globally_disabled

        if gpu_globally_disabled():
            bm = None
        else:
            try:
                import cupy as cp

                from ._fe_batched_mi import binned_mi_from_codes_gpu

                n_cls = int(y_va.max()) + 1 if y_va.size else 1
                Yp_d = cp.asarray(Yp)
                xc_d = cp.asarray(np.ascontiguousarray(xc.astype(np.int64)))
                bm = np.asarray(binned_mi_from_codes_gpu(Yp_d, xc_d, kx_per_col=[n_cls] * n_perm, ky=int(base_nb)), dtype=np.float64)
            except Exception as e:
                logger.debug("binned_mi_from_codes_gpu (base) failed, falling back to the CPU path: %s", e)
                bm = None
        if bm is None:
            bm = np.empty(n_perm, dtype=np.float64)
            for _p in range(n_perm):
                bm[_p] = _binned_mi(xc_f, Yp[:, _p], nbins=base_nb, discrete=True)

    # SMOOTH-refinement competitor: finer (2*nbins) location binning of raw x. Leg-independent.
    xc_fine = _x_codes(x[va], nbins=2 * nbins)
    fine_mi = _binned_mi(
        xc_fine.astype(np.float64), y[va], nbins=max(2 * nbins, int(xc_fine.max()) + 1), discrete=True,
    )
    smooth_gain = float(fine_mi - base_mi)
    return {
        "va": va, "xc": xc, "y_va": y_va, "base_mi": base_mi, "base_nb": base_nb,
        "n_perm": n_perm, "Yp": Yp, "bm": bm, "smooth_gain": smooth_gain,
    }


def _heldout_incremental_mi_from_prep(prep: Optional[dict], leg: np.ndarray) -> tuple[float, float]:
    """Leg-dependent completion of :func:`_heldout_incremental_mi` given a source column's cached ``prep``
    (:func:`_heldout_incremental_mi_prep`): only ``joint`` / ``joint_mi`` / the null's joint-side term (``jm``)
    are recomputed per leg; everything else is reused from ``prep``."""
    if prep is None:
        return 0.0, 0.0
    va = prep["va"]
    leg = np.asarray(leg, dtype=np.float64).ravel()
    if leg.size != va.size:
        return 0.0, 0.0
    xc = prep["xc"]
    y_va = prep["y_va"]
    base_mi = prep["base_mi"]
    legc = np.asarray(leg[va], dtype=np.float64)
    # 3-cell leg codes {-1,0,+1} -> {0,1,2}; joint code = xc * 3 + legcode.
    leg_code = np.searchsorted(np.array([-1.0, 0.0, 1.0]), legc)
    leg_code = np.clip(leg_code, 0, 2)
    joint = xc * 3 + leg_code
    joint_mi = _binned_mi(joint.astype(np.float64), y_va, nbins=int(joint.max()) + 1, discrete=True)
    leg_incr_raw = float(joint_mi - base_mi)
    # PERMUTATION-NULL DEBIAS: subtract the worst-shuffle incremental MI so the leg's finite-sample plug-in
    # bias (the extra leg cells inflate the joint MI even on noise) cancels. The permutations themselves
    # (``Yp``) and the null's xc-side term (``bm``) are cached in ``prep``; only the joint-side term is redone.
    n_perm = prep["n_perm"]
    Yp = prep["Yp"]
    bm = prep["bm"]
    if n_perm > 0 and Yp is not None and bm is not None:
        joint_f = joint.astype(np.float64)
        joint_nb = int(joint.max()) + 1
        null = None
        # Same opt-out gate as the bm side above: the host fallback below is exact, so declining costs only speed.
        from ._gpu_policy import gpu_globally_disabled

        if not gpu_globally_disabled():
            try:
                import cupy as cp

                from ._fe_batched_mi import binned_mi_from_codes_gpu

                n_cls = int(y_va.max()) + 1 if y_va.size else 1
                Yp_d = cp.asarray(Yp)
                joint_d = cp.asarray(np.ascontiguousarray(joint.astype(np.int64)))
                jm = np.asarray(binned_mi_from_codes_gpu(Yp_d, joint_d, kx_per_col=[n_cls] * n_perm, ky=int(joint_nb)), dtype=np.float64)
                null = jm - bm
            except Exception as e:
                logger.debug("binned_mi_from_codes_gpu (joint) failed, falling back to the CPU path: %s", e)
                null = None
        if null is None:
            null = np.empty(n_perm, dtype=np.float64)
            for _p in range(n_perm):
                null[_p] = _binned_mi(joint_f, Yp[:, _p], nbins=joint_nb, discrete=True) - bm[_p]
        # Subtract the MAX incremental MI over the shuffles: the leg survives only if its incremental MI beats
        # every shuffled-y replicate - a permutation test (p < 1/(n_perm+1)) on the leg's extra cells. The
        # extra contingency cells inflate the joint MI even on noise (finite-sample plug-in bias) and that bias
        # has high variance on the small held-out slice, so a pure-noise leg can still hit a large positive raw
        # incr; subtracting the worst shuffle cancels both the bias and its spread. A genuine localized
        # step/bump clears the null by a wide margin; a noise leg lands at <= 0.
        leg_incr = float(leg_incr_raw - float(null.max()))
    else:
        leg_incr = leg_incr_raw
    return leg_incr, prep["smooth_gain"]


def _heldout_incremental_mi(
    x: np.ndarray, leg: np.ndarray, y: np.ndarray, *, nbins: int = 10,
) -> tuple[float, float]:
    """Held-out INCREMENTAL MI of adding ``leg`` to the binned raw column ``x``,
    scored on the ``%3`` stride slice, PLUS the gain of a SMOOTH refinement
    competitor. Returns ``(leg_incr, smooth_gain)``:

    * ``leg_incr = MI(y; [bin_{nbins}(x), leg])_va - MI(y; bin_{nbins}(x))_va`` -
      what the localized Haar leg adds ON TOP of the coarse binned raw column.
    * ``smooth_gain = MI(y; bin_{2*nbins}(x))_va - MI(y; bin_{nbins}(x))_va`` -
      what simply refining the raw column's binning (a SMOOTH, location-only
      refinement, no contrast structure) adds over the same coarse baseline. This
      is the complementarity competitor: a SMOOTH signal (sin, monotone) is
      captured by finer location-binning, so ``smooth_gain`` dominates; a LOCALIZED
      step/contrast is captured by the leg's sign within a cell, so ``leg_incr``
      dominates (the leg nails a sharp discontinuity that finer uniform binning
      only resolves slowly).

    Why ``leg_incr`` (not the naive leg-MI-vs-raw-MI uplift the spline / Fourier
    path uses): a localized target ``y`` is a FUNCTION of x in a sub-window, so
    binned raw x already scores HIGH marginal MI and a single leg's marginal MI
    sits BELOW it -> uplift < 1 -> the genuine localized leg is wrongly dropped
    (the same trap the monotone hinge hit, but here for a non-monotone leg). The
    incremental MI conditions on raw x and so measures exactly the localized value
    the wavelet adds. The split is the same deterministic ``%3`` stride the
    scale-selection + the hinge / adaptive-Fourier detectors use (no RNG,
    recipe-free).

    Thin wrapper over :func:`_heldout_incremental_mi_prep` + :func:`_heldout_incremental_mi_from_prep`; a
    caller scoring MULTIPLE legs of the same source column should call those directly and cache the prep
    (see :func:`hybrid_wavelet_fe_with_recipes`) instead of recomputing it per leg via this wrapper."""
    return _heldout_incremental_mi_from_prep(_heldout_incremental_mi_prep(x, y, nbins=nbins), leg)


@njit(cache=True, parallel=True)
def _wavelet_legs_mi_batch_njit(z, yb_tr, yb_va, tr_idx, va_idx, n_y_classes, js, ks):
    """``prange``-parallel batch of the per-leg train/held-out ``_binned_mi`` scoring
    :func:`_select_wavelet_legs`'s CPU fallback ran serially, one (j, k) leg at a time.

    2M-row cProfile on combo ``c0079_b01d8c82``: ``_select_wavelet_legs`` was 380.1s cumtime (28% of the
    whole suite's wall), with ``_binned_mi`` itself costing 207.9s across 1200 calls -- a Python
    ``for j: for k:`` loop calling ``_binned_mi`` TWICE per leg (train + held-out), each call re-paying
    ``np.unique``/``np.searchsorted``/``np.bincount`` dispatch overhead on top of building the leg's `{-1,
    0, +1}` values via boolean masks. The leg's own values are ALWAYS exactly 3 classes (see
    ``_dyadic_haar_leg``'s docstring) and the target codes (``yb_tr``/``yb_va``) are already precomputed
    ONCE per source column (fit-constant across every leg) -- so a leg's MI can be computed inline, with
    the leg's ternary code (0=leg==-1, 1=leg==0, 2=leg==+1, matching ``np.unique([-1,0,1])``-then-
    ``searchsorted`` exactly) built directly from ``z`` instead of materialising the ``{-1,0,+1}`` array
    first. Fixed-size (3, n_y_classes) joint histograms per leg (rather than ``_binned_mi``'s dynamically-
    sized ``np.unique``-derived alphabet) give the IDENTICAL MI value: a leg class or y class absent from
    a given (leg, split) pair contributes a literal zero count either way, and the plug-in MI sum already
    skips zero-probability cells (``if pa <= 0: continue``) -- so the fixed alphabet is mathematically
    equivalent, not an approximation. ``tr_idx``/``va_idx`` are the row positions of the deterministic
    ``%3`` train/held-out split (``np.flatnonzero(tr)``/``np.flatnonzero(va)``), aligned with
    ``yb_tr``/``yb_va``. Bit-identical to the original per-leg ``_binned_mi`` calls (~1e-15 FP-reduction-
    order only); wired at the sole CPU fallback call site, byte-identical candidate-list order preserved
    (``js``/``ks`` built in the same ``for j: for k:`` enumeration order as before)."""
    n_legs = js.shape[0]
    n_tr = tr_idx.shape[0]
    n_va = va_idx.shape[0]
    mi_tr = np.zeros(n_legs, dtype=np.float64)
    mi_va = np.zeros(n_legs, dtype=np.float64)
    for li in prange(n_legs):
        j = js[li]
        k = ks[li]
        width = 1.0 / (2**j)
        left = k * width
        mid = left + width / 2.0
        right = left + width

        joint_tr = np.zeros((3, n_y_classes), dtype=np.int64)
        ca_tr = np.zeros(3, dtype=np.int64)
        for ii in range(n_tr):
            i = tr_idx[ii]
            zi = z[i]
            if zi >= left and zi < mid:
                code = 2
            elif zi >= mid and zi < right:
                code = 0
            else:
                code = 1
            yc = yb_tr[ii]
            joint_tr[code, yc] += 1
            ca_tr[code] += 1
        cb_tr = np.zeros(n_y_classes, dtype=np.int64)
        for a in range(3):
            for b in range(n_y_classes):
                cb_tr[b] += joint_tr[a, b]
        mi = 0.0
        nf = float(n_tr)
        for a in range(3):
            pa = ca_tr[a] / nf
            if pa <= 0.0:
                continue
            for b in range(n_y_classes):
                cab = joint_tr[a, b]
                if cab > 0:
                    pab = cab / nf
                    pb = cb_tr[b] / nf
                    mi += pab * np.log(pab / (pa * pb))
        mi_tr[li] = mi if mi > 0.0 else 0.0

        joint_va = np.zeros((3, n_y_classes), dtype=np.int64)
        ca_va = np.zeros(3, dtype=np.int64)
        for ii in range(n_va):
            i = va_idx[ii]
            zi = z[i]
            if zi >= left and zi < mid:
                code = 2
            elif zi >= mid and zi < right:
                code = 0
            else:
                code = 1
            yc = yb_va[ii]
            joint_va[code, yc] += 1
            ca_va[code] += 1
        cb_va = np.zeros(n_y_classes, dtype=np.int64)
        for a in range(3):
            for b in range(n_y_classes):
                cb_va[b] += joint_va[a, b]
        mi = 0.0
        nf = float(n_va)
        for a in range(3):
            pa = ca_va[a] / nf
            if pa <= 0.0:
                continue
            for b in range(n_y_classes):
                cab = joint_va[a, b]
                if cab > 0:
                    pab = cab / nf
                    pb = cb_va[b] / nf
                    mi += pab * np.log(pab / (pa * pb))
        mi_va[li] = mi if mi > 0.0 else 0.0
    return mi_tr, mi_va


def _select_wavelet_legs(
    x: np.ndarray,
    y: np.ndarray,
    lo: float,
    span: float,
    *,
    max_scale: int = _WAVELET_MAX_SCALE,
    max_legs: int = _WAVELET_MAX_LEGS,
    scale_sigma: float = _WAVELET_SCALE_SIGMA,
    return_arrays: bool = False,
) -> list:
    """Held-out scale-selection: rank the dyadic Haar legs by TRAIN-side marginal
    MI, keep only those whose HELD-OUT marginal MI clears a noise-aware MAD floor.

    The candidate explosion control. For each ``(j, k)`` with ``j <= max_scale``:
    build the leg on the WHOLE column, split rows on the deterministic ``%3``
    stride (no RNG), compute the leg's MI vs y on the TRAIN rows (the ranking
    key) and on the HELD-OUT rows (the validation). A leg is ADMITTED iff its
    held-out MI exceeds ``median + scale_sigma * 1.4826 * MAD`` of all candidate
    legs' held-out MIs AND an absolute floor ``_WAVELET_MIN_HELDOUT_MI``. The top
    ``max_legs`` admitted legs (by train MI) are returned as ``(j, k)`` tuples.

    Pure noise -> every leg's held-out MI sits in the noise band -> none clears
    -> empty list (no wavelet). A genuine localized leg is a multi-sigma outlier
    in held-out MI -> admitted. Returns ``[]`` on too-few rows / degenerate x.

    ``return_arrays`` (default False, byte-identical legacy return of ``(j, k)`` tuples): when True, each
    admitted entry is ``(j, k, leg_array)`` - ``leg_array`` is the SAME ``_dyadic_haar_leg(z, j, k)`` array
    already built here to rank the candidate, so :func:`generate_wavelet_features` can reuse it instead of
    rebuilding an identical array from scratch for every survivor. The GPU-batched delegate path (which
    returns bare ``(j, k)`` tuples only) rebuilds the array for survivors when ``return_arrays=True`` - still
    correct, just without the reuse (the batched twin lives in a separate module)."""
    # BATCHED born-on-device path under STRICT (default OFF -> CPU below, byte-identical). The batched twin
    # scores all candidate legs' train+held-out MI in two device workloads (one cp.bincount each) instead of
    # ~2 per-leg _binned_mi calls; parity-pinned to return the SAME admitted legs (test_wavelet_batched_mi_parity).
    if _binnedmi_gpu_enabled(n=int(np.asarray(x).size)):
        try:
            from ._wavelet_basis_fe_batched import select_wavelet_legs_batched
            _legs = select_wavelet_legs_batched(x, y, lo, span, max_scale=max_scale, max_legs=max_legs, scale_sigma=scale_sigma)
            if not return_arrays:
                return _legs
            _z = np.clip((np.asarray(x, dtype=np.float64).ravel() - lo) / span, 0.0, 1.0)
            return [(j, k, _dyadic_haar_leg(_z, j, k)) for j, k in _legs]
        except Exception as e:  # nosec B110 - optional dependency import guard
            logger.debug("GPU-resident leg-selection path failed, falling back to the host path: %s", e)
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    n = x.size
    if n != y.size or n < _WAVELET_MIN_ROWS:
        return []
    if span <= 1e-12 or float(np.std(x)) < 1e-12:
        return []
    z = np.clip((x - lo) / span, 0.0, 1.0)
    idx = np.arange(n)
    va = (idx % 3) == 0
    tr = ~va
    if int(tr.sum()) < 64 or int(va.sum()) < 32:
        return []
    # The y-subsets y[tr]/y[va] are invariant across all legs in this call; bin
    # them ONCE here and thread the codes into _binned_mi (byte-identical to the
    # per-leg inline binning) instead of re-binning per leg.
    yb_tr = _bin_y_codes(y[tr])
    yb_va = _bin_y_codes(y[va])
    cand: list[tuple] = []  # (train_mi, heldout_mi, j, k)
    leg_arrays: dict = {}
    js_list: list = []
    ks_list: list = []
    for j in range(int(max_scale) + 1):
        for k in range(2**j):
            leg = _dyadic_haar_leg(z, j, k)
            nz_left = int(np.count_nonzero(leg > 0))
            nz_right = int(np.count_nonzero(leg < 0))
            # Require enough rows in each non-zero half-cell for a trustworthy MI.
            if nz_left < _WAVELET_MIN_HALF_ROWS or nz_right < _WAVELET_MIN_HALF_ROWS:
                continue
            js_list.append(j)
            ks_list.append(k)
            if return_arrays:
                leg_arrays[(j, k)] = leg
    if not js_list:
        return []
    # Fused parallel-njit batch (see _wavelet_legs_mi_batch_njit's docstring): scores every surviving
    # leg's train+held-out MI in ONE prange dispatch instead of 2 _binned_mi calls per leg. Preserves
    # the exact (j, k) enumeration order above, so `cand`'s tie-breaking order is unchanged.
    js_arr = np.asarray(js_list, dtype=np.int64)
    ks_arr = np.asarray(ks_list, dtype=np.int64)
    tr_idx = np.flatnonzero(tr).astype(np.int64)
    va_idx = np.flatnonzero(va).astype(np.int64)
    n_y_classes = int(max(int(yb_tr.max()), int(yb_va.max()))) + 1
    mi_tr_arr, mi_va_arr = _wavelet_legs_mi_batch_njit(
        np.ascontiguousarray(z, dtype=np.float64),
        np.ascontiguousarray(yb_tr, dtype=np.int64),
        np.ascontiguousarray(yb_va, dtype=np.int64),
        tr_idx, va_idx, n_y_classes, js_arr, ks_arr,
    )
    cand = [(float(mi_tr_arr[i]), float(mi_va_arr[i]), int(js_arr[i]), int(ks_arr[i])) for i in range(js_arr.size)]
    if not cand:
        return []
    heldout = np.array([c[1] for c in cand], dtype=np.float64)
    if heldout.size >= 4:
        med = float(np.median(heldout))
        mad = float(np.median(np.abs(heldout - med)))
        floor = med + scale_sigma * 1.4826 * mad
    else:
        # Too few candidates for a robust MAD; fall back to the absolute floor.
        floor = 0.0
    floor = max(floor, _WAVELET_MIN_HELDOUT_MI)
    admitted = [c for c in cand if c[1] >= floor]
    if not admitted:
        return []
    # Rank survivors by TRAIN MI (the held-out floor already validated them).
    admitted.sort(key=lambda c: c[0], reverse=True)
    top = admitted[: int(max_legs)]
    if return_arrays:
        return [(int(c[2]), int(c[3]), leg_arrays[(int(c[2]), int(c[3]))]) for c in top]
    return [(int(c[2]), int(c[3])) for c in top]
