"""Haar wavelet / localized multiresolution basis FE (backlog #13, 2026-06-09).

A NEW univariate operator for a signal shape the catalog cannot capture: a
**localized bump / multiscale piecewise structure** -- ``y`` jumps only inside a
narrow sub-interval of x, or has step/contrast structure at SEVERAL scales at
once. The closest existing operators all have the WRONG form:

* Fourier is **GLOBAL** -- a localized bump forces a long tone sum and the
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
detector at scale ``2^{-j}`` centred on a dyadic position -- a multiresolution
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
   its raw source's MI -- a second, pool-level self-limit.

Why MI-gateable (unlike the hinge)
----------------------------------
A Haar leg ``psi_{j,k}`` is NON-monotone in x (it is +1 then -1 then 0), so it is
NOT MI-invariant by the data-processing inequality -- a leg in the RIGHT window
carries genuine MARGINAL MI about a localized target (unlike a single Fourier
phase-leg, whose MI is split across sin/cos, or the monotone hinge/isotonic legs
that MI cannot see). So the wavelet routes through the NORMAL MI-uplift gate, no
deferred-materialisation / re-add dance is needed (contrast the hinge, backlog
#11, which is MI-invariant and needs the protection roster).

Leak-safe replay
----------------
The recipe (kind ``"orth_wavelet"``) stores only ``(lo, span)`` + the dyadic
``(j, k)`` -- NO y -- so ``transform`` replay is the closed-form indicator
``_dyadic_haar_leg(clip((x - lo) / span, 0, 1), j, k)``. The scale SELECTION
consumes y at FIT time (like every supervised FE here -- spline knot placement,
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
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

logger = logging.getLogger(__name__)

__all__ = [
    "generate_wavelet_features",
    "hybrid_wavelet_fe_with_recipes",
    "build_orth_wavelet_recipe",
    "_apply_orth_wavelet",
    "_dyadic_haar_leg",
    "_select_wavelet_legs",
]


# Coarsest..finest dyadic scales. j=0 is the root contrast (left vs right half of
# the whole support, = psi_{0,0}); j=3 resolves features at 1/8 of the support.
# Beyond j=3 a leg spans < ~1/16 of the range and on n<=4000 its half-cells hold
# too few rows for a trustworthy held-out MI -- so cap at 3 (the backlog's
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


def _dyadic_haar_leg(z: np.ndarray, j: int, k: int, dtype=np.float32) -> np.ndarray:
    """Closed-form Haar wavelet indicator ``psi_{j,k}(z)`` for ``z`` in [0, 1].

    ``+1`` on the LEFT half ``[k/2^j, (k+0.5)/2^j)``, ``-1`` on the RIGHT half
    ``[(k+0.5)/2^j, (k+1)/2^j)``, ``0`` outside. Pure function of ``z`` -- no y,
    no fitted state beyond the (j, k) integers, so it replays leak-free.

    The output is allocated in ``dtype`` (float32 by default, the large-n working
    dtype). The leg holds only the exact values {-1, 0, +1}, which are bit-exact in
    float32 -> every downstream consumer (binned MI via np.unique/searchsorted, the
    engineered column) is bit-identical to float64 while halving the (n_scales, n)
    working-array footprint (e.g. (10, 1M) = 76 MiB -> 38 MiB). The dyadic-cell
    boolean masks are computed against the float64 ``z`` axis, so the cell
    membership (and hence the leg) does not depend on the output dtype."""
    width = 1.0 / (2 ** int(j))
    left = int(k) * width
    mid = left + width / 2.0
    right = left + width
    leg = np.zeros_like(z, dtype=dtype)
    leg[(z >= left) & (z < mid)] = 1.0
    leg[(z >= mid) & (z < right)] = -1.0
    return leg


def _binned_mi(feat: np.ndarray, y: np.ndarray, nbins: int = 10) -> float:
    """Plug-in binned MI(feat; y) in nats. y is treated as discrete classes if it
    has <= 20 unique values, else quantile-binned into ``nbins``. Used only for
    the held-out scale-selection ranking (the pool-level admission reuses the
    project's ``_mi_classif_batch``)."""
    feat = np.asarray(feat, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    n = feat.size
    if n == 0 or n != y.size:
        return 0.0
    # Feature is a Haar leg taking values in {-1, 0, +1} -> use those as classes
    # directly (3 cells); avoids quantile-binning a ternary column.
    uniq_f = np.unique(feat)
    if uniq_f.size <= nbins:
        fb = np.searchsorted(uniq_f, feat)
    else:
        edges = np.quantile(feat, np.linspace(0.0, 1.0, nbins + 1)[1:-1])
        fb = np.digitize(feat, edges)
    if np.issubdtype(y.dtype, np.integer) and np.unique(y).size <= 20:
        yb = y.astype(np.int64)
    elif np.unique(y).size <= 20:
        uy = np.unique(y)
        yb = np.searchsorted(uy, y)
    else:
        edges_y = np.quantile(y, np.linspace(0.0, 1.0, nbins + 1)[1:-1])
        yb = np.digitize(y, edges_y)
    # Joint-histogram MI: the prior O(|fa|*|yb|*n) double loop recomputed an O(n) boolean
    # mask per contingency cell; a single bincount over the dense joint code yields the same
    # plug-in counts. Bit-identical by construction -- pa/pb/pab are still count/n float64 and
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


def _heldout_incremental_mi(
    x: np.ndarray, leg: np.ndarray, y: np.ndarray, *, nbins: int = 10,
) -> tuple[float, float]:
    """Held-out INCREMENTAL MI of adding ``leg`` to the binned raw column ``x``,
    scored on the ``%3`` stride slice, PLUS the gain of a SMOOTH refinement
    competitor. Returns ``(leg_incr, smooth_gain)``:

    * ``leg_incr = MI(y; [bin_{nbins}(x), leg])_va - MI(y; bin_{nbins}(x))_va`` --
      what the localized Haar leg adds ON TOP of the coarse binned raw column.
    * ``smooth_gain = MI(y; bin_{2*nbins}(x))_va - MI(y; bin_{nbins}(x))_va`` --
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
    recipe-free)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    leg = np.asarray(leg, dtype=np.float64).ravel()
    y = np.asarray(y).ravel()
    n = x.size
    if n != leg.size or n != y.size or n < _WAVELET_MIN_ROWS:
        return 0.0, 0.0
    idx = np.arange(n)
    va = (idx % 3) == 0
    if int(va.sum()) < 32:
        return 0.0, 0.0
    xc = _x_codes(x[va], nbins=nbins)
    legc = np.asarray(leg[va], dtype=np.float64)
    # 3-cell leg codes {-1,0,+1} -> {0,1,2}; joint code = xc * 3 + legcode.
    leg_code = np.searchsorted(np.array([-1.0, 0.0, 1.0]), legc)
    leg_code = np.clip(leg_code, 0, 2)
    joint = xc * 3 + leg_code
    base_mi = _binned_mi(xc.astype(np.float64), y[va], nbins=max(nbins, int(xc.max()) + 1))
    joint_mi = _binned_mi(joint.astype(np.float64), y[va], nbins=int(joint.max()) + 1)
    leg_incr = float(joint_mi - base_mi)
    # SMOOTH-refinement competitor: finer (2*nbins) location binning of raw x.
    xc_fine = _x_codes(x[va], nbins=2 * nbins)
    fine_mi = _binned_mi(
        xc_fine.astype(np.float64), y[va], nbins=max(2 * nbins, int(xc_fine.max()) + 1),
    )
    smooth_gain = float(fine_mi - base_mi)
    return leg_incr, smooth_gain


def _select_wavelet_legs(
    x: np.ndarray,
    y: np.ndarray,
    lo: float,
    span: float,
    *,
    max_scale: int = _WAVELET_MAX_SCALE,
    max_legs: int = _WAVELET_MAX_LEGS,
    scale_sigma: float = _WAVELET_SCALE_SIGMA,
) -> list[tuple[int, int]]:
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
    """
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
    cand: list[tuple] = []  # (train_mi, heldout_mi, j, k)
    for j in range(int(max_scale) + 1):
        for k in range(2 ** j):
            leg = _dyadic_haar_leg(z, j, k)
            nz_left = int(np.count_nonzero(leg > 0))
            nz_right = int(np.count_nonzero(leg < 0))
            # Require enough rows in each non-zero half-cell for a trustworthy MI.
            if nz_left < _WAVELET_MIN_HALF_ROWS or nz_right < _WAVELET_MIN_HALF_ROWS:
                continue
            mi_tr = _binned_mi(leg[tr], y[tr])
            mi_va = _binned_mi(leg[va], y[va])
            cand.append((mi_tr, mi_va, j, k))
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
    return [(int(c[2]), int(c[3])) for c in admitted[: int(max_legs)]]


def generate_wavelet_features(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    y: Optional[np.ndarray] = None,
    max_scale: int = _WAVELET_MAX_SCALE,
    max_legs: int = _WAVELET_MAX_LEGS,
    scale_sigma: float = _WAVELET_SCALE_SIGMA,
    dedup_collinear_sources: bool = True,
    dedup_corr_threshold: float = 0.999,
    feature_dtype=np.float32,
) -> tuple[pd.DataFrame, dict]:
    """For each numeric column, held-out-select a small dyadic Haar leg set and
    emit the legs, returning the columns alongside the per-column fit meta needed
    to build leak-safe recipes.

    Parameters
    ----------
    X : DataFrame
        Source frame. Only numeric columns are processed; non-numeric skipped.
    cols : sequence of column names, optional
        Columns to expand. None = all numeric columns.
    y : array-like, optional
        Target. Consulted ONLY by the held-out scale-selection (which legs carry
        held-out MI). Never read for the emitted column VALUE, so the recipe
        replay stays leakage-free / y-independent.
    max_scale : int
        Finest dyadic scale j (default 3 -> scales 0..3, <= 15 candidate legs).
    max_legs : int
        Hard cap on emitted legs per column after selection (candidate control).

    Returns
    -------
    (engineered_X, meta)
        engineered_X : DataFrame of new columns named
            ``"{col}__haar_j{j}k{k}"`` (the leg ``psi_{j,k}``).
        meta : dict mapping each emitted column name to the recipe metadata
            ``{"src": col, "j": int, "k": int, "lo": float, "span": float}``.
    """
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if y is None:
        # Scale-selection is supervised; with no y there is nothing to validate.
        return pd.DataFrame(index=X.index), {}
    if dedup_collinear_sources:
        from ._orthogonal_univariate_fe import _dedup_collinear_source_cols
        cols = _dedup_collinear_source_cols(
            X, list(cols), corr_threshold=dedup_corr_threshold,
        )
    y_arr = np.asarray(y).ravel()
    if y_arr.size != len(X):
        return pd.DataFrame(index=X.index), {}
    out_cols: dict = {}
    meta: dict = {}
    for col in cols:
        if col not in X.columns or not pd.api.types.is_numeric_dtype(X[col]):
            continue
        x = np.asarray(X[col].to_numpy(), dtype=np.float64)
        finite_mask = np.isfinite(x)
        if not finite_mask.any():
            continue
        if not finite_mask.all():
            # A wavelet basis over a NaN column is unsound: the nanmean-imputed wavelet becomes a missingness proxy that displaces the genuine
            # missingness-FE columns, and the recipe replay does not impute (transform() emits all-NaN). Skip; the missingness signal belongs to the
            # dedicated missingness-FE family.
            continue
        xf = x[np.isfinite(x)]
        lo = float(xf.min())
        hi = float(xf.max())
        span = max(hi - lo, 1e-12)
        try:
            legs = _select_wavelet_legs(
                x, y_arr, lo, span,
                max_scale=max_scale, max_legs=max_legs, scale_sigma=scale_sigma,
            )
        except Exception as exc:
            logger.warning(
                "generate_wavelet_features: scale-select on col=%r raised %r; "
                "skipping wavelet for that column.", col, exc,
            )
            continue
        z = np.clip((x - lo) / span, 0.0, 1.0)
        for (j, k) in legs:
            leg = _dyadic_haar_leg(z, j, k, dtype=feature_dtype)
            if float(np.std(leg)) <= 1e-12:
                continue
            name = f"{col}__haar_j{j}k{k}"
            out_cols[name] = leg
            meta[name] = {
                "src": col, "j": int(j), "k": int(k),
                "lo": float(lo), "span": float(span),
            }
    return pd.DataFrame(out_cols, index=X.index), meta


def build_orth_wavelet_recipe(
    *, name: str, src_name: str, j: int, k: int, lo: float, span: float,
) -> "EngineeredRecipe":
    """Frozen recipe for one Haar wavelet basis column ``psi_{j,k}(z)`` where
    ``z = clip((X[src_name] - lo) / span, 0, 1)`` with the dyadic ``(j, k)`` and
    ``(lo, span)`` fixed at fit time.

    Replay is closed-form in the source column alone -- no y reference captured,
    so ``transform`` is leakage-free by construction. Mirrors
    ``build_orth_spline_recipe`` (store basis params + ``lo``/``span``, replay a
    closed-form basis function of x)."""
    from .engineered_recipes import EngineeredRecipe
    return EngineeredRecipe(
        name=name,
        kind="orth_wavelet",
        src_names=(str(src_name),),
        extra={
            "j": int(j), "k": int(k),
            "lo": float(lo), "span": float(span),
        },
    )


def _apply_orth_wavelet(recipe, X) -> np.ndarray:
    """Replay one Haar wavelet basis column from the stored ``(j, k, lo, span)``
    -- a pure function of the source column (no y). Mirrors ``_apply_orth_spline``.
    """
    from .engineered_recipes import _extract_column
    if len(recipe.src_names) != 1:
        raise ValueError(
            f"orth_wavelet recipe '{recipe.name}' must have exactly 1 "
            f"src_names; got {len(recipe.src_names)}"
        )
    for key in ("j", "k", "lo", "span"):
        if key not in recipe.extra:
            raise KeyError(
                f"orth_wavelet recipe '{recipe.name}' missing '{key}' in extra. "
                f"Re-fit MRMR to regenerate."
            )
    name = recipe.src_names[0]
    j = int(recipe.extra["j"])
    k = int(recipe.extra["k"])
    lo = float(recipe.extra["lo"])
    span = max(float(recipe.extra["span"]), 1e-12)
    vals = np.asarray(_extract_column(X, name), dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.all():
        fill = float(np.nanmean(vals[finite])) if finite.any() else 0.0
        vals = np.where(finite, vals, fill)
    z = np.clip((vals - lo) / span, 0.0, 1.0)
    return _dyadic_haar_leg(z, j, k)


def hybrid_wavelet_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    max_scale: int = _WAVELET_MAX_SCALE,
    max_legs: int = _WAVELET_MAX_LEGS,
    scale_sigma: float = _WAVELET_SCALE_SIGMA,
    top_k: int = 8,
    min_incr_mi: float = _WAVELET_MIN_INCR_MI,
    smooth_complement_ratio: float = _WAVELET_SMOOTH_COMPLEMENT_RATIO,
    nbins: int = 10,
    feature_dtype=np.float32,
    **_legacy_ignored,
) -> tuple[pd.DataFrame, list, list, pd.DataFrame]:
    """Haar wavelet basis FE + held-out-incremental-MI selection, returning
    leak-safe recipes. Returns ``(X_augmented, appended_names, recipes, scores)``.

    Four self-limiting bounds keep the candidate count small AND make the operator
    complementary (it adds legs only where a localized contrast genuinely sharpens
    y beyond a SMOOTH refinement of raw x, and stays silent on smooth / noise):

    1. :func:`generate_wavelet_features` already held-out-scale-selected a small
       dyadic leg set per column (the noise-aware held-out MAD floor over the
       candidate legs' held-out MIs + the ``max_legs`` cap), so pure noise emits
       NO leg to score here.
    2. Each surviving leg must clear the held-out INCREMENTAL MI floor
       (``leg_incr >= min_incr_mi`` on the ``%3`` slice): the joint
       ``MI(y; [bin(x), leg])`` must beat ``MI(y; bin(x))`` by an absolute margin
       -- the leg sharpens y BEYOND what the coarse raw column already says.
    3. COMPLEMENTARITY GUARD: the leg's incremental MI must also exceed
       ``smooth_complement_ratio`` x the SMOOTH-refinement gain (what finer
       location-only binning of raw x adds over the same coarse baseline). On a
       SMOOTH (sin / monotone) column the smooth refinement dominates -> the leg is
       rejected (the global Fourier basis owns that regime, complementarity); on a
       LOCALIZED step / contrast the leg dominates -> admitted.
    4. ``top_k`` caps the survivors per fit.

    Why the incremental gate, NOT the naive leg-MI-vs-raw-MI uplift the spline /
    Fourier path uses: a localized target ``y`` is a FUNCTION of x in a sub-window,
    so binned raw x already scores HIGH marginal MI and a single leg's marginal MI
    sits BELOW it -> uplift < 1 -> the genuine localized leg is wrongly dropped
    (the same trap the monotone hinge hit, but here for a non-monotone leg). The
    incremental MI conditions on raw x and so measures exactly the localized value
    the wavelet adds. A Haar leg is NON-monotone -> it is MI-VISIBLE (unlike the
    monotone hinge / isotonic, which need a held-out LINEAR-usability gate), so the
    statistic here is MI-based, just conditioned on raw x.

    ``scores`` is the full per-leg ranking (incr_mi, smooth_gain, passed flag;
    winners + rejects).
    """
    engineered, meta = generate_wavelet_features(
        X, cols=cols, y=y,
        max_scale=max_scale, max_legs=max_legs, scale_sigma=scale_sigma,
        feature_dtype=feature_dtype,
    )
    _empty_cols = [
        "engineered_col", "source_col", "incr_mi", "smooth_gain", "passed",
    ]
    if engineered.empty:
        return X.copy(), [], [], pd.DataFrame(columns=_empty_cols)
    # y -> discrete class codes for the binned joint-MI gate; bin continuous y.
    y_arr = np.asarray(y).ravel()
    if not np.issubdtype(y_arr.dtype, np.integer) or np.unique(y_arr).size > 20:
        try:
            y_codes = pd.qcut(
                pd.Series(y_arr), q=min(nbins, max(2, np.unique(y_arr).size)),
                labels=False, duplicates="drop",
            ).to_numpy()
            y_codes = np.where(np.isfinite(y_codes), y_codes, 0).astype(np.int64)
        except Exception:
            y_codes = np.zeros(y_arr.size, dtype=np.int64)
    else:
        y_codes = y_arr.astype(np.int64)
    rows = []
    for name in engineered.columns:
        m = meta.get(name, {})
        src = str(m.get("src", name.split("__", 1)[0]))
        if src in X.columns and pd.api.types.is_numeric_dtype(X[src]):
            x_src = np.asarray(X[src].to_numpy(), dtype=np.float64)
            finite = np.isfinite(x_src)
            if not finite.all():
                x_src = np.where(
                    finite, x_src,
                    float(np.nanmean(x_src[finite])) if finite.any() else 0.0,
                )
        else:
            x_src = engineered[name].to_numpy(dtype=np.float64)
        incr, smooth_gain = _heldout_incremental_mi(
            x_src, engineered[name].to_numpy(), y_codes, nbins=nbins,
        )
        # Two-condition admission: (a) absolute incremental floor, (b) the leg
        # beats the smooth-refinement competitor (complementarity guard).
        passed = bool(
            (incr >= float(min_incr_mi))
            and (incr >= float(smooth_complement_ratio) * max(smooth_gain, 0.0))
        )
        rows.append({
            "engineered_col": name, "source_col": src,
            "incr_mi": float(incr),
            "smooth_gain": float(smooth_gain),
            "passed": passed,
        })
    scores = pd.DataFrame(rows).sort_values(
        "incr_mi", ascending=False,
    ).reset_index(drop=True)
    qualified = scores[scores["passed"]]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    recipes = []
    for name in keep:
        if name not in meta:
            continue
        m = meta[name]
        recipes.append(build_orth_wavelet_recipe(
            name=name, src_name=str(m["src"]),
            j=int(m["j"]), k=int(m["k"]),
            lo=float(m["lo"]), span=float(m["span"]),
        ))
    return X_aug, keep, recipes, scores
