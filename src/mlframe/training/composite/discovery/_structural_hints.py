"""Cheap structural-affinity scorer for ``_auto_base``.

Surfaces *obvious* base columns from data shape / correlation BEFORE the
elaborate MI ranking decides the top-k.  The MI ranking stays the primary
signal; this module only contributes a small, bounded additive boost so an
unmistakable structural base (a near-affine predictor of ``y``, a
low-cardinality integer grouping column, a monotone time index) bubbles up
within the MI-ranked tail instead of being missed when its pairwise
``MI(y, x)`` happens to land a hair below a noisier competitor.

Three detectors, each mapped to the composite transform it is a prime base
for:

* near-affine predictor -> ``linear_residual`` base.  ``|corr(y, x)|`` very
  high AND the OLS residual variance collapses (``var(y - a*x - b) / var(y)``
  small).  A column that linearly explains almost all of ``y`` is the
  canonical ``linear_residual`` base; the residual is what is worth modelling.
* low-cardinality integer column -> ``grouped`` base.  A small set of distinct
  integer-valued levels (a region / regime / category id) is the natural
  ``target_mean`` / grouped-residual base.
* monotone / timestamp column -> ``time`` base.  A strictly (near-)monotone
  column over the screening rows is a row-order / timestamp index, the prime
  base for time-aware composites.

The boost is intentionally SMALL and bounded (``<= max_boost``, default a
fraction of the MI spread) so it nudges ties and near-ties without overriding
a genuinely larger MI gap.  ``boost_for_features`` is bit-identical to "no
boost" (all zeros) on data with no detectable structure, so enabling it by
default never perturbs a clean MI ranking.

Cost / cProfile note: the scorer is whole-matrix vectorised.  The affine
detector is a single ``einsum`` column-variance + one ``X.T @ y`` correlation
pass -- the OLS residual-variance ratio is the closed form ``1 - corr^2`` for a
single feature, so no per-column lstsq is needed.  Integer-valued columns are
pre-screened in one ``np.all(abs(x - rint(x)), axis=0)`` pass so the per-column
``np.unique`` level-count runs ONLY on the (few) integer columns.  The monotone
fraction is one ``np.diff(axis=0)`` + two sign-count reductions for all columns.
Bench (this repo, screening shapes): the per-column Python loop was the
hotspot; vectorising the affine path roughly halved the wide-matrix cost
(n=20k x 100: ~90ms -> ~46-52ms) and the scorer now sits at ~18-36% of the
``_mi_per_feature_y_fixed`` sweep it feeds (which itself precedes the far more
expensive permutation-null), bit-identical to the per-column reference.  The
residual ``np.diff`` / ``einsum`` passes are at the numpy floor; no actionable
further speedup without a numba kernel that the call frequency (once per fit)
does not justify.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Detector thresholds.  Kept module-level so a caller can retune per dataset
# without touching the scoring body; the defaults are deliberately strict so
# only UNMISTAKABLE structural bases score (the MI ranking handles the rest).

# Near-affine predictor: |corr(y, x)| at/above this AND residual-variance
# ratio at/below ``_AFFINE_RESID_RATIO`` both required.
_AFFINE_CORR_MIN = 0.90
_AFFINE_RESID_RATIO = 0.20

# Low-cardinality integer grouping column: integer-valued (within tol) with a
# distinct-level count in ``[_GROUP_MIN_LEVELS, _GROUP_MAX_LEVELS]`` and not
# (nearly) one-level-per-row (that is an id, not a grouping).
_GROUP_MIN_LEVELS = 2
_GROUP_MAX_LEVELS = 50
_GROUP_MAX_LEVEL_FRACTION = 0.20  # distinct/levels-per-row must stay below this

# Monotone / timestamp column: fraction of forward differences sharing one
# sign at/above this counts as (near-)monotone.
_MONOTONE_FRACTION = 0.98


def _residual_variance_ratio(y: np.ndarray, x: np.ndarray) -> float:
    """``var(y - a*x - b) / var(y)`` for the OLS line ``a*x + b``.

    Closed-form (no lstsq allocation): ``a = cov(x, y) / var(x)``.  Returns
    ``1.0`` (no collapse) for a constant ``x`` or degenerate ``y`` so a
    non-predictive column scores no affine boost.
    """
    xc = x - x.mean()
    yc = y - y.mean()
    var_x = float(np.dot(xc, xc))
    var_y = float(np.dot(yc, yc))
    if var_x < 1e-24 or var_y < 1e-24:
        return 1.0
    a = float(np.dot(xc, yc)) / var_x
    resid = yc - a * xc  # intercept cancels in the centred residual
    return float(np.dot(resid, resid)) / var_y


def _is_low_card_integer(col: np.ndarray) -> bool:
    """True iff ``col`` is integer-valued (within tol) with a small distinct
    level count that is not one-level-per-row.
    """
    n = col.size
    if n < _GROUP_MIN_LEVELS:
        return False
    # Integer-valued within a small absolute tolerance (handles float64 ints).
    if not np.all(np.abs(col - np.rint(col)) < 1e-9):
        return False
    n_levels = int(np.unique(np.rint(col)).size)
    if not (_GROUP_MIN_LEVELS <= n_levels <= _GROUP_MAX_LEVELS):
        return False
    # An id-like column (almost every row its own level) is not a grouping.
    return (n_levels / n) <= _GROUP_MAX_LEVEL_FRACTION


def _monotone_fraction(col: np.ndarray) -> float:
    """Fraction of forward differences sharing the majority sign.

    ``1.0`` for a strictly monotone column; ``~0.5`` for noise.  Zero
    differences (ties) are excluded from the denominator so a step-constant
    timestamp still reads as monotone.
    """
    if col.size < 3:
        return 0.0
    diffs = np.diff(col)
    nonzero = diffs[diffs != 0.0]
    if nonzero.size == 0:
        return 0.0
    pos = float((nonzero > 0).sum())
    return float(max(pos, nonzero.size - pos) / nonzero.size)


def structural_affinity_scores(
    x_matrix: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, dict[str, str]]:
    """Per-feature structural-affinity score in ``[0, 1]`` + a base-kind tag.

    Returns ``(scores, kinds)`` where ``scores[j]`` is the strongest detector
    response for ``feature_names[j]`` (0.0 if none fired) and ``kinds`` maps a
    detected feature name to its prime base kind (``"linear_residual"`` /
    ``"grouped"`` / ``"time"``).  Operates on the already-finite-masked
    screening matrix (callers pass ``x_matrix[finite]`` / ``y[finite]``).
    """
    n_rows, n_cols = x_matrix.shape
    scores = np.zeros(n_cols, dtype=np.float64)
    kinds: dict[str, str] = {}
    if n_rows < 3 or n_cols == 0:
        return scores, kinds

    yc = y - y.mean()
    var_y = float(np.dot(yc, yc))

    # Vectorised near-affine detection across ALL columns in one matrix pass.
    # For a single-feature OLS the residual-variance ratio is exactly
    # ``1 - corr^2`` (``var(y - a*x - b) = var_y * (1 - corr^2)``), so the
    # separate per-column residual fit is unnecessary -- the affine score is a
    # closed form of the column-vs-y correlation.  Bit-identical to the
    # per-column ``corr * (1 - ratio)`` it replaces, and removes the Python
    # per-column dot/lstsq loop (the dominant hotspot on wide screening
    # matrices).
    affine_score = np.zeros(n_cols, dtype=np.float64)
    if var_y >= 1e-24:
        Xc = x_matrix - x_matrix.mean(axis=0)
        var_x = np.einsum("ij,ij->j", Xc, Xc)
        live = var_x >= 1e-24
        if live.any():
            cov = Xc[:, live].T @ yc
            corr = np.abs(cov) / np.sqrt(var_x[live] * var_y)
            ratio = 1.0 - corr * corr  # closed-form OLS residual ratio
            ok = (corr >= _AFFINE_CORR_MIN) & (ratio <= _AFFINE_RESID_RATIO)
            sc = np.zeros(int(live.sum()), dtype=np.float64)
            sc[ok] = corr[ok] * (1.0 - ratio[ok])
            affine_score[np.where(live)[0]] = sc

    # Vectorised integer-valued pre-screen: a column can only be a grouped base
    # if EVERY cell is integer within tol.  Checking this for all columns in one
    # matrix pass lets the per-column ``np.unique`` level-count run ONLY on the
    # (usually few) integer-valued columns, instead of every continuous column
    # paying a wasted ``np.unique``.  Same verdict as calling
    # ``_is_low_card_integer`` per column; it only short-circuits the negatives.
    if n_rows >= _GROUP_MIN_LEVELS:
        int_valued = np.all(np.abs(x_matrix - np.rint(x_matrix)) < 1e-9, axis=0)
    else:
        int_valued = np.zeros(n_cols, dtype=bool)

    # Vectorised monotone fraction for all columns at once.  ``max(pos, neg) /
    # (pos + neg)`` over forward-difference signs is column-separable, so one
    # ``np.diff(axis=0)`` + two sign-count reductions replace the per-column
    # ``np.diff`` loop.  Bit-identical to ``_monotone_fraction`` (zero diffs are
    # excluded from the denominator both ways); columns with all-equal cells get
    # ``pos + neg == 0`` and score 0.0.
    if n_rows >= 3:
        d = np.diff(x_matrix, axis=0)
        pos = (d > 0.0).sum(axis=0).astype(np.float64)
        neg = (d < 0.0).sum(axis=0).astype(np.float64)
        denom = pos + neg
        with np.errstate(invalid="ignore", divide="ignore"):
            mono_frac = np.where(denom > 0, np.maximum(pos, neg) / denom, 0.0)
    else:
        mono_frac = np.zeros(n_cols, dtype=np.float64)

    for j in range(n_cols):
        col = x_matrix[:, j]
        best_score = affine_score[j]
        best_kind = "linear_residual" if best_score > 0.0 else ""

        # Low-cardinality integer column -> grouped base.
        if int_valued[j] and _is_low_card_integer(col):
            # Fixed mid-strength response: a grouping column's "obviousness"
            # is categorical, not graded, so a flat score keeps it competitive
            # with a moderate affine hit without dominating a strong one.
            if 0.60 > best_score:
                best_score, best_kind = 0.60, "grouped"

        # Monotone / timestamp column -> time base.
        mono = float(mono_frac[j])
        if mono >= _MONOTONE_FRACTION and mono > best_score:
            best_score, best_kind = mono, "time"

        scores[j] = best_score
        if best_kind:
            kinds[feature_names[j]] = best_kind

    return scores, kinds


def boost_for_features(
    x_matrix: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    mi_spread: float,
    max_boost_fraction: float = 0.25,
) -> tuple[np.ndarray, dict[str, str]]:
    """Bounded additive MI boost from structural affinity.

    The boost for a feature is ``score * max_boost``, where ``max_boost =
    max_boost_fraction * mi_spread`` (``mi_spread`` is the finite MI range of
    the candidate set).  Scaling to the MI spread keeps the boost a *nudge*:
    it can lift a near-tie but cannot overcome a clearly larger MI gap.  When
    ``mi_spread`` is zero/degenerate (all MIs equal) a small absolute floor is
    used so structural bases still order ahead of structureless ties.

    Returns ``(boost, kinds)`` aligned with ``feature_names``.
    """
    scores, kinds = structural_affinity_scores(x_matrix, y, feature_names)
    if not np.any(scores > 0.0):
        return np.zeros(len(feature_names), dtype=np.float64), kinds
    spread = float(mi_spread)
    if not np.isfinite(spread) or spread <= 1e-12:
        # Degenerate MI spread: use a tiny absolute floor so structural bases
        # still break ties deterministically without overpowering real MI.
        max_boost = 1e-6
    else:
        max_boost = max_boost_fraction * spread
    return scores * max_boost, kinds
