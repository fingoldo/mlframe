"""Jackknife estimators + the BCa/percentile CI reducer for ``bootstrap.py``.

Split out from ``bootstrap.py`` to keep that file below the 1k-line monolith threshold (the five
jackknife/CI helpers below run to ~240 lines together, including their algorithmic-derivation
docstrings). Behaviour preserved bit-for-bit; every name is re-exported from ``bootstrap`` so existing
imports (``from mlframe.evaluation.bootstrap import ...``) continue to work.

Contents: ``_ci_from_samples`` (percentile / BCa reduction of a bootstrap distribution) and the four
leave-one-out jackknife estimators that feed its BCa acceleration term -- the generic O(n * max_n) gather
form (``_jackknife_metric`` / ``_jackknife_metric_idx``) plus the O(n) / O(n log n) closed-form fast paths
for mean-decomposable metrics (``_jackknife_mean_metric``) and ROC-AUC (``_jackknife_auc``).
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# iter417: bind math.isfinite for the per-iter scalar check (see bootstrap.py's own copy of this
# rationale). np.isfinite on a single Python float pays the full numpy dispatcher (~1.0us / call); the
# C-implemented math.isfinite is 7.5x faster (0.13us / call).
import math

_isfinite = math.isfinite


def _ci_from_samples(
    samples: np.ndarray,
    point: float,
    alpha: float,
    method: str,
    jackknife: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Reduce a bootstrap distribution to a (lo, hi) CI by percentile or BCa.

    ``method="percentile"`` is the plain Efron percentile interval (symmetric, fast). ``method="bca"`` is the
    bias-corrected and accelerated interval (Efron 1987): it shifts the percentile cut-points to correct for
    (a) median bias of the bootstrap distribution relative to the point estimate (``z0``) and (b) skew of the
    sampling distribution (``acceleration`` from the jackknife). On skewed / bounded metrics (AUC near 1.0,
    Pearson r) percentile silently UNDER-COVERS; BCa recovers close-to-nominal coverage. When the BCa inputs
    are degenerate (no jackknife, all-equal samples, non-finite z0/a) BCa gracefully falls back to percentile.
    """
    lo_pct = (alpha / 2.0) * 100.0
    hi_pct = (1.0 - alpha / 2.0) * 100.0

    def _pct_pair(p_lo: float, p_hi: float) -> tuple[float, float]:
        """Fetch both percentile cut-points from ``samples`` in a single ``np.percentile`` call (see rationale below)."""
        # CPX24: one np.percentile call over the [lo, hi] vector instead of two
        # separate calls. Each np.percentile internally np.partition's the array;
        # two calls partition the SAME samples twice. The vectorised single call
        # partitions once and is BIT-IDENTICAL (==) to the two scalar calls
        # (verified across n=1k-10k). Pure post-processing of the CI cut-points,
        # no RNG touched -- the resample draws are already complete here.
        both = np.percentile(samples, [p_lo, p_hi])
        return float(both[0]), float(both[1])

    if method != "bca":
        return _pct_pair(lo_pct, hi_pct)

    n_s = samples.shape[0]
    # z0: bias correction = inverse-normal of the fraction of resamples below the point estimate.
    n_below = int(np.count_nonzero(samples < point))
    if n_below == 0 or n_below == n_s:
        return _pct_pair(lo_pct, hi_pct)
    z0 = stats.norm.ppf(n_below / n_s)

    # a: acceleration from the jackknife skew of the metric (Efron 1987 eq 6.6). No jackknife -> percentile.
    if jackknife is None or jackknife.shape[0] < 3:
        return _pct_pair(lo_pct, hi_pct)
    jk_mean = jackknife.mean()
    diffs = jk_mean - jackknife
    denom = 6.0 * (np.sum(diffs**2) ** 1.5)
    if denom == 0.0 or not np.isfinite(denom):
        return _pct_pair(lo_pct, hi_pct)
    a = float(np.sum(diffs**3) / denom)

    z_lo = stats.norm.ppf(alpha / 2.0)
    z_hi = stats.norm.ppf(1.0 - alpha / 2.0)
    a_lo = stats.norm.cdf(z0 + (z0 + z_lo) / (1.0 - a * (z0 + z_lo)))
    a_hi = stats.norm.cdf(z0 + (z0 + z_hi) / (1.0 - a * (z0 + z_hi)))
    if not (np.isfinite(a_lo) and np.isfinite(a_hi)) or a_lo >= a_hi:
        return _pct_pair(lo_pct, hi_pct)
    return _pct_pair(a_lo * 100.0, a_hi * 100.0)


def _jackknife_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    max_n: int = 2000,
) -> Optional[np.ndarray]:
    """Leave-one-out jackknife of ``metric_fn`` for the BCa acceleration term.

    Returns the ``(n,)`` leave-one-out metric values, or ``None`` if the jackknife is infeasible. For n > ``max_n``
    the full LOO is O(n^2) in metric calls, so we sub-sample to a deterministic stride of ``max_n`` rows -- the
    acceleration estimate is a low-order skew summary and tolerates sub-sampling far better than the percentile
    cut-points themselves. Failed / non-finite LOO evaluations are dropped.
    """
    n = y_true.shape[0]
    if n < 3:
        return None
    if n <= max_n:
        sel = np.arange(n)
    else:
        sel = np.linspace(0, n - 1, max_n).astype(np.int64)
    keep_mask = np.ones(n, dtype=bool)
    out = np.empty(sel.shape[0], dtype=np.float64)
    w = 0
    for i in sel:
        keep_mask[i] = False
        try:
            v = float(metric_fn(y_true[keep_mask], y_pred[keep_mask]))
        except Exception as exc:
            logger.debug("jackknife LOO metric failed at i=%d: %r", i, exc, exc_info=True)
            keep_mask[i] = True
            continue
        keep_mask[i] = True
        if not _isfinite(v):
            continue
        out[w] = v
        w += 1
    if w < 3:
        return None
    return out[:w]


def _jackknife_metric_idx(
    n: int,
    metric_fn_idx: Callable[[np.ndarray], float],
    max_n: int = 2000,
) -> Optional[np.ndarray]:
    """Leave-one-out jackknife for an INDEX-aware metric ``fn(idx) -> float`` (BCa acceleration term).

    Mirrors ``_jackknife_metric`` but feeds the metric the leave-one-out index array instead of pre-sliced views,
    so an index-aware metric (e.g. the pre-sorted AUC resampler) reuses its precomputed base structure.
    """
    if n < 3:
        return None
    sel = np.arange(n) if n <= max_n else np.linspace(0, n - 1, max_n).astype(np.int64)
    full = np.arange(n, dtype=np.int64)
    # CPX24: boolean mask-flip gather instead of per-iter np.delete(full, i).
    # np.delete allocates a fresh length-(n-1) array AND pays searchsorted /
    # range-rebuild dispatch every iteration; full[mask] is a single fancy-index
    # gather with the mask flipped in/out per iter (no per-call delete dispatch).
    # Bit-identical: full[mask] yields the same ascending indices delete did
    # (mirrors the already-mask-based _jackknife_metric above). No RNG here.
    keep_mask = np.ones(n, dtype=bool)
    out = np.empty(sel.shape[0], dtype=np.float64)
    w = 0
    for i in sel:
        keep_mask[i] = False
        try:
            v = float(metric_fn_idx(full[keep_mask]))
        except Exception as exc:
            logger.debug("jackknife-idx LOO metric failed at i=%d: %r", i, exc, exc_info=True)
            keep_mask[i] = True
            continue
        keep_mask[i] = True
        if not _isfinite(v):
            continue
        out[w] = v
        w += 1
    if w < 3:
        return None
    return out[:w]


def _jackknife_mean_metric(
    y_true: np.ndarray,
    per_row: np.ndarray,
    *,
    requires_both_classes: bool,
    reduce_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    max_n: int = 2000,
) -> Optional[np.ndarray]:
    """O(n) exact leave-one-out jackknife for a metric of the form ``metric == reduce_fn(mean(per_row))``.

    The generic ``_jackknife_metric`` re-gathers the ``n-1`` kept rows and re-runs ``metric_fn`` for each of the
    ``max_n`` sampled leave-out points -- O(max_n * n) row copies + O(max_n) full metric evaluations, which is ~11s
    for a single log-loss / Brier jackknife at n=300k (each of 2000 LOO points copies a ~300k array). But when the
    metric is ``g`` applied to the MEAN of per-row contributions (log-loss = mean of per-row cross-entropy; Brier =
    mean of squared errors; RMSE = sqrt(mean of squared errors)), the leave-one-out value is algebraic:
    ``LOO_i = g((S - per_row_i) / (n-1))`` with ``S = sum(per_row)`` computed ONCE. That is O(n) total (517x at
    n=300k: 10.9s -> 21ms), and matches the gather path to floating-point sum-reduction order (~1e-15 on the LOO
    values, <=1e-13 on the resulting BCa CI bounds -- negligible for the acceleration skewness term).

    ``requires_both_classes`` mirrors the metric's degenerate-value contract: log-loss returns NaN (and the gather
    jackknife skips) when the leave-one-out set collapses to a single class, so those indices are skipped here too;
    Brier / RMSE are defined on any set and skip nothing. Returns ``None`` (caller falls back to the generic gather
    jackknife) when the per-row contributions are non-finite -- e.g. probabilities out of range -- so the exact
    contract of the slow path is preserved on the degenerate inputs it was written for.
    """
    per_row = np.asarray(per_row, dtype=np.float64).ravel()
    n = per_row.shape[0]
    if n < 3 or not np.all(np.isfinite(per_row)):
        return None
    sel = np.arange(n) if n <= max_n else np.linspace(0, n - 1, max_n).astype(np.int64)
    total = float(per_row.sum())
    loo = (total - per_row[sel]) / (n - 1)
    if reduce_fn is not None:
        loo = reduce_fn(loo)
    if requires_both_classes:
        yt = np.asarray(y_true).ravel()
        n_pos = int(np.count_nonzero(yt == 1))
        pos_after = n_pos - (yt[sel] == 1).astype(np.int64)
        loo = loo[(pos_after != 0) & (pos_after != n - 1)]
    loo = loo[np.isfinite(loo)]
    return loo if loo.shape[0] >= 3 else None


def _jackknife_auc(y_true: np.ndarray, scores: np.ndarray, *, max_n: int = 2000) -> Optional[np.ndarray]:
    """O(n log n) exact leave-one-out jackknife of ROC-AUC via Mann-Whitney placement values.

    ROC-AUC is NOT a mean of per-row contributions, so ``_jackknife_mean_metric`` does not apply -- but the tie-aware
    (midrank) Mann-Whitney AUC ``= (concordant + 0.5*ties) / (n_pos*n_neg)`` DOES decompose through per-observation
    PLACEMENT values, so leaving out one observation is O(1) once the placements are computed. The generic
    ``_jackknife_metric_idx`` instead recomputes an O(n log n) AUC for each of the ``max_n`` leave-out points
    (O(max_n * n log n)) -- ~34s for a single AUC jackknife at n=300k. This computes both rankings ONCE and derives
    every leave-one-out AUC algebraically: BIT-IDENTICAL to re-running ``fast_roc_auc`` on the n-1 kept rows (0.0 on
    both continuous and tied scores; both use the same midrank convention), 159x faster (33.7s -> 0.21s at n=300k).

    Placement of a positive = negatives it beats + 0.5 ties = ``rank_in_pooled - rank_within_positives``; the sum over
    positives is the total concordant mass. Leaving out positive i removes its placement from the numerator and one
    positive from the denominator; leaving out a negative removes the positives-above-it count it contributed. Skips
    leave-out points that collapse a class (AUC undefined -> the gather path returns NaN and drops them too). Returns
    ``None`` (caller falls back to the exact gather jackknife) on non-finite scores or fewer than 3 usable LOO points.
    """
    y = np.asarray(y_true).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = y.shape[0]
    if n < 3 or not np.all(np.isfinite(s)):
        return None
    pos = y == 1
    n_pos = int(np.count_nonzero(pos))
    n_neg = n - n_pos
    if n_pos < 2 or n_neg < 2:
        return None  # a single leave-out cannot keep both classes non-trivial; let the gather path handle it
    x = s[pos]
    z = s[~pos]
    ranks_pooled = stats.rankdata(np.concatenate([x, z]), method="average")
    plc_pos = ranks_pooled[:n_pos] - stats.rankdata(x, method="average")  # negatives each positive beats (+0.5 ties)
    plc_neg = ranks_pooled[n_pos:] - stats.rankdata(z, method="average")  # positives each negative beats (+0.5 ties)
    total = float(plc_pos.sum())  # total concordant (+0.5 tie) mass
    pos_of_row = np.full(n, -1, dtype=np.int64)
    pos_of_row[np.flatnonzero(pos)] = np.arange(n_pos)
    neg_of_row = np.full(n, -1, dtype=np.int64)
    neg_of_row[np.flatnonzero(~pos)] = np.arange(n_neg)
    sel = np.arange(n) if n <= max_n else np.linspace(0, n - 1, max_n).astype(np.int64)
    out = np.empty(sel.shape[0], dtype=np.float64)
    w = 0
    for i in sel:
        if pos[i]:
            if n_pos - 1 < 1 or n_neg < 1:
                continue
            out[w] = (total - plc_pos[pos_of_row[i]]) / ((n_pos - 1) * n_neg)
        else:
            if n_neg - 1 < 1 or n_pos < 1:
                continue
            # negative i contributed (n_pos - plc_neg_i) concordant units (the positives ranked ABOVE it).
            out[w] = (total - (n_pos - plc_neg[neg_of_row[i]])) / (n_pos * (n_neg - 1))
        w += 1
    loo = out[:w]
    loo = loo[np.isfinite(loo)]
    return loo if loo.shape[0] >= 3 else None
