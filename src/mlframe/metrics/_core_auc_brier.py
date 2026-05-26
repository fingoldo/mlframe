"""AUC + Brier score kernels for ``mlframe.metrics.core``.

Carved from ``core.py``. Public symbols are re-exported from the parent.
"""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd
import polars as pl

from ._numba_params import NUMBA_NJIT_PARAMS, _PARALLEL_REDUCTION_THRESHOLD


def fast_roc_auc_unstable(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC variant using unstable (quicksort) argsort -- 2-3x faster.

    Use ONLY for callers where tie-breaking determinism on tied scores is
    immaterial: bootstrap resampling (the resample randomness already
    dominates any tie-order effect), ad-hoc per-fold metric reports
    inside CV searches, and any monte-carlo loop where the consumer
    cares about the distribution of AUC, not the byte-identical scalar.

    Stable sort is needed when two runs must produce the same AUC byte-
    identically on data with tied scores. For float64 predictions from
    real models, exact ties are rare (~0% on continuous probabilities)
    and the metric difference vs the stable variant is <1e-12 in
    practice. Where ties are common (binned / dummy classifier output),
    use ``fast_roc_auc`` instead.

    bench-validated 2026-05-27 iter336 (c0083 honest_diagnostics
    bootstrap path)::

        n=20k    stable=1.85 ms  unstable=0.67 ms   2.75x
        n=200k   stable=25.7 ms  unstable=11.4 ms   2.25x

    On c0091 / c0083 binary classification combos the bootstrap block
    runs ~6000 _auc calls per process; the swap saves ~3-4 s per
    process on n=20k val splits, ~50 s on n=200k test splits.
    """
    if isinstance(y_true, (pd.Series, pl.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_score, (pd.Series, pl.Series)):
        y_score = y_score.to_numpy()
    if y_score.ndim == 2:
        y_score = y_score[:, -1]
    # No ``kind=stable``: numpy default quicksort is 2-3x faster and
    # numerically identical when scores have no exact ties (the dominant
    # case for float64 model outputs).
    desc_score_indices = np.argsort(y_score)[::-1]
    return fast_numba_auc_nonw(y_true=y_true, y_score=y_score, desc_score_indices=desc_score_indices)


def fast_roc_auc(y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:
    """Compute ROC AUC efficiently using numba.

    Note: np.argsort needs to stay out of njitted func.

    bench-attempt-rejected (2026-05-26, c0091 iter316): tried folding the
    ``np.argsort(kind="stable")`` into the numba kernel via
    ``np.argsort(kind="mergesort")`` (numba's only stable sort). Bench
    ``profiling/bench_fast_roc_auc_argsort_inside.py``::

        n=2000   current=0.13 ms  proposed=0.12 ms  speedup=1.05x
        n=20000  current=1.79 ms  proposed=1.80 ms  speedup=1.00x
        n=200000 current=26.3 ms  proposed=29.5 ms  speedup=0.89x
        n=1M     current=156 ms   proposed=190 ms   speedup=0.82x

    Numpy's stable sort C implementation is 11-22pct faster than numba's
    mergesort on n>=200k, where the bootstrap loop spends most of its
    time. Per-call Python ``_wrapfunc`` overhead exists but is dwarfed
    by the sort itself, so removing it does not move the needle. Numpy
    argsort stays outside.

    See ``fast_roc_auc_unstable`` for the 2-3x faster variant that drops
    the stable-sort guarantee -- safe for bootstrap / monte-carlo
    callers where tie-breaking determinism is immaterial.
    """
    # **kwargs absorbs sklearn's unexpected params. Explicitly reject sample_weight rather than silently ignoring it.
    if "sample_weight" in kwargs and kwargs["sample_weight"] is not None:
        raise NotImplementedError(
            "fast_roc_auc does not support sample_weight; use sklearn.metrics.roc_auc_score"
        )

    if isinstance(y_true, (pd.Series, pl.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_score, (pd.Series, pl.Series)):
        y_score = y_score.to_numpy()
    if y_score.ndim == 2:
        y_score = y_score[:, -1]
    desc_score_indices = np.argsort(y_score, kind="stable")[::-1]  # Wave 57: stable sort for reproducibility on tied scores
    return fast_numba_auc_nonw(y_true=y_true, y_score=y_score, desc_score_indices=desc_score_indices)


@numba.njit(**NUMBA_NJIT_PARAMS)
def fast_numba_auc_nonw(y_true: np.ndarray, y_score: np.ndarray, desc_score_indices: np.ndarray) -> float:
    """code taken from fastauc lib."""
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    last_counted_fps = 0
    last_counted_tps = 0
    tps, fps = 0, 0
    auc = 0

    l = len(y_true) - 1
    for i in range(l + 1):
        tps += y_true[i]
        fps += 1 - y_true[i]
        if i == l or y_score[i + 1] != y_score[i]:
            auc += (fps - last_counted_fps) * (last_counted_tps + tps)
            last_counted_fps = fps
            last_counted_tps = tps
    tmp = tps * fps * 2
    if tmp > 0:
        return auc / tmp
    else:
        # Single-class data: ROC AUC is undefined
        return np.nan


def fast_aucs(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Compute both ROC AUC and PR AUC efficiently."""
    if isinstance(y_true, (pd.Series, pl.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_score, (pd.Series, pl.Series)):
        y_score = y_score.to_numpy()
    if y_score.ndim == 2:
        y_score = y_score[:, -1]
    desc_score_indices = np.argsort(y_score, kind="stable")[::-1]  # Wave 57: stable sort for reproducibility on tied scores
    return fast_numba_aucs(y_true=y_true, y_score=y_score, desc_score_indices=desc_score_indices)


@numba.njit(**NUMBA_NJIT_PARAMS)
def fast_numba_aucs(y_true: np.ndarray, y_score: np.ndarray, desc_score_indices: np.ndarray) -> tuple[float, float]:
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    total_pos = np.sum(y_true_sorted)
    total_neg = len(y_true_sorted) - total_pos
    if total_pos == 0 or total_neg == 0:
        # Single-class data: both ROC AUC and PR AUC are undefined
        return np.nan, np.nan

    # Variables for ROC AUC
    last_counted_fps = 0
    last_counted_tps = 0
    tps, fps = 0, 0
    roc_auc = 0.0

    # Variables for PR AUC. sklearn.average_precision_score computes
    #   AP = sum_n (R_n - R_{n-1}) * P_n
    # starting from R_0 = 0 (implicit anchor). The previous implementation already matches
    # this; we explicitly document the starting (R=0) anchor here. No behavioral change
    # needed - parity test below verifies |our - sklearn| < 1e-8.
    prev_recall = 0.0
    pr_auc = 0.0

    n = len(y_true_sorted)
    for i in range(n):
        tps += y_true_sorted[i]
        fps += 1 - y_true_sorted[i]

        if i == n - 1 or y_score_sorted[i + 1] != y_score_sorted[i]:
            # Update ROC AUC
            delta_fps = fps - last_counted_fps
            sum_tps = last_counted_tps + tps
            roc_auc += delta_fps * sum_tps
            last_counted_fps = fps
            last_counted_tps = tps

            # sklearn AP: sum over thresholds of (R_n - R_{n-1}) * P_n (current precision).
            current_precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
            current_recall = tps / total_pos
            delta_recall = current_recall - prev_recall
            pr_auc += delta_recall * current_precision  # Riemann sum
            prev_recall = current_recall

    # Normalize ROC AUC
    denom_roc = tps * fps * 2
    if denom_roc > 0:
        roc_auc /= denom_roc
    else:
        # Should not reach here due to early return, but handle defensively
        roc_auc = np.nan

    return roc_auc, pr_auc


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_brier_score_loss_seq(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return np.mean((y_true - y_prob) ** 2)


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_brier_score_loss_par(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Parallel variant. ~7.7x faster than seq at N=10M (verified
    on 8-thread numba runtime). Loses to seq below N~50k due to
    thread-spawn overhead -- the public ``fast_brier_score_loss``
    wrapper auto-dispatches based on N."""
    n = len(y_true)
    s = 0.0
    for i in numba.prange(n):
        d = y_true[i] - y_prob[i]
        s += d * d
    return s / n


# Crossover threshold for parallel kernels. See ``_numba_params.py`` (SSOT).
# Sum-reduction kernels (brier, log loss, prf1 counts) parallel-win from
# N~50-100k upwards. Multilabel row-loop kernels (subset accuracy, jaccard)
# win from N~10-50k. Conservative thresholds chosen to avoid the lose-band
# at low N.


def fast_brier_score_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (mean squared error of probabilities), auto seq/par.

    Sequential numba kernel below ~100k rows (cold-start cost
    of the parallel runtime exceeds the per-element gain). Parallel
    kernel above the threshold -- 7.7x faster at N=10M on an 8-thread
    runtime. Tunable via ``_PARALLEL_REDUCTION_THRESHOLD``.
    """
    if len(y_true) >= _PARALLEL_REDUCTION_THRESHOLD:
        return _fast_brier_score_loss_par(y_true, y_prob)
    return _fast_brier_score_loss_seq(y_true, y_prob)


# Backward-compat alias - older code and tests import `brier_score_loss` from this module.
# Keep the name visible but route it to the renamed fast_brier_score_loss so the intent is clear.
brier_score_loss = fast_brier_score_loss


def brier_and_precision_score(
    y_true,
    y_proba,
    precision_threshold: float = 0.5,
    brier_threshold: float = 0.25,
) -> float:
    """precision_score - fast_brier_score_loss when both thresholds pass, else 0.

    Brier must be <= brier_threshold and precision must be >= precision_threshold
    (at a 0.5 decision boundary) for a non-zero result. Useful as a conservative
    optimisation objective that rewards only models that are simultaneously
    calibrated AND precise at the top.
    """
    from sklearn.metrics import precision_score

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    brier = fast_brier_score_loss(y_true=y_true, y_prob=y_proba)
    if brier > brier_threshold:
        return 0.0
    y_pred = (y_proba > 0.5).astype(int)
    try:
        precision = precision_score(y_true, y_pred, zero_division=0)
    except Exception:
        return 0.0
    if precision < precision_threshold:
        return 0.0
    return float(precision - brier)


def make_brier_precision_scorer(precision_threshold: float = 0.5, brier_threshold: float = 0.25):
    """Return an sklearn scorer wrapping brier_and_precision_score (needs_proba=True)."""
    from sklearn.metrics import make_scorer

    # New sklearn (>=1.4) replaces `needs_proba` with `response_method`; fall back
    # to the legacy kwarg for older versions.
    try:
        return make_scorer(
            brier_and_precision_score,
            response_method="predict_proba",
            greater_is_better=True,
            precision_threshold=precision_threshold,
            brier_threshold=brier_threshold,
        )
    except TypeError:
        return make_scorer(
            brier_and_precision_score,
            needs_proba=True,
            greater_is_better=True,
            precision_threshold=precision_threshold,
            brier_threshold=brier_threshold,
        )
