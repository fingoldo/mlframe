"""Additional classification metrics (binary + multiclass).

Numba-accelerated, dispatcher-gated kernels that complement
``_core_auc_brier.py`` / ``_log_loss_and_separation.py`` /
``_core_precision_mape.py``.

Public API (re-exported from ``mlframe.metrics.core``):
  Binary:
    * ``ks_statistic``                  - Kolmogorov-Smirnov
    * ``matthews_corrcoef_binary``      - MCC
    * ``cohen_kappa_binary``            - Cohen's kappa
    * ``balanced_accuracy_binary``      - (TPR+TNR)/2
    * ``g_mean_binary``                 - sqrt(TPR*TNR)
    * ``brier_skill_score``             - BSS vs marginal baseline
    * ``gini_from_auc``                 - 2*AUC - 1
    * ``specificity_npv_fpr_fnr``       - (Specificity, NPV, FPR, FNR) tuple
    * ``f_beta_score``                  - F-beta for arbitrary beta
    * ``spiegelhalter_z``               - Spiegelhalter Z-test for calibration
    * ``lift_at_k``                     - Lift @ top-k%

  Multiclass:
    * ``top_k_accuracy``                - top-k accuracy (k=1,3,5,...)
    * ``matthews_corrcoef_multiclass``  - Gorodkin multiclass MCC
    * ``ranked_probability_score``      - RPS (ordinal CRPS-analog)
"""
from __future__ import annotations

from math import erfc, sqrt
from typing import Optional, Tuple

import numba
import numpy as np

from .._numba_params import _PARALLEL_REDUCTION_THRESHOLD, NUMBA_NJIT_PARAMS

# ---------- helpers ----------


@numba.njit(**NUMBA_NJIT_PARAMS)
def _confusion_counts_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (TP, FP, TN, FN) for binary 0/1 labels.

    Robust to any int/bool dtype: cast happens at call site. The kernel
    treats anything != 0 as positive.
    """
    tp = 0; fp = 0; tn = 0; fn = 0
    for i in range(y_true.shape[0]):
        if y_true[i] != 0:
            if y_pred[i] != 0:
                tp += 1
            else:
                fn += 1
        else:
            if y_pred[i] != 0:
                fp += 1
            else:
                tn += 1
    return tp, fp, tn, fn


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _confusion_counts_binary_par(
    y_true: np.ndarray, y_pred: np.ndarray, nthr: int,
) -> Tuple[int, int, int, int]:
    """Per-thread accumulators avoid the `+= scalar` reduction-detection
    edge case that fails on tuples (numba only auto-reduces scalars in
    prange). Final reduction outside the prange."""
    tp_a = np.zeros(nthr, dtype=np.int64)
    fp_a = np.zeros(nthr, dtype=np.int64)
    tn_a = np.zeros(nthr, dtype=np.int64)
    fn_a = np.zeros(nthr, dtype=np.int64)
    n = y_true.shape[0]
    for i in numba.prange(n):
        t = numba.get_thread_id()
        if y_true[i] != 0:
            if y_pred[i] != 0:
                tp_a[t] += 1
            else:
                fn_a[t] += 1
        else:
            if y_pred[i] != 0:
                fp_a[t] += 1
            else:
                tn_a[t] += 1
    return int(tp_a.sum()), int(fp_a.sum()), int(tn_a.sum()), int(fn_a.sum())


def _confusion_counts_binary_dispatch(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Public dispatcher: int64 cast + par/seq pick."""
    yt = np.ascontiguousarray(y_true).astype(np.int64, copy=False)
    yp = np.ascontiguousarray(y_pred).astype(np.int64, copy=False)
    if yt.shape[0] >= _PARALLEL_REDUCTION_THRESHOLD:
        result = _confusion_counts_binary_par(yt, yp, numba.get_num_threads())
    else:
        result = _confusion_counts_binary(yt, yp)
    return (int(result[0]), int(result[1]), int(result[2]), int(result[3]))


def precision_recall_f1_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Precision / Recall / F1 derived from confusion counts (no data pass).

    Bit-identical to ``compute_pr_recall_f1_metrics`` by construction (same
    closed form on the same TP/FP/FN). Lets a caller that already holds the
    binary confusion counts skip a second full-n scan over the same arrays.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def matthews_corrcoef_from_counts(tp: int, fp: int, tn: int, fn: int) -> float:
    """MCC derived from confusion counts (no data pass). Bit-identical to
    ``matthews_corrcoef_binary`` -- same closed form on the same counts."""
    num = float(tp) * tn - float(fp) * fn
    d1 = tp + fp
    d2 = tp + fn
    d3 = tn + fp
    d4 = tn + fn
    if d1 == 0 or d2 == 0 or d3 == 0 or d4 == 0:
        return 0.0
    denom = sqrt(float(d1) * d2 * d3 * d4)
    return num / denom


# ---------- KS statistic ----------

# The inline-ordered kernel (indexes through ``order``, no gather temporaries) wins in TWO regions, with a losing middle band:
#   - n < _KS_FUSED_MAX_N: the double-indirect scan is cheap and skipping the two N-length gathers wins 1.3-1.7x (small per-class report arrays).
#   - n >= _KS_INLINE_ORDERED_MIN_N: the two N-length fancy-index gathers (int64 + float64) become memory-bandwidth-bound and the inline scan wins ~1.05x@200k / ~1.2x@500k bit-identically (bench_ks_desc_order_inline_gather_iter72).
# Between the two the pre-gathered contiguous-scan reference is fastest. Retune per hardware.
_KS_FUSED_MAX_N = 2048
_KS_INLINE_ORDERED_MIN_N = 150_000


@numba.njit(**NUMBA_NJIT_PARAMS)
def _ks_statistic_kernel(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Single-pass Kolmogorov-Smirnov on sorted scores.

    Pre-condition: ``y_score`` is sorted ascending and ``y_true`` is
    reordered to match. Returns max |F1(s) - F0(s)| over the score axis,
    where F1 / F0 are class-conditional empirical CDFs.

    Tie handling: rows with identical y_score must be folded into a SINGLE
    CDF jump (both class CDFs step at the same x). Without this fold the
    kernel reports spurious intermediate |F1 - F0| differences purely
    from the y_true ordering within the tied group (e.g. all-identical
    scores would report KS ~ 0.33 instead of the correct 0).
    """
    n = y_true.shape[0]
    n_pos = 0
    for i in range(n):
        if y_true[i] != 0:
            n_pos += 1
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    inv_pos = 1.0 / n_pos
    inv_neg = 1.0 / n_neg
    cum_pos = 0.0
    cum_neg = 0.0
    ks = 0.0
    i = 0
    while i < n:
        # Accumulate everything at this score level before we check |diff|.
        j = i
        cur = y_score[i]
        while j < n and y_score[j] == cur:
            if y_true[j] != 0:
                cum_pos += inv_pos
            else:
                cum_neg += inv_neg
            j += 1
        d = cum_pos - cum_neg
        if d < 0.0:
            d = -d
        if d > ks:
            ks = d
        i = j
    return ks


@numba.njit(**NUMBA_NJIT_PARAMS)
def _ks_statistic_kernel_ordered(order: np.ndarray, y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Single-pass KS that indexes through ``order`` (an ascending argsort of
    ``y_score``) instead of consuming pre-gathered arrays. Bit-identical to
    ``_ks_statistic_kernel(y_true[order], y_score[order])`` but avoids the two
    N-length gather temporaries -- the gather happens inline in the scan.

    Tie handling matches the reference: rows sharing a y_score value fold into
    a SINGLE CDF jump before |F1 - F0| is checked (see ``_ks_statistic_kernel``).
    """
    n = order.shape[0]
    n_pos = 0
    for i in range(n):
        if y_true[i] != 0:
            n_pos += 1
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    inv_pos = 1.0 / n_pos
    inv_neg = 1.0 / n_neg
    cum_pos = 0.0
    cum_neg = 0.0
    ks = 0.0
    i = 0
    while i < n:
        j = i
        cur = y_score[order[i]]
        while j < n and y_score[order[j]] == cur:
            if y_true[order[j]] != 0:
                cum_pos += inv_pos
            else:
                cum_neg += inv_neg
            j += 1
        d = cum_pos - cum_neg
        if d < 0.0:
            d = -d
        if d > ks:
            ks = d
        i = j
    return ks


def _ks_statistic_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Reference KS: argsort + pre-gathered single-pass kernel. Kept callable
    for bit-identity regression tests against the fused default path."""
    yt = np.asarray(y_true).astype(np.int64, copy=False)
    ys = np.asarray(y_score, dtype=np.float64)
    if yt.shape[0] == 0:
        return np.nan
    order = np.argsort(ys, kind="quicksort")
    return float(_ks_statistic_kernel(yt[order], ys[order]))


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray, desc_order: Optional[np.ndarray] = None) -> float:
    """Kolmogorov-Smirnov statistic between class-conditional score CDFs.

    Returns max_s |F_neg(s) - F_pos(s)| in [0, 1]. Higher is better;
    0 = no discrimination, 1 = perfect separation. Standard in credit
    scoring / fraud / churn.

    NaN when either class is empty (no class-conditional CDF defined).

    ``desc_order`` is an optional precomputed DESCENDING argsort of
    ``y_score`` (e.g. the one ``fast_aucs_per_group_optimized`` already
    builds for AUC over the same scores). When supplied, KS reuses it
    (reversed to ascending) instead of running a second independent
    argsort over the same array -- bit-identical because the KS kernel
    folds tied scores into a single CDF jump, so within-tie ordering of
    the order array never affects the statistic.
    """
    yt = np.asarray(y_true).astype(np.int64, copy=False)
    ys = np.asarray(y_score, dtype=np.float64)
    n = yt.shape[0]
    if n == 0:
        return np.nan
    if desc_order is not None and desc_order.shape[0] == n:
        order = desc_order[::-1]
        if n < _KS_FUSED_MAX_N or n >= _KS_INLINE_ORDERED_MIN_N:
            return float(_ks_statistic_kernel_ordered(np.ascontiguousarray(order), yt, ys))
        return float(_ks_statistic_kernel(yt[order], ys[order]))
    # bench-attempt-rejected (_benchmarks/bench_ks_shared_sort.py): sharing this
    # sort with the AUC score-desc argsort is bit-identical but unimplementable
    # -- the report's AUC sort is inside the batched/GPU compute_batch_aucs
    # (returns scalars, not orders); threading an (N,K) order matrix back adds
    # 8*N*K. Keep own sort.
    # standalone-replace rejected (_benchmarks/bench_ks_statistic_njit.py): np.argsort
    # dominates so no all-sizes win; the in-kernel-argsort variant is 0.3-0.5x.
    # BUT _ks_statistic_kernel_ordered (indexes through order inline, zero gather
    # temporaries) is gated in for n < _KS_FUSED_MAX_N (1.3-1.7x; tiny per-class arrays)
    # AND for n >= _KS_INLINE_ORDERED_MIN_N (~1.05x@200k / ~1.2x@500k; the two N-length
    # gathers turn memory-bandwidth-bound) -- bit-identical, bench_ks_desc_order_inline_gather_iter72.
    # The middle band keeps the pre-gathered contiguous-scan reference.
    order = np.argsort(ys, kind="quicksort")
    if n < _KS_FUSED_MAX_N or n >= _KS_INLINE_ORDERED_MIN_N:
        return float(_ks_statistic_kernel_ordered(order, yt, ys))
    return float(_ks_statistic_kernel(yt[order], ys[order]))


# ---------- Matthews correlation coefficient (binary) ----------


def matthews_corrcoef_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MCC for binary 0/1 labels.

    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Range [-1, 1]; 0 = random; +1 = perfect; -1 = inverted.

    Returns 0.0 when the denominator is 0 (matches sklearn's convention
    when one row or column of the confusion matrix is all zeros).
    """
    tp, fp, tn, fn = _confusion_counts_binary_dispatch(y_true, y_pred)
    return matthews_corrcoef_from_counts(tp, fp, tn, fn)


# ---------- Cohen's kappa (binary) ----------


def cohen_kappa_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cohen's kappa for binary 0/1 labels.

    kappa = (p_o - p_e) / (1 - p_e)
    p_o = observed agreement, p_e = chance agreement under independence.
    Range (-1, 1]; 0 = chance; 1 = perfect.
    """
    tp, fp, tn, fn = _confusion_counts_binary_dispatch(y_true, y_pred)
    n = tp + fp + tn + fn
    if n == 0:
        return np.nan
    p_o = (tp + tn) / n
    row_pos = (tp + fn) / n
    col_pos = (tp + fp) / n
    p_e = row_pos * col_pos + (1.0 - row_pos) * (1.0 - col_pos)
    if 1.0 - p_e == 0:
        return 0.0
    return (p_o - p_e) / (1.0 - p_e)


# ---------- Balanced accuracy / G-mean ----------


def balanced_accuracy_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """(TPR + TNR) / 2. Correct under imbalance where plain accuracy lies."""
    tp, fp, tn, fn = _confusion_counts_binary_dispatch(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (tpr + tnr)


def g_mean_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Geometric mean of sensitivity (TPR) and specificity (TNR).

    Heavily penalises models that sacrifice one class entirely; preferred
    over accuracy for highly imbalanced datasets.
    """
    tp, fp, tn, fn = _confusion_counts_binary_dispatch(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sqrt(tpr * tnr)


# ---------- Brier Skill Score ----------


@numba.njit(**NUMBA_NJIT_PARAMS)
def _brier_skill_score_kernel(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """BSS = 1 - Brier(model) / Brier(marginal_baseline).

    The marginal baseline predicts every row at the empirical positive
    rate; Brier(baseline) = p_bar * (1 - p_bar).
    """
    n = y_true.shape[0]
    pos = 0
    bs_model = 0.0
    for i in range(n):
        if y_true[i] != 0:
            pos += 1
            d = 1.0 - y_score[i]
        else:
            d = y_score[i]
        bs_model += d * d
    bs_model /= n
    p_bar = pos / n
    bs_base = p_bar * (1.0 - p_bar)
    if bs_base == 0.0:
        return np.nan
    return float(1.0 - bs_model / bs_base)


def brier_skill_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Brier Skill Score against the marginal-prevalence baseline.

    1 = perfect probability; 0 = no skill over predicting the base rate;
    negative = worse than predicting the base rate every row.

    NaN when y_true is constant (the marginal baseline is also perfect).

    iter611: dropped the unconditional ``dtype=np.float64`` cast on
    y_score (same pattern as iter595-608). Kernel has 3 ops/element
    (cmp != 0, sub-or-copy, mul-add) -- inside the iter597 safe band.
    Bench n=25k: y_score float64 1.12x, float32 1.16x. Bit-equiv.
    bench-attempt-rejected for ``ks_statistic`` -- argsort dominates
    (~2ms / call), the cast is sub-5% of total cost, so the speedup
    is within timing noise; ks_statistic keeps its cast."""
    yt = np.ascontiguousarray(y_true).astype(np.int64, copy=False)
    ys = np.ascontiguousarray(y_score)
    if yt.shape[0] == 0:
        return np.nan
    return float(_brier_skill_score_kernel(yt, ys))


def brier_skill_score_from_brier(brier_loss: float, y_true: np.ndarray) -> float:
    """BSS derived from an already-computed model Brier loss + the empirical positive rate.

    Brier(model) is identical to ``mean((y_true - y_score)**2)`` -- the exact quantity the
    calibration report already computes once via ``fast_brier_score_loss``. The marginal baseline
    Brier is the closed form ``p_bar * (1 - p_bar)``, so the whole BSS needs only the prevalence
    (one ``sum`` over the int labels), not a second full-n model-Brier scan. Numerically equivalent
    to ``brier_skill_score`` to FP reduction-order (~1e-15)."""
    yt = np.ascontiguousarray(y_true).astype(np.int64, copy=False)
    n = yt.shape[0]
    if n == 0:
        return np.nan
    p_bar = float(yt.sum()) / n
    bs_base = p_bar * (1.0 - p_bar)
    if bs_base == 0.0:
        return np.nan
    return float(1.0 - float(brier_loss) / bs_base)


# ---------- Gini ----------


def gini_from_auc(auc: float) -> float:
    """Gini = 2 * AUC - 1.

    Trivial closed-form but exposed explicitly because the banking /
    credit-scoring convention reports Gini, not AUC, in dashboards.
    NaN propagates.
    """
    return 2.0 * auc - 1.0


# ---------- Specificity / NPV / FPR / FNR ----------


def specificity_npv_fpr_fnr(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Companion to (precision, recall) on the negative class.

    Returns (Specificity=TN/(TN+FP), NPV=TN/(TN+FN), FPR=FP/(FP+TN),
    FNR=FN/(FN+TP)). Each component is 0.0 when its denominator is 0.
    """
    tp, fp, tn, fn = _confusion_counts_binary_dispatch(y_true, y_pred)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return spec, npv, fpr, fnr


# ---------- F-beta ----------


def f_beta_score(
    y_true: np.ndarray, y_pred: np.ndarray, beta: float = 1.0,
) -> float:
    """Generalised F-beta. beta=1 -> F1; beta=2 -> recall weighted 4x more than precision; beta=0.5 -> precision weighted 4x more.

    Returns 0.0 when both precision and recall are 0.
    """
    tp, fp, _tn, fn = _confusion_counts_binary_dispatch(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


# ---------- Spiegelhalter Z ----------


@numba.njit(**NUMBA_NJIT_PARAMS)
def _spiegelhalter_z_kernel(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Returns (numerator, denominator-variance) for Spiegelhalter's Z."""
    n = y_true.shape[0]
    num = 0.0
    var = 0.0
    for i in range(n):
        p = y_score[i]
        # Clip into [1e-15, 1-1e-15] to avoid the (1-2p)*0 degenerate when
        # p hits exactly 0 or 1; this matches sklearn's log_loss convention.
        if p < 1e-15:
            p = 1e-15
        elif p > 1.0 - 1e-15:
            p = 1.0 - 1e-15
        y = 1.0 if y_true[i] != 0 else 0.0
        num += (y - p) * (1.0 - 2.0 * p)
        var += (1.0 - 2.0 * p) * (1.0 - 2.0 * p) * p * (1.0 - p)
    return num, var


def spiegelhalter_z(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Spiegelhalter's Z-test for binary-classification calibration.

    Z = sum_i (y_i - p_i)(1 - 2 p_i) / sqrt(sum_i (1 - 2 p_i)^2 p_i (1 - p_i))

    Under perfect calibration Z ~ N(0, 1). Returns (Z, two-sided p-value).
    |Z| > 1.96 ~= p < 0.05 -> calibration significantly off. Stronger
    statistical test than the ECE which has no associated p-value.
    """
    yt = np.ascontiguousarray(y_true).astype(np.int64, copy=False)
    ys = np.ascontiguousarray(y_score, dtype=np.float64)
    if yt.shape[0] == 0:
        return np.nan, np.nan
    num, var = _spiegelhalter_z_kernel(yt, ys)
    if var <= 0.0:
        return np.nan, np.nan
    z = num / sqrt(var)
    # Two-sided p-value via the complementary error function (avoids scipy).
    p_value = erfc(abs(z) / sqrt(2.0))
    return float(z), float(p_value)


# ---------- Lift @ k ----------


def lift_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k_pct: float = 10.0,
) -> float:
    """Lift at top-k%: fraction of positives captured in top-k% by score,
    divided by k%. A model that predicts the prevalence everywhere has
    lift=1.0; lift>1 = model concentrates positives at the top.

    Standard in marketing / credit ranking. ``k_pct`` in (0, 100].
    """
    if not (0.0 < k_pct <= 100.0):
        raise ValueError(f"k_pct must be in (0, 100], got {k_pct}")
    yt = np.asarray(y_true).astype(np.int64, copy=False)
    ys = np.asarray(y_score, dtype=np.float64)
    n = yt.shape[0]
    if n == 0:
        return np.nan
    n_pos = int(yt.sum())
    if n_pos == 0:
        return np.nan
    cutoff = max(1, int(np.ceil(n * k_pct / 100.0)))
    # argpartition for top-k cutoff selection (no full sort needed).
    top_idx = np.argpartition(-ys, cutoff - 1)[:cutoff]
    captured = int(yt[top_idx].sum())
    p_top = captured / cutoff
    p_overall = n_pos / n
    return float(p_top / p_overall)


# ============================================================================
# Multiclass
# ============================================================================


@numba.njit(**NUMBA_NJIT_PARAMS)
def _top_k_hits_kernel(topk_idx: np.ndarray, y_true: np.ndarray, K: int) -> int:
    """Count rows whose true class appears in the row's top-k index set.

    Bit-identical to the pure-Python hit-count loop it replaces (same
    ``0 <= ti < K`` guard, same break-on-first-match membership scan);
    only the interpreted Python loop is lowered into njit machine code.
    """
    n, k = topk_idx.shape
    hits = 0
    for i in range(n):
        ti = y_true[i]
        if 0 <= ti < K:
            for j in range(k):
                if topk_idx[i, j] == ti:
                    hits += 1
                    break
    return hits


def top_k_accuracy(
    y_true: np.ndarray, probs_NK: np.ndarray, k: int = 1,
) -> float:
    """Fraction of rows where the true class is in the top-k predicted
    probabilities.

    NaN when k >= K (number of classes) since top-k>=K is trivially 1.0
    and uninformative; we surface that to the caller as NaN with a
    warning rather than reporting a meaningless 100%.
    """
    yt = np.asarray(y_true).astype(np.int64, copy=False)
    p = np.asarray(probs_NK, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"probs_NK must be 2-D (N, K), got shape {p.shape}")
    n, K = p.shape
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    if k >= K:
        return np.nan  # trivial, not informative
    # argpartition on -p[i] picks top-k indices per row; partial-sort is
    # O(K) per row vs full sort's O(K log K). The per-row membership count
    # is then done in an njit kernel (was a pure-Python double loop, the
    # hot path at report scale) -- bit-identical, 1.9-5.5x over n in
    # 10k..1M (bench_top_k_accuracy_kernel).
    topk_idx = np.ascontiguousarray(np.argpartition(-p, k - 1, axis=1)[:, :k])
    hits = _top_k_hits_kernel(topk_idx, yt, K)
    return hits / n if n > 0 else np.nan


@numba.njit(**NUMBA_NJIT_PARAMS)
def _multiclass_confusion_kernel(
    y_true: np.ndarray, y_pred: np.ndarray, K: int,
) -> np.ndarray:
    """Returns K x K confusion matrix C[true, pred]."""
    C = np.zeros((K, K), dtype=np.int64)
    for i in range(y_true.shape[0]):
        t = y_true[i]
        p = y_pred[i]
        if 0 <= t < K and 0 <= p < K:
            C[t, p] += 1
    return C


def matthews_corrcoef_multiclass(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: Optional[int] = None,
) -> float:
    """Gorodkin multiclass MCC.

    R_K = (n * sum_k C_kk - sum_k t_k p_k) / (sqrt(n^2 - sum_k p_k^2) * sqrt(n^2 - sum_k t_k^2))

    where C is the K x K confusion matrix, t_k = row sum, p_k = column sum.
    Reduces to the binary MCC when K=2. Returns 0.0 when either
    denominator factor is 0.
    """
    yt = np.asarray(y_true).astype(np.int64, copy=False)
    yp = np.asarray(y_pred).astype(np.int64, copy=False)
    if n_classes is None:
        K = int(max(int(yt.max()) if yt.size > 0 else 0, int(yp.max()) if yp.size > 0 else 0) + 1)
    else:
        K = int(n_classes)
    if K < 2:
        return np.nan
    C = _multiclass_confusion_kernel(yt, yp, K)
    n = int(C.sum())
    if n == 0:
        return np.nan
    t = C.sum(axis=1).astype(np.float64)  # true totals
    p = C.sum(axis=0).astype(np.float64)  # pred totals
    diag = float(np.trace(C))
    tp_dot = float((t * p).sum())
    t2 = float((t * t).sum())
    p2 = float((p * p).sum())
    n2 = float(n) * n
    denom2 = (n2 - t2) * (n2 - p2)
    if denom2 <= 0.0:
        return 0.0
    return (n * diag - tp_dot) / sqrt(denom2)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _rps_kernel(y_true: np.ndarray, probs_NK: np.ndarray) -> float:
    """Ranked Probability Score for ordinal multiclass.

    RPS = (1 / (K - 1)) * mean_i sum_{k=0..K-2} (cumP_i,k - cumY_i,k)^2

    where cumP and cumY are cumulative distributions across the K
    ordered classes. Equivalent to the CRPS evaluated on the discrete
    ordinal scale. Smaller is better. Strict proper scoring rule for
    ordinal targets where class order is meaningful (e.g. star ratings,
    severity grades).
    """
    n, K = probs_NK.shape
    if K < 2:
        return np.nan
    total = 0.0
    for i in range(n):
        cum_p = 0.0
        rps_i = 0.0
        ti = y_true[i]
        for k in range(K - 1):
            cum_p += probs_NK[i, k]
            cum_y = 1.0 if ti <= k else 0.0
            d = cum_p - cum_y
            rps_i += d * d
        total += rps_i / (K - 1)
    return float(total / n)


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _rps_kernel_par(y_true: np.ndarray, probs_NK: np.ndarray) -> float:
    """Parallel (prange) Ranked Probability Score over ``n`` samples and ``K`` ordinal classes; ``nan`` when ``K < 2``."""
    n, K = probs_NK.shape
    if K < 2:
        return np.nan
    total = 0.0
    for i in numba.prange(n):
        cum_p = 0.0
        rps_i = 0.0
        ti = y_true[i]
        for k in range(K - 1):
            cum_p += probs_NK[i, k]
            cum_y = 1.0 if ti <= k else 0.0
            d = cum_p - cum_y
            rps_i += d * d
        total += rps_i / (K - 1)
    return float(total / n)


def ranked_probability_score(
    y_true: np.ndarray, probs_NK: np.ndarray,
) -> float:
    """Ranked Probability Score (RPS).

    Proper scoring rule for ordinal multiclass / multinomial targets.
    Generalises the Brier score to ordered K classes by penalising
    distance between cumulative predicted and observed distributions.

    Lower is better; range [0, 1]. Reduces to Brier when K=2.
    """
    yt = np.ascontiguousarray(y_true).astype(np.int64, copy=False)
    p = np.ascontiguousarray(probs_NK, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != yt.shape[0]:
        raise ValueError(
            f"shape mismatch: y_true={yt.shape}, probs_NK={p.shape}",
        )
    if yt.shape[0] >= _PARALLEL_REDUCTION_THRESHOLD:
        return float(_rps_kernel_par(yt, p))
    return float(_rps_kernel(yt, p))


# ============================================================================
# Fused single-pass blocks
# ============================================================================
#
# Bench (mlframe.metrics._benchmarks.bench_extended_metric_blocks) -
# measured on Win11 / numba 0.58 / 16-thread Ryzen 2026-05-28:
#
#   Binary confusion block (11 metrics: accuracy, balanced_accuracy, MCC,
#   Cohen-kappa, F1, F0.5, F2, precision, recall, specificity, NPV, FPR/FNR,
#   G-mean):
#     N=10k:   separate 11 calls=0.39 ms   fused=0.06 ms  (6.56x)
#     N=500k:  separate 11 calls=26.0 ms   fused=2.9 ms   (8.86x)
#     N=5M:    separate 11 calls=239.2 ms  fused=33.1 ms  (7.22x)
#
#   Binary probability block (Brier, log_loss, base_rate, BSS,
#   Spiegelhalter Z, Spiegelhalter p):
#     N=10k:   separate 4 calls=0.21 ms    fused=0.11 ms  (1.95x)
#     N=500k:  separate 4 calls=6.2 ms     fused=2.5 ms   (2.53x)
#     N=5M:    separate 4 calls=58.8 ms    fused=20.1 ms  (2.93x)
#
# Binary probability speedup is lower because each separate kernel is
# already a single-pass numba reduction; the win comes only from RAM-walk
# reuse + amortising 4 numba dispatch overheads into 1. All fused-block
# returns are bit-identical to the separate calls except for fp ordering
# jitter in the parallel-accumulator paths (max |diff| < 1e-13 at N=5M).


# binary/multiclass confusion+probability block kernels carved to _classification_extras_blocks.py (1k-LOC ceiling).
from ._classification_calibration import (
    _hosmer_lemeshow_kernel,
    accuracy_ratio,
    hosmer_lemeshow_test,
)
from ._classification_extras_blocks import (
    _binary_confusion_block_kernel_seq,
    _binary_probability_block_kernel_par,
    _binary_probability_block_kernel_seq,
    fast_binary_confusion_metrics_block,
    fast_binary_probability_metrics_block,
    fast_multiclass_confusion_metrics_block,
)
