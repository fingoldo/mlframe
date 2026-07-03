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
from typing import Tuple

import numba
import numpy as np

from .._numba_params import _PARALLEL_REDUCTION_THRESHOLD, NUMBA_NJIT_PARAMS, _check_equal_length

# ---------- helpers ----------
from ._classification_extras import _confusion_counts_binary_par, _multiclass_confusion_kernel  # noqa: E402 (cycle-safe)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _binary_confusion_block_kernel_seq(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> Tuple[int, int, int, int]:
    """Sequential confusion-counts. Same body as
    ``_confusion_counts_binary`` but unrolled here so the fused
    public API doesn't pay the cross-module call overhead."""
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


def fast_binary_confusion_metrics_block(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> dict:
    """Compute 11 confusion-matrix metrics in ONE pass over (y_true, y_pred).

    Returns:
        accuracy, balanced_accuracy, MCC, Cohen_kappa,
        F1, F0_5, F2, precision, recall, specificity, NPV,
        FPR, FNR, G_mean
    plus _TP/_FP/_TN/_FN for downstream callers.

    Single pass; ~8-9x faster than 11 separate fast_* calls (see bench
    comment in this module).
    """
    _check_equal_length(y_true, y_pred)
    yt = np.ascontiguousarray(y_true).astype(np.int64, copy=False)
    yp = np.ascontiguousarray(y_pred).astype(np.int64, copy=False)
    if yt.shape[0] >= _PARALLEL_REDUCTION_THRESHOLD:
        tp, fp, tn, fn = _confusion_counts_binary_par(yt, yp, numba.get_num_threads())
    else:
        tp, fp, tn, fn = _binary_confusion_block_kernel_seq(yt, yp)
    n = tp + fp + tn + fn

    # All derived in scalar arithmetic from the 4 counts.
    accuracy = (tp + tn) / n if n > 0 else np.nan
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    balanced_accuracy = 0.5 * (tpr + tnr)
    g_mean = sqrt(tpr * tnr)

    # MCC
    mcc_num = float(tp) * tn - float(fp) * fn
    d1 = tp + fp; d2 = tp + fn; d3 = tn + fp; d4 = tn + fn
    if d1 == 0 or d2 == 0 or d3 == 0 or d4 == 0:
        mcc = 0.0
    else:
        mcc = mcc_num / sqrt(float(d1) * d2 * d3 * d4)

    # Cohen's kappa
    if n == 0:
        cohen_kappa = np.nan
    else:
        p_o = (tp + tn) / n
        row_pos = (tp + fn) / n
        col_pos = (tp + fp) / n
        p_e = row_pos * col_pos + (1.0 - row_pos) * (1.0 - col_pos)
        cohen_kappa = 0.0 if (1.0 - p_e) == 0 else (p_o - p_e) / (1.0 - p_e)

    # F-beta family - precision/recall reused
    def _fb(b: float) -> float:
        b2 = b * b
        d = b2 * precision + tpr
        return 0.0 if d == 0 else (1.0 + b2) * precision * tpr / d
    f1 = _fb(1.0)
    f0_5 = _fb(0.5)
    f2 = _fb(2.0)

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "MCC": float(mcc),
        "Cohen_kappa": float(cohen_kappa),
        "F1": float(f1),
        "F0_5": float(f0_5),
        "F2": float(f2),
        "precision": float(precision),
        "recall": float(tpr),
        "specificity": float(tnr),
        "NPV": float(npv),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "G_mean": float(g_mean),
        "_TP": int(tp), "_FP": int(fp), "_TN": int(tn), "_FN": int(fn),
    }


@numba.njit(**NUMBA_NJIT_PARAMS)
def _binary_probability_block_kernel_seq(
    y_true: np.ndarray, y_score: np.ndarray,
) -> Tuple[float, float, float, float, int]:
    """Fused single-pass over (y_true, y_score). Returns:
        (sum_brier, sum_logloss, sh_num, sh_var, n_pos)
    from which Brier, log-loss, Spiegelhalter Z, base rate, BSS all derive.
    """
    n = y_true.shape[0]
    sum_brier = 0.0
    sum_log = 0.0
    sh_num = 0.0
    sh_var = 0.0
    n_pos = 0
    for i in range(n):
        p = y_score[i]
        if p < 1e-15:
            p = 1e-15
        elif p > 1.0 - 1e-15:
            p = 1.0 - 1e-15
        if y_true[i] != 0:
            n_pos += 1
            d = 1.0 - p
            sum_log -= np.log(p)
        else:
            d = p
            sum_log -= np.log(1.0 - p)
        sum_brier += d * d
        # Spiegelhalter Z accumulators
        y = 1.0 if y_true[i] != 0 else 0.0
        sh_num += (y - p) * (1.0 - 2.0 * p)
        sh_var += (1.0 - 2.0 * p) * (1.0 - 2.0 * p) * p * (1.0 - p)
    return sum_brier, sum_log, sh_num, sh_var, n_pos


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _binary_probability_block_kernel_par(
    y_true: np.ndarray, y_score: np.ndarray, n_threads: int,
):
    n = y_true.shape[0]
    chunk = (n + n_threads - 1) // n_threads
    l_brier = np.zeros(n_threads, dtype=np.float64)
    l_log = np.zeros(n_threads, dtype=np.float64)
    l_num = np.zeros(n_threads, dtype=np.float64)
    l_var = np.zeros(n_threads, dtype=np.float64)
    l_pos = np.zeros(n_threads, dtype=np.int64)
    for tid in numba.prange(n_threads):
        s_brier = 0.0; s_log = 0.0; s_num = 0.0; s_var = 0.0
        np_count = 0
        start = tid * chunk
        end = start + chunk
        if end > n:
            end = n
        for i in range(start, end):
            p = y_score[i]
            if p < 1e-15:
                p = 1e-15
            elif p > 1.0 - 1e-15:
                p = 1.0 - 1e-15
            if y_true[i] != 0:
                np_count += 1
                d = 1.0 - p
                s_log -= np.log(p)
            else:
                d = p
                s_log -= np.log(1.0 - p)
            s_brier += d * d
            y = 1.0 if y_true[i] != 0 else 0.0
            s_num += (y - p) * (1.0 - 2.0 * p)
            s_var += (1.0 - 2.0 * p) * (1.0 - 2.0 * p) * p * (1.0 - p)
        l_brier[tid] = s_brier
        l_log[tid] = s_log
        l_num[tid] = s_num
        l_var[tid] = s_var
        l_pos[tid] = np_count
    return (
        float(l_brier.sum()),
        float(l_log.sum()),
        float(l_num.sum()),
        float(l_var.sum()),
        int(l_pos.sum()),
    )


def fast_binary_probability_metrics_block(
    y_true: np.ndarray, y_score: np.ndarray,
) -> dict:
    """Compute 6 probabilistic-binary metrics in ONE pass over (y_true, y_score).

    Returns:
        Brier, log_loss, base_rate, BSS,
        Spiegelhalter_Z, Spiegelhalter_p

    Single pass; ~3-4x faster than separate Brier + log_loss + Spiegelhalter
    + base-rate calls. NOT fused with KS / ROC AUC / PR AUC (those need
    sorted scores, an O(N log N) prerequisite that the un-sorted single
    pass cannot share).

    Use this AS a complement to compute_batch_aucs (which already does
    ROC+PR AUC fused). Together they cover the full probabilistic-binary
    metric block in 2 passes total (1 sort-pass + 1 raw pass).
    """
    _check_equal_length(y_true, y_score)
    yt = np.ascontiguousarray(y_true).astype(np.int64, copy=False)
    ys = np.ascontiguousarray(y_score, dtype=np.float64)
    n = yt.shape[0]
    # The standalone fast_brier_score_loss / fast_log_loss_binary return NaN on out-of-[0,1] or NaN probabilities;
    # this fused block previously CLAMPED such scores to [1e-15, 1-1e-15], reporting a plausible-looking Brier /
    # log_loss for what is actually an invalid-probability model -- two code paths disagreeing on the same input.
    # Match the standalone contract: an out-of-range / non-finite score set yields an all-NaN block, surfacing the
    # defect instead of hiding it. (Values legitimately AT 0.0/1.0 are in-range and still clamped for the log.)
    invalid_scores = n > 0 and not (np.isfinite(ys).all() and ys.min() >= 0.0 and ys.max() <= 1.0)
    if n == 0 or invalid_scores:
        return {
            "Brier": np.nan, "log_loss": np.nan, "base_rate": np.nan,
            "BSS": np.nan, "Spiegelhalter_Z": np.nan, "Spiegelhalter_p": np.nan,
        }
    if n >= _PARALLEL_REDUCTION_THRESHOLD:
        sum_brier, sum_log, sh_num, sh_var, n_pos = _binary_probability_block_kernel_par(yt, ys, numba.get_num_threads())
    else:
        sum_brier, sum_log, sh_num, sh_var, n_pos = _binary_probability_block_kernel_seq(yt, ys)
    brier = sum_brier / n
    log_loss = sum_log / n
    base_rate = n_pos / n
    bs_base = base_rate * (1.0 - base_rate)
    bss = (1.0 - brier / bs_base) if bs_base > 0.0 else np.nan
    if sh_var <= 0.0:
        sh_z = np.nan
        sh_p = np.nan
    else:
        sh_z = sh_num / sqrt(sh_var)
        sh_p = erfc(abs(sh_z) / sqrt(2.0))
    return {
        "Brier": float(brier),
        "log_loss": float(log_loss),
        "base_rate": float(base_rate),
        "BSS": float(bss),
        "Spiegelhalter_Z": float(sh_z),
        "Spiegelhalter_p": float(sh_p),
    }


# ----- Multiclass confusion block -----


def fast_multiclass_confusion_metrics_block(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int,
) -> dict:
    """Compute multiclass confusion-derived metrics in ONE pass.

    Returns:
        accuracy, balanced_accuracy, MCC_multiclass,
        per_class_precision, per_class_recall, per_class_f1,
        macro_precision, macro_recall, macro_f1,
        micro_precision, micro_recall, micro_f1,
        weighted_precision, weighted_recall, weighted_f1
    Plus _confusion_matrix (K x K) for downstream callers.

    Bench (mlframe.metrics._benchmarks.bench_classification_blocks):
        N=10k K=10:    separate 7 calls=0.45 ms  fused=0.09 ms (5.0x)
        N=500k K=10:   separate 7 calls=12.2 ms  fused=2.0 ms  (6.1x)
        N=5M K=10:     separate 7 calls=119 ms   fused=20 ms   (6.0x)
    """
    _check_equal_length(y_true, y_pred)
    yt = np.ascontiguousarray(y_true).astype(np.int64, copy=False)
    yp = np.ascontiguousarray(y_pred).astype(np.int64, copy=False)
    K = int(n_classes)
    C = _multiclass_confusion_kernel(yt, yp, K)
    n = int(C.sum())
    if n == 0 or K < 2:
        return {
            "accuracy": np.nan, "balanced_accuracy": np.nan,
            "MCC_multiclass": np.nan,
            "macro_precision": np.nan, "macro_recall": np.nan, "macro_f1": np.nan,
            "micro_precision": np.nan, "micro_recall": np.nan, "micro_f1": np.nan,
            "weighted_precision": np.nan, "weighted_recall": np.nan, "weighted_f1": np.nan,
            "per_class_precision": np.full(K, np.nan, dtype=np.float64),
            "per_class_recall": np.full(K, np.nan, dtype=np.float64),
            "per_class_f1": np.full(K, np.nan, dtype=np.float64),
            "_confusion_matrix": C,
        }
    diag = np.diag(C).astype(np.float64)
    row_sums = C.sum(axis=1).astype(np.float64)  # support (true totals)
    col_sums = C.sum(axis=0).astype(np.float64)  # predicted totals
    # Per-class precision / recall / F1 (NaN-safe via maximum(*, eps))
    precision = np.where(col_sums > 0, diag / np.maximum(col_sums, 1), 0.0)
    recall = np.where(row_sums > 0, diag / np.maximum(row_sums, 1), 0.0)
    f1_denom = precision + recall
    f1 = np.where(f1_denom > 0, 2 * precision * recall / np.maximum(f1_denom, 1e-30), 0.0)
    # Macro
    macro_precision = float(precision.mean())
    macro_recall = float(recall.mean())
    macro_f1 = float(f1.mean())
    # Weighted (by true support)
    sup_total = row_sums.sum()
    weighted_precision = float((precision * row_sums).sum() / sup_total)
    weighted_recall = float((recall * row_sums).sum() / sup_total)
    weighted_f1 = float((f1 * row_sums).sum() / sup_total)
    # Micro = accuracy for single-label multiclass (sklearn convention)
    accuracy = float(diag.sum() / n)
    micro_precision = accuracy
    micro_recall = accuracy
    micro_f1 = accuracy
    # Balanced accuracy = mean per-class recall
    balanced_accuracy = float(recall.mean())
    # Gorodkin multiclass MCC
    tp_dot = float((row_sums * col_sums).sum())
    t2 = float((row_sums * row_sums).sum())
    p2 = float((col_sums * col_sums).sum())
    n2 = float(n) * n
    denom2 = (n2 - t2) * (n2 - p2)
    mcc_multi = (n * diag.sum() - tp_dot) / sqrt(denom2) if denom2 > 0.0 else 0.0
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "MCC_multiclass": float(mcc_multi),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "_confusion_matrix": C,
    }


