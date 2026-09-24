"""Sample-weighted twin of the batched per-class ICE kernel.

A weighted fit (fairness / inverse-frequency / recency weights) early-stopped on an ICE that counted every row once, so the
metric deciding when to stop described a different sample from the loss being minimised. This kernel takes a per-row
weight vector and weights every component the same way: Brier is the weighted mean squared error, a calibration bin's
mass, mean prediction and observed frequency are weight sums, and ROC / PR AUC accumulate weighted true and false
positives (the trapezoid and step sums of sklearn's ``sample_weight`` form). Coverage still counts bins that hold at least
one row.

The caller supplies each row's bin (``bin_index``), so the uniform and the equal-mass strategies share one kernel.
With every weight equal to 1 and uniform bins it returns exactly what ``_batch_per_class_ice_kernel_serial`` returns (pinned by
``tests/metrics/test_ice_weighted_kernel.py``); the unweighted kernels stay as they are and remain the path for
unweighted calls.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(fastmath=False, cache=True, nogil=True)
def _weighted_class_ice(
    y_t: np.ndarray,
    y_p: np.ndarray,
    w: np.ndarray,
    desc_idx: np.ndarray,
    bin_idx: np.ndarray,
    nbins: int,
    use_weights: bool,
    mae_weight: float,
    std_weight: float,
    brier_loss_weight: float,
    roc_auc_weight: float,
    pr_auc_weight: float,
    min_roc_auc: float,
    roc_auc_penalty: float,
    coverage_weight: float,
) -> float:
    """ICE of one class column under per-row weights ``w``; ``bin_idx`` is each row's calibration bin (see ``bin_index``)."""
    N = y_t.shape[0]
    w_total = 0.0
    s = 0.0
    for i in range(N):
        d = float(y_t[i]) - y_p[i]
        s += w[i] * d * d
        w_total += w[i]
    brier = s / w_total if w_total > 0 else 1.0

    counts = np.zeros(nbins, dtype=np.int64)
    mass = np.zeros(nbins, dtype=np.float64)
    true_mass = np.zeros(nbins, dtype=np.float64)
    pred_mass = np.zeros(nbins, dtype=np.float64)
    for i in range(N):
        ind = bin_idx[i]
        counts[ind] += 1
        mass[ind] += w[i]
        true_mass[ind] += w[i] * y_t[i]
        pred_mass[ind] += w[i] * y_p[i]

    n_nonempty = 0
    n_scored = 0
    for b in range(nbins):
        if counts[b] > 0:
            n_nonempty += 1
            if mass[b] > 0:
                n_scored += 1
    freqs_pred = np.empty(n_scored, dtype=np.float64)
    freqs_true = np.empty(n_scored, dtype=np.float64)
    hits = np.empty(n_scored, dtype=np.float64)
    ptr = 0
    for b in range(nbins):
        if counts[b] > 0 and mass[b] > 0:
            freqs_pred[ptr] = pred_mass[b] / mass[b]
            freqs_true[ptr] = true_mass[b] / mass[b]
            hits[ptr] = mass[b]
            ptr += 1

    if n_scored > 0:
        if use_weights:
            bw = np.empty(n_scored, dtype=np.float64)
            bw_sum = 0.0
            for b in range(n_scored):
                bw[b] = hits[b] ** 0.8
                bw_sum += bw[b]
            if bw_sum > 0:
                for b in range(n_scored):
                    bw[b] /= bw_sum
            cal_mae = 0.0
            for b in range(n_scored):
                cal_mae += abs(freqs_pred[b] - freqs_true[b]) * bw[b]
            cal_var = 0.0
            for b in range(n_scored):
                d = abs(freqs_pred[b] - freqs_true[b]) - cal_mae
                cal_var += d * d * bw[b]
            cal_std = np.sqrt(cal_var)
        else:
            cal_mae = 0.0
            for b in range(n_scored):
                cal_mae += abs(freqs_pred[b] - freqs_true[b])
            cal_mae /= n_scored
            cal_var = 0.0
            for b in range(n_scored):
                d = abs(freqs_pred[b] - freqs_true[b]) - cal_mae
                cal_var += d * d
            cal_std = np.sqrt(cal_var / n_scored)
    else:
        cal_mae = 1.0
        cal_std = 1.0

    total_pos = 0.0
    total_neg = 0.0
    for i in range(N):
        total_pos += w[i] * y_t[i]
        total_neg += w[i] * (1 - y_t[i])
    if total_pos <= 0.0 or total_neg <= 0.0:
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        last_fps = 0.0
        last_tps = 0.0
        tps = 0.0
        fps = 0.0
        roc_acc = 0.0
        pr_acc = 0.0
        prev_recall = 0.0
        for j in range(N):
            i = desc_idx[j]
            yi = y_t[i]
            tps += w[i] * yi
            fps += w[i] * (1 - yi)
            if j == N - 1 or y_p[desc_idx[j + 1]] != y_p[i]:
                roc_acc += (fps - last_fps) * (last_tps + tps)
                last_fps = fps
                last_tps = tps
                precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
                recall = tps / total_pos
                pr_acc += (recall - prev_recall) * precision
                prev_recall = recall
        denom = tps * fps * 2
        roc_auc = roc_acc / denom if denom > 0 else np.nan
        pr_auc = pr_acc

    coverage = n_nonempty / nbins if nbins > 0 else 1.0
    base_loss = brier * brier_loss_weight + cal_mae * mae_weight + cal_std * std_weight + (1.0 - coverage) * coverage_weight
    roc_term = 0.0 if np.isnan(roc_auc) else np.abs(roc_auc - 0.5) * roc_auc_weight
    pr_term = 0.0 if np.isnan(pr_auc) else pr_auc * pr_auc_weight
    ice = base_loss - roc_term - pr_term
    threshold_width = min_roc_auc - 0.5
    if threshold_width > 0.0 and not np.isnan(roc_auc):
        deficit = threshold_width - np.abs(roc_auc - 0.5)
        if deficit > 0.0:
            ice += (deficit / threshold_width) * roc_auc_penalty
    return ice


def bin_index(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray, nbins: int, strategy: str) -> np.ndarray:
    """Each row's calibration bin, int64 in [0, nbins).

    ``"uniform"``: the batched kernel's equal-width grid (``floor((p - min) * nbins / span)``, clamped). ``"quantile"``:
    edges at equal fractions of the WEIGHT mass (the weighted form of equal-population bins), deduplicated like
    ``calibration_binning``; a row goes to the last edge at or below it. Fewer than two distinct edges falls back to uniform.
    """
    p = np.asarray(y_pred, dtype=np.float64)
    if strategy == "quantile" and p.size:
        order = np.argsort(p, kind="stable")
        cum = np.cumsum(np.asarray(sample_weight, dtype=np.float64)[order])
        if cum[-1] > 0:
            targets = np.linspace(0.0, 1.0, nbins + 1) * cum[-1]
            pos = np.minimum(np.searchsorted(cum, targets, side="left"), p.size - 1)
            edges = np.unique(p[order][pos])
            if edges.size >= 2:
                return np.searchsorted(edges[1:-1], p, side="right").astype(np.int64)
    lo, hi = (float(p.min()), float(p.max())) if p.size else (0.0, 0.0)
    span = hi - lo
    if span <= 0:
        return np.zeros(p.size, dtype=np.int64)
    return np.clip(np.floor((p - lo) * (nbins / span)), 0, nbins - 1).astype(np.int64)


@numba.njit(fastmath=False, cache=True, nogil=True)
def batch_per_class_ice_weighted(
    y_true_NK: np.ndarray,
    y_pred_NK: np.ndarray,
    desc_idx_NK: np.ndarray,
    bin_idx_NK: np.ndarray,
    sample_weight: np.ndarray,
    nbins: int,
    use_weights: bool,
    mae_weight: float,
    std_weight: float,
    brier_loss_weight: float,
    roc_auc_weight: float,
    pr_auc_weight: float,
    min_roc_auc: float,
    roc_auc_penalty: float,
    coverage_weight: float = 0.0,
) -> np.ndarray:
    """Per-class ICE, shape (K,), with every component weighted by ``sample_weight`` (shape (N,), non-negative)."""
    K = y_true_NK.shape[1]
    out = np.empty(K, dtype=np.float64)
    for k in range(K):
        out[k] = _weighted_class_ice(
            np.ascontiguousarray(y_true_NK[:, k]), np.ascontiguousarray(y_pred_NK[:, k]), sample_weight,
            np.ascontiguousarray(desc_idx_NK[:, k]), np.ascontiguousarray(bin_idx_NK[:, k]), nbins, use_weights, mae_weight, std_weight, brier_loss_weight,
            roc_auc_weight, pr_auc_weight, min_roc_auc, roc_auc_penalty, coverage_weight,
        )
    return out
