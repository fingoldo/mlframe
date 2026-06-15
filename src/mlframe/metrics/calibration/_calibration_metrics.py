"""Calibration metric kernels (CMAEW, ECE, Brier-decomposition) for ``mlframe.metrics.core``.

Split out from ``core.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; every moved symbol is
re-exported from ``core`` so existing
``from mlframe.metrics.core import fast_calibration_metrics`` (and the
other moved names) imports continue to work.

What lives here:
  - ``calibration_metrics_from_freqs`` (CMAEW + coverage from pre-binned freqs)
  - ``compute_ece_and_brier_decomposition`` (Murphy 1973 decomposition +
    ECE on a single data-adaptive binning pass)
  - ``fast_calibration_metrics`` (one-shot binning + CMAEW wrapper)

The binning grid is shared with ``_calibration_plot.fast_calibration_binning``
so ECE/REL bin boundaries match CMAEW exactly across these kernels.
"""
from __future__ import annotations

from math import floor

import numpy as np
import numba

from .._numba_params import NUMBA_NJIT_PARAMS
from ._calibration_plot import _fast_calibration_binning_serial


@numba.njit(**NUMBA_NJIT_PARAMS)
def calibration_metrics_from_freqs(
    freqs_predicted: np.ndarray,
    freqs_true: np.ndarray,
    hits: np.ndarray,
    nbins: int,
    array_size: int,
    use_weights: bool = True,
    use_log_weighting: bool = False,
    use_sqrt_weighting: bool = False,
    use_power_weighting: bool = True,
):
    # Rounding precision must be >= 1 decimal place even for small nbins (previously
    # int(np.log10(5)) == 0 meant integer-rounding, collapsing all bins together).
    _round_prec = max(1, int(np.ceil(np.log10(max(nbins, 2)))))
    calibration_coverage = len(set(np.round(freqs_predicted, _round_prec))) / nbins
    if len(hits) > 0:
        diffs = np.abs((freqs_predicted - freqs_true))
        if use_weights:

            if use_log_weighting:
                weights = np.log1p(hits)
            elif use_sqrt_weighting:
                weights = np.sqrt(hits)
            elif use_power_weighting:
                alpha = 0.8  # adjust between (0, 1)
                weights = hits**alpha
            else:
                weights = hits.astype(np.float64)

            # Normalize weights to sum to 1. A prior ``+1e-6`` constant was arbitrary and
            # mattered whenever weights.sum() was small (few bins, low hits counts) -- it
            # biased the weighted MAE toward zero. Guard against the only legitimate zero
            # case explicitly instead.
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum

            calibration_mae = np.sum(diffs * weights)
            calibration_std = np.sqrt(np.sum(((diffs - calibration_mae) ** 2) * weights))
        else:
            calibration_mae = np.mean(diffs)
            calibration_std = np.sqrt(np.mean(((diffs - calibration_mae) ** 2)))
    else:
        calibration_mae, calibration_std = 1.0, 1.0

    return calibration_mae, calibration_std, calibration_coverage


@numba.njit(**NUMBA_NJIT_PARAMS)
def compute_ece_and_brier_decomposition(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nbins: int,
):
    """ECE plus Murphy 1973 Brier score decomposition.

    Returns ``(ece, reliability, resolution, uncertainty, brier_binned)``.

    ECE = sum_k (n_k / N) * |p_mean_k - acc_k|
    BinnedBrier = REL - RES + UNC      (exact identity by construction)
        REL = sum_k (n_k / N) * (p_mean_k - acc_k)^2
        RES = sum_k (n_k / N) * (acc_k - base_rate)^2
        UNC = base_rate * (1 - base_rate)
    where p_mean_k is the *mean predicted probability* in bin k (NOT the bin
    centre), acc_k is the observed positive rate in bin k, base_rate is the
    overall positive rate.

    The kernel does its own binning over [min(y_pred), max(y_pred)] using the
    same data-adaptive grid as fast_calibration_binning - so ECE/REL bin
    boundaries match CMAEW exactly. Per-bin p_mean (not bin centre) is used so
    the Murphy identity ``BinnedBrier == REL - RES + UNC`` holds exactly to fp
    precision; this matters for the test asserting the identity, and for users
    checking REL when they care about absolute magnitude. Raw Brier (computed
    by ``fast_brier_score_loss``) differs from BinnedBrier by the within-bin
    variance of predictions; that gap shrinks with finer binning.

    Returns 1.0/1.0/0.0/0.0/1.0 on empty input - mirrors degenerate handling
    elsewhere in the calibration pipeline.
    """
    n = len(y_true)
    if n == 0:
        return 1.0, 1.0, 0.0, 0.0, 1.0

    base_rate = 0.0
    for i in range(n):
        base_rate += y_true[i]
    base_rate /= n

    # Min/max span - same data-adaptive grid as fast_calibration_binning.
    min_val = 1.0
    max_val = 0.0
    for i in range(n):
        p = y_pred[i]
        if p > max_val:
            max_val = p
        if p < min_val:
            min_val = p
    span = max_val - min_val

    pred_sum = np.zeros(nbins, dtype=np.float64)
    true_sum = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)

    if span > 0:
        multiplier = (nbins - 1) / span
        for i in range(n):
            p = y_pred[i]
            ind = int(floor((p - min_val) * multiplier))
            counts[ind] += 1
            pred_sum[ind] += p
            true_sum[ind] += y_true[i]
    else:
        # All predictions identical - one bin holds everything.
        for i in range(n):
            counts[0] += 1
            pred_sum[0] += y_pred[i]
            true_sum[0] += y_true[i]

    ece = 0.0
    reliability = 0.0
    resolution = 0.0
    inv_n = 1.0 / n
    for k in range(nbins):
        if counts[k] == 0:
            continue
        w = counts[k] * inv_n
        p_mean = pred_sum[k] / counts[k]
        acc = true_sum[k] / counts[k]
        diff = p_mean - acc
        ece += w * abs(diff)
        reliability += w * diff * diff
        resolution += w * (acc - base_rate) ** 2
    uncertainty = base_rate * (1.0 - base_rate)
    brier_binned = reliability - resolution + uncertainty
    return ece, reliability, resolution, uncertainty, brier_binned


@numba.njit(**NUMBA_NJIT_PARAMS)
def compute_ece_debiased(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nbins: int,
):
    """Bias-corrected binned ECE (Kumar et al. NeurIPS 2019; Roelofs et al. AISTATS 2022).

    The plug-in binned ECE ``sum_b (n_b/N)*|conf_b - acc_b|`` is a POSITIVELY biased estimator of the
    population ECE: ``acc_b`` is a noisy Bernoulli-rate estimate, and ``E[|conf_b - acc_b|]`` is inflated
    by the per-bin sampling noise (Jensen on the absolute value). A perfectly calibrated model (true ECE 0)
    therefore reports a spurious positive ECE that grows with nbins and shrinks with bin population.

    Correction: the per-bin squared gap satisfies ``E[g_b^2] = (conf_b - true_acc_b)^2 + Var(acc_b)`` with
    ``Var(acc_b) = acc_b*(1-acc_b)/(n_b-1)`` (unbiased Bernoulli variance of the bin mean). Subtracting the
    noise term and clamping at 0 gives a debiased squared gap; the debiased ECE is
    ``sum_b (n_b/N)*sqrt(max(g_b^2 - Var(acc_b), 0))``. Bins with ``n_b < 2`` have no variance estimate and
    keep the raw gap. This removes the noise floor so a calibrated model scores ~0 rather than a positive
    artefact, without changing the verdict on a genuinely miscalibrated model (the true-gap term dominates).

    Uses the SAME data-adaptive equal-width binning grid as ``compute_ece_and_brier_decomposition`` so the
    debiased and plug-in estimates are directly comparable. Returns 1.0 on empty input (degenerate handling
    mirrors the rest of the calibration pipeline).
    """
    n = len(y_true)
    if n == 0:
        return 1.0

    min_val = 1.0
    max_val = 0.0
    for i in range(n):
        p = y_pred[i]
        if p > max_val:
            max_val = p
        if p < min_val:
            min_val = p
    span = max_val - min_val

    pred_sum = np.zeros(nbins, dtype=np.float64)
    true_sum = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)

    if span > 0:
        multiplier = (nbins - 1) / span
        for i in range(n):
            p = y_pred[i]
            ind = int(floor((p - min_val) * multiplier))
            counts[ind] += 1
            pred_sum[ind] += p
            true_sum[ind] += y_true[i]
    else:
        for i in range(n):
            counts[0] += 1
            pred_sum[0] += y_pred[i]
            true_sum[0] += y_true[i]

    ece = 0.0
    inv_n = 1.0 / n
    for k in range(nbins):
        nk = counts[k]
        if nk == 0:
            continue
        w = nk * inv_n
        p_mean = pred_sum[k] / nk
        acc = true_sum[k] / nk
        gap = abs(p_mean - acc)
        if nk >= 2:
            var_acc = acc * (1.0 - acc) / (nk - 1)
            corrected = gap * gap - var_acc
            if corrected < 0.0:
                corrected = 0.0
            ece += w * np.sqrt(corrected)
        else:
            ece += w * gap
    return ece


@numba.njit(**NUMBA_NJIT_PARAMS)
def fast_calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray, nbins: int = 100, use_weights: bool = False, verbose: int = 0):
    # Call the serial njit binning kernel directly: ``fast_calibration_binning`` is a plain-Python size dispatcher (not njit), so referencing it from inside this nopython body fails type inference. This wrapper is a one-shot small-n convenience path, so the serial kernel is the right njit-callable choice.
    freqs_predicted, freqs_true, hits = _fast_calibration_binning_serial(y_true, y_pred, nbins)
    if verbose:
        print(freqs_predicted, freqs_true)
    return calibration_metrics_from_freqs(
        freqs_predicted=freqs_predicted, freqs_true=freqs_true, hits=hits, nbins=nbins, array_size=len(y_true), use_weights=use_weights
    )


@numba.njit(**NUMBA_NJIT_PARAMS)
def integral_calibration_error_from_metrics(
    calibration_mae: float,
    calibration_std: float,
    calibration_coverage: float,
    brier_loss: float,
    roc_auc: float,
    pr_auc: float,
    mae_weight: float = 3,
    std_weight: float = 2,
    roc_auc_weight: float = 1.5,
    pr_auc_weight: float = 0.1,
    brier_loss_weight: float = 0.8,
    min_roc_auc: float = 0.54,
    roc_auc_penalty: float = 0.00,
) -> float:
    """Compute Integral Calibration Error (ICE) from base ML metrics.

    ICE is a weighted sum of baseline losses minus rewards for sharp ranking
    (roc_auc, pr_auc). When ``roc_auc`` is weaker than ``min_roc_auc``, a
    penalty smoothly ramps up from 0 at the threshold to ``roc_auc_penalty``
    at the worst case ``roc_auc == 0.5`` (complete random). The ramp is
    linear in the deficit and symmetric about 0.5 (so an inverted ranker at
    e.g. 0.45 is penalised the same as one at 0.55-epsilon, matching the
    symmetric reward term ``-|roc_auc-0.5|*roc_auc_weight``).

    Keeping ``roc_auc_penalty`` as the "max penalty" knob preserves the
    prior semantics: old callers that set e.g. 3.0 still get a 3.0 bump at
    auc=0.5. What changed: the penalty now tapers smoothly to 0 as auc
    approaches ``min_roc_auc`` instead of dropping off a step cliff -- this
    avoids jumpy early-stopping curves that could fixate just inside the
    penalty zone when the step was large.
    """
    # Guard against NaN roc_auc/pr_auc (single-class eval set, zero-variance
    # scores, etc.; fast_aucs_per_group_optimized returns NaN in those cases).
    # Without this guard the entire ICE becomes NaN, which silently breaks
    # early-stopping comparisons (NaN > best is always False, so the trainer
    # gets stuck on iteration-1 best instead of failing loud).
    base_loss = (
        brier_loss * brier_loss_weight
        + calibration_mae * mae_weight
        + calibration_std * std_weight
    )
    roc_term = 0.0 if np.isnan(roc_auc) else np.abs(roc_auc - 0.5) * roc_auc_weight
    pr_term = 0.0 if np.isnan(pr_auc) else pr_auc * pr_auc_weight
    res = base_loss - roc_term - pr_term
    threshold_width = min_roc_auc - 0.5
    if threshold_width > 0.0 and not np.isnan(roc_auc):
        deficit = threshold_width - np.abs(roc_auc - 0.5)
        if deficit > 0.0:
            res += (deficit / threshold_width) * roc_auc_penalty
    return res
