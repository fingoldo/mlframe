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
    # Coverage = fraction of the nbins grid that actually carries predictions. This is a stable structural
    # quantity (a populated bin stays populated under tiny score perturbations), unlike the prior measure
    # (distinct values of ``round(freqs_predicted, prec)`` / nbins), which was a float-rounding artifact: a
    # perturbation of ~10**-prec across a bin boundary flipped two pockets into one and jumped the value.
    populated = 0
    for _b in range(len(hits)):
        if hits[_b] > 0:
            populated += 1
    calibration_coverage = populated / nbins
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
    """Plug-in ECE plus Murphy 1973 Brier score decomposition on a FIXED [0,1] grid.

    Returns ``(ece, reliability, resolution, uncertainty, brier_binned)``.

    ECE = sum_k (n_k / N) * |p_mean_k - acc_k|
    BinnedBrier = REL - RES + UNC      (exact identity by construction)
        REL = sum_k (n_k / N) * (p_mean_k - acc_k)^2
        RES = sum_k (n_k / N) * (acc_k - base_rate)^2
        UNC = base_rate * (1 - base_rate)
    where p_mean_k is the *mean predicted probability* in bin k (NOT the bin centre), acc_k is the observed
    positive rate in bin k, base_rate is the overall positive rate.

    These are the PLUG-IN estimators: ``acc_k`` is a noisy finite-sample bin mean, so even a perfectly calibrated
    model reports a positive ECE/REL noise floor that grows with nbins. The bias-corrected twins
    ``compute_ece_debiased`` / ``compute_brier_decomposition_debiased`` are the DEFAULT headline path in
    ``fast_calibration_report`` (``ece_debiased`` / ``brier_debiased`` default True); this kernel remains the
    explicit ``ece_debiased=False`` opt-out and the exact-identity reference.

    Binning uses a FIXED ``[0, 1]`` equal-width grid (probabilities live on ``[0,1]``), NOT the old data-adaptive
    ``[min(y_pred), max(y_pred)]`` grid. The data-adaptive grid keyed on the sample min/max made the bin
    boundaries -- and hence ECE/REL/RES -- non-comparable across datasets and across bootstrap/per-window
    resamples (each draw had different extrema), silently biasing those comparisons. The fixed grid restores
    comparability. Per-bin p_mean (not bin centre) keeps the Murphy identity ``BinnedBrier == REL - RES + UNC``
    exact to fp precision.

    Cross-module ECE note: this ECE shares the EQUAL-WIDTH-[0,1] partition with
    ``calibration/policy._ece_score`` but still differs from the EQUAL-MASS (equal-count) ECE in
    ``calibration/quality.estimate_calibration_quality_binned``, which argsorts predictions into
    equal-population pockets. Equal-mass and equal-width ECE are NOT comparable on the same inputs --
    compare ECE only within one binning scheme.

    Returns 1.0/1.0/0.0/0.0/1.0 on empty input - mirrors degenerate handling elsewhere in the calibration pipeline.
    """
    n = len(y_true)
    if n == 0:
        return 1.0, 1.0, 0.0, 0.0, 1.0

    base_rate = 0.0
    for i in range(n):
        base_rate += y_true[i]
    base_rate /= n

    pred_sum = np.zeros(nbins, dtype=np.float64)
    true_sum = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)

    # Fixed [0,1] grid: bin index = floor(p * nbins), clamped so p == 1.0 lands in the last bin. Comparable across
    # datasets / resamples because the boundaries do not depend on the sample's observed min/max.
    for i in range(n):
        p = y_pred[i]
        ind = int(floor(p * nbins))
        if ind >= nbins:
            ind = nbins - 1
        elif ind < 0:
            ind = 0
        counts[ind] += 1
        pred_sum[ind] += p
        true_sum[ind] += y_true[i]

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

    Uses the SAME fixed ``[0, 1]`` equal-width binning grid as ``compute_ece_and_brier_decomposition`` so the
    debiased and plug-in estimates are directly comparable AND comparable across datasets / resamples (the grid
    does not depend on the sample's observed min/max). Returns 1.0 on empty input (degenerate handling mirrors
    the rest of the calibration pipeline).
    """
    n = len(y_true)
    if n == 0:
        return 1.0

    pred_sum = np.zeros(nbins, dtype=np.float64)
    true_sum = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)

    for i in range(n):
        p = y_pred[i]
        ind = int(floor(p * nbins))
        if ind >= nbins:
            ind = nbins - 1
        elif ind < 0:
            ind = 0
        counts[ind] += 1
        pred_sum[ind] += p
        true_sum[ind] += y_true[i]

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
def compute_brier_decomposition_debiased(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nbins: int,
):
    """Bias-corrected Murphy/Brier reliability + resolution (Broecker 2009).

    The plug-in binned Brier decomposition terms ``REL = sum_b w_b (conf_b - acc_b)^2`` and
    ``RES = sum_b w_b (acc_b - base_rate)^2`` are POSITIVELY biased by the same per-bin Bernoulli sampling
    noise that inflates the plug-in ECE: ``acc_b`` is a noisy estimate of the bin's true positive rate, so
    ``E[(conf_b - acc_b)^2] = (conf_b - true_acc_b)^2 + Var(acc_b)`` and likewise the squared centred term in
    RES carries a ``+Var(acc_b)`` (the bias enters with the SAME sign because both terms square a quantity
    that contains ``acc_b``). A perfectly calibrated model has true REL 0 yet the plug-in REL is strictly
    positive and grows with nbins -- exactly the headline-overstatement problem qual-1 fixed for ECE.

    Correction (Broecker 2009, "Reliability, sufficiency, and the decomposition of proper scores", QJRMS):
    BOTH terms carry the same per-bin noise inflation, so the unbiased estimates subtract the within-bin variance
    ``Var(acc_b) = acc_b*(1-acc_b)/(n_b-1)`` from EACH: ``REL_db = REL_plugin - sum_b w_b Var(acc_b)`` and
    ``RES_db = RES_plugin - sum_b w_b Var(acc_b)``. Because the SAME amount leaves both, ``REL_db - RES_db ==
    REL_plugin - RES_plugin`` and the Murphy identity ``BinnedBrier = REL - RES + UNC`` (and the BinnedBrier value
    itself) is preserved EXACTLY -- only the split between the (now-unbiased) reliability and resolution shifts.
    REL is clamped at 0 (true reliability is non-negative); the clamp can break the exact REL-RES cancellation in
    the rare bins where the plug-in REL term falls below its noise floor, which is the deliberate accuracy choice.
    Bins with n_b<2 have no variance estimate and contribute their raw squared term.

    Returns ``(reliability_debiased, resolution_debiased, uncertainty, brier_binned)``. ``uncertainty`` is unchanged
    from the plug-in; ``brier_binned`` is recomputed from the debiased terms (equals the plug-in BinnedBrier when no
    REL bin clamps). Uses the SAME fixed ``[0, 1]`` equal-width grid as ``compute_ece_and_brier_decomposition`` so the
    debiased and plug-in decompositions are directly comparable AND comparable across datasets / resamples. Returns
    1.0/0.0/0.0/1.0 on empty input.
    """
    n = len(y_true)
    if n == 0:
        return 1.0, 0.0, 0.0, 1.0

    base_rate = 0.0
    for i in range(n):
        base_rate += y_true[i]
    base_rate /= n

    pred_sum = np.zeros(nbins, dtype=np.float64)
    true_sum = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)

    for i in range(n):
        p = y_pred[i]
        ind = int(floor(p * nbins))
        if ind >= nbins:
            ind = nbins - 1
        elif ind < 0:
            ind = 0
        counts[ind] += 1
        pred_sum[ind] += p
        true_sum[ind] += y_true[i]

    reliability = 0.0
    resolution = 0.0
    inv_n = 1.0 / n
    for k in range(nbins):
        nk = counts[k]
        if nk == 0:
            continue
        w = nk * inv_n
        p_mean = pred_sum[k] / nk
        acc = true_sum[k] / nk
        diff = p_mean - acc
        rel_term = diff * diff
        res_term = (acc - base_rate) ** 2
        if nk >= 2:
            var_acc = acc * (1.0 - acc) / (nk - 1)
            rel_term -= var_acc
            if rel_term < 0.0:
                rel_term = 0.0
            res_term -= var_acc
        reliability += w * rel_term
        resolution += w * res_term
    uncertainty = base_rate * (1.0 - base_rate)
    brier_binned = reliability - resolution + uncertainty
    return reliability, resolution, uncertainty, brier_binned


@numba.njit(**NUMBA_NJIT_PARAMS)
def compute_ece_brier_full_and_debiased(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nbins: int,
):
    """Fused single-pass plug-in + debiased ECE / Brier decomposition on the shared fixed ``[0,1]`` grid.

    Returns ``(ece_plugin, rel_plugin, res_plugin, unc, brier_binned_plugin,
               ece_debiased, rel_debiased, res_debiased, brier_binned_debiased)``.

    The default ``fast_calibration_report`` headline path needs BOTH the plug-in decomposition
    (``compute_ece_and_brier_decomposition``) AND the two debiased estimators (``compute_ece_debiased``,
    ``compute_brier_decomposition_debiased``). Those three kernels each rebuild the IDENTICAL
    ``counts``/``pred_sum``/``true_sum`` histogram from the same ``y_true``/``y_pred`` -- three O(n) binning
    passes where one suffices. This kernel bins ONCE and emits every reduction; it is bit-identical to calling
    the three kernels separately BY CONSTRUCTION (same binning arithmetic, same per-bin formulas, same
    ``base_rate``/``inv_n``). The report dispatches here when both debiased flags are on; the standalone
    kernels remain for the opt-out paths and other callers.

    Returns the empty-input sentinels of all three kernels on ``n == 0``.
    """
    n = len(y_true)
    if n == 0:
        return 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0

    base_rate = 0.0
    for i in range(n):
        base_rate += y_true[i]
    base_rate /= n

    pred_sum = np.zeros(nbins, dtype=np.float64)
    true_sum = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)

    for i in range(n):
        p = y_pred[i]
        ind = int(floor(p * nbins))
        if ind >= nbins:
            ind = nbins - 1
        elif ind < 0:
            ind = 0
        counts[ind] += 1
        pred_sum[ind] += p
        true_sum[ind] += y_true[i]

    ece = 0.0
    reliability = 0.0
    resolution = 0.0
    ece_db = 0.0
    rel_db = 0.0
    res_db = 0.0
    inv_n = 1.0 / n
    for k in range(nbins):
        nk = counts[k]
        if nk == 0:
            continue
        w = nk * inv_n
        p_mean = pred_sum[k] / nk
        acc = true_sum[k] / nk
        diff = p_mean - acc
        # Plug-in terms (compute_ece_and_brier_decomposition).
        ece += w * abs(diff)
        reliability += w * diff * diff
        res_term_plugin = (acc - base_rate) ** 2
        resolution += w * res_term_plugin
        # Debiased terms (compute_ece_debiased + compute_brier_decomposition_debiased).
        rel_term = diff * diff
        res_term = res_term_plugin
        if nk >= 2:
            var_acc = acc * (1.0 - acc) / (nk - 1)
            # ECE debiased.
            corrected = diff * diff - var_acc
            if corrected < 0.0:
                corrected = 0.0
            ece_db += w * np.sqrt(corrected)
            # Brier REL/RES debiased.
            rel_term -= var_acc
            if rel_term < 0.0:
                rel_term = 0.0
            res_term -= var_acc
        else:
            ece_db += w * abs(diff)
        rel_db += w * rel_term
        res_db += w * res_term
    uncertainty = base_rate * (1.0 - base_rate)
    brier_binned = reliability - resolution + uncertainty
    brier_binned_db = rel_db - res_db + uncertainty
    return (
        ece, reliability, resolution, uncertainty, brier_binned,
        ece_db, rel_db, res_db, brier_binned_db,
    )


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
