"""Batched per-class ICE (Integral Calibration Error) njit kernel + its serial/parallel dispatch.

Split out from ``_classification_report.py`` to keep that file below the 1k-line monolith threshold
(CLAUDE.md: "Monolith split via re-export"). Behaviour preserved bit-for-bit; every moved symbol is
re-exported from ``_classification_report`` so existing
``from mlframe.metrics.classification._classification_report import _batch_per_class_ice_kernel`` (and
the other moved names) imports continue to work.
"""
from __future__ import annotations

import logging
from typing import cast

import numpy as np
import numba
from pyutilz.performance.kernel_tuning.registry import kernel_tuner

logger = logging.getLogger(__name__)


@numba.njit(fastmath=False, cache=True, nogil=True, parallel=True)
def _batch_per_class_ice_kernel(
    y_true_NK: np.ndarray,
    y_pred_NK: np.ndarray,
    desc_idx_NK: np.ndarray,
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
    """Batched per-class ICE: one numba dispatch, prange over K.

    Inlines the work of ``fast_ice_only`` (Brier + calibration binning +
    AUC + ICE combination) so the Python ``for class_id in range(K)``
    loop in ``compute_probabilistic_multiclass_error`` collapses to a
    single Python->numba transition. On 1M-row multiclass workloads
    this drops the Python-glue overhead from ~10-20 ms per call * K
    classes to ~10-20 ms total per call.

    Inputs:
        y_true_NK : (N, K) int8 -- per-class indicator matrix
        y_pred_NK : (N, K) float64 -- per-class predicted probability

    Returns ice_per_class : (K,) float64.

    Bit-exact equivalent of looping ``fast_ice_only`` per class
    (verified against the legacy form in
    ``bench_compute_multiclass_error.py``).

    ``desc_idx_NK`` is the per-class descending-score order (shape (N, K)), computed ONCE by the caller via numpy's
    C ``np.argsort(-y_pred_NK, axis=0)``. numba's own ``np.argsort`` is markedly slower than numpy's (measured 3.6x on
    the AUC portion at N=1M binary; bench_ice_argsort_variants.py), and the AUC/PR walk below only accumulates at
    tie-run boundaries, so it is INVARIANT to the within-tie order -- any valid descending order gives a bit-identical
    ROC/PR AUC. So the sort is hoisted out to numpy and passed in, leaving the kernel a pure single-dispatch reduction.

    bench-attempt-rejected (2026-05-21, c0146 / iter133): fusing the
    Brier + min/max passes (3 N-passes -> 2) saved only 1.04x at
    N=1M/K=3, 1.01-1.02x smaller. Argsort + AUC walk dominates the
    kernel; pre-argsort pass fusion is below the measurable speedup
    floor. Bench: profiling/bench_batch_ice_kernel_pass_fusion.py.

    ``coverage_weight`` (default 0.0, bit-identical to prior behaviour) mirrors
    ``integral_calibration_error_from_metrics``'s coverage term: ``(1 - n_nonempty/nbins) * coverage_weight``
    added to each class's ``base_loss``, using this kernel's own already-computed ``n_nonempty``/``nbins``.
    """
    N = y_true_NK.shape[0]
    K = y_true_NK.shape[1]
    ice_per_class = np.empty(K, dtype=np.float64)

    for k in numba.prange(K):
        y_t = y_true_NK[:, k]
        y_p = y_pred_NK[:, k]

        # ---- Brier loss (mean squared error vs indicator) ----
        s = 0.0
        for i in range(N):
            d = float(y_t[i]) - y_p[i]
            s += d * d
        brier = s / N if N > 0 else 1.0

        # ---- Calibration binning (uniform-strategy, fixed nbins) ----
        # Replicates fast_calibration_binning + calibration_metrics_from_freqs
        # logic inline so the kernel stays single-entry.
        # Seed from the first sample (not a fixed [1.0, 0.0]) -- mirrors
        # _fast_calibration_binning_serial ("gold") so predictions outside [0,1] bin against the actual data
        # range instead of one sentinel never being touched when a class's predicted-probability column is
        # entirely <0 or entirely >1.
        min_val = y_p[0]
        max_val = y_p[0]
        for i in range(N):
            v = y_p[i]
            if v > max_val:
                max_val = v
            if v < min_val:
                min_val = v
        span = max_val - min_val
        pockets_pred = np.zeros(nbins, dtype=np.int64)
        pockets_true = np.zeros(nbins, dtype=np.int64)
        if span > 0:
            multiplier = (nbins - 1) / span
            for i in range(N):
                ind = int(np.floor((y_p[i] - min_val) * multiplier))
                # FP-boundary clamp (same guard as the gold serial kernel): at y_p[i] == max_val this is
                # exactly nbins-1 in exact arithmetic, but floating-point rounding can push it to nbins,
                # which would write out of the pockets_pred/pockets_true bounds under @njit (bounds-checking off).
                if ind < 0:
                    ind = 0
                elif ind >= nbins:
                    ind = nbins - 1
                pockets_pred[ind] += 1
                pockets_true[ind] += y_t[i]
        else:
            for i in range(N):
                pockets_pred[0] += 1
                pockets_true[0] += y_t[i]

        # Collapse to non-empty bins
        n_nonempty = 0
        for b in range(nbins):
            if pockets_pred[b] > 0:
                n_nonempty += 1
        freqs_pred = np.empty(n_nonempty, dtype=np.float64)
        freqs_true = np.empty(n_nonempty, dtype=np.float64)
        hits = np.empty(n_nonempty, dtype=np.int64)
        ptr = 0
        for b in range(nbins):
            if pockets_pred[b] > 0:
                freqs_pred[ptr] = min_val + (b + 0.5) * span / nbins
                freqs_true[ptr] = pockets_true[b] / pockets_pred[b]
                hits[ptr] = pockets_pred[b]
                ptr += 1

        # ---- Calibration MAE / std / coverage ----
        # (calibration_metrics_from_freqs inlined with power-weighting on)
        if n_nonempty > 0:
            # Compute weights (power_weighting alpha=0.8 default of use_weights)
            if use_weights:
                weights = np.empty(n_nonempty, dtype=np.float64)
                for b in range(n_nonempty):
                    weights[b] = hits[b] ** 0.8
                w_sum = 0.0
                for b in range(n_nonempty):
                    w_sum += weights[b]
                if w_sum > 0:
                    for b in range(n_nonempty):
                        weights[b] /= w_sum
                # Weighted MAE
                cal_mae = 0.0
                for b in range(n_nonempty):
                    cal_mae += abs(freqs_pred[b] - freqs_true[b]) * weights[b]
                # Weighted std around weighted-mean MAE
                cal_var = 0.0
                for b in range(n_nonempty):
                    d = abs(freqs_pred[b] - freqs_true[b]) - cal_mae
                    cal_var += d * d * weights[b]
                cal_std = np.sqrt(cal_var)
            else:
                # Unweighted
                cal_mae = 0.0
                for b in range(n_nonempty):
                    cal_mae += abs(freqs_pred[b] - freqs_true[b])
                cal_mae /= n_nonempty
                cal_var = 0.0
                for b in range(n_nonempty):
                    d = abs(freqs_pred[b] - freqs_true[b]) - cal_mae
                    cal_var += d * d
                cal_std = np.sqrt(cal_var / n_nonempty)
        else:
            cal_mae = 1.0
            cal_std = 1.0

        # ---- ROC AUC + PR AUC (fast_numba_aucs body inline) ----
        # Walk through (y_t, y_p) in score-desc order, accumulating TP / FP / current_precision / current_recall as in
        # sklearn.average_precision_score. The descending order is precomputed by the caller (numpy C argsort, hoisted
        # out because numba's argsort is ~3.6x slower); the walk emits only at tie-run boundaries so it is invariant to
        # the within-tie order the sort chose -- bit-identical ROC/PR AUC for any valid descending permutation.
        desc_idx = desc_idx_NK[:, k]
        y_t_sorted = y_t[desc_idx]
        y_p_sorted = y_p[desc_idx]
        # int64 accumulator, not the bare Python literal 0: `y_t_sorted[i]` is an int8 array element
        # (``y_true_NK`` is int8 per this kernel's own docstring), and a plain `0 + int8_scalar` narrows
        # the running total to int8 under numpy's runtime value-based casting -- silently wraps around
        # under normal numba JIT (numba's type inference widens differently and never surfaced this),
        # but raises `OverflowError: Python integer N out of bounds for int8` under NUMBA_DISABLE_JIT=1
        # (pure Python/numpy execution, used to measure @njit body coverage) once a class has >127
        # positives. Any per-class positive count above 127 was a genuine, silent correctness risk in
        # compiled mode too, not just a NUMBA_DISABLE_JIT-only artifact.
        total_pos = np.int64(0)
        for i in range(N):
            total_pos += y_t_sorted[i]
        total_neg = N - total_pos
        if total_pos == 0 or total_neg == 0:
            roc_auc = np.nan
            pr_auc = np.nan
        else:
            # int64, same reason as total_pos above: fps/tps accumulate int8 elements (`yi`/`1 - yi`)
            # and `denom_roc = tps * fps * 2` below would also risk int8 overflow on wide classes.
            last_fps = np.int64(0)
            last_tps = np.int64(0)
            tps = np.int64(0)
            fps = np.int64(0)
            roc_acc = 0.0
            pr_acc = 0.0
            prev_recall = 0.0
            for i in range(N):
                yi = y_t_sorted[i]
                tps += yi
                fps += 1 - yi
                if i == N - 1 or y_p_sorted[i + 1] != y_p_sorted[i]:
                    delta_fps = fps - last_fps
                    sum_tps = last_tps + tps
                    roc_acc += delta_fps * sum_tps
                    last_fps = fps
                    last_tps = tps
                    current_precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
                    current_recall = tps / total_pos
                    delta_recall = current_recall - prev_recall
                    pr_acc += delta_recall * current_precision
                    prev_recall = current_recall
            denom_roc = tps * fps * 2
            if denom_roc > 0:
                roc_auc = roc_acc / denom_roc
            else:
                roc_auc = np.nan
            pr_auc = pr_acc

        # ---- Combine into ICE (integral_calibration_error_from_metrics body) ----
        coverage = n_nonempty / nbins if nbins > 0 else 1.0
        cov_term = (1.0 - coverage) * coverage_weight
        base_loss = brier * brier_loss_weight + cal_mae * mae_weight + cal_std * std_weight + cov_term
        roc_term = 0.0 if np.isnan(roc_auc) else np.abs(roc_auc - 0.5) * roc_auc_weight
        pr_term = 0.0 if np.isnan(pr_auc) else pr_auc * pr_auc_weight
        ice = base_loss - roc_term - pr_term
        threshold_width = min_roc_auc - 0.5
        if threshold_width > 0.0 and not np.isnan(roc_auc):
            deficit = threshold_width - np.abs(roc_auc - 0.5)
            if deficit > 0.0:
                ice += (deficit / threshold_width) * roc_auc_penalty

        ice_per_class[k] = ice

    return ice_per_class


# Serial twin of ``_batch_per_class_ice_kernel`` -- identical body, ``range(K)`` instead of
# ``numba.prange(K)``. ``parallel=True`` only pays off when there is enough TOTAL work (N*K) to
# amortize numba's fixed per-call thread-pool dispatch cost (~10-40ms, independent of how much
# work each thread actually does); K itself is almost always small in real callers (1 for binary,
# rarely above ~10 for multiclass), so the trip count of ``prange(K)`` alone was never the amortizing
# factor the original design assumed. Measured (same-process A/B, warm, best-of-20-200,
# bench_ice_kernel_parallel_vs_serial.py): serial is 10-955x FASTER than parallel at every N<=100k
# tested (any K), and stays faster up to N*K~=500k (ratio ~1.05-1.32x parallel/serial, i.e. parallel
# still loses). Parallel only starts winning once N*K reaches ~4M (0.43-0.66x, i.e. 1.5-2.3x
# speedup) -- see ``_ice_kernel_dispatch`` below for the threshold this calibrates.
@numba.njit(fastmath=False, cache=True, nogil=True, parallel=False)
def _batch_per_class_ice_kernel_serial(
    y_true_NK: np.ndarray,
    y_pred_NK: np.ndarray,
    desc_idx_NK: np.ndarray,
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
    """Serial twin of ``_batch_per_class_ice_kernel`` -- see that function's docstring for the algorithm; see this
    module's ``_ice_kernel_dispatch`` for why a serial variant exists and when it is selected."""
    N = y_true_NK.shape[0]
    K = y_true_NK.shape[1]
    ice_per_class = np.empty(K, dtype=np.float64)

    for k in range(K):
        y_t = y_true_NK[:, k]
        y_p = y_pred_NK[:, k]

        # ---- Brier loss (mean squared error vs indicator) ----
        s = 0.0
        for i in range(N):
            d = float(y_t[i]) - y_p[i]
            s += d * d
        brier = s / N if N > 0 else 1.0

        # ---- Calibration binning (uniform-strategy, fixed nbins) ----
        min_val = y_p[0]
        max_val = y_p[0]
        for i in range(N):
            v = y_p[i]
            if v > max_val:
                max_val = v
            if v < min_val:
                min_val = v
        span = max_val - min_val
        pockets_pred = np.zeros(nbins, dtype=np.int64)
        pockets_true = np.zeros(nbins, dtype=np.int64)
        if span > 0:
            multiplier = (nbins - 1) / span
            for i in range(N):
                ind = int(np.floor((y_p[i] - min_val) * multiplier))
                if ind < 0:
                    ind = 0
                elif ind >= nbins:
                    ind = nbins - 1
                pockets_pred[ind] += 1
                pockets_true[ind] += y_t[i]
        else:
            for i in range(N):
                pockets_pred[0] += 1
                pockets_true[0] += y_t[i]

        n_nonempty = 0
        for b in range(nbins):
            if pockets_pred[b] > 0:
                n_nonempty += 1
        freqs_pred = np.empty(n_nonempty, dtype=np.float64)
        freqs_true = np.empty(n_nonempty, dtype=np.float64)
        hits = np.empty(n_nonempty, dtype=np.int64)
        ptr = 0
        for b in range(nbins):
            if pockets_pred[b] > 0:
                freqs_pred[ptr] = min_val + (b + 0.5) * span / nbins
                freqs_true[ptr] = pockets_true[b] / pockets_pred[b]
                hits[ptr] = pockets_pred[b]
                ptr += 1

        if n_nonempty > 0:
            if use_weights:
                weights = np.empty(n_nonempty, dtype=np.float64)
                for b in range(n_nonempty):
                    weights[b] = hits[b] ** 0.8
                w_sum = 0.0
                for b in range(n_nonempty):
                    w_sum += weights[b]
                if w_sum > 0:
                    for b in range(n_nonempty):
                        weights[b] /= w_sum
                cal_mae = 0.0
                for b in range(n_nonempty):
                    cal_mae += abs(freqs_pred[b] - freqs_true[b]) * weights[b]
                cal_var = 0.0
                for b in range(n_nonempty):
                    d = abs(freqs_pred[b] - freqs_true[b]) - cal_mae
                    cal_var += d * d * weights[b]
                cal_std = np.sqrt(cal_var)
            else:
                cal_mae = 0.0
                for b in range(n_nonempty):
                    cal_mae += abs(freqs_pred[b] - freqs_true[b])
                cal_mae /= n_nonempty
                cal_var = 0.0
                for b in range(n_nonempty):
                    d = abs(freqs_pred[b] - freqs_true[b]) - cal_mae
                    cal_var += d * d
                cal_std = np.sqrt(cal_var / n_nonempty)
        else:
            cal_mae = 1.0
            cal_std = 1.0

        desc_idx = desc_idx_NK[:, k]
        y_t_sorted = y_t[desc_idx]
        y_p_sorted = y_p[desc_idx]
        total_pos = np.int64(0)
        for i in range(N):
            total_pos += y_t_sorted[i]
        total_neg = N - total_pos
        if total_pos == 0 or total_neg == 0:
            roc_auc = np.nan
            pr_auc = np.nan
        else:
            last_fps = np.int64(0)
            last_tps = np.int64(0)
            tps = np.int64(0)
            fps = np.int64(0)
            roc_acc = 0.0
            pr_acc = 0.0
            prev_recall = 0.0
            for i in range(N):
                yi = y_t_sorted[i]
                tps += yi
                fps += 1 - yi
                if i == N - 1 or y_p_sorted[i + 1] != y_p_sorted[i]:
                    delta_fps = fps - last_fps
                    sum_tps = last_tps + tps
                    roc_acc += delta_fps * sum_tps
                    last_fps = fps
                    last_tps = tps
                    current_precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
                    current_recall = tps / total_pos
                    delta_recall = current_recall - prev_recall
                    pr_acc += delta_recall * current_precision
                    prev_recall = current_recall
            denom_roc = tps * fps * 2
            if denom_roc > 0:
                roc_auc = roc_acc / denom_roc
            else:
                roc_auc = np.nan
            pr_auc = pr_acc

        coverage = n_nonempty / nbins if nbins > 0 else 1.0
        cov_term = (1.0 - coverage) * coverage_weight
        base_loss = brier * brier_loss_weight + cal_mae * mae_weight + cal_std * std_weight + cov_term
        roc_term = 0.0 if np.isnan(roc_auc) else np.abs(roc_auc - 0.5) * roc_auc_weight
        pr_term = 0.0 if np.isnan(pr_auc) else pr_auc * pr_auc_weight
        ice = base_loss - roc_term - pr_term
        threshold_width = min_roc_auc - 0.5
        if threshold_width > 0.0 and not np.isnan(roc_auc):
            deficit = threshold_width - np.abs(roc_auc - 0.5)
            if deficit > 0.0:
                ice += (deficit / threshold_width) * roc_auc_penalty

        ice_per_class[k] = ice

    return ice_per_class


# Per-host serial/parallel crossover via the canonical kernel_tuning_cache (NO hardcoded threshold --
# feedback_use_kernel_tuning_cache_for_gpu / feedback_fastest_default_with_dispatch). Dev-box measurement
# found the confirmed-loss region at N*K<=500k (parallel 1.05-1.32x SLOWER) and the confirmed-win region
# at N*K~=4M (parallel 1.5-2.3x faster) -- the fallback threshold below (used only pre-sweep / on tuner
# failure) is deliberately close to the loss side for that reason, but the REAL per-host decision comes
# from the tuner's measured sweep, not this constant.
_ICE_KERNEL_SWEEP_N = [1_000, 100_000, 2_000_000]
_ICE_KERNEL_SWEEP_K = [1, 8, 20]
_ICE_KERNEL_SALT = 1
_ICE_KWARGS_TUPLE = (10, True, 3.0, 2.0, 0.8, 1.5, 0.1, 0.54, 0.0, 0.0)


def _make_ice_kernel_inputs(dims: dict) -> tuple:
    """(y_true_NK, y_pred_NK, desc_idx_NK, *_ICE_KWARGS_TUPLE) at the sweep's (n, k) cell, with realistic
    tied-score density (quantized model outputs)."""
    n, k = int(dims["n"]), int(dims["k"])
    rng = np.random.default_rng(0)
    y_true = (rng.random((n, k)) > 0.5).astype(np.int8)
    y_pred = np.round(rng.random((n, k)), 3)
    desc_idx = np.ascontiguousarray(np.argsort(-y_pred, axis=0).astype(np.int64))
    return (y_true, y_pred, desc_idx, *_ICE_KWARGS_TUPLE)


def _run_ice_kernel_sweep() -> list:
    """Serial-vs-parallel wall-clock sweep over the (n, k) grid -> kernel_tuning_cache regions."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {
        "serial": lambda *a: _batch_per_class_ice_kernel_serial(*a),
        "parallel": lambda *a: _batch_per_class_ice_kernel(*a),
    }
    return cast(list, sweep_backend_grid(
        variants,
        {"n": _ICE_KERNEL_SWEEP_N, "k": _ICE_KERNEL_SWEEP_K},
        _make_ice_kernel_inputs,
        reference="serial", repeats=5, equiv_atol=0.0, equiv_rtol=0.0,
    ))


def _ice_kernel_fallback_choice(n: int, k: int) -> str:
    """Pre-sweep / tuner-failure fallback: parallel above the dev-box-measured N*K crossover (see the
    module comment above for the confirmed-loss/confirmed-win bracket)."""
    return "parallel" if int(n) * int(k) >= 2_000_000 else "serial"


_ICE_KERNEL_PARALLELISM_SPEC = kernel_tuner(
    kernel_name="ice_kernel_parallelism",
    variant_fns=(_batch_per_class_ice_kernel_serial, _batch_per_class_ice_kernel),
    tuner=_run_ice_kernel_sweep,
    axes={"n": _ICE_KERNEL_SWEEP_N, "k": _ICE_KERNEL_SWEEP_K},
    fallback=_ice_kernel_fallback_choice,
    gpu_capable=False,
    salt=_ICE_KERNEL_SALT,
    cli_label="ice_kernel_parallelism",
)


def _ice_kernel_dispatch(
    y_true_NK: np.ndarray,
    y_pred_NK: np.ndarray,
    desc_idx_NK: np.ndarray,
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
    """Dispatch to the serial or parallel ``_batch_per_class_ice_kernel*`` variant via the per-host
    kernel_tuning_cache (``_ICE_KERNEL_PARALLELISM_SPEC``). Bit-identical output either way: each class's
    ICE is computed independently of which thread (if any) runs it."""
    n, k = y_true_NK.shape[0], y_true_NK.shape[1]
    try:
        choice = _ICE_KERNEL_PARALLELISM_SPEC.choose(n=int(n), k=int(k))
    except Exception as e:
        logger.debug("ice_kernel_parallelism choose() failed, using the size-based fallback: %s", e)
        choice = _ice_kernel_fallback_choice(int(n), int(k))
    fn = _batch_per_class_ice_kernel if choice == "parallel" else _batch_per_class_ice_kernel_serial
    return np.asarray(
        fn(
            y_true_NK, y_pred_NK, desc_idx_NK, nbins, use_weights,
            mae_weight, std_weight, brier_loss_weight, roc_auc_weight, pr_auc_weight,
            min_roc_auc, roc_auc_penalty, coverage_weight,
        )
    )
