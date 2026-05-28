"""Bench: fused extended metric blocks vs separate kernel calls.

Run::

    python -m mlframe.metrics._benchmarks.bench_extended_metric_blocks

Confirms the per-block speedup numbers cited in the docstrings of
``fast_regression_metrics_block_extended`` / the binary confusion and
probability blocks / the multilabel block. Numbers feed directly into
the in-source perf comments per the project rule
"document the achieved performance of the chosen variant".

The bench warms each kernel once (numba JIT) before timing, runs three
size regimes (10k / 500k / 5M for regression+binary; 10k / 100k / 1M
for multilabel where K=20 multiplies the cost), and prints a small
table per block.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from mlframe.metrics.core import (
    # Regression block
    fast_regression_metrics_block_extended,
    fast_mean_absolute_error,
    fast_root_mean_squared_error,
    fast_max_error,
    fast_r2_score,
    fast_mean_bias_error,
    fast_mape_mean,
    fast_smape,
    fast_wmape,
    fast_cv_rmse,
    fast_nash_sutcliffe,
    fast_explained_variance,
    fast_pearson_corr,
    # Binary confusion block
    fast_binary_confusion_metrics_block,
    matthews_corrcoef_binary,
    cohen_kappa_binary,
    balanced_accuracy_binary,
    g_mean_binary,
    specificity_npv_fpr_fnr,
    f_beta_score,
    # Binary probability block
    fast_binary_probability_metrics_block,
    brier_skill_score,
    spiegelhalter_z,
    fast_log_loss_binary,
    fast_brier_score_loss,
    # Multilabel block
    fast_multilabel_classification_metrics_block,
    hamming_loss,
    subset_accuracy,
    jaccard_score_multilabel,
    multilabel_f1_macro,
    multilabel_f1_micro,
    multilabel_f1_weighted,
)


def _timeit(fn: Callable, *args, n_repeat: int = 5) -> float:
    """Median of n_repeat runs in milliseconds. Median dodges the GC-
    spike outliers a single timed run would catch."""
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3


def _bench_regression_block():
    print("\n=== Regression extended block ===")
    print(f"{'N':>10} {'sep ms':>10} {'fused ms':>10} {'speedup':>10}")
    rng = np.random.default_rng(0)
    for N in (10_000, 500_000, 5_000_000):
        y = rng.standard_normal(N).astype(np.float64) * 10 + 100
        p = y + rng.standard_normal(N).astype(np.float64) * 2

        # Warm numba caches.
        fast_regression_metrics_block_extended(y[:64], p[:64])
        fast_mean_absolute_error(y[:64], p[:64])

        def separate():
            fast_mean_absolute_error(y, p)
            fast_root_mean_squared_error(y, p)
            fast_max_error(y, p)
            fast_r2_score(y, p)
            fast_mean_bias_error(y, p)
            fast_mape_mean(y, p)
            fast_smape(y, p)
            fast_wmape(y, p)
            fast_cv_rmse(y, p)
            fast_nash_sutcliffe(y, p)
            fast_explained_variance(y, p)
            fast_pearson_corr(y, p)

        def fused():
            fast_regression_metrics_block_extended(y, p)

        t_sep = _timeit(separate)
        t_fus = _timeit(fused)
        print(f"{N:>10_} {t_sep:>10.2f} {t_fus:>10.2f} {t_sep/t_fus:>9.2f}x")


def _bench_binary_confusion_block():
    print("\n=== Binary confusion block ===")
    print(f"{'N':>10} {'sep ms':>10} {'fused ms':>10} {'speedup':>10}")
    rng = np.random.default_rng(1)
    for N in (10_000, 500_000, 5_000_000):
        y_true = (rng.uniform(size=N) > 0.7).astype(np.int64)
        y_pred = (rng.uniform(size=N) > 0.7).astype(np.int64)
        # warm
        fast_binary_confusion_metrics_block(y_true[:64], y_pred[:64])
        matthews_corrcoef_binary(y_true[:64], y_pred[:64])

        def separate():
            matthews_corrcoef_binary(y_true, y_pred)
            cohen_kappa_binary(y_true, y_pred)
            balanced_accuracy_binary(y_true, y_pred)
            g_mean_binary(y_true, y_pred)
            specificity_npv_fpr_fnr(y_true, y_pred)
            f_beta_score(y_true, y_pred, beta=1.0)
            f_beta_score(y_true, y_pred, beta=0.5)
            f_beta_score(y_true, y_pred, beta=2.0)
            # precision/recall via specificity_npv_fpr_fnr already; add accuracy
            (y_true == y_pred).mean()

        def fused():
            fast_binary_confusion_metrics_block(y_true, y_pred)

        t_sep = _timeit(separate)
        t_fus = _timeit(fused)
        print(f"{N:>10_} {t_sep:>10.2f} {t_fus:>10.2f} {t_sep/t_fus:>9.2f}x")


def _bench_binary_probability_block():
    print("\n=== Binary probability block ===")
    print(f"{'N':>10} {'sep ms':>10} {'fused ms':>10} {'speedup':>10}")
    rng = np.random.default_rng(2)
    for N in (10_000, 500_000, 5_000_000):
        y_true = (rng.uniform(size=N) > 0.7).astype(np.int64)
        y_score = np.clip(0.3 + 0.4 * y_true + rng.normal(0, 0.1, N), 0.001, 0.999)
        # warm
        fast_binary_probability_metrics_block(y_true[:64], y_score[:64])
        fast_log_loss_binary(y_true[:64], y_score[:64])

        def separate():
            fast_brier_score_loss(y_true, y_score)
            fast_log_loss_binary(y_true, y_score)
            brier_skill_score(y_true, y_score)
            spiegelhalter_z(y_true, y_score)

        def fused():
            fast_binary_probability_metrics_block(y_true, y_score)

        t_sep = _timeit(separate)
        t_fus = _timeit(fused)
        print(f"{N:>10_} {t_sep:>10.2f} {t_fus:>10.2f} {t_sep/t_fus:>9.2f}x")


def _bench_multilabel_block():
    print("\n=== Multilabel block (K=20) ===")
    print(f"{'N':>10} {'sep ms':>10} {'fused ms':>10} {'speedup':>10}")
    rng = np.random.default_rng(3)
    K = 20
    for N in (10_000, 100_000, 1_000_000):
        y_true = (rng.uniform(size=(N, K)) > 0.7).astype(np.int64)
        y_pred = (rng.uniform(size=(N, K)) > 0.7).astype(np.int64)
        # warm
        fast_multilabel_classification_metrics_block(y_true[:64], y_pred[:64])
        hamming_loss(y_true[:64], y_pred[:64])

        def separate():
            hamming_loss(y_true, y_pred)
            subset_accuracy(y_true, y_pred)
            jaccard_score_multilabel(y_true, y_pred)
            multilabel_f1_macro(y_true, y_pred)
            multilabel_f1_micro(y_true, y_pred)
            multilabel_f1_weighted(y_true, y_pred)

        def fused():
            fast_multilabel_classification_metrics_block(y_true, y_pred)

        t_sep = _timeit(separate)
        t_fus = _timeit(fused)
        print(f"{N:>10_} {t_sep:>10.2f} {t_fus:>10.2f} {t_sep/t_fus:>9.2f}x")


def main():
    _bench_regression_block()
    _bench_binary_confusion_block()
    _bench_binary_probability_block()
    _bench_multilabel_block()


if __name__ == "__main__":
    main()
