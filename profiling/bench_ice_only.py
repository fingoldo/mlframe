"""Microbench: current fast_calibration_report vs a proposed ICE-only
fastpath for compute_probabilistic_multiclass_error's fairness fan-out.

Why this bench exists
---------------------
The 2026-04-19 profile_metrics_blocks run (230s total) showed
``fast_calibration_report`` at 1708 calls / 43.6 s cumulative. The
culprit: ``compute_probabilistic_multiclass_error`` (1700 calls from
compute_fairness_metrics' per-bin fan-out) uses only ``ice`` or
``brier_loss`` from the return tuple but pays for ``fast_log_loss``,
``compute_pr_recall_f1_metrics``, and the title string every call.

This script measures, at the per-call level, what fraction of
``fast_calibration_report`` time is spent on the discarded work, so we
know whether the proposed bypass is worth the extra code path.

Usage:
    python -m mlframe.profiling.bench_ice_only
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mlframe.metrics import (
    fast_calibration_report,
    fast_brier_score_loss,
    fast_calibration_binning,
    calibration_metrics_from_freqs,
    fast_aucs_per_group_optimized,
    integral_calibration_error_from_metrics,
)
from mlframe.metrics import prewarm_numba_cache


def fast_ice_only(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nbins: int = 10,
    use_weights: bool = True,
    **ice_kwargs,
) -> float:
    """Proposed lite path: compute ICE without log_loss/PR/F1/title.

    Matches fast_calibration_report's ICE computation exactly — same
    helpers, same ICE formula, same kwargs plumbing.
    """
    if len(y_true) == 0:
        return 1.0

    brier_loss = fast_brier_score_loss(y_true=y_true, y_prob=y_pred)
    freqs_predicted, freqs_true, hits = fast_calibration_binning(
        y_true=y_true, y_pred=y_pred, nbins=nbins
    )
    cal_mae, cal_std, cal_cov = calibration_metrics_from_freqs(
        freqs_predicted=freqs_predicted,
        freqs_true=freqs_true,
        hits=hits,
        nbins=nbins,
        array_size=len(y_true),
        use_weights=use_weights,
    )
    roc_auc, pr_auc, _ = fast_aucs_per_group_optimized(
        y_true=y_true, y_score=y_pred, group_ids=None
    )
    ice = integral_calibration_error_from_metrics(
        calibration_mae=cal_mae,
        calibration_std=cal_std,
        calibration_coverage=cal_cov,
        brier_loss=brier_loss,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        **ice_kwargs,
    )
    return ice


def bench(n_samples: int, n_iters: int) -> None:
    rng = np.random.default_rng(42)
    # Imbalanced binary — closer to prod fairness-bin data
    y_true = (rng.random(n_samples) < 0.3).astype(np.int8)
    y_pred = np.clip(
        0.3 + 0.25 * y_true + rng.normal(0, 0.2, n_samples), 0.01, 0.99
    ).astype(np.float64)

    print(f"\n=== n_samples={n_samples}, n_iters={n_iters} ===")

    # Warm both paths
    _ = fast_calibration_report(y_true, y_pred, show_plots=False)
    _ = fast_ice_only(y_true, y_pred)

    # Bench full
    t0 = time.perf_counter()
    ice_full_last = None
    for _ in range(n_iters):
        _, _, _, _, _, _, ice_full, *_ = fast_calibration_report(
            y_true=y_true, y_pred=y_pred, use_weights=True, nbins=10, show_plots=False,
        )
        ice_full_last = ice_full
    t_full = time.perf_counter() - t0

    # Bench lite
    t0 = time.perf_counter()
    ice_lite_last = None
    for _ in range(n_iters):
        ice_lite_last = fast_ice_only(
            y_true=y_true, y_pred=y_pred, use_weights=True, nbins=10,
        )
    t_lite = time.perf_counter() - t0

    # Correctness
    drift = abs(ice_full_last - ice_lite_last)
    print(f"  full: {t_full*1000:7.1f} ms ({t_full/n_iters*1e6:6.1f} µs/call)")
    print(f"  lite: {t_lite*1000:7.1f} ms ({t_lite/n_iters*1e6:6.1f} µs/call)")
    print(f"  speedup: {t_full/t_lite:.2f}×")
    print(f"  ICE match:  full={ice_full_last:.10f}  lite={ice_lite_last:.10f}  drift={drift:.2e}")
    assert drift < 1e-9, f"ICE value drifted by {drift:.2e} — lite path is NOT equivalent"
    print("  ✓ equivalence asserted (drift < 1e-9)")


if __name__ == "__main__":
    print("Prewarming numba cache...")
    prewarm_numba_cache()
    print("Done.\n")

    # Prod fairness bins: ~1000-5000 rows per bin typically
    for n in (500, 2_000, 10_000, 50_000):
        bench(n_samples=n, n_iters=500)

    # Large single-shot (like the 8 direct report_probabilistic_model_perf calls)
    bench(n_samples=100_000, n_iters=50)
