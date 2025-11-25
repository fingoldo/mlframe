"""
Profile fast_calibration_report function to identify bottlenecks.

Usage:
    python -m mlframe.profiling.profile_calibration_report
"""

import time
import cProfile
import pstats
import io
import numpy as np
from line_profiler import LineProfiler

from mlframe.metrics import (
    fast_calibration_report,
    prewarm_numba_cache,
    brier_score_loss,
    fast_calibration_binning,
    calibration_metrics_from_freqs,
    fast_aucs_per_group_optimized,
    integral_calibration_error_from_metrics,
    fast_log_loss,
    compute_pr_recall_f1_metrics,
)


def create_test_data(n: int = 100_000):
    """Create synthetic binary classification data."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, n).astype(np.float64)
    y_pred = np.clip(y_true + np.random.randn(n) * 0.3, 0.01, 0.99)
    return y_true, y_pred


def profile_components(y_true, y_pred, n_iterations=100):
    """Profile individual components of fast_calibration_report."""
    print(f"\nProfiling individual components ({n_iterations} iterations, {len(y_true):,} samples):")
    print("=" * 70)

    # Component timings
    components = {}

    # 1. brier_score_loss
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = brier_score_loss(y_true, y_pred)
    components['brier_score_loss'] = time.perf_counter() - start

    # 2. fast_calibration_binning
    start = time.perf_counter()
    for _ in range(n_iterations):
        freqs_pred, freqs_true, hits = fast_calibration_binning(y_true, y_pred, nbins=10)
    components['fast_calibration_binning'] = time.perf_counter() - start

    # 3. calibration_metrics_from_freqs (use cached results)
    freqs_pred, freqs_true, hits = fast_calibration_binning(y_true, y_pred, nbins=10)
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = calibration_metrics_from_freqs(freqs_pred, freqs_true, hits, nbins=10, array_size=len(y_true))
    components['calibration_metrics_from_freqs'] = time.perf_counter() - start

    # 4. fast_aucs_per_group_optimized
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = fast_aucs_per_group_optimized(y_true, y_pred, group_ids=None)
    components['fast_aucs_per_group_optimized'] = time.perf_counter() - start

    # 5. integral_calibration_error_from_metrics
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = integral_calibration_error_from_metrics(0.01, 0.01, 0.9, 0.25, 0.7, 0.7)
    components['integral_calibration_error_from_metrics'] = time.perf_counter() - start

    # 6. fast_log_loss
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = fast_log_loss(y_true, y_pred)
    components['fast_log_loss'] = time.perf_counter() - start

    # 7. compute_pr_recall_f1_metrics
    y_pred_binary = (y_pred >= 0.5).astype(np.int64)
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = compute_pr_recall_f1_metrics(y_true.astype(np.int64), y_pred_binary)
    components['compute_pr_recall_f1_metrics'] = time.perf_counter() - start

    # 8. np.argsort (standalone)
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = np.argsort(y_pred)[::-1]
    components['np.argsort'] = time.perf_counter() - start

    # 9. Full fast_calibration_report
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = fast_calibration_report(y_true, y_pred, show_plots=False)
    components['fast_calibration_report (full)'] = time.perf_counter() - start

    # Sort by time and print
    sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)
    total = components['fast_calibration_report (full)']

    for name, t in sorted_components:
        pct = (t / total) * 100 if 'full' not in name else 100
        per_call = t / n_iterations * 1000  # ms
        print(f"{name:40s}: {t:7.3f}s ({pct:5.1f}%) - {per_call:.3f}ms/call")

    return components


def profile_with_cprofile(y_true, y_pred, n_iterations=100):
    """Profile with cProfile for detailed breakdown."""
    print(f"\n\ncProfile analysis ({n_iterations} iterations):")
    print("=" * 70)

    def run_iterations():
        for _ in range(n_iterations):
            fast_calibration_report(y_true, y_pred, show_plots=False)

    profiler = cProfile.Profile()
    profiler.enable()
    run_iterations()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(30)
    print(stream.getvalue())


def main():
    print("Pre-warming Numba JIT cache...")
    prewarm_numba_cache()
    print("Numba cache warmed.\n")

    # Create test data
    y_true, y_pred = create_test_data(n=100_000)
    print(f"Test data: {len(y_true):,} samples")

    # Warm up all functions once
    print("Warming up functions...")
    _ = fast_calibration_report(y_true[:1000], y_pred[:1000], show_plots=False)

    # Profile components
    components = profile_components(y_true, y_pred, n_iterations=100)

    # cProfile analysis
    profile_with_cprofile(y_true, y_pred, n_iterations=50)


if __name__ == "__main__":
    main()
