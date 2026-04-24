"""Numerical stability + speed benchmarks for feature_engineering.numerical
moment kernels.

Compares:
- ``naive_mean_var_two_pass_seq``  (current pattern in compute_simple_stats_numba)
- ``naive_moments_two_pass_seq``   (current pattern in compute_moments_slope_mi)
- ``welford_mean_var_seq``         (Welford single-pass)
- ``welford_moments_seq``          (Pébay generalised online moments)
- ``kahan_sum_seq``                (Kahan-Babuška-Neumaier compensated sum)

Reports for each input distribution:
- Relative error vs ``np.mean / np.var / scipy.stats.skew / .kurtosis``
- Sign-flip count (for skew/kurt where sign matters)
- Runtime ratio vs naive

Hard input distributions:
1. ``large_mean_small_var`` — x = 1e6 + N(0, 0.1) — catastrophic cancellation domain
2. ``normal`` — x ~ N(0, 1) — easy baseline
3. ``lognormal`` — x = exp(N(0, 1)) — wide range
4. ``high_kurtosis`` — mixture of two normals (heavy tails)
5. ``small_variance`` — 1e9 + N(0, 1e-3) — extreme cancellation
6. ``long_array`` — N=1e7 normal — drift accumulation

Run as: pytest tests/feature_engineering/test_numerical_stability_bench.py -s -v --no-cov

Or directly: python -m mlframe.tests.feature_engineering.test_numerical_stability_bench
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis

from mlframe.feature_engineering._numerical_stable import (
    welford_mean_var_seq,
    welford_moments_seq,
    kahan_sum_seq,
    kahan_two_pass_var_seq,
    naive_mean_var_two_pass_seq,
    naive_moments_two_pass_seq,
)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def _make_distributions(seed=0):
    """Return dict[name] -> ndarray. Sized to expose the issues without
    blowing up wall-clock."""
    rng = np.random.default_rng(seed)
    return {
        "normal_N1k": rng.standard_normal(1_000),
        "normal_N100k": rng.standard_normal(100_000),
        "normal_N1M": rng.standard_normal(1_000_000),
        "large_mean_small_var": 1e6 + rng.standard_normal(100_000) * 0.1,
        "small_variance_1e9": 1e9 + rng.standard_normal(100_000) * 1e-3,
        "lognormal": np.exp(rng.standard_normal(100_000)),
        "high_kurtosis": np.concatenate([
            rng.standard_normal(90_000),
            rng.standard_normal(10_000) * 10.0,
        ]),
    }


# ---------------------------------------------------------------------------
# Reference implementations (high precision)
# ---------------------------------------------------------------------------


def _ref_mean_var(arr):
    return float(np.mean(arr)), float(np.var(arr, ddof=0))


def _ref_skew_kurt(arr):
    return float(sp_skew(arr, bias=True)), float(sp_kurtosis(arr, fisher=True, bias=True))


def _rel_err(actual, ref, eps=1e-15):
    if abs(ref) < eps:
        return abs(actual - ref)  # absolute fallback for ref ~ 0
    return abs(actual - ref) / abs(ref)


# ---------------------------------------------------------------------------
# Warmup (numba JIT)
# ---------------------------------------------------------------------------


def _warmup():
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    welford_mean_var_seq(arr)
    welford_moments_seq(arr)
    kahan_sum_seq(arr)
    kahan_two_pass_var_seq(arr)
    naive_mean_var_two_pass_seq(arr)
    naive_moments_two_pass_seq(arr)


# ---------------------------------------------------------------------------
# Benchmarks (executed as pytest functions for `-s` reporting)
# ---------------------------------------------------------------------------


def test_bench_mean_var_precision():
    """Compare naive 2-pass / Welford / Kahan-2pass on variance precision."""
    _warmup()
    distributions = _make_distributions(seed=42)
    print()
    print(f"{'distribution':<26} {'naive_2p':>14} {'welford':>14} {'kahan_2p':>14} {'best':>10}")
    print("-" * 80)
    for name, arr in distributions.items():
        ref_mean, ref_var = _ref_mean_var(arr)
        _, n_var, _ = naive_mean_var_two_pass_seq(arr)
        _, w_var, _ = welford_mean_var_seq(arr)
        _, k_var, _ = kahan_two_pass_var_seq(arr)
        e_n = _rel_err(n_var, ref_var)
        e_w = _rel_err(w_var, ref_var)
        e_k = _rel_err(k_var, ref_var)
        best = min((e_n, "naive"), (e_w, "welford"), (e_k, "kahan_2p"))
        print(f"{name:<26} {e_n:>14.3e} {e_w:>14.3e} {e_k:>14.3e} {best[1]:>10}")
    # NOTE: no hard assertion. Benchmark surfaces interesting nuances:
    # - Welford WINS big on long arrays (N>10k) — recovers ~log10(N) digits
    # - Welford LOSES on `large_mean + smooth` two-pass cases — running mean
    #   accumulates per-element rounding that the exact two-pass mean avoids
    # - Best-of-both: 2-pass with Kahan compensation in the inner sum
    #   (separate kernel `_kahan_two_pass_var_seq` — TODO add)


def test_bench_moments_precision():
    """Print precision delta for skew + kurt: naive vs Welford-Pébay."""
    _warmup()
    distributions = _make_distributions(seed=42)
    print()
    header = f"{'distribution':<26} {'naive_skew_err':>14} {'welf_skew_err':>14} {'naive_kurt_err':>14} {'welf_kurt_err':>14}"
    print(header); print("-" * len(header))
    for name, arr in distributions.items():
        ref_skew, ref_kurt = _ref_skew_kurt(arr)
        _, _, n_skew, n_kurt, _ = naive_moments_two_pass_seq(arr)
        _, _, w_skew, w_kurt, _ = welford_moments_seq(arr)
        es_n = _rel_err(n_skew, ref_skew)
        es_w = _rel_err(w_skew, ref_skew)
        ek_n = _rel_err(n_kurt, ref_kurt)
        ek_w = _rel_err(w_kurt, ref_kurt)
        print(f"{name:<26} {es_n:>14.3e} {es_w:>14.3e} {ek_n:>14.3e} {ek_w:>14.3e}")


def test_bench_kahan_sum_vs_naive():
    """Kahan compensated sum vs naive Python loop sum."""
    _warmup()
    distributions = _make_distributions(seed=42)
    print()
    print(f"{'distribution':<26} {'naive_sum_relerr':>18} {'kahan_sum_relerr':>18} {'improvement':>12}")
    print("-" * 80)
    for name, arr in distributions.items():
        # Reference: numpy's pairwise sum (already partially compensated)
        # is the best we can get without external high-precision lib.
        ref_sum = float(np.sum(arr))
        # Naive Python loop
        s = 0.0
        for x in arr:
            s += x
        naive_sum = s
        khn = kahan_sum_seq(arr)
        e_naive = _rel_err(naive_sum, ref_sum)
        e_kahan = _rel_err(khn, ref_sum)
        imp = e_naive / max(e_kahan, 1e-300)
        print(f"{name:<26} {e_naive:>18.3e} {e_kahan:>18.3e} {imp:>12.1f}x")


def test_bench_runtime_overhead():
    """Wall-clock ratio: Welford / naive on N=100K and N=1M."""
    _warmup()
    rng = np.random.default_rng(0)
    print()
    print(f"{'kernel':<30} {'N':>10} {'time_ms':>12} {'vs_naive':>12}")
    print("-" * 70)
    for N in (100_000, 1_000_000):
        arr = rng.standard_normal(N)
        # mean+var
        t0 = time.perf_counter()
        for _ in range(50):
            naive_mean_var_two_pass_seq(arr)
        t_naive = (time.perf_counter() - t0) / 50 * 1000
        t0 = time.perf_counter()
        for _ in range(50):
            welford_mean_var_seq(arr)
        t_welford = (time.perf_counter() - t0) / 50 * 1000
        print(f"{'naive_mean_var_2pass':<30} {N:>10} {t_naive:>12.3f} {'1.00':>12}")
        print(f"{'welford_mean_var':<30} {N:>10} {t_welford:>12.3f} {t_welford/t_naive:>12.2f}")
        # moments
        t0 = time.perf_counter()
        for _ in range(50):
            naive_moments_two_pass_seq(arr)
        t_naive_m = (time.perf_counter() - t0) / 50 * 1000
        t0 = time.perf_counter()
        for _ in range(50):
            welford_moments_seq(arr)
        t_welford_m = (time.perf_counter() - t0) / 50 * 1000
        print(f"{'naive_moments_2pass':<30} {N:>10} {t_naive_m:>12.3f} {'1.00':>12}")
        print(f"{'welford_moments':<30} {N:>10} {t_welford_m:>12.3f} {t_welford_m/t_naive_m:>12.2f}")


def test_bench_skew_sign_flips_on_hard():
    """Catastrophic case: 1e9 + small noise. Naive may flip sign or NaN.

    Asserts Welford preserves sign. Naive may legitimately fail."""
    _warmup()
    rng = np.random.default_rng(123)
    arr = 1e9 + rng.standard_normal(50_000) * 1e-5
    # Inject a known-asymmetric tail
    arr[:1000] += 1e-3
    ref_skew, ref_kurt = _ref_skew_kurt(arr)
    _, _, n_skew, n_kurt, _ = naive_moments_two_pass_seq(arr)
    _, _, w_skew, w_kurt, _ = welford_moments_seq(arr)
    print()
    print(f"hard catastrophic case (1e9 + N(0, 1e-5)):")
    print(f"  ref_skew={ref_skew:.6e} ref_kurt={ref_kurt:.6e}")
    print(f"  naive_skew={n_skew:.6e} naive_kurt={n_kurt:.6e}")
    print(f"  welford_skew={w_skew:.6e} welford_kurt={w_kurt:.6e}")
    # Welford skew sign must match reference
    if not np.isnan(ref_skew):
        assert np.sign(w_skew) == np.sign(ref_skew) or abs(ref_skew) < 1e-3, (
            f"Welford skew sign-flipped: ref={ref_skew}, welford={w_skew}"
        )


if __name__ == "__main__":
    # Direct invocation bypasses pytest collection (Win32 cold-start hassle)
    print("=" * 80)
    print("MEAN/VAR PRECISION")
    print("=" * 80)
    test_bench_mean_var_precision()
    print()
    print("=" * 80)
    print("MOMENTS (skew + kurt) PRECISION")
    print("=" * 80)
    test_bench_moments_precision()
    print()
    print("=" * 80)
    print("KAHAN COMPENSATED SUM")
    print("=" * 80)
    test_bench_kahan_sum_vs_naive()
    print()
    print("=" * 80)
    print("RUNTIME OVERHEAD")
    print("=" * 80)
    test_bench_runtime_overhead()
    print()
    print("=" * 80)
    print("HARD CATASTROPHIC CASE")
    print("=" * 80)
    test_bench_skew_sign_flips_on_hard()
