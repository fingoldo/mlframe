"""cProfile micro-benchmark for the T-batch transforms (trailing rolling_quantile_ratio, grouped recurrent trio,
grouped quantile/monotonic, box_cox_y, seasonal_residual, volatility_normalized_residual, asinh_residual_multi,
linear_residual_multi_robust, nadaraya_watson_residual, gaussian_copula_residual).

Profiles fit + forward + inverse per transform at a representative shape and prints the per-op best-of-N wall times
plus a cProfile top-15 for the two heaviest entries (nadaraya_watson_residual and the grouped fits).

Run::

    CUDA_VISIBLE_DEVICES="" python -m mlframe.training.composite.transforms._benchmarks.bench_t_batch_transforms
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY


def _best_of(fn, *args, reps: int = 7, **kwargs):
    best = float("inf")
    r = None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best, r


def _make(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = np.abs(rng.normal(10.0, 3.0, n)) + 1.0
    y = base * rng.uniform(0.8, 1.2, n) + rng.normal(scale=0.5, size=n)
    base2 = np.column_stack([base, rng.normal(size=n)])
    groups = (np.arange(n) // max(n // 20, 1)).astype(np.int64)
    return y, base, base2, groups


_BATCH = (
    "rolling_quantile_ratio", "rolling_quantile_ratio_centered",
    "ewma_residual_grouped", "rolling_quantile_ratio_grouped", "frac_diff_grouped",
    "quantile_residual_grouped", "monotonic_residual_grouped",
    "box_cox_y", "seasonal_residual", "volatility_normalized_residual",
    "asinh_residual_multi", "linear_residual_multi_robust",
    "nadaraya_watson_residual", "gaussian_copula_residual",
)


def main() -> None:
    n = 200_000
    y, base, base2, groups = _make(n)
    print(f"=== T-batch transform bench, n={n} ===")
    for name in _BATCH:
        t = TRANSFORMS_REGISTRY[name]
        b = base2 if "multi" in name else (None if not t.requires_base else base)
        kw = {"groups": groups} if t.requires_groups else {}
        t_fit, params = _best_of(t.fit, y, b, reps=3, **kw)
        t_fwd, T = _best_of(t.forward, y, b, params, reps=5, **kw)
        t_inv, _ = _best_of(t.inverse, T, b, params, reps=5, **kw)
        print(f"{name:34s} fit {t_fit*1e3:9.2f} ms   forward {t_fwd*1e3:9.2f} ms   inverse {t_inv*1e3:9.2f} ms")

    print("\n=== cProfile: nadaraya_watson_residual fit+forward+inverse (x5) ===")
    t = TRANSFORMS_REGISTRY["nadaraya_watson_residual"]
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        p = t.fit(y, base)
        T = t.forward(y, base, p)
        t.inverse(T, base, p)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(15)
    print(s.getvalue())

    print("=== cProfile: quantile_residual_grouped + monotonic_residual_grouped fit (x5) ===")
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        TRANSFORMS_REGISTRY["quantile_residual_grouped"].fit(y, base, groups=groups)
        TRANSFORMS_REGISTRY["monotonic_residual_grouped"].fit(y, base, groups=groups)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(15)
    print(s.getvalue())


if __name__ == "__main__":
    main()
