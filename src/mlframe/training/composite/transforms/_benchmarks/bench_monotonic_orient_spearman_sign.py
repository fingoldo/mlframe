"""Bench: orient ``_monotonic_residual_fit`` by ``_spearman_sign`` instead of ``scipy.stats.spearmanr``.

``_monotonic_residual_fit`` only consumes the SIGN of the Spearman correlation (it flips the
spline orientation, never scales it). The full ``scipy.stats.spearmanr`` computes tie-averaged
ranks + Pearson-on-ranks (two argsorts + rankdata machinery, ~60% cumulative of the fit by
cProfile) just to throw the magnitude away. ``_spearman_sign`` computes the sign from ordinal
ranks (argsort-of-argsort) + a rank covariance.

Run:
    CUDA_VISIBLE_DEVICES="" python -m mlframe.training.composite.transforms._benchmarks.bench_monotonic_orient_spearman_sign

Measured (store py 3.14.3, this host, best-of-9 / median):
    isolated orient      n=50k  1.36x   n=200k 1.38x
    full _monotonic_residual_fit
                         n=50k  36.69 -> 31.61 ms (+13.8%)
                         n=200k 177.23 -> 154.54 ms (+12.8%)
    direction sign-identical to scipy.stats.spearmanr: 0 mismatches / 500 random
    cases (continuous + tied + negative-slope).

RESOLVED (2026-06-24): shipped as the default orient path; sign-identical, so the fitted
spline (and every downstream prediction) is bit-identical. No gate needed.
"""
from __future__ import annotations

import time

import numpy as np


def _old_orient(base_clean: np.ndarray, y_clean: np.ndarray) -> int:
    """The pre-change orient: full scipy.stats.spearmanr, sign of rho."""
    from scipy.stats import spearmanr

    rho, _ = spearmanr(base_clean, y_clean)
    return 1 if (rho is None or not np.isfinite(rho) or rho >= 0) else -1


def _bench() -> None:
    import importlib

    import mlframe.training.composite.transforms.nonlinear as M

    importlib.reload(M)
    rng = np.random.default_rng(2)

    print("== isolated orient ==")
    for n in (50_000, 200_000):
        base = rng.normal(size=n)
        y = np.tanh(base) + rng.normal(size=n) * 0.3
        _old_orient(base, y)
        M._spearman_sign(base, y)
        to, tn = [], []
        for _ in range(9):
            s = time.perf_counter(); _old_orient(base, y); to.append(time.perf_counter() - s)
            s = time.perf_counter(); M._spearman_sign(base, y); tn.append(time.perf_counter() - s)
        print(f"  n={n}: spearmanr {np.median(to) * 1e3:.2f}ms  fast {np.median(tn) * 1e3:.2f}ms  "
              f"speedup {np.median(to) / np.median(tn):.2f}x")

    print("== full _monotonic_residual_fit ==")
    for n in (50_000, 200_000):
        base = rng.normal(size=n)
        y = np.tanh(base) + rng.normal(size=n) * 0.3
        M._monotonic_residual_fit(y, base)
        t = []
        for _ in range(7):
            s = time.perf_counter(); M._monotonic_residual_fit(y, base); t.append(time.perf_counter() - s)
        print(f"  n={n}: full fit {np.median(t) * 1e3:.2f} ms")

    print("== sign identity vs scipy ==")
    mism = 0
    N = 500
    for s in range(N):
        n = int(rng.integers(500, 4000))
        base = rng.normal(size=n)
        if s % 3 == 0:
            base = np.round(base)
        y = rng.choice([1, -1]) * np.tanh(base) + rng.normal(size=n) * rng.uniform(0.1, 1.5)
        if _old_orient(base, y) != M._spearman_sign(base, y):
            mism += 1
    print(f"  direction mismatches: {mism} / {N}")


if __name__ == "__main__":
    _bench()
