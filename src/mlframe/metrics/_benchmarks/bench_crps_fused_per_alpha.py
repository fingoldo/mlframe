"""Benchmark: CRPS-from-quantiles per-alpha pinball loop vs fused matrix kernel.

FINDING #1 (metrics/quantile.py crps_from_quantiles): the per-alpha mean pinball
vector was built by a Python loop that calls ``pinball_loss(y, p[:, k], a[k])`` K
times -- each call copies a strided column of the C-contiguous (N, K) matrix and
crosses a fresh JIT boundary. The fused ``_fast_pinball_per_alpha(y, P, alphas)``
kernel already exists (used by ``pinball_loss_per_alpha``) and scores every alpha
in one row-major pass. This times BEFORE (loop) vs AFTER (fused) and asserts
bit-identity of the resulting CRPS scalar on representative (N, K).
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.metrics.quantile import (
    _fast_pinball_per_alpha,
    pinball_loss,
    crps_from_quantiles,
)


def _crps_loop(y, p, a):
    """Original loop path (BEFORE)."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    per_alpha = np.empty(a.shape[0], dtype=np.float64)
    for k in range(a.shape[0]):
        per_alpha[k] = pinball_loss(y, p[:, k], float(a[k]))
    integral = float(np.sum((a[1:] - a[:-1]) * (per_alpha[1:] + per_alpha[:-1]) * 0.5))
    if a[0] > 0.0:
        pin_lo_edge = pinball_loss(y, p[:, 0], 0.0)
        integral += float(0.5 * a[0] * (pin_lo_edge + per_alpha[0]))
    if a[-1] < 1.0:
        pin_hi_edge = pinball_loss(y, p[:, -1], 1.0)
        integral += float(0.5 * (1.0 - a[-1]) * (per_alpha[-1] + pin_hi_edge))
    return 2.0 * integral


def _crps_fused(y, p, a):
    """Fused path (AFTER) -- mirrors the shipped crps_from_quantiles body."""
    y = np.asarray(y, dtype=np.float64)
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    a = np.asarray(a, dtype=np.float64)
    per_alpha = _fast_pinball_per_alpha(np.ascontiguousarray(y), p, np.ascontiguousarray(a))
    integral = float(np.sum((a[1:] - a[:-1]) * (per_alpha[1:] + per_alpha[:-1]) * 0.5))
    if a[0] > 0.0:
        pin_lo_edge = pinball_loss(y, p[:, 0], 0.0)
        integral += float(0.5 * a[0] * (pin_lo_edge + per_alpha[0]))
    if a[-1] < 1.0:
        pin_hi_edge = pinball_loss(y, p[:, -1], 1.0)
        integral += float(0.5 * (1.0 - a[-1]) * (per_alpha[-1] + pin_hi_edge))
    return 2.0 * integral


def _time(fn, *args, repeat=5):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(0)
    configs = [(10_000, 10), (10_000, 19), (100_000, 10), (100_000, 19), (1_000_000, 19)]
    # warm up JIT
    y0 = rng.standard_normal(1000)
    p0 = np.sort(rng.standard_normal((1000, 10)), axis=1)
    a0 = np.linspace(0.05, 0.95, 10)
    _crps_loop(y0, p0, a0)
    _crps_fused(y0, p0, a0)

    print(f"{'N':>10} {'K':>4} {'before(ms)':>12} {'after(ms)':>12} {'speedup':>9} {'bit-identical':>14}")
    for n, k in configs:
        y = rng.standard_normal(n)
        p = np.sort(rng.standard_normal((n, k)), axis=1)
        a = np.linspace(0.05, 0.95, k)
        r_before = _crps_loop(y, p, a)
        r_after = _crps_fused(y, p, a)
        identical = r_before == r_after
        t_before = _time(_crps_loop, y, p, a) * 1e3
        t_after = _time(_crps_fused, y, p, a) * 1e3
        print(f"{n:>10} {k:>4} {t_before:>12.3f} {t_after:>12.3f} {t_before / t_after:>8.2f}x {str(identical):>14}")
        # also verify shipped function equals loop
        assert crps_from_quantiles(y, p, a) == r_before or np.isclose(
            crps_from_quantiles(y, p, a), r_before, rtol=0, atol=0
        ), "shipped != loop"


if __name__ == "__main__":
    main()
