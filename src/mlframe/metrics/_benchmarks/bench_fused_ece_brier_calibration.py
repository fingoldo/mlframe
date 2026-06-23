"""PERF bench: fused single-pass ECE/Brier (plug-in + debiased) vs three separate binning passes.

The default ``fast_calibration_report`` headline path (``ece_debiased=True``, ``brier_debiased=True``)
calls three kernels that each rebuild the IDENTICAL counts/pred_sum/true_sum histogram over the same
``y_true``/``y_pred``:

    compute_ece_and_brier_decomposition(...)        # plug-in pass 1
    compute_ece_debiased(...)                        # pass 2 (same binning)
    compute_brier_decomposition_debiased(...)        # pass 3 (same binning)

``compute_ece_brier_full_and_debiased`` bins ONCE and emits every reduction. Bit-identical by
construction. This bench measures the wall-time of OLD (three kernels) vs NEW (one fused kernel),
warmed, best-of-N, across realistic ``n`` and ``nbins``.

Run::

    python -m mlframe.metrics._benchmarks.bench_fused_ece_brier_calibration
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.metrics.calibration._calibration_metrics import (
    compute_ece_and_brier_decomposition,
    compute_ece_debiased,
    compute_brier_decomposition_debiased,
    compute_ece_brier_full_and_debiased,
)


def _old(y, p, nbins):
    ece, rel, res, unc, br = compute_ece_and_brier_decomposition(y, p, nbins)
    ece = compute_ece_debiased(y, p, nbins)
    rel, res, unc, br = compute_brier_decomposition_debiased(y, p, nbins)
    return ece, rel, res, unc, br


def _new(y, p, nbins):
    out = compute_ece_brier_full_and_debiased(y, p, nbins)
    # report-equivalent unpack: debiased ECE + debiased REL/RES/UNC
    return out[5], out[6], out[7], out[3], out[8]


def _best_of(fn, y, p, nbins, reps, inner):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn(y, p, nbins)
        dt = (time.perf_counter() - t0) / inner
        best = min(best, dt)
    return best


def _identity(y, p, nbins):
    o = _old(y, p, nbins)
    nnew = _new(y, p, nbins)
    return max(abs(a - b) for a, b in zip(o, nnew))


def main():
    rng = np.random.default_rng(0)
    # warm both paths (numba JIT)
    yw = (rng.uniform(0, 1, 256) < 0.5).astype(np.float64)
    pw = rng.uniform(0, 1, 256)
    _old(yw, pw, 15)
    _new(yw, pw, 15)

    print(f"{'n':>9} {'nbins':>6} {'OLD us':>10} {'NEW us':>10} {'speedup':>8} {'max|diff|':>12}")
    for n in (1_000, 10_000, 100_000, 1_000_000):
        for nbins in (15, 100):
            p = np.clip(rng.uniform(0, 1, n), 1e-6, 1 - 1e-6)
            y = (rng.uniform(0, 1, n) < p).astype(np.float64)
            inner = max(1, 2_000_000 // n)
            reps = 7
            told = _best_of(_old, y, p, nbins, reps, inner) * 1e6
            tnew = _best_of(_new, y, p, nbins, reps, inner) * 1e6
            d = _identity(y, p, nbins)
            print(f"{n:>9} {nbins:>6} {told:>10.2f} {tnew:>10.2f} {told / tnew:>7.2f}x {d:>12.2e}")


if __name__ == "__main__":
    main()
