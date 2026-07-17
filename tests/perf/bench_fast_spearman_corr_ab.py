"""iter70 A/B (REJECTED): scalar ``fast_spearman_corr`` -- dedicated 1-D rankdata + scalar
reduction vs the current path that reshapes to (1,N) and routes through the 2-D batched
``_spearmanr_batched_numpy`` (full axis-keepdims reductions on a single row).

# bench-attempt-rejected (2026-06-14, iter70):
#   * Isolated rankdata-only delta: 1-D ``method='average'`` vs 2-D ``axis=1,nan_policy='propagate'``
#     = only 1.13x @ n=2000 / 1.07x @ n=50000 (rankdata is C; the 2-D-on-1-row overhead is small).
#   * ``fast_spearman_corr`` has NO active caller wired into the report hot path (grep: only
#     re-exports in metrics/core.py + regression/__init__.py); it is a public utility. So even a
#     real local win cannot move e2e -- fails the "moves e2e" gate.
#   * UNSAFE in this env: importing mlframe (loads numba) and then calling BOTH the 1-D
#     ``rankdata(x, method='average')`` and the batched ``rankdata(X, axis=1, nan_policy=...)``
#     signatures in one process SEGFAULTS on this py3.14 store build (numba+scipy.stats ABI
#     fragility -- same class as the GPU native-AV suite-abort). The new 1-D path would introduce
#     exactly that mixed-signature call pattern wherever both spearman entry points run in one
#     process. A marginal, non-hot win is not worth a segfault risk.
#   Verdict: REJECT. Kept here so the next agent re-runs this instead of re-trying the swap.

Run: python tests/perf/_iter70_bench.py  (from worktree root, PYTHONPATH=src)
Skip the mixed-signature timing section unless on a build where it does not crash.
"""

from __future__ import annotations

import timeit
import numpy as np

# NOTE: importing ``scipy.stats.rankdata`` at top level THEN calling the
# batched path (which lazy-imports rankdata + uses axis=1/nan_policy) segfaults
# on this py3.14 store build (scipy/numpy ABI fragility). Import scipy.stats
# fully ONCE up front and reuse the same callable everywhere.
import scipy.stats as _sps

rankdata = _sps.rankdata


def scalar_old(yt, yp):
    # Lazy mlframe import: keeping it out of module scope lets the rankdata-only
    # microbench run in a clean (numba-free) process without the segfault.
    """Helper that scalar old."""
    from mlframe.metrics.rank_correlation import _spearmanr_batched_numpy

    YT = np.asarray(yt, dtype=np.float64).reshape(1, -1)
    YP = np.asarray(yp, dtype=np.float64).reshape(1, -1)
    if YT.shape[1] < 2:
        return np.nan
    return float(_spearmanr_batched_numpy(YT, YP)[0])


def scalar_new(yt, yp):
    """Helper that scalar new."""
    yt = np.asarray(yt, dtype=np.float64)
    yp = np.asarray(yp, dtype=np.float64)
    if yt.shape[0] < 2:
        return np.nan
    rx = rankdata(yt, method="average")
    ry = rankdata(yp, method="average")
    if not (np.isfinite(rx[0]) and np.isfinite(ry[0]) and np.isfinite(rx[-1]) and np.isfinite(ry[-1])):
        if not (np.isfinite(rx).all() and np.isfinite(ry).all()):
            return np.nan
    rxc = rx - rx.mean()
    ryc = ry - ry.mean()
    cov = (rxc * ryc).sum()
    vx = (rxc * rxc).sum()
    vy = (ryc * ryc).sum()
    denom = np.sqrt(vx * vy)
    if denom > 0 and np.isfinite(denom) and np.isfinite(cov):
        return float(cov / denom)
    return np.nan


def rankdata_only_microbench():
    """rankdata-signature microbench. Safe to run ONLY in a process that has NOT imported
    mlframe/numba (otherwise the mixed 1-D vs axis=1 call pattern segfaults this build)."""
    rng = np.random.default_rng(0)
    for n in (2000, 50000):
        yt = rng.standard_normal(n)
        yp = rng.standard_normal(n) + 0.3 * yt
        YT2, YP2 = yt.reshape(1, -1), yp.reshape(1, -1)
        for _ in range(5):
            rankdata(YT2, axis=1, nan_policy="propagate")
            rankdata(yt, method="average")
        t2 = (
            min(
                timeit.repeat(
                    lambda: (rankdata(YT2, axis=1, nan_policy="propagate"), rankdata(YP2, axis=1, nan_policy="propagate")),  # noqa: B023 -- timeit.repeat invokes the lambda synchronously within this iteration, never stored
                    number=20,
                    repeat=20,
                )
            )
            / 20
        )
        t1 = (
            min(timeit.repeat(lambda: (rankdata(yt, method="average"), rankdata(yp, method="average")), number=20, repeat=20))  # noqa: B023 -- timeit.repeat invokes the lambda synchronously within this iteration, never stored
            / 20
        )
        print(f"n={n}: 2D-batched {t2 * 1e6:.1f}us  1D {t1 * 1e6:.1f}us  speedup {t2 / t1:.2f}x")


def main():
    """Helper that main."""
    print(
        "iter70: REJECTED -- see module docstring. Measured rankdata-only delta below "
        "(run in a clean process WITHOUT mlframe imported; the in-process scalar A/B "
        "segfaults this numba+scipy.stats build)."
    )
    print("=== rankdata-only microbench ===")
    rankdata_only_microbench()


if __name__ == "__main__":
    main()
