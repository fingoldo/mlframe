"""iter110 bench: rankgauss replay (`apply_rankgauss`) at n=10M.

Seam: `apply_rankgauss` / `generate_rankgauss_features` computed the average tie rank as `(lo + hi - 1)/2` with TWO full
`np.searchsorted` sweeps (side="left" and side="right") over the test column. The two sweeps differ only where a test
value exactly equals a fit value (a tie). For continuous data -- the canonical rank-Gauss input -- there are no exact
ties, so `hi == lo` and the result is exactly `lo - 0.5`; the entire second sweep is wasted. `_avg_tie_rank` does one
sweep, probes `fit_sorted[lo] == vals` to detect any tie, and only runs the exact two-sweep path when a tie actually
exists -- bit-identical on BOTH continuous and tied/discrete inputs.

Run (from worktree root, PYTHONPATH=src):
    python -m mlframe.feature_selection._benchmarks.bench_rankgauss_replay_iter110

Result @10M (kernel-isolated, best-of-5, this host): avg_rank OLD two-sweep 27.70s -> NEW one-sweep 22.24s (~1.25x),
bit-identical on continuous AND tied. End-to-end `apply_rankgauss` carries the same saving (the searchsorted sweeps
dominate the replay).
"""
from __future__ import annotations

import sys

sys.modules["cupy"] = None  # type: ignore[assignment]  # block cupy native import (py3.14 cold-import segfault under contention)

import time  # noqa: E402

import numpy as np  # noqa: E402


def _avg_rank_two_sweep(fit_sorted: np.ndarray, vals: np.ndarray) -> np.ndarray:
    lo = np.searchsorted(fit_sorted, vals, side="left")
    hi = np.searchsorted(fit_sorted, vals, side="right")
    return (lo + hi - 1) / 2.0


def main() -> None:
    import scipy.stats  # noqa: F401  # warm native import before mlframe (py3.14 cold-import segfault guard)
    import numba  # noqa: F401

    from mlframe.feature_selection.filters._extra_fe_families import _avg_tie_rank

    rng = np.random.default_rng(1)
    n_fit, n_test = 2_000_000, 10_000_000
    fit_sorted = np.sort(rng.standard_normal(n_fit))
    x_cont = rng.standard_normal(n_test)
    x_tied = rng.integers(0, 1000, n_test).astype(np.float64)
    fit_tied = np.sort(rng.integers(0, 1000, n_fit).astype(np.float64))

    assert np.array_equal(_avg_rank_two_sweep(fit_sorted, x_cont), _avg_tie_rank(fit_sorted, x_cont)), "continuous diverged"  # nosec B101 - internal invariant check in src/mlframe/feature_selection/_benchmarks, not reachable with untrusted input
    assert np.array_equal(_avg_rank_two_sweep(fit_tied, x_tied), _avg_tie_rank(fit_tied, x_tied)), "tied diverged"  # nosec B101 - internal invariant check in src/mlframe/feature_selection/_benchmarks, not reachable with untrusted input
    print("identity: continuous OK, tied OK")

    for name, fn in (("old_two_sweep", _avg_rank_two_sweep), ("new_one_sweep", _avg_tie_rank)):
        fn(fit_sorted, x_cont[:1000])
        ts = []
        for _ in range(5):
            t = time.perf_counter()
            fn(fit_sorted, x_cont)
            ts.append(time.perf_counter() - t)
        print(f"{name}: best={min(ts):.4f}s med={sorted(ts)[2]:.4f}s")


if __name__ == "__main__":
    main()
