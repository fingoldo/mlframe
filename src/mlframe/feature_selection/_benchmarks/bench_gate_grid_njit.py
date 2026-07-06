"""iter53 bench (REJECTED): njit tau-grid build kernels for ``_conditional_gate_fe`` vs the numpy per-tau loop.

``cheap_conditional_gate_scan`` builds, per candidate, a (n, 17-tau) feature matrix via a Python loop of 17 ``np.where`` /
``(cv>tau)*av`` calls. This bench prototyped fusing each grid into one njit kernel. VERDICT: REJECTED (not in prod defaults).

Measured (warm, this dev box, 2026-06-13):
* ISOLATED njit(parallel) build IS a win: select 1.84x@533 / 1.71x@1667 / 2.18x@5000 / 1.28x@12000; mask 1.60-2.46x. Bit-identical.
* But END-TO-END inside ``cheap_conditional_gate_scan`` the njit path is 0.89-0.90x (SLOWER): the build is only ~0.4s of the
  ~4s scan; the parallel kernel's per-candidate spawn + core contention with the MI prange (``_gate_grid_mi``) over 648
  candidates swamps the per-call saving. Single-thread njit is slower even in isolation at n=12000 (890us vs 765us numpy --
  numpy's vectorised ``np.where`` over a contiguous column is memory-bandwidth-bound at the floor). A numpy broadcast build
  (``cv[:,None]>taus[None,:]``) regresses 0.65x@n12000 (the (n,17) bool temporary blows the cache).

So the numpy per-tau loop stays in prod. Re-test = re-run this bench. The kernels are kept here (not in prod) per REJECTED != DELETED.

    python -m mlframe.feature_selection._benchmarks.bench_gate_grid_njit
"""
import time

import numba
import numpy as np
from numba import prange

_TAUS = np.round(np.linspace(0.1, 0.9, 17), 4)


@numba.njit(parallel=True, cache=True)
def _build_select_grid_njit_par(cv, av, bv, taus):
    n = cv.shape[0]
    nt = taus.shape[0]
    out = np.empty((n, nt), dtype=np.float64)
    for j in prange(nt):
        tau = taus[j]
        for i in range(n):
            out[i, j] = av[i] if cv[i] > tau else bv[i]
    return out


@numba.njit(parallel=True, cache=True)
def _build_mask_grid_njit_par(cv, av, taus):
    n = cv.shape[0]
    nt = taus.shape[0]
    out = np.empty((n, nt), dtype=np.float64)
    for j in prange(nt):
        tau = taus[j]
        for i in range(n):
            out[i, j] = av[i] if cv[i] > tau else 0.0
    return out


def _numpy_mask(cv, av, taus):
    f = np.empty((cv.shape[0], len(taus)), dtype=np.float64)
    for j, t in enumerate(taus):
        f[:, j] = (cv > t).astype(np.float64) * av
    return f


def _numpy_select(cv, av, bv, taus):
    f = np.empty((cv.shape[0], len(taus)), dtype=np.float64)
    for j, t in enumerate(taus):
        f[:, j] = np.where(cv > t, av, bv)
    return f


def _bench(fn, args, R=200, reps=5):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(R):
            fn(*args)
        best = min(best, (time.perf_counter() - t) / R * 1e6)
    return best


def main():
    rng = np.random.default_rng(1)
    for n in (533, 1667, 5000, 12000):
        cv = rng.standard_normal(n)
        av = rng.standard_normal(n)
        bv = rng.standard_normal(n)
        taus = np.quantile(cv, _TAUS)
        _build_mask_grid_njit_par(cv, av, taus)
        _build_select_grid_njit_par(cv, av, bv, taus)
        assert np.array_equal(_build_mask_grid_njit_par(cv, av, taus), _numpy_mask(cv, av, taus))  # nosec B101 - internal invariant check in src/mlframe/feature_selection/_benchmarks, not reachable with untrusted input
        assert np.array_equal(_build_select_grid_njit_par(cv, av, bv, taus), _numpy_select(cv, av, bv, taus))  # nosec B101 - internal invariant check in src/mlframe/feature_selection/_benchmarks, not reachable with untrusted input
        mn = _bench(_numpy_mask, (cv, av, taus))
        mj = _bench(_build_mask_grid_njit_par, (cv, av, taus))
        sn = _bench(_numpy_select, (cv, av, bv, taus))
        sj = _bench(_build_select_grid_njit_par, (cv, av, bv, taus))
        print(f"n={n:5d} mask numpy={mn:7.1f}us njit={mj:7.1f}us {mn / mj:4.2f}x | "
              f"select numpy={sn:7.1f}us njit={sj:7.1f}us {sn / sj:4.2f}x  (ISOLATED win; e2e REJECT, see docstring)")


if __name__ == "__main__":
    main()
