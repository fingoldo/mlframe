"""A/B bench for ``_quantile_residual_assign_bins``: numpy ``searchsorted`` vs njit.

The bin-assignment maps n base values into a small number (``n_bins``, ~10-20) of
quantile bins via ``np.searchsorted(edges[1:-1], base, "right")``. With very few
cut points the per-element binary search + the separate ``np.clip`` pass leave a
linear-scan njit kernel room to win (single pass, branch-light, no clip temp).

Run: CUDA_VISIBLE_DEVICES="" python -m mlframe.training.composite.transforms._benchmarks.bench_quantile_assign_bins_searchsorted
"""
import time

import numpy as np
import numba

import mlframe.training.composite.transforms.nonlinear as nl


def _reference_searchsorted(base, edges):
    """Pre-optimization numpy reference (the OLD side of the A/B). The prod
    function ``nl._quantile_residual_assign_bins`` now routes through the njit
    kernel, so the OLD baseline must be reproduced here rather than read live."""
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    n_bins = edges.size - 1
    if n_bins <= 1:
        return np.zeros(base_f.size, dtype=np.intp)
    return np.clip(np.searchsorted(edges[1:-1], base_f, side="right"), 0, n_bins - 1)


@numba.njit(cache=True)
def _assign_njit(base_f, inner_edges, n_bins):
    n = base_f.size
    out = np.empty(n, dtype=np.intp)
    m = inner_edges.size
    for i in range(n):
        x = base_f[i]
        # searchsorted side="right": first index where inner_edges[idx] > x
        lo = 0
        hi = m
        while lo < hi:
            mid = (lo + hi) >> 1
            if inner_edges[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        b = lo
        if b > n_bins - 1:
            b = n_bins - 1
        out[i] = b
    return out


def _assign_njit_wrap(base, edges):
    base_f = np.asarray(base, dtype=np.float64).reshape(-1)
    n_bins = edges.size - 1
    if n_bins <= 1:
        return np.zeros(base_f.size, dtype=np.intp)
    return _assign_njit(base_f, np.ascontiguousarray(edges[1:-1]), n_bins)


@numba.njit(cache=True, parallel=True)
def _assign_njit_par_linscan(base_f, inner_edges, n_bins):
    n = base_f.size
    out = np.empty(n, dtype=np.intp)
    m = inner_edges.size
    for i in numba.prange(n):
        x = base_f[i]
        if x != x:  # NaN sorts as +inf in np.searchsorted -> top bin after clip
            out[i] = n_bins - 1
            continue
        b = 0
        # count cuts <= x (side="right"): equals searchsorted-right index.
        for j in range(m):
            if inner_edges[j] <= x:
                b += 1
            else:
                break
        if b > n_bins - 1:
            b = n_bins - 1
        out[i] = b
    return out


def _assign_par_wrap(base, edges):
    base_f = np.ascontiguousarray(np.asarray(base, dtype=np.float64).reshape(-1))
    n_bins = edges.size - 1
    if n_bins <= 1:
        return np.zeros(base_f.size, dtype=np.intp)
    return _assign_njit_par_linscan(base_f, np.ascontiguousarray(edges[1:-1]), n_bins)


def _best(fn, iters):
    fn()
    best = float("inf")
    for _ in range(7):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        best = min(best, (time.perf_counter() - t) / iters)
    return best


def main():
    rng = np.random.default_rng(0)
    for n in (10_000, 200_000, 1_000_000):
        y = rng.standard_normal(n).cumsum()
        base = rng.standard_normal(n)
        qparams = nl._quantile_residual_fit(y, base)
        edges = np.asarray(qparams["bin_edges"], dtype=np.float64)

        # identity gate
        a = _reference_searchsorted(base, edges)
        b = _assign_njit_wrap(base, edges)
        c = _assign_par_wrap(base, edges)
        assert np.array_equal(a, b), f"MISMATCH bin n={n}: {(a != b).sum()} diff"  # nosec B101 - internal invariant check in src/mlframe/training/composite/transforms/_benchmarks, not reachable with untrusted input
        assert np.array_equal(a, c), f"MISMATCH par n={n}: {(a != c).sum()} diff"  # nosec B101 - internal invariant check in src/mlframe/training/composite/transforms/_benchmarks, not reachable with untrusted input

        # adversarial identity: NaN / +-inf / out-of-range
        adv = np.array([np.nan, np.inf, -np.inf, 1e300, -1e300, 0.0], dtype=np.float64)
        assert np.array_equal(_reference_searchsorted(adv, edges), _assign_par_wrap(adv, edges)), "MISMATCH adversarial"  # nosec B101 - internal invariant check in src/mlframe/training/composite/transforms/_benchmarks, not reachable with untrusted input

        old = _best(lambda: _reference_searchsorted(base, edges), 100)
        new = _best(lambda: _assign_njit_wrap(base, edges), 100)
        par = _best(lambda: _assign_par_wrap(base, edges), 100)  # the shipped prod path
        print(f"n={n:>9}: edges={edges.size}  old={old*1e3:8.3f}ms  bin_njit={new*1e3:7.3f}ms ({old/new:4.2f}x)  par_linscan={par*1e3:7.3f}ms ({old/par:4.2f}x)")


if __name__ == "__main__":
    main()
