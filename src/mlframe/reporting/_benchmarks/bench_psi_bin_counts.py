"""Four ways to bin a column against ~10 quantile edges, for the PSI drift heatmap.

``np.histogram`` sorts each slice before counting whenever the bins are non-uniform, and quantile edges
never are -- that sort was 40% of the whole chart's runtime. This bench is what chose the replacement and
what rejected the other two candidates, so a re-attempt starts from the measurement rather than the idea.
"""
import time
import numpy as np
from numba import njit, prange

@njit(cache=True, parallel=True)
def _counts_njit(v, edges, nbins):
    """One parallel pass: each row walks the tiny edge list, private per-thread histograms."""
    nthreads = 8
    part = np.zeros((nthreads, nbins), dtype=np.int64)
    n = v.shape[0]
    chunk = (n + nthreads - 1) // nthreads
    for t in prange(nthreads):
        lo = t * chunk
        hi = min(lo + chunk, n)
        for i in range(lo, hi):
            x = v[i]
            if not np.isfinite(x):
                continue
            b = 0
            for e in range(1, nbins):
                if x >= edges[e]:
                    b = e
                else:
                    break
            part[t, b] += 1
    out = np.zeros(nbins, dtype=np.int64)
    for t in range(nthreads):
        for b in range(nbins):
            out[b] += part[t, b]
    return out

def _hist(v, edges):
    """numpy's own, the current implementation."""
    return np.histogram(v[np.isfinite(v)], bins=edges)[0]

def _ladder(v, edges):
    """One vectorised comparison pass per edge, then difference the cumulative counts."""
    f = v[np.isfinite(v)]
    cum = np.empty(len(edges), dtype=np.int64)
    for i, e in enumerate(edges):
        cum[i] = np.count_nonzero(f < e)
    cum[-1] = f.size
    return np.diff(cum)

def _ss(v, edges, nbins):
    """searchsorted + bincount, the variant already rejected once."""
    f = v[np.isfinite(v)]
    idx = np.searchsorted(edges, f, side="right") - 1
    np.clip(idx, 0, nbins - 1, out=idx)
    return np.bincount(idx, minlength=nbins)

def t(f, r=7):
    """Best-of-r wall time in ms."""
    f()
    best = 1e9
    for _ in range(r):
        a = time.perf_counter(); f(); best = min(best, time.perf_counter() - a)
    return best * 1000

rng = np.random.default_rng(0)
base = rng.standard_normal(100_000)
edges = np.unique(np.quantile(base, np.linspace(0, 1, 11)))
edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
nb = len(edges) - 1
print(f"{'n':>10} {'histogram':>11} {'ladder':>9} {'searchsort':>11} {'njit':>9}   identical")
for n in (10_000, 100_000, 1_000_000):
    v = rng.standard_normal(n)
    v[rng.random(n) < 0.02] = np.nan
    a, b, c, d = _hist(v, edges), _ladder(v, edges), _ss(v, edges, nb), _counts_njit(v, edges, nb)
    same = np.array_equal(a, b) and np.array_equal(a, c) and np.array_equal(a, d)
    print(f"{n:>10,} {t(lambda: _hist(v, edges)):>9.2f}ms {t(lambda: _ladder(v, edges)):>7.2f}ms "
          f"{t(lambda: _ss(v, edges, nb)):>9.2f}ms {t(lambda: _counts_njit(v, edges, nb)):>7.2f}ms   {same}")
