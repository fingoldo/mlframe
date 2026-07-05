"""CPX24 microbench: three independent bootstrap.py hot-path candidates.

(1) _jackknife_metric_idx LOO loop: per-iter ``np.delete(full, i)`` (O(n) copy +
    searchsorted overhead) vs a boolean mask-flip ``full[mask]`` gather. Both are
    O(n) per iter so the loop stays O(n^2) either way, but delete carries extra
    per-call dispatch. Measure the per-iter delta and the full-loop delta.

(2) _auc_structural_components: rankdata called 3x (pooled concat, x-self, y-self).
    The three rankings are over DIFFERENT arrays (x|y, x, y) so they are NOT
    derivable from one another -- this checks whether any sharing is even possible
    and measures the cost so the REJECT verdict is numeric, not hand-waved.

(3) _ci_from_samples percentile path: np.percentile(samples, lo) + (.., hi) each
    sort the SAME array -> 2 sorts. Presort once + linear-interp lookup -> 1 sort.
    Fires once per metric per bootstrap_metric call.

Run: python src/mlframe/evaluation/_benchmarks/bench_cpx24_loo_rank_percentile.py
"""
from __future__ import annotations

import time
import numpy as np
from scipy import stats


def _best_of(fn, n_iter, repeats=7):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        dt = time.perf_counter() - t0
        best = min(best, dt / n_iter)
    return best


# ---------------- (1) LOO np.delete vs mask-flip ----------------
def bench_loo(n, max_n=2000):
    sel = np.arange(n) if n <= max_n else np.linspace(0, n - 1, max_n).astype(np.int64)
    full = np.arange(n, dtype=np.int64)

    # metric stand-in: realistic O(len) work on the index array (sum gather).
    data = np.random.default_rng(0).standard_normal(n)

    def metric_idx(idx):
        return float(data[idx].mean())

    def old_loop():
        out = np.empty(sel.shape[0])
        w = 0
        for i in sel:
            loo_idx = np.delete(full, i)
            out[w] = metric_idx(loo_idx)
            w += 1
        return out

    def new_loop():
        out = np.empty(sel.shape[0])
        mask = np.ones(n, dtype=bool)
        w = 0
        for i in sel:
            mask[i] = False
            out[w] = metric_idx(full[mask])
            mask[i] = True
            w += 1
        return out

    r_old = old_loop()
    r_new = new_loop()
    ident = np.array_equal(r_old, r_new)
    t_old = _best_of(old_loop, 1, repeats=5)
    t_new = _best_of(new_loop, 1, repeats=5)
    return t_old, t_new, ident


# ---------------- (2) rankdata x3 sharing feasibility ----------------
def bench_rank(n_pos, n_neg, ties=False):
    rng = np.random.default_rng(1)
    if ties:
        x = rng.integers(0, 10, size=n_pos).astype(float)
        y = rng.integers(0, 10, size=n_neg).astype(float)
    else:
        x = rng.standard_normal(n_pos)
        y = rng.standard_normal(n_neg)

    def old3():
        ranks_all = stats.rankdata(np.concatenate([x, y]), method="average")
        rx_all = ranks_all[:n_pos]
        ry_all = ranks_all[n_pos:]
        rx_self = stats.rankdata(x, method="average")
        ry_self = stats.rankdata(y, method="average")
        return rx_all, ry_all, rx_self, ry_self

    # Attempt: derive self-ranks from pooled argsort? Not possible in general.
    # Document feasibility: self-rank of x depends only on x's internal order,
    # pooled rank mixes y in -> no shared sort reuse without re-ranking.
    res = old3()
    t_old = _best_of(old3, 1, repeats=5)
    return t_old, res


# ---------------- (3) double-sort percentile -> single sort ----------------
def _pct_interp(sorted_s, q):
    # numpy 'linear' interpolation percentile on an ALREADY-sorted array.
    n = sorted_s.shape[0]
    pos = (q / 100.0) * (n - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_s[lo] * (1.0 - frac) + sorted_s[hi] * frac


def bench_percentile(n):
    rng = np.random.default_rng(2)
    samples = rng.standard_normal(n)
    lo_pct, hi_pct = 2.5, 97.5

    def old():
        return float(np.percentile(samples, lo_pct)), float(np.percentile(samples, hi_pct))

    def new():
        s = np.sort(samples)
        return float(_pct_interp(s, lo_pct)), float(_pct_interp(s, hi_pct))

    o = old()
    nw = new()
    ident = o == nw
    close = np.allclose(o, nw, rtol=0, atol=1e-12)
    t_old = _best_of(old, 5, repeats=7)
    t_new = _best_of(new, 5, repeats=7)
    return t_old, t_new, ident, close, o, nw


if __name__ == "__main__":
    print("=== (1) LOO delete vs mask-flip (per full-loop, sec) ===")
    for n in (500, 1000, 2000, 5000):
        t_old, t_new, ident = bench_loo(n)
        print(f"  n={n:6d}  old={t_old*1e3:9.3f}ms  new={t_new*1e3:9.3f}ms  " f"speedup={t_old/t_new:5.2f}x  identical={ident}")

    print("\n=== (2) rankdata x3 (per call, sec); 3 rankings over DIFFERENT arrays ===")
    for npos, nneg in ((500, 500), (5000, 5000), (1000, 50000)):
        t_old, _ = bench_rank(npos, nneg)
        print(f"  n_pos={npos:6d} n_neg={nneg:6d}  cost={t_old*1e3:8.3f}ms")
    print("  Feasibility: self-rank(x) and self-rank(y) need x-only / y-only order;")
    print("  pooled rank mixes both -> the 3 are mathematically distinct, no reuse.")

    print("\n=== (3) percentile double-sort vs single-sort (per call) ===")
    for n in (1000, 2000, 5000, 10000):
        t_old, t_new, ident, close, o, nw = bench_percentile(n)
        print(f"  n={n:6d}  old={t_old*1e6:9.2f}us  new={t_new*1e6:9.2f}us  " f"speedup={t_old/t_new:5.2f}x  exact_eq={ident}  atol1e-12={close}")
