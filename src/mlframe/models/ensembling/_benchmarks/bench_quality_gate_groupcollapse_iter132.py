"""iter132 bench: np.add.at -> np.bincount for the group-collapse in quality_gate.filter_ensemble_members at n=10M.

Run:
    PYTHONPATH=src CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 python src/mlframe/models/ensembling/_benchmarks/bench_quality_gate_groupcollapse_iter132.py

The group-collapse path (group_ids + sample_weight both supplied) scatters per-sample weights into per-group accumulators via
np.add.at over n samples. np.bincount(inverse, weights=...) does the same sum reduction in compiled C and is bit-identical
(integer-indexed sum, order-irrelevant up to FP). Bench compares both at n=10M for a few group counts and member counts.

VERDICT (iter132, 2026-06-15, REJECTED): the historical np.add.at penalty no longer holds on this numpy -- bincount is only
1.01-1.02x at n=10M (G in {1k,100k}, M=5), output exact. The add.at -> bincount rewrite is not worth the diff churn here.
    N=10M G=  1000 M=5  add.at 0.1908  bincount 0.1885  1.01x exact=True
    N=10M G=100000 M=5  add.at 0.2161  bincount 0.2147  1.01x exact=True
    N= 1M G=  1000 M=5  add.at 0.0192  bincount 0.0189  1.02x exact=True
Kept as a committed negative result; re-run on an older numpy may still show the classic 2-3x.
"""

import sys

sys.modules["cupy"] = None
import time

import numpy as np
import scipy.stats  # noqa: F401


def old_collapse(arr, inv, sw, G):
    M = arr.shape[0]
    w_sum = np.zeros(G)
    np.add.at(w_sum, inv, sw)
    count = np.zeros(G)
    np.add.at(count, inv, 1.0)
    wsum = np.zeros((M, G))
    for mi in range(M):
        np.add.at(wsum[mi], inv, arr[mi] * sw)
    return w_sum, count, wsum


def new_collapse(arr, inv, sw, G):
    M = arr.shape[0]
    w_sum = np.bincount(inv, weights=sw, minlength=G)
    count = np.bincount(inv, minlength=G).astype(np.float64)
    wsum = np.empty((M, G))
    for mi in range(M):
        wsum[mi] = np.bincount(inv, weights=arr[mi] * sw, minlength=G)
    return w_sum, count, wsum


def main():
    rng = np.random.default_rng(0)
    for N, G, M in [(10_000_000, 1000, 5), (10_000_000, 100_000, 5), (1_000_000, 1000, 5)]:
        gids = rng.integers(0, G, size=N)
        uniq, inv = np.unique(gids, return_inverse=True)
        Gr = uniq.shape[0]
        sw = rng.random(N)
        arr = rng.standard_normal((M, N))

        # warm
        new_collapse(arr[:, :100], inv[:100], sw[:100], Gr)

        bo = bn = 1e9
        for _ in range(3):
            t = time.perf_counter()
            o = old_collapse(arr, inv, sw, Gr)
            bo = min(bo, time.perf_counter() - t)
            t = time.perf_counter()
            nn = new_collapse(arr, inv, sw, Gr)
            bn = min(bn, time.perf_counter() - t)
        ident = all(np.allclose(a, b, rtol=0, atol=1e-9) for a, b in zip(o, nn))
        exact = all(np.array_equal(a, b) for a, b in zip(o, nn))
        print("N=%9d G=%6d M=%d  add.at %.4f  bincount %.4f  %.2fx  close(1e-9)=%s exact=%s" % (N, Gr, M, bo, bn, bo / bn, ident, exact))


if __name__ == "__main__":
    main()
