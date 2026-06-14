"""Bench: stable group-segmentation via np.argsort(stable) vs njit counting sort, for integer gids.
Proves bit-identical sort_idx/starts/ends and the speedup at 10M."""
import sys
sys.modules['cupy'] = None
import scipy.stats  # noqa
import numba  # noqa
import time
import numpy as np
from numba import njit


def baseline(group_ids):
    g = np.ascontiguousarray(group_ids)
    n = g.size
    sort_idx = np.argsort(g, kind="stable")
    g_sorted = g[sort_idx]
    bnd = np.where(g_sorted[1:] != g_sorted[:-1])[0] + 1
    starts = np.concatenate(([0], bnd)).astype(np.intp)
    ends = np.concatenate((bnd, [n])).astype(np.intp)
    return sort_idx, starts, ends


@njit(cache=True)
def _stable_counting_argsort_int(g, gmin, span):
    # Stable counting sort returning sort_idx ordered by (gid, original index).
    n = g.shape[0]
    counts = np.zeros(span + 1, dtype=np.int64)
    for i in range(n):
        counts[g[i] - gmin] += 1
    # prefix offsets
    offsets = np.empty(span + 1, dtype=np.int64)
    acc = 0
    nonempty = 0
    for b in range(span + 1):
        offsets[b] = acc
        if counts[b] > 0:
            nonempty += 1
        acc += counts[b]
    sort_idx = np.empty(n, dtype=np.intp)
    cursor = offsets.copy()
    for i in range(n):  # ascending i => stable within group
        b = g[i] - gmin
        sort_idx[cursor[b]] = i
        cursor[b] += 1
    starts = np.empty(nonempty, dtype=np.intp)
    ends = np.empty(nonempty, dtype=np.intp)
    k = 0
    for b in range(span + 1):
        if counts[b] > 0:
            starts[k] = offsets[b]
            ends[k] = offsets[b] + counts[b]
            k += 1
    return sort_idx, starts, ends


def fast(group_ids):
    g = np.ascontiguousarray(group_ids)
    gmin = int(g.min()); gmax = int(g.max())
    span = gmax - gmin
    return _stable_counting_argsort_int(g, gmin, span)


def main():
    n = 10_000_000
    rng = np.random.default_rng(0)
    for n_groups in (200_000, 10_000):
        gids = rng.integers(0, n_groups, size=n).astype(np.int64)
        fast(gids[:1000])  # warm
        tb, tf = [], []
        for _ in range(3):
            t0 = time.perf_counter(); b = baseline(gids); tb.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); f = fast(gids); tf.append(time.perf_counter() - t0)
        # identity
        ok_idx = np.array_equal(b[0], f[0])
        ok_s = np.array_equal(b[1], f[1])
        ok_e = np.array_equal(b[2], f[2])
        print(f"n_groups={n_groups}: base best={min(tb):.3f} fast best={min(tf):.3f} speedup={min(tb)/min(tf):.2f}x  identity idx={ok_idx} starts={ok_s} ends={ok_e}")


if __name__ == "__main__":
    main()
