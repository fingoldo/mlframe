"""Bench: stable group-segmentation via np.argsort(stable) vs njit counting sort, for integer gids.
Proves bit-identical sort_idx/starts/ends and the speedup at 10M."""
import sys
sys.modules['cupy'] = None
import scipy.stats  # noqa
import numba  # noqa
import time
import numpy as np

from mlframe.feature_engineering._grouped_segments import _stable_counting_segments_int as _stable_counting_argsort_int


def baseline(group_ids):
    g = np.ascontiguousarray(group_ids)
    n = g.size
    sort_idx = np.argsort(g, kind="stable")
    g_sorted = g[sort_idx]
    bnd = np.where(g_sorted[1:] != g_sorted[:-1])[0] + 1
    starts = np.concatenate(([0], bnd)).astype(np.intp)
    ends = np.concatenate((bnd, [n])).astype(np.intp)
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
