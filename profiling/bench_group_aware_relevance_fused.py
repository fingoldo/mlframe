"""A/B bench for the fused, chunk-parallel njit rewrite of ``group_aware_relevance``'s per-group loop.

cProfile on the 2M-row LTR combo showed 236.7s self-time in the Python per-group loop, of which
np.quantile alone cost 109.5s cumtime across 324k dispatches (162k query groups x 2 calls/group) --
see profiling/profile_ltr_2m.pstats. This bench validates selection-equivalence against the pre-fusion
per-group Python/numpy reference and measures the real wall-clock win at LTR-realistic group counts and
sizes (small groups, ~10-30 rows/query -- the case that made per-group Python dispatch the bottleneck).
"""

import time

import numpy as np

from mlframe.training.ranking._ranker_fs import _binned_mi, group_aware_relevance


def _reference_group_aware_relevance(cols, arr, y, groups, bins=8):
    """Pre-fusion per-group Python/numpy reference (the implementation being replaced)."""
    out = {}
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="mergesort")
    gs = groups[order]
    arr_s = arr[order]
    y_s = y[order]
    boundaries = np.flatnonzero(gs[1:] != gs[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [gs.size]))
    sizes = (stops - starts).astype(np.float64)
    contributing_total = float(sizes[sizes >= 4].sum()) or 1.0
    ncols = len(cols)
    acc = np.zeros(ncols, dtype=np.float64)
    for b in range(starts.size):
        s, e = int(starts[b]), int(stops[b])
        if e - s < 4:
            continue
        y_g, block, w = y_s[s:e], arr_s[s:e], float(e - s)
        for j in range(ncols):
            acc[j] += w * _binned_mi(block[:, j], y_g, bins=bins)
    for j, name in enumerate(cols):
        out[name] = acc[j] / contributing_total
    return out


def _make_ltr_like(n_groups, avg_group_size, ncols, seed):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(max(4, avg_group_size - 8), avg_group_size + 8, size=n_groups)
    n = int(sizes.sum())
    groups = np.repeat(np.arange(n_groups), sizes)
    y_true = rng.standard_normal(n_groups)
    y = np.repeat(y_true, sizes) + rng.standard_normal(n) * 0.3
    arr = rng.standard_normal((n, ncols))
    arr[:, 0] = y + rng.standard_normal(n) * 0.2  # one genuinely relevant column
    order = rng.permutation(n)
    return [f"f{j}" for j in range(ncols)], arr[order], y[order], groups[order]


def main():
    cols, arr, y, groups = _make_ltr_like(n_groups=8000, avg_group_size=15, ncols=20, seed=0)

    # warm JIT
    cols_w, arr_w, y_w, groups_w = _make_ltr_like(n_groups=200, avg_group_size=15, ncols=20, seed=1)
    group_aware_relevance(cols_w, arr_w, y_w, groups_w)

    t0 = time.perf_counter()
    ref = _reference_group_aware_relevance(cols, arr, y, groups)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    got = group_aware_relevance(cols, arr, y, groups)
    t_new = time.perf_counter() - t0

    worst = max(abs(got[c] - ref[c]) for c in cols)
    print(f"reference: {t_ref:.3f}s")
    print(f"fused:     {t_new:.3f}s")
    print(f"speedup:   {t_ref / t_new:.2f}x")
    print(f"worst abs diff: {worst:.3e}")
    print(f"selection-equivalence (< 1e-9): {worst < 1e-9}")
    assert got["f0"] > 0.05, "the genuinely relevant column must still score high"


if __name__ == "__main__":
    main()
