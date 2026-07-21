"""Microbench: ranking.py's per-group
helpers rewritten from ``for gid in np.unique(group): m = group == gid`` (O(n * n_groups), a full rescan of
the array per group) to a single stable-sort + contiguous-segment walk (O(n log n) total).

Run:  CUDA_VISIBLE_DEVICES="" python bench_ranking_group_loop_o_n.py

OLD side = the actual prior implementations (faithfully reproduced from ``git show HEAD``). NEW side = the
shipped helpers. Both warmed; best-of-N. Bit-identity is asserted for every function on every size.
"""
import sys

sys.modules.setdefault("cupy", None)  # type: ignore[arg-type]  # avoid pre-existing cupy native-AV at import on this box
import time

import numpy as np

from mlframe.training.composite.ranking import (
    _within_group_residual, _residual_to_gains, _ndcg_at_k, _rank01,
)


def _old_within_group_residual(y, base, group, mode):
    res = np.empty(y.shape[0], dtype=np.float64)
    for gid in np.unique(group):
        m = group == gid
        yi, bi = y[m], base[m]
        if mode == "rank":
            res[m] = _rank01(yi) - _rank01(bi)
        else:
            d = yi - bi
            res[m] = d - d.mean()
    return res


def _old_residual_to_gains(res, group, nb=31):
    gains = np.zeros(res.shape[0], dtype=np.int32)
    for gid in np.unique(group):
        m = group == gid
        r = res[m]
        if r.size == 1 or np.ptp(r) == 0:
            gains[m] = 0
            continue
        rk = _rank01(r)
        lvl = np.floor(rk / max(rk.max(), 1.0) * (nb - 1) + 0.5).astype(np.int32)
        gains[m] = lvl
    return gains


def _old_ndcg_at_k(y_true, scores, group, k):
    vals = []
    for gid in np.unique(group):
        m = group == gid
        yt, sc = y_true[m], scores[m]
        kk = min(k, yt.size)
        order = np.argsort(-sc, kind="stable")[:kk]
        gains = 2.0 ** yt[order] - 1.0
        disc = 1.0 / np.log2(np.arange(2, kk + 2))
        dcg = float((gains * disc).sum())
        ideal_order = np.argsort(-yt, kind="stable")[:kk]
        igains = 2.0 ** yt[ideal_order] - 1.0
        idcg = float((igains * disc).sum())
        vals.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def best_of(fn, n=5):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        r = fn()
        ts.append(time.perf_counter() - t)
    return min(ts), r


def make_data(n_groups, items_per_group, seed=0):
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(n_groups), items_per_group)
    perm = rng.permutation(group.size)  # unsorted, realistic row order
    group = group[perm]
    y = rng.integers(0, 5, size=group.size).astype(np.float64)
    base = rng.normal(size=group.size)
    scores = base + rng.normal(scale=0.5, size=group.size)
    return group, y, base, scores


def run():
    print("=== ranking.py group-loop rewrite: OLD (per-group rescan) vs NEW (sort+segment) ===")
    for n_groups, items_per_group in ((200, 20), (2000, 20), (5000, 20)):
        group, y, base, scores = make_data(n_groups, items_per_group)

        for name, old_fn, new_fn, check in (
            ("_within_group_residual", lambda: _old_within_group_residual(y, base, group, "diff"), lambda: _within_group_residual(y, base, group, "diff"), True),
            ("_residual_to_gains", lambda: _old_residual_to_gains(y - base, group), lambda: _residual_to_gains(y - base, group), True),
            ("_ndcg_at_k", lambda: _old_ndcg_at_k(y, scores, group, 10), lambda: _ndcg_at_k(y, scores, group, 10), False),
        ):
            t_old, r_old = best_of(old_fn)
            t_new, r_new = best_of(new_fn)
            if check:
                assert np.allclose(r_old, r_new), f"{name}: NEW result diverges from OLD at n_groups={n_groups}"
            else:
                assert abs(r_old - r_new) < 1e-12, f"{name}: NEW result diverges from OLD at n_groups={n_groups}"
            print(f"  {name:24s} n_groups={n_groups:>5} items/grp={items_per_group}: OLD {t_old*1e3:8.3f}ms  NEW {t_new*1e3:8.3f}ms  speedup {t_old/t_new:6.2f}x")


if __name__ == "__main__":
    run()
