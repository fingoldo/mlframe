"""Bench: public ``ranking.py`` ndcg_at_k / map_at_k / mrr -- per-group Python
loop dispatch vs whole-batch njit kernel @ realistic LTR scale.

The public ``ndcg_at_k`` / ``map_at_k`` / ``mrr`` in ``metrics/ranking.py`` still
loop over groups in Python and dispatch a single-group njit kernel per group
(``for i in range(n_groups): _ndcg_one_query(...)``) -- ~n_groups Python->njit
transitions per call. The sibling ``_ranking_extras`` family and
``compute_ranking_summary`` were already converted to whole-batch prange kernels;
these three public entry points were left on the per-group Python loop.

NEW path: dispatch ONE prange kernel (``_per_query_ndcg_kernel`` /
``_per_query_mrr_kernel`` already exist; this bench also prototypes the
``_per_query_map_kernel`` that the fix adds), then NaN-aware reduce in numpy.

OLD baseline = the actual prior code, loaded via ``git show`` into a temp leaf
module so we A/B two real artifacts (per the A/B methodology).

Run: CUDA_VISIBLE_DEVICES="" python bench_ranking_public_batch_dispatch_cpx24.py
"""
import scipy.stats  # noqa: F401  (py3.14 ABI prewarm before mlframe import)
import numba  # noqa: F401
import numpy as np
import time
import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_METRICS_DIR = os.path.dirname(_HERE)


def _load_leaf(name, fname, src_dir=_METRICS_DIR):
    pkg = "mlframe.metrics"
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [_METRICS_DIR]
        sys.modules[pkg] = m
    spec = importlib.util.spec_from_file_location(f"{pkg}.{name}", os.path.join(src_dir, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{pkg}.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


_load_leaf("_numba_params", "_numba_params.py")
rk = _load_leaf("ranking", "ranking.py")


# --- OLD per-group Python-loop implementations (the pre-fix prod code) ---
def _old_ndcg_at_k(y_true, y_score, group_ids, k=10):
    syt, sys_, gs = rk._iter_group_slices(y_true, y_score, group_ids)
    ng = len(gs) - 1
    if ng == 0:
        return float("nan")
    acc = 0.0; nv = 0
    for i in range(ng):
        s, e = gs[i], gs[i + 1]
        v = rk._ndcg_one_query(syt[s:e], sys_[s:e], k)
        if not np.isnan(v):
            acc += v; nv += 1
    return acc / nv if nv else float("nan")


def _old_map_at_k(y_true, y_score, group_ids, k=10):
    syt, sys_, gs = rk._iter_group_slices(y_true, y_score, group_ids)
    ng = len(gs) - 1
    if ng == 0:
        return float("nan")
    acc = 0.0; nv = 0
    for i in range(ng):
        s, e = gs[i], gs[i + 1]
        v = rk._map_one_query(syt[s:e], sys_[s:e], k)
        if not np.isnan(v):
            acc += v; nv += 1
    return acc / nv if nv else float("nan")


def _old_mrr(y_true, y_score, group_ids):
    syt, sys_, gs = rk._iter_group_slices(y_true, y_score, group_ids)
    ng = len(gs) - 1
    if ng == 0:
        return float("nan")
    acc = 0.0; nv = 0
    for i in range(ng):
        s, e = gs[i], gs[i + 1]
        v = rk._mrr_one_query(syt[s:e], sys_[s:e])
        if not np.isnan(v):
            acc += v; nv += 1
    return acc / nv if nv else float("nan")


def make_data(n, groups_per, seed=0):
    rng = np.random.default_rng(seed)
    ng = n // groups_per
    gids = np.repeat(np.arange(ng), groups_per)
    yt = rng.integers(0, 5, size=n).astype(np.float64)
    ys = rng.standard_normal(n)
    return yt, ys, gids


def best(fn, n=7):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts), float(np.median(ts))


if __name__ == "__main__":
    SHAPES = [(50_000, 10), (200_000, 10), (199_998, 6)]
    # warm both paths
    yt, ys, g = make_data(2000, 10)
    for k in (1, 10):
        _old_ndcg_at_k(yt, ys, g, k); rk.ndcg_at_k(yt, ys, g, k)
        _old_map_at_k(yt, ys, g, k); rk.map_at_k(yt, ys, g, k)
    _old_mrr(yt, ys, g); rk.mrr(yt, ys, g)

    for n, gp in SHAPES:
        yt, ys, g = make_data(n, gp)
        ng = n // gp
        print(f"\n=== n={n:_} groups_per={gp} n_groups={ng:_} ===")
        for name, old_f, new_f in [
            ("ndcg@10", lambda: _old_ndcg_at_k(yt, ys, g, 10), lambda: rk.ndcg_at_k(yt, ys, g, 10)),
            ("map@10", lambda: _old_map_at_k(yt, ys, g, 10), lambda: rk.map_at_k(yt, ys, g, 10)),
            ("mrr", lambda: _old_mrr(yt, ys, g), lambda: rk.mrr(yt, ys, g)),
        ]:
            o_best, o_med = best(old_f)
            n_best, n_med = best(new_f)
            ov, nv = old_f(), new_f()
            ident = "IDENTICAL" if (ov == nv or (np.isnan(ov) and np.isnan(nv))) else f"DIFF old={ov} new={nv}"
            print(f"  {name:9s} OLD {o_best*1e3:7.2f}ms  NEW {n_best*1e3:7.2f}ms  "
                  f"speedup {o_best/n_best:5.2f}x  {ident}")
