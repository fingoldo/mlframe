"""Bench: per-group Python-loop dispatch vs whole-batch njit for LTR metrics @200k.

Each of dcg_at_k / expected_reciprocal_rank / hit_at_k / precision_at_k loops
over groups in Python and dispatches a single-group njit kernel per group. At
n=200k with ~20k groups that is ~20k Python->njit dispatches per metric.
"""
import scipy.stats  # noqa: F401  (py3.14 ABI prewarm before mlframe import)
import numba  # noqa: F401
import numpy as np
import time
import cProfile
import pstats
import io

# ``mlframe.metrics.core`` native-segfaults at import on py3.14 (eager numba warmup),
# so load the leaf module directly, bypassing the package ``__init__`` chain.
import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_METRICS_DIR = os.path.dirname(_HERE)


def _load_leaf(name, fname):
    pkg = "mlframe.metrics"
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [_METRICS_DIR]
        sys.modules[pkg] = m
    spec = importlib.util.spec_from_file_location(f"{pkg}.{name}", os.path.join(_METRICS_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{pkg}.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


_load_leaf("_numba_params", "_numba_params.py")
_rk = _load_leaf("_ranking_extras", "_ranking_extras.py")
dcg_at_k = _rk.dcg_at_k
expected_reciprocal_rank = _rk.expected_reciprocal_rank
hit_at_k = _rk.hit_at_k
precision_at_k = _rk.precision_at_k


def make_data(n=200_000, groups_per=10, seed=0):
    rng = np.random.default_rng(seed)
    ngroups = n // groups_per
    gids = np.repeat(np.arange(ngroups), groups_per)
    yt = rng.integers(0, 5, size=n).astype(np.float64)
    ys = rng.standard_normal(n)
    return yt, ys, gids


def best(fn, n=5):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts), float(np.median(ts))


if __name__ == "__main__":
    yt, ys, gids = make_data()
    for f in (dcg_at_k, expected_reciprocal_rank, hit_at_k, precision_at_k):
        f(yt[:100], ys[:100], gids[:100], k=10)

    print("ngroups", gids.max() + 1)
    for name, f in [("dcg", dcg_at_k), ("err", expected_reciprocal_rank), ("hit", hit_at_k), ("prec", precision_at_k)]:
        print(name, "best/med ms", [round(x * 1e3, 2) for x in best(lambda f=f: f(yt, ys, gids, k=10))])

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        dcg_at_k(yt, ys, gids, k=10)
        expected_reciprocal_rank(yt, ys, gids, k=10)
        hit_at_k(yt, ys, gids, k=10)
        precision_at_k(yt, ys, gids, k=10)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(20)
    print(s.getvalue())
