"""cProfile harness for ``models.lgbm_defaults.default_lgbm_params``.

Run: ``python -m mlframe.models._benchmarks.bench_lgbm_defaults_dart_heuristic``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.models.lgbm_defaults import default_lgbm_params

# Fixed probe matrices reused across calls -- the redundancy probe itself (not matrix construction) is what
# this benchmark needs to profile, matching how a caller would pass an already-materialized X.
_rng = np.random.default_rng(0)
_X_WIDE_INDEPENDENT = _rng.normal(size=(1000, 500))
_X_NARROW_REDUNDANT = np.tile(_rng.normal(size=(1000, 20)), (1, 10)) + 0.02 * _rng.normal(size=(1000, 200))


def _run(n_calls: int) -> None:
    for _ in range(n_calls):
        default_lgbm_params(objective="regression")
        default_lgbm_params(objective="regression", n_features=500)
        default_lgbm_params(objective="regression", n_features=500, n_estimators=200)
        # adaptive extra_trees path, small (below floor) and large (at/above floor) tree-count regimes.
        default_lgbm_params(objective="regression", auto_extra_trees=True, n_estimators=30)
        default_lgbm_params(objective="regression", auto_extra_trees=True, n_estimators=300)
        # redundancy-probe dart path, wide-independent (probe declines) and narrow-redundant (probe triggers).
        default_lgbm_params(objective="regression", auto_dart_redundancy=True, X=_X_WIDE_INDEPENDENT)
        default_lgbm_params(objective="regression", auto_dart_redundancy=True, X=_X_NARROW_REDUNDANT)


if __name__ == "__main__":
    for n_calls in [1000, 100000, 1000000]:
        t0 = time.perf_counter()
        _run(n_calls)
        wall = time.perf_counter() - t0
        print(f"n_calls={n_calls:>8} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1000000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(10)
    print(buf.getvalue())
