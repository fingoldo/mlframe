"""iter76 @200k cProfile harness: composite-discovery screening hot path.

Profiles the repeatable per-screen-pass kernels at n=200000 over F features and B candidate bases:
prebin the feature matrix once, then per base compute mi_y (per-feature MI excluding the base col)
and the abs-corr leak guard. This is the shape CompositeTargetDiscovery._auto_base runs on prod.

Run:
    python -m mlframe.training.composite.discovery._benchmarks.prof_screening_iter76
"""

from __future__ import annotations

import cProfile
import pstats
import sys
from io import StringIO

import scipy.stats  # noqa: F401  (import-order: avoid py3.14 numba+scipy native segfault)
import numba  # noqa: F401
import numpy as np

from mlframe.training.composite.discovery.screening import (
    _prebin_feature_columns,
    _mi_per_feature_prebinned,
    _mi_to_target_prebinned,
    _safe_abs_corr_all,
    _mi_per_feature_y_fixed,
)

_N = 200_000
_F = 100
_B = 8
_NBINS = 16


def _build(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((_N, _F)).astype(np.float32)
    y = np.abs(0.6 * x[:, 0] + 0.3 * x[:, 1] - 0.2 * x[:, 2]).astype(np.float64) + 0.5
    y += rng.standard_normal(_N) * 0.1
    return x, y


def _workload(x, y, prebinned):
    # Per-base screen: mi_y by exclusion + leak-corr guard, B bases.
    acc = 0.0
    for b in range(_B):
        acc += _mi_to_target_prebinned(prebinned, y, nbins=_NBINS, aggregation="mean", exclude_col=b)
        corr = _safe_abs_corr_all(x[:, b].astype(np.float64), x)
        acc += float(corr.sum())
    return acc


def main():
    x, y = _build()
    # warm numba + page-in
    prebinned = _prebin_feature_columns(x, nbins=_NBINS)
    _workload(x, y, prebinned)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3):
        prebinned = _prebin_feature_columns(x, nbins=_NBINS)
        _workload(x, y, prebinned)
    pr.disable()

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(30)
    out = s.getvalue()
    # filter to mlframe-own + key numpy frames
    print(f"screening prof  n={_N} F={_F} B={_B} nbins={_NBINS} py={sys.version.split()[0]}")
    for line in out.splitlines():
        if ("mlframe" in line or "screening" in line or "{method" in line
                or "ncalls" in line or "function calls" in line or "_corr_numba" in line):
            print(line)


if __name__ == "__main__":
    main()
