"""Shared timing/verification helpers for the ``bench_*_smote_synthesize_vectorized.py`` bench family:
small utilities independently duplicated across those scripts, consolidated here so a fix can't
silently drift out of sync across copies. Each benchmark script stays independently runnable from
this same ``_benchmarks/`` directory -- only the literal duplicated bodies move here.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np


def _best(fn: Callable, *a, n: int = 15) -> float:
    """Warm ``fn(*a)`` once, then return its best (minimum) wall-clock time over ``n`` repeats."""
    fn(*a)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn(*a)
        ts.append(time.perf_counter() - t)
    return min(ts)


def smote_bit_identity_and_speed_main(old: Callable, new: Callable) -> None:
    """Shared ``main()`` for a SMOTE-family old-vs-new vectorization bench: verify bit-identity on
    one seed, then report the best-of-15 timing speedup."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 30)).astype(np.float32)
    a = old(X, 5000, 5, 1)
    b = new(X, 5000, 5, 1)
    assert np.array_equal(a, b), "not bit-identical"  # nosec B101 - internal invariant check in src/mlframe/feature_engineering/_benchmarks, not reachable with untrusted input
    t_old = _best(old, X, 5000, 5, 1)
    t_new = _best(new, X, 5000, 5, 1)
    print(f"OLD={t_old * 1000:.2f}ms  NEW={t_new * 1000:.2f}ms  speedup={t_old / t_new:.2f}x  bit-identical=True")
