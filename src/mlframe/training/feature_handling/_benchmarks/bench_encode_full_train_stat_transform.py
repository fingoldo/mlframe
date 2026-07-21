"""Bench: LeakageSafeEncoder.transform per-row encode loop vs vectorised factorize+gather at n=10M.

The `_encode_with_full_train_stat` path runs a pure-Python `for i, c in enumerate(cats)` dict-lookup loop over EVERY held-out row
(the production scoring path). At n=10M with a small number of unique categories this is a 10M-iteration Python loop where the
output is a deterministic function of the category only -- so the per-category value can be computed once over the (small) unique
set and gathered back with `pd.factorize` + numpy take. This is bit-identical by construction (same arithmetic per category).

Run: python -m mlframe.training.feature_handling._benchmarks.bench_encode_full_train_stat_transform
"""
from __future__ import annotations

import sys
sys.modules.setdefault("cupy", None)  # type: ignore[arg-type]  # avoid cold cupy import segfault on py3.14
import time
from typing import Literal

import numpy as np

from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder


def _make_data(n: int, n_cat: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    cats_train = np.array([f"cat_{i}" for i in range(n_cat)], dtype=object)
    train_idx = rng.integers(0, n_cat, size=n)
    X_train = cats_train[train_idx]
    y = (rng.random(n) < 0.3).astype(np.int64)
    # transform set: same cats plus a few unseen
    test_idx = rng.integers(0, n_cat + 2, size=n)
    pool = np.array([f"cat_{i}" for i in range(n_cat + 2)], dtype=object)
    X_test = pool[test_idx]
    return X_train, y, X_test


def bench(
    method: Literal["target_mean", "target_m_estimate", "target_james_stein", "target_loo", "woe"],
    n: int = 10_000_000,
    n_cat: int = 200,
    reps: int = 3,
):
    X_train, y, X_test = _make_data(n, n_cat)
    enc = LeakageSafeEncoder(method=method, cv=3)
    enc.fit_transform(X_train, y)

    best = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = enc.transform(X_test)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, out


if __name__ == "__main__":
    for method in ("target_mean", "woe", "target_james_stein"):
        dt, out = bench(method)
        print(f"{method:24s} best={dt*1000:9.1f} ms  out[:3]={out[:3]}")
