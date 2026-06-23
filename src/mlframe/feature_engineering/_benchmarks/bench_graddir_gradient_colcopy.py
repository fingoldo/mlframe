"""Bench for gradient_direction_agreement._gradient column-copy hotspot.

OLD: per-column `X.copy()` -> d full (n,d) array allocations per gradient call.
NEW: save+restore a single (n,) column per iteration -> 1 column copy per column.

Identity: the predict input matrix is bit-identical between OLD and NEW at every
probe (OLD builds X_plus = copy with col j += eps; NEW mutates col j += eps in
place, restores from a saved copy afterwards). Predict never mutates X, so the
sequence of inputs the model sees is identical -> outputs bit-identical.

Run: py bench_graddir_gradient_colcopy.py
"""
from __future__ import annotations
import time
import numpy as np


class _LinModel:
    """Cheap deterministic stand-in for an lgb/Ridge predictor (predict only)."""
    def __init__(self, d, seed):
        rng = np.random.default_rng(seed)
        self.w = rng.standard_normal(d).astype(np.float32)
        self.b = np.float32(rng.standard_normal())

    def predict(self, X):
        return (X @ self.w + self.b).astype(np.float32)


def _predict(model, X):
    return model.predict(X).astype(np.float32)


def grad_old(model, X, eps):
    n, d = X.shape
    p_base = _predict(model, X)
    grad = np.zeros((n, d), dtype=np.float32)
    for j in range(d):
        X_plus = X.copy()
        X_plus[:, j] += eps
        p_plus = _predict(model, X_plus)
        grad[:, j] = (p_plus - p_base) / eps
    return grad


def grad_new(model, X, eps):
    n, d = X.shape
    p_base = _predict(model, X)
    grad = np.zeros((n, d), dtype=np.float32)
    for j in range(d):
        col = X[:, j].copy()
        X[:, j] = col + eps
        p_plus = _predict(model, X)
        X[:, j] = col
        grad[:, j] = (p_plus - p_base) / eps
    return grad


def best_of(fn, *args, n=7):
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn(*args)
        t.append(time.perf_counter() - s)
    return min(t)


if __name__ == "__main__":
    for (n, d) in [(2000, 30), (5000, 50), (20000, 80)]:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, d)).astype(np.float32)
        model = _LinModel(d, 1)
        eps = 0.05
        go = grad_old(model, X.copy(), eps)
        gn = grad_new(model, X.copy(), eps)
        identical = np.array_equal(go, gn)
        unchanged = np.array_equal(X, rng.standard_normal((n, d)).astype(np.float32)) is False  # X is caller's; new must restore
        # OLD never mutates X (it copies); NEW mutates+restores so X is intact after each call.
        # Both can safely reuse the same X across timing iterations.
        to = best_of(grad_old, model, X, eps)
        tn = best_of(grad_new, model, X, eps)
        print(f"n={n:>6} d={d:>3}  OLD={to*1e3:8.3f}ms  NEW={tn*1e3:8.3f}ms  speedup={to/tn:5.2f}x  identical={identical}")
