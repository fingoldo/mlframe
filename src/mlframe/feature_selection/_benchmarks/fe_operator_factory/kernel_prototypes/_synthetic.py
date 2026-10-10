"""Synthetic case-2 data and the materialise-then-score reference shared by the kernel prototypes, their benchmarks and their tests."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._fe_cpu_batch import cpu_fe_batch_mi

NB, NCLS = 10, 10


def make(n, K=10, G=8, seed=0):
    """``K`` (u, v) column pairs from ``c, d ~ U(0,1)``, a grid ``T`` of ``G`` offsets per pair (scaled by ``u``'s spread) and decile codes of ``y = log(2c) sin(d/3) + noise``."""
    rng = np.random.default_rng(seed)
    c, d = rng.random(n), rng.random(n)
    y = np.log(2 * c) * np.sin(d / 3) + rng.random(n) / 5
    yc = np.searchsorted(np.quantile(y, np.linspace(0, 1, 11)[1:-1]), y).astype(np.int64)
    us = [np.log(c), c, np.sqrt(c), c * c, -np.log(c), np.log(c + 0.1), np.exp(c), 1 / (c + 0.2), np.abs(c - 0.5), np.sin(3 * c)]
    vs = [np.sin(d), d, np.cos(d), d * d, np.sqrt(d), np.sin(2 * d), np.log(d + 0.1), -d, np.abs(d - 0.5), np.exp(-d)]
    U = np.ascontiguousarray(np.stack(us[:K]))
    V = np.ascontiguousarray(np.stack(vs[:K]))
    sp = U.std(1, keepdims=True)
    T = np.linspace(-0.5, 1.5, G)[None, :] * sp
    return U, V, T, yc


def ref(U, V, T, yc):
    """Reference: materialise the scrubbed ``(K*G, n)`` candidate matrix and score it with the repo's ``cpu_fe_batch_mi``; returns ``(MI (K, G), matrix)``."""
    K, n = U.shape
    G = T.shape[1]
    M = np.empty((n, K * G))
    for k in range(K):
        for g in range(G):
            M[:, k * G + g] = np.nan_to_num((U[k] + T[k, g]) * V[k], nan=0.0, posinf=0.0, neginf=0.0)
    return cpu_fe_batch_mi(M, yc, NB).reshape(K, G), M
