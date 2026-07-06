"""Bench + identity for the bincount-based joint-histogram build (CPX9).

The pre-fix joint build was a pure-Python ``joint[xb[i],yb[i]]+=1`` loop outside
njit in _mah / _pid_decomposition / _chao_shen. Replaced by
``np.bincount(xb*K_y+yb).reshape(K_x,K_y)`` (C-level), bit-identical (integer
counts). Run: python bench_joint_hist_build.py
"""
from __future__ import annotations

import time

import numpy as np


def _loop_2d(xb, yb, K_x, K_y):
    joint = np.zeros((K_x, K_y), dtype=np.float64)
    for i in range(xb.size):
        joint[xb[i], yb[i]] += 1.0
    return joint


def _vec_2d(xb, yb, K_x, K_y):
    return np.bincount(xb * K_y + yb, minlength=K_x * K_y).reshape(K_x, K_y).astype(np.float64)


def _loop_3d(x1, x2, y, K1, K2, K3):
    joint = np.zeros((K1, K2, K3), dtype=np.float64)
    for i in range(x1.size):
        joint[x1[i], x2[i], y[i]] += 1.0
    return joint


def _vec_3d(x1, x2, y, K1, K2, K3):
    return np.bincount((x1 * K2 + x2) * K3 + y, minlength=K1 * K2 * K3).reshape(K1, K2, K3).astype(np.float64)


def _best(fn, *a, reps=7):
    t = []
    for _ in range(reps):
        s = time.perf_counter(); fn(*a); t.append(time.perf_counter() - s)
    return min(t)


def main():
    rng = np.random.default_rng(0)
    for n in (5000, 50000):
        K_x, K_y = 16, 10
        xb = rng.integers(0, K_x, n).astype(np.int64)
        yb = rng.integers(0, K_y, n).astype(np.int64)
        assert np.array_equal(_loop_2d(xb, yb, K_x, K_y), _vec_2d(xb, yb, K_x, K_y))  # nosec B101 - internal invariant check in src/mlframe/feature_selection/filters/_benchmarks, not reachable with untrusted input
        to = _best(_loop_2d, xb, yb, K_x, K_y); tv = _best(_vec_2d, xb, yb, K_x, K_y)
        print(f"2D n={n} K={K_x}x{K_y}: LOOP {to*1e3:.2f}ms -> bincount {tv*1e3:.3f}ms ({to/tv:.0f}x) identity OK")

        K1, K2, K3 = 4, 4, 3
        x1 = rng.integers(0, K1, n).astype(np.int64)
        x2 = rng.integers(0, K2, n).astype(np.int64)
        y = rng.integers(0, K3, n).astype(np.int64)
        assert np.array_equal(_loop_3d(x1, x2, y, K1, K2, K3), _vec_3d(x1, x2, y, K1, K2, K3))  # nosec B101 - internal invariant check in src/mlframe/feature_selection/filters/_benchmarks, not reachable with untrusted input
        to = _best(_loop_3d, x1, x2, y, K1, K2, K3); tv = _best(_vec_3d, x1, x2, y, K1, K2, K3)
        print(f"3D n={n} K={K1}x{K2}x{K3}: LOOP {to*1e3:.2f}ms -> bincount {tv*1e3:.3f}ms ({to/tv:.0f}x) identity OK")


if __name__ == "__main__":
    main()
