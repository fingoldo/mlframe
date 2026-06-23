"""Bench: squared-euclidean broadcast temporary vs GEMM decomposition in anchor_attention.

``_score_rows_against_anchors`` (softmax-weighted similarity) and the two hard-assignment blocks
(``train_dists = np.sum((X[:,None,:]-anchors[None,:,:])**2, axis=2)`` followed by ``argmin``/``nanargmin``)
all build a (n_rows, n_anchors, d) broadcast cube. The same squared distances follow from
``||x||^2 - 2 x.a + ||a||^2`` via one GEMM (X @ anchors.T), allocating only (n_rows, n_anchors).

Shapes mirror Mode B at the documented n_anchors=16-64 / N<20k scale. Reports broadcast-vs-GEMM wall
(best-of-N, warm) for both the softmax-scoring path and the argmin hard-assign path, plus the max
|softmax delta| and the argmin-disagreement count so the selection-equivalence claim is grounded.
"""
from __future__ import annotations

import time

import numpy as np


def _dists_broadcast(X, anchors):
    return np.sum((X[:, None, :] - anchors[None, :, :]) ** 2, axis=2)


def _dists_gemm(X, anchors):
    x_sq = np.einsum("ij,ij->i", X, X)[:, None]
    a_sq = np.einsum("ij,ij->i", anchors, anchors)[None, :]
    d = x_sq - 2.0 * (X @ anchors.T) + a_sq
    np.maximum(d, 0.0, out=d)
    return d


def _softmax_from_dists(dists, temp):
    logits = -dists / (temp + 1e-9)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def _best_of(fn, *args, n=30):
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(0)
    temp = 1.0
    for n_rows, d, n_anchors in [(4000, 30, 32), (8000, 50, 32), (16000, 50, 64), (4000, 100, 32)]:
        X = rng.standard_normal((n_rows, d)).astype(np.float32)
        anchors = rng.standard_normal((n_anchors, d)).astype(np.float32)
        _dists_broadcast(X, anchors)
        _dists_gemm(X, anchors)
        tb = _best_of(_dists_broadcast, X, anchors)
        tg = _best_of(_dists_gemm, X, anchors)
        db = _dists_broadcast(X, anchors)
        dg = _dists_gemm(X, anchors)
        sb = _softmax_from_dists(db, temp)
        sg = _softmax_from_dists(dg, temp)
        max_delta = float(np.max(np.abs(sb - sg)))
        argmin_disagree = int(np.sum(np.argmin(db, axis=1) != np.argmin(dg, axis=1)))
        print(f"n={n_rows} d={d} K={n_anchors}: broadcast={tb*1e3:.3f}ms gemm={tg*1e3:.3f}ms "
              f"speedup={tb/tg:.2f}x  max|softmax delta|={max_delta:.2e}  argmin_disagree={argmin_disagree}/{n_rows}")


if __name__ == "__main__":
    main()
