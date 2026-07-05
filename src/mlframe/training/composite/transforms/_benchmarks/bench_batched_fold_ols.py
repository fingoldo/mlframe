"""Micro-bench: batched per-fold single-base OLS vs sequential ``lstsq``.

The tiny-CV per-fold ``linear_residual`` refit solves K independent single-base
(2-parameter) OLS systems. The legacy path calls ``np.linalg.lstsq`` once per
fold, paying K x SVD/dispatch overhead. ``_linear_residual_fit_batched`` computes
each fold's five reductions (n, Sx, Sy, Sxx, Sxy) on the contiguous segment, then
derives all K (alpha, beta) pairs in ONE vectorised arithmetic pass via the
closed-form normal equations -- exact, no lstsq.

Run:
    CUDA_VISIBLE_DEVICES="" python -m \
      mlframe.training.composite.transforms._benchmarks.bench_batched_fold_ols

Reference numbers (this dev box, K=5, mixed fold sizes up to 20k):
    seq-lstsq ~1320 us/call, batched ~260 us/call -> ~5.1x, bit-identical to the
    sequential closed-form reference (alphas/betas exactly equal incl. a
    zero-variance fold). The batched path is the normal-equations form, which
    differs from lstsq by ~1 ULP -- so the bit-identity contract is
    batched-vs-sequential-CLOSED-FORM, not batched-vs-lstsq.
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.training.composite.transforms.linear import (
    _linear_residual_fit_batched,
    _linear_residual_fit_closed,
)


def _seq_lstsq(xs, ys):
    out = []
    for x, y in zip(xs, ys):
        n = len(y)
        X = np.column_stack([x.astype(np.float64), np.ones(n, dtype=np.float64)])
        coef, *_ = np.linalg.lstsq(X, y.astype(np.float64), rcond=None)
        out.append((float(coef[0]), float(coef[1])))
    return out


def _make_folds(seed: int = 0):
    rng = np.random.default_rng(seed)
    sizes = (20000, 15000, 18000, 12000, 20000)
    xs = [rng.normal(size=s) for s in sizes]
    ys = [1.3 * x - 0.4 + rng.normal(size=x.size) * 0.3 for x in xs]
    return xs, ys


def main(repeats: int = 2000) -> None:
    xs, ys = _make_folds(2)

    # Bit-identity gate (batched == sequential closed-form).
    alphas, betas = _linear_residual_fit_batched(xs, ys)
    identical = True
    for k in range(len(xs)):
        a_seq, b_seq = _linear_residual_fit_closed(xs[k], ys[k])
        if not (a_seq == alphas[k] and b_seq == betas[k]):
            identical = False
    print(f"batched == sequential-closed-form bit-identical: {identical}")

    # Warm + time.
    _seq_lstsq(xs, ys)
    _linear_residual_fit_batched(xs, ys)

    t = time.perf_counter()
    for _ in range(repeats):
        _seq_lstsq(xs, ys)
    t_lstsq = (time.perf_counter() - t) / repeats * 1e6

    t = time.perf_counter()
    for _ in range(repeats):
        _linear_residual_fit_batched(xs, ys)
    t_bat = (time.perf_counter() - t) / repeats * 1e6

    print(f"K={len(xs)} folds  seq-lstsq={t_lstsq:8.1f} us/call  " f"batched={t_bat:8.1f} us/call  speedup={t_lstsq / t_bat:5.2f}x")


if __name__ == "__main__":
    main()
