"""Iter-47 benchmark for ``_yj_forward`` candidates.

Hotspot identified by ``_profile_fuzz_1m`` 500k seed=99 cb-only:
``composite_unary_transforms.py:118(_yj_forward)`` -- 5.80s tottime
on 72 calls (Brent neg-loglik scan during ``yeo_johnson_y_fit``).

Variants benchmarked:
- ``cur``: current production (fancy indexing + per-side np.power).
- ``where``: compute both branches over the full array + ``np.where``.
- ``numba_par``: numba @njit(parallel=True) per-element scalar branches.
"""
from __future__ import annotations

import time

import numpy as np
import numba


def _yj_forward_cur(y: np.ndarray, lam: float) -> np.ndarray:
    out = np.empty_like(y, dtype=np.float64)
    nonneg = y >= 0.0
    pos = y[nonneg]
    neg = y[~nonneg]
    if abs(lam) < 1e-12:
        out[nonneg] = np.log1p(pos)
    else:
        out[nonneg] = (np.power(pos + 1.0, lam) - 1.0) / lam
    if abs(lam - 2.0) < 1e-12:
        out[~nonneg] = -np.log1p(-neg)
    else:
        out[~nonneg] = -(np.power(-neg + 1.0, 2.0 - lam) - 1.0) / (2.0 - lam)
    return out


def _yj_forward_where(y: np.ndarray, lam: float) -> np.ndarray:
    abs_y = np.abs(y)
    if abs(lam) < 1e-12:
        pos_branch = np.log1p(abs_y)
    else:
        pos_branch = (np.power(abs_y + 1.0, lam) - 1.0) / lam
    if abs(lam - 2.0) < 1e-12:
        neg_branch = -np.log1p(abs_y)
    else:
        neg_branch = -(np.power(abs_y + 1.0, 2.0 - lam) - 1.0) / (2.0 - lam)
    return np.where(y >= 0.0, pos_branch, neg_branch)


@numba.njit(cache=True, fastmath=True, parallel=True)
def _yj_forward_numba_par(y: np.ndarray, lam: float) -> np.ndarray:
    n = y.shape[0]
    out = np.empty(n, dtype=np.float64)
    lam_is_zero = abs(lam) < 1e-12
    lam_is_two = abs(lam - 2.0) < 1e-12
    inv_lam = 0.0 if lam_is_zero else 1.0 / lam
    two_minus_lam = 2.0 - lam
    inv_2ml = 0.0 if lam_is_two else 1.0 / two_minus_lam
    for i in numba.prange(n):
        yi = y[i]
        if yi >= 0.0:
            if lam_is_zero:
                out[i] = np.log1p(yi)
            else:
                out[i] = ((yi + 1.0) ** lam - 1.0) * inv_lam
        else:
            if lam_is_two:
                out[i] = -np.log1p(-yi)
            else:
                out[i] = -((-yi + 1.0) ** two_minus_lam - 1.0) * inv_2ml
    return out


@numba.njit(cache=True, fastmath=True)
def _yj_forward_numba_serial(y: np.ndarray, lam: float) -> np.ndarray:
    n = y.shape[0]
    out = np.empty(n, dtype=np.float64)
    lam_is_zero = abs(lam) < 1e-12
    lam_is_two = abs(lam - 2.0) < 1e-12
    inv_lam = 0.0 if lam_is_zero else 1.0 / lam
    two_minus_lam = 2.0 - lam
    inv_2ml = 0.0 if lam_is_two else 1.0 / two_minus_lam
    for i in range(n):
        yi = y[i]
        if yi >= 0.0:
            if lam_is_zero:
                out[i] = np.log1p(yi)
            else:
                out[i] = ((yi + 1.0) ** lam - 1.0) * inv_lam
        else:
            if lam_is_two:
                out[i] = -np.log1p(-yi)
            else:
                out[i] = -((-yi + 1.0) ** two_minus_lam - 1.0) * inv_2ml
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    sizes = [50_000, 200_000, 405_000, 1_000_000]
    lams = np.linspace(-1.8, 3.8, 12).tolist()
    print(f"# Iter-47 _yj_forward bench (median of 3, 12-lambda Brent-like sweep)\n")
    print(f"{'n':>10}  {'cur (ms)':>10}  {'where (ms)':>10}  {'nb-ser (ms)':>12}  {'nb-par (ms)':>12}")
    for n in sizes:
        y = rng.standard_normal(n).astype(np.float64)
        # Warm JIT
        _ = _yj_forward_numba_par(y, 1.0)
        _ = _yj_forward_numba_serial(y, 1.0)
        # Correctness check
        for lam in [-1.5, -0.5, 0.0, 0.5, 1.0, 2.0, 2.5, 3.5]:
            a = _yj_forward_cur(y, lam)
            b = _yj_forward_where(y, lam)
            c = _yj_forward_numba_par(y, lam)
            d = _yj_forward_numba_serial(y, lam)
            for name, x in [("where", b), ("nb-par", c), ("nb-ser", d)]:
                if not np.allclose(a, x, equal_nan=True, rtol=1e-10, atol=1e-10):
                    md = np.abs(a - x).max()
                    raise AssertionError(f"{name} mismatch at n={n} lam={lam}: maxdiff={md}")

        def _bench(fn):
            t = []
            for _ in range(3):
                s = time.perf_counter()
                for lam in lams:
                    fn(y, lam)
                t.append(time.perf_counter() - s)
            return sorted(t)[1] * 1000

        cur = _bench(_yj_forward_cur)
        wh = _bench(_yj_forward_where)
        nb_s = _bench(_yj_forward_numba_serial)
        nb_p = _bench(_yj_forward_numba_par)
        print(
            f"{n:>10,}  {cur:>10.1f}  {wh:>10.1f}  {nb_s:>12.1f}  {nb_p:>12.1f}"
            f"   (par speedup vs cur = {cur/nb_p:.2f}x)"
        )


if __name__ == "__main__":
    main()
