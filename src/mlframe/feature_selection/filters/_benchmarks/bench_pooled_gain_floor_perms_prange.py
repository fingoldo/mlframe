"""Bench: parallelise the order-1 maxT permutation-null kernel over the K shuffles.

``_pooled_gain_floor_perms_njit`` (the order-1 Westfall-Young maxT floor in
``_permutation_null.py``) is a single-thread ``@numba.njit`` that loops over K
(default 200) independent target shuffles, rescoring the whole order-1 candidate
pool per shuffle. The shuffle loop is embarrassingly parallel: each iteration k
writes only ``maxes[k]`` and reads shared read-only inputs, so a
``parallel=True`` + ``prange`` over k (with a per-thread ``counts`` scratch) is
bit-identical (each shuffle's FP entropy reduction order is unchanged) and scales
with cores.

OLD = current single-thread njit (HEAD).
NEW = prange-over-nperm twin.

Realistic shapes: the order-1 floor fires on WIDE pools (>= screen_fdr_min_features
= 30 candidates), K=200 default, n in {10k, 100k}. Wide-pool embedding/TF-IDF
screens are exactly the p>>sqrt(n) regime this floor targets.

Run: CUDA_VISIBLE_DEVICES="" python .../bench_pooled_gain_floor_perms_prange.py
"""
from __future__ import annotations

import time

import numba
import numpy as np
from numba import prange


@numba.njit(cache=True)
def _old_kernel(scaled_flat, offsets, joint_card, h_x, mm_bias, h_y, y_perms, inv_n):
    nperm = y_perms.shape[0]
    n = y_perms.shape[1]
    ncand = offsets.shape[0] - 1
    maxes = np.empty(nperm, dtype=np.float64)
    max_jc = 0
    for j in range(ncand):
        if joint_card[j] > max_jc:
            max_jc = joint_card[j]
    counts = np.empty(max_jc, dtype=np.float64)
    for k in range(nperm):
        yp = y_perms[k]
        best = 0.0
        for j in range(ncand):
            jc = joint_card[j]
            for t in range(jc):
                counts[t] = 0.0
            s0 = offsets[j]
            for i in range(n):
                counts[scaled_flat[s0 + i] + yp[i]] += 1.0
            h_xy = 0.0
            for t in range(jc):
                c = counts[t]
                if c > 0.0:
                    p = c * inv_n
                    h_xy -= p * np.log(p)
            mi = h_x[j] + h_y - h_xy - mm_bias[j]
            if mi > best:
                best = mi
        maxes[k] = best
    return maxes


@numba.njit(cache=True, parallel=True)
def _new_kernel(scaled_flat, offsets, joint_card, h_x, mm_bias, h_y, y_perms, inv_n):
    nperm = y_perms.shape[0]
    n = y_perms.shape[1]
    ncand = offsets.shape[0] - 1
    maxes = np.empty(nperm, dtype=np.float64)
    max_jc = 0
    for j in range(ncand):
        if joint_card[j] > max_jc:
            max_jc = joint_card[j]
    for k in prange(nperm):
        counts = np.empty(max_jc, dtype=np.float64)  # per-thread scratch
        yp = y_perms[k]
        best = 0.0
        for j in range(ncand):
            jc = joint_card[j]
            for t in range(jc):
                counts[t] = 0.0
            s0 = offsets[j]
            for i in range(n):
                counts[scaled_flat[s0 + i] + yp[i]] += 1.0
            h_xy = 0.0
            for t in range(jc):
                c = counts[t]
                if c > 0.0:
                    p = c * inv_n
                    h_xy -= p * np.log(p)
            mi = h_x[j] + h_y - h_xy - mm_bias[j]
            if mi > best:
                best = mi
        maxes[k] = best
    return maxes


def _make_inputs(n, n_cand, nperm, nbins_x=12, nbins_y=10, seed=0):
    rng = np.random.default_rng(seed)
    inv_n = 1.0 / n
    scaled = []
    joint_card = []
    h_x = []
    mm_bias = []
    for _ in range(n_cand):
        xc = rng.integers(0, nbins_x, size=n).astype(np.int64)
        xcounts = np.bincount(xc, minlength=nbins_x).astype(np.float64)
        px = xcounts[xcounts > 0] * inv_n
        scaled.append((xc * nbins_y).astype(np.int32))
        joint_card.append(nbins_x * nbins_y)
        h_x.append(float(-(px * np.log(px)).sum()))
        mm_bias.append((nbins_x - 1) * (nbins_y - 1) / (2.0 * n))
    y_codes = rng.integers(0, nbins_y, size=n).astype(np.int32)
    ycounts = np.bincount(y_codes, minlength=nbins_y).astype(np.float64)
    py = ycounts[ycounts > 0] * inv_n
    h_y = float(-(py * np.log(py)).sum())
    y_perm = y_codes.copy()
    y_perms = np.empty((nperm, n), dtype=np.int32)
    for k in range(nperm):
        rng.shuffle(y_perm)
        y_perms[k] = y_perm
    scaled_flat = np.concatenate(scaled).astype(np.int32)
    offsets = np.arange(n_cand + 1, dtype=np.int64) * n
    jc = np.asarray(joint_card, dtype=np.int64)
    hx = np.asarray(h_x, dtype=np.float64)
    mm = np.asarray(mm_bias, dtype=np.float64)
    return scaled_flat, offsets, jc, hx, mm, h_y, y_perms, inv_n


def _best_of(fn, args, reps=7):
    t = []
    for _ in range(reps):
        s = time.perf_counter()
        out = fn(*args)
        t.append(time.perf_counter() - s)
    return min(t), out


def main():
    shapes = [
        (10_000, 50, 200),
        (10_000, 200, 200),
        (100_000, 50, 200),
        (100_000, 200, 200),
    ]
    for n, n_cand, nperm in shapes:
        args = _make_inputs(n, n_cand, nperm)
        # warm JIT
        _old_kernel(*args)
        _new_kernel(*args)
        t_old, o_old = _best_of(_old_kernel, args)
        t_new, o_new = _best_of(_new_kernel, args)
        max_abs = float(np.max(np.abs(o_old - o_new)))
        q_old = float(np.quantile(o_old, 0.95))
        q_new = float(np.quantile(o_new, 0.95))
        print(
            f"n={n:>7} n_cand={n_cand:>4} K={nperm}: "
            f"OLD={t_old*1e3:8.2f}ms NEW={t_new*1e3:8.2f}ms "
            f"speedup={t_old/t_new:5.2f}x  max|maxes diff|={max_abs:.2e}  "
            f"floor OLD={q_old:.8f} NEW={q_new:.8f}"
        )


if __name__ == "__main__":
    main()
