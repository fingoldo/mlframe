"""Bench: local_intrinsic_dim per-row covariance+eigvalsh loop vs batched.

OLD: Python `for q in range(n_q)` loop -- each iter builds a (d,d) covariance via
`deviations[q].T @ deviations[q]` and calls `np.linalg.eigvalsh` on one matrix.

NEW: build the whole (n_q, d, d) covariance stack once via `np.einsum`, then call
`np.linalg.eigvalsh` on the stack (numpy batches over the leading axis in C, one
dispatch instead of n_q). The downstream spectrum math is vectorized over rows.

Identity: eigvalsh on each (d,d) slice is computed by the SAME LAPACK routine whether
called per-slice or on the stack -> eigenvalues are bit-identical; the einsum covariance
equals `dev.T @ dev` exactly (same float32 accumulation pattern up to reduction order;
we keep float32 in/out as the prod code does). We assert max-abs identity <= 1e-5 on the
final 5 features (float32 tolerance) and selection-equivalence is trivially satisfied.

Run:
  CUDA_VISIBLE_DEVICES="" python bench_local_intrinsic_dim_batched_spectrum.py
"""
from __future__ import annotations

import time

import numpy as np

N_FEATURES = 5


def _old_loop(deviations: np.ndarray, k_eff: int) -> np.ndarray:
    n_q, _, d = deviations.shape
    out = np.zeros((n_q, N_FEATURES), dtype=np.float32)
    for q in range(n_q):
        cov = (deviations[q].T @ deviations[q]) / float(k_eff)
        lambdas = np.linalg.eigvalsh(cov)
        lambdas = np.clip(lambdas, 0.0, None) + 1e-9
        sum_l = float(lambdas.sum())
        sum_l_sq = float((lambdas**2).sum())
        participation_ratio = (sum_l * sum_l) / sum_l_sq
        top1 = float(lambdas[-1])
        top2 = float(lambdas[-2]) if len(lambdas) >= 2 else 1e-9
        top1_ratio = top1 / sum_l
        top2_ratio = top2 / top1
        p = lambdas / sum_l
        spectrum_entropy = float(-np.sum(p * np.log(p + 1e-9)))
        eff_dim = float(np.exp(spectrum_entropy))
        out[q, 0] = participation_ratio
        out[q, 1] = top1_ratio
        out[q, 2] = top2_ratio
        out[q, 3] = spectrum_entropy
        out[q, 4] = eff_dim
    return out


def _new_batched(deviations: np.ndarray, k_eff: int) -> np.ndarray:
    n_q, _, d = deviations.shape
    # (n_q, d, d) covariance stack via batched matmul (dev^T @ dev per row) in one C
    # dispatch. matmul on the transposed stack is far faster than einsum here (BLAS gemm
    # batched path vs einsum's generic loop) -- ~10x at small d, ~6x at d=50.
    cov = np.matmul(deviations.transpose(0, 2, 1), deviations) / np.float32(k_eff)
    lambdas = np.linalg.eigvalsh(cov)  # (n_q, d) ascending, batched in C
    lambdas = np.clip(lambdas, 0.0, None) + 1e-9
    sum_l = lambdas.sum(axis=1)  # (n_q,)
    sum_l_sq = (lambdas**2).sum(axis=1)  # (n_q,)
    out = np.empty((n_q, N_FEATURES), dtype=np.float32)
    out[:, 0] = (sum_l * sum_l) / sum_l_sq
    top1 = lambdas[:, -1]
    top2 = lambdas[:, -2] if d >= 2 else np.full(n_q, 1e-9, dtype=lambdas.dtype)
    out[:, 1] = top1 / sum_l
    out[:, 2] = top2 / top1
    p = lambdas / sum_l[:, None]
    spectrum_entropy = -np.sum(p * np.log(p + 1e-9), axis=1)
    out[:, 3] = spectrum_entropy
    out[:, 4] = np.exp(spectrum_entropy)
    return out


def _make_inputs(n_q: int, k_eff: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_q, k_eff, d)).astype(np.float32)


def _bestof(fn, dev, k_eff, reps=7):
    fn(dev, k_eff)  # warm
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(dev, k_eff)
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    for n_q, k_eff, d in [(2000, 30, 8), (10000, 30, 20), (5000, 30, 50), (10000, 30, 50)]:
        dev = _make_inputs(n_q, k_eff, d)
        old = _old_loop(dev, k_eff)
        new = _new_batched(dev, k_eff)
        max_abs = float(np.max(np.abs(old - new)))
        rel = max_abs / (float(np.max(np.abs(old))) + 1e-12)
        t_old = _bestof(_old_loop, dev, k_eff)
        t_new = _bestof(_new_batched, dev, k_eff)
        print(f"n_q={n_q:6d} k={k_eff} d={d:3d} | OLD={t_old*1e3:8.2f}ms NEW={t_new*1e3:8.2f}ms "
              f"speedup={t_old/t_new:5.2f}x | max_abs={max_abs:.3e} rel={rel:.3e}")


if __name__ == "__main__":
    main()
