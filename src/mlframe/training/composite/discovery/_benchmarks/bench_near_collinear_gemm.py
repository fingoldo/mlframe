"""Near-collinear keep-mask: per-pair njit walk vs one correlation matrix per call.

The dedup walk compares every column against each column already kept with a full pass over n rows per pair, serially,
reading down the columns of a row-major matrix. The same decision only needs the pairwise |corr| matrix, which is one
GEMM on the standardised columns (and three masked GEMMs when the matrix has holes, to get each pair's joint-finite
moments). The greedy walk then runs on the (B, B) matrix, where B is the column count, not the row count.

Run::

    python -m mlframe.training.composite.discovery._benchmarks.bench_near_collinear_gemm

Prints the per-call wall time of each backend and asserts the masks are identical.
"""

from __future__ import annotations

import time

import numpy as np

from .._collinear_numba import near_collinear_keep_mask_fast
from .._eval_stats import _near_collinear_keep_mask_numpy


def abs_corr_matrix(fm: np.ndarray) -> np.ndarray:
    """Pairwise ``|corr|`` over each pair's jointly finite rows, as a dense ``(B, B)`` matrix.

    All-finite input takes one GEMM on the centred columns. With holes, the per-pair joint sums come from GEMMs of the
    zero-filled values and of the finite mask, which is what makes the pair's own mean and variance available without a
    Python loop over pairs.
    """
    x = np.asarray(fm, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.all():
        xc = x - x.mean(axis=0, keepdims=True)
        ss = np.einsum("ij,ij->j", xc, xc)
        denom = np.sqrt(np.outer(ss, ss))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(xc.T @ xc) / denom
        corr[~np.isfinite(corr)] = 0.0
        return corr
    x0 = np.where(finite, x, 0.0)
    m = finite.astype(np.float64)
    n_pair = m.T @ m                      # rows finite in both columns
    s_a = x0.T @ m                        # sum of column i over the pair's support
    s_ab = x0.T @ x0                      # sum of products over the pair's support
    s_aa = (x0 * x0).T @ m                # sum of squares of column i over the pair's support
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = s_ab - s_a * s_a.T / n_pair
        va = s_aa - s_a * s_a / n_pair
        vb = va.T
        corr = np.abs(cov) / np.sqrt(va * vb)
    corr[~np.isfinite(corr)] = 0.0
    corr[n_pair < 3] = 0.0                # too few joint rows to decide, as the reference skips
    return corr


def keep_mask_from_corr(corr: np.ndarray, thr: float) -> np.ndarray:
    """The reference greedy walk, reading the precomputed matrix instead of re-scanning the rows."""
    n_cols = corr.shape[0]
    keep = np.ones(n_cols, dtype=bool)
    kept: list[int] = []
    for j in range(n_cols):
        if any(corr[j, k] > thr for k in kept):
            keep[j] = False
            continue
        kept.append(j)
    return keep


def _matrix(n: int, b: int, *, nan_frac: float = 0.0, seed: int = 0) -> np.ndarray:
    """A screen-like matrix: correlated blocks, one exact duplicate, one constant column, optional holes."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, max(1, b // 4)))
    cols = [base[:, i % base.shape[1]] + rng.normal(scale=0.35, size=n) for i in range(b - 2)]
    cols.append(cols[0].copy())            # exact duplicate
    cols.append(np.full(n, 2.5))           # constant
    x = np.column_stack(cols).astype(np.float32)
    if nan_frac:
        holes = rng.random(x.shape) < nan_frac
        x[holes] = np.nan
    return x


def _time(fn, *args, repeats: int = 3) -> float:
    """Median wall seconds over ``repeats`` runs."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    """Time both backends on a few shapes and assert the keep-masks match."""
    thr = 0.995
    for n, b, nan_frac in ((20000, 60, 0.0), (80000, 120, 0.0), (20000, 60, 0.05)):
        fm = _matrix(n, b, nan_frac=nan_frac)
        kernel = near_collinear_keep_mask_fast(fm, corr_threshold=thr, reference_fn=_near_collinear_keep_mask_numpy)
        gemm = keep_mask_from_corr(abs_corr_matrix(fm), thr)
        same = bool(np.array_equal(kernel, gemm))
        t_kernel = _time(lambda: near_collinear_keep_mask_fast(fm, corr_threshold=thr, reference_fn=_near_collinear_keep_mask_numpy))
        t_gemm = _time(lambda: keep_mask_from_corr(abs_corr_matrix(fm), thr))
        print(
            f"n={n:>6} b={b:>4} nan={nan_frac:>4}: kernel {t_kernel * 1000:9.1f} ms | matrix {t_gemm * 1000:8.1f} ms "
            f"| {t_kernel / max(t_gemm, 1e-9):5.1f}x | identical mask: {same} | kept {int(kernel.sum())}/{b}"
        )
        assert same, "the matrix path must reproduce the kernel's keep-mask exactly"


if __name__ == "__main__":
    main()
