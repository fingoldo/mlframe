"""Family-wise (maxT) permutation gate for a set of candidate columns.

A per-candidate test at alpha lets the chance of at least one noise candidate passing grow with the number of candidates scored. This gate
builds the null distribution of the MAXIMUM plug-in MI over the whole candidate set under label permutations and keeps a candidate only when
the exact permutation p-value ``(1 + R) / (B + 1)`` is at most alpha, where R counts the null maxima at or above the observed value and B is
the number of permutations. Under the null the pass probability is at most alpha for any family size, provided B + 1 >= 1 / alpha.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

FAMILY_MAXT_PERMUTATIONS = 999
FAMILY_EARLY_BLOCK = 256
_FAMILY_DEVICE_MIN_WORK = 3_000_000_000


@njit(cache=True)
def _mi_one_column_njit(col: np.ndarray, n_x: int, y_codes: np.ndarray, n_y: int) -> float:
    """Plug-in MI (nats) of one integer-coded column with ``y_codes``; a serial kernel so the caller can parallelise over columns."""
    n = y_codes.shape[0]
    joint = np.zeros((n_x, n_y), dtype=np.int64)
    for i in range(n):
        joint[col[i], y_codes[i]] += 1
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for a in range(n_x):
        if px[a] == 0:
            continue
        for b in range(n_y):
            nab = joint[a, b]
            if nab == 0:
                continue
            mi += (nab / n) * np.log((nab * n) / (px[a] * py[b]))
    return mi


@njit(cache=True, parallel=True)
def mi_all_columns_njit(cols: np.ndarray, n_x: np.ndarray, y_codes: np.ndarray, n_y: int) -> np.ndarray:
    """Plug-in MI (nats) of every row of ``cols`` (k x n integer codes) with ``y_codes``, one column per parallel task."""
    k = cols.shape[0]
    out = np.zeros(k, dtype=np.float64)
    for j in prange(k):
        out[j] = _mi_one_column_njit(cols[j], n_x[j], y_codes, n_y)
    return out


@njit(cache=True)
def _mi_one_column_into(col: np.ndarray, n_x: int, y_codes: np.ndarray, n_y: int, joint: np.ndarray) -> float:
    """Same plug-in MI as :func:`_mi_one_column_njit`, counting into a caller-owned buffer (no allocation per column and permutation).

    ``joint`` is an int64 array of at least ``n_x * n_y`` cells; it is zeroed here. The integer counts, and the order of the summation, are
    those of the allocating kernel, so the values are identical.
    """
    n = y_codes.shape[0]
    cells = n_x * n_y
    for c in range(cells):
        joint[c] = 0
    for i in range(n):
        joint[col[i] * n_y + y_codes[i]] += 1
    mi = 0.0
    for a in range(n_x):
        pxa = 0
        for b in range(n_y):
            pxa += joint[a * n_y + b]
        if pxa == 0:
            continue
        for b in range(n_y):
            nab = joint[a * n_y + b]
            if nab == 0:
                continue
            pyb = 0
            for aa in range(n_x):
                pyb += joint[aa * n_y + b]
            mi += (nab / n) * np.log((nab * n) / (pxa * pyb))
    return mi


@njit(cache=True, parallel=True)
def family_null_max_njit(cols: np.ndarray, n_x: np.ndarray, y_perms: np.ndarray, n_y: int) -> np.ndarray:
    """Maximum MI over every column of ``cols`` for each row of ``y_perms`` (a block of label permutations).

    Parallel over permutations, serial over columns inside a task: the per-permutation work is K x n, so one task per permutation keeps every
    core busy without a kernel launch per permutation.
    """
    b_count = y_perms.shape[0]
    k = cols.shape[0]
    out = np.zeros(b_count, dtype=np.float64)
    cap = 1
    for j in range(k):
        if n_x[j] > cap:
            cap = n_x[j]
    for b in prange(b_count):
        best = 0.0
        joint = np.zeros(cap * n_y, dtype=np.int64)
        for j in range(k):
            mi = _mi_one_column_into(cols[j], n_x[j], y_perms[b], n_y, joint)
            if mi > best:
                best = mi
        out[b] = best
    return out


def family_null_max(cols: np.ndarray, n_x: np.ndarray, y: np.ndarray, n_y: int, n_perm: int, rng: np.random.Generator, block: int = 256) -> np.ndarray:
    """Null maxima over ``n_perm`` label permutations drawn from ``rng`` in the same order as a per-permutation loop (results are identical)."""
    out = np.empty(int(n_perm), dtype=np.float64)
    # STRICT GPU mode (MLFRAME_FE_GPU_STRICT=1, or AUTO at scale) runs the null on the device; the permutations are drawn on the host in
    # the same order either way, so the decisions do not depend on the path.
    from ._fe_gpu_strict import fe_gpu_strict_enabled
    from ._family_maxt_cuda import device_supported, family_null_max_device

    # Size-gated device dispatch: the null costs B*K*n histogram updates, so a large family goes to the device even outside STRICT (the
    # pair family gate runs 9999 permutations; on the CPU an n=30000 fit exceeded the 600 s test watchdog). The floor is a measured
    # crossover on the dev GPU (GTX 1050 Ti: 1.26x at 6e9 updates, slower below) and belongs in the kernel-tuning cache; TODO(KTC).
    _work = int(n_perm) * int(cols.shape[0]) * int(y.shape[0])
    use_device = device_supported(n_x, int(n_y)) and (bool(fe_gpu_strict_enabled()) or _work >= _FAMILY_DEVICE_MIN_WORK)
    for start in range(0, int(n_perm), block):
        m = min(block, int(n_perm) - start)
        perms = np.empty((m, y.shape[0]), dtype=np.int64)
        for i in range(m):
            perms[i] = y[rng.permutation(y.shape[0])]
        if use_device:
            out[start : start + m] = family_null_max_device(cols, n_x, perms, int(n_y))
        else:
            out[start : start + m] = family_null_max_njit(cols, n_x, perms, int(n_y))
    return out


def exact_pvalue_pass(observed: np.ndarray, null_max: np.ndarray, alpha: float) -> np.ndarray:
    """Exact permutation test: True where ``(1 + R) / (B + 1) <= alpha``, R = number of null maxima at or above the observed value.

    The test can reject only when ``B + 1 >= 1 / alpha``; with fewer permutations every candidate fails, which is the conservative outcome.
    """
    null_sorted = np.sort(np.asarray(null_max, dtype=np.float64))
    obs = np.asarray(observed, dtype=np.float64)
    # number of null maxima >= each observed value, via binary search on the sorted null
    exceed = null_sorted.shape[0] - np.searchsorted(null_sorted, obs, side="left")
    passes: np.ndarray = (1.0 + exceed) / (null_sorted.shape[0] + 1.0) <= float(alpha)
    return passes


def _quantile_codes(v: np.ndarray, n_bins: int) -> tuple[np.ndarray, int]:
    """Integer quantile-bin codes of a finite column and the number of codes it uses."""
    edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1)))
    codes = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, max(len(edges) - 2, 0)).astype(np.int64)
    return codes, int(codes.max()) + 1


def family_maxt_threshold(X: np.ndarray, y_codes: np.ndarray, n_bins: int, n_perm: int, seed: int):
    """Observed plug-in MI per candidate column and the family null maxima under ``n_perm`` label permutations.

    ``X`` is rows x candidates; ``y_codes`` holds the integer target codes. Returns ``(observed, null_max)``; the decision is taken by
    :func:`exact_pvalue_pass`. Shared by the engineered-candidate gate and the empty-support fallback.
    """
    n_y = int(y_codes.max()) + 1
    codes = []
    n_x = []
    for j in range(X.shape[1]):
        v = X[:, j]
        bad = ~np.isfinite(v)
        if bad.any():
            v = v.copy()
            v[bad] = np.median(v[~bad]) if (~bad).any() else 0.0
        c, k = _quantile_codes(v, n_bins)
        codes.append(c)
        n_x.append(k)
    cols = np.ascontiguousarray(np.stack(codes, axis=0), dtype=np.int64)
    nx = np.asarray(n_x, dtype=np.int64)
    observed = mi_all_columns_njit(cols, nx, y_codes, n_y)
    null_max = family_null_max(cols, nx, y_codes, n_y, n_perm, np.random.default_rng(seed))
    return observed, null_max


def family_maxt_pass(candidates, y, n_bins: int = 10, alpha: float = 0.01, n_perm: int = FAMILY_MAXT_PERMUTATIONS, seed: int = 0) -> np.ndarray:
    """Boolean mask over the candidate columns: True where the exact family-wise permutation test passes at ``alpha``.

    ``candidates`` is a 2-D array-like (rows x candidates) or a DataFrame. Non-finite values are replaced by the column median before binning,
    so a constant or all-NaN candidate simply gets MI 0 and cannot pass.
    """
    X = np.asarray(candidates, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    yv = np.asarray(y).ravel()
    _, y_codes = np.unique(yv, return_inverse=True)
    y_codes = y_codes.astype(np.int64).ravel()
    observed, null_max = family_maxt_threshold(X, y_codes, n_bins, n_perm, seed)
    return exact_pvalue_pass(observed, null_max, alpha)


def family_pass_batches(batches, y_codes: np.ndarray, alpha: float = 0.01, n_perm: int = FAMILY_MAXT_PERMUTATIONS, seed: int = 0) -> list:
    """Family-wise pass flags for several batches of candidate codes that form ONE family (e.g. every configuration of one pair).

    ``batches`` is a list of rows x K_b integer code matrices. The null is the maximum MI over the WHOLE family under the same label permutations,
    so a batch cannot pass merely because it was tested on its own. All columns of the family are scored in one kernel call per block of
    permutations. Returns one bool per batch from the exact permutation test at ``alpha``.

    Early rejection: when every batch's best observed MI is at or below the median of the first ``FAMILY_EARLY_BLOCK`` null maxima, the
    remaining permutations are skipped and every batch is rejected.
    """
    y = np.ascontiguousarray(np.asarray(y_codes, dtype=np.int64).ravel())
    n_y = int(y.max()) + 1
    mats = [np.asarray(c, dtype=np.int64) for c in batches]
    widths = [m.shape[1] for m in mats]
    cols = np.ascontiguousarray(np.hstack(mats).T)
    n_x = (cols.max(axis=1) + 1).astype(np.int64)
    observed = mi_all_columns_njit(cols, n_x, y, n_y)
    bounds = np.cumsum([0, *widths])
    observed_max = np.array([float(observed[bounds[j] : bounds[j + 1]].max()) if widths[j] else 0.0 for j in range(len(widths))], dtype=np.float64)
    rng = np.random.default_rng(seed)
    B = int(n_perm)
    first = family_null_max(cols, n_x, y, n_y, min(B, FAMILY_EARLY_BLOCK), rng)
    if observed_max.size and float(observed_max.max()) <= float(np.median(first)):
        return [False for _ in observed_max]
    # Exact early stop per batch: the pass test is (1 + R) / (B + 1) <= alpha with R the count of null maxima at or above the batch's
    # observed maximum. R only grows, so a batch whose count already violates the bound fails for every B; the null still spans the whole
    # family, but the permutation loop ends as soon as no batch is alive.
    R = np.zeros(observed_max.shape[0], dtype=np.int64)
    R += (first[:, None] >= observed_max[None, :]).sum(axis=0)
    done = first.shape[0]
    alive = (1.0 + R) / (B + 1.0) <= float(alpha)
    while done < B and alive.any():
        m = min(FAMILY_EARLY_BLOCK, B - done)
        blk = family_null_max(cols, n_x, y, n_y, m, rng)
        done += m
        R += (blk[:, None] >= observed_max[None, :]).sum(axis=0)
        alive &= (1.0 + R) / (B + 1.0) <= float(alpha)
    return [bool(p) for p in alive]


__all__ = [
    "FAMILY_EARLY_BLOCK",
    "FAMILY_MAXT_PERMUTATIONS",
    "exact_pvalue_pass",
    "family_maxt_pass",
    "family_maxt_threshold",
    "family_null_max",
    "family_pass_batches",
    "mi_all_columns_njit",
]
