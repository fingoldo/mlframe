"""Family-wise (maxT) permutation gate for a set of candidate columns.

A per-candidate test at alpha lets the chance of at least one noise candidate passing grow with the number of candidates scored. This gate
builds the null distribution of the MAXIMUM plug-in MI over the whole candidate set under label permutations and keeps a candidate only when
the exact permutation p-value ``(1 + R) / (B + 1)`` is at most alpha, where R counts the null maxima at or above the observed value and B is
the number of permutations. Under the null the pass probability is at most alpha for any family size, provided B + 1 >= 1 / alpha.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
from numba import njit, prange

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import ArrayLike

FAMILY_MAXT_PERMUTATIONS = 999
FAMILY_EARLY_BLOCK = 256
FAMILY_PROBE_BLOCK = 32  # permutations drawn before deciding that a family far inside the null needs no more
# Fallback crossover (B*K*n histogram updates) used when the per-host kernel-tuning cache has no entry. Measured on GTX 1050 Ti: the device
# null is 1.26x the CPU kernel at 6e9 and slower below; the per-host sweep (``_family_null_backend``) replaces this constant.
_FAMILY_DEVICE_MIN_WORK = 3_000_000_000


def _family_null_backend(n_rows: int, n_cols: int, work: int) -> str:
    """'device' or 'cpu' for one family null, from the per-host kernel-tuning cache (async sweep), falling back to the work crossover."""
    fallback = "device" if work >= _FAMILY_DEVICE_MIN_WORK else "cpu"
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        res = KernelTuningCache.load_or_create().get_or_tune(
            "family_null_backend",
            dims={"n_rows": int(n_rows), "n_cols": int(n_cols)},
            tuner=_run_family_null_sweep,
            axes=["n_rows", "n_cols"],
            fallback={"kernel_choice": fallback},
            code_version=_FAMILY_NULL_CODE_VERSION,
            async_sweep=True,
        )
        if isinstance(res, str):
            return res
        if res:
            return str(res.get("kernel_choice", fallback))
    except Exception as exc:
        import logging

        logging.getLogger("mlframe.feature_selection.filters.mrmr").debug("family null backend selection failed: %r", exc)
    return fallback


_FAMILY_NULL_CODE_VERSION = "family_null_backend-v1-2026-10-16"


def _run_family_null_sweep():
    """Per-host CPU-vs-device sweep of the family null over (n_rows, n_cols); returns [] when the helper is unavailable."""
    try:
        from pyutilz.dev.benchmarking import sweep_backend_grid
        from ._family_maxt_cuda import family_null_max_device
    except Exception as exc:
        import logging

        logging.getLogger("mlframe.feature_selection.filters.mrmr").debug("family null sweep unavailable (%s: %s); no regions recorded", type(exc).__name__, exc)
        return []

    def _make_inputs(dims):
        """Make inputs."""
        n = int(dims["n_rows"])
        K = int(dims["n_cols"])
        B = 99
        rng = np.random.default_rng(0)
        cols = rng.integers(0, 8, (K, n)).astype(np.int64)
        nx = np.full(K, 8, dtype=np.int64)
        y = rng.integers(0, 4, n).astype(np.int64)
        perms = np.stack([y[rng.permutation(n)] for _ in range(B)]).astype(np.int64)
        return (cols, nx, perms, 4)

    return list(
        sweep_backend_grid(
            {"cpu": lambda c, nx, p, ny: family_null_max_njit(c, nx, p, ny), "device": lambda c, nx, p, ny: family_null_max_device(c, nx, p, ny)},
            {"n_rows": [2000, 20000], "n_cols": [200, 2000]},
            _make_inputs,
            reference="cpu",
            repeats=2,
            equiv_rtol=1e-12,
            equiv_atol=1e-15,
            decision_key="kernel_choice",
        )
        or []
    )


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
    out = np.zeros(k, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
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
    out = np.zeros(b_count, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
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


_PERM_BLOCK_CACHE_BYTES = 256 * 1024 * 1024
_perm_block_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
_perm_block_cache_bytes = 0
_perm_block_cache_lock = threading.Lock()  # move_to_end, insert and the byte-budget eviction are one read-modify-write on shared state


def _permutation_block(rng: np.random.Generator, y: np.ndarray, m: int) -> np.ndarray:
    """Row-permutation indices of the next ``m`` label permutations drawn from ``rng``.

    The family gate rebuilds the same permutation stream for every candidate family (each call starts from the same seed), and drawing it
    costs more than the MI kernel itself. The block is cached under the generator state it was drawn from, and the stored state after the
    draw is restored on a hit, so the stream the caller sees is exactly the one a fresh draw would produce. The cache is bounded (LRU, bytes).
    """
    global _perm_block_cache_bytes
    state_before = rng.bit_generator.state
    key = (y.shape[0], int(m), hash(y.tobytes()), repr(state_before))
    n = y.shape[0]
    with _perm_block_cache_lock:
        hit = _perm_block_cache.get(key)
        if hit is not None:
            _perm_block_cache.move_to_end(key)
            idx, state_after = hit
            rng.bit_generator.state = state_after
            cached: np.ndarray = idx
            return cached
        idx_dtype = np.uint16 if n <= np.iinfo(np.uint16).max else np.int32
        idx = np.empty((m, n), dtype=idx_dtype)
        for i in range(m):
            idx[i] = rng.permutation(n)
        _perm_block_cache[key] = (idx, rng.bit_generator.state)
        _perm_block_cache_bytes += idx.nbytes
        while _perm_block_cache_bytes > _PERM_BLOCK_CACHE_BYTES and len(_perm_block_cache) > 1:
            _, (old, _) = _perm_block_cache.popitem(last=False)
            _perm_block_cache_bytes -= old.nbytes
    return idx


def family_null_max(cols: np.ndarray, n_x: np.ndarray, y: np.ndarray, n_y: int, n_perm: int, rng: np.random.Generator, block: int = 256) -> np.ndarray:
    """Null maxima over ``n_perm`` label permutations drawn from ``rng`` in the same order as a per-permutation loop (results are identical)."""
    out = np.empty(int(n_perm), dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    # STRICT GPU mode (MLFRAME_FE_GPU_STRICT=1, or AUTO at scale) runs the null on the device; the permutations are drawn on the host in
    # the same order either way, so the decisions do not depend on the path.
    from ._fe_gpu_strict import fe_gpu_strict_enabled
    from ._family_maxt_cuda import device_supported, family_null_max_device

    # Size-gated device dispatch: the null costs B*K*n histogram updates, so a large family goes to the device even outside STRICT (the
    # pair family gate runs 9999 permutations; on the CPU an n=30000 fit exceeded the 600 s test watchdog). The floor is a measured
    # crossover on the dev GPU (GTX 1050 Ti: 1.26x at 6e9 updates, slower below) and belongs in the kernel-tuning cache; TODO(KTC).
    _work = int(n_perm) * int(cols.shape[0]) * int(y.shape[0])
    use_device = device_supported(n_x, int(n_y)) and (
        True if fe_gpu_strict_enabled() else _family_null_backend(int(y.shape[0]), int(cols.shape[0]), _work) == "device"
    )
    for start in range(0, int(n_perm), block):
        m = min(block, int(n_perm) - start)
        perms = y[_permutation_block(rng, y, m)]
        if use_device:
            out[start : start + m] = family_null_max_device(cols, n_x, perms, int(n_y))
        else:
            out[start : start + m] = family_null_max_njit(cols, n_x, perms, int(n_y))
    return out


def exact_pvalue_pass(observed: np.ndarray, null_max: np.ndarray, alpha: float) -> np.ndarray:
    """Exact permutation test: True where ``(1 + R) / (B + 1) <= alpha``, R = number of null maxima at or above the observed value.

    The test can reject only when ``B + 1 >= 1 / alpha``; with fewer permutations every candidate fails, which is the conservative outcome.
    """
    # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    null_sorted = np.sort(np.asarray(null_max, dtype=np.float64))
    obs = np.asarray(observed, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    # number of null maxima >= each observed value, via binary search on the sorted null
    exceed = null_sorted.shape[0] - np.searchsorted(null_sorted, obs, side="left")
    passes: np.ndarray = (1.0 + exceed) / (null_sorted.shape[0] + 1.0) <= float(alpha)
    return passes


def _quantile_codes(v: np.ndarray, n_bins: int) -> tuple[np.ndarray, int]:
    """Integer quantile-bin codes of a finite column and the number of codes it uses."""
    edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1)))
    codes = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, max(len(edges) - 2, 0)).astype(np.int64)
    return codes, int(codes.max()) + 1


def _column_values(X, j: int) -> np.ndarray:
    """Column ``j`` of a 2-D array or a DataFrame at the fit's float dtype (one column at a time, never the whole matrix)."""
    if hasattr(X, "iloc"):
        return np.asarray(X.iloc[:, j].to_numpy(), dtype=_crit_np_dtype())
    return np.asarray(X[:, j], dtype=_crit_np_dtype())


def _family_codes(X, n_bins: int):
    """Quantile codes of every candidate column as a (K, n) int16 matrix (codes are < n_bins) and the per-column code counts.

    Filled column by column: no full-width float copy of the whole candidate matrix and no int64 code matrix (at 50000 x 3795 those were 1.5 GB each).
    """
    n_rows, n_cols = X.shape
    cols = np.empty((n_cols, n_rows), dtype=np.int16)
    nx = np.empty(n_cols, dtype=np.int64)
    for j in range(n_cols):
        v = _column_values(X, j)
        bad = ~np.isfinite(v)
        if bad.any():
            v = v.copy()
            v[bad] = np.median(v[~bad]) if (~bad).any() else 0.0
        c, k = _quantile_codes(v, n_bins)
        cols[j] = c
        nx[j] = k
    return cols, nx


def family_maxt_pass(candidates: "ArrayLike | pd.DataFrame", y: "ArrayLike", n_bins: int = 10, alpha: float = 0.01, n_perm: int = FAMILY_MAXT_PERMUTATIONS, seed: int = 0) -> np.ndarray:
    """Boolean mask over the candidate columns: True where the exact family-wise permutation test passes at ``alpha``.

    ``candidates`` is a 2-D array-like (rows x candidates) or a DataFrame. Non-finite values are replaced by the column median before binning,
    so a constant or all-NaN candidate simply gets MI 0 and cannot pass.
    """
    X = candidates if hasattr(candidates, "iloc") else np.asarray(candidates)
    if getattr(X, "ndim", 2) == 1:
        X = np.asarray(X)[:, None]
    cols, nx = _family_codes(X, n_bins)
    return family_maxt_pass_codes(cols, nx, y, alpha=alpha, n_perm=n_perm, seed=seed)


def family_maxt_pass_codes(cols: np.ndarray, nx: np.ndarray, y: "ArrayLike", alpha: float = 0.01, n_perm: int = FAMILY_MAXT_PERMUTATIONS, seed: int = 0) -> np.ndarray:
    """:func:`family_maxt_pass` on codes already built by :func:`_family_codes` (a caller that streams its candidates in chunks keeps only
    the int16 codes of every chunk and gates the whole family once)."""
    yv = np.asarray(y).ravel()
    _, y_codes = np.unique(yv, return_inverse=True)
    y_codes = np.ascontiguousarray(y_codes.astype(np.int64).ravel())
    n_y = int(y_codes.max()) + 1
    observed = mi_all_columns_njit(cols, nx, y_codes, n_y)
    return np.asarray(_family_flags(cols, nx, y_codes, n_y, observed, np.arange(observed.shape[0] + 1), alpha, n_perm, seed), dtype=np.bool_)


# Analytic triage of a family before any permutation is drawn. The permutation test spends B x K x n histogram updates to learn what the G-test
# tail already says when a candidate is far from the null: a pair family of 1944 columns at 10000 rows needed 4-7 s per block of 256 permutations
# and 39 blocks to confirm candidates whose MI is 50-100 times the null maximum.
#   pass  : Bonferroni over every column of the family, sum_j P(MI_j >= M) with the chi-square tail of 2*n*M at that column's degrees of freedom.
#           The family maximum can exceed M only if some column does, so this bounds the exact maxT p-value from above; the pass margin below
#           absorbs a tail error of that factor.
#   fail  : the batch's own best column already exceeds its single-column tail by more than _ANALYTIC_FAIL_FACTOR x alpha. The family maximum is at
#           least that column's null MI, so the exact exceedance rate is at least as large and the permutation test would reject.
#   gray  : everything in between goes to the exact permutation test, as before.
# Both bounds rest on the chi-square null of the plug-in MI, which needs dense tables: the family is triaged only when the average expected
# count per cell is at least _ANALYTIC_DENSITY x the sparsity floor of the analytic null; sparse tables use the permutation test alone.
_ANALYTIC_PASS_MARGIN = 0.01
_ANALYTIC_FAIL_FACTOR = 3.0
_ANALYTIC_DENSITY = 4.0


@njit(cache=True, parallel=True)
def _occupied_bins(cols: np.ndarray, n_x: np.ndarray) -> np.ndarray:
    """Number of distinct codes present in every row of ``cols`` (k x n integer codes)."""
    k = cols.shape[0]
    n = cols.shape[1]
    out = np.zeros(k, dtype=np.int64)
    for j in prange(k):
        seen = np.zeros(n_x[j], dtype=np.bool_)
        c = 0
        for i in range(n):
            v = cols[j, i]
            if not seen[v]:
                seen[v] = True
                c += 1
        out[j] = c
    return out


def _analytic_triage(cols, n_x, y, n_y, observed, starts, alpha) -> "np.ndarray | None":
    """Per-batch status from the chi-square null: +1 certified pass, -1 certified fail, 0 undecided; None when the analytic null does not apply.

    ``starts`` are the batch boundaries into ``observed`` (batch b spans ``starts[b]:starts[b + 1]``).
    """
    from ._analytic_mi_null import _chi2_sf, _min_expected_cell, analytic_null_enabled

    if not analytic_null_enabled():
        return None
    n_rows = int(cols.shape[1])
    ny_occ = int(np.count_nonzero(np.bincount(y, minlength=int(n_y))))
    if ny_occ < 2 or n_rows <= 0:
        return None
    nx_occ = _occupied_bins(cols, n_x)
    if n_rows / float(max(int(nx_occ.max()), 1) * ny_occ) < _ANALYTIC_DENSITY * _min_expected_cell():
        return None
    df = (nx_occ - 1) * (ny_occ - 1)
    starts = np.asarray(starts)
    n_batches = len(starts) - 1
    widths = np.diff(starts)
    nonempty = widths > 0
    best = np.zeros(n_batches, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    best_df = np.zeros(n_batches, dtype=np.int64)
    if bool(np.all(widths == 1)):
        best[:] = observed
        best_df[:] = df
    else:
        for b in np.flatnonzero(nonempty):
            seg = observed[starts[b] : starts[b + 1]]
            j = int(np.argmax(seg))
            best[b] = seg[j]
            best_df[b] = df[starts[b] + j]
    g = 2.0 * n_rows * best
    # Bonferroni upper bound: columns are grouped by their degrees of freedom (a handful of distinct values), constant columns never exceed M > 0.
    p_bonf = np.zeros(n_batches, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    for d in np.unique(df[df > 0]):
        # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
        tail = np.asarray(_chi2_sf(g, float(d)), dtype=np.float64)
        p_bonf += float(np.count_nonzero(df == d)) * tail
    p_single = np.ones(n_batches, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    pos = nonempty & (best > 0.0) & (best_df > 0)
    for d in np.unique(best_df[pos]):
        sel = pos & (best_df == d)
        # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
        p_single[sel] = np.asarray(_chi2_sf(g[sel], float(d)), dtype=np.float64)
    status = np.zeros(n_batches, dtype=np.int8)
    status[pos & (p_bonf <= alpha * _ANALYTIC_PASS_MARGIN)] = 1
    status[~pos | (p_single > alpha * _ANALYTIC_FAIL_FACTOR)] = -1
    return status


def _permutation_pass_flags(cols, n_x, y, n_y, observed_max, alpha, n_perm, seed) -> np.ndarray:
    """Exact family-wise permutation test for batches with the given observed maxima (the null spans every column of ``cols``)."""
    rng = np.random.default_rng(seed)
    B = int(n_perm)
    # Probe: when every batch's best MI is at or below the median of the first few null maxima, at least half of the permutations reach it, so the
    # exceedance rate is far above alpha and every batch fails. Half of FAMILY_PROBE_BLOCK draws at or above it is what a true rate under
    # 5% produces with probability about 1e-11, so a small block decides the weak families at a fraction of a full block's cost.
    probe = family_null_max(cols, n_x, y, n_y, min(B, FAMILY_PROBE_BLOCK), rng)
    if observed_max.size and float(observed_max.max()) <= float(np.median(probe)):
        return np.zeros(observed_max.shape[0], dtype=np.bool_)
    first = np.concatenate([probe, family_null_max(cols, n_x, y, n_y, max(min(B, FAMILY_EARLY_BLOCK) - probe.shape[0], 0), rng)])
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
    return np.asarray(alive, dtype=np.bool_)


def _family_flags(cols, n_x, y, n_y, observed, starts, alpha, n_perm, seed) -> list:
    """Pass flag per batch of one family: analytic triage first, the exact permutation test for what it leaves undecided."""
    starts = np.asarray(starts)
    n_batches = len(starts) - 1
    maxima = [float(observed[starts[j] : starts[j + 1]].max()) if starts[j + 1] > starts[j] else 0.0 for j in range(n_batches)]
    # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    observed_max = np.array(maxima, dtype=np.float64)
    flags = np.zeros(n_batches, dtype=np.bool_)
    status = _analytic_triage(cols, n_x, y, n_y, observed, starts, alpha)
    if status is None:
        flags[:] = _permutation_pass_flags(cols, n_x, y, n_y, observed_max, alpha, n_perm, seed)
        return [bool(f) for f in flags]
    flags[status > 0] = True
    gray = status == 0
    if gray.any():
        flags[gray] = _permutation_pass_flags(cols, n_x, y, n_y, observed_max[gray], alpha, n_perm, seed)
    return [bool(f) for f in flags]


def _stack_codes_transposed(mats, widths) -> np.ndarray:
    """The (K_total, n) code matrix of the batches in the narrowest integer width that holds their codes, built in one pass.

    The family of one pair spans every configuration scored for it (hundreds of batches of ~1500 columns at 30000 rows); upcasting each batch to
    int64 and then stacking and transposing it held that family three times at eight bytes a code, which exhausted host memory on a wide frame.
    """
    # bench-attempt-rejected (2026-10-16): one np.concatenate + transposed copy instead of this loop: 0.27 s vs 0.19 s for 1944 single-column int8 batches at 10000 rows.
    n = int(mats[0].shape[0])
    top = max((int(m.max()) for m in mats if m.size), default=0)
    dt = np.int8 if top <= 127 else (np.int16 if top <= 32767 else np.int64)
    cols = np.empty((int(sum(widths)), n), dtype=dt)
    at = 0
    for m, w in zip(mats, widths):
        cols[at : at + w] = m.T
        at += w
    return cols


def family_pass_batches(batches: "Sequence[np.ndarray]", y_codes: np.ndarray, alpha: float = 0.01, n_perm: int = FAMILY_MAXT_PERMUTATIONS, seed: int = 0) -> list:
    """Family-wise pass flags for several batches of candidate codes that form ONE family (e.g. every configuration of one pair).

    ``batches`` is a list of rows x K_b integer code matrices. The null is the maximum MI over the WHOLE family under the same label permutations,
    so a batch cannot pass merely because it was tested on its own. All columns of the family are scored in one kernel call per block of
    permutations. Returns one bool per batch at ``alpha``: batches far from the null are decided by the chi-square triage, the rest by the
    exact permutation test (see :func:`_family_flags`).

    Early rejection: when every undecided batch's best observed MI is at or below the median of the first ``FAMILY_PROBE_BLOCK`` null
    maxima, the remaining permutations are skipped and those batches are rejected.
    """
    y = np.ascontiguousarray(np.asarray(y_codes, dtype=np.int64).ravel())
    n_y = int(y.max()) + 1
    mats = [np.asarray(c) for c in batches]
    widths = [m.shape[1] for m in mats]
    cols = _stack_codes_transposed(mats, widths)
    n_x = cols.max(axis=1).astype(np.int64) + 1
    observed = mi_all_columns_njit(cols, n_x, y, n_y)
    return _family_flags(cols, n_x, y, n_y, observed, np.cumsum([0, *widths]), alpha, n_perm, seed)


__all__ = [
    "FAMILY_EARLY_BLOCK",
    "FAMILY_MAXT_PERMUTATIONS",
    "exact_pvalue_pass",
    "family_maxt_pass",
    "family_null_max",
    "family_pass_batches",
    "mi_all_columns_njit",
]
