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
from typing import TYPE_CHECKING, Any

import numpy as np

from mlframe.feature_selection.filters._family_columns_dedupe import unique_columns
from mlframe._numba_parallel_guard import parallel_kernel_entry
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
from numba import njit, prange

from ._family_codes_njit import quantile_codes_block
from ._family_maxt_null_v2_njit import (
    BITSET_MAX_BINS,
    BITSET_MAX_CLASSES,
    bitset_groups,
    label_bitsets_batch,
    label_bitsets_from_index,
    mi_all_columns_counts as mi_all_columns_njit,
    null_max_bitsets,
)

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


_FAMILY_NULL_CODE_VERSION = "family_null_backend-v2-bitsets"


def _run_family_null_sweep():
    """Per-host CPU-vs-device sweep of the family null over (n_rows, n_cols); returns [] when the helper is unavailable."""
    try:
        from pyutilz.dev.benchmarking import sweep_backend_grid
    except Exception as exc:
        import logging

        logging.getLogger("mlframe.feature_selection.filters.mrmr").debug("family null sweep unavailable (%s: %s); no regions recorded", type(exc).__name__, exc)
        return []

    def _cpu(c, nx, p, ny):
        """Host bit-set null (the kernel family_null_max runs on the host)."""
        return null_max_bitsets(bitset_groups(c, nx), nx, p, ny)

    def _device(c, nx, p, ny):
        """Device bit-set null, including the upload of the column bit sets."""
        import cupy as cp

        from ._family_maxt_cuda_bits import DeviceBitsets, null_max_bits_device

        prep = bitset_groups(c, nx)
        n = c.shape[1]
        return null_max_bits_device(DeviceBitsets(cp, prep, nx, n), label_bitsets_batch(p, ny, (n + 63) // 64), _label_const(prep[0], p[0], ny))

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
            {"cpu": _cpu, "device": _device},
            {"n_rows": [2000, 20000], "n_cols": [200, 2000]},
            _make_inputs,
            reference="cpu",
            repeats=2,
            equiv_rtol=1e-12,
            equiv_atol=1e-12,
            decision_key="kernel_choice",
        )
        or []
    )


@njit(cache=True)
def _mi_one_column_into(col: np.ndarray, n_x: int, y_codes: np.ndarray, n_y: int, joint: np.ndarray) -> float:
    """Plug-in MI of one column, counting into a caller-owned buffer (no allocation per column and permutation).

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


@njit(cache=True, parallel=True)
def _fill_permutations(idx: np.ndarray, base_seed: int) -> None:
    """Fisher-Yates permutations of ``range(n)`` into the rows of ``idx``, one parallel task per row; row ``i`` is seeded ``base_seed + i`` so it does not depend on the thread schedule."""
    m = idx.shape[0]
    n = idx.shape[1]
    for r in prange(m):
        np.random.seed((base_seed + r) % 4294967295)
        for i in range(n):
            idx[r, i] = i
        for i in range(n - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            tmp = idx[r, i]
            idx[r, i] = idx[r, j]
            idx[r, j] = tmp


def _permutation_block(rng: np.random.Generator, y: np.ndarray, m: int) -> np.ndarray:
    """Row-permutation indices of the next ``m`` label permutations of the stream defined by ``rng``.

    One draw is taken from ``rng`` (the block seed); permutation ``i`` of the block is the Fisher-Yates shuffle seeded ``seed + i``, generated in parallel (50,000 sequential
    ``rng.permutation`` calls were 19.5 s of a 146 s CPU fit at n=20000). The block is a function of the generator state alone, as the cache below needs.

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
        _fill_permutations(idx, int(rng.integers(0, 2**31 - 1)))
        _perm_block_cache[key] = (idx, rng.bit_generator.state)
        _perm_block_cache_bytes += idx.nbytes
        while _perm_block_cache_bytes > _PERM_BLOCK_CACHE_BYTES and len(_perm_block_cache) > 1:
            _, (old, _) = _perm_block_cache.popitem(last=False)
            _perm_block_cache_bytes -= old.nbytes
    return idx


def _device_null_oom_errors() -> tuple[type[BaseException], ...]:
    """Exceptions that mean the device could not hold this null block (pool limit or allocation failure), so the host kernel takes over.

    Narrow on purpose: only cupy's out-of-memory and runtime errors, never a bug in the scoring code."""
    try:
        from cupy.cuda import memory as _cp_memory
        from cupy_backends.cuda.api import runtime as _cp_runtime

        return (_cp_memory.OutOfMemoryError, _cp_runtime.CUDARuntimeError)
    except ImportError:
        return ()


def _null_device_wanted(cols: np.ndarray, y: np.ndarray, n_perm: int) -> bool:
    """Whether the family null for this call runs on the device (the same decision ``family_null_max`` takes for its blocks)."""
    from ._fe_gpu_strict import fe_gpu_strict_enabled
    from ._gpu_policy import gpu_globally_disabled

    _work = int(n_perm) * int(cols.shape[0]) * int(y.shape[0])
    if gpu_globally_disabled():
        # MLFRAME_DISABLE_GPU=1 / CUDA_VISIBLE_DEVICES="" is the project-wide "no GPU on this run": it wins over STRICT and over the tuned backend choice (the tuned choice was "device" on this
        # host and a "pure CPU" fit still ran the null on the GPU, 3.2 s of device work in the CPU-mode profile).
        return False
    if fe_gpu_strict_enabled():
        return True
    return _family_null_backend(int(y.shape[0]), int(cols.shape[0]), _work) == "device"


def family_null_max(cols: np.ndarray, n_x: np.ndarray, y: np.ndarray, n_y: int, n_perm: int, rng: np.random.Generator, block: int = 256, acc: Any = None) -> np.ndarray | None:
    """Null maxima over ``n_perm`` label permutations drawn from ``rng`` in the same order as a per-permutation loop (results are identical).

    ``acc``: an optional device float64 vector of length ``n_perm``. The maxima are folded into it (elementwise, in place) and the
    function returns ``None``, so a caller that scores several column groups against the same draw reads the result back once."""
    out = np.empty(int(n_perm), dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    # STRICT GPU mode (MLFRAME_FE_GPU_STRICT=1, or AUTO at scale) runs the null on the device; the permutations are drawn on the host in
    # the same order either way, so the decisions do not depend on the path.
    from ._family_maxt_cuda import device_supported, family_null_max_device

    # Size-gated device dispatch: the null costs B*K*n count updates, so a large family goes to the device even outside STRICT (the
    # pair family gate runs 9999 permutations; on the CPU an n=30000 fit exceeded the 600 s test watchdog). The floor is a measured
    # crossover on the dev GPU and belongs in the kernel-tuning cache; TODO(KTC).
    want_device = _null_device_wanted(cols, y, n_perm)
    # Bin x class counts as popcounts of bit sets (host and device): the column bit sets are built once per call and reused by every block;
    # columns with many bins or labels with many classes keep the histogram kernels.
    bit_prep = None
    dev_bits = None
    if cols.shape[0] and int(n_x.max()) <= BITSET_MAX_BINS and int(n_y) <= BITSET_MAX_CLASSES:
        # Parallel=True column-prange kernels: reachable from the pair sweep's joblib threads; see mlframe._numba_parallel_guard.
        with parallel_kernel_entry():
            bit_prep = bitset_groups(cols, n_x)
        if want_device:
            dev_bits = _device_bitsets(bit_prep, n_x, int(y.shape[0]))
    use_device = want_device and bit_prep is None and device_supported(n_x, int(n_y))
    py_const = _label_const(bit_prep[0], y, int(n_y)) if (dev_bits is not None and bit_prep is not None) else 0.0
    words = (int(y.shape[0]) + 63) // 64
    for start in range(0, int(n_perm), block):
        m = min(block, int(n_perm) - start)
        perm_idx = _permutation_block(rng, y, m)
        # The bitset / njit branches run parallel=True prange kernels; the GPU branches do not, but holding the guard
        # across the whole block keeps the null's kernel entries serialised with every other FE fan-out thread.
        with parallel_kernel_entry():
            if dev_bits is not None:
                from ._family_maxt_cuda_bits import null_max_bits_device

                # The class bit sets are packed straight from the permutation indices: the (block, n) matrix of permuted labels is never built.
                labels = label_bitsets_from_index(y, perm_idx, int(n_y), words)
                if acc is not None:
                    null_max_bits_device(dev_bits, labels, py_const, acc=acc[start : start + m])
                    continue
                out[start : start + m] = null_max_bits_device(dev_bits, labels, py_const)
                continue
            perms = y[perm_idx]
            if bit_prep is not None:
                out[start : start + m] = null_max_bitsets(bit_prep, n_x, perms, int(n_y))
            elif use_device:
                try:
                    out[start : start + m] = family_null_max_device(cols, n_x, perms, int(n_y))
                except _device_null_oom_errors():
                    # The device pool's limit (or a busy card) refused this block: the same maxima come from the host kernel,
                    # since the permutations were drawn on the host in the same order either way.
                    out[start : start + m] = family_null_max_njit(cols, n_x, perms, int(n_y))
            else:
                out[start : start + m] = family_null_max_njit(cols, n_x, perms, int(n_y))
            if acc is not None:
                import cupy as cp

                cp.maximum(acc[start : start + m], cp.asarray(out[start : start + m]), out=acc[start : start + m])
    if acc is not None:
        return None
    return out


def _device_bitsets(prep, n_x: np.ndarray, n: int):
    """The family's column bit sets on the device, or ``None`` when there is no usable device (the host bit sets then count)."""
    try:
        import cupy as cp

        from ._family_maxt_cuda_bits import DeviceBitsets

        return DeviceBitsets(cp, prep, n_x, n)
    except Exception as exc:  # no cupy / no device / out of memory: the host kernel gives the same maxima
        import logging

        logging.getLogger("mlframe.feature_selection.filters.mrmr").debug("device bit sets unavailable (%r); counting on the host", exc)
        return None


def _label_const(table: np.ndarray, y: np.ndarray, n_y: int) -> float:
    """``n log n - sum_b xlogx(py_b)`` of a label vector (the same for every permutation of it)."""
    py = np.bincount(np.asarray(y, dtype=np.int64), minlength=n_y)
    return float(table[y.shape[0]] - sum(float(table[c]) for c in py))


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
    """Integer quantile-bin codes of a finite column and the number of codes it uses (exact codes for a column with at most ``n_bins`` distinct values)."""
    levels = np.unique(v)
    if levels.size <= n_bins:
        # The quantile edges of a two-valued column are [lo, hi] and the upper bin is clipped away: both values got code 0, MI 0, and a binary
        # candidate could never pass the family test.
        return np.searchsorted(levels, v).astype(np.int64), int(levels.size)
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
    block = max(1, min(n_cols, (128 << 20) // (4 * max(n_rows, 1))))  # a block of float columns of at most 128 MB, coded in parallel
    for j0 in range(0, n_cols, block):
        j1 = min(j0 + block, n_cols)
        v = np.empty((j1 - j0, n_rows), dtype=_crit_np_dtype())
        for j in range(j0, j1):
            col = _column_values(X, j)
            bad = ~np.isfinite(col)
            if bad.any():
                col = col.copy()
                col[bad] = np.median(col[~bad]) if (~bad).any() else 0.0
            v[j - j0] = col
        with parallel_kernel_entry():
            quantile_codes_block(v, int(n_bins), cols[j0:j1], nx[j0:j1])
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
    from ._y_encoding import encode_y_for_classif_mi

    # A continuous target is quantile-binned first: against one class per row the "MI" is the entropy of the candidate's own binning, the same under every
    # permutation, so no candidate could pass the family test.
    yv = encode_y_for_classif_mi(np.asarray(y).ravel())
    _, y_codes = np.unique(yv, return_inverse=True)
    y_codes = np.ascontiguousarray(y_codes.astype(np.int64).ravel())
    n_y = int(y_codes.max()) + 1
    with parallel_kernel_entry():
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


def _analytic_triage(cols, n_x, y, n_y, observed, starts, alpha, inverse=None) -> "np.ndarray | None":
    """Per-batch status from the chi-square null: +1 certified pass, -1 certified fail, 0 undecided; None when the analytic null does not apply.

    ``starts`` are the batch boundaries into ``observed`` (batch b spans ``starts[b]:starts[b + 1]``). ``cols`` / ``n_x`` hold the DISTINCT columns of the family and ``inverse`` maps each column
    of ``observed`` to its row in them (``None``: no repeated columns). The Bonferroni sum keeps counting every column of the family: the chi-square tail is an approximation, and the pass margin
    was calibrated against that count (summing over the distinct columns only let the batches through that the exact test rejects).
    """
    from ._analytic_mi_null import analytic_null_enabled

    if not analytic_null_enabled():
        return None
    n_rows = int(cols.shape[1])
    if n_rows <= 0:
        return None
    with parallel_kernel_entry():
        nx_occ = _occupied_bins(cols, n_x)
    return _analytic_triage_core(n_rows, nx_occ, y, n_y, observed, starts, alpha, inverse)


def _analytic_triage_core(n_rows, nx_occ, y, n_y, observed, starts, alpha, inverse=None) -> "np.ndarray | None":
    """:func:`_analytic_triage` from the row count and the occupied-bin count of every distinct column, so a caller that holds the columns on the device needs no host copy of them."""
    from ._analytic_mi_null import _chi2_sf, _min_expected_cell

    ny_occ = int(np.count_nonzero(np.bincount(y, minlength=int(n_y))))
    if ny_occ < 2 or n_rows <= 0:
        return None
    if n_rows / float(max(int(nx_occ.max()), 1) * ny_occ) < _ANALYTIC_DENSITY * _min_expected_cell():
        return None
    df_distinct = (nx_occ - 1) * (ny_occ - 1)
    df = df_distinct if inverse is None else df_distinct[inverse]
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
    return _permutation_pass_flags_with(lambda m, rng: family_null_max(cols, n_x, y, n_y, m, rng), observed_max, alpha, n_perm, seed)


def _permutation_pass_flags_with(null_max, observed_max, alpha, n_perm, seed) -> np.ndarray:
    """:func:`_permutation_pass_flags` with the null supplied as ``null_max(m, rng)``: the ``m`` next null maxima over the whole family, drawn from ``rng``."""
    rng = np.random.default_rng(seed)
    B = int(n_perm)
    # Probe: when every batch's best MI is at or below the median of the first few null maxima, at least half of the permutations reach it, so the
    # exceedance rate is far above alpha and every batch fails. Half of FAMILY_PROBE_BLOCK draws at or above it is what a true rate under
    # 5% produces with probability about 1e-11, so a small block decides the weak families at a fraction of a full block's cost.
    probe = null_max(min(B, FAMILY_PROBE_BLOCK), rng)
    if observed_max.size and float(observed_max.max()) <= float(np.median(probe)):
        return np.zeros(observed_max.shape[0], dtype=np.bool_)
    first = np.concatenate([probe, null_max(max(min(B, FAMILY_EARLY_BLOCK) - probe.shape[0], 0), rng)])
    # Exact early stop per batch: the pass test is (1 + R) / (B + 1) <= alpha with R the count of null maxima at or above the batch's
    # observed maximum. R only grows, so a batch whose count already violates the bound fails for every B; the null still spans the whole
    # family, but the permutation loop ends as soon as no batch is alive.
    R = np.zeros(observed_max.shape[0], dtype=np.int64)
    R += (first[:, None] >= observed_max[None, :]).sum(axis=0)
    done = first.shape[0]
    alive = (1.0 + R) / (B + 1.0) <= float(alpha)
    while done < B and alive.any():
        m = min(FAMILY_EARLY_BLOCK, B - done)
        blk = null_max(m, rng)
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
    # The tests depend on the set of DISTINCT columns (see _family_columns_dedupe): a family of operator configurations repeats many code columns.
    distinct = unique_columns(cols)
    inverse = None
    if distinct is not None:
        keep, inverse = distinct
        cols, n_x = cols[keep], n_x[keep]
    status = _analytic_triage(cols, n_x, y, n_y, observed, starts, alpha, inverse)
    if status is None:
        flags[:] = _permutation_pass_flags(cols, n_x, y, n_y, observed_max, alpha, n_perm, seed)
        return [bool(f) for f in flags]
    flags[status > 0] = True
    gray = status == 0
    if gray.any():
        flags[gray] = _permutation_pass_flags(cols, n_x, y, n_y, observed_max[gray], alpha, n_perm, seed)
    return [bool(f) for f in flags]


def _columns_contiguous(a: np.ndarray) -> bool:
    """Whether the columns of the 2-D block ``a`` follow each other in memory within a row; the stride of an axis of length 1 carries no meaning (a single-column view may report any value)."""
    return bool(a.shape[1] <= 1 or a.strides[1] == a.itemsize)


def _column_runs(mats, widths) -> list:
    """``(block, merged)`` pairs: the batches as 2-D blocks, with every run of batches that are adjacent column blocks of one array merged into a single strided view (``merged`` True).

    The pair family passes one single-column view per configuration, all cut from one code matrix in order; copying them one by one reads each column at the row stride of the whole matrix
    and starts a parallel region per column. Two batches merge when they have the same dtype, row count and strides, the columns are contiguous within a row, and the second starts exactly one
    batch width after the first in memory, so the merged view addresses nothing outside the batches' own memory. Anything else stays a block of its own.
    """
    runs: list = []
    for m, w in zip(mats, widths):
        if runs:
            head, width = runs[-1]
            if (
                m.ndim == 2
                and head.ndim == 2
                and m.dtype == head.dtype
                and m.shape[0] == head.shape[0]
                and m.strides[0] == head.strides[0]
                and _columns_contiguous(m)
                and _columns_contiguous(head)
                and m.__array_interface__["data"][0] == head.__array_interface__["data"][0] + width * head.itemsize
            ):
                runs[-1][1] = width + int(w)
                continue
        runs.append([m, int(w)])
    blocks = []
    for head, width in runs:
        if width == head.shape[1]:
            blocks.append((head, False))
        else:
            blocks.append((np.lib.stride_tricks.as_strided(head, shape=(head.shape[0], width), strides=(head.strides[0], head.itemsize), writeable=False), True))
    return blocks


def _stack_codes_transposed(mats, widths) -> np.ndarray:
    """The (K_total, n) code matrix of the batches in the narrowest integer width that holds their codes, built in one pass.

    The family of one pair spans every configuration scored for it (hundreds of batches of ~1500 columns at 30000 rows); upcasting each batch to
    int64 and then stacking and transposing it held that family three times at eight bytes a code, which exhausted host memory on a wide frame.
    Batches that are adjacent column blocks of one array are transposed together (see :func:`_column_runs`).
    """
    # bench-attempt-rejected (2026-10-16): one np.concatenate + transposed copy instead of this loop: 0.27 s vs 0.19 s for 1944 single-column int8 batches at 10000 rows.
    n = int(mats[0].shape[0])
    blocks = _column_runs(mats, widths)
    top = max((int(b.max()) for b, _ in blocks if b.size), default=0)
    dt = np.int8 if top <= np.iinfo(np.int8).max else (np.int16 if top <= np.iinfo(np.int16).max else np.int64)
    cols = np.empty((int(sum(widths)), n), dtype=dt)
    at = 0
    for b, merged in blocks:
        # A parallel blocked transpose into the shared buffer: numpy's strided ``m.T`` copy was the largest host cost of the family gate on a wide pair pool. A merged strided block is read
        # in place (rows of a tile are contiguous in memory); a batch on its own is made contiguous first, as before.
        with parallel_kernel_entry():
            _transpose_into(b if merged else np.ascontiguousarray(b), cols, at)
        at += int(b.shape[1])
    return cols


@njit(cache=True, parallel=True)
def _transpose_into(src, dst, at):
    """``dst[at + j, i] = src[i, j]`` for every element of the ``(n, w)`` block ``src``, in cache-sized tiles, parallel over the rows of ``dst``."""
    n, w = src.shape
    tile = 64
    for j0 in prange((w + tile - 1) // tile):
        j_lo = j0 * tile
        j_hi = min(j_lo + tile, w)
        for i0 in range(0, n, tile):
            i_hi = min(i0 + tile, n)
            for j in range(j_lo, j_hi):
                for i in range(i0, i_hi):
                    dst[at + j, i] = src[i, j]


def family_pass_batches(batches: "Sequence[np.ndarray]", y_codes: np.ndarray, alpha: float = 0.01, n_perm: int = FAMILY_MAXT_PERMUTATIONS, seed: int = 0) -> list:
    """Family-wise pass flags for several batches of candidate codes that form ONE family (e.g. every configuration of one pair).

    ``batches`` is a list of rows x K_b integer code matrices. The null is the maximum MI over the WHOLE family under the same label permutations,
    so a batch cannot pass merely because it was tested on its own. All columns of the family are scored in one kernel call per block of
    permutations. Returns one bool per batch at ``alpha``: batches far from the null are decided by the chi-square triage, the rest by the
    exact permutation test (see :func:`_family_flags`).

    Early rejection: when every undecided batch's best observed MI is at or below the median of the first ``FAMILY_PROBE_BLOCK`` null
    maxima, the remaining permutations are skipped and those batches are rejected.
    """
    # Resident device batches are scored on the device without copying them back; the host gate below answers whenever that path does not apply.
    from ._family_maxt_gate_device import family_pass_batches_device

    on_device = family_pass_batches_device(batches, y_codes, alpha, n_perm, seed)
    if on_device is not None:
        return on_device
    y = np.ascontiguousarray(np.asarray(y_codes, dtype=np.int64).ravel())
    n_y = int(y.max()) + 1
    from ._feature_engineering_pairs._pairs_chunk_codes import resolve_code_columns

    mats = [np.asarray(c) for c in resolve_code_columns(batches)]  # lazy pair-chunk columns come back in one transfer per chunk
    widths = [m.shape[1] for m in mats]
    # A family whose stacked code matrix does not fit the free host memory is scored one group of columns at a time (same decisions, see _family_maxt_stream).
    from ._family_maxt_stream import family_pass_streamed, plan_stream_groups

    groups = plan_stream_groups(mats, widths, int(mats[0].shape[0]))
    if groups is not None:
        return family_pass_streamed(mats, widths, y, n_y, alpha, n_perm, seed, groups)
    cols = _stack_codes_transposed(mats, widths)
    n_x = cols.max(axis=1).astype(np.int64) + 1
    with parallel_kernel_entry():
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
