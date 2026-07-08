"""Numba-JIT inner loop for the per-base near-collinear keep-mask walk.

``near_collinear_keep_mask`` (in ``_eval_stats``) de-duplicates a base's
``x_remaining`` by walking columns left to right and dropping any column whose
absolute Pearson correlation with an already-kept column exceeds a threshold.
That walk is ``O(B^2)`` pure-numpy Pearson + boolean fancy-indexing; on a base
with 50+ candidate remaining columns it is a measurable hot spot inside the MI
baseline sweep ``fit`` runs once per screened base.

This module carves the pair-walk into a ``numba.njit`` kernel
(:func:`_keep_mask_kernel`) that operates on the full ``(n, B)`` matrix plus a
precomputed finite mask, computing each pair's correlation in registers with no
per-pair fancy-index allocation. The public entry point is
:func:`near_collinear_keep_mask_fast`, a size-aware dispatcher: it routes tiny
inputs (``B < _MIN_COLS`` or ``n < _MIN_ROWS``) to the numpy reference (the JIT
overhead never amortises there per the numba ladder) and larger inputs to the
JIT kernel.

Bit-identity contract
---------------------
The kernel's per-pair reductions (``mean`` / ``dot`` lowered by numba) can differ
from numpy's BLAS / pairwise-reduce by ~1 ULP. That ULP can only flip a
keep/drop decision when a pair's correlation lands within ~1e-12 of the
threshold -- which is measure-zero on continuous data but reachable on tied /
discrete columns. The kernel therefore reports any pair whose correlation lands
in a tiny band around the threshold as "borderline"; the wrapper re-decides ONLY
those borderline pairs with the EXACT numpy primitives the reference uses
(``a.mean()`` / ``np.dot``). Every non-borderline decision is already separated
from the threshold by far more than the ULP noise, so the resulting mask is
bit-identical to :func:`near_collinear_keep_mask` for all inputs -- continuous,
discrete, tied, NaN-holed, and exact-duplicate (corr exactly 1.0). The
regression test pins this across seeds plus a degenerate constant column.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import numpy as np

try:
    import numba as _numba

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numba is a hard dep; allow graceful skip.
    _numba = None  # type: ignore
    _HAS_NUMBA = False

# Per-suite keep-mask cache. The keep-mask is a pure function of the (base-dropped)
# feature matrix + threshold; discovery runs once per target, so when two targets
# share the byte-identical matrix for a base (the common case -- the leakage filter
# usually drops the same columns) the O(B^2 * n) walk is recomputed identically.
# Key = blake2b of the contiguous buffer + shape + dtype + threshold -> collision-safe
# so a hit returns ONLY a byte-identical recompute (bit-identical by construction).
# The hash is O(matrix bytes) but the kernel is O(B^2 * n) (~17 s at 80k x 120 vs
# ~50 ms hash), so a miss never regresses. Bounded FIFO + per-entry size cap so a
# 100 GB workload never pins large masks (the mask itself is just n_cols bools).
_KEEP_MASK_CACHE: "OrderedDict[Any, np.ndarray]" = OrderedDict()
_KEEP_MASK_CACHE_MAX_ENTRIES: int = 64
_KEEP_MASK_HASH_MAX_BYTES: int = 2_000_000_000  # skip caching beyond ~2 GB matrices.


def _keep_mask_cache_key(fm: np.ndarray, thr: float):
    """Collision-safe content signature for the keep-mask cache, or None to skip.

    Hashes the full contiguous buffer (no source-frame copy -- ``fm`` is already an
    ascontiguousarray the kernel owns). Returns None above the byte cap so very large
    matrices recompute rather than pay a multi-GB hash.
    """
    nbytes = int(fm.nbytes)
    if nbytes == 0 or nbytes > _KEEP_MASK_HASH_MAX_BYTES:
        return None
    h = hashlib.blake2b(digest_size=16)
    h.update(fm.tobytes())
    return (fm.shape, fm.dtype.str, float(thr), h.digest())

# Numba JIT wins the O(B^2) pair walk only once both the column count and the
# row count are large enough to amortise the ~register-loop setup vs numpy's
# vectorised dot; below these the numpy reference's lower per-call overhead wins
# (and the one-shot JIT compile never pays back on a handful of columns).
_MIN_COLS: int = 10
_MIN_ROWS: int = 256

# Half-width of the "borderline" band around the threshold. A pair whose
# kernel-computed |corr| lands within this of the threshold is re-decided with
# the exact numpy primitives so a ~1 ULP kernel/BLAS reduction difference can
# never flip the keep/drop decision. Far wider than the ~1e-15 ULP noise yet
# narrow enough that borderline pairs are vanishingly rare on real data.
_BORDERLINE_BAND: float = 1e-9

# Variance floor below which a column is "constant on the joint support" -> no
# correlation (kept). Matches the numpy reference exactly.
_VAR_FLOOR: float = 1e-24


if _HAS_NUMBA:

    @_numba.njit(cache=True, fastmath=False, parallel=True)
    def _column_stats_allfinite(fm):  # type: ignore[no-untyped-def]
        """Per-column mean + centred sum-of-squares for an all-finite matrix.

        One parallel pass over the (n, B) matrix; ``var[j]`` is the centred
        sum-of-squares (NOT divided by n) so a later cross-term divides by the
        same un-normalised scale, matching the reference's ``dot(dev, dev)``.
        """
        n = fm.shape[0]
        n_cols = fm.shape[1]
        mean = np.empty(n_cols, dtype=np.float64)
        var = np.empty(n_cols, dtype=np.float64)
        for j in _numba.prange(n_cols):
            s = 0.0
            for i in range(n):
                s += fm[i, j]
            m = s / n
            v = 0.0
            for i in range(n):
                d = fm[i, j] - m
                v += d * d
            mean[j] = m
            var[j] = v
        return mean, var

    @_numba.njit(cache=True, fastmath=False)
    def _keep_mask_kernel_allfinite(fm, mean, var, thr, band):  # type: ignore[no-untyped-def]
        """All-finite fast path: single cross-term pass per kept pair.

        Equivalent to :func:`_keep_mask_kernel` when every row is finite, but
        with the per-pair mean/variance pulled from the precomputed ``mean`` /
        ``var`` arrays so each pair costs ONE pass over n (the cross term) rather
        than two. ``var[k]`` is the centred sum-of-squares (un-normalised),
        matching the reference. Borderline pairs are still flagged for exact
        numpy re-decision so the mask stays bit-identical.
        """
        n = fm.shape[0]
        n_cols = fm.shape[1]
        keep = np.ones(n_cols, dtype=np.int8)
        borderline = np.zeros(n_cols, dtype=np.int8)
        kept_idx = np.empty(n_cols, dtype=np.int64)
        n_kept = 0
        for j in range(n_cols):
            drop = False
            j_borderline = False
            mj = mean[j]
            vj = var[j]
            for ki in range(n_kept):
                k = kept_idx[ki]
                vk = var[k]
                if vj < _VAR_FLOOR or vk < _VAR_FLOOR:
                    continue
                mk = mean[k]
                vab = 0.0
                for i in range(n):
                    vab += (fm[i, j] - mj) * (fm[i, k] - mk)
                corr = abs(vab / np.sqrt(vj * vk))
                if abs(corr - thr) <= band:
                    j_borderline = True
                    drop = True
                    break
                if corr > thr:
                    drop = True
                    break
            if drop:
                keep[j] = 0
                if j_borderline:
                    borderline[j] = 1
            else:
                kept_idx[n_kept] = j
                n_kept += 1
        return keep, borderline

    @_numba.njit(cache=True, fastmath=False)
    def _keep_mask_kernel(fm, finite, thr, band):  # type: ignore[no-untyped-def]
        """Walk columns left to right, return (keep, borderline) int8 arrays.

        ``keep[j]`` is 1 when column j survives the kernel's own decision.
        ``borderline[j]`` is 1 when j's DROP/keep hinged on a pair whose |corr|
        was within ``band`` of ``thr`` (the wrapper rechecks those exactly).

        Reductions here mirror the numpy reference's two-pass mean + dot but are
        lowered by numba, so a non-borderline decision is trusted directly while
        a borderline one is flagged for exact numpy re-decision.
        """
        n = fm.shape[0]
        n_cols = fm.shape[1]
        keep = np.ones(n_cols, dtype=np.int8)
        borderline = np.zeros(n_cols, dtype=np.int8)
        # Indices of columns kept so far (compacted into the front of kept_idx).
        kept_idx = np.empty(n_cols, dtype=np.int64)
        n_kept = 0
        for j in range(n_cols):
            drop = False
            j_borderline = False
            for ki in range(n_kept):
                k = kept_idx[ki]
                # First pass: count jointly-finite rows + accumulate the two
                # means over exactly those rows.
                n_pair = 0
                sa = 0.0
                sb = 0.0
                for i in range(n):
                    if finite[i, j] and finite[i, k]:
                        sa += fm[i, j]
                        sb += fm[i, k]
                        n_pair += 1
                if n_pair < 3:
                    continue
                ma = sa / n_pair
                mb = sb / n_pair
                va = 0.0
                vb = 0.0
                vab = 0.0
                for i in range(n):
                    if finite[i, j] and finite[i, k]:
                        da = fm[i, j] - ma
                        db = fm[i, k] - mb
                        va += da * da
                        vb += db * db
                        vab += da * db
                if va < _VAR_FLOOR or vb < _VAR_FLOOR:
                    continue
                corr = abs(vab / np.sqrt(va * vb))
                if abs(corr - thr) <= band:
                    # Too close to the threshold to trust the lowered reduction;
                    # the wrapper will re-decide this column exactly.
                    j_borderline = True
                    drop = True
                    break
                if corr > thr:
                    drop = True
                    break
            if drop:
                keep[j] = 0
                if j_borderline:
                    borderline[j] = 1
            else:
                kept_idx[n_kept] = j
                n_kept += 1
        return keep, borderline


def _ref_pair_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    """Exact numpy |Pearson| for one jointly-finite pair, matching the reference.

    Returns ``None`` for the "skip" cases (constant on the joint support) so the
    caller treats the pair as uncorrelated, identical to the numpy reference.
    """
    a_dev = a - a.mean()
    b_dev = b - b.mean()
    va = float(np.dot(a_dev, a_dev))
    vb = float(np.dot(b_dev, b_dev))
    if va < _VAR_FLOOR or vb < _VAR_FLOOR:
        return None
    return abs(float(np.dot(a_dev, b_dev)) / np.sqrt(va * vb))


def _recheck_column_exact(
    feature_matrix: np.ndarray,
    finite: np.ndarray,
    j: int,
    kept_idx: list[int],
    thr: float,
) -> bool:
    """Exact numpy re-decision for one borderline column ``j``.

    Returns True when ``j`` is DROPPED (correlates above ``thr`` with a kept
    column), using the identical arithmetic of the numpy reference so the
    decision is bit-identical to ``near_collinear_keep_mask``.
    """
    fin_j = finite[:, j]
    col_j = feature_matrix[:, j]
    for k in kept_idx:
        pair = fin_j & finite[:, k]
        if int(pair.sum()) < 3:
            continue
        corr = _ref_pair_corr(col_j[pair], feature_matrix[:, k][pair])
        if corr is None:
            continue
        if corr > thr:
            return True
    return False


def near_collinear_keep_mask_fast(
    feature_matrix: np.ndarray,
    *,
    corr_threshold: float,
    reference_fn,
) -> np.ndarray:
    """Size-aware dispatcher for the near-collinear keep-mask walk.

    Routes tiny inputs to ``reference_fn`` (the pure-numpy reference) and larger
    inputs to the numba kernel, then re-decides any borderline column exactly so
    the returned mask is bit-identical to the reference for all inputs.
    ``reference_fn`` is injected (rather than imported) to avoid an import cycle
    with ``_eval_stats``.
    """
    if feature_matrix.ndim != 2:
        raise ValueError("near_collinear_keep_mask expects a 2-D matrix")
    n_rows, n_cols = feature_matrix.shape
    thr = float(corr_threshold)
    # Cheap exits identical to the reference (also avoids JIT on trivial inputs).
    if n_cols < 2 or n_rows < 3 or not (thr < 1.0):
        return np.ones(n_cols, dtype=bool)
    if not _HAS_NUMBA:
        return np.asarray(reference_fn(feature_matrix, corr_threshold=thr))
    # Backend choice goes through the kernel_tuning_cache (per-host measured crossover)
    # with the hardcoded _MIN_COLS / _MIN_ROWS gate as the fallback + an env-var
    # force-override; bit-identical either way (borderline columns re-decided exactly).
    from ._ktc_dispatch import choose_collinear_backend

    if choose_collinear_backend(n_rows, n_cols, min_rows=_MIN_ROWS, min_cols=_MIN_COLS) == "numpy":
        return np.asarray(reference_fn(feature_matrix, corr_threshold=thr))

    fm = np.ascontiguousarray(feature_matrix, dtype=np.float64)
    # Content-keyed cache hit -> a byte-identical matrix already produced this exact
    # mask; return a copy so the caller can mutate freely without corrupting the entry.
    _ck = _keep_mask_cache_key(fm, thr)
    if _ck is not None:
        _hit = _KEEP_MASK_CACHE.get(_ck)
        if _hit is not None:
            _KEEP_MASK_CACHE.move_to_end(_ck)
            return np.asarray(_hit.copy())
    finite = np.isfinite(fm)
    # All-finite fast path (the common case after the leakage filter): precompute
    # per-column mean+ssq once, then each kept pair costs ONE cross-term pass over
    # n instead of the two-pass per-pair mean/variance recompute. Borderline pairs
    # are still re-decided exactly below, so the mask is bit-identical either way.
    if finite.all():
        mean, var = _column_stats_allfinite(fm)
        keep_i8, borderline_i8 = _keep_mask_kernel_allfinite(fm, mean, var, thr, _BORDERLINE_BAND)
    else:
        keep_i8, borderline_i8 = _keep_mask_kernel(fm, finite, thr, _BORDERLINE_BAND)
    keep = keep_i8.astype(bool)
    if borderline_i8.any():
        # Replay the left-to-right walk, re-deciding only the borderline columns with
        # the exact numpy primitives. The kept set is rebuilt in column order so a
        # re-decided KEEP can correctly shadow a later borderline column too.
        kept_idx: list[int] = []
        for j in range(n_cols):
            if borderline_i8[j]:
                dropped = _recheck_column_exact(fm, finite, j, kept_idx, thr)
                keep[j] = not dropped
            if keep[j]:
                kept_idx.append(j)
    if _ck is not None:
        if len(_KEEP_MASK_CACHE) >= _KEEP_MASK_CACHE_MAX_ENTRIES:
            _KEEP_MASK_CACHE.popitem(last=False)
        _KEEP_MASK_CACHE[_ck] = keep.copy()
    return np.asarray(keep)


if _HAS_NUMBA:
    @_numba.njit(cache=True)
    def _block_gather_kernel(arr, perm, block_len):  # type: ignore[no-untyped-def]
        # Materialise a block-permuted copy of ``arr`` in one pass: emit each permuted block's
        # real (in-bounds) elements in order, dropping the trailing short block's padding. This
        # fuses the index build + gather of the prior numpy path (broadcast (n_blocks, block_len)
        # template -> ravel -> boolean ``idx < m`` mask -> ``arr[idx]``), eliminating the O(n_blocks*block_len)
        # int64 temp + mask. Element order is identical to the numpy path for the same ``perm`` draw,
        # so the block shuffle is bit-identical for both the int64 bin-code and the float value path.
        m = arr.shape[0]
        out = np.empty(m, dtype=arr.dtype)
        w = 0
        for bi in range(perm.shape[0]):
            start = perm[bi] * block_len
            for j in range(block_len):
                pos = start + j
                if pos < m:
                    out[w] = arr[pos]
                    w += 1
        return out


def block_shuffle_gather(arr: np.ndarray, perm: np.ndarray, block_len: int) -> np.ndarray:
    """Block-permuted copy of ``arr`` for a precomputed block permutation ``perm``.

    ``perm`` MUST come from ``rng.permutation(n_blocks)`` so the null distribution's RNG draw is
    unchanged vs the legacy inline path. Falls back to the numpy broadcast+mask gather when numba
    is unavailable; both produce identical element order, so the block shuffle is bit-identical."""
    if _HAS_NUMBA:
        return np.asarray(_block_gather_kernel(arr, np.ascontiguousarray(perm, dtype=np.int64), int(block_len)))
    m = arr.size
    idx = (perm[:, None] * block_len + np.arange(block_len)[None, :]).ravel()
    idx = idx[idx < m]
    return np.asarray(arr[idx])


def _warm_collinear_kernel() -> None:
    """Compile the kernel at import on a tiny matrix so the first real call is hot."""
    if not _HAS_NUMBA:
        return
    try:
        warm = np.zeros((4, 2), dtype=np.float64)
        warm[:, 0] = np.arange(4.0)
        warm[:, 1] = np.arange(4.0)
        fin = np.isfinite(warm)
        _keep_mask_kernel(warm, fin, 0.99, _BORDERLINE_BAND)
        _mean, _var = _column_stats_allfinite(warm)
        _keep_mask_kernel_allfinite(warm, _mean, _var, 0.99, _BORDERLINE_BAND)
        _block_gather_kernel(np.arange(4, dtype=np.int64), np.arange(2, dtype=np.int64), 2)
        _block_gather_kernel(np.arange(4, dtype=np.float32), np.arange(2, dtype=np.int64), 2)
    except Exception:  # pragma: no cover - warming is best-effort.  # nosec B110 - best-effort/optional path, no module logger
        pass


_warm_collinear_kernel()
