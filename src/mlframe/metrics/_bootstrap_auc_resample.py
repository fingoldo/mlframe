"""Bootstrap AUC resamplers: the fused one-pass kernels, the batched parallel variant and the factory.

Every resampler is bit-identical to ``fast_roc_auc_unstable`` on the same resample. ``_core_auc_brier``
re-exports each name, so callers and the metrics package are unchanged.
"""
from __future__ import annotations

import functools
import threading
from typing import Callable

import numba
import numpy as np
import pandas as pd
import polars as pl

from ._auc_labels import _binary_labels_01, _has_nan_score
from ._numba_params import NUMBA_NJIT_PARAMS, _check_equal_length


@numba.njit(**NUMBA_NJIT_PARAMS)
def _resample_desc_order_counting(resample_rank: np.ndarray, n: int) -> np.ndarray:
    """Descending sort order of a resample, built in O(n) by counting-sort over
    pre-computed base ascending ranks instead of an O(n log n) argsort.

    ``resample_rank[k]`` = ascending rank (0..n-1) of the base element selected
    at resample position ``k``. Bucket counts -> prefix offsets -> scatter gives
    the ascending order; the caller reverses for descending. Bit-identical to
    ``np.argsort(y_score[idx])[::-1]`` when base scores are all-distinct (every
    rank unique, so resample-position tie-break never differs); on tied base
    scores the positional tie-break differs, so the caller GATES this off.

    Retained building block: the resampler now uses the fused single-pass
    ``_fused_resample_auc`` (no separate desc-order array); this counting-sort
    stays available for the desc-order-as-array path and as a fallback primitive."""
    m = resample_rank.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    for k in range(m):
        counts[resample_rank[k]] += 1
    # prefix offsets (exclusive scan)
    offsets = np.empty(n, dtype=np.int64)
    acc = 0
    for r in range(n):
        offsets[r] = acc
        acc += counts[r]
    asc_order = np.empty(m, dtype=np.int64)
    for k in range(m):
        r = resample_rank[k]
        asc_order[offsets[r]] = k
        offsets[r] += 1
    return asc_order[::-1]  # descending


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fused_resample_auc(idx: np.ndarray, base_rank: np.ndarray, y_by_rank: np.ndarray, n: int, counts: np.ndarray, ones: np.ndarray) -> float:
    """Single-pass resample ROC AUC: bin counts+positives per ascending base rank,
    then walk ranks descending accumulating tps/fps. Fuses base_rank gather +
    desc-order build + duplicate y/score gather + the AUC walk into ONE njit pass
    (no desc-index array, no ``[::-1]`` reverse, no second/third gather). Returns
    the same AUC as ``fast_numba_auc_nonw`` fed the counting desc-order; bit-
    identical to ``fast_roc_auc_unstable(y[idx], score[idx])`` on tie-free base
    scores (each rank is unique, so the descending-rank walk matches the desc
    score walk). Caller GATES this on all-distinct base scores.

    ``counts``/``ones`` are CALLER-OWNED ``n``-length int64 scratch buffers, reset
    to 0 here rather than reallocated. The bootstrap resampler calls this ~1000x
    per bootstrap; ``np.zeros(n, ...)`` fresh per call paid a full malloc+zero of
    an ``n``-length int64 array EVERY call (2M-row cProfile: this was the #1
    mlframe hotspot by tottime, 7.884s/1001 calls -- allocate-once + reset-in-place
    turns 1001 mallocs into 1, with the same in-kernel zero-fill cost either way)."""
    counts[:] = 0
    ones[:] = 0
    m = idx.shape[0]
    for k in range(m):
        r = base_rank[idx[k]]
        counts[r] += 1
        ones[r] += y_by_rank[r]
    last_fps = np.int64(0)
    last_tps = np.int64(0)
    tps = np.int64(0)
    fps = np.int64(0)
    auc = np.int64(0)
    for r in range(n - 1, -1, -1):
        c = counts[r]
        if c == 0:
            continue
        pos = ones[r]
        neg = c - pos
        tps += pos
        fps += neg
        auc += (fps - last_fps) * (last_tps + tps)
        last_fps = fps
        last_tps = tps
    tmp = tps * fps * 2
    if tmp > 0:
        return float(auc / tmp)
    return np.nan


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fused_resample_auc_grouped(idx: np.ndarray, group_of_base: np.ndarray, y_base: np.ndarray, ngroups: int, counts: np.ndarray, ones: np.ndarray) -> float:
    """Tie-aware single-pass resample ROC AUC over DISTINCT-score groups.

    Generalises ``_fused_resample_auc`` to tied base scores: instead of binning by
    unique ascending rank (which folds one element per bin and would emit a spurious
    AUC boundary at every tied element), it bins by ``group_of_base[i]`` = the index
    of ``i``'s DISTINCT score value in ascending order (0..ngroups-1). The descending
    walk then advances one boundary per distinct score -- exactly the ``y_score[i+1]
    != y_score[i]`` boundary folding that ``fast_numba_auc_nonw`` performs.

    Bit-identical to ``fast_roc_auc_unstable(y_true[idx], y_score[idx])`` on ANY base
    scores (tied or tie-free): the AUC value is invariant to the within-tie argsort
    order (cumulative tps/fps are order-independent inside a tied run, and the run
    contributes a single trapezoid at its last element), so we need only the per-group
    resampled positive/negative counts, not the positional tie-break. Integer
    accumulation matches the reference exactly for labels in {0,1} and n < 2**53.

    ``y_base[i]`` is the {0,1} label of base index ``i`` (gathered once by the caller).
    O(m + ngroups) per resample -- no per-resample argsort. ``counts``/``ones`` are
    CALLER-OWNED ``ngroups``-length int64 scratch buffers (reset to 0 here, not
    reallocated) -- see :func:`_fused_resample_auc`'s docstring for the rationale."""
    counts[:] = 0
    ones[:] = 0
    m = idx.shape[0]
    for k in range(m):
        bi = idx[k]
        g = group_of_base[bi]
        counts[g] += 1
        ones[g] += y_base[bi]
    last_fps = np.int64(0)
    last_tps = np.int64(0)
    tps = np.int64(0)
    fps = np.int64(0)
    auc = np.int64(0)
    for g in range(ngroups - 1, -1, -1):
        c = counts[g]
        if c == 0:
            continue
        pos = ones[g]
        neg = c - pos
        tps += pos
        fps += neg
        auc += (fps - last_fps) * (last_tps + tps)
        last_fps = fps
        last_tps = tps
    tmp = tps * fps * 2
    if tmp > 0:
        return float(auc / tmp)
    return np.nan


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fused_resample_auc_batch_parallel(idxs: np.ndarray, base_rank: np.ndarray, y_by_rank: np.ndarray, n: int) -> np.ndarray:
    """``prange``-parallel twin of :func:`_fused_resample_auc` for a WHOLE bootstrap distribution at once.

    ``idxs`` is ``(R, m)`` int64 -- ``R`` independent resamples, each of size ``m`` (typically ``m == n``),
    pre-generated by the caller (bit-identical index generation to the serial single-resample path). Each
    ``numba.prange`` iteration is a fully independent resample: it allocates its OWN private ``counts``/
    ``ones`` int64 buffers (numba privatises locals declared inside the parallel loop body) and runs the
    exact same counting + descending-walk as :func:`_fused_resample_auc`, so ``out[r]`` is bit-identical to
    ``_fused_resample_auc(idxs[r], base_rank, y_by_rank, n, ...)``.

    This is the "fully-njit resample loop (rng + gather + kernel in one nogil pass)" the module's own prior
    audit flagged as the real lever over ``bootstrap_metrics``'s joblib-threading path (docstring there:
    "MEASURED ... threading ~1.0x@4 / 1.3x@8 / 1.5x@16 ... stays OPT-IN ... the per-resample cost is
    dominated by the GIL-held index generation + fancy-index gather, not the nogil kernels"). Threading
    across SEPARATE Python-level njit calls never releases the GIL for the RNG draw / dispatch overhead
    between calls; a single ``parallel=True`` njit call does the RNG draw ONCE (vectorised, outside the
    parallel region) and runs every resample's kernel body truly concurrently across OS threads with no
    per-resample Python round-trip. Memory: ``R`` private ``n``-length int64 pairs are transient (freed
    per-iteration, not all R held at once) -- peak scratch is ``O(threads * n)``, not ``O(R * n)``.

    Caller owns ``idxs`` materialisation (memory-bound at large ``R * n``; chunk externally if needed)."""
    r_count = idxs.shape[0]
    out = np.empty(r_count, dtype=np.float64)
    for r in numba.prange(r_count):
        counts = np.zeros(n, dtype=np.int64)
        ones = np.zeros(n, dtype=np.int64)
        idx = idxs[r]
        m = idx.shape[0]
        for k in range(m):
            rk = base_rank[idx[k]]
            counts[rk] += 1
            ones[rk] += y_by_rank[rk]
        last_fps = np.int64(0)
        last_tps = np.int64(0)
        tps = np.int64(0)
        fps = np.int64(0)
        auc = np.int64(0)
        for rk in range(n - 1, -1, -1):
            c = counts[rk]
            if c == 0:
                continue
            pos = ones[rk]
            neg = c - pos
            tps += pos
            fps += neg
            auc += (fps - last_fps) * (last_tps + tps)
            last_fps = fps
            last_tps = tps
        tmp = tps * fps * 2
        out[r] = auc / tmp if tmp > 0 else np.nan
    return out


def bootstrap_auc_distribution_parallel(y_true: np.ndarray, y_score: np.ndarray, n_bootstrap: int, random_state: "int | None" = None, chunk_size: int = 200) -> "np.ndarray | None":
    """Compute a FULL bootstrap AUC distribution (``n_bootstrap`` resamples) via the batched
    ``prange``-parallel kernel, gated on the same all-distinct-base-scores condition as
    :func:`make_bootstrap_auc_resampler`'s fast path.

    Chunks the resample dimension at ``chunk_size`` to bound peak memory to
    ``O(chunk_size * n)`` for the index matrix (the dominant allocation at large ``n`` -- e.g.
    ``chunk_size=200`` at ``n=2e6`` caps the index chunk at ~320MB instead of materialising all
    ``n_bootstrap`` resamples' indices at once). Returns ``None`` when the tie-free gate fails (ties
    present); the caller should fall back to :func:`make_bootstrap_auc_resampler`'s grouped/exact path.

    Bit-identical to calling ``make_bootstrap_auc_resampler(y_true, y_score)`` once per resample IF fed
    the same per-resample index arrays -- see ``bench_fused_resample_auc_batch_parallel.py`` for the A/B
    against the serial per-call loop (same RNG draw scheme: ``np.random.default_rng(random_state).integers``).
    """
    if isinstance(y_true, (pd.Series, pl.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_score, (pd.Series, pl.Series)):
        y_score = y_score.to_numpy()
    if y_score.ndim == 2:
        y_score = y_score[:, -1]
    _check_equal_length(y_true, y_score)
    if _has_nan_score(y_score):
        return np.full(n_bootstrap, np.nan)  # same NaN convention as fast_roc_auc (audit 25-F8)
    y_true = np.ascontiguousarray(_binary_labels_01(y_true))
    y_score = np.ascontiguousarray(y_score)
    n = y_score.shape[0]

    asc_order = np.argsort(y_score)
    sorted_score = y_score[asc_order]
    tie_free = True if n < 2 else bool(np.all(sorted_score[1:] != sorted_score[:-1]))
    if not tie_free:
        return None

    base_rank = np.empty(n, dtype=np.int64)
    base_rank[asc_order] = np.arange(n, dtype=np.int64)
    y_by_rank = np.ascontiguousarray(y_true[asc_order].astype(np.int64))

    rng = np.random.default_rng(random_state)
    out = np.empty(n_bootstrap, dtype=np.float64)
    for lo in range(0, n_bootstrap, chunk_size):
        hi = min(lo + chunk_size, n_bootstrap)
        idxs = rng.integers(0, n, size=(hi - lo, n), dtype=np.int64)
        out[lo:hi] = _fused_resample_auc_batch_parallel(idxs, base_rank, y_by_rank, n)
    return out


def exact_bootstrap_auc_resample(y_true: np.ndarray, y_score: np.ndarray, idx: np.ndarray) -> float:
    """Recompute AUC exactly on the resampled (index-gathered) ``y_true``/``y_score`` -- the fallback
    bootstrap AUC resampler used when the fast fused path isn't applicable."""
    from mlframe.metrics._core_auc_brier import fast_roc_auc_unstable  # parent is fully loaded by the time a resample runs
    return fast_roc_auc_unstable(y_true[idx], y_score[idx])


def make_bootstrap_auc_resampler(y_true: np.ndarray, y_score: np.ndarray) -> Callable[[np.ndarray], float]:
    """Factory: pre-argsort the BASE score vector ONCE, return a callable
    ``resampler(idx) -> float`` that scores each bootstrap resample without
    re-argsorting the n-length resampled vector (1000x O(n log n) -> O(n)).

    The returned closure scores each resample in ONE fused njit pass
    (``_fused_resample_auc``): it bins counts + positives per ascending base rank
    and walks ranks descending accumulating tps/fps -- no per-resample argsort, no
    desc-index array, no ``[::-1]``, and no duplicate y/score gather. The AUC is
    bit-identical to ``fast_roc_auc_unstable(y_true[idx], y_score[idx])`` on tie-
    free float64 scores. Measured on the 1000-bootstrap loop: 1.72x@50k /
    2.16x@200k over the prior 4-pass resampler, maxdiff 0.0 (see
    ``training/_benchmarks/bench_fused_bootstrap_auc_resampler.py``).

    GATE: the fast path requires all-distinct base scores (tie-free). On tied /
    discrete / low-cardinality base scores ``np.argsort`` breaks ties by
    resample position, which the counting path cannot reproduce, so the
    resampler falls back to the exact per-resample argsort path (still the
    fastest correct path on tied data). The fast path is DEFAULT-ON whenever
    the gate condition holds -- no opt-in flag."""
    if isinstance(y_true, (pd.Series, pl.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_score, (pd.Series, pl.Series)):
        y_score = y_score.to_numpy()
    if y_score.ndim == 2:
        y_score = y_score[:, -1]
    _check_equal_length(y_true, y_score)
    if _has_nan_score(y_score):
        return lambda idx: float("nan")  # same NaN convention as fast_roc_auc (audit 25-F8)
    y_true = np.ascontiguousarray(_binary_labels_01(y_true))
    y_score = np.ascontiguousarray(y_score)
    n = y_score.shape[0]

    asc_order = np.argsort(y_score)  # ascending; once
    sorted_score = y_score[asc_order]
    # all-distinct gate: no two adjacent sorted scores equal -> tie-free
    tie_free = True if n < 2 else bool(np.all(sorted_score[1:] != sorted_score[:-1]))

    if not tie_free:
        # Tied base scores. The exact per-resample argsort fallback re-sorts the whole
        # n-length resampled vector every call (O(n log n) x ~3000 resamples). Because the
        # AUC value is invariant to the within-tie argsort order, we can score each resample
        # in O(n + K) over the K distinct-score GROUPS instead (``_fused_resample_auc_grouped``),
        # matching numpy's positional tie-break WITHOUT reproducing it -- the folded trapezoid
        # only needs per-group resampled +/- counts. Bit-identical to the exact path (verified
        # in tests/metrics/test_bootstrap_auc_resampler_tied.py). Requires binary {0,1} labels;
        # anything else keeps the exact fallback (below).
        y_int = np.ascontiguousarray(y_true).ravel()
        uniq_labels = np.unique(y_int)
        binary_ok = uniq_labels.size <= 2 and bool(np.all((uniq_labels == 0) | (uniq_labels == 1)))
        if binary_ok:
            # group_of_base[i] = index of base i's distinct score (ascending order), 0..K-1
            boundaries = sorted_score[1:] != sorted_score[:-1]
            group_by_rank = np.empty(n, dtype=np.int64)
            group_by_rank[0] = 0
            if n > 1:
                group_by_rank[1:] = np.cumsum(boundaries)
            ngroups = int(group_by_rank[-1]) + 1 if n > 0 else 0
            group_of_base = np.empty(n, dtype=np.int64)
            group_of_base[asc_order] = group_by_rank
            y_base = np.ascontiguousarray(y_int.astype(np.int64))
            _grouped_tls = threading.local()

            def _resampler_grouped(idx: np.ndarray) -> float:
                """Fast bootstrap AUC resampler for tied scores: groups by distinct score value so ties are counted correctly without re-sorting the resampled indices."""
                bufs = getattr(_grouped_tls, "bufs", None)
                if bufs is None:
                    bufs = (np.zeros(ngroups, dtype=np.int64), np.zeros(ngroups, dtype=np.int64))
                    _grouped_tls.bufs = bufs  # per-thread scratch, see 25-F9
                return float(_fused_resample_auc_grouped(idx, group_of_base, y_base, ngroups, bufs[0], bufs[1]))
            return _resampler_grouped

        return functools.partial(exact_bootstrap_auc_resample, y_true, y_score)

    # base_rank[i] = ascending rank of base index i (inverse permutation)
    base_rank = np.empty(n, dtype=np.int64)
    base_rank[asc_order] = np.arange(n, dtype=np.int64)
    # y_by_rank[r] = label of the base element at ascending rank r (gather once)
    y_by_rank = np.ascontiguousarray(y_true[asc_order].astype(np.int64))
    # Scratch buffers are reset in-kernel per call (see _fused_resample_auc's docstring). They are per THREAD: the
    # resampler is called from joblib's thread backend, and one shared buffer would be written by two resamples at once (audit 25-F9).
    _tls = threading.local()

    def _resampler_fast(idx: np.ndarray) -> float:
        """Fast bootstrap AUC resampler for the no-ties case: reuses the pre-computed base ranking so each resample avoids a fresh full sort."""
        bufs = getattr(_tls, "fast", None)
        if bufs is None:
            bufs = (np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64))
            _tls.fast = bufs
        return float(_fused_resample_auc(idx, base_rank, y_by_rank, n, bufs[0], bufs[1]))

    return _resampler_fast
