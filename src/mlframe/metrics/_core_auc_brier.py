"""AUC + Brier score kernels for ``mlframe.metrics.core``.

Carved from ``core.py``. Public symbols are re-exported from the parent.
"""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd
import polars as pl

from ._numba_params import NUMBA_NJIT_PARAMS, _PARALLEL_REDUCTION_THRESHOLD, _check_equal_length


import os as _os


# iter338 (2026-05-27): central dispatcher for ``argsort(y_score)[::-1]``
# in metric kernels. Default UNSTABLE (numpy quicksort) -- 2-3x faster than
# stable sort and numerically identical on continuous-valued ML predictions
# (the dominant case; exact ties are essentially impossible on float64
# probabilities from MLP / boosters / linear models). Pathological tie-
# heavy inputs (binned predictions, dummy classifiers, calibrators that
# bucket-snap probabilities) opt back via env var
# ``MLFRAME_METRICS_STABLE_SORT=1``. Bench c0083 / c0091 honest_diagnostics
# bootstrap loop: 2.25x (n=200k) to 2.75x (n=20k); deviations from the
# stable variant under 1e-12 on Gaussian / uniform predictions.
# GPU argsort gate -- the size threshold is everything. ISOLATED micro-bench (cupy vs numpy argsort+transfer) on this
# host (RTX 500 Ada laptop): 0.42x@10k, 1.95x@50k, 3.92x@200k, 4.94x@1M. A metric run issues MANY argsorts of MIXED
# size (full-array per-class AUC at N, plus many small per-bin/bootstrap-resample sorts), so the gate MUST send only
# the big ones to the GPU: routing ALL of them (gate=1) regressed the c0023 200k suite ~8% (the small sorts pay
# H2D/D2H/sync overhead the tight micro-bench hides), but size-gating at 50k -- GPU for argsorts >= 50k, CPU below --
# nets a CONSISTENT ~10% END-TO-END win at 200k (A/B: CPU 9.37/8.29s vs GPU 7.98/7.89s, both GPU runs beat both CPU).
# Default = 50k (the measured isolated crossover). Tune per host via MLFRAME_METRICS_ARGSORT_GPU_MIN_N (huge = force
# CPU). Stable-sort opt-in always stays on CPU. (Lesson: never reject on the extremes -- sweep the gated middle.)
_GPU_ARGSORT_MIN_N = int(_os.environ.get("MLFRAME_METRICS_ARGSORT_GPU_MIN_N", "50000"))
_GPU_ARGSORT_AVAILABLE: "bool | None" = None

# iter97 (2026-06-14): parallel bucket-split argsort for the large-N CPU path. The metric kernels' descending argsort is
# tie-order-INVARIANT (AUC uses fractional ranks; KS folds tied scores into a single CDF jump), so we may pick any sort
# whose output orders y_score identically -- the within-bucket tie-break order is immaterial. A linear-range bucketise
# (parallel per-thread histogram + serial scatter) groups indices into B value-ordered buckets, each bucket is argsorted
# in parallel (numpy on cache-resident slices), then the concatenation is reversed for descending. Measured crossover
# (8-thread, this host): 1.46x@100k / 1.62x@500k / 2.33x@1M / 4.01x@5M, y_score-order identical to np.argsort. Gated to
# the unstable CPU default at N >= _PAR_BUCKET_ARGSORT_MIN_N; the stable-sort opt-in and the GPU path are untouched.
# Tune the gate per host via MLFRAME_METRICS_ARGSORT_PAR_MIN_N (huge value = force scalar numpy).
_PAR_BUCKET_ARGSORT_MIN_N = int(_os.environ.get("MLFRAME_METRICS_ARGSORT_PAR_MIN_N", "200000"))


@numba.njit(cache=True, nogil=True, parallel=True)
def _bucket_hist(y_score: np.ndarray, lo: float, inv_w: float, nbuckets: int, nthr: int) -> np.ndarray:
    """Per-thread linear-range bucket histogram, reduced to a single (nbuckets,) count vector."""
    n = y_score.shape[0]
    local = np.zeros((nthr, nbuckets), dtype=np.int64)
    for t in numba.prange(nthr):
        s = t * n // nthr
        e = (t + 1) * n // nthr
        for i in range(s, e):
            b = int((y_score[i] - lo) * inv_w)
            if b < 0:
                b = 0
            elif b >= nbuckets:
                b = nbuckets - 1
            local[t, b] += 1
    return local.sum(axis=0)


@numba.njit(cache=True, nogil=True)
def _bucket_scatter(y_score: np.ndarray, lo: float, inv_w: float, nbuckets: int, offsets: np.ndarray, n: int) -> np.ndarray:
    """Scatter sample indices into their value-ordered bucket (serial: each bucket cursor is independent)."""
    out = np.empty(n, dtype=np.int64)
    pos = offsets.copy()
    for i in range(n):
        b = int((y_score[i] - lo) * inv_w)
        if b < 0:
            b = 0
        elif b >= nbuckets:
            b = nbuckets - 1
        out[pos[b]] = i
        pos[b] += 1
    return out


@numba.njit(cache=True, nogil=True, parallel=True)
def _bucket_sort_within(idx: np.ndarray, y_score: np.ndarray, bnd: np.ndarray, nbuckets: int, res: np.ndarray) -> None:
    """Argsort each bucket's index slice by its y_score (ascending) in parallel; write into res in-place."""
    for b in numba.prange(nbuckets):
        s = bnd[b]
        e = bnd[b + 1]
        if e > s:
            sub = idx[s:e]
            order = np.argsort(y_score[sub])
            for k in range(e - s):
                res[s + k] = sub[order[k]]


def _argsort_desc_par_bucket(y_score: np.ndarray) -> np.ndarray:
    """Descending argsort via parallel bucket-split. Orders y_score identically to ``np.argsort(y_score)[::-1]``;
    within-bucket tie-break order may differ (immaterial to the tie-invariant AUC / KS consumers)."""
    n = y_score.shape[0]
    y64 = np.ascontiguousarray(y_score, dtype=np.float64)
    lo = float(y64.min())
    hi = float(y64.max())
    if not (hi > lo):  # constant column (or non-finite collapse) -> nothing to order; mirror numpy's index sequence
        return np.argsort(y64)[::-1].copy()
    nbuckets = max(64, min(8192, n // 256))
    inv_w = nbuckets / (hi - lo)
    counts = _bucket_hist(y64, lo, inv_w, nbuckets, numba.get_num_threads())
    bnd = np.empty(nbuckets + 1, dtype=np.int64)
    bnd[0] = 0
    acc = 0
    for b in range(nbuckets):
        acc += int(counts[b])
        bnd[b + 1] = acc
    idx = _bucket_scatter(y64, lo, inv_w, nbuckets, bnd[:nbuckets].copy(), n)
    res = np.empty(n, dtype=np.int64)
    _bucket_sort_within(idx, y64, bnd, nbuckets, res)
    return res[::-1].copy()


def _gpu_argsort_available() -> bool:
    """cupy + a visible CUDA device (cached once). Probes cupy directly (this path IS cupy, not numba) so it does not
    depend on CUDA_HOME / numba's cached CUDA detection. The metrics argsort is the unstable default, so a GPU radix
    sort is a valid backend -- fast_aucs uses tie-order-invariant fractional ranks (verified byte-identical AUC)."""
    global _GPU_ARGSORT_AVAILABLE
    if _GPU_ARGSORT_AVAILABLE is None:
        try:
            import cupy as cp
            _GPU_ARGSORT_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            _GPU_ARGSORT_AVAILABLE = False
    return _GPU_ARGSORT_AVAILABLE


def _argsort_desc_for_metrics(y_score: np.ndarray) -> np.ndarray:
    """Descending argsort used by every metric kernel; stable-opt-in via env, GPU-dispatched for very large N."""
    if _os.environ.get("MLFRAME_METRICS_STABLE_SORT") == "1":
        return np.argsort(y_score, kind="stable")[::-1]
    n = y_score.shape[0] if hasattr(y_score, "shape") else len(y_score)
    if n >= _GPU_ARGSORT_MIN_N and _gpu_argsort_available():
        try:
            import cupy as cp
            return cp.asnumpy(cp.argsort(cp.asarray(y_score))[::-1])
        except Exception:
            pass  # GPU OOM / transient device error -> exact CPU fallback
    if n >= _PAR_BUCKET_ARGSORT_MIN_N:
        return _argsort_desc_par_bucket(y_score)
    return np.argsort(y_score)[::-1]


def fast_roc_auc_unstable(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC variant using unstable (quicksort) argsort -- 2-3x faster.

    Use ONLY for callers where tie-breaking determinism on tied scores is
    immaterial: bootstrap resampling (the resample randomness already
    dominates any tie-order effect), ad-hoc per-fold metric reports
    inside CV searches, and any monte-carlo loop where the consumer
    cares about the distribution of AUC, not the byte-identical scalar.

    Stable sort is needed when two runs must produce the same AUC byte-
    identically on data with tied scores. For float64 predictions from
    real models, exact ties are rare (~0% on continuous probabilities)
    and the metric difference vs the stable variant is <1e-12 in
    practice. Where ties are common (binned / dummy classifier output),
    use ``fast_roc_auc`` instead.

    bench-validated 2026-05-27 iter336 (c0083 honest_diagnostics
    bootstrap path)::

        n=20k    stable=1.85 ms  unstable=0.67 ms   2.75x
        n=200k   stable=25.7 ms  unstable=11.4 ms   2.25x

    On c0091 / c0083 binary classification combos the bootstrap block
    runs ~6000 _auc calls per process; the swap saves ~3-4 s per
    process on n=20k val splits, ~50 s on n=200k test splits.
    """
    if isinstance(y_true, (pd.Series, pl.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_score, (pd.Series, pl.Series)):
        y_score = y_score.to_numpy()
    if y_score.ndim == 2:
        y_score = y_score[:, -1]
    # No ``kind=stable``: numpy default quicksort is 2-3x faster and
    # numerically identical when scores have no exact ties (the dominant
    # case for float64 model outputs).
    desc_score_indices = np.argsort(y_score)[::-1]
    return fast_numba_auc_nonw(y_true=y_true, y_score=y_score, desc_score_indices=desc_score_indices)


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
def _fused_resample_auc(idx: np.ndarray, base_rank: np.ndarray, y_by_rank: np.ndarray, n: int) -> float:
    """Single-pass resample ROC AUC: bin counts+positives per ascending base rank,
    then walk ranks descending accumulating tps/fps. Fuses base_rank gather +
    desc-order build + duplicate y/score gather + the AUC walk into ONE njit pass
    (no desc-index array, no ``[::-1]`` reverse, no second/third gather). Returns
    the same AUC as ``fast_numba_auc_nonw`` fed the counting desc-order; bit-
    identical to ``fast_roc_auc_unstable(y[idx], score[idx])`` on tie-free base
    scores (each rank is unique, so the descending-rank walk matches the desc
    score walk). Caller GATES this on all-distinct base scores."""
    counts = np.zeros(n, dtype=np.int64)
    ones = np.zeros(n, dtype=np.int64)
    m = idx.shape[0]
    for k in range(m):
        r = base_rank[idx[k]]
        counts[r] += 1
        ones[r] += y_by_rank[r]
    last_fps = 0
    last_tps = 0
    tps = 0
    fps = 0
    auc = 0
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
        return auc / tmp
    return np.nan


def make_bootstrap_auc_resampler(y_true: np.ndarray, y_score: np.ndarray):
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
    y_true = np.ascontiguousarray(y_true)
    y_score = np.ascontiguousarray(y_score)
    n = y_score.shape[0]

    asc_order = np.argsort(y_score)  # ascending; once
    sorted_score = y_score[asc_order]
    # all-distinct gate: no two adjacent sorted scores equal -> tie-free
    tie_free = n < 2 or bool(np.all(sorted_score[1:] != sorted_score[:-1]))

    if not tie_free:
        def _resampler_exact(idx: np.ndarray) -> float:
            return fast_roc_auc_unstable(y_true[idx], y_score[idx])
        return _resampler_exact

    # base_rank[i] = ascending rank of base index i (inverse permutation)
    base_rank = np.empty(n, dtype=np.int64)
    base_rank[asc_order] = np.arange(n, dtype=np.int64)
    # y_by_rank[r] = label of the base element at ascending rank r (gather once)
    y_by_rank = np.ascontiguousarray(y_true[asc_order].astype(np.int64))

    def _resampler_fast(idx: np.ndarray) -> float:
        return float(_fused_resample_auc(idx, base_rank, y_by_rank, n))

    return _resampler_fast


def fast_roc_auc(y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:
    """Compute ROC AUC efficiently using numba.

    Note: np.argsort needs to stay out of njitted func.

    bench-attempt-rejected (2026-05-26, c0091 iter316): tried folding the
    ``np.argsort(kind="stable")`` into the numba kernel via
    ``np.argsort(kind="mergesort")`` (numba's only stable sort). Bench
    ``profiling/bench_fast_roc_auc_argsort_inside.py``::

        n=2000   current=0.13 ms  proposed=0.12 ms  speedup=1.05x
        n=20000  current=1.79 ms  proposed=1.80 ms  speedup=1.00x
        n=200000 current=26.3 ms  proposed=29.5 ms  speedup=0.89x
        n=1M     current=156 ms   proposed=190 ms   speedup=0.82x

    Numpy's stable sort C implementation is 11-22pct faster than numba's
    mergesort on n>=200k, where the bootstrap loop spends most of its
    time. Per-call Python ``_wrapfunc`` overhead exists but is dwarfed
    by the sort itself, so removing it does not move the needle. Numpy
    argsort stays outside.

    bench-attempt-rejected (2026-05-28, c0027_90010291 iter552): tried
    folding numba's default (quicksort) argsort into the AUC kernel with
    a reverse-iteration walk. Numba's quicksort is markedly slower than
    numpy's C-optimised quicksort even with the Python<->numba dispatch
    saved::

        n=5k     numpy+kern=0.13 ms  numba fused=0.39 ms  0.34x
        n=20k    numpy+kern=0.58 ms  numba fused=1.84 ms  0.32x
        n=50k    numpy+kern=1.66 ms  numba fused=7.60 ms  0.22x
        n=200k   numpy+kern=20.5 ms  numba fused=44.3 ms  0.46x

    Conclusion: numpy argsort is at the algorithmic floor for this kernel.
    The c0027 honest_diagnostics bootstrap argsort (5.1s tottime / 6419
    calls / ~800us per call at the resampled ~50k bootstrap size) is the
    floor cost on this code path. Documented so the next agent does not
    re-attempt a third sort-fusion variant.

    See ``fast_roc_auc_unstable`` for the 2-3x faster variant that drops
    the stable-sort guarantee -- safe for bootstrap / monte-carlo
    callers where tie-breaking determinism is immaterial.
    """
    # **kwargs absorbs sklearn's extra params (sklearn's scorer forwards sample_weight when the caller fits with weights).
    sample_weight = kwargs.get("sample_weight")

    if isinstance(y_true, (pd.Series, pl.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_score, (pd.Series, pl.Series)):
        y_score = y_score.to_numpy()
    if y_score.ndim == 2:
        y_score = y_score[:, -1]
    _check_equal_length(y_true, y_score)
    desc_score_indices = _argsort_desc_for_metrics(y_score)  # iter338: dispatcher (unstable default, MLFRAME_METRICS_STABLE_SORT=1 to opt back)
    if sample_weight is not None:
        if isinstance(sample_weight, (pd.Series, pl.Series)):
            sample_weight = sample_weight.to_numpy()
        sample_weight = np.ascontiguousarray(np.asarray(sample_weight), dtype=np.float64)
        return fast_numba_auc_weighted(
            y_true=np.asarray(y_true, dtype=np.float64), y_score=y_score, sample_weight=sample_weight, desc_score_indices=desc_score_indices
        )
    return fast_numba_auc_nonw(y_true=y_true, y_score=y_score, desc_score_indices=desc_score_indices)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _roc_curve_kernel(y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray,
                      desc_score_indices: np.ndarray, weighted: bool):
    """Cumulative TP/FP sweep over descending-sorted scores -> (fps, tps, thr) at DISTINCT-score boundaries.

    Mirrors sklearn's ``_binary_clf_curve``: walk samples in descending score order accumulating the true-positive
    and false-positive mass; a threshold row is emitted only at the LAST index of each tied-score run (sklearn keeps
    ``np.where(np.diff(y_score))`` plus the final point), so intermediate thresholds within a tie run are dropped.
    Returns the raw ``fps`` (cumulative negatives), ``tps`` (cumulative positives), and the corresponding thresholds
    at those boundaries. The caller prepends the (0,0) origin and normalises to FPR/TPR.
    """
    n = desc_score_indices.shape[0]
    # First pass: count distinct-score boundaries to size the output exactly (no dynamic list in njit).
    n_thresh = 0
    for i in range(n):
        idx = desc_score_indices[i]
        if i == n - 1 or y_score[desc_score_indices[i + 1]] != y_score[idx]:
            n_thresh += 1

    fps = np.empty(n_thresh, dtype=np.float64)
    tps = np.empty(n_thresh, dtype=np.float64)
    thr = np.empty(n_thresh, dtype=np.float64)

    tp = 0.0
    fp = 0.0
    j = 0
    for i in range(n):
        idx = desc_score_indices[i]
        yt = y_true[idx]
        if weighted:
            w = sample_weight[idx]
        else:
            w = 1.0
        if yt != 0:
            tp += w
        else:
            fp += w
        if i == n - 1 or y_score[desc_score_indices[i + 1]] != y_score[idx]:
            fps[j] = fp
            tps[j] = tp
            thr[j] = y_score[idx]
            j += 1
    return fps, tps, thr


def fast_roc_curve(y_true, y_score, *, sample_weight=None):
    """ROC curve matching ``sklearn.metrics.roc_curve`` semantics: ``(fpr, tpr, thresholds)``.

    Own-implementation replacement for sklearn's ``roc_curve``. A single descending-score sweep (numba njit)
    accumulates cumulative true-/false-positive mass and emits one point per DISTINCT score (ties collapse to a
    single threshold, exactly as sklearn does via ``np.diff``). The returned arrays follow sklearn's conventions:

    - A leading ``(fpr=0, tpr=0)`` point is prepended (the "classify nothing positive" operating point).
    - ``thresholds[0]`` is ``+inf`` (sklearn >=1.3 uses ``np.inf`` rather than ``max(score)+1`` for this anchor),
      so ``thresholds`` is one longer than the swept points and strictly decreasing thereafter.
    - ``fpr``/``tpr`` are normalised by the total negative / positive weight.

    Single-class inputs (all positives or all negatives) yield ``nan`` in the undefined-rate axis, mirroring
    sklearn (which warns and produces ``nan`` FPR or TPR). Empty input returns three empty-ish arrays with just
    the prepended origin / inf anchor, again matching sklearn's degenerate output shape.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary ground-truth labels (0/1, or {-1,1}; any non-zero is treated as the positive class after the
        ``== 1`` normalisation sklearn applies -- pass 0/1 for exact parity).
    y_score : array-like of shape (n,)
        Target scores (probabilities or decision-function values). A 2-D ``(n, 2)`` proba array uses column -1.
    sample_weight : array-like of shape (n,), optional
        Per-sample weights; when given, TP/FP mass accumulates weighted (sklearn-equivalent).

    Returns
    -------
    fpr, tpr, thresholds : np.ndarray
        ``fpr`` and ``tpr`` have identical length; ``thresholds`` matches that length (its first entry is +inf).
    """
    if isinstance(y_true, (pd.Series, pl.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_score, (pd.Series, pl.Series)):
        y_score = y_score.to_numpy()
    y_score = np.asarray(y_score)
    if y_score.ndim == 2:
        y_score = y_score[:, -1]
    y_score = np.ascontiguousarray(y_score, dtype=np.float64)
    # sklearn treats the positive class as ``y_true == 1``; normalise to {0.0, 1.0} so {-1,1} / bool inputs match.
    y_true = np.ascontiguousarray(np.asarray(y_true) == 1, dtype=np.float64)
    _check_equal_length(y_true, y_score)

    n = y_score.shape[0]
    if n == 0:
        # sklearn returns arrays with a single nan/inf-anchored point on empty input.
        return (np.array([np.nan]), np.array([np.nan]), np.array([np.inf]))

    if sample_weight is not None:
        if isinstance(sample_weight, (pd.Series, pl.Series)):
            sample_weight = sample_weight.to_numpy()
        sample_weight = np.ascontiguousarray(np.asarray(sample_weight), dtype=np.float64)
        weighted = True
    else:
        sample_weight = np.empty(0, dtype=np.float64)
        weighted = False

    desc_score_indices = _argsort_desc_for_metrics(y_score)
    desc_score_indices = np.ascontiguousarray(desc_score_indices, dtype=np.int64)
    fps, tps, thr = _roc_curve_kernel(y_true, y_score, sample_weight, desc_score_indices, weighted)

    # sklearn drops "collinear" intermediate points that lie on a straight segment (optimal thresholds only).
    # We keep every distinct-score point (a strict superset of sklearn's kept points and identical FPR/TPR vertices);
    # collinear-drop only removes redundant points on a line, so the curve, its AUC and np.trapz are unchanged. To
    # match sklearn's EXACT array we replicate its collinear filter below.
    total_pos = tps[-1]
    total_neg = fps[-1]

    # Collinear-point drop (sklearn >=0.24) is applied to the RAW swept points, BEFORE prepending the origin -- the
    # leading (0,0) shifts the second-difference window otherwise and off-by-ones the kept set.
    if fps.shape[0] > 2:
        optimal = _roc_optimal_idxs(fps, tps)
        fps = fps[optimal]
        tps = tps[optimal]
        thr = thr[optimal]

    # Prepend the (0, 0) origin. sklearn: tps = r_[0, tps], fps = r_[0, fps], thresholds = r_[inf, thresholds].
    fps = np.concatenate((np.array([0.0]), fps))
    tps = np.concatenate((np.array([0.0]), tps))
    thr = np.concatenate((np.array([np.inf]), thr))

    if total_pos <= 0:
        tpr = np.full(tps.shape[0], np.nan)
    else:
        tpr = tps / total_pos
    if total_neg <= 0:
        fpr = np.full(fps.shape[0], np.nan)
    else:
        fpr = fps / total_neg
    return fpr, tpr, thr


def _roc_optimal_idxs(fps: np.ndarray, tps: np.ndarray) -> np.ndarray:
    """Indices of the non-collinear ROC vertices (sklearn's ``drop_intermediate`` optimum filter).

    sklearn keeps a point when the second difference of ``(fps, tps)`` is non-zero -- i.e. the slope changes --
    plus both endpoints. Points strictly interior to a straight segment are redundant (the curve, AUC and every
    interpolation are identical without them), so dropping them yields sklearn's exact array without altering shape.
    """
    d1 = np.diff(fps, prepend=fps[0])
    d2 = np.diff(tps, prepend=tps[0])
    # A vertex is kept if the incremental step differs from the next step in either axis (slope change), or it is
    # an endpoint. Replicates ``np.r_[True, np.logical_or(np.diff(fps,2), np.diff(tps,2)), True]``.
    keep = np.ones(fps.shape[0], dtype=np.bool_)
    if fps.shape[0] > 2:
        second = np.logical_or(np.diff(fps, 2) != 0, np.diff(tps, 2) != 0)
        keep[1:-1] = second
    return np.flatnonzero(keep)


@numba.njit(**NUMBA_NJIT_PARAMS)
def fast_numba_auc_nonw(y_true: np.ndarray, y_score: np.ndarray, desc_score_indices: np.ndarray) -> float:
    """code taken from fastauc lib."""
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    last_counted_fps = 0
    last_counted_tps = 0
    tps, fps = 0, 0
    auc = 0

    l = len(y_true) - 1
    for i in range(l + 1):
        tps += y_true[i]
        fps += 1 - y_true[i]
        if i == l or y_score[i + 1] != y_score[i]:
            auc += (fps - last_counted_fps) * (last_counted_tps + tps)
            last_counted_fps = fps
            last_counted_tps = tps
    tmp = tps * fps * 2
    if tmp > 0:
        return auc / tmp
    else:
        # Single-class data: ROC AUC is undefined
        return np.nan


@numba.njit(**NUMBA_NJIT_PARAMS)
def fast_numba_auc_weighted(
    y_true: np.ndarray, y_score: np.ndarray, sample_weight: np.ndarray, desc_score_indices: np.ndarray
) -> float:
    """Weighted ROC AUC via the same tie-aware trapezoidal scan as ``fast_numba_auc_nonw``.

    Equivalent to ``sklearn.metrics.roc_auc_score(..., sample_weight=w)``: each sample contributes its weight to the
    cumulative true/false-positive mass rather than a unit count. Reduces to the unweighted kernel when all weights are 1.
    """
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    w = sample_weight[desc_score_indices]

    last_counted_fps = 0.0
    last_counted_tps = 0.0
    tps = 0.0
    fps = 0.0
    auc = 0.0

    l = len(y_true) - 1
    for i in range(l + 1):
        wi = w[i]
        tps += y_true[i] * wi
        fps += (1.0 - y_true[i]) * wi
        if i == l or y_score[i + 1] != y_score[i]:
            auc += (fps - last_counted_fps) * (last_counted_tps + tps)
            last_counted_fps = fps
            last_counted_tps = tps
    tmp = tps * fps * 2.0
    if tmp > 0:
        return auc / tmp
    else:
        return np.nan


def fast_aucs(y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> tuple[float, float]:
    """Compute both ROC AUC and PR AUC efficiently."""
    # Unlike ``fast_roc_auc``, the fused ROC+PR kernel has no weighted variant.
    # Raise rather than silently ignoring sample_weight (sklearn's scorer forwards
    # it when fit with weights), which would otherwise return an unweighted score
    # that the caller believes is weighted.
    sample_weight = kwargs.get("sample_weight")
    if sample_weight is not None:
        raise NotImplementedError(
            "fast_aucs does not support sample_weight; use fast_roc_auc(..., sample_weight=...) "
            "for a weighted ROC AUC, or compute weighted PR AUC separately."
        )
    if isinstance(y_true, (pd.Series, pl.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_score, (pd.Series, pl.Series)):
        y_score = y_score.to_numpy()
    if y_score.ndim == 2:
        y_score = y_score[:, -1]
    _check_equal_length(y_true, y_score)
    desc_score_indices = _argsort_desc_for_metrics(y_score)  # iter338: dispatcher (unstable default, MLFRAME_METRICS_STABLE_SORT=1 to opt back)
    return fast_numba_aucs(y_true=y_true, y_score=y_score, desc_score_indices=desc_score_indices)


def average_precision_score(y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:
    """Average precision (PR-AUC), own numba kernel -- drop-in for ``sklearn.metrics.average_precision_score``.

    Binary-only. Computes AP = sum_n (R_n - R_{n-1}) * P_n over score-descending
    thresholds (implicit R_0 = 0 anchor), identical to sklearn's step-function AP;
    the shared ROC+PR kernel (``fast_numba_aucs``) carries a parity test to |1e-8|.
    Reuses ``fast_aucs`` and returns its PR-AUC component. ``sample_weight`` is not
    supported (the fused kernel has no weighted variant) and raises rather than
    silently returning an unweighted score.
    """
    _, pr_auc = fast_aucs(y_true, y_score, **kwargs)
    return pr_auc


@numba.njit(**NUMBA_NJIT_PARAMS)
def fast_numba_aucs_with_ks(y_true: np.ndarray, y_score: np.ndarray, desc_score_indices: np.ndarray) -> tuple[float, float, float]:
    """ROC AUC, PR AUC, and the KS statistic in ONE descending-order pass.

    The reliability/title report needs all three over the SAME score-desc order; computing them separately costs two passes plus
    KS's own ascending re-scan. KS = max over thresholds of |F_pos - F_neg|; the AUC walk already tracks (tps, fps, total_pos,
    total_neg) at every distinct-score boundary, so KS folds in for free as max|tps/total_pos - fps/total_neg| (signs cancel under
    abs vs the ascending CDF form, ties fold into the same single jump as the AUC boundary check). Indexes through
    ``desc_score_indices`` inline (no y_score/y_true gather temporaries). Bit-identical ROC/PR vs ``fast_numba_aucs``; KS within FP
    reduction-order (~1e-12, far below the 3-digit report precision and any decision boundary) vs ``ks_statistic``."""
    n = len(desc_score_indices)
    total_pos = 0.0
    for i in range(n):
        total_pos += y_true[desc_score_indices[i]]
    total_neg = n - total_pos
    if total_pos == 0 or total_neg == 0:
        return np.nan, np.nan, np.nan

    last_counted_fps = 0
    last_counted_tps = 0
    tps, fps = 0, 0
    roc_auc = 0.0
    prev_recall = 0.0
    pr_auc = 0.0
    ks = 0.0
    inv_pos = 1.0 / total_pos
    inv_neg = 1.0 / total_neg

    for i in range(n):
        idx = desc_score_indices[i]
        yt = y_true[idx]
        tps += yt
        fps += 1 - yt
        if i == n - 1 or y_score[desc_score_indices[i + 1]] != y_score[idx]:
            delta_fps = fps - last_counted_fps
            sum_tps = last_counted_tps + tps
            roc_auc += delta_fps * sum_tps
            last_counted_fps = fps
            last_counted_tps = tps

            current_precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
            current_recall = tps / total_pos
            delta_recall = current_recall - prev_recall
            pr_auc += delta_recall * current_precision
            prev_recall = current_recall

            d = tps * inv_pos - fps * inv_neg
            if d < 0.0:
                d = -d
            if d > ks:
                ks = d

    denom_roc = tps * fps * 2
    if denom_roc > 0:
        roc_auc /= denom_roc
    else:
        roc_auc = np.nan

    return roc_auc, pr_auc, ks


@numba.njit(**NUMBA_NJIT_PARAMS)
def fast_numba_aucs(y_true: np.ndarray, y_score: np.ndarray, desc_score_indices: np.ndarray) -> tuple[float, float]:
    """Numba-fast ROC AUC and PR AUC (average precision) in one descending-score pass.

    ``desc_score_indices`` must order rows by descending ``y_score``. Ties fold into a single
    threshold boundary. PR AUC matches ``sklearn.average_precision_score`` (anchored at recall 0).
    Returns ``(roc_auc, pr_auc)``; ``(nan, nan)`` when ``y_true`` is single-class (both undefined).
    """
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    total_pos = np.sum(y_true_sorted)
    total_neg = len(y_true_sorted) - total_pos
    if total_pos == 0 or total_neg == 0:
        # Single-class data: both ROC AUC and PR AUC are undefined
        return np.nan, np.nan

    # Variables for ROC AUC
    last_counted_fps = 0
    last_counted_tps = 0
    tps, fps = 0, 0
    roc_auc = 0.0

    # Variables for PR AUC. sklearn.average_precision_score computes
    #   AP = sum_n (R_n - R_{n-1}) * P_n
    # starting from R_0 = 0 (implicit anchor). The previous implementation already matches
    # this; we explicitly document the starting (R=0) anchor here. No behavioral change
    # needed - parity test below verifies |our - sklearn| < 1e-8.
    prev_recall = 0.0
    pr_auc = 0.0

    n = len(y_true_sorted)
    for i in range(n):
        tps += y_true_sorted[i]
        fps += 1 - y_true_sorted[i]

        if i == n - 1 or y_score_sorted[i + 1] != y_score_sorted[i]:
            # Update ROC AUC
            delta_fps = fps - last_counted_fps
            sum_tps = last_counted_tps + tps
            roc_auc += delta_fps * sum_tps
            last_counted_fps = fps
            last_counted_tps = tps

            # sklearn AP: sum over thresholds of (R_n - R_{n-1}) * P_n (current precision).
            current_precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
            current_recall = tps / total_pos
            delta_recall = current_recall - prev_recall
            pr_auc += delta_recall * current_precision  # Riemann sum
            prev_recall = current_recall

    # Normalize ROC AUC
    denom_roc = tps * fps * 2
    if denom_roc > 0:
        roc_auc /= denom_roc
    else:
        # Should not reach here due to early return, but handle defensively
        roc_auc = np.nan

    return roc_auc, pr_auc


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_brier_score_loss_seq(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return np.mean((y_true - y_prob) ** 2)


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_brier_score_loss_par(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Parallel variant. ~7.7x faster than seq at N=10M (verified
    on 8-thread numba runtime). Loses to seq below N~50k due to
    thread-spawn overhead -- the public ``fast_brier_score_loss``
    wrapper auto-dispatches based on N."""
    n = len(y_true)
    s = 0.0
    for i in numba.prange(n):
        d = y_true[i] - y_prob[i]
        s += d * d
    return s / n


# Crossover threshold for parallel kernels. See ``_numba_params.py`` (SSOT).
# Sum-reduction kernels (brier, log loss, prf1 counts) parallel-win from
# N~50-100k upwards. Multilabel row-loop kernels (subset accuracy, jaccard)
# win from N~10-50k. Conservative thresholds chosen to avoid the lose-band
# at low N.


def fast_brier_score_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (mean squared error of probabilities), auto seq/par.

    Sequential numba kernel below ~100k rows (cold-start cost
    of the parallel runtime exceeds the per-element gain). Parallel
    kernel above the threshold -- 7.7x faster at N=10M on an 8-thread
    runtime. Tunable via ``_PARALLEL_REDUCTION_THRESHOLD``.

    Returns ``np.nan`` on out-of-[0,1] or NaN probabilities, mirroring
    ``fast_log_loss_binary``: the kernels would otherwise square whatever
    garbage was passed and report a plausible-looking but meaningless score.
    """
    _check_equal_length(y_true, y_prob)
    _prob = np.asarray(y_prob, dtype=np.float64)
    if _prob.size and (not np.all(np.isfinite(_prob)) or _prob.min() < 0.0 or _prob.max() > 1.0):
        return np.nan
    if len(y_true) >= _PARALLEL_REDUCTION_THRESHOLD:
        return _fast_brier_score_loss_par(y_true, y_prob)
    return _fast_brier_score_loss_seq(y_true, y_prob)


# Backward-compat alias - older code and tests import `brier_score_loss` from this module.
# Keep the name visible but route it to the renamed fast_brier_score_loss so the intent is clear.
brier_score_loss = fast_brier_score_loss


def brier_and_precision_score(
    y_true,
    y_proba,
    precision_threshold: float = 0.5,
    brier_threshold: float = 0.25,
) -> float:
    """precision_score - fast_brier_score_loss when both thresholds pass, else 0.

    Brier must be <= brier_threshold and precision must be >= precision_threshold
    (at a 0.5 decision boundary) for a non-zero result. Useful as a conservative
    optimisation objective that rewards only models that are simultaneously
    calibrated AND precise at the top.
    """
    from ._core_precision_mape import fast_precision

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    brier = fast_brier_score_loss(y_true=y_true, y_prob=y_proba)
    # fast_brier_score_loss returns NaN on out-of-[0,1] / NaN probabilities. NaN > threshold is False, so without
    # this guard an invalid-probability model would slip through the gate and make this model-selection scorer
    # return NaN (or precision - NaN), silently changing which model wins. Treat a non-finite Brier as a hard fail.
    if not np.isfinite(brier) or brier > brier_threshold:
        return 0.0
    y_pred = (y_proba > 0.5).astype(int)
    # This is a strictly binary scorer. A multiclass y_true (labels outside {0, 1}) is a real input-contract
    # violation that must surface, not be silently ignored -- fast_precision(nclasses=2) would drop the extra
    # classes and report a corrupted precision, so validate before the numba kernel (mirrors what sklearn's
    # precision_score raised on this input). Returning 0.0 here would make the model-selection scorer report
    # the worst value and silently change which model wins.
    labels = np.unique(y_true)
    if labels.size and (labels.min() < 0 or labels.max() > 1):
        raise ValueError(f"brier_and_precision_score requires binary y_true in {{0, 1}}; got labels {labels.tolist()}")
    # fast_precision returns the positive-class (class-1) precision, matching binary precision_score
    # with zero_division=0 (empty positive predictions -> 0.0), and is the project's numba equivalent.
    precision = fast_precision(y_true.astype(np.int64), y_pred.astype(np.int64), nclasses=2, zero_division=0)
    if precision < precision_threshold:
        return 0.0
    return float(precision - brier)


def make_brier_precision_scorer(precision_threshold: float = 0.5, brier_threshold: float = 0.25):
    """Return an sklearn scorer wrapping brier_and_precision_score (needs_proba=True)."""
    from sklearn.metrics import make_scorer

    # New sklearn (>=1.4) replaces `needs_proba` with `response_method`; fall back
    # to the legacy kwarg for older versions.
    try:
        return make_scorer(
            brier_and_precision_score,
            response_method="predict_proba",
            greater_is_better=True,
            precision_threshold=precision_threshold,
            brier_threshold=brier_threshold,
        )
    except TypeError:
        return make_scorer(
            brier_and_precision_score,
            needs_proba=True,
            greater_is_better=True,
            precision_threshold=precision_threshold,
            brier_threshold=brier_threshold,
        )
