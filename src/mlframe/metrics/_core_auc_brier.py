"""AUC + Brier score kernels for ``mlframe.metrics.core``.

Carved from ``core.py``. Public symbols are re-exported from the parent.
"""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd
import polars as pl

from ._numba_params import NUMBA_NJIT_PARAMS, _PARALLEL_REDUCTION_THRESHOLD


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


def fast_aucs(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Compute both ROC AUC and PR AUC efficiently."""
    if isinstance(y_true, (pd.Series, pl.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_score, (pd.Series, pl.Series)):
        y_score = y_score.to_numpy()
    if y_score.ndim == 2:
        y_score = y_score[:, -1]
    desc_score_indices = _argsort_desc_for_metrics(y_score)  # iter338: dispatcher (unstable default, MLFRAME_METRICS_STABLE_SORT=1 to opt back)
    return fast_numba_aucs(y_true=y_true, y_score=y_score, desc_score_indices=desc_score_indices)


@numba.njit(**NUMBA_NJIT_PARAMS)
def fast_numba_aucs(y_true: np.ndarray, y_score: np.ndarray, desc_score_indices: np.ndarray) -> tuple[float, float]:
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
    """
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
    from sklearn.metrics import precision_score

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    brier = fast_brier_score_loss(y_true=y_true, y_prob=y_proba)
    if brier > brier_threshold:
        return 0.0
    y_pred = (y_proba > 0.5).astype(int)
    # precision_score on valid binary input with zero_division=0 does not raise; a failure here
    # signals a real input-contract violation (multiclass y, shape mismatch). Returning 0.0 would
    # silently make this model-selection scorer report the worst value, corrupting which model wins.
    precision = precision_score(y_true, y_pred, zero_division=0)
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
