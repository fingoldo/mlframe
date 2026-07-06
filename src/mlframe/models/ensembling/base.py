"""Leaf module: shared helpers, constants and numba probe for ``mlframe.models.ensembling``.

Carved out so the sibling modules (``_ensembling_score``,
``_ensembling_process_method``, ``_ensembling_predict``,
``_ensembling_quality_gate``) can import their shared dependencies from a
leaf instead of from ``ensembling.py`` itself.

That dodges the ``ensembling -> sibling -> ensembling`` import-cycle that
``tests/test_meta/test_no_import_cycles.py`` flags as a hard fail (the
parent re-imports the sibling at the bottom; the sibling needs helpers
defined in the parent above).

Every name here is bit-for-bit identical to the pre-split definition in
``ensembling.py``; the parent re-exports each name so historical
``from mlframe.models.ensembling import X`` imports continue to resolve.
"""


from __future__ import annotations

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger("mlframe.models.ensembling")

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from typing import Optional, Sequence

from scipy.stats import rankdata
import psutil
from joblib import delayed
import pandas as pd, numpy as np

from pyutilz.parallel import parallel_run, cpu_count_physical
from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names, compute_numerical_aggregates_numba, get_basic_feature_names

SIMPLE_ENSEMBLING_METHODS: list = "arithm harm median quad qube geo".split()

# Optional numba accelerator for the per-member MAE/STD reduction.
# Compiled once at import time so the hot path never pays the JIT cost. The size dispatcher
# (`_per_member_mae_std`) routes to the JIT version only when input is large enough to dominate
# the call-overhead -- for typical 5-model x ~1k-row ensembles the pure-numpy path is faster.
try:  # pragma: no cover -- env-dependent
    import numba as _numba

    # Wave 25 P1 fix (2026-05-20): switched from the naive E[X^2]-E[X]^2
    # variance formula to a two-pass mean+deviation form, which is
    # numerically stable for the regression-scale inputs (|d| ~ 1e3+) that
    # the previous form lost precision on. The ``< 0`` clamp left behind
    # in the source proved the cancellation was real. ``fastmath=False``
    # now so reductions stay associative-stable; for K independent members
    # the prange parallelism still gives the speedup we want.
    @_numba.njit(parallel=True, fastmath=False, cache=True)
    def _per_member_mae_std_njit(arr, median_preds):
        K = arr.shape[0]
        N = arr.shape[1]
        out_mae = np.empty(K, dtype=np.float64)
        out_std = np.empty(K, dtype=np.float64)
        if arr.ndim == 2:
            for k in _numba.prange(K):
                # Pass 1: compute mean(|d|).
                _s_diff = 0.0
                for i in range(N):
                    d = arr[k, i] - median_preds[i]
                    if d < 0:
                        d = -d
                    _s_diff += d
                mae = _s_diff / N
                # Pass 2: accumulate (|d| - mae)^2 directly. No catastrophic
                # cancellation: each summand is non-negative.
                _s_dev_sq = 0.0
                for i in range(N):
                    d = arr[k, i] - median_preds[i]
                    if d < 0:
                        d = -d
                    dev = d - mae
                    _s_dev_sq += dev * dev
                _var = _s_dev_sq / N
                out_mae[k] = mae
                out_std[k] = _var ** 0.5
        else:
            # 3-D (K, N, C): per-COLUMN MAE & std over the N axis, then averaged
            # across the C columns -- matches the numpy path exactly
            # (mae_per_col / std_per_col -> mean over C). The std MUST be
            # computed per column (anchored at that column's own mean) and only
            # then averaged: a pooled N*C series anchored at the global mean is a
            # DIFFERENT statistic (it folds in between-column variance), which is
            # the ~1e-4 disagreement that previously kept 3-D off the numba path.
            # MAE is unaffected (pooled mean == per-column-then-averaged), but it
            # is accumulated per column here for symmetry. Two-pass per column
            # (mean, then sum of squared deviations) keeps the same
            # no-catastrophic-cancellation guarantee as the 2-D branch.
            C = arr.shape[2]
            for k in _numba.prange(K):
                mae_acc = 0.0
                std_acc = 0.0
                for c in range(C):
                    # Pass 1: this column's mean of |d| over N.
                    _s_diff = 0.0
                    for i in range(N):
                        d = arr[k, i, c] - median_preds[i, c]
                        if d < 0:
                            d = -d
                        _s_diff += d
                    mae_c = _s_diff / N
                    # Pass 2: this column's deviation sum-of-squares anchored at
                    # its own mean (each summand non-negative -> stable).
                    _s_dev_sq = 0.0
                    for i in range(N):
                        d = arr[k, i, c] - median_preds[i, c]
                        if d < 0:
                            d = -d
                        dev = d - mae_c
                        _s_dev_sq += dev * dev
                    mae_acc += mae_c
                    std_acc += (_s_dev_sq / N) ** 0.5
                out_mae[k] = mae_acc / C
                out_std[k] = std_acc / C
        return out_mae, out_std

    _HAS_NUMBA_PER_MEMBER = True

    @_numba.njit(parallel=True, fastmath=False, cache=True)
    def _rrf_aggregate_probs_njit(preds_arr: np.ndarray, k: int) -> np.ndarray:
        """Parallel-over-M reciprocal-rank fusion of a (M, N, K) probability
        tensor. Each member's argsort over the N axis is independent across
        members; ``prange`` exploits multi-core CPUs to amortise the dominant
        O(N log N) sort cost.

        fastmath=False because ``1.0 / (k + rank + 1)`` is exact float
        division; fast-math reciprocal-approximations would drift the
        per-row sums on long N.

        Memory: allocates a (M, N, K) float64 intermediate (M*N*K*8 bytes).
        The dispatcher in ``_rrf_aggregate_probs`` guards via input-size
        threshold so this kernel only fires when the allocation fits.

        Bench results (2026-05-19, ``mlframe._benchmarks.bench_ensemble_rrf``)
        show 2.65x-4.06x speedup vs numpy across (M=5/10/20) x (N=10k/100k/1M)
        x (K=2/3) on the dev host; equivalence to ~1e-16 max abs delta.
        """
        M, N, K = preds_arr.shape
        per_member_recip = np.zeros((M, N, K), dtype=np.float64)
        for m in _numba.prange(M):
            for k_class in range(K):
                col_m = -preds_arr[m, :, k_class]
                # Stable sort so equal probabilities form contiguous runs in
                # `order`. numba's argsort supports only "quicksort"/"mergesort";
                # mergesort is the stable one.
                order = np.argsort(col_m, kind="mergesort")
                # Canonical RRF assigns TIED items EQUAL (averaged) ranks so
                # genuine ties contribute identical reciprocal-rank mass
                # regardless of array index. Detect tied runs of equal
                # `col_m` value and give every member of a run the average of
                # the 1-based positions it spans. Matches scipy
                # rankdata(method="average") used in the numpy fallback.
                n_pos = 0
                while n_pos < N:
                    j = n_pos + 1
                    while j < N and col_m[order[j]] == col_m[order[n_pos]]:
                        j += 1
                    # positions n_pos..j-1 (0-based) tie; average 1-based rank
                    avg_rank = (n_pos + 1 + j) / 2.0  # mean of (n_pos+1 .. j)
                    recip = 1.0 / (k + avg_rank)
                    for t in range(n_pos, j):
                        per_member_recip[m, order[t], k_class] = recip
                    n_pos = j
        aggregated = np.zeros((N, K), dtype=np.float64)
        for m in range(M):
            for n in range(N):
                for ki in range(K):
                    aggregated[n, ki] += per_member_recip[m, n, ki]
        if K > 1:
            for n in _numba.prange(N):
                row_sum = 0.0
                for ki in range(K):
                    row_sum += aggregated[n, ki]
                if row_sum > 0.0:
                    inv = 1.0 / row_sum
                    for ki in range(K):
                        aggregated[n, ki] *= inv
        return aggregated
except Exception:  # pragma: no cover
    _HAS_NUMBA_PER_MEMBER = False
    _per_member_mae_std_njit = None
    _rrf_aggregate_probs_njit = None


from mlframe.system import try_import_cupy  # noqa: E402

_, _HAS_CUPY = try_import_cupy()  # pragma: no cover -- env-dependent


def _stacked_corrcoef(M: np.ndarray) -> np.ndarray:
    """Correlation matrix of (K, N) stacked vectors with a size-dispatcher.

    Replaces the previous O(K^2) Python pair loop. Routes to cupy when both available and (K>50 OR
    N>1M); falls back to plain numpy otherwise. Both paths emit a (K, K) ndarray of Pearson
    correlations; constant rows surface as NaN entries (caller is expected to filter).
    """
    K = M.shape[0]
    N = M.shape[1] if M.ndim > 1 else 1
    use_cupy = _HAS_CUPY and (K > 50 or N > 1_000_000)
    if use_cupy:
        try:
            import cupy as _cp

            M_gpu = _cp.asarray(M)
            corr_gpu = _cp.corrcoef(M_gpu)
            return _cp.asnumpy(corr_gpu)
        except Exception as e:  # pragma: no cover -- defensive
            logger.debug("swallowed exception in base.py: %s", e)
            pass
    return np.corrcoef(M)


# Rank-fusion methods are NOT moment-based (RRF / Borda operate on rank
# positions, not on raw values), so they live in their own bucket and are
# not in SIMPLE_ENSEMBLING_METHODS by default. Classification flavours in
# score_ensemble opt-in by extending the iteration list at call-site;
# regression must skip RRF entirely (no rank notion on continuous y).
RANK_FUSION_METHODS: list = ["rrf", "rank_average"]

_MEANS_COLS: list = "arimean,quadmean,qubmean,geomean,harmmean".split(",")

basic_features_names = get_basic_feature_names(
    return_drawdown_stats=False,
    return_profit_factor=False,
    whiten_means=False,
)


# =============================================================================
# Streaming-accumulator Protocol -- 2026-04-24 Session 2
# =============================================================================
#
# Pluggable interface for ensemble aggregation that consumes one (N, K)
# probability matrix at a time, without materialising the full (M, N, K)
# tensor. Today's `ensemble_probabilistic_predictions` uses materialised
# `_preds_arr` (1 copy peak) -- fine up to ~M=6, N=1M, K=5 (~240MB). For
# bigger frames (prod 9M-row), implementations of this Protocol drop the
# peak to O(N*K) by streaming.
#
# Currently provided:
# - `_WelfordAccumulator` -- single-pass mean / variance via Welford. Used
#   directly by future big-frame ensembling path.
#
# Planned:
# - `_KahanTwoPassAccumulator` -- exact mean+var via Kahan-2pass (best precision)
# - `_PSquaredQuantileAccumulator` -- fixed-memory streaming quantile for median
# - `_TDigestAccumulator` -- alternative quantile sketch
#
# See docs/NUMERICAL_STABILITY_REPORT.md for empirical comparison of
# Welford / Kahan / naive on 7 synthetic distributions.


class StreamingAccumulator:
    """Protocol for single-pass probability-matrix aggregation.

    Concrete implementations:
    - ``push(arr)``: consume one (N, K) probability matrix
    - ``result()``: return ``dict[str, np.ndarray]`` with computed
      statistics ('mean', 'std', 'min', 'max', 'geomean', etc.)

    Memory budget: O(N*K) per accumulator instance (a few persistent
    accumulator arrays). Independent of M (number of models in ensemble).
    """

    def push(self, arr: np.ndarray) -> None:
        raise NotImplementedError

    def result(self) -> dict:
        raise NotImplementedError


class _WelfordAccumulator(StreamingAccumulator):
    """Welford single-pass mean+var+min+max for (N, K) probability matrices.

    Extends the classical scalar Welford to elementwise (N, K) updates.
    Memory: 4 persistent arrays of (N, K) -- mean, M2, min, max.

    Numerical stability: same as scalar Welford (each delta = arr - mean
    operates on differences of similar magnitude to var, no catastrophic
    cancellation). Per benchmarks (docs/NUMERICAL_STABILITY_REPORT.md):
    Welford recovers 19-37x precision on long arrays of well-conditioned
    data; loses to Kahan-2pass on extreme cancellation cases (large mean +
    smooth variance) -- for those use ``_KahanTwoPassAccumulator`` (TBD).

    Usage::

        acc = _WelfordAccumulator(shape=(N, K))
        for model in models:
            acc.push(model.predict_proba(X_val))
        stats = acc.result()  # {'mean': (N,K), 'std': (N,K), 'min': ..., 'max': ...}
    """

    def __init__(self, shape: tuple, dtype=np.float64):
        self.n = 0
        self.mean = np.zeros(shape, dtype=dtype)
        self.M2 = np.zeros(shape, dtype=dtype)
        self.min: Optional[np.ndarray] = None
        self.max: Optional[np.ndarray] = None
        self._dtype = dtype

    def push(self, arr: np.ndarray) -> None:
        if arr.shape != self.mean.shape:
            raise ValueError(f"_WelfordAccumulator.push: expected shape {self.mean.shape}, " f"got {arr.shape}")
        arr = arr.astype(self._dtype, copy=False)
        self.n += 1
        delta = arr - self.mean
        self.mean += delta / self.n
        delta2 = arr - self.mean  # mean already updated
        self.M2 += delta * delta2
        if self.min is None:
            self.min = arr.copy()
            self.max = arr.copy()
        else:
            np.minimum(self.min, arr, out=self.min)
            np.maximum(self.max, arr, out=self.max)

    def result(self) -> dict:
        if self.n == 0:
            return {"mean": None, "std": None, "var": None, "min": None, "max": None, "n": 0}
        var = self.M2 / max(self.n - 1, 1)  # sample variance (ddof=1)
        return {
            "mean": self.mean,
            "var": var,
            "std": np.sqrt(var),
            "min": self.min,
            "max": self.max,
            "n": self.n,
        }

    @staticmethod
    def combine(a: "_WelfordAccumulator", b: "_WelfordAccumulator") -> "_WelfordAccumulator":
        """Exact merge of two Welford accumulators (Chan et al. 1979).

        Enables parallelisation: split models across threads, each thread
        builds its own Welford, then combine at the end. Result is
        bit-identical to the single-stream version (no precision loss).
        """
        if a.n == 0:
            return b
        if b.n == 0:
            return a
        n = a.n + b.n
        delta = b.mean - a.mean
        out = _WelfordAccumulator(shape=a.mean.shape, dtype=a._dtype)
        out.n = n
        out.mean = a.mean + delta * b.n / n
        out.M2 = a.M2 + b.M2 + (delta**2) * a.n * b.n / n
        out.min = np.minimum(a.min, b.min) if (a.min is not None and b.min is not None) else (a.min if a.min is not None else b.min)
        out.max = np.maximum(a.max, b.max) if (a.max is not None and b.max is not None) else (a.max if a.max is not None else b.max)
        return out


# Streaming median (P^2-Quantile sketch, Jain & Chlamtac 1985) intentionally
# NOT implemented (explicit-design-decision (wave 69, 2026-05-20)). For typical
# M=5-10 ensembling members, exact materialised median via np.nanmedian(preds, axis=0)
# is O(1) wallclock + O(M*N*K) memory which fits the budget. P^2 wins only on
# big-M streams (CV with 100+ folds) and a correct (N, K)-vectorised impl
# needs a numba per-cell loop. Welford-mode median raises NotImplementedError
# in ensemble_probabilistic_predictions_streaming as the explicit signal-to-caller.

# *****************************************************************************************************************************************************
# Core ensembling functionality
# *****************************************************************************************************************************************************


def batch_numaggs(predictions: np.ndarray, get_numaggs_names_len: int, numaggs_kwds: dict, means_only: bool = True) -> np.ndarray:
    """Row-wise numeric-aggregate features for a (N, K) prediction matrix.

    Builds a ``(N, get_numaggs_names_len)`` float32 feature matrix by
    applying :func:`compute_numerical_aggregates_numba` to each row of
    ``predictions``. Shared by the ensembling-diversity check and quality
    gate so each member-prediction batch only pays the numba dispatch
    once per fold.
    """
    row_features = np.empty(shape=(len(predictions), get_numaggs_names_len), dtype=np.float32)
    for i in range(len(predictions)):
        arr = predictions[i, :]
        if means_only:
            numerical_features = compute_numerical_aggregates_numba(arr, geomean_log_mode=False, directional_only=False)
        else:
            numerical_features = compute_numaggs(arr=arr, **numaggs_kwds)
        row_features[i, :] = numerical_features
    return row_features


def enrich_ensemble_preds_with_numaggs(
    predictions: np.ndarray,
    models_names: Sequence = None,
    means_only: bool = False,
    keep_probs: bool = True,
    numaggs_kwds: dict = None,
    n_jobs: int = 1,
    only_physical_cores: bool = True,
) -> pd.DataFrame:
    """Probs are non-negative that allows more averages to be applied"""

    if models_names is None:
        models_names = []
    if numaggs_kwds is None:
        numaggs_kwds = {"whiten_means": False}

    if predictions.shape[1] >= 10:
        numaggs_kwds.update(dict(directional_only=False, return_hurst=True, return_entropy=True))
    else:
        numaggs_kwds.update(dict(directional_only=False, return_hurst=False, return_entropy=False))

    if means_only:
        numaggs_names = basic_features_names
    else:
        numaggs_names = list(get_numaggs_names(**numaggs_kwds))

    if keep_probs:
        probs_fields_names = models_names if models_names else [f"p{i}" for i in range(predictions.shape[1])]
    else:
        probs_fields_names = []

    if n_jobs == -1:
        n_jobs = cpu_count_physical() if only_physical_cores else (psutil.cpu_count(logical=True) or 1)

    if n_jobs and n_jobs != 1:
        batch_numaggs_results = parallel_run(
            [
                delayed(batch_numaggs)(predictions=arr, means_only=means_only, get_numaggs_names_len=len(numaggs_names), numaggs_kwds=numaggs_kwds)
                for arr in np.array_split(predictions, n_jobs)
            ],
            backend=None,
        )
        if keep_probs:
            idx = predictions.shape[1]
        else:
            idx = 0
        row_features = np.empty(shape=(len(predictions), len(numaggs_names) + idx), dtype=np.float32)
        row_features[:, idx:] = np.concatenate(batch_numaggs_results)
        if keep_probs:
            row_features[:, :idx] = predictions
    else:
        row_features = []
        for i in range(len(predictions)):
            arr = predictions[i, :]
            if means_only:
                numerical_features = compute_numerical_aggregates_numba(arr, geomean_log_mode=False, directional_only=False, whiten_means=False)
            else:
                numerical_features = compute_numaggs(arr=arr, **numaggs_kwds)
            if keep_probs:
                line = arr.tolist()
            else:
                line = []
            line.extend(numerical_features)

            row_features.append(line)

    columns = probs_fields_names + numaggs_names

    res = pd.DataFrame(data=row_features, columns=columns)
    if means_only:
        return res[probs_fields_names + _MEANS_COLS]
    else:
        return res


def _rrf_aggregate_probs(preds_arr: np.ndarray, k: int = 60) -> np.ndarray:
    """Per-class RRF aggregation of a stacked (M, N, K) probability tensor.

    Each member's per-class column is ranked across the N rows independently
    (descending: higher probability -> rank 1). The reciprocal-rank score
    ``1/(k + rank)`` is summed across members. The per-row K-vector is then
    re-normalised to sum to 1 so the output remains a proper probability
    distribution (per-row min-max-to-sum-1 is monotone in the RRF score and
    keeps the simplex invariant that AUC / logloss expect).

    The 1-D scalar-binary case is handled implicitly: when K=1, the re-norm
    step collapses to all-ones which is wrong for binary -- callers passing
    a (N, 1) shape get the raw RRF score back (no normalisation), and the
    binary path in score_ensemble drives this via the (N, 2) two-column
    probability matrix that classifiers already emit.
    """
    if preds_arr.ndim != 3:
        # Promote (M, N) -> (M, N, 1) for the scalar / 1-D path
        preds_arr = preds_arr.reshape(preds_arr.shape[0], preds_arr.shape[1], -1)
    M, N, K = preds_arr.shape
    # The K=1 binary/scalar path returns the raw RRF score (no per-row normalisation), which is NOT a probability and is calibrated only up to monotone rank order. Production callers stamping AUC / logloss on the result hit silently miscalibrated outputs; surface a one-line WARN so the path is grep-able. sklearn classifiers should always pass (M, N, 2) two-column probabilities; integrators feeding (M, N) decision_function-style scores must rank-aggregate elsewhere or sigmoid-transform first.
    if K == 1:
        logger.warning(
            "[_rrf_aggregate_probs] received K=1 (scalar / 1-column) inputs (M=%d, N=%d); output is the raw reciprocal-rank score and is NOT a calibrated probability. "
            "Pass two-column (N, 2) probabilities for binary classifiers, or wrap the result with sigmoid / min-max scaling before treating it as a probability.",
            M, N,
        )

    # Numba parallel-over-M fastpath. Bench (2026-05-19,
    # mlframe._benchmarks.bench_ensemble_rrf) shows 2.6-4.1x speedup across
    # (M=5/10/20) x (N=10k/100k/1M) x (K=2/3) -- njit wins at every
    # measured point, equivalence to ~1e-16 max abs delta. The dispatcher
    # gates on (a) numba available, (b) float64 input, (c) the (M, N, K)
    # intermediate fits in ~512MB so the kernel doesn't blow up RAM on
    # extreme (M, N) combinations. K=1 binary-score path also routes
    # through njit; the final normalisation is K>1 only inside the kernel.
    _intermediate_bytes = int(M) * int(N) * int(K) * 8
    _use_njit = (
        _HAS_NUMBA_PER_MEMBER
        and _rrf_aggregate_probs_njit is not None
        and preds_arr.dtype == np.float64
        and preds_arr.flags["C_CONTIGUOUS"]
        and _intermediate_bytes < 512 * 1024 * 1024  # 512MB allocation guard
    )
    if _use_njit:
        aggregated = _rrf_aggregate_probs_njit(preds_arr, int(k))
        if aggregated.shape[1] == 1:
            return aggregated[:, 0]
        return aggregated

    aggregated = np.zeros((N, K), dtype=np.float64)
    for k_class in range(K):
        # (M, N) -> per-column ranks across N rows for each member, descending.
        col = preds_arr[:, :, k_class]  # (M, N)
        # Canonical RRF assigns TIED items EQUAL (averaged) ranks so genuine
        # ties contribute identical reciprocal-rank mass regardless of their
        # array index. The prior argsort-of-argsort broke ties by position,
        # injecting index-dependent fusion noise on equal probabilities. Use
        # scipy rankdata(method="average") over the N axis. rankdata ranks
        # ascending (1=smallest); we want 1=largest, so rank on -col. This
        # CHANGES fused scores on ties vs. the old positional path (a
        # correctness fix, not bit-identical).
        ranks = rankdata(-col, method="average", axis=1)  # (M, N), 1-based avg ranks
        rr = 1.0 / (k + ranks.astype(np.float64))
        aggregated[:, k_class] = rr.sum(axis=0)

    # Re-normalise per-row to sum-1 when K > 1 (proper probability simplex).
    # K == 1 (1-D binary score path) returns the raw RRF score; the wrapper
    # call-site clip-to-[0,1] step handles bounding.
    if K > 1:
        row_sums = aggregated.sum(axis=1, keepdims=True)
        # Guard against degenerate row_sum==0 (all members tied identically across all rows)
        safe = np.where(row_sums > 0, row_sums, 1.0)
        aggregated = aggregated / safe

    if aggregated.shape[1] == 1:
        # Restore (N,) for 1-D input.
        return aggregated[:, 0]
    return aggregated


def rrf_ensemble(
    probs_list: Sequence[np.ndarray],
    k: int = 60,
) -> np.ndarray:
    """Reciprocal Rank Fusion ensemble of classifier probability outputs.

    Scale-invariant probabilistic blend: ranks each member's per-class column
    across the n_samples axis (descending), accumulates ``1/(k + rank)`` and
    re-normalises rows to a probability distribution. Survives wildly
    heterogeneous probability scales (calibrated CB ~ [0.1, 0.9], raw sigmoid
    /100 ~ [0.005, 0.01]) where arithmetic / geometric mean would be dominated
    by the larger-scale member.

    Use this for CLASSIFICATION ensembles only (binary -> (n_samples, 2) or
    (n_samples,) probability vectors; multiclass -> (n_samples, K)). It has
    no rank meaning for regression (continuous y has no rank-1 vs rank-N
    distinction in the same sense).

    Parameters
    ----------
    probs_list : sequence of arrays
        Each member's probability output. All must share shape:
        - (n_samples,) for binary positive-class probability
        - (n_samples, K) for K-class probabilities
    k : int
        RRF damping constant; 60 is the TREC default. Smaller k emphasises
        the top of each member's ranking; larger k flattens it.

    Returns
    -------
    np.ndarray
        Aggregated probabilities, same shape as each member.

    Raises
    ------
    ValueError
        On empty list, mismatched shapes, or non-positive k.
    """
    if not probs_list:
        raise ValueError("rrf_ensemble: probs_list is empty")
    if k <= 0:
        raise ValueError(f"rrf_ensemble: k must be > 0, got {k!r}")
    arrs = [np.asarray(p, dtype=np.float64) for p in probs_list]
    base_shape = arrs[0].shape
    for i, a in enumerate(arrs):
        if a.shape != base_shape:
            raise ValueError(
                f"rrf_ensemble: member {i} shape {a.shape!r} != member 0 shape {base_shape!r}"
            )
    stacked = np.stack(arrs, axis=0)
    return _rrf_aggregate_probs(stacked, k=k)


def combine_probs(
    stacked: np.ndarray,
    flavour: str,
    *,
    rrf_k: int = 60,
    sample_weight: Optional[np.ndarray] = None,
    ensure_prob_limits: bool = True,
    precomputed_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Single canonical per-flavour ensemble math, shared by train and predict.

    ``stacked`` is a (K, N, ...) tensor (already filtered to the post-gate member set).
    The function ONLY does the per-flavour reduction + clip + NaN-fallback; it does NOT
    run the outlier-member gate, the diversity scan, or the high-corr WARN. Those steps
    are train-time only and live inside ``ensemble_probabilistic_predictions`` /
    ``score_ensemble`` because they require ground-truth predictions (val / test / oof)
    that predict-time replay does not have.

    Extracted to eliminate the previous train-vs-predict math drift (Arch-1):
    pre-extraction the predict path reimplemented every flavour by hand, missing the
    train-side zero-handling / clip-before-blend / NaN-fallback. Replay drift surfaced
    when any member's preds approached 0 / 1 (harm) or had NaN rows (any). One helper,
    one set of semantics; predict-time output now matches train-stamp within fp64 tol.

    ``precomputed_weights`` (NNLS-derived, length M aligned with ``stacked`` axis 0) replace
    the uniform ``1/M`` weight used by the arithm/harm/quad/qube/geo flavours. ``median`` and
    ``rrf`` ignore the weights silently (no canonical weighted-median/weighted-rank-fusion is
    in scope here). Validation: shape must be (M,), all entries finite and non-negative; the
    function renormalises (warn-and-correct) if ``abs(sum(w)-1) > 1e-3``.
    """
    flav = (flavour or "").lower()
    if flav in ("", "arith", "mean"):
        flav = "arithm"
    elif flav in ("harmonic",):
        flav = "harm"
    elif flav in ("geomean", "geometric"):
        flav = "geo"
    elif flav in ("quadratic",):
        flav = "quad"
    elif flav in ("cubic",):
        flav = "qube"

    if ensure_prob_limits and flav in ("arithm", "harm", "quad", "qube", "median"):
        stacked = np.clip(stacked, 0.0, 1.0)

    weights_arr: Optional[np.ndarray] = None
    if precomputed_weights is not None:
        weights_arr = np.asarray(precomputed_weights, dtype=np.float64).reshape(-1)
        M = stacked.shape[0]
        if weights_arr.shape != (M,):
            raise ValueError(
                f"combine_probs: precomputed_weights shape {weights_arr.shape} does not match "
                f"stacked member axis (M={M},). Caller must align weights with the post-gate "
                f"member ordering of ``stacked``."
            )
        if not np.all(np.isfinite(weights_arr)):
            raise ValueError("combine_probs: precomputed_weights contains non-finite entries (NaN/inf).")
        if (weights_arr < 0).any():
            raise ValueError("combine_probs: precomputed_weights must be non-negative (NNLS contract).")
        _wsum = float(weights_arr.sum())
        if _wsum <= 0.0:
            raise ValueError("combine_probs: precomputed_weights sum to zero; no member contributes.")
        if abs(_wsum - 1.0) > 1e-3:
            logger.warning(
                "[ensemble] combine_probs: precomputed_weights sum=%.6f deviates from 1.0; renormalising.",
                _wsum,
            )
            weights_arr = weights_arr / _wsum
        elif _wsum != 1.0:
            weights_arr = weights_arr / _wsum

    if flav == "harm":
        # Harmonic mean: when any model predicts exactly 0, HM is defined as 0.
        any_zero = (stacked == 0).any(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / stacked
            if weights_arr is not None:
                # Weighted harmonic mean: 1 / sum_i w_i * (1 / p_i). Weights normalised to sum=1.
                w_shape = (weights_arr.shape[0],) + (1,) * (inv.ndim - 1)
                combined = 1.0 / np.sum(weights_arr.reshape(w_shape) * inv, axis=0)
            else:
                inv_sum = np.sum(inv, axis=0)
                combined = len(stacked) / inv_sum
        if any_zero.any():
            combined = np.where(any_zero, 0.0, combined)
    elif flav == "arithm":
        if weights_arr is not None:
            combined = np.average(stacked, axis=0, weights=weights_arr)
        else:
            combined = np.mean(stacked, axis=0)
    elif flav == "median":
        if sample_weight is not None:
            try:
                combined = np.quantile(stacked, 0.5, axis=0, weights=sample_weight, method="inverted_cdf")
            except TypeError:
                combined = np.median(stacked, axis=0)
        else:
            # ``np.median`` over ``np.quantile(stacked, 0.5, axis=0)`` -- same
            # rationale as the iter119 fix in ``compute_member_quality_gate``:
            # ``np.quantile`` with q=0.5 falls back to apply_along_axis, while
            # ``np.median`` uses numpy's dedicated C reduction. Bench at the
            # c0056 multilabel-chain shapes (K=3, N=40k, C=3) shows 11 ms ->
            # 7 ms (~1.5x); 2-D (K=3, N=200k): 19 ms -> 11 ms (~1.7x). Both
            # propagate NaN identically -- the existing non_finite_mask
            # arith-fallback further below catches any NaN cells either way.
            combined = np.median(stacked, axis=0)
    elif flav == "quad":
        sq = stacked * stacked
        if weights_arr is not None:
            combined = np.sqrt(np.maximum(np.average(sq, axis=0, weights=weights_arr), 0.0))
        else:
            combined = np.sqrt(np.mean(sq, axis=0))
    elif flav == "qube":
        # Underflow guard: for all-positive but extremely-small probs, cbrt(mean(p^3))
        # can lose precision near 1e-100 because p^3 underflows below float64 min.
        # Clip the pre-cube floor at the cube root of the smallest safe float64 (~1e-103).
        _safe = np.clip(stacked, 1e-103, None) if (stacked > 0).all() else stacked
        cu = _safe * _safe * _safe
        if weights_arr is not None:
            combined = np.cbrt(np.average(cu, axis=0, weights=weights_arr))
        else:
            combined = np.cbrt(np.mean(cu, axis=0))
    elif flav == "geo":
        with np.errstate(divide="ignore"):
            log_stack = np.log(np.clip(stacked, 1e-300, None))
            if weights_arr is not None:
                combined = np.exp(np.average(log_stack, axis=0, weights=weights_arr))
            else:
                combined = np.exp(np.mean(log_stack, axis=0))
    elif flav == "rrf":
        combined = _rrf_aggregate_probs(stacked, k=int(rrf_k))
    elif flav == "rank_average":
        # Rank-average fusion (mean of per-member row-ranks). Lazy import breaks the base<->selection cycle. Like RRF it
        # is a scale-invariant RANK score (not a calibrated probability); ``weights_arr`` is honoured when supplied.
        from .selection import rank_average_blend
        combined = rank_average_blend(stacked, normalise=True, weights=weights_arr)
    else:
        # Unrecognised flavour -> arithmetic mean fallback (matches the legacy predict-side default).
        if weights_arr is not None:
            combined = np.average(stacked, axis=0, weights=weights_arr)
        else:
            combined = np.mean(stacked, axis=0)

    # NaN/inf fallback to arithmetic mean. Train side ran this AFTER the flavour reduce;
    # predict now does the same so a single NaN cell doesn't poison the whole batch.
    non_finite_mask = ~np.isfinite(combined)
    if non_finite_mask.any():
        _arith = np.mean(stacked, axis=0)
        # Wave 78 (2026-05-21): hard-assert shape contract -- np.where broadcasts
        # silently on shape mismatch, which would silently produce wrong-shape
        # ensemble output if a future flavour returns a different reduce shape.
        assert combined.shape == _arith.shape, (
            f"ensemble combine: shape mismatch combined={combined.shape} vs arith fallback={_arith.shape}"
        )  # nosec B101 - internal invariant / dev-time sanity check, not a security gate
        combined = np.where(non_finite_mask, _arith, combined)

    if ensure_prob_limits:
        combined = np.clip(combined, 0.0, 1.0)

    return combined


def build_predictive_kwargs(train_data, test_data, val_data, is_regression: bool):
    """
    Build predictive_kwargs dict for classification or regression tasks.

    Parameters
    ----------
    train_data, test_data, val_data : tuple | np.ndarray | None
        Either a tuple (predictions, indices), or just predictions, or None.
    is_regression : bool
        Whether the task is regression (True) or classification (False).

    Returns
    -------
    dict
        predictive_kwargs containing appropriately filtered and flattened arrays.
    """

    def process(data, flatten=False):
        # Case 1: None -> None
        if data is None:
            return None

        # Case 2: Tuple or list -> try to unpack (preds, indices)
        if isinstance(data, (tuple, list)):
            if len(data) == 2:
                preds, indices = data
                if preds is None:
                    return None
                if indices is None:
                    result = preds
                else:
                    result = preds[indices]
            else:
                # Unexpected tuple/list length
                result = data[0]
        else:
            # Case 3: raw ndarray (no indices)
            result = data

        return result.flatten() if (flatten and result is not None) else result

    if not is_regression:
        return dict(
            train_probs=process(train_data),
            test_probs=process(test_data),
            val_probs=process(val_data),
        )
    else:
        return dict(
            train_preds=process(train_data, flatten=True),
            test_preds=process(test_data, flatten=True),
            val_preds=process(val_data, flatten=True),
        )


def compute_high_correlation_pairs(
    members: Sequence,
    member_tags: Sequence[str],
    threshold: float = 0.98,
) -> tuple[list[dict], Optional[str]]:
    """Return pairs of ensemble members whose predictions are correlated above ``threshold`` plus the split that fed the check.

    Diversity is checked once on whichever prediction array is universally available across members, in this precedence:
    ``val_preds -> test_preds -> train_preds -> val_probs -> test_probs -> train_probs``. Probabilistic outputs collapse
    to a single column (last) for the correlation proxy; full multinomial diversity is overkill for near-duplicate detection.
    Members with a constant vector on the chosen split (std == 0) or fewer than 2 finite shared samples are skipped, not flagged.

    No mutation here - the caller decides what to do (WARN, persist, drop). Today the only caller (``score_ensemble``) just warns.
    """
    pairs: list[dict] = []
    if len(members) < 2:
        return pairs, None
    arrays: list[np.ndarray] = []
    split_used: Optional[str] = None
    for attr in ("val_preds", "test_preds", "train_preds"):
        cand = [getattr(m, attr, None) for m in members]
        if all(p is not None for p in cand):
            arrays = [np.asarray(p, dtype=np.float64).ravel() for p in cand]
            split_used = attr
            break
    # When no preds attribute is available, fall back to probs. For multiclass (C>=3) we compute
    # per-class correlation matrices and AVERAGE the off-diagonal pair entries across classes --
    # the pre-fix flatten-then-Pearson interleaved per-row class entries into a single long vector
    # and computed Pearson over that, which mixes intra-row class structure with inter-row variation
    # and is not a meaningful diversity measure. Binary (C==2) collapses to a 1-column proxy because
    # the two columns are perfectly linearly dependent (sum to 1); the per-class average over the
    # two redundant columns equals the single-column Pearson by construction.
    probs_arrays: list[np.ndarray] = []
    if not arrays:
        for attr in ("val_probs", "test_probs", "train_probs"):
            cand = [getattr(m, attr, None) for m in members]
            if all(p is not None for p in cand):
                probs_arrays = [np.asarray(p, dtype=np.float64) for p in cand]
                split_used = attr
                break
    if arrays:
        if not all(a.size == arrays[0].size and a.size >= 2 for a in arrays):
            return pairs, split_used
        M_stack = np.vstack(arrays)  # (K, N)
        corr_matrix = _pairwise_corr_or_nan(M_stack)
        if corr_matrix is None:
            return pairs, split_used
        K_use = corr_matrix.shape[0]
        idx_use = np.arange(K_use)
        return _emit_pairs_above_threshold(corr_matrix, idx_use, member_tags, threshold, split_used)
    if probs_arrays:
        # All probs arrays must share (N, C) (or (N,) which we promote to (N, 1)).
        norm = []
        for a in probs_arrays:
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            norm.append(a)
        probs_arrays = norm
        if not all(a.shape == probs_arrays[0].shape and a.shape[0] >= 2 for a in probs_arrays):
            return pairs, split_used
        K_members = len(probs_arrays)
        n_classes = probs_arrays[0].shape[1]
        if n_classes == 1:
            # Single-column / binary-as-1D path: stack as (K, N) and reuse the standard corr.
            M_stack = np.vstack([a.ravel() for a in probs_arrays])
            corr_matrix = _pairwise_corr_or_nan(M_stack)
            if corr_matrix is None:
                return pairs, split_used
            return _emit_pairs_above_threshold(corr_matrix, np.arange(corr_matrix.shape[0]), member_tags, threshold, split_used)
        # Multiclass per-class correlation, then average the off-diagonal entries across classes.
        per_class_corrs: list[np.ndarray] = []
        for _ci in range(n_classes):
            M_stack_ci = np.vstack([a[:, _ci] for a in probs_arrays])
            corr_ci = _pairwise_corr_or_nan(M_stack_ci, return_full_shape=True, original_k=K_members)
            if corr_ci is not None:
                per_class_corrs.append(corr_ci)
        if not per_class_corrs:
            return pairs, split_used
        stack_corrs = np.stack(per_class_corrs, axis=0)
        with np.errstate(invalid="ignore"):
            avg_corr = np.nanmean(stack_corrs, axis=0)
        return _emit_pairs_above_threshold(avg_corr, np.arange(K_members), member_tags, threshold, split_used)
    return pairs, split_used


def _pairwise_corr_or_nan(M_stack: np.ndarray, *, return_full_shape: bool = False, original_k: Optional[int] = None) -> Optional[np.ndarray]:
    """Compute the (K, K) Pearson corr matrix of a stacked (K, N) array.

    Masks NaN-bearing columns and constant-row members (std==0). Returns ``None`` when fewer than
    2 finite columns OR fewer than 2 non-constant members remain. When ``return_full_shape=True``
    AND ``original_k`` is supplied, returns a (original_k, original_k) matrix with NaN-padded rows
    for skipped members; callers averaging across classes can then ``np.nanmean`` over per-class
    matrices of the same shape.
    """
    K = M_stack.shape[0]
    finite_cols = np.all(np.isfinite(M_stack), axis=0)
    if int(finite_cols.sum()) < 2:
        if return_full_shape and original_k is not None:
            return np.full((original_k, original_k), np.nan, dtype=np.float64)
        return None
    M_finite = M_stack[:, finite_cols]
    stds = M_finite.std(axis=1)
    nonconst = stds > 0
    if int(nonconst.sum()) < 2:
        if return_full_shape and original_k is not None:
            return np.full((original_k, original_k), np.nan, dtype=np.float64)
        return None
    M_use = M_finite[nonconst]
    K_use = M_use.shape[0]
    idx_use = np.flatnonzero(nonconst)
    corr_used = _stacked_corrcoef(M_use)
    if return_full_shape and original_k is not None:
        out = np.full((original_k, original_k), np.nan, dtype=np.float64)
        # Vectorised NaN-padded scatter: ``out[idx_use, idx_use]`` block-assign via ``np.ix_``
        # replaces the O(K_use^2) Python double loop (bit-identical; 5-17x at K=10-20).
        out[np.ix_(idx_use, idx_use)] = corr_used
        return out
    # When return_full_shape=False the caller expects an indexed-by-use matrix and will iterate
    # via idx_use externally; we return the dense submatrix and let the caller pass idx_use to
    # ``_emit_pairs_above_threshold``.
    # We need to expose idx_use to the caller; pack the corr_used into a full (K, K) NaN-padded
    # matrix so the iteration sites stay symmetric.
    out = np.full((K, K), np.nan, dtype=np.float64)
    # Same vectorised np.ix_ block-scatter as the return_full_shape branch above (bit-identical).
    out[np.ix_(idx_use, idx_use)] = corr_used
    return out


def _emit_pairs_above_threshold(
    corr_matrix: np.ndarray,
    idx_use: np.ndarray,
    member_tags: Sequence[str],
    threshold: float,
    split_used: Optional[str],
) -> tuple[list[dict], Optional[str]]:
    pairs: list[dict] = []
    K = corr_matrix.shape[0]
    for ii in range(K):
        for jj in range(ii + 1, K):
            corr = float(corr_matrix[ii, jj])
            if not np.isfinite(corr):
                continue
            if abs(corr) > threshold:
                m1 = member_tags[ii] if ii < len(member_tags) else f"member_{ii}"
                m2 = member_tags[jj] if jj < len(member_tags) else f"member_{jj}"
                pairs.append({"m1": m1, "m2": m2, "corr": corr})
    return pairs, split_used


from .member_metrics import (  # noqa: E402,F401
    _PER_MEMBER_KERNEL_NAME,
    _PER_MEMBER_NUMBA_FLOOR_ELEMENTS,
    _per_member_mae_std,
    _per_member_use_numba,
)


