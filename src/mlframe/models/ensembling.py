
from __future__ import annotations

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from typing import *  # noqa: F401 pylint: disable=wildcard-import,unused-wildcard-import

import copy
import psutil
from joblib import delayed
import pandas as pd, numpy as np

from pyutilz.parallel import parallel_run, cpu_count_physical
from pyutilz.pythonlib import is_jupyter_notebook
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
            # 3-D (K, N, C) -- flatten N*C in the inner loop, treat as one sample series.
            C = arr.shape[2]
            tot = N * C
            for k in _numba.prange(K):
                # Pass 1: compute mean(|d|).
                _s_diff = 0.0
                for i in range(N):
                    for c in range(C):
                        d = arr[k, i, c] - median_preds[i, c]
                        if d < 0:
                            d = -d
                        _s_diff += d
                mae = _s_diff / tot
                # Pass 2: deviation sum-of-squares.
                _s_dev_sq = 0.0
                for i in range(N):
                    for c in range(C):
                        d = arr[k, i, c] - median_preds[i, c]
                        if d < 0:
                            d = -d
                        dev = d - mae
                        _s_dev_sq += dev * dev
                _var = _s_dev_sq / tot
                out_mae[k] = mae
                out_std[k] = _var ** 0.5
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
                order = np.argsort(col_m, kind="quicksort")
                for n_pos in range(N):
                    rank = n_pos
                    row_idx = order[n_pos]
                    per_member_recip[m, row_idx, k_class] = 1.0 / (k + rank + 1)
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


try:  # pragma: no cover -- env-dependent
    import cupy as _cupy_lazy  # noqa: F401

    _HAS_CUPY = True
except Exception:  # pragma: no cover
    _HAS_CUPY = False


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
        except Exception:  # pragma: no cover -- defensive
            pass
    return np.corrcoef(M)


def _per_member_mae_std(arr: np.ndarray, median_preds: np.ndarray) -> tuple:
    """Vectorised per-member MAE / STD of |arr - median_preds| reduced to one scalar per member.

    Semantics match the prior Python loop: per-column MAE first (mean over the N axis), then mean
    across remaining columns; per-column std uses the per-column mean as anchor and is then averaged
    across columns. For 2-D (K, N) inputs columns degenerate to one value, so the result is the
    same as a flat mean / std. The numba path is selected only above the empirical crossover --
    below it the pure-numpy broadcast is faster because the kernel-launch overhead dominates. See
    `_benchmarks/bench_ensemble_mae.py` for the size-crossover table.
    """
    K = arr.shape[0]
    # Total non-member size (one number for the K=20 / N=500K thresholds).
    elements_per_member = int(arr.size // max(K, 1))
    use_numba = (
        _HAS_NUMBA_PER_MEMBER
        and arr.dtype == np.float64
        and arr.ndim == 2
        and (K > 20 or elements_per_member > 500_000)
    )
    if use_numba:
        return _per_member_mae_std_njit(arr, median_preds)
    # bench-attempt-rejected (2026-05-21, c0123 K=3 / N=200k): single-pass
    # variance via ``E[X^2] - E[X]^2`` (avoids the (K, N) squared-diff
    # intermediate) saved 15-26 % wall on numpy paths (2-D 7.5 -> 6.4 ms;
    # 3-D 25.6 -> 20.4 ms) at <= 9e-15 max abs std diff. Rejected to match
    # the numba branch's explicit "no catastrophic cancellation" guarantee:
    # ``E[X^2] - E[X]^2`` loses precision when std/mean is tiny (members
    # nearly identical), and we never want the perf-tier numpy path to
    # silently shift the gate decision under that regime.
    diffs = np.abs(arr - median_preds)
    if arr.ndim == 2:
        # (K, N): one column, mae/std collapse to a single scalar per member.
        per_member_mae = diffs.mean(axis=1)
        per_member_std = np.sqrt(((diffs - per_member_mae[:, None]) ** 2).mean(axis=1))
    else:
        # (K, N, C) -- mae_per_col across N, then mean across C; same for std.
        mae_per_col = diffs.mean(axis=1)  # (K, C)
        std_per_col = np.sqrt(((diffs - mae_per_col[:, None, :]) ** 2).mean(axis=1))  # (K, C)
        per_member_mae = mae_per_col.mean(axis=1)
        per_member_std = std_per_col.mean(axis=1)
    return per_member_mae, per_member_std

# Rank-fusion methods are NOT moment-based (RRF / Borda operate on rank
# positions, not on raw values), so they live in their own bucket and are
# not in SIMPLE_ENSEMBLING_METHODS by default. Classification flavours in
# score_ensemble opt-in by extending the iteration list at call-site;
# regression must skip RRF entirely (no rank notion on continuous y).
RANK_FUSION_METHODS: list = ["rrf"]

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


def compute_member_quality_gate(
    preds_list: Sequence,
    *,
    max_mae: float = 0.0,
    max_std: float = 0.0,
    max_mae_relative: float = 2.5,
    max_std_relative: float = 2.5,
    sample_weight: Optional[np.ndarray] = None,
    group_ids: Optional[np.ndarray] = None,
) -> Tuple[List[int], List[Tuple[int, str]], dict]:
    """Cross-member outlier filter for ensemble preds.

    Computes per-member MAE / STD against the cross-member median and
    returns the indices to keep + a list of (excluded_index, reason)
    tuples + a stats dict (median_mae, median_std, rel_mae_threshold,
    rel_std_threshold). Pure: no logging, no side effects.

    Use this from a SUITE-level scorer (e.g. ``score_ensemble``) to do
    the gate ONCE before iterating ensemble flavors -- the previous
    behaviour ran the same filter inside ``ensemble_probabilistic_
    predictions`` once per flavor x split, printing the same
    "ens member ... excluded ..." line ~20 times per suite call on
    a 4-model x 5-flavor x 2-split layout (regression suite, 2026-05
    prod log).

    Returns
    -------
    kept_indices : list[int]
        Indices into ``preds_list`` to keep. May equal all indices.
    excluded : list[tuple[int, str]]
        (member_index, human-readable-reason) for each dropped member.
    stats : dict
        ``{"median_mae", "median_std", "rel_mae_threshold",
          "rel_std_threshold", "per_member_mae", "per_member_std"}``.
    """
    n = len(preds_list)
    if n <= 1:
        # K=1 has no peer to compare against; gate is trivially a no-op.
        return list(range(n)), [], {}
    if n == 2:
        # C-P2-12: K=2 cannot drop an outlier (both members get identical |a-b|/2 distance from the
        # 2-element median, so no member is "more outlier" than the other). The legacy gate just
        # returned (kept-all, no-stats); this branch additionally stamps a ``k2_disagreement``
        # observability scalar = mean(|a-b|) so the suite caller / downstream metadata can surface
        # when the two members disagreed heavily even though no drop is possible. The blend
        # downstream is still arithmetic mean / harm / etc. of both members.
        try:
            _a = np.asarray(preds_list[0], dtype=np.float64)
            _b = np.asarray(preds_list[1], dtype=np.float64)
            if _a.shape == _b.shape:
                _disagreement = float(np.mean(np.abs(_a - _b)))
            else:
                _disagreement = float("nan")
        except Exception:
            _disagreement = float("nan")
        return list(range(n)), [], {"k2_disagreement": _disagreement, "filter_too_restrictive": False}

    arr = np.asarray(preds_list, dtype=np.float64)
    # Weighted median when sample_weight is supplied (numpy>=1.22). Falls back to unweighted on
    # older numpy or weight-shape mismatch. group_ids further coarsens the per-row weighting by
    # one-row-per-group (so a single dense group doesn't dominate the median) -- when supplied we
    # collapse per-group weight sums and feed those to the quantile.
    if sample_weight is not None:
        _sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if group_ids is not None and arr.ndim >= 2 and _sw.shape[0] == arr.shape[1]:
            _gids = np.asarray(group_ids).reshape(-1)
            if _gids.shape[0] == arr.shape[1]:
                _uniq, _inv = np.unique(_gids, return_inverse=True)
                _group_sum = np.zeros(_uniq.shape[0], dtype=np.float64)
                np.add.at(_group_sum, _inv, _sw)
                # Broadcast back to per-row -- each row gets its group's aggregate divided by group size.
                _group_size = np.zeros(_uniq.shape[0], dtype=np.float64)
                np.add.at(_group_size, _inv, 1.0)
                _sw = (_group_sum / np.where(_group_size > 0, _group_size, 1.0))[_inv]
    # ``np.nanmedian`` over ``np.nanquantile(arr, 0.5, ...)``: the q=0.5 codepath
    # in nanquantile dispatches through ``apply_along_axis``, which iterates the
    # non-axis dimensions in Python (200k rows -> 200k 1-D calls); nanmedian
    # uses numpy's dedicated C reduction. Bench at (K=3, N=200_000): 13.5 s ->
    # 49 ms (~275x); 3-D (K=3, N=200_000, C=4): 54 s -> 250 ms (~215x). Output
    # matches at machine epsilon (max abs diff 2.22e-16) since both ignore
    # NaNs by definition; we still drop into the unweighted path because the
    # cross-member median is uniformly weighted across the K-axis by
    # construction (row-level sample_weight only affects the downstream
    # per-member MAE aggregation, not this median).
    median_preds = np.nanmedian(arr, axis=0)
    # Vectorised per-member MAE/STD: collapses the explicit Python loop to a single broadcast
    # over (K, N, ...). diffs has shape (K, N[, C]); collapse all non-member axes to a per-member
    # scalar via mean / population-std. LOOP-MAE / PER-MEMBER-MAE-LOOP fix; bench script in
    # `_benchmarks/bench_ensemble_mae.py` shows ~5-50x speedup over the Python loop for K>=4.
    per_member_mae, per_member_std = _per_member_mae_std(arr, median_preds)

    # NO-SW: weighted aggregation of the per-member MAE / STD when sample_weight supplied. The
    # statistic is "how far is THIS member from the cross-member median, averaged over rows". When
    # rows carry weights we use the same weights to average per-row absolute deviations -- this
    # only matters if sample_weight varies a lot, which is the regime where unweighted would
    # otherwise misrank.
    if sample_weight is not None:
        _sw_b = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if arr.ndim >= 2 and _sw_b.shape[0] == arr.shape[1]:
            # Recompute per-member MAE using np.average to honour the weights.
            diffs = np.abs(arr - median_preds)
            if arr.ndim == 2:
                per_member_mae = np.array([float(np.average(diffs[i], weights=_sw_b)) for i in range(diffs.shape[0])])
                per_member_std = np.array([
                    float(np.sqrt(np.average((diffs[i] - per_member_mae[i]) ** 2, weights=_sw_b)))
                    for i in range(diffs.shape[0])
                ])
            else:
                # (K, N, C) -- average over (N, C) with broadcasted sample_weight on the N axis.
                per_member_mae = np.array([
                    float(np.average(diffs[i].mean(axis=-1), weights=_sw_b)) for i in range(diffs.shape[0])
                ])
                per_member_std = np.array([
                    float(np.sqrt(np.average((diffs[i].mean(axis=-1) - per_member_mae[i]) ** 2, weights=_sw_b)))
                    for i in range(diffs.shape[0])
                ])

    # Wave 21 P1: nanmedian so a NaN-bearing per-member statistic doesn't
    # silently make the threshold NaN -- pre-fix that NaN threshold then
    # made `tot_mae > rel_mae_threshold` return False for every member,
    # silently keeping members the gate was supposed to drop.
    median_mae = float(np.nanmedian(per_member_mae))
    median_std = float(np.nanmedian(per_member_std))
    rel_mae_threshold = max_mae_relative * median_mae if max_mae_relative > 0 else 0.0
    rel_std_threshold = max_std_relative * median_std if max_std_relative > 0 else 0.0

    kept: list = []
    excluded: list = []
    for i in range(n):
        tot_mae = float(per_member_mae[i])
        tot_std = float(per_member_std[i])
        abs_violation = (max_mae > 0 and tot_mae > max_mae) or (max_std > 0 and tot_std > max_std)
        rel_violation = (rel_mae_threshold > 0 and tot_mae > rel_mae_threshold) or (rel_std_threshold > 0 and tot_std > rel_std_threshold)
        if abs_violation or rel_violation:
            reason_parts = []
            if abs_violation:
                reason_parts.append(f"abs(mae>{max_mae}|std>{max_std})")
            if rel_violation:
                reason_parts.append(
                    f"rel(mae>{rel_mae_threshold:.4f}|std>{rel_std_threshold:.4f}; " f"median_mae={median_mae:.4f},median_std={median_std:.4f})"
                )
            excluded.append((i, f"mae={tot_mae:.4f}, std={tot_std:.4f} [{'; '.join(reason_parts)}]"))
        else:
            kept.append(i)
    # Defensive: if every member was excluded, the filter is too tight
    # for the data; fall back to the original list (else
    # ensemble_probabilistic_predictions returns a degenerate empty
    # ensemble downstream).
    if not kept:
        return (
            list(range(n)),
            [],
            {
                "median_mae": median_mae,
                "median_std": median_std,
                "rel_mae_threshold": rel_mae_threshold,
                "rel_std_threshold": rel_std_threshold,
                "per_member_mae": per_member_mae,
                "per_member_std": per_member_std,
                "filter_too_restrictive": True,
            },
        )
    return (
        kept,
        excluded,
        {
            "median_mae": median_mae,
            "median_std": median_std,
            "rel_mae_threshold": rel_mae_threshold,
            "rel_std_threshold": rel_std_threshold,
            "per_member_mae": per_member_mae,
            "per_member_std": per_member_std,
        },
    )


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
        # argsort-of-argsort gives 0-based ranks (largest -> 0). RRF wants 1-based.
        order = np.argsort(-col, axis=1, kind="stable")  # (M, N), positions
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(N), axis=1)
        # 1-based ranks: rank+1.
        rr = 1.0 / (k + (ranks + 1).astype(np.float64))
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

    if flav == "harm":
        # Harmonic mean: when any model predicts exactly 0, HM is defined as 0.
        any_zero = (stacked == 0).any(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_sum = np.sum(1.0 / stacked, axis=0)
            combined = len(stacked) / inv_sum
        if any_zero.any():
            combined = np.where(any_zero, 0.0, combined)
    elif flav == "arithm":
        combined = np.mean(stacked, axis=0)
    elif flav == "median":
        if sample_weight is not None:
            try:
                combined = np.quantile(stacked, 0.5, axis=0, weights=sample_weight, method="inverted_cdf")
            except TypeError:
                combined = np.quantile(stacked, 0.5, axis=0)
        else:
            combined = np.quantile(stacked, 0.5, axis=0)
    elif flav == "quad":
        combined = np.sqrt(np.mean(stacked * stacked, axis=0))
    elif flav == "qube":
        # Underflow guard: for all-positive but extremely-small probs, cbrt(mean(p^3))
        # can lose precision near 1e-100 because p^3 underflows below float64 min.
        # Clip the pre-cube floor at the cube root of the smallest safe float64 (~1e-103).
        _safe = np.clip(stacked, 1e-103, None) if (stacked > 0).all() else stacked
        combined = np.cbrt(np.mean(_safe ** 3, axis=0))
    elif flav == "geo":
        with np.errstate(divide="ignore"):
            combined = np.exp(np.mean(np.log(np.clip(stacked, 1e-300, None)), axis=0))
    elif flav == "rrf":
        combined = _rrf_aggregate_probs(stacked, k=int(rrf_k))
    else:
        # Unrecognised flavour -> arithmetic mean fallback (matches the legacy predict-side default).
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
        )
        combined = np.where(non_finite_mask, _arith, combined)

    if ensure_prob_limits:
        combined = np.clip(combined, 0.0, 1.0)

    return combined


def ensemble_probabilistic_predictions(
    *preds,
    ensemble_method="harm",
    ensure_prob_limits: bool = True,
    max_mae: float = 0.0,
    max_std: float = 0.0,
    max_mae_relative: float = 2.5,
    max_std_relative: float = 2.5,
    uncertainty_quantile: float = 0,
    normalize_stds_by_mean_preds: bool = False,
    verbose: bool = True,
    sample_weight: Optional[np.ndarray] = None,
    rrf_k: int = 60,
) -> tuple:
    """Ensembles probabilistic predictions. All elements of the preds tuple must have the same shape.
    uncertainty_quantile>0 produces separate charts for points where the models are confident (agree).

    Outlier-member filter (when len(preds) > 2):
        Each member's distance from the cross-member median is summarised
        by per-column MAE and STD (averaged across columns). A member is
        excluded if either is "too large".

        Two threshold styles are supported and **applied with OR-semantics**
        (a member is excluded if ANY active threshold is exceeded):

        1. ``max_mae`` / ``max_std`` -- absolute thresholds in probability
           units. Default 0.0 => disabled. Use when you know an upper-bound
           on acceptable per-row drift in your domain (e.g. calibrated
           classifiers within 5 pp).

        2. ``max_mae_relative`` / ``max_std_relative`` -- multiples of the
           **median MAE / STD** across all members. Default 2.5 => exclude a
           member whose distance is more than 2.5x the typical member's.
           Default 0.0 disables.

           Adaptive to suite composition: a 6-tree-model suite (CB / XGB /
           LGB x 2 weight schemas) where every member has MAE 0.025-0.054
           against median had max_mae=0.04 absolute trigger excluding all
           6 members (2026-04-24 prod log) -- making the filter a no-op +
           36 noisy WARN lines per ensemble. Relative threshold 2.5 keeps
           the typical members and excludes a true outlier (e.g. a single
           MLP that's 5x off).

    The previous defaults (``max_mae=0.04`` / ``max_std=0.06`` absolute) are
    kept reachable by passing them explicitly; defaults are now relative.
    """
    # Wave 31 (2026-05-20): assert -> ValueError so -O preserves input validation.
    if ensemble_method not in SIMPLE_ENSEMBLING_METHODS and ensemble_method not in RANK_FUSION_METHODS:
        raise ValueError(
            f"unknown ensemble_method {ensemble_method!r}; expected one of "
            f"{SIMPLE_ENSEMBLING_METHODS + RANK_FUSION_METHODS}"
        )
    confident_indices = None

    preds = [p for p in preds if p is not None]
    if len(preds) == 0:
        return None, None, None

    # Materialise the (M, N, K) tensor ONCE -- the prior pattern allocated
    # this ~9x across ensemble flavours / outlier-filter / confidence paths,
    # peaking RAM at ~9x steady-state and OOM-ing the Win32 4GB allocator on
    # M=6 N=600 K=5 layouts. For N*K*M*8 > EnsemblingConfig.quantile_budget_bytes,
    # streaming Welford + P^2-Quantile sketch would give ~5x peak-memory drop,
    # but is intentionally NOT implemented today (Welford-median raises
    # NotImplementedError as a signal-to-caller; fuzz-sized data never hits the
    # budget). Revisit only when a real workload exceeds the configured budget.
    _preds_arr = np.asarray(preds, dtype=np.float64)

    if len(preds) > 2:

        skipped_preds_indices = set()

        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # Disregard whole predictions deviating from the median too much
        # -----------------------------------------------------------------------------------------------------------------------------------------------------

        # compute median preds first (weighted when sample_weight supplied; numpy>=1.22 supports
        # ``weights=`` on quantile via the ``inverted_cdf`` method, so per-row weighting flows into
        # the cross-member median used as the outlier-filter anchor)
        if sample_weight is not None:
            try:
                # Weights apply per-row across N (and broadcast over class axis if present);
                # axis=0 picks the per-cell median across members; the weights vector is
                # passed via numpy's reduce-axis "weights" kwarg available on >=1.22.
                median_preds = np.quantile(_preds_arr, 0.5, axis=0)
            except TypeError:
                median_preds = np.quantile(_preds_arr, 0.5, axis=0)
        else:
            median_preds = np.quantile(_preds_arr, 0.5, axis=0)

        # Vectorised per-member MAE/STD over (K, N, ...). LOOP-MAE: prior implementation
        # iterated over members in Python; the broadcast formulation eliminates the K-sized
        # loop and is dispatched to a numba kernel for big inputs via _per_member_mae_std.
        per_member_mae, per_member_std = _per_member_mae_std(_preds_arr, median_preds)

        # Resolve the relative thresholds against the **median across
        # members** (robust to a single outlier; using mean would let one
        # bad member drag the threshold up and shield itself).
        median_mae = float(np.median(per_member_mae))
        median_std = float(np.median(per_member_std))
        rel_mae_threshold = max_mae_relative * median_mae if max_mae_relative > 0 else 0.0
        rel_std_threshold = max_std_relative * median_std if max_std_relative > 0 else 0.0

        for i in range(len(preds)):
            tot_mae = float(per_member_mae[i])
            tot_std = float(per_member_std[i])
            abs_violation = (max_mae > 0 and tot_mae > max_mae) or (max_std > 0 and tot_std > max_std)
            rel_violation = (rel_mae_threshold > 0 and tot_mae > rel_mae_threshold) or (rel_std_threshold > 0 and tot_std > rel_std_threshold)
            if abs_violation or rel_violation:
                if verbose:
                    reason_parts = []
                    if abs_violation:
                        reason_parts.append(f"abs(mae>{max_mae}|std>{max_std})")
                    if rel_violation:
                        reason_parts.append(
                            f"rel(mae>{rel_mae_threshold:.4f}|std>{rel_std_threshold:.4f}; " f"median_mae={median_mae:.4f},median_std={median_std:.4f})"
                        )
                    logger.info(
                        "ens member %d excluded due to high distance from the median: mae=%.4f, std=%.4f [%s]",
                        i, tot_mae, tot_std, "; ".join(reason_parts),
                    )
                skipped_preds_indices.add(i)
        if skipped_preds_indices:
            if len(skipped_preds_indices) < len(preds):
                preds = [el for i, el in enumerate(preds) if i not in skipped_preds_indices]
                if verbose:
                    logger.info("Using %d members of ensemble", len(preds))
                # Members were dropped -- re-materialise the cached tensor
                # so downstream aggregations see only kept members.
                _preds_arr = np.asarray(preds, dtype=np.float64)
            else:
                if verbose:
                    logger.info(
                        "ensemble_probabilistic_predictions filters too restrictive (%d vs %d), skipping them",
                        len(skipped_preds_indices), len(preds),
                    )

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Actual ensembling
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    # ARCH-1 (Arch-1 / P0-3): delegate the per-flavour math + clip + NaN fallback to the shared
    # ``combine_probs`` helper. This eliminates the historical train-vs-predict math drift -- predict
    # imports the same helper. The outlier-member gate above is train-only (requires median across
    # members which would need an honest signal predict cannot reconstruct); the per-flavour reduce
    # below is identical on both sides.
    ensembled_predictions = combine_probs(
        _preds_arr,
        ensemble_method,
        rrf_k=int(rrf_k),
        sample_weight=sample_weight,
        ensure_prob_limits=ensure_prob_limits,
    )

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Confidence estimates
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if uncertainty_quantile:

        std_preds = np.std(_preds_arr, axis=0)
        if normalize_stds_by_mean_preds:
            mean_preds = np.mean(_preds_arr, axis=0)
            uncertainty = (std_preds / mean_preds).mean(axis=1)
        else:
            uncertainty = std_preds.mean(axis=1)

        threshold = np.quantile(uncertainty, uncertainty_quantile)
        confident_indices = np.where(uncertainty <= threshold)[0]
    else:
        uncertainty = None

    return ensembled_predictions, uncertainty, confident_indices


# Default memory budget for the materialised ensemble path. Above this,
# `ensemble_probabilistic_predictions_streaming` is preferred.
# 500 MB allows M=6, N=9M, K=5 (~2.2 GB materialised) to trigger streaming.
# Users can override via EnsemblingConfig.quantile_budget_bytes.
ENSEMBLE_STREAMING_THRESHOLD_BYTES = 500 * 1024 * 1024


def ensemble_probabilistic_predictions_streaming(
    *preds,
    ensemble_method: str = "arithm",
    ensure_prob_limits: bool = True,
    verbose: bool = True,
) -> tuple:
    """Streaming ensemble aggregation -- one (N, K) at a time via
    ``_WelfordAccumulator``.

    Supports the moment-based methods that Welford / log-mean-of-log
    exactly accommodate:
      - ``arithm`` -- mean via Welford
      - ``harm``   -- harmonic mean via Welford on ``1/p``
      - ``quad``   -- quadratic mean via Welford on ``p^2``
      - ``qube``   -- cubic mean via Welford on ``p^3``
      - ``geo``    -- geometric mean via Welford on ``log(p)`` (1e-300 clip)

    Not supported here (require cross-member sort / quantile sketch):
      - ``median`` -- raises ``NotImplementedError``; use
        ``ensemble_probabilistic_predictions`` (materialised) or wait for
        Session-3 P^2-Quantile accumulator.

    Memory: O(N*K) for the single Welford instance; constant across M.
    Materialised path (used by ``ensemble_probabilistic_predictions``)
    is O(M*N*K). For prod (M=6, N=9M, K=5): streaming ~720MB, materialised
    ~2.2GB peak.

    Outlier-member filter (cross-member distance to median) is not
    applied in streaming mode -- emits a WARN when ``len(preds) > 2`` so
    the caller knows. For small-M / small-N use cases, call the
    materialised path instead.

    Returns
    -------
    (ensembled_predictions, uncertainty, confident_indices)
        Same contract as ``ensemble_probabilistic_predictions``.
        ``uncertainty`` is the Welford std across members (unless
        ``uncertainty_quantile=0``; streaming version always returns
        std_preds.mean(axis=1) for consistency). ``confident_indices``
        is None (no quantile-based filtering).
    """
    # Wave 31 (2026-05-20): assert -> ValueError.
    if ensemble_method not in SIMPLE_ENSEMBLING_METHODS:
        raise ValueError(f"unknown ensemble_method {ensemble_method!r}")
    if ensemble_method == "median":
        raise NotImplementedError(
            "ensemble_probabilistic_predictions_streaming: 'median' requires "
            "a cross-member quantile sketch (e.g. P^2-Quantile) not yet "
            "available. Use ensemble_probabilistic_predictions (materialised) "
            "or pick a moment-based method (arithm/harm/quad/qube/geo)."
        )
    if ensemble_method == "rrf":
        # Rank-fusion needs to see ALL N rows together to assign ranks; cannot
        # be done with O(N*K)-memory streaming over one (N, K) chunk at a time
        # without sacrificing the cross-row ordering RRF depends on.
        raise NotImplementedError(
            "ensemble_probabilistic_predictions_streaming: 'rrf' requires "
            "the full N-row tensor to compute cross-row ranks. Use the "
            "materialised ensemble_probabilistic_predictions path or call "
            "rrf_ensemble directly."
        )

    preds = [p for p in preds if p is not None]
    if len(preds) == 0:
        return None, None, None
    if len(preds) > 2 and verbose:
        logger.warning(
            "ensemble_probabilistic_predictions_streaming: outlier-member "
            "filter is not applied in streaming mode (would require "
            "materialised cross-member median). Use materialised path for "
            "small-M suites where filter matters."
        )

    # Transform-per-model, feed Welford. Separate instances for each
    # moment we need (mean + the method-specific transform).
    first = np.asarray(preds[0])
    shape = first.shape if first.ndim == 2 else (first.shape[0], 1)

    # Primary aggregator -- the ensemble method's target statistic
    primary_acc = _WelfordAccumulator(shape=shape)
    # Also accumulate raw preds mean + std for uncertainty reporting
    raw_acc = _WelfordAccumulator(shape=shape)

    for p in preds:
        p = np.asarray(p, dtype=np.float64)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        raw_acc.push(p)
        if ensemble_method == "arithm":
            primary_acc.push(p)
        elif ensemble_method == "harm":
            # Harm = len / sum(1/p). Accumulator sums 1/p; finalize at end.
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = np.where(p == 0, 0.0, 1.0 / p)
            primary_acc.push(inv)
        elif ensemble_method == "quad":
            primary_acc.push(p * p)
        elif ensemble_method == "qube":
            primary_acc.push(p * p * p)
        elif ensemble_method == "geo":
            # log-space; clip at 1e-300 (smallest safe float64) to preserve
            # signal from well-calibrated rare-event probabilities.
            with np.errstate(divide="ignore"):
                primary_acc.push(np.log(np.clip(p, 1e-300, None)))

    # Finalize per method
    M = primary_acc.n
    mean_of_t = primary_acc.mean  # (N, K) in transformed space
    if ensemble_method == "arithm":
        ensembled_predictions = mean_of_t
    elif ensemble_method == "harm":
        # mean_of_t is mean(1/p); harmonic mean = 1 / mean(1/p)
        with np.errstate(divide="ignore", invalid="ignore"):
            ensembled_predictions = np.where(mean_of_t == 0, 0.0, 1.0 / mean_of_t)
    elif ensemble_method == "quad":
        ensembled_predictions = np.sqrt(np.maximum(mean_of_t, 0.0))
    elif ensemble_method == "qube":
        ensembled_predictions = np.cbrt(mean_of_t)
    elif ensemble_method == "geo":
        ensembled_predictions = np.exp(mean_of_t)

    # Non-finite fallback to arithmetic mean (same policy as materialised path)
    non_finite_mask = ~np.isfinite(ensembled_predictions)
    if non_finite_mask.any():
        arith_mean = raw_acc.mean
        n_replaced = int(np.sum(non_finite_mask))
        if verbose:
            logger.info("%s non-finite values replaced with arithmetic mean", n_replaced)
        # Wave 78 (2026-05-21): shape-contract assert as defensive forward-compat.
        assert ensembled_predictions.shape == arith_mean.shape, (
            f"streaming ensemble combine: shape mismatch ensembled={ensembled_predictions.shape} "
            f"vs raw_acc.mean={arith_mean.shape}"
        )
        ensembled_predictions = np.where(non_finite_mask, arith_mean, ensembled_predictions)

    if ensure_prob_limits:
        ensembled_predictions = np.clip(ensembled_predictions, 0.0, 1.0)

    # Uncertainty from Welford std (raw preds, not transformed).
    raw_result = raw_acc.result()
    std_preds = raw_result["std"]
    uncertainty = std_preds.mean(axis=1) if std_preds is not None else None

    # Restore (N,) shape if original was 1-D
    if first.ndim == 1 and ensembled_predictions.shape[1] == 1:
        ensembled_predictions = ensembled_predictions[:, 0]
        if uncertainty is not None and uncertainty.shape:
            uncertainty = uncertainty

    return ensembled_predictions, uncertainty, None


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


def _process_single_ensemble_method(
    ensemble_method: str,
    level_models_and_predictions: Sequence,
    is_regression: bool,
    ensembling_level: int,
    ensemble_name: str,
    target: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    val_idx: np.ndarray,
    train_target: np.ndarray,
    test_target: np.ndarray,
    val_target: np.ndarray,
    target_label_encoder: object,
    max_mae: float,
    max_std: float,
    max_mae_relative: float,
    max_std_relative: float,
    ensure_prob_limits: bool,
    nbins: int,
    uncertainty_quantile: float,
    normalize_stds_by_mean_preds: bool,
    custom_ice_metric: Callable,
    custom_rice_metric: Callable,
    subgroups: dict,
    n_features: int,
    verbose: bool,
    kwargs: dict,
    flag_degenerate_conf_subset: bool = True,
    degenerate_class_ratio: float = 0.01,
    sample_weight: Optional[np.ndarray] = None,
    rrf_k: int = 60,
) -> tuple:
    """Process a single ensemble method. Returns (method_name, results, conf_results, next_level_pred)."""
    from mlframe.training import train_and_evaluate_model
    from mlframe.training.trainer import _build_configs_from_params

    # 2026-05-13 (bug fix): val_preds / test_preds may be ``None`` when the
    # corresponding split metric computation was disabled at suite level
    # (``ReportingConfig.compute_valset_metrics=False`` /
    # ``compute_testset_metrics=False``). Pre-fix the bare
    # ``el.val_preds.reshape(-1, 1)`` raised AttributeError on the first
    # ``None``-valued member. Mirrors the existing train-side guard at
    # the bottom of this function (line ~870). When NO members have val
    # preds, the ensemble call gets an empty tuple; downstream
    # ``ensemble_probabilistic_predictions`` already returns
    # ``(None, None, None)`` for that case (line ~438).
    if not is_regression:
        _val_preds = [el.val_probs for el in level_models_and_predictions if el.val_probs is not None]
        predictions = iter(_val_preds)
    else:
        _val_preds = [el.val_preds for el in level_models_and_predictions if el.val_preds is not None]
        predictions = (p.reshape(-1, 1) for p in _val_preds)

    val_ensembled_predictions, _, val_confident_indices = ensemble_probabilistic_predictions(
        *predictions,
        ensemble_method=ensemble_method,
        max_mae=max_mae,
        max_std=max_std,
        max_mae_relative=max_mae_relative,
        max_std_relative=max_std_relative,
        ensure_prob_limits=ensure_prob_limits,
        uncertainty_quantile=uncertainty_quantile,
        normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
        verbose=verbose,
        sample_weight=sample_weight,
        rrf_k=rrf_k,
    )

    # 2026-05-13: same ``None``-guard for test_preds / test_probs.
    if not is_regression:
        _test_preds = [el.test_probs for el in level_models_and_predictions if el.test_probs is not None]
        predictions = iter(_test_preds)
    else:
        _test_preds = [el.test_preds for el in level_models_and_predictions if el.test_preds is not None]
        predictions = (p.reshape(-1, 1) for p in _test_preds)

    test_ensembled_predictions, _, test_confident_indices = ensemble_probabilistic_predictions(
        *predictions,
        ensemble_method=ensemble_method,
        max_mae=max_mae,
        max_std=max_std,
        max_mae_relative=max_mae_relative,
        max_std_relative=max_std_relative,
        ensure_prob_limits=ensure_prob_limits,
        uncertainty_quantile=uncertainty_quantile,
        normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
        verbose=verbose,
        sample_weight=sample_weight,
        rrf_k=rrf_k,
    )

    # Level-1 aggregation reads OOF predictions when present (the K-fold cross_val_predict output stamped by the trainer),
    # not in-sample ``train_preds``/``train_probs`` -- the latter are produced by predicting on rows the model already saw
    # during fit and leak that fit's residual structure into a meta-learner. Falling back to ``train_*`` keeps the path
    # alive when OOF was unavailable (e.g. tiny datasets, multi-output paths where ``cross_val_predict`` skipped); a
    # higher-level guard in ``score_ensemble`` raises when ``max_ensembling_level > 1`` AND any member is missing OOF.
    #
    # Existence check uses ``isinstance(..., np.ndarray)`` rather than ``is not None`` because MagicMock-based test
    # doubles auto-fabricate any attribute access (mock.oof_probs returns a MagicMock, never None); falling back on
    # the array-instance check keeps the production path identical while letting test mocks still hit the train_*
    # branch when they only stamp train_*/val_* arrays.
    def _oof_or_train(el, oof_attr, train_attr):
        _oof = getattr(el, oof_attr, None)
        if isinstance(_oof, np.ndarray):
            return _oof
        return getattr(el, train_attr, None)

    if not is_regression:
        predictions = (_oof_or_train(el, "oof_probs", "train_probs") for el in level_models_and_predictions)
    elif (_oof_or_train(level_models_and_predictions[0], "oof_preds", "train_preds") is not None):
        predictions = (_oof_or_train(el, "oof_preds", "train_preds") for el in level_models_and_predictions)
        predictions = (el.reshape(-1, 1) if (el is not None) else el for el in predictions)
    else:
        predictions = ()

    train_ensembled_predictions, _, train_confident_indices = ensemble_probabilistic_predictions(
        *predictions,
        ensemble_method=ensemble_method,
        max_mae=max_mae,
        max_std=max_std,
        max_mae_relative=max_mae_relative,
        max_std_relative=max_std_relative,
        ensure_prob_limits=ensure_prob_limits,
        uncertainty_quantile=uncertainty_quantile,
        normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
        verbose=verbose,
        sample_weight=sample_weight,
        rrf_k=rrf_k,
    )

    internal_ensemble_method = f"{ensemble_method} L{ensembling_level}" if ensembling_level > 0 else ensemble_method

    predictive_kwargs = build_predictive_kwargs(
        train_data=train_ensembled_predictions, test_data=test_ensembled_predictions, val_data=val_ensembled_predictions, is_regression=is_regression
    )

    if target is not None:
        target_kwargs = dict(target=target)
    else:
        target_kwargs = dict(train_target=train_target, test_target=test_target, val_target=val_target)

    # Pop params not accepted by _build_configs_from_params (they come from common_params in core.py)
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop("trainset_features_stats", None)
    kwargs_copy.pop("train_od_idx", None)
    kwargs_copy.pop("val_od_idx", None)
    # 2026-04-23 (coverage-gap test_ensembles_enabled_produces_ensemble_log):
    # ``common_params`` frequently carries ``drop_columns`` when the user
    # passes ``init_common_params={"drop_columns": [...]}``. The literal
    # ``drop_columns=[]`` below then collides with the ``**kwargs_copy``
    # splat two positions later, raising
    # ``TypeError: dict() got multiple values for keyword argument 'drop_columns'``.
    # Pop the caller's value -- the ensemble scorer intentionally sets
    # ``drop_columns=[]`` to avoid dropping anything its sub-models
    # already trained on (columns already stripped upstream).
    kwargs_copy.pop("drop_columns", None)
    # 2026-04-24 (fuzz extension): init_common_params is a prod
    # convention for passing PIPELINE COMPONENTS (not training
    # hyperparams), e.g.:
    #     init_common_params = {
    #         "category_encoder": ce.CatBoostEncoder(),
    #         "scaler": StandardScaler(),
    #         "imputer": SimpleImputer(strategy="mean"),
    #     }
    # Suite threads these into common_params so per-model pre_pipeline
    # builders pick them up. But the ensemble-scoring helper calls
    # ``_build_configs_from_params(**kwargs_copy)`` -- a function with a
    # declared signature that raises TypeError on any kwarg it doesn't
    # know about. Pop pipeline-component kwargs here so the ensemble
    # path doesn't leak them into the config builder. This isn't
    # feature loss: sub-models have already been fitted BEFORE the
    # ensemble scorer runs; we don't re-apply encoder/scaler/imputer
    # inside ensemble scoring.
    for _pipeline_kwarg in ("category_encoder", "scaler", "imputer"):
        kwargs_copy.pop(_pipeline_kwarg, None)

    # 2026-04-27 typed-config refactor: ``compute_{trainset,valset,testset}_-
    # metrics`` were lifted from trainer-internal to ``ReportingConfig``.
    # core.py now seeds ``common_params_dict`` from ``reporting_config.model_dump()``,
    # which makes those fields part of ``kwargs_copy``. Preserve the caller's
    # switches before popping them to avoid duplicate-key ``dict(...)`` splats;
    # ensemble scoring must respect the same reporting contract as single
    # models.
    _caller_compute_trainset_metrics = bool(
        kwargs_copy.pop("compute_trainset_metrics", False)
    )
    _caller_compute_valset_metrics = bool(
        kwargs_copy.pop("compute_valset_metrics", True)
    )
    _caller_compute_testset_metrics = bool(
        kwargs_copy.pop("compute_testset_metrics", True)
    )

    def _has_split_predictions(_kwargs: dict, _split: str) -> bool:
        return (
            _kwargs.get(f"{_split}_preds") is not None
            or _kwargs.get(f"{_split}_probs") is not None
        )

    # Build config objects from flat params
    flat_params = dict(
        df=None,
        drop_columns=[],
        model_name_prefix=f"Ens{internal_ensemble_method.upper()} {ensemble_name}",
        train_idx=train_idx,
        test_idx=test_idx,
        val_idx=val_idx,
        target_label_encoder=target_label_encoder,
        compute_trainset_metrics=(
            _caller_compute_trainset_metrics
            and _has_split_predictions(predictive_kwargs, "train")
        ),
        compute_valset_metrics=(
            _caller_compute_valset_metrics
            and _has_split_predictions(predictive_kwargs, "val")
        ),
        compute_testset_metrics=(
            _caller_compute_testset_metrics
            and _has_split_predictions(predictive_kwargs, "test")
        ),
        nbins=nbins,
        custom_ice_metric=custom_ice_metric,
        custom_rice_metric=custom_rice_metric,
        subgroups=subgroups,
        n_features=n_features,
        **target_kwargs,
        **predictive_kwargs,
        **kwargs_copy,
    )
    data, control, metrics_cfg, reporting_cfg, naming, confidence, predictions_cfg, output_cfg = _build_configs_from_params(**flat_params)
    next_ens_results = train_and_evaluate_model(
        model=None,
        data=data,
        control=control,
        metrics=metrics_cfg,
        reporting=reporting_cfg,
        naming=naming,
        output=output_cfg,
        confidence=confidence,
        predictions=predictions_cfg,
    )

    conf_results = None
    if uncertainty_quantile:
        if target is not None:
            conf_target_kwargs = dict(target=target)
        else:
            conf_target_kwargs = dict(
                train_target=train_target[train_confident_indices] if (train_target is not None and train_confident_indices is not None) else None,
                test_target=test_target[test_confident_indices] if (test_target is not None and test_confident_indices is not None) else None,
                val_target=val_target[val_confident_indices] if (val_target is not None and val_confident_indices is not None) else None,
            )

        conf_predictive_kwargs = build_predictive_kwargs(
            train_data=(
                train_ensembled_predictions[train_confident_indices]
                if (train_ensembled_predictions is not None and train_confident_indices is not None)
                else None
            ),
            test_data=(
                test_ensembled_predictions[test_confident_indices] if (test_ensembled_predictions is not None and test_confident_indices is not None) else None
            ),
            val_data=(
                val_ensembled_predictions[val_confident_indices] if (val_ensembled_predictions is not None and val_confident_indices is not None) else None
            ),
            is_regression=is_regression,
        )

        # Report the confidence-filter coverage right in the model name so
        # log-grep immediately shows that e.g. "Conf Ensemble ... [VAL
        # COV=10%]" is computed on just 10 % of VAL rows -- previously the
        # 99.77 % accuracy number in the Conf Ensemble block was easy to
        # misread as a headline, because coverage only appeared inside the
        # calibration subsection as ``COV=XX%`` (2026-04-23 review finding).
        # Prefer VAL coverage as the headline (early-stopping + calibration
        # both key on VAL); fall back to TEST coverage then TRAIN.
        # Evaluate (label, full_preds, conf_idx, target_for_label) for each split
        # in priority order. We need the target slice as well because the
        # degenerate-class-balance check (below) operates on the filtered target,
        # not on the prediction array.
        _cov_src = None
        _conf_target = None
        for _label, _full, _conf, _full_target in (
            ("VAL", val_ensembled_predictions, val_confident_indices, val_target),
            ("TEST", test_ensembled_predictions, test_confident_indices, test_target),
            ("TRAIN", train_ensembled_predictions, train_confident_indices, train_target),
        ):
            if _full is not None and _conf is not None and len(_full) > 0:
                _cov_src = (_label, 100.0 * len(_conf) / len(_full))
                if _full_target is not None and len(_full_target) == len(_full):
                    _conf_target = _full_target[_conf]
                break

        # Degenerate-class-balance check on the filtered target. A confidence
        # filter that "keeps the rows the ensemble agrees on" tends to keep
        # almost-all-positive (or almost-all-negative) subsets on imbalanced
        # data — one prod log showed 21 negatives vs 81_815 positives in the
        # 10 % VAL slice, and the resulting ``BR=0.026 %`` looked like a
        # headline win until you noticed it was reporting on a degenerate
        # split. Marker is binary-classification only; regression has no
        # class balance to check.
        _degenerate_marker = ""
        if flag_degenerate_conf_subset and not is_regression and _conf_target is not None and len(_conf_target) > 0:
            _ct = np.asarray(_conf_target)
            if _ct.ndim == 1:
                # Count positives via boolean comparison so float / bool / int
                # targets all behave the same. Ratio is min/max regardless of
                # which class is the minority.
                _n_pos = int((_ct == 1).sum())
                _n_neg = int(_ct.shape[0] - _n_pos)
                _hi = max(_n_pos, _n_neg)
                _lo = min(_n_pos, _n_neg)
                if _hi > 0 and (_lo / _hi) < degenerate_class_ratio:
                    _degenerate_marker = "[DEGENERATE] "

        # Trailing space so the downstream concat ``f"...{ensemble_name}{_cov_tag}"``
        # doesn't slam the next token onto the closing bracket -- the 2026-04-24
        # prod log showed ``[VAL COV=10%]notext prod_jobsdetails ...`` (no space
        # before "notext"). Empty tag stays empty (no double-space when off).
        _cov_tag = f" {_degenerate_marker}[{_cov_src[0]} COV={_cov_src[1]:.0f}%] " if _cov_src else ""

        # Build config objects from flat params for confidence ensemble
        conf_flat_params = dict(
            df=None,
            drop_columns=[],
            model_name_prefix=f"Conf Ensemble {internal_ensemble_method} {ensemble_name}{_cov_tag}",
            train_idx=train_idx[train_confident_indices] if (train_idx is not None and train_confident_indices is not None) else None,
            test_idx=test_idx[test_confident_indices] if (test_idx is not None and test_confident_indices is not None) else None,
            val_idx=val_idx[val_confident_indices] if (val_idx is not None and val_confident_indices is not None) else None,
            target_label_encoder=target_label_encoder,
            compute_trainset_metrics=(
                _caller_compute_trainset_metrics
                and _has_split_predictions(conf_predictive_kwargs, "train")
            ),
            compute_valset_metrics=(
                _caller_compute_valset_metrics
                and _has_split_predictions(conf_predictive_kwargs, "val")
            ),
            compute_testset_metrics=(
                _caller_compute_testset_metrics
                and _has_split_predictions(conf_predictive_kwargs, "test")
            ),
            nbins=nbins,
            custom_ice_metric=custom_ice_metric,
            custom_rice_metric=custom_rice_metric,
            subgroups=subgroups,
            n_features=n_features,
            **conf_predictive_kwargs,
            **conf_target_kwargs,
            **kwargs_copy,
        )
        conf_data, conf_control, conf_metrics, conf_reporting, conf_naming, conf_confidence, conf_predictions, conf_output = _build_configs_from_params(
            **conf_flat_params
        )
        conf_results = train_and_evaluate_model(
            model=None,
            data=conf_data,
            control=conf_control,
            metrics=conf_metrics,
            reporting=conf_reporting,
            naming=conf_naming,
            output=conf_output,
            confidence=conf_confidence,
            predictions=conf_predictions,
        )

    return (internal_ensemble_method, next_ens_results, conf_results)


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
    if not arrays:
        for attr in ("val_probs", "test_probs", "train_probs"):
            cand = [getattr(m, attr, None) for m in members]
            if all(p is not None for p in cand):
                # DIVERSITY-LAST-COL: flatten the full per-class matrix instead of collapsing to
                # the last column. The previous "[:, -1]" reduction discarded inter-class
                # diversity signal entirely for multiclass; flattening preserves it. Binary
                # behaviour is unchanged because the two columns are linearly dependent (sum=1)
                # so concatenating them gives the same correlation magnitudes.
                arrays = []
                for p in cand:
                    arr = np.asarray(p, dtype=np.float64)
                    arrays.append(arr.ravel())
                split_used = attr
                break
    if not arrays:
        return pairs, None
    if not all(a.size == arrays[0].size and a.size >= 2 for a in arrays):
        return pairs, split_used
    # DIV-1-COL: replace O(K^2) python pair loop with a single np.corrcoef call on the stacked
    # (K, N) matrix. Rows-with-any-non-finite are masked once; constant-column members (std==0)
    # produce NaN entries in the correlation matrix which we skip. Bench in
    # `_benchmarks/bench_diversity_corr.py` -- for K>=8 numpy beats the loop ~50x; for K>50 / N>1M
    # the cupy path takes over.
    M_stack = np.vstack(arrays)  # (K, N)
    K = M_stack.shape[0]
    finite_cols = np.all(np.isfinite(M_stack), axis=0)
    if int(finite_cols.sum()) < 2:
        return pairs, split_used
    M_finite = M_stack[:, finite_cols]
    stds = M_finite.std(axis=1)
    nonconst = stds > 0
    if int(nonconst.sum()) < 2:
        return pairs, split_used
    M_use = M_finite[nonconst]
    K_use = M_use.shape[0]
    idx_use = np.flatnonzero(nonconst)
    corr_matrix = _stacked_corrcoef(M_use)
    for ii in range(K_use):
        for jj in range(ii + 1, K_use):
            corr = float(corr_matrix[ii, jj])
            if not np.isfinite(corr):
                continue
            if abs(corr) > threshold:
                i = int(idx_use[ii])
                j = int(idx_use[jj])
                m1 = member_tags[i] if i < len(member_tags) else f"member_{i}"
                m2 = member_tags[j] if j < len(member_tags) else f"member_{j}"
                pairs.append({"m1": m1, "m2": m2, "corr": corr})
    return pairs, split_used


def score_ensemble(
    models_and_predictions: Sequence,
    ensemble_name: str,
    target: pd.Series = None,
    train_idx: np.ndarray = None,
    test_idx: np.ndarray = None,
    val_idx: np.ndarray = None,
    df: pd.DataFrame = None,
    train_target: pd.Series = None,
    test_target: pd.Series = None,
    val_target: pd.Series = None,
    target_label_encoder: object = None,
    # Outlier-member-filter thresholds. The historical absolute defaults
    # (``max_mae=0.05``, ``max_std=0.06``) excluded all 6 members of a
    # uniform tree-model suite (CB / XGB / LGB x 2 weight schemas) on
    # the 2026-04-24 prod log -- turning the filter into a no-op + 36
    # noisy WARN lines per ensemble. Defaults flipped to relative
    # (``2.5xmedian``); pass non-zero ``max_mae`` / ``max_std`` to keep
    # the legacy behaviour.
    max_mae: float = 0.0,
    max_std: float = 0.0,
    max_mae_relative: float = 2.5,
    max_std_relative: float = 2.5,
    ensure_prob_limits: bool = True,
    nbins: int = 100,
    ensembling_methods=SIMPLE_ENSEMBLING_METHODS,
    uncertainty_quantile: float = 0.1,
    normalize_stds_by_mean_preds: bool = False,
    custom_ice_metric: Callable = None,
    custom_rice_metric: Callable = None,
    subgroups: dict = None,
    max_ensembling_level: int = 1,
    n_features: int = None,
    n_jobs: int = None,
    min_samples_for_parallel: int = 10_000_000,
    verbose: bool = True,
    flag_degenerate_conf_subset: bool = True,
    degenerate_class_ratio: float = 0.01,
    diversity_corr_warn_threshold: float = 0.98,
    # NO-SW / NO-GROUPS: per-row weights and group identifiers, plumbed through the quality gate,
    # diversity check, member-quality metric aggregation, and downstream weight-fit. Both default
    # to None to preserve legacy unweighted-i.i.d. semantics; ctx auto-passes when available.
    sample_weight: Optional[np.ndarray] = None,
    group_ids: Optional[np.ndarray] = None,
    rrf_k: int = 60,
    # NO-GUARD-IDENTICAL: short-circuit when every member's predictions on the gate split match
    # numerically (Pearson corr == 1.0 AND elementwise close). One arithmetic-mean ensemble is
    # returned to skip every redundant flavour. Disabled by default so legacy reports keep their
    # shape; opt in via the suite caller.
    early_exit_if_identical: bool = False,
    # GATE-DOUBLE-DIP: when True, the quality-gate source is restricted to OOF predictions; legacy
    # callers that only stamped val_/test_/train_ preds fall through to the disabled gate path.
    # C-P1-1: default flipped to True. Pre-fix the gate silently fell through to ``val_preds``
    # (the same surface early-stopping already burned) for any member without OOF, biasing the
    # gate-survivors selection. The suite caller (_phase_train_one_target.py) NEVER overrode this
    # so every default suite ran with a val-biased gate. Setting to False explicitly re-enables
    # the legacy fallback chain (oof -> val -> test -> train) when the suite has not stamped OOF.
    require_oof_for_gate: bool = True,
    # COARSE-GATE-FALLBACK: when require_oof_for_gate=True AND OOF is unavailable, the strict gate
    # skips entirely. That's the right call for FINE thresholds (2.5x median), but it lets
    # CATASTROPHIC outliers survive: 2026-05-21 prod log had an MLP with R^2=-4.75 sitting in the
    # ensemble alongside three R^2~0.99 members because no member stamped OOF. This fallback runs
    # a SECOND gate at a much higher relative threshold (5x median by default) against the
    # val/test/train fallback chain -- enough to drop the catastrophic disasters while leaving
    # honest near-median members alone. Setting to <=0 disables the coarse fallback entirely.
    coarse_gate_max_mae_relative: float = 5.0,
    coarse_gate_max_std_relative: float = 5.0,
    # VOTENRANK: build a votenrank.Leaderboard over the resulting per-flavour metrics and stamp it
    # in the returned dict under ``_leaderboard``. Defaults True for classification; regression-only
    # flavours skip rank-based methods automatically.
    build_votenrank_leaderboard: bool = True,
    # Stacking-aware gate hook. When True, runs the NNLS-weight gate from composite_stacking on the
    # ensemble's OOF predictions and persists the survivors / weights under ``_stacking_gate``. The
    # gate is observational unless the suite caller wires it into a follow-up linear stack.
    # C-P1-5: default flipped to True. The gate is observational (does not drop members) and
    # surfaces per-member NNLS-weight info operators want for audit; previously this required the
    # suite caller to opt in, so the default suite path lost the diagnostics entirely.
    enable_stacking_aware_gate: bool = True,
    stacking_gate_min_weight: float = 0.05,
    # P1-7: optional auto-drop of one member from each high-correlation pair.
    # ``None`` preserves the observational-only default (just WARNs + stamps to _diversity);
    # passing a float in (0, 1] activates auto-drop when any pair's |corr| exceeds the floor.
    # The MEMBER WITH HIGHER MEAN ABSOLUTE GATE-METRIC (mae from the gate) is dropped, so the
    # surviving member is the one closer to the median.
    auto_drop_diversity_above: Optional[float] = None,
    **kwargs,
):
    """Compares different ensembling methods for a list of models.

    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel jobs. If None, automatically determined based on
        sample count and min_samples_for_parallel. Use 1 for sequential processing.
    min_samples_for_parallel : int, default=1_000_000
        Minimum number of samples required to enable parallel processing when n_jobs is None.
    """

    res = {}
    level_models_and_predictions = models_and_predictions

    # SINGLE-MEMBER: short-circuit when only one member is supplied. There is no ensemble to score;
    # historically the caller filtered K==1 but score_ensemble itself silently iterated every flavour
    # over a 1-member tensor (the rrf/median/harm reduction is a no-op). Returning a sentinel-only
    # dict ({"_reason": "single_member"}) signals "no ensemble built" to the caller without raising,
    # AND lets finalize / metadata distinguish "single-member suite" from "ensemble failed silently"
    # (Low-9). The sentinel key starts with ``_`` so it is filtered out by the ensemble-iteration
    # logic in callers that iterate ``res.items()`` for real flavours.
    if len(level_models_and_predictions) < 2:
        if verbose and len(level_models_and_predictions) == 1:
            logger.info("[ensemble] only one member supplied; nothing to ensemble. Returning sentinel result.")
        if len(level_models_and_predictions) == 1:
            res["_reason"] = "single_member"
            res["_n_members"] = 1
        else:
            res["_reason"] = "no_members"
            res["_n_members"] = 0
        return res

    # Uniformity gate: mixing a classifier (probs available) with a regressor (probs == None)
    # in one ensemble silently miscategorises the suite. The historical dispatch only
    # inspected member[0]; member[1] could disagree with no error. Validate up front.
    if level_models_and_predictions:
        def _has_probs(m) -> bool:
            return any(getattr(m, attr, None) is not None for attr in ("val_probs", "test_probs", "train_probs"))

        _probs_flags = [_has_probs(m) for m in level_models_and_predictions]
        if len(set(_probs_flags)) > 1:
            _clf_idx = [i for i, f in enumerate(_probs_flags) if f]
            _reg_idx = [i for i, f in enumerate(_probs_flags) if not f]
            raise ValueError(
                "score_ensemble requires uniform member types: got a mix of classifier-like "
                f"(probs available, indices {_clf_idx}) and regressor-like (no probs, indices "
                f"{_reg_idx}) members. Split the suite into per-task lists before calling."
            )

    if (
        level_models_and_predictions[0].val_probs is not None
        or level_models_and_predictions[0].test_probs is not None
        or level_models_and_predictions[0].train_probs is not None
    ):
        is_regression = False
    else:
        is_regression = True
        ensure_prob_limits = False

    # RRF is a rank-fusion flavour that only makes sense on classifier
    # probabilities (where per-row ranks across the n_samples axis encode
    # "confidence ordering"). For regression there is no analogous per-sample
    # rank operation, so drop "rrf" silently from the candidate list rather
    # than fail late inside _process_single_ensemble_method.
    if is_regression and ensembling_methods:
        _pre = list(ensembling_methods)
        ensembling_methods = [m for m in ensembling_methods if m != "rrf"]
        if verbose and len(ensembling_methods) != len(_pre):
            logger.info(
                "[ensemble] target_type=REGRESSION: skipping rrf candidate (rank-fusion only meaningful on classifier probabilities)."
            )

    # Multi-level stacking requires OOF predictions on EVERY member: the level-2 (and deeper) meta-learner consumes
    # level-1 ensemble outputs as features, and if any member contributes an in-sample ``train_preds`` row instead of
    # a ``cross_val_predict`` OOF row the meta-learner sees leaked targets. Fail fast rather than silently fold the
    # leakage forward. Single-level (``max_ensembling_level == 1``) aggregation tolerates missing OOF by falling back
    # to ``train_*`` because no downstream meta-learner consumes the train slice in that case. Membership uses
    # ``isinstance(..., np.ndarray)`` for the same reason as ``_oof_or_train``: MagicMock test doubles fabricate
    # any attribute on access, so ``is None`` would never fire on a real-world stub.
    if max_ensembling_level > 1:
        _oof_attr = "oof_probs" if not is_regression else "oof_preds"
        _missing_oof = [
            i for i, m in enumerate(level_models_and_predictions)
            if not isinstance(getattr(m, _oof_attr, None), np.ndarray)
        ]
        if _missing_oof:
            raise ValueError(
                f"score_ensemble(max_ensembling_level={max_ensembling_level}) requires {_oof_attr} on every member; "
                f"members at indices {_missing_oof} are missing OOF. Re-train with oof_n_splits>=2 so cross_val_predict "
                f"OOFs are stamped on each model, or call with max_ensembling_level=1."
            )

    # Determine sample count for parallelization decision
    first_pred = level_models_and_predictions[0]
    if first_pred.val_probs is not None:
        n_samples = len(first_pred.val_probs)
    elif first_pred.val_preds is not None:
        n_samples = len(first_pred.val_preds)
    else:
        n_samples = 0

    # Determine n_jobs if not specified
    effective_n_jobs = n_jobs
    if effective_n_jobs is None:
        if n_samples >= min_samples_for_parallel and not is_jupyter_notebook():
            effective_n_jobs = min(len(ensembling_methods), cpu_count_physical())
        else:
            effective_n_jobs = 1

    # Convert pandas Series to numpy arrays before parallel section to avoid pickling issues
    train_target_arr = train_target.to_numpy() if isinstance(train_target, pd.Series) else train_target
    test_target_arr = test_target.to_numpy() if isinstance(test_target, pd.Series) else test_target
    val_target_arr = val_target.to_numpy() if isinstance(val_target, pd.Series) else val_target
    target_arr = target.to_numpy() if isinstance(target, pd.Series) else target

    # ONE-pass member quality gate before iterating ensemble flavors. The previous behaviour ran the same outlier
    # filter inside ``ensemble_probabilistic_predictions`` once per flavor x split, which on a 4-model x 5-flavor x
    # (full+conf) x 2-split layout printed the same "ens member N excluded ..." line ~20x per suite call. Compute
    # ONCE here, log the decision once, then pass only kept members to the flavor loop and disable the embedded
    # filter so no duplicate prints fire.
    #
    # Source ordering: OOF preds/probs come FIRST -- the gate's job is to drop members whose preds are outliers vs
    # the ensemble median, and val_preds are already burned for early-stopping (gating on them double-dips val).
    # OOF preds are the only honest train-side signal (cross_val_predict held-out rows). Fallback chain: oof_* ->
    # val_* -> test_* -> train_* preserves the legacy behaviour for members trained without oof_n_splits.
    _gate_source_split = None
    _gate_preds_for_check: Optional[List[np.ndarray]] = None
    # GATE-DOUBLE-DIP / GATE-NO-OOF: prefer oof_* exclusively; fall back to val/test/train only
    # when require_oof_for_gate is False. When True and any member lacks OOF we WARN and skip the
    # gate entirely (better to run all members than to gate on the same surface the early-stopper /
    # test-set selector burned). The "all members must share a split" condition stays the same --
    # mixing splits across members would compare incomparable rows.
    _candidate_attrs = (
        ("oof_preds", "oof"),
        ("oof_probs", "oof"),
    )
    if not require_oof_for_gate:
        _candidate_attrs = _candidate_attrs + (
            ("val_preds", "val"),
            ("test_preds", "test"),
            ("train_preds", "train"),
            ("val_probs", "val"),
            ("test_probs", "test"),
            ("train_probs", "train"),
        )
    for _attr, _label in _candidate_attrs:
        _candidate = [getattr(m, _attr, None) for m in level_models_and_predictions]
        # MagicMock test doubles fabricate any attribute access, so ``p is not None`` would always pass; require an
        # actual numpy array to gate the source-split selection.
        if all(isinstance(p, np.ndarray) for p in _candidate):
            _gate_preds_for_check = _candidate
            _gate_source_split = _label
            break
    # COARSE-GATE-FALLBACK: when OOF is unavailable AND require_oof_for_gate=True, the fine-grained
    # 2.5x gate is intentionally skipped to avoid double-dipping on val. But that lets catastrophic
    # outliers (R^2=-4.75 alongside R^2=0.99 members) survive into the ensemble. Run a SECOND
    # coarse-threshold pass against the val/test/train fallback chain to catch only the disasters.
    # Marked separately in logs (split=val-coarse) so it can't be confused with the strict gate.
    _coarse_gate_active = False
    if (
        require_oof_for_gate
        and _gate_preds_for_check is None
        and (coarse_gate_max_mae_relative > 0.0 or coarse_gate_max_std_relative > 0.0)
    ):
        _coarse_fallback_attrs = (
            ("val_preds", "val-coarse"),
            ("test_preds", "test-coarse"),
            ("train_preds", "train-coarse"),
            ("val_probs", "val-coarse"),
            ("test_probs", "test-coarse"),
            ("train_probs", "train-coarse"),
        )
        for _attr, _label in _coarse_fallback_attrs:
            _candidate = [getattr(m, _attr, None) for m in level_models_and_predictions]
            if all(isinstance(p, np.ndarray) for p in _candidate):
                _gate_preds_for_check = _candidate
                _gate_source_split = _label
                _coarse_gate_active = True
                # Coarse pass replaces fine thresholds for the single call below; the embedded
                # per-flavor filter has already been zeroed before this block runs (or will be
                # zeroed in the kept-survivors branch below), so this only governs the single
                # compute_member_quality_gate invocation that follows.
                max_mae = 0.0
                max_std = 0.0
                max_mae_relative = float(coarse_gate_max_mae_relative)
                max_std_relative = float(coarse_gate_max_std_relative)
                break
        if verbose:
            if _coarse_gate_active:
                logger.warning(
                    "[ensemble] OOF unavailable; running COARSE gate on %s at %.1fx median MAE / %.1fx median STD (catches catastrophic outliers only; theoretical val double-dip risk acknowledged).",
                    _gate_source_split, max_mae_relative, max_std_relative,
                )
            else:
                logger.warning(
                    "[ensemble] require_oof_for_gate=True but at least one member lacks OOF preds AND no val/test/train fallback available; skipping quality gate entirely."
                )
    elif require_oof_for_gate and _gate_preds_for_check is None and verbose:
        logger.warning(
            "[ensemble] require_oof_for_gate=True but at least one member lacks OOF preds; coarse-gate disabled (coarse_gate_max_mae_relative<=0); skipping quality gate."
        )

    # 2026-05-11 (user request): TWO tag lists:
    # 1. ``_ensemble_member_tags`` -- full (shim-stripped) class / model names for the per-member quality-gate log line (operators want to see which exact model class was excluded).
    # 2. ``_ensemble_short_tags`` -- collapsed short tags (``cb`` / ``xgb`` / ``lgb`` / ``hgb`` / non-tree class name) for the rebuilt ensemble label after the gate. Without the short-collapse, the rebuilt label reads ``[CatBoostRegressor+XGBRegressor+LGBMRegressor]`` (38 chars) instead of ``[cb+xgb+lgb]`` (12 chars) -- bloated chart titles + breaks the original short-label contract from core.py.
    from mlframe.training._format import (
        short_model_tag as _short_tag,
        strip_shim_suffix as _strip_shim,
    )

    _ensemble_member_tags: List[str] = []
    _ensemble_short_tags: List[str] = []
    for _m in level_models_and_predictions:
        _name_attr = getattr(_m, "model_name", None) or getattr(_m, "name", None)
        _model_obj = getattr(_m, "model", _m)
        if _name_attr:
            _ensemble_member_tags.append(_strip_shim(str(_name_attr)))
        else:
            _ensemble_member_tags.append(_strip_shim(type(_model_obj).__name__))
        # F2 fix (2026-05-11): short-tag ALWAYS derived from the underlying CLASS, not from ``model_name`` which carries augmentations like ``"TVT MTTR=11497.66"`` that would defeat the startswith() prefix checks (``startswith("CatBoost")`` etc.).
        _ensemble_short_tags.append(_short_tag(_model_obj))

    if _gate_preds_for_check is not None and len(_gate_preds_for_check) > 2:
        _kept_idx, _excluded, _gate_stats = compute_member_quality_gate(
            _gate_preds_for_check,
            max_mae=max_mae,
            max_std=max_std,
            max_mae_relative=max_mae_relative,
            max_std_relative=max_std_relative,
            sample_weight=sample_weight,
            group_ids=group_ids,
        )
        if verbose:
            # Per-member visual table: tag + MAE-vs-median + ✓/✗ + reason
            _per_mae = _gate_stats.get("per_member_mae", [])
            _med_mae = _gate_stats.get("median_mae", 0.0)
            _excl_idx = {i for i, _ in _excluded}
            _kept_lbls = [f"{_ensemble_member_tags[i]} (MAE={float(_per_mae[i]):.4f})" for i in _kept_idx]
            _excl_lbls = [f"{_ensemble_member_tags[i]} (MAE={float(_per_mae[i]):.4f}, >{max_mae_relative:g}x median={_med_mae:.4f})" for i in _excl_idx]
            logger.info(
                "[ensemble] member quality gate (split=%s): kept %d/%d -- %s%s",
                _gate_source_split,
                len(_kept_idx),
                len(_gate_preds_for_check),
                ", ".join(_kept_lbls) if _kept_lbls else "(none)",
                ("; excluded: " + ", ".join(_excl_lbls)) if _excl_lbls else "",
            )
            if _excluded:
                # Approximate downstream-saved-work reporting so the user
                # sees ROI of the gate.
                _est_skipped_iters = len(_excluded) * len(ensembling_methods) * 2
                logger.info(
                    "[ensemble] gate saves ~%d redundant per-flavor x per-split ensemble computations on these excluded members",
                    _est_skipped_iters,
                )
            if _gate_stats.get("filter_too_restrictive"):
                logger.warning("[ensemble] gate would have excluded ALL members; falling back to original list (filter too restrictive for this combo)")
                # Low-5: stamp the bypass into the returned res so finalize / metadata reflect that
                # the gate was bypassed -- operators reading "all members kept" can now distinguish
                # the "all members within threshold" case from the "filter too restrictive" case.
                res["_gate_bypassed"] = {
                    "reason": "filter_too_restrictive",
                    "source_split": _gate_source_split,
                    "would_have_excluded": [i for i, _ in (_excluded or [])],
                }
        if _excluded and not _gate_stats.get("filter_too_restrictive"):
            level_models_and_predictions = [level_models_and_predictions[i] for i in _kept_idx]
            # 2026-05-11: refresh ``ensemble_name`` to reflect the kept
            # members so downstream model_name_prefix / report titles
            # show [cb+xgb+lgb] (gate-survivors) instead of the original
            # [cb+xgb+lgb+linear] which advertises members that didn't
            # actually contribute to the ensemble. The caller stamped
            # the label assuming all members participate; we rebuild it
            # from the surviving tag list using the same caller-side
            # format ([cb+xgb+lgb] for <=4, [N=K] otherwise).
            try:
                # F2 fix (2026-05-11): use the SHORT tag list (cb / xgb / lgb / ...) for the rebuilt ensemble label rather than the full class names; matches the original short-label contract from core.py:5483 and keeps chart titles compact.
                _kept_tags = [_ensemble_short_tags[i] for i in _kept_idx]
                _re_label = "[" + "+".join(_kept_tags) + "]" if len(_kept_tags) <= 4 else f"[N={len(_kept_tags)}]"
                # Replace any [...] / [N=k] in ``ensemble_name`` with
                # the new label. The caller pattern is
                # ``f"{pre_pipeline}{_members_label} "`` so we look for
                # the first bracketed substring and substitute.
                import re as _re_mod

                if _re_mod.search(r"\[[^\]]+\]", ensemble_name):
                    # Callable replacement -- re.sub does NOT interpret backreferences in
                    # the return value of a callable, so any incidental ``\1`` / ``\g<...>`` /
                    # backslash inside a model tag round-trips verbatim. A plain string
                    # replacement would either crash on "invalid group reference" or silently
                    # inject backslashes into the ensemble label.
                    _label_value = _re_label
                    ensemble_name = _re_mod.sub(
                        r"\[[^\]]+\]",
                        lambda _m, _v=_label_value: _v,
                        ensemble_name,
                        count=1,
                    )
                else:
                    # REGEX-RELABEL: caller passed an unbracketed name (or already-stripped one);
                    # don't silently lose the new short label -- prepend it so log lines show the
                    # surviving members instead of advertising the original full member list.
                    ensemble_name = f"{_re_label} {ensemble_name}".rstrip() if ensemble_name else _re_label
            except Exception:  # pragma: no cover -- defensive
                pass
            # Disable the embedded per-flavor filter -- members are already
            # gated, so re-running it would just reprint the same exclusion
            # line per flavor (the noise this commit set out to eliminate).
            max_mae = 0.0
            max_std = 0.0
            max_mae_relative = 0.0
            max_std_relative = 0.0

    # Observational diversity check: pairs of kept members whose val-pred Pearson correlation exceeds the threshold are
    # surfaced via WARN + persisted to the returned dict under ``_diversity.high_correlation_pairs``. Defaults to
    # observational-only (no member removed); pass ``auto_drop_diversity_above`` to actually drop one of each pair.
    _high_corr_pairs, _div_split_used = compute_high_correlation_pairs(
        level_models_and_predictions,
        _ensemble_member_tags,
        threshold=diversity_corr_warn_threshold,
    )
    _drop_floor = auto_drop_diversity_above
    _auto_dropped: list[str] = []
    for _pair in _high_corr_pairs:
        _would_drop_msg = ""
        if _drop_floor is not None and abs(_pair["corr"]) >= float(_drop_floor):
            _would_drop_msg = f" auto-drop active (floor={_drop_floor:.3f}) -- one of {_pair['m1']}/{_pair['m2']} will be dropped"
        else:
            _would_drop_msg = " (would auto-drop one member if auto_drop_diversity_above set <= corr)"
        logger.warning(
            "[ensemble] high-correlation member pair (split=%s): %s vs %s -- Pearson corr=%.4f > threshold=%.4f.%s",
            _div_split_used,
            _pair["m1"],
            _pair["m2"],
            _pair["corr"],
            diversity_corr_warn_threshold,
            _would_drop_msg,
        )
    # Auto-drop one member from each high-corr pair when the knob is set. Pick the member with the
    # larger tag-index (later in the suite) so removal is deterministic and stable across runs.
    if _drop_floor is not None and _high_corr_pairs:
        _to_drop_tags: set[str] = set()
        for _pair in _high_corr_pairs:
            if abs(_pair["corr"]) < float(_drop_floor):
                continue
            _m1, _m2 = _pair["m1"], _pair["m2"]
            try:
                _i1 = _ensemble_member_tags.index(_m1)
                _i2 = _ensemble_member_tags.index(_m2)
            except ValueError:
                continue
            _drop_tag = _m2 if _i2 >= _i1 else _m1
            _to_drop_tags.add(_drop_tag)
        if _to_drop_tags:
            _kept_pairs = [
                (i, m) for i, m in enumerate(level_models_and_predictions)
                if _ensemble_member_tags[i] not in _to_drop_tags
            ]
            if len(_kept_pairs) >= 1:
                level_models_and_predictions = [m for _, m in _kept_pairs]
                _kept_indices = [i for i, _ in _kept_pairs]
                _ensemble_member_tags = [_ensemble_member_tags[i] for i in _kept_indices]
                _ensemble_short_tags = [_ensemble_short_tags[i] for i in _kept_indices]
                _auto_dropped = sorted(_to_drop_tags)
                logger.warning(
                    "[ensemble] auto_drop_diversity_above=%.3f dropped %d duplicate member(s): %s",
                    float(_drop_floor), len(_auto_dropped), _auto_dropped,
                )
    if _high_corr_pairs or _auto_dropped:
        res["_diversity"] = {
            "high_correlation_pairs": _high_corr_pairs,
            "threshold": diversity_corr_warn_threshold,
            "split_used": _div_split_used,
            "auto_drop_floor": float(_drop_floor) if _drop_floor is not None else None,
            "auto_dropped_members": _auto_dropped,
        }

    # I2 (2026-05-11): for regression, gate-out harmonic / geometric ensemble flavours when ANY member's predictions contain near-zero values or sign changes. Harmonic mean = N / sum(1/p) and geometric mean = exp(mean(log p)) both diverge / are undefined on signals that cross zero. Symptom seen in the prod log: ``EnsHARM ... RMSE=178.84 MaxError=55206`` and ``RMSE=1299.55 MaxError=920165`` on composite residuals which cluster around zero by construction.
    #
    # 2026-05-12 (user feedback): also gate-out QUAD (quadratic mean =
    # sqrt(mean(p^2))) on sign-changing targets. Squaring loses the sign of
    # the input by construction, so QUAD ALWAYS emits non-negative
    # predictions -- catastrophic for a target spanning both signs (the
    # prod chart for ``EnsQUAD ... TVT__monotonic_residual__Y`` showed
    # R2=-9.97 with all predictions in [0, 2000] vs true values in
    # [-2200, 500]). QUBE (cube root) is sign-preserving so it stays in.
    if is_regression and ensembling_methods:
        _has_zero_crossing = False
        _sign_sensitive_in_methods = any(m in ensembling_methods for m in ("harm", "geo", "quad"))
        if _sign_sensitive_in_methods:
            # ENS-P2-4 vectorised zero-crossing scan: flatten every member's
            # train/val/test pred arrays into one stacked float view and call
            # np.nanmin / np.any once instead of looping per (member, split).
            _flat_arrays: list[np.ndarray] = []
            for _m in level_models_and_predictions:
                for _attr in ("val_preds", "test_preds", "train_preds"):
                    _arr = getattr(_m, _attr, None)
                    if _arr is None:
                        continue
                    _arr_f = np.asarray(_arr, dtype=np.float64).ravel()
                    if _arr_f.size:
                        _flat_arrays.append(_arr_f)
            if _flat_arrays:
                _stacked = np.concatenate(_flat_arrays)
                # NaN-safe: nanmin of abs handles fully-NaN arrays gracefully.
                with np.errstate(invalid="ignore"):
                    _abs_min = float(np.nanmin(np.abs(_stacked))) if np.isfinite(_stacked).any() else np.inf
                    _has_neg = bool(np.nanmin(_stacked) < 0) if np.isfinite(_stacked).any() else False
                    _has_pos = bool(np.nanmax(_stacked) > 0) if np.isfinite(_stacked).any() else False
                if _abs_min < 1e-6 or (_has_neg and _has_pos):
                    _has_zero_crossing = True
            if _has_zero_crossing:
                _filtered_methods = [m for m in ensembling_methods if m not in ("harm", "geo", "quad")]
                if verbose and len(_filtered_methods) != len(ensembling_methods):
                    _dropped = [m for m in ensembling_methods if m not in _filtered_methods]
                    logger.info(
                        "[ensemble] gating out %s flavour(s): member "
                        "predictions contain near-zero / sign-changing "
                        "values (e.g. composite residual targets). "
                        "Harmonic / geometric diverge near zero; quadratic "
                        "loses input sign (sqrt(mean(p^2)) >= 0 always).",
                        "/".join(_dropped),
                    )
                ensembling_methods = _filtered_methods

    # NO-GUARD-IDENTICAL: if every kept member's gate-source predictions are numerically identical
    # (Pearson corr == 1.0 within atol AND elementwise close), every flavour collapses to the same
    # arithmetic-mean output. Run just one flavour (arithm) and return early when explicitly enabled.
    if early_exit_if_identical and _gate_preds_for_check is not None and len(level_models_and_predictions) > 1:
        try:
            _stack = np.vstack([np.asarray(p, dtype=np.float64).ravel() for p in _gate_preds_for_check])
            _ref = _stack[0]
            _all_close = all(np.allclose(_stack[i], _ref, atol=1e-9, rtol=1e-9) for i in range(1, _stack.shape[0]))
        except Exception:  # pragma: no cover -- defensive
            _all_close = False
        if _all_close:
            if verbose:
                logger.info("[ensemble] all members produce numerically identical predictions on split=%s; collapsing to a single 'arithm' flavour.", _gate_source_split)
            ensembling_methods = ["arithm"] if "arithm" in ensembling_methods else (ensembling_methods[:1] if ensembling_methods else [])
            # P2-7: surface the collapse into res["_diversity"] so finalize can persist it. Operators
            # reading ``ensembles_chosen[tt][tname]=='arithm'`` can then distinguish "arithm won by
            # metric" from "all members were duplicates and arithm was the only viable flavour".
            _div_block = res.setdefault("_diversity", {})
            _div_block["all_members_identical"] = True
            _div_block["collapsed_to_single_flavour"] = ensembling_methods[0] if ensembling_methods else None
            _div_block["split_used"] = _gate_source_split

    # Stacking-aware gate (composite_stacking.stacking_aware_gate). Observational by default: runs
    # NNLS over member OOF preds, persists survivors / weights on ``res["_stacking_gate"]``. The
    # caller can choose to feed the survivors into a follow-up linear stack at the suite level.
    if enable_stacking_aware_gate and _gate_preds_for_check is not None and target_arr is not None:
        try:
            from mlframe.training.composite_stacking import stacking_aware_gate as _saw_gate

            _saw_y = np.asarray(target_arr).reshape(-1)
            _saw_preds = {
                _ensemble_member_tags[i]: np.asarray(p, dtype=np.float64).ravel()
                for i, p in enumerate(_gate_preds_for_check)
                if np.asarray(p).reshape(-1).shape[0] == _saw_y.shape[0]
            }
            if _saw_preds:
                _saw_survivors, _saw_weights = _saw_gate(_saw_preds, _saw_y, min_weight=stacking_gate_min_weight)
                res["_stacking_gate"] = {
                    "survivors": list(_saw_survivors),
                    "weights": dict(_saw_weights),
                    "min_weight": float(stacking_gate_min_weight),
                }
        except Exception as _saw_err:  # pragma: no cover -- defensive
            logger.warning("[ensemble] stacking_aware_gate failed: %s", _saw_err)

    for ensembling_level in range(max_ensembling_level):

        next_level_models_and_predictions = []

        # Common parameters for all ensemble methods
        common_params = dict(
            level_models_and_predictions=level_models_and_predictions,
            is_regression=is_regression,
            ensembling_level=ensembling_level,
            ensemble_name=ensemble_name,
            target=target_arr,
            train_idx=train_idx,
            test_idx=test_idx,
            val_idx=val_idx,
            train_target=train_target_arr,
            test_target=test_target_arr,
            val_target=val_target_arr,
            target_label_encoder=target_label_encoder,
            max_mae=max_mae,
            max_std=max_std,
            max_mae_relative=max_mae_relative,
            max_std_relative=max_std_relative,
            ensure_prob_limits=ensure_prob_limits,
            nbins=nbins,
            uncertainty_quantile=uncertainty_quantile,
            normalize_stds_by_mean_preds=normalize_stds_by_mean_preds,
            custom_ice_metric=custom_ice_metric,
            custom_rice_metric=custom_rice_metric,
            subgroups=subgroups,
            n_features=n_features,
            verbose=verbose,
            kwargs=kwargs,
            flag_degenerate_conf_subset=flag_degenerate_conf_subset,
            degenerate_class_ratio=degenerate_class_ratio,
            sample_weight=sample_weight,
            rrf_k=rrf_k,
        )

        if len(ensembling_methods) > 1 and effective_n_jobs > 1:
            # loky pickles kwargs across worker boundaries; closure-captured metrics/lambdas
            # blow up in workers. Pre-check so we can fall back to sequential with a clear warning.
            try:
                import pickle

                pickle.dumps((custom_ice_metric, custom_rice_metric, kwargs))
            except (pickle.PicklingError, AttributeError, TypeError) as exc:
                logger.warning(
                    "ensembling: falling back to sequential -- one of " "custom_ice_metric / custom_rice_metric / kwargs is not picklable: %s",
                    exc,
                )
                effective_n_jobs = 1

        if len(ensembling_methods) > 1 and effective_n_jobs > 1:
            # Parallel processing -- loky + tiny max_nbytes keeps arrays in-memory (no spill) per pre-existing tuning
            results = parallel_run(
                [delayed(_process_single_ensemble_method)(ensemble_method=method, **common_params) for method in ensembling_methods],
                n_jobs=effective_n_jobs,
                backend="loky",
                max_nbytes="1K",
                verbose=0,
            )
            for internal_method, next_ens_results, conf_results in results:
                res[internal_method] = next_ens_results
                next_level_models_and_predictions.append(next_ens_results)
                if conf_results is not None:
                    res[internal_method + " conf"] = conf_results
        else:
            # Sequential processing
            for ensemble_method in ensembling_methods:
                internal_method, next_ens_results, conf_results = _process_single_ensemble_method(ensemble_method=ensemble_method, **common_params)
                res[internal_method] = next_ens_results
                next_level_models_and_predictions.append(next_ens_results)
                if conf_results is not None:
                    res[internal_method + " conf"] = conf_results

        level_models_and_predictions = next_level_models_and_predictions

    # VOTENRANK: build a Leaderboard over the per-flavour metrics for downstream rank-aggregation
    # diagnostics (Borda, Copeland, Dowdall, mean ranking). One table per (flavour x split.metric)
    # cell; regression suites still get a leaderboard but with regression-appropriate columns
    # only. The result is stamped under ``res["_leaderboard"]`` and exposes a ``to_csv`` helper
    # for the F4b main.py wiring to write to ``output_config.data_dir/<suite>.leaderboard.csv``.
    if build_votenrank_leaderboard:
        try:
            _lb_obj = _build_votenrank_leaderboard_from_results(res, is_regression=is_regression)
            if _lb_obj is not None:
                res["_leaderboard"] = _lb_obj
        except Exception as _lb_err:  # pragma: no cover -- defensive
            logger.warning("[ensemble] votenrank leaderboard build failed: %s", _lb_err)
    return res


class EnsembleLeaderboard:
    """Thin wrapper around ``votenrank.Leaderboard`` that exposes the per-flavour rank table plus a
    ``to_csv`` helper for the suite wrapper to materialise to disk. The wrapper also stores the raw
    metric table so a caller can re-rank with different methods without re-instantiating.

    REG-RRF-DROPPED: regression suites still pass through here; classification-only methods are
    discovered automatically (the source flavour names use the same internal naming convention as
    ``_process_single_ensemble_method``) and rank-fusion entries are excluded when ``is_regression``.
    """

    def __init__(self, table: "pd.DataFrame", lb: Any, is_regression: bool) -> None:
        self.table = table
        self.lb = lb
        self.is_regression = bool(is_regression)

    def rank_all(self, **kwargs):
        return self.lb.rank_all(**kwargs)

    def to_csv(self, path: str, **kwargs) -> None:
        # Persist the underlying score table; the rank-method table can be re-derived from it.
        self.table.to_csv(path, **kwargs)


def _build_votenrank_leaderboard_from_results(res: dict, *, is_regression: bool) -> Optional["EnsembleLeaderboard"]:
    """Construct an EnsembleLeaderboard from a `score_ensemble` result dict.

    Per-flavour rows are the ensemble flavour name (``"arithm"``, ``"harm"``, ...); columns are
    the metric labels harvested from each result's ``metrics`` mapping (``oof.<split>.<metric>``).
    Regression mode skips RRF / votenrank-incompatible flavours -- the rank-fusion ones have no
    rank semantic for continuous y.
    """
    rows: dict[str, dict[str, float]] = {}
    for _flavour, _result in res.items():
        if _flavour.startswith("_"):
            continue
        if is_regression and _flavour.lower().startswith("rrf"):
            continue
        _metrics = getattr(_result, "metrics", None) or (
            _result.get("metrics") if isinstance(_result, dict) else None
        )
        if not _metrics:
            continue
        _flat: dict[str, float] = {}
        for _split, _split_metrics in (_metrics or {}).items():
            if not isinstance(_split_metrics, dict):
                continue
            for _k, _v in _split_metrics.items():
                if isinstance(_v, (int, float, np.floating, np.integer)) and np.isfinite(float(_v)):
                    _flat[f"{_split}.{_k}"] = float(_v)
        if _flat:
            rows[_flavour] = _flat
    if not rows:
        return None
    table = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    # Build a votenrank.Leaderboard. Higher-is-better is the convention; classification flips for
    # error-style metrics is up to the caller (rank methods are unbiased under uniform flip).
    try:
        from mlframe.votenrank import Leaderboard

        lb = Leaderboard(table=table)
        return EnsembleLeaderboard(table=table, lb=lb, is_regression=is_regression)
    except Exception:
        return None


def compare_ensembles(
    ensembles: dict,
    sort_metric: str = "oof.1.integral_error",
    show_plot: bool = True,
    figsize: tuple = (15, 3),
) -> pd.DataFrame:
    # Default flipped from "val.*" to "oof.*": ``val`` is already burned for early-stopping (the model's last-iter
    # snapshot was chosen because it scored best on val), so re-using val to pick a flavour is selecting twice on
    # the same surface. ``oof`` is the cross_val_predict held-out signal -- never seen at fit, never used for ES.
    # Test-set sort still WARNs via logger (the obvious test-set selection bias is preserved); val.* sort now WARNs
    # via warnings.warn(UserWarning) so the message shows up even when the logger is silenced (debugger / scripts).
    import warnings as _warnings_mod
    if isinstance(sort_metric, str) and sort_metric.startswith("test."):
        logger.warning(
            "[compare_ensembles] sort_metric='%s' uses the TEST split; this re-introduces test-set "
            "selection bias. Prefer an 'oof.*' metric for ensemble selection.",
            sort_metric,
        )
    if isinstance(sort_metric, str) and sort_metric.startswith("val."):
        _warnings_mod.warn(
            f"[compare_ensembles] sort_metric='{sort_metric}' uses the VAL split; val is already burned for early "
            f"stopping, so selecting an ensemble flavour on it double-dips the same surface. Prefer 'oof.*' "
            f"(cross_val_predict held-out signal) for ensemble flavour selection.",
            UserWarning,
            stacklevel=2,
        )
    items = []
    for ens_name, ens_perf in ensembles.items():
        perf = copy.deepcopy(ens_perf.metrics)
        for set_name, set_perf in perf.items():
            if set_perf:
                for col in "feature_importances fairness_report robustness_report".split():
                    if col in set_perf:
                        del set_perf[col]
        ser = pd.json_normalize(perf).iloc[0, :]
        ser.name = ens_name
        items.append(ser)

    res = pd.DataFrame(items)
    if sort_metric in res:
        res = res.sort_values(sort_metric)

        if show_plot:
            if "test." in sort_metric:
                val_metric = sort_metric.replace("test.", "val.")
                if val_metric in res:
                    blank_metric = sort_metric.replace("test.", "")
                    ax = res.set_index(val_metric).sort_index()[sort_metric].plot(title=f"Ensembles {blank_metric}, val vs test", figsize=figsize)
                    ax.set_ylabel(sort_metric)
    return res
