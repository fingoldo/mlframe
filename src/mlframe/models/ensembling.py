
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


# P^2-Quantile streaming sketch (Jain & Chlamtac 1985) DEFERRED.
#
# Honest ROI assessment after prototyping:
# - For mlframe ensembling (typical M=5-10 models per suite), exact
#   materialised median via `np.quantile(preds, 0.5)` is already O(1)
#   in wallclock (sort of 5-10 floats per cell is ~instant) and O(M*N*K)
#   in memory (fine up to N*K*8*M < ~500MB, covers the typical case).
# - P^2 gives O(1) memory and O(1) time per sample, useful for **big-M**
#   streams (CV with 100+ folds, online feature quantiles over huge N)
#   but NOT for the M=5-10 ensembling regime.
# - Correct vectorisation over (N, K) cells requires per-cell state
#   transitions that don't reduce to numpy broadcasts (each cell has
#   its own marker trajectory + different bucket placement per update).
#   A naive scalar-per-i vectorisation (attempted Session 4) gave ~100%
#   relative error -- the per-cell direction `d_sign` varies across cells,
#   breaking the scalar marker-shift logic.
#
# Correct implementation paths:
# - numba-jitted per-cell loop (O(1) per cell, full N*K parallel via
#   @numba.njit(parallel=True)) -- ~1h focused work + bench
# - pure-python per-cell loop (O(N*K) Python overhead per push -- slow)
# - existing `crick.TDigest` / `pytdigest` C++ binding -- external dep
#
# When the big-M case arises in prod, revisit. Leaving the Welford
# primitive (this module's `_WelfordAccumulator`) as the streaming
# aggregator for mean/std/min/max. Median in streaming mode raises
# NotImplementedError in `ensemble_probabilistic_predictions_streaming`.
#
# Tracked as: TODO(session-5+) P^2-Quantile numba-jit per-cell impl

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
    if n <= 2:
        # Filter only fires for 3+ members; ensembling 2 members
        # doesn't have a well-defined median to outlier-test against.
        return list(range(n)), [], {}

    arr = np.asarray(preds_list, dtype=np.float64)
    median_preds = np.quantile(arr, 0.5, axis=0)
    per_member_mae = np.empty(n, dtype=np.float64)
    per_member_std = np.empty(n, dtype=np.float64)
    for i, pred in enumerate(preds_list):
        diffs = np.abs(pred - median_preds)
        mae_per_col = diffs.mean(axis=0)
        std_per_col = np.sqrt(((diffs - mae_per_col) ** 2).mean(axis=0))
        per_member_mae[i] = mae_per_col.mean()
        per_member_std[i] = std_per_col.mean()

    median_mae = float(np.median(per_member_mae))
    median_std = float(np.median(per_member_std))
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
    assert ensemble_method in SIMPLE_ENSEMBLING_METHODS
    confident_indices = None

    preds = [p for p in preds if p is not None]
    if len(preds) == 0:
        return None, None, None

    # 2026-04-24: dedup memory churn. Pre-2026-04-24, this function called
    # `np.array(preds)` ~9 times across the various ensemble methods,
    # outlier-filter, and confidence paths -- each call materialised a full
    # (M, N, K) tensor, peaking RAM at ~9x the steady-state cost. On
    # multi_5 x ensembles x 2-weight_schemas (M=6, N=600, K=5) the peak hit
    # native C++ allocator's Win32 4GB ceiling and OOM'd. Materialising
    # ONCE here eliminates that churn -- full Welford-streaming refactor
    # for the big-N case (N=9M+) is tracked as TODO below.
    #
    # TODO(future): For N*K*M*8 > EnsemblingConfig.quantile_budget_bytes,
    # switch to streaming Welford accumulators (mean/std/geomean via
    # log-mean/M2) + P^2-Quantile sketch for median/quantile aggregations.
    # Estimated gain: ~5x peak-memory drop on prod-sized frames; not
    # needed for fuzz-sized data.
    _preds_arr = np.asarray(preds, dtype=np.float64)

    if len(preds) > 2:

        skipped_preds_indices = set()

        # -----------------------------------------------------------------------------------------------------------------------------------------------------
        # Disregard whole predictions deviating from the median too much
        # -----------------------------------------------------------------------------------------------------------------------------------------------------

        # compute median preds first
        median_preds = np.quantile(_preds_arr, 0.5, axis=0)

        # Per-member distance summary (vectorised over all columns at once).
        # Old code did a Python loop over columns; on 6-member x 1-column
        # ensembles that's only 6 iterations, but on multi-output cases this
        # is cleaner and ~free.
        per_member_mae = np.empty(len(preds), dtype=np.float64)
        per_member_std = np.empty(len(preds), dtype=np.float64)
        for i, pred in enumerate(preds):
            diffs = np.abs(pred - median_preds)
            mae_per_col = diffs.mean(axis=0)  # (n_cols,)
            std_per_col = np.sqrt(((diffs - mae_per_col) ** 2).mean(axis=0))
            per_member_mae[i] = mae_per_col.mean()
            per_member_std[i] = std_per_col.mean()

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
                    print(f"ens member {i} excluded due to high distance from the median: " f"mae={tot_mae:.4f}, std={tot_std:.4f} [{'; '.join(reason_parts)}]")
                skipped_preds_indices.add(i)
        if skipped_preds_indices:
            if len(skipped_preds_indices) < len(preds):
                preds = [el for i, el in enumerate(preds) if i not in skipped_preds_indices]
                if verbose:
                    print(f"Using {len(preds)} members of ensemble")
                # Members were dropped -- re-materialise the cached tensor
                # so downstream aggregations see only kept members.
                _preds_arr = np.asarray(preds, dtype=np.float64)
            else:
                if verbose:
                    print(f"ensemble_probabilistic_predictions filters too restrictive ({len(skipped_preds_indices)} vs {len(preds)}), skipping them")

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Actual ensembling
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if ensemble_method == "harm":
        # Harmonic mean: if any model predicts exactly 0, HM is defined as 0.
        # Plain ``1 / pred`` triggers RuntimeWarning ("divide by zero") and
        # produces ``inf``, which ``1/mean(...)`` then maps back to 0 -- correct
        # numerically but noisy in logs (observed 2026-04-23 prod run). Mask the
        # zeros explicitly so the common path stays warning-free.
        any_zero = (_preds_arr == 0).any(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_sum = np.sum(1.0 / _preds_arr, axis=0)
            ensembled_predictions = len(_preds_arr) / inv_sum
        if any_zero.any():
            ensembled_predictions = np.where(any_zero, 0.0, ensembled_predictions)
    elif ensemble_method == "arithm":
        ensembled_predictions = np.mean(_preds_arr, axis=0)
    elif ensemble_method == "median":
        ensembled_predictions = np.quantile(_preds_arr, 0.5, axis=0)
    elif ensemble_method == "quad":
        ensembled_predictions = np.sqrt(np.mean(_preds_arr**2, axis=0))
    elif ensemble_method == "qube":
        ensembled_predictions = np.cbrt(np.mean(_preds_arr**3, axis=0))
    elif ensemble_method == "geo":
        # Use log-sum-exp via log-mean for numerical stability on large M.
        # Floor at 1e-300 (smallest safe float64) instead of 1e-12 to
        # preserve precision for legitimately rare events from well-
        # calibrated boosted trees.
        with np.errstate(divide="ignore"):
            ensembled_predictions = np.exp(np.mean(np.log(np.clip(_preds_arr, 1e-300, None)), axis=0))

    # Replace non-finite values (NaN, inf) with arithmetic mean fallback
    non_finite_mask = ~np.isfinite(ensembled_predictions)
    if non_finite_mask.any():
        arith_mean = np.mean(_preds_arr, axis=0)
        n_replaced = np.sum(non_finite_mask)
        if verbose:
            logger.info("%s non-finite values replaced with arithmetic mean", n_replaced)
        ensembled_predictions = np.where(non_finite_mask, arith_mean, ensembled_predictions)

    if ensure_prob_limits:
        ensembled_predictions = np.clip(ensembled_predictions, 0.0, 1.0)

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
    assert ensemble_method in SIMPLE_ENSEMBLING_METHODS, f"unknown ensemble_method {ensemble_method!r}"
    if ensemble_method == "median":
        raise NotImplementedError(
            "ensemble_probabilistic_predictions_streaming: 'median' requires "
            "a cross-member quantile sketch (e.g. P^2-Quantile) not yet "
            "available. Use ensemble_probabilistic_predictions (materialised) "
            "or pick a moment-based method (arithm/harm/quad/qube/geo)."
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
    )

    if not is_regression:
        predictions = (el.train_probs for el in level_models_and_predictions)
    elif level_models_and_predictions[0].train_preds is not None:
        predictions = (el.train_preds for el in level_models_and_predictions)
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

    # 2026-05-10: ONE-pass member quality gate before iterating ensemble
    # flavors. The previous behaviour ran the same outlier filter inside
    # ``ensemble_probabilistic_predictions`` once per flavor x split,
    # which on a 4-model x 5-flavor x (full+conf) x 2-split layout
    # printed the same "ens member N excluded ..." line ~20x per suite
    # call. Compute on val_preds (or test/train fallback) ONCE here,
    # log the decision once, then pass only kept members to the flavor
    # loop and disable the embedded filter so no duplicate prints fire.
    _gate_source_split = None
    _gate_preds_for_check: Optional[List[np.ndarray]] = None
    for _attr, _label in (
        ("val_preds", "val"),
        ("test_preds", "test"),
        ("train_preds", "train"),
        ("val_probs", "val"),
        ("test_probs", "test"),
        ("train_probs", "train"),
    ):
        _candidate = [getattr(m, _attr, None) for m in level_models_and_predictions]
        if all(p is not None for p in _candidate):
            _gate_preds_for_check = _candidate
            _gate_source_split = _label
            break

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
            except Exception:  # pragma: no cover -- defensive
                pass
            # Disable the embedded per-flavor filter -- members are already
            # gated, so re-running it would just reprint the same exclusion
            # line per flavor (the noise this commit set out to eliminate).
            max_mae = 0.0
            max_std = 0.0
            max_mae_relative = 0.0
            max_std_relative = 0.0

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
            for _m in level_models_and_predictions:
                for _attr in ("val_preds", "test_preds", "train_preds"):
                    _arr = getattr(_m, _attr, None)
                    if _arr is None:
                        continue
                    _arr_f = np.asarray(_arr, dtype=np.float64)
                    if _arr_f.size == 0:
                        continue
                    _abs_min = float(np.nanmin(np.abs(_arr_f)))
                    _has_neg = bool(np.any(_arr_f < 0))
                    _has_pos = bool(np.any(_arr_f > 0))
                    if _abs_min < 1e-6 or (_has_neg and _has_pos):
                        _has_zero_crossing = True
                        break
                if _has_zero_crossing:
                    break
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
    return res


def compare_ensembles(
    ensembles: dict,
    sort_metric: str = "val.1.integral_error",
    show_plot: bool = True,
    figsize: tuple = (15, 3),
) -> pd.DataFrame:
    # Default flipped from "test.*" to "val.*" 2026-05-15: sorting by a test-set metric biases ensemble selection
    # toward the holdout (test-set selection bias). Caller may still pass an explicit "test.*" for manual override --
    # we WARN once so the override is visible in the log.
    if isinstance(sort_metric, str) and sort_metric.startswith("test."):
        logger.warning(
            "[compare_ensembles] sort_metric='%s' uses the TEST split; this re-introduces test-set "
            "selection bias. Prefer a 'val.*' metric for ensemble selection.",
            sort_metric,
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
