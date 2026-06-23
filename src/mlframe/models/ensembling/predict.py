"""Probabilistic ensemble prediction entry points for ``mlframe.models.ensembling``.

Split out of ``ensembling.py`` to keep the parent below the 1k-line monolith
threshold. ``ensemble_probabilistic_predictions`` and its streaming variant
are re-exported from the parent so historical
``from mlframe.models.ensembling import ensemble_probabilistic_predictions``
imports continue to resolve.

The two functions are the canonical blend entry points -- they combine N
member predictions into one ensemble prediction via the configured method
(arithm / harm / median / quad / qube / geo / rrf). The streaming variant
threads through ``_WelfordAccumulator`` so we never materialise the full
(K, N, classes) array in memory for large ensembles.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

# Shared helpers live in the leaf so siblings can import them without
# re-entering the parent (which would close an import cycle).
from .base import (
    RANK_FUSION_METHODS,
    SIMPLE_ENSEMBLING_METHODS,
    _WelfordAccumulator,
    _per_member_mae_std,
    combine_probs,
    rrf_ensemble,
)

# Share the parent's logger name so caplog filters in the test suite
# pick up records emitted from this sibling.
logger = logging.getLogger("mlframe.models.ensembling")


# CPX17: the outlier-member gate (cross-member median + per-member MAE/STD + threshold decision) is flavour-INVARIANT --
# it depends only on the member-prediction array + the four thresholds, NOT on ``ensemble_method``. ``score_ensemble`` fans
# this function over n_flavours per split, so the same gate was recomputed n_flavours times (median+MAE measured at ~55% of
# the per-call wall on a 50k x 16 split). Memoise the gate per process keyed on the member-array identities + thresholds so
# the second..n-th flavour for a given split reuses the first flavour's decision. The cache holds the member arrays alive,
# so ``id()`` cannot be reused for a different array while the entry is live. Bounded to the last few member sets (3 splits
# x a couple of in-flight ensembles); cleared on demand for tests.
_GATE_CACHE_MAXSIZE = 16
_gate_cache: "dict[tuple, tuple]" = {}
_gate_cache_order: list = []


def _clear_gate_cache() -> None:
    _gate_cache.clear()
    _gate_cache_order.clear()


def _compute_outlier_gate(
    preds: list,
    preds_arr: np.ndarray,
    max_mae: float,
    max_std: float,
    max_mae_relative: float,
    max_std_relative: float,
) -> tuple:
    """Flavour-invariant outlier-member gate. Returns (skipped_indices: frozenset, median_mae, median_std, rel_mae_threshold,
    rel_std_threshold, per_member_mae, per_member_std). Memoised on the member-array identities + thresholds (CPX17)."""
    key = (
        tuple(id(p) for p in preds),
        float(max_mae), float(max_std), float(max_mae_relative), float(max_std_relative),
    )
    cached = _gate_cache.get(key)
    if cached is not None:
        # Retained refs (cached[0]) keep the arrays alive so the id()-key cannot alias a freed-then-reused array.
        return cached[1]

    # Cross-member median used as outlier-filter anchor: median along the MEMBER axis (axis=0 of (M, N, K)). ``sample_weight``
    # is a per-ROW vector so it is meaningless on this member-axis reduction (the member axis is uniformly weighted by
    # construction) -- intentionally unweighted here. ``np.median`` (dedicated C reduction) over ``np.quantile(q=0.5)`` --
    # bit-identical for the unweighted member-axis median, ~1.4x faster on the ensemble anchor shape.
    median_preds = np.median(preds_arr, axis=0)
    # Vectorised per-member MAE/STD over (K, N, ...); dispatched to a numba kernel for big inputs via _per_member_mae_std.
    per_member_mae, per_member_std = _per_member_mae_std(preds_arr, median_preds)

    # Relative thresholds resolved against the MEDIAN across members (robust to a single outlier; mean would let one bad
    # member drag the threshold up and shield itself).
    median_mae = float(np.median(per_member_mae))
    median_std = float(np.median(per_member_std))
    rel_mae_threshold = max_mae_relative * median_mae if max_mae_relative > 0 else 0.0
    rel_std_threshold = max_std_relative * median_std if max_std_relative > 0 else 0.0

    skipped: set = set()
    for i in range(len(preds)):
        tot_mae = float(per_member_mae[i])
        tot_std = float(per_member_std[i])
        abs_violation = (max_mae > 0 and tot_mae > max_mae) or (max_std > 0 and tot_std > max_std)
        rel_violation = (rel_mae_threshold > 0 and tot_mae > rel_mae_threshold) or (rel_std_threshold > 0 and tot_std > rel_std_threshold)
        if abs_violation or rel_violation:
            skipped.add(i)

    result = (
        frozenset(skipped),
        median_mae, median_std, rel_mae_threshold, rel_std_threshold,
        per_member_mae, per_member_std,
    )
    _gate_cache[key] = (preds, result)  # hold ``preds`` so the id()-key stays valid for the entry's lifetime.
    _gate_cache_order.append(key)
    if len(_gate_cache_order) > _GATE_CACHE_MAXSIZE:
        _gate_cache.pop(_gate_cache_order.pop(0), None)
    return result


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
    precomputed_weights: Optional[np.ndarray] = None,
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

    # Filter out None preds while keeping ``precomputed_weights`` aligned with the surviving
    # member ordering. Without this re-alignment a None member would push every later weight
    # one position upstream of its actual prediction.
    _orig_preds = list(preds)
    _keep_mask = [p is not None for p in _orig_preds]
    preds = [p for p, k in zip(_orig_preds, _keep_mask) if k]
    if precomputed_weights is not None and len(_orig_preds) != len(preds):
        precomputed_weights = np.asarray(precomputed_weights, dtype=np.float64).reshape(-1)
        if precomputed_weights.shape[0] == len(_orig_preds):
            precomputed_weights = precomputed_weights[np.asarray(_keep_mask, dtype=bool)]
    if len(preds) == 0:
        raise ValueError(
            "ensemble_probabilistic_predictions: no non-None member predictions to ensemble "
            f"(received {len(_orig_preds)} member(s), all None). Provide at least one non-None prediction."
        )

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

        # Cross-member median used as outlier-filter anchor: take the median along the MEMBER axis
        # (axis=0 of (M, N, K)). ``sample_weight`` is a per-ROW vector (length N), not a per-member
        # vector, so applying it directly to ``np.quantile(axis=0)`` is meaningless -- the member
        # axis is uniformly weighted by construction. Pre-fix the code claimed "weighted via
        # ``inverted_cdf``" but never passed a ``weights=`` kwarg in either try/except branch, so
        # the supposed weighting silently degraded to unweighted; the comment was a no-op promise.
        # If the gate ever needs per-row weighting it must aggregate along axis=1 separately, not
        # via this anchor. Document the choice here so downstream readers don't re-add the broken
        # weighted-quantile call. The downstream per-member MAE in ``_per_member_mae_std`` still
        # weights rows when sample_weight is propagated via ``_ensembling_quality_gate``.
        # CPX17: median + per-member MAE/STD + threshold decision are flavour-invariant; computed once per member-set and
        # memoised so the n_flavours fan-out in ``score_ensemble`` reuses one gate per split instead of recomputing it.
        (
            _gate_skipped,
            median_mae, median_std, rel_mae_threshold, rel_std_threshold,
            per_member_mae, per_member_std,
        ) = _compute_outlier_gate(preds, _preds_arr, max_mae, max_std, max_mae_relative, max_std_relative)
        skipped_preds_indices = set(_gate_skipped)

        if verbose and skipped_preds_indices:
            for i in sorted(skipped_preds_indices):
                tot_mae = float(per_member_mae[i])
                tot_std = float(per_member_std[i])
                abs_violation = (max_mae > 0 and tot_mae > max_mae) or (max_std > 0 and tot_std > max_std)
                rel_violation = (rel_mae_threshold > 0 and tot_mae > rel_mae_threshold) or (rel_std_threshold > 0 and tot_std > rel_std_threshold)
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
        if skipped_preds_indices:
            if len(skipped_preds_indices) < len(preds):
                _kept_mask = np.array([i not in skipped_preds_indices for i in range(len(preds))], dtype=bool)
                preds = [el for i, el in enumerate(preds) if _kept_mask[i]]
                if precomputed_weights is not None:
                    _pw = np.asarray(precomputed_weights, dtype=np.float64).reshape(-1)
                    if _pw.shape[0] == _kept_mask.shape[0]:
                        precomputed_weights = _pw[_kept_mask]
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
        precomputed_weights=precomputed_weights,
    )

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Confidence estimates
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if uncertainty_quantile:

        # ddof=1 (sample std across members) to match the streaming path's _WelfordAccumulator.result(), which
        # reports std = sqrt(M2 / max(n-1, 1)). Both paths now report the same Bessel-corrected member spread.
        # With a single member (M=1) ddof=1 would divide by zero; mirror Welford's max(n-1,1) by reporting 0.
        std_preds = np.std(_preds_arr, axis=0, ddof=1) if _preds_arr.shape[0] > 1 else np.zeros(_preds_arr.shape[1:], dtype=np.float64)
        if normalize_stds_by_mean_preds:
            mean_preds = np.mean(_preds_arr, axis=0)
            # A class whose mean prediction is ~0 would yield inf/nan from the
            # division and poison the quantile threshold + confident-index
            # selection for every row; treat its relative spread as 0 instead.
            rel_std = np.where(np.abs(mean_preds) > 1e-12, std_preds / np.where(np.abs(mean_preds) > 1e-12, mean_preds, 1.0), 0.0)
            uncertainty = rel_std.mean(axis=1)
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
    ensemble_method: str = "harm",
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

    _n_received = len(preds)
    preds = [p for p in preds if p is not None]
    if len(preds) == 0:
        raise ValueError(
            "ensemble_probabilistic_predictions_streaming: no non-None member predictions to ensemble "
            f"(received {_n_received} member(s), all None). Provide at least one non-None prediction."
        )
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


