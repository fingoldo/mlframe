"""Fold-score / shard-score aggregators shared by slice-stable ES and robust CV-selector.

Five aggregation modes:

- ``mean``                  — classical CV-selector baseline; bit-identical to ``np.mean``.
- ``mean_minus_std``        — Tim-Masters-style penalty; ``alpha * std`` added to the score in the
                              "worse" direction (so ``direction="min"`` adds, ``"max"`` subtracts).
- ``median_minus_mad``      — robust analogue using median + MAD; for K >= ~10 and heavy-tail data.
- ``t_lcb``                 — one-sided Student-t LCB / UCB: ``mean +/- t_{K-1, conf} * std/sqrt(K)``.
                              Uses ``ddof=1`` (Bessel-corrected) and a correlation inflation factor
                              for the Nadeau-Bengio dependent-fold bias. The factor defaults to the
                              Nadeau-Bengio ``sqrt(1 + K * test_frac/train_frac)`` std-multiplier
                              WHENEVER the caller supplies ``split_geometry`` — the naive ``1.0`` is
                              only used when neither an explicit factor nor a split geometry is given.
- ``quantile``              — non-parametric quantile penalty; auto-flips to upper quantile for
                              ``direction="min"`` and lower quantile for ``direction="max"``.

The function returns a *single* penalty-augmented score where ``direction="min"`` means smaller
is better (RMSE, log-loss): worse-shard variance pushes the returned value UP. ``direction="max"``
(AUC, accuracy) pushes it DOWN, so the caller's existing argmax / argmin works unchanged.
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np


AggregateMode = Literal["mean", "mean_minus_std", "median_minus_mad", "t_lcb", "quantile"]
Direction = Literal["min", "max"]

# Sentinel for ``correlation_inflation``: resolve the factor from ``split_geometry`` (Nadeau-Bengio) when geometry is
# supplied, else fall back to the naive ``1.0``. A bare float still overrides this and is applied verbatim.
AUTO_INFLATION: str = "auto"


def nadeau_bengio_inflation(k: int, test_frac: float, train_frac: float | None = None) -> float:
    """Nadeau-Bengio std-inflation factor for repeated / overlapping CV variance.

    The Nadeau-Bengio (2003) corrected-resampled estimator inflates the variance of the K-fold mean from the naive
    ``sigma^2 / K`` to ``(1/K + test_frac/train_frac) * sigma^2`` to compensate for the train-set overlap that makes
    the per-fold errors POSITIVELY correlated (so the naive ``std/sqrt(K)`` standard error is over-optimistic and the
    resulting CV-RMSE confidence intervals are too tight). Since :func:`aggregate_fold_scores` multiplies the *std*
    (``se = std/sqrt(K)`` for ``t_lcb``; ``std`` for ``mean_minus_std``), the std-multiplier that turns the naive
    ``sigma/sqrt(K)`` into the NB standard error is the ratio of the two standard deviations::

        factor = sqrt( (1/K + test_frac/train_frac) / (1/K) ) = sqrt( 1 + K * test_frac/train_frac )

    For standard (non-overlapping) K-fold, ``test_frac = 1/K`` and ``train_frac = (K-1)/K``, so the factor reduces to
    ``sqrt(1 + K/(K-1))`` (e.g. K=5 -> ~1.5; K=10 -> ~1.45; K=3 -> ~1.58) -- matching the ~1.5 default historically
    hardcoded for ``SliceStableESConfig.correlation_inflation``.

    Parameters
    ----------
    k
        Number of folds / shards / resamples (``K``). Must be ``>= 2``.
    test_frac
        Fraction of rows held out per fold (``n_test / n_total``). For K-fold this is ``1/K``; for a single train/test
        resample repeated ``K`` times it is the actual held-out fraction.
    train_frac
        Fraction of rows trained on per fold (``n_train / n_total``). Defaults to ``1 - test_frac`` (the standard
        disjoint train/test split). Pass it explicitly only for overlapping schemes where train+test != whole.
    """
    if k < 2:
        return 1.0
    tf = float(test_frac)
    trf = float(train_frac) if train_frac is not None else (1.0 - tf)
    if trf <= 0.0 or tf <= 0.0:
        return 1.0
    return float(np.sqrt(1.0 + float(k) * tf / trf))


def _resolve_inflation(
    correlation_inflation: float | str,
    split_geometry: tuple[int, float] | tuple[int, float, float] | None,
) -> float:
    """Resolve the effective std-inflation factor.

    An explicit numeric ``correlation_inflation`` is returned verbatim (caller knows best). The ``AUTO_INFLATION``
    sentinel resolves to the Nadeau-Bengio factor when ``split_geometry`` is supplied, else to the naive ``1.0``.
    """
    if not isinstance(correlation_inflation, str):
        return float(correlation_inflation)
    if correlation_inflation != AUTO_INFLATION:
        raise ValueError(
            f"aggregate_fold_scores: correlation_inflation must be a float or {AUTO_INFLATION!r}, "
            f"got {correlation_inflation!r}"
        )
    if split_geometry is None:
        return 1.0
    k = int(split_geometry[0])
    test_frac = float(split_geometry[1])
    train_frac = float(split_geometry[2]) if len(split_geometry) >= 3 else None
    return nadeau_bengio_inflation(k, test_frac, train_frac)


def aggregate_fold_scores(
    fold_scores: Sequence[float],
    *,
    mode: AggregateMode = "mean",
    direction: Direction = "min",
    alpha: float = 1.0,
    confidence: float = 0.9,
    quantile_level: float = 0.9,
    correlation_inflation: float | str = AUTO_INFLATION,
    split_geometry: tuple[int, float] | tuple[int, float, float] | None = None,
) -> float:
    """Aggregate per-fold (or per-shard) scores into a single penalty-augmented value.

    Parameters
    ----------
    fold_scores
        Per-fold / per-shard scores. Must have at least 2 entries for any non-``mean`` mode
        (std / quantile / t-LCB require dispersion).
    mode
        See module docstring.
    direction
        ``"min"`` for loss-like metrics, ``"max"`` for score-like. Determines sign of penalty
        and which tail of the quantile we read for ``mode="quantile"``.
    alpha
        Penalty coefficient for ``mean_minus_std`` and ``median_minus_mad``.
    confidence
        One-sided Student-t confidence for ``mode="t_lcb"`` (e.g. 0.9 = 90% one-sided LCB / UCB).
    quantile_level
        Quantile read for ``mode="quantile"``. For ``direction="min"`` this is the UPPER
        quantile (worst-case shard); for ``direction="max"`` we auto-flip to ``1 - q``.
    correlation_inflation
        Multiplier applied to std before the t-LCB / mean-std penalty, used to compensate for the Nadeau-Bengio
        dependent-fold variance underestimate (repeated / overlapping CV folds share training rows, so per-fold errors
        are positively correlated and the naive ``std/sqrt(K)`` SE is over-optimistic). An explicit float is applied
        verbatim. The default :data:`AUTO_INFLATION` sentinel resolves to the Nadeau-Bengio
        ``sqrt(1 + K * test_frac/train_frac)`` factor when ``split_geometry`` is supplied, and to ``1.0`` (no
        correction) otherwise -- so callers that pass their split geometry get an honest (un-over-confident) interval
        by default, while callers that pass neither stay bit-identical to the naive aggregator.
    split_geometry
        ``(k, test_frac)`` or ``(k, test_frac, train_frac)`` describing the CV split. Used only to derive the default
        Nadeau-Bengio inflation factor when ``correlation_inflation`` is left at :data:`AUTO_INFLATION`. ``test_frac``
        is the per-fold held-out fraction (``1/K`` for K-fold); ``train_frac`` defaults to ``1 - test_frac``.
    """
    inflation = _resolve_inflation(correlation_inflation, split_geometry)
    arr = np.asarray(fold_scores, dtype=float)
    if arr.size == 0:
        raise ValueError("aggregate_fold_scores: fold_scores is empty")

    if mode == "mean":
        return float(np.mean(arr))

    if arr.size < 2 and mode != "mean":
        return float(np.mean(arr))

    sign = 1.0 if direction == "min" else -1.0

    if mode == "mean_minus_std":
        spread = float(np.std(arr, ddof=1)) * inflation
        return float(np.mean(arr)) + sign * float(alpha) * spread

    if mode == "median_minus_mad":
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return med + sign * float(alpha) * mad

    if mode == "t_lcb":
        from scipy.stats import t as _t

        k = arr.size
        se = float(np.std(arr, ddof=1) / np.sqrt(k)) * inflation
        tq = float(_t.ppf(float(confidence), df=k - 1))
        return float(np.mean(arr)) + sign * tq * se

    if mode == "quantile":
        q = float(quantile_level) if direction == "min" else 1.0 - float(quantile_level)
        return float(np.quantile(arr, q))

    raise ValueError(f"aggregate_fold_scores: unknown mode={mode!r}")


def compute_pareto_frontier(
    points: Sequence[tuple[float, float]],
    *,
    mean_direction: Direction = "min",
) -> list[int]:
    """Indices of non-dominated (Pareto-optimal) ``(mean, std)`` points.

    A point ``i`` dominates ``j`` iff its mean is at least as good (per ``mean_direction``)
    AND its std is at least as low, with at least one strict. The frontier is the set of
    non-dominated indices, returned in input order.

    O(T log T) via mean-sort + running best-std scan. For ``mean_direction="min"`` we sort
    ascending by mean and keep points whose std improves; for ``"max"`` we sort descending.
    Mean ties are broken by ascending std so the lower-std point is scanned first and the
    higher-std equal-mean point is correctly rejected as dominated.
    """
    if len(points) == 0:
        return []
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"compute_pareto_frontier: expected (T, 2) array, got {arr.shape}")
    n = arr.shape[0]

    means = arr[:, 0]
    stds = arr[:, 1]
    order = np.lexsort((stds, means)) if mean_direction == "min" else np.lexsort((stds, -means))
    best_std = np.inf
    keep: list[int] = []
    for idx in order:
        s = float(stds[idx])
        if s < best_std:
            best_std = s
            keep.append(int(idx))
    keep.sort()
    return keep


def select_from_pareto(
    frontier_indices: Sequence[int],
    iter_means: Sequence[float],
    iter_stds: Sequence[float],
    iter_shard_scores: Sequence[Sequence[float]],
    *,
    risk_quantile: float = 0.9,
    direction: Direction = "min",
) -> int:
    """Pick a single ``best_iter`` from the Pareto frontier by per-iteration risk quantile.

    For each iteration on the frontier, compute the non-parametric risk quantile across its
    K shard scores (upper quantile for ``direction="min"``, lower for ``"max"``), then return
    the global iteration index with the best risk-adjusted score. Smaller ``risk_quantile`` is
    aggressive (close to mean); ``0.9-0.95`` is conservative.
    """
    if not frontier_indices:
        raise ValueError("select_from_pareto: empty frontier")
    best_score: float | None = None
    best_iter: int = int(frontier_indices[0])
    sign_better = -1.0 if direction == "min" else 1.0
    for idx in frontier_indices:
        scores = np.asarray(iter_shard_scores[idx], dtype=float)
        # An iteration with no shard scores has no defined risk quantile (np.quantile([]) warns + returns NaN); skip it rather than poison the comparison.
        if scores.size == 0:
            continue
        q = float(risk_quantile) if direction == "min" else 1.0 - float(risk_quantile)
        risk_score = float(np.quantile(scores, q))
        if best_score is None or sign_better * (risk_score - best_score) > 0:
            best_score = risk_score
            best_iter = int(idx)
    return best_iter
