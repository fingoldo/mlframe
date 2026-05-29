"""Fold-score / shard-score aggregators shared by slice-stable ES and robust CV-selector.

Five aggregation modes:

- ``mean``                  — classical CV-selector baseline; bit-identical to ``np.mean``.
- ``mean_minus_std``        — Tim-Masters-style penalty; ``alpha * std`` added to the score in the
                              "worse" direction (so ``direction="min"`` adds, ``"max"`` subtracts).
- ``median_minus_mad``      — robust analogue using median + MAD; for K >= ~10 and heavy-tail data.
- ``t_lcb``                 — one-sided Student-t LCB / UCB: ``mean +/- t_{K-1, conf} * std/sqrt(K)``.
                              Uses ``ddof=1`` (Bessel-corrected) and an opt-in correlation
                              inflation factor for the Nadeau-Bengio dependent-fold bias.
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


def aggregate_fold_scores(
    fold_scores: Sequence[float],
    *,
    mode: AggregateMode = "mean",
    direction: Direction = "min",
    alpha: float = 1.0,
    confidence: float = 0.9,
    quantile_level: float = 0.9,
    correlation_inflation: float = 1.0,
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
        Multiplier applied to std before the t-LCB / mean-std penalty, used to compensate for
        the Nadeau-Bengio dependent-fold variance underestimate. ``1.0`` is naive (no correction).
    """
    arr = np.asarray(fold_scores, dtype=float)
    if arr.size == 0:
        raise ValueError("aggregate_fold_scores: fold_scores is empty")

    if mode == "mean":
        return float(np.mean(arr))

    if arr.size < 2 and mode != "mean":
        return float(np.mean(arr))

    sign = 1.0 if direction == "min" else -1.0

    if mode == "mean_minus_std":
        spread = float(np.std(arr, ddof=1)) * float(correlation_inflation)
        return float(np.mean(arr)) + sign * float(alpha) * spread

    if mode == "median_minus_mad":
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return med + sign * float(alpha) * mad

    if mode == "t_lcb":
        from scipy.stats import t as _t

        k = arr.size
        se = float(np.std(arr, ddof=1) / np.sqrt(k)) * float(correlation_inflation)
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
    """
    if len(points) == 0:
        return []
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"compute_pareto_frontier: expected (T, 2) array, got {arr.shape}")
    n = arr.shape[0]

    means = arr[:, 0]
    stds = arr[:, 1]
    order = np.argsort(means) if mean_direction == "min" else np.argsort(-means)
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
        q = float(risk_quantile) if direction == "min" else 1.0 - float(risk_quantile)
        risk_score = float(np.quantile(scores, q))
        if best_score is None or sign_better * (risk_score - best_score) > 0:
            best_score = risk_score
            best_iter = int(idx)
    return best_iter
