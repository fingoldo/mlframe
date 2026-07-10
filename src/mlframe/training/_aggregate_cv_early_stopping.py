"""Aggregate-CV early stopping: pick ONE global best round from the mean per-round metric across folds.

Standard early stopping picks each fold's OWN best round, then the final refit uses the average (or median)
of those per-fold best rounds. That average-of-argmaxes is noisier than it looks: each fold's individual
best-round choice is itself a noisy estimate (picked by argmax-ing a validation curve that jitters up/down
round-to-round), so per-fold best rounds can vary widely even when the TRUE optimal round is the same
across folds. Ubiquant's 2nd place team instead trained every fold to a large fixed round count, averaged
the validation metric ACROSS folds at each round, and picked the round that maximizes that AGGREGATE curve
— averaging the CURVES (which cancels per-fold noise) before taking the argmax, rather than averaging
already-noisy per-fold argmaxes.
"""
from __future__ import annotations

import numpy as np


def select_best_iteration_by_aggregate_cv(
    per_fold_metric_curves: np.ndarray,
    maximize: bool = True,
) -> dict:
    """Pick the single best round from the fold-averaged validation-metric curve.

    Parameters
    ----------
    per_fold_metric_curves
        ``(n_folds, n_rounds)`` array — the validation metric at every round, for every fold, all folds
        trained to the SAME round count (no per-fold early stopping applied beforehand).
    maximize
        ``True`` for a metric where higher is better (AUC, correlation, accuracy); ``False`` for a loss
        (log-loss, RMSE) where lower is better.

    Returns
    -------
    dict
        ``{"best_round", "aggregate_curve", "best_aggregate_metric", "per_fold_best_rounds"}``.
        ``best_round`` is the 0-indexed round maximizing (or minimizing) the fold-averaged curve —
        this is the single round to use for the FINAL refit. ``per_fold_best_rounds`` is included for
        comparison/diagnostics (each fold's own naive argmax), NOT the recommended selection.
    """
    curves = np.asarray(per_fold_metric_curves, dtype=np.float64)
    if curves.ndim != 2:
        raise ValueError(f"select_best_iteration_by_aggregate_cv: expected 2D (n_folds, n_rounds); got shape {curves.shape}")
    if curves.shape[1] == 0:
        raise ValueError("select_best_iteration_by_aggregate_cv: need at least 1 round")

    aggregate_curve = curves.mean(axis=0)
    best_round = int(np.argmax(aggregate_curve)) if maximize else int(np.argmin(aggregate_curve))
    per_fold_best_rounds = curves.argmax(axis=1) if maximize else curves.argmin(axis=1)

    return {
        "best_round": best_round,
        "aggregate_curve": aggregate_curve,
        "best_aggregate_metric": float(aggregate_curve[best_round]),
        "per_fold_best_rounds": per_fold_best_rounds.astype(np.int64),
    }


__all__ = ["select_best_iteration_by_aggregate_cv"]
