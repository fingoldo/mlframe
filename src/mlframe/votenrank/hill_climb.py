"""Hill-climbing ensemble selection: greedy weighted forward selection from a pool of OOF predictions.

Given a library of base-model OOF predictions, start from the single best model and repeatedly add (WITH
replacement — the same model can be re-added, which is how it gets implicit weight) whichever candidate most
improves the blended validation score; stop when no candidate helps. This "Caruana-style" greedy ensemble
selection is a concrete alternative to a fixed equal-weight or single-shot-optimized blend: bad/noisy models
in the pool are never added (their inclusion never improves the score), while a genuinely complementary
model gets added repeatedly and so ends up with proportionally higher effective weight — no separate weight-
optimization step needed.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def hill_climb_ensemble(
    oof_preds: Sequence[np.ndarray],
    y_true: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    maximize: bool = True,
    max_iterations: int = 100,
    tol: float = 0.0,
    random_state: Optional[int] = None,
) -> dict:
    """Greedily build a with-replacement ensemble that maximizes (or minimizes) ``metric_fn``.

    Parameters
    ----------
    oof_preds
        Sequence of ``(n_samples,)`` OOF prediction arrays, one per candidate base model.
    y_true
        ``(n_samples,)`` ground truth aligned to every array in ``oof_preds``.
    metric_fn
        ``metric_fn(y_true, y_pred) -> float``.
    maximize
        ``True`` when higher ``metric_fn`` is better (AUC, correlation); ``False`` for a loss (RMSE, log-loss).
    max_iterations
        Hard cap on how many models can be added (including the initial best-single-model seed).
    tol
        A candidate addition is accepted only if it improves the score by strictly more than ``tol``
        (``0.0`` accepts any improvement, however tiny; raise to require a more meaningful gain and stop
        earlier / avoid overfitting to validation noise).
    random_state
        Currently unused (the greedy search is deterministic given ``oof_preds`` order); accepted for API
        stability if a randomized tie-break / bagged variant is added later.

    Returns
    -------
    dict
        ``selected_indices`` — list of model indices added, in order (with repeats: a repeated index means
        that model got extra effective weight). ``weights`` — ``(n_models,)`` array, each model's final
        blend weight (count of appearances / total appearances). ``ensemble_pred`` — the final blended
        ``(n_samples,)`` prediction. ``score`` — the final ensemble's ``metric_fn`` value.
        ``history`` — list of scores after each accepted step (length == number of accepted additions).
    """
    del random_state  # reserved for a future randomized-tie-break variant
    n_models = len(oof_preds)
    if n_models == 0:
        raise ValueError("hill_climb_ensemble: oof_preds is empty")

    preds = [np.asarray(p, dtype=np.float64) for p in oof_preds]
    y = np.asarray(y_true)

    single_scores = [float(metric_fn(y, p)) for p in preds]
    best_start = int(np.argmax(single_scores) if maximize else np.argmin(single_scores))

    selected = [best_start]
    running_sum = preds[best_start].copy()
    current_score = single_scores[best_start]
    history = [current_score]

    def _better(a: float, b: float) -> bool:
        return (a > b + tol) if maximize else (a < b - tol)

    for _ in range(max_iterations - 1):
        k = len(selected)
        best_candidate_score = current_score
        best_candidate_idx: Optional[int] = None
        best_candidate_sum: Optional[np.ndarray] = None
        for j in range(n_models):
            trial_sum = running_sum + preds[j]
            trial_pred = trial_sum / (k + 1)
            trial_score = float(metric_fn(y, trial_pred))
            if _better(trial_score, best_candidate_score):
                best_candidate_score = trial_score
                best_candidate_idx = j
                best_candidate_sum = trial_sum
        if best_candidate_idx is None:
            break
        selected.append(best_candidate_idx)
        running_sum = best_candidate_sum
        current_score = best_candidate_score
        history.append(current_score)

    weights = np.zeros(n_models, dtype=np.float64)
    for idx in selected:
        weights[idx] += 1.0
    weights /= len(selected)

    return {
        "selected_indices": selected,
        "weights": weights,
        "ensemble_pred": running_sum / len(selected),
        "score": current_score,
        "history": history,
    }


__all__ = ["hill_climb_ensemble"]
