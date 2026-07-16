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


def _hill_climb_single_path(
    preds: list[np.ndarray],
    y: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    maximize: bool,
    max_iterations: int,
    tol: float,
    model_order: Optional[np.ndarray] = None,
    start_idx: Optional[int] = None,
) -> dict:
    """One greedy hill-climb run; factored out so the bagged variant can call it repeatedly.

    ``model_order`` (candidate scan order at each step) and ``start_idx`` (forced seed model) are the two
    knobs a randomized restart perturbs; both default to the original deterministic behavior (natural order,
    best-single-model seed) when left unset, so this is a pure refactor for the single-call path.
    """
    n_models = len(preds)
    order = np.arange(n_models) if model_order is None else model_order

    single_scores = [float(metric_fn(y, p)) for p in preds]
    if start_idx is None:
        start_idx = int(np.argmax(single_scores) if maximize else np.argmin(single_scores))

    selected = [start_idx]
    running_sum = preds[start_idx].copy()
    current_score = single_scores[start_idx]
    history = [current_score]

    def _better(a: float, b: float) -> bool:
        """Whether score a beats score b by more than tol, respecting the maximize/minimize direction."""
        return (a > b + tol) if maximize else (a < b - tol)

    for _ in range(max_iterations - 1):
        k = len(selected)
        best_candidate_score = current_score
        best_candidate_idx: Optional[int] = None
        best_candidate_sum: Optional[np.ndarray] = None
        for j in order:
            j = int(j)
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


def hill_climb_ensemble(
    oof_preds: Sequence[np.ndarray],
    y_true: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    maximize: bool = True,
    max_iterations: int = 100,
    tol: float = 0.0,
    random_state: Optional[int] = None,
    n_bags: int = 1,
    randomize_start: bool = False,
    randomize_order: bool = False,
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
        Seed for the bagged/randomized-restart variant below. Unused (and the search stays fully
        deterministic) when ``n_bags == 1`` and neither ``randomize_start`` nor ``randomize_order`` is set —
        i.e. under every default-argument call.
    n_bags
        Opt-in bagging: when > 1, run the greedy search this many times (each with a randomized starting
        model and/or randomized candidate-scan order, per the flags below) and average the resulting weight
        vectors, then re-blend with those averaged weights. Reduces overfitting to a single greedy path's
        idiosyncrasies on a small/noisy OOF set. ``n_bags == 1`` (the default) reproduces the exact original
        single-path behavior — this parameter is a pure opt-in, the default path is untouched.
    randomize_start
        When bagging (``n_bags > 1``), draw each bag's seed model uniformly at random instead of always the
        single best model. Ignored when ``n_bags == 1``.
    randomize_order
        When bagging (``n_bags > 1``), shuffle the candidate-model scan order independently for each bag
        (breaks ties in a random rather than index-order-biased way). Ignored when ``n_bags == 1``.

    Returns
    -------
    dict
        ``selected_indices`` — list of model indices added, in order (with repeats: a repeated index means
        that model got extra effective weight); for a bagged run this is the path of the single best-scoring
        bag (kept for inspection), while ``weights``/``ensemble_pred``/``score`` reflect the bag-averaged
        blend. ``weights`` — ``(n_models,)`` array, each model's final blend weight (count of appearances /
        total appearances, averaged across bags when ``n_bags > 1``). ``ensemble_pred`` — the final blended
        ``(n_samples,)`` prediction. ``score`` — the final ensemble's ``metric_fn`` value.
        ``history`` — list of scores after each accepted step (length == number of accepted additions) of the
        representative bag (or the single run when ``n_bags == 1``). ``bag_scores`` — list of each bag's own
        final score (only present when ``n_bags > 1``).
    """
    n_models = len(oof_preds)
    if n_models == 0:
        raise ValueError("hill_climb_ensemble: oof_preds is empty")

    preds = [np.asarray(p, dtype=np.float64) for p in oof_preds]
    y = np.asarray(y_true)

    if n_bags <= 1:
        # exact original code path: no randomization, single deterministic greedy run.
        return _hill_climb_single_path(preds, y, metric_fn, maximize, max_iterations, tol)

    rng = np.random.default_rng(random_state)
    bag_results = []
    for _ in range(n_bags):
        order = rng.permutation(n_models) if randomize_order else None
        start_idx = int(rng.integers(0, n_models)) if randomize_start else None
        bag_results.append(_hill_climb_single_path(preds, y, metric_fn, maximize, max_iterations, tol, order, start_idx))

    avg_weights = np.mean([r["weights"] for r in bag_results], axis=0)
    ensemble_pred = np.zeros_like(preds[0])
    for idx, w in enumerate(avg_weights):
        ensemble_pred += w * preds[idx]
    avg_score = float(metric_fn(y, ensemble_pred))

    bag_scores = [float(r["score"]) for r in bag_results]
    best_bag = bag_results[int(np.argmax(bag_scores) if maximize else np.argmin(bag_scores))]

    return {
        "selected_indices": best_bag["selected_indices"],
        "weights": avg_weights,
        "ensemble_pred": ensemble_pred,
        "score": avg_score,
        "history": best_bag["history"],
        "bag_scores": bag_scores,
    }


__all__ = ["hill_climb_ensemble"]
