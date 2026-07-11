"""``apply_sticky_state_persistence_floor``: enforce a minimum probability floor on the currently-active class.

Source: dd_2nd_nasa-airport-config.md -- "Minimum Configuration Support ... a learned parameter enforcing a
minimum predicted-probability floor for the currently active configuration ... 'one of the most important
aspects of our final submission.'" For "state persists unless there's strong evidence of change" multiclass
sequence tasks (airport configuration, regime/status flags), a per-step classifier can flicker between classes
on noisy borderline probabilities even when the true state hasn't changed; flooring the active class's
probability (renormalizing the rest) biases the decision toward persistence, only flipping when the model's
evidence against the current state clearly exceeds the floor.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def apply_sticky_state_persistence_floor(probs: np.ndarray, active_class: np.ndarray, floor: float) -> np.ndarray:
    """Clip each row's active-class probability to ``floor``, renormalizing the remaining mass.

    Parameters
    ----------
    probs
        ``(n, k)`` predicted class-probability matrix.
    active_class
        ``(n,)`` integer index (into the ``k`` classes) of the currently-active class per row.
    floor
        Minimum probability to enforce for the active class, in ``[0, 1)``. Rows where the active class's
        raw probability already exceeds ``floor`` are returned unchanged.

    Returns
    -------
    np.ndarray
        ``(n, k)`` probability matrix, each row still summing to 1. Returns the input array UNCOPIED when no
        row needs flooring (copy-on-write) -- callers that mutate the result in place should copy first if
        they also hold a reference to the original ``probs`` array.
    """
    probs_src = np.asarray(probs, dtype=np.float64)
    active = np.asarray(active_class)
    n = probs_src.shape[0]
    row_idx = np.arange(n)

    active_prob = probs_src[row_idx, active]
    needs_floor = active_prob < floor

    if not needs_floor.any():
        return probs_src

    # only pay the full-array copy when at least one row actually needs flooring -- the common case (floor
    # tuned low, or most predictions already dominant) skips it entirely; measured as ~2.2s of 11.1s cProfile
    # total (200000 rows x20 classes x200 calls) when the copy was unconditional.
    probs_arr = probs_src.copy()
    rest_mass = 1.0 - active_prob[needs_floor]
    target_rest_mass = 1.0 - floor
    scale = np.where(rest_mass > 0, target_rest_mass / rest_mass, 0.0)

    rows_to_fix = row_idx[needs_floor]
    probs_arr[rows_to_fix] *= scale[:, None]
    probs_arr[rows_to_fix, active[rows_to_fix]] = floor

    return np.asarray(probs_arr)


def optimize_persistence_floor(
    probs: np.ndarray,
    active_class: np.ndarray,
    y_true: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_thresholds: int = 50,
) -> dict:
    """Sweep candidate floor values in ``[0, 1)`` and return the one maximizing ``metric_fn`` on the argmax
    of the floored probabilities -- the same sweep-and-pick-argmax shape as
    :func:`mlframe.calibration.threshold_optimizer.optimize_decision_threshold`, specialized to a full
    probability-matrix transform rather than a scalar-per-row binary threshold.

    Parameters
    ----------
    probs, active_class
        See :func:`apply_sticky_state_persistence_floor`.
    y_true
        ``(n,)`` true class indices.
    metric_fn
        ``metric_fn(y_true, y_pred_classes) -> float``, HIGHER is better (e.g. accuracy).
    n_thresholds
        Number of candidate floor values swept over ``[0, 1)``.

    Returns
    -------
    dict
        Same shape as :func:`optimize_decision_threshold`'s return value (``{"threshold": ..., "score": ...}``),
        with ``"threshold"`` being the optimal floor value.
    """

    # optimize_decision_threshold's binary-threshold sweep expects a scalar score per row, not a full
    # probability-matrix transform, so it isn't directly reusable here -- sweep the floor value directly.
    best_floor, best_score = 0.0, -np.inf
    for floor in np.linspace(0.0, 1.0, n_thresholds, endpoint=False):
        floored = apply_sticky_state_persistence_floor(probs, active_class, floor)
        pred_classes = np.argmax(floored, axis=1)
        score = metric_fn(y_true, pred_classes)
        if score > best_score:
            best_score = score
            best_floor = float(floor)

    return {"threshold": best_floor, "score": best_score}


__all__ = ["apply_sticky_state_persistence_floor", "optimize_persistence_floor"]
