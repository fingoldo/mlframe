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

from typing import Callable, Union

import numpy as np


def apply_sticky_state_persistence_floor(probs: np.ndarray, active_class: np.ndarray, floor: Union[float, np.ndarray]) -> np.ndarray:
    """Clip each row's active-class probability to ``floor``, renormalizing the remaining mass.

    Parameters
    ----------
    probs
        ``(n, k)`` predicted class-probability matrix.
    active_class
        ``(n,)`` integer index (into the ``k`` classes) of the currently-active class per row. Raises
        ``ValueError`` if any value falls outside ``[0, k)``.
    floor
        Minimum probability to enforce for the active class, in ``[0, 1)`` -- raises ``ValueError`` outside
        that range (``floor >= 1`` would drive the renormalized "rest" probability mass negative). Rows
        where the active class's raw probability already exceeds ``floor`` are returned unchanged. Either a
        single scalar applied to every class uniformly, or a ``(k,)`` per-class floor vector -- different
        classes can genuinely have different persistence tendencies (e.g. a state that almost never
        spontaneously changes vs. one that flips often), so a single global floor is a compromise between
        them. Opt-in: passing a scalar reproduces the original uniform-floor behavior bit-for-bit.

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
    k = probs_src.shape[1]

    floor_arr = np.asarray(floor, dtype=np.float64)
    if np.any(floor_arr < 0.0) or np.any(floor_arr >= 1.0):
        raise ValueError(f"apply_sticky_state_persistence_floor: floor must be in [0, 1), got {floor!r}.")
    if n and (np.any(active < 0) or np.any(active >= k)):
        raise ValueError(f"apply_sticky_state_persistence_floor: active_class must index into [0, {k}), got {active_class!r}.")

    row_idx = np.arange(n)

    active_prob = probs_src[row_idx, active]

    # scalar floor stays a scalar through the arithmetic below (no per-row array materialized) so the
    # uniform-floor path is bit-identical to the pre-per-class-vector implementation.
    floor_per_row: Union[float, np.ndarray] = float(floor_arr) if floor_arr.ndim == 0 else floor_arr[active]

    needs_floor = active_prob < floor_per_row

    if not needs_floor.any():
        return probs_src

    # only pay the full-array copy when at least one row actually needs flooring -- the common case (floor
    # tuned low, or most predictions already dominant) skips it entirely; measured as ~2.2s of 11.1s cProfile
    # total (200000 rows x20 classes x200 calls) when the copy was unconditional.
    probs_arr = probs_src.copy()
    row_floor = floor_per_row if isinstance(floor_per_row, float) else floor_per_row[needs_floor]
    rest_mass = 1.0 - active_prob[needs_floor]
    target_rest_mass = 1.0 - row_floor
    scale = np.where(rest_mass > 0, target_rest_mass / rest_mass, 0.0)

    rows_to_fix = row_idx[needs_floor]
    probs_arr[rows_to_fix] *= scale[:, None]
    probs_arr[rows_to_fix, active[rows_to_fix]] = row_floor

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


def optimize_persistence_floor_per_class(
    probs: np.ndarray,
    active_class: np.ndarray,
    y_true: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_thresholds: int = 50,
    n_classes: int | None = None,
    n_passes: int = 2,
) -> dict:
    """Coordinate-wise per-class floor search: opt-in generalization of :func:`optimize_persistence_floor`
    that tunes an independent floor for each class instead of one scalar shared by all classes.

    A single global floor is necessarily a compromise when classes have genuinely different persistence
    tendencies (a state that almost never spontaneously changes needs a high floor; one that flips often is
    hurt by any floor at all) -- this sweeps each class's floor in turn, holding the others fixed at their
    current-best value, for ``n_passes`` rounds (coordinate ascent; a full ``n_thresholds**k`` grid is
    exponential in the number of classes and unnecessary since classes interact only through renormalized
    mass, not directly).

    Parameters
    ----------
    probs, active_class, y_true, metric_fn, n_thresholds
        See :func:`optimize_persistence_floor`.
    n_classes
        Number of classes ``k``; defaults to ``probs.shape[1]``.
    n_passes
        Number of coordinate-ascent sweeps over all classes. Each pass costs ``k * n_thresholds`` evaluations.

    Returns
    -------
    dict
        ``{"floor": np.ndarray of shape (k,), "score": float}`` -- the per-class floor vector maximizing
        ``metric_fn``, suitable for passing straight to :func:`apply_sticky_state_persistence_floor`.
    """
    k = probs.shape[1] if n_classes is None else n_classes
    floors = np.zeros(k, dtype=np.float64)
    candidates = np.linspace(0.0, 1.0, n_thresholds, endpoint=False)

    def _score(candidate_floors: np.ndarray) -> float:
        """Apply the given per-class persistence floors and score the resulting argmax predictions with ``metric_fn``."""
        floored = apply_sticky_state_persistence_floor(probs, active_class, candidate_floors)
        pred_classes = np.argmax(floored, axis=1)
        return float(metric_fn(y_true, pred_classes))

    best_score = _score(floors)
    for _ in range(n_passes):
        improved = False
        for c in range(k):
            best_c_value, best_c_score = floors[c], best_score
            for candidate in candidates:
                if candidate == floors[c]:
                    continue
                trial = floors.copy()
                trial[c] = candidate
                score = _score(trial)
                if score > best_c_score:
                    best_c_score = score
                    best_c_value = float(candidate)
            if best_c_value != floors[c]:
                improved = True
            floors[c] = best_c_value
            best_score = best_c_score
        if not improved:
            break

    return {"floor": floors, "score": best_score}


__all__ = ["apply_sticky_state_persistence_floor", "optimize_persistence_floor", "optimize_persistence_floor_per_class"]
