"""Simplex-constrained (non-negative, sum-to-one) weight blending via constrained optimization.

Distinct from ``hill_climb_ensemble`` (discrete greedy forward selection, with-replacement weights implicit
in selection counts): this solves for CONTINUOUS blend weights directly via ``scipy.optimize.minimize`` under
the simplex constraint (weights >= 0, sum to 1), minimizing a chosen CV metric across a pool of OOF
predictions -- the technique a DrivenData Pover-T-Tests team used to blend ~100 cheap model variants, a
lower-variance alternative to a trained stacker when the pool is many cheap variants of few model families.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def constrained_weight_blend(
    oof_preds: Sequence[np.ndarray],
    y_true: np.ndarray,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    n_restarts: int = 5,
    random_state: int = 0,
) -> dict:
    """Solve for non-negative, sum-to-one blend weights minimizing ``loss_fn`` over a pool of OOF predictions.

    Parameters
    ----------
    oof_preds
        Sequence of ``(n_samples,)`` OOF prediction arrays, one per candidate model variant.
    y_true
        ``(n_samples,)`` ground truth.
    loss_fn
        ``loss_fn(y_true, y_pred) -> float``, LOWER is better (e.g. log-loss, RMSE).
    n_restarts
        Number of random simplex-feasible starting points; SLSQP is local, so multiple restarts reduce the
        chance of a poor local optimum. The best (lowest-loss) result across restarts is kept.
    random_state
        Seed for the restart starting points.

    Returns
    -------
    dict
        ``weights`` ``(n_models,)`` (non-negative, sums to 1), ``ensemble_pred`` ``(n_samples,)``,
        ``loss`` (the achieved ``loss_fn`` value).
    """
    from scipy.optimize import minimize

    n_models = len(oof_preds)
    if n_models == 0:
        raise ValueError("constrained_weight_blend: oof_preds is empty")
    preds = np.stack([np.asarray(p, dtype=np.float64) for p in oof_preds], axis=0)  # (n_models, n_samples)
    y = np.asarray(y_true)

    def _objective(w: np.ndarray) -> float:
        blended = np.tensordot(w, preds, axes=(0, 0))
        return float(loss_fn(y, blended))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_models

    rng = np.random.default_rng(random_state)
    best_weights = None
    best_loss = float("inf")
    for i in range(n_restarts):
        if i == 0:
            w0 = np.full(n_models, 1.0 / n_models)
        else:
            w0 = rng.dirichlet(np.ones(n_models))
        result = minimize(_objective, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 200})
        w = np.clip(result.x, 0.0, None)
        w = w / w.sum() if w.sum() > 0 else np.full(n_models, 1.0 / n_models)
        loss = _objective(w)
        if loss < best_loss:
            best_loss = loss
            best_weights = w

    assert best_weights is not None
    ensemble_pred = np.tensordot(best_weights, preds, axes=(0, 0))
    return {"weights": best_weights, "ensemble_pred": ensemble_pred, "loss": best_loss}


__all__ = ["constrained_weight_blend"]
