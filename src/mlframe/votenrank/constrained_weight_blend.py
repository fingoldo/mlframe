"""Simplex-constrained (non-negative, sum-to-one) weight blending via constrained optimization.

Distinct from ``hill_climb_ensemble`` (discrete greedy forward selection, with-replacement weights implicit
in selection counts): this solves for CONTINUOUS blend weights directly via ``scipy.optimize.minimize`` under
the simplex constraint (weights >= 0, sum to 1), minimizing a chosen CV metric across a pool of OOF
predictions -- the technique a DrivenData Pover-T-Tests team used to blend ~100 cheap model variants, a
lower-variance alternative to a trained stacker when the pool is many cheap variants of few model families.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def _solve_simplex_weights(
    preds: np.ndarray,
    y: np.ndarray,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    n_restarts: int,
    random_state: int,
) -> tuple[np.ndarray, float]:
    """Core SLSQP multi-restart simplex solve, shared by the dense path and the sparse-subset refit."""
    from scipy.optimize import minimize

    n_models = preds.shape[0]

    def _objective(w: np.ndarray) -> float:
        """Loss of the weighted blend of preds under candidate weights w."""
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
    return best_weights, best_loss


def constrained_weight_blend(
    oof_preds: Sequence[np.ndarray],
    y_true: np.ndarray,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    n_restarts: int = 5,
    random_state: int = 0,
    max_nonzero_weights: Optional[int] = None,
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
    max_nonzero_weights
        Opt-in sparsity cap. When set (and below ``n_models``), a plain L1 penalty is useless here because
        every feasible simplex point already has L1 norm 1 -- instead this does dense-solve-then-prune-then-refit:
        solve the unconstrained-sparsity (dense) simplex problem first, keep the ``max_nonzero_weights`` models
        with the largest dense weight, then re-solve the same simplex problem restricted to just that subset.
        Discarding near-zero-weight models rarely hurts quality (that's exactly what near-zero weight means),
        and the subset refit recovers most of what the drop cost. Leave ``None`` for the original dense behavior
        (bit-identical to omitting this parameter).

    Returns
    -------
    dict
        ``weights`` ``(n_models,)`` (non-negative, sums to 1; zero for models pruned by ``max_nonzero_weights``),
        ``ensemble_pred`` ``(n_samples,)``, ``loss`` (the achieved ``loss_fn`` value),
        ``selected_indices`` (indices of models with nonzero weight), ``n_nonzero`` (count of nonzero weights).
    """
    n_models = len(oof_preds)
    if n_models == 0:
        raise ValueError("constrained_weight_blend: oof_preds is empty")
    preds = np.stack([np.asarray(p, dtype=np.float64) for p in oof_preds], axis=0)  # (n_models, n_samples)
    y = np.asarray(y_true)

    best_weights, best_loss = _solve_simplex_weights(preds, y, loss_fn, n_restarts, random_state)

    if max_nonzero_weights is not None and 0 < max_nonzero_weights < n_models:
        top_idx = np.argsort(best_weights)[::-1][:max_nonzero_weights]
        top_idx = np.sort(top_idx)
        sub_weights, sub_loss = _solve_simplex_weights(preds[top_idx], y, loss_fn, n_restarts, random_state)
        best_weights = np.zeros(n_models, dtype=np.float64)
        best_weights[top_idx] = sub_weights
        best_loss = sub_loss

    selected_indices = np.flatnonzero(best_weights > 0.0)
    ensemble_pred = np.tensordot(best_weights, preds, axes=(0, 0))
    return {
        "weights": best_weights,
        "ensemble_pred": ensemble_pred,
        "loss": best_loss,
        "selected_indices": selected_indices,
        "n_nonzero": int(selected_indices.size),
    }


__all__ = ["constrained_weight_blend"]
