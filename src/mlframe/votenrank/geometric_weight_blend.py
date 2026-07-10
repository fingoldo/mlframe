"""Weighted-power (geometric-mean) blending via constrained optimization on fitted per-model exponents.

Source: 1st_otto-group-product-classification.md's final blend -- ``0.85 * [XGBOOST^0.65 * NN^0.35] + 0.15 *
[ET]``, a geometric mean of two model outputs with tuned exponents, then arithmetic-blended with a third
model. A geometric mean (``prod(p_i^w_i)``, equivalently ``exp(sum(w_i * log(p_i)))``) is a different
inductive bias than the arithmetic mean ``constrained_weight_blend`` already provides: it punishes any single
model being confidently WRONG far more than an arithmetic average does (a near-zero factor from one model
collapses the whole product), which often helps for probability-like outputs on log-loss/AUC tasks where
models occasionally disagree sharply.

Mirrors ``constrained_weight_blend``'s optimization machinery (SLSQP, Dirichlet-seeded multi-restart) so the
two blend modes share the same calling convention and restart-robustness guarantees; the objective and
constraint differ (log-space weighted sum instead of a raw weighted sum, exponents non-negative but NOT
required to sum to 1 -- unlike a convex arithmetic blend, geometric-mean exponents are a genuinely different
parameterization where sum-to-one isn't the natural constraint).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def geometric_weight_blend(
    oof_preds: Sequence[np.ndarray],
    y_true: np.ndarray,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    n_restarts: int = 5,
    random_state: int = 0,
    eps: float = 1e-7,
) -> dict:
    """Solve for non-negative per-model exponents minimizing ``loss_fn`` under a geometric-mean blend.

    Parameters
    ----------
    oof_preds
        Sequence of ``(n_samples,)`` OOF prediction arrays (probability-like, i.e. positive), one per
        candidate model.
    y_true
        ``(n_samples,)`` ground truth.
    loss_fn
        ``loss_fn(y_true, y_pred) -> float``, LOWER is better (e.g. log-loss).
    n_restarts
        Number of random Dirichlet-seeded starting points (SLSQP is local; multiple restarts reduce the
        chance of a poor local optimum). The best (lowest-loss) result across restarts is kept.
    random_state
        Seed for the restart starting points.
    eps
        Floor applied to predictions before taking ``log`` (keeps the objective finite near 0).

    Returns
    -------
    dict
        ``exponents`` ``(n_models,)`` (non-negative), ``ensemble_pred`` ``(n_samples,)`` (the geometric-mean
        blend, ``exp(sum(w_i * log(p_i)))``), ``loss`` (the achieved ``loss_fn`` value).
    """
    from scipy.optimize import minimize

    n_models = len(oof_preds)
    if n_models == 0:
        raise ValueError("geometric_weight_blend: oof_preds is empty")
    preds = np.stack([np.clip(np.asarray(p, dtype=np.float64), eps, None) for p in oof_preds], axis=0)  # (n_models, n_samples)
    log_preds = np.log(preds)
    y = np.asarray(y_true)

    def _blend(w: np.ndarray) -> np.ndarray:
        return np.asarray(np.exp(np.tensordot(w, log_preds, axes=(0, 0))))

    def _objective(w: np.ndarray) -> float:
        return float(loss_fn(y, _blend(w)))

    bounds = [(0.0, 3.0)] * n_models  # exponents needn't sum to 1; cap avoids runaway optimization on flat regions

    rng = np.random.default_rng(random_state)
    best_weights = None
    best_loss = float("inf")
    for i in range(n_restarts):
        w0 = np.full(n_models, 1.0 / n_models) if i == 0 else rng.uniform(0.0, 1.5, size=n_models)
        result = minimize(_objective, w0, method="SLSQP", bounds=bounds, options={"maxiter": 200})
        w = np.clip(result.x, 0.0, None)
        loss = _objective(w)
        if loss < best_loss:
            best_loss = loss
            best_weights = w

    assert best_weights is not None
    ensemble_pred = _blend(best_weights)
    return {"exponents": best_weights, "ensemble_pred": ensemble_pred, "loss": best_loss}


__all__ = ["geometric_weight_blend"]
