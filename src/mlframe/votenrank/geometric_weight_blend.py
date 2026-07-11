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
    alpha: float | None = None,
    fit_alpha: bool = False,
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
    alpha
        Opt-in hybrid mode. When set (and ``fit_alpha`` is False), the returned ``ensemble_pred`` is the
        convex combination ``alpha * geometric_blend + (1 - alpha) * arithmetic_blend`` where both blends
        share the same fitted exponents (the arithmetic side uses them L1-normalized as weights). A pure
        geometric blend (``alpha=1``, the default when this param is omitted) can catastrophically zero out
        an otherwise-good row's prediction whenever any single model emits a near-zero probability for that
        row, since one factor near 0 collapses the whole product; blending in the arithmetic-mean term
        guards against that failure mode while ``alpha`` close to 1 keeps most of the geometric mean's
        sharper punishment of confidently-wrong models. Ignored (treated as unset) when ``fit_alpha=True``.
    fit_alpha
        When True, fit ``alpha`` in ``[0, 1]`` via a bounded SLSQP minimization of ``loss_fn`` on the hybrid
        blend above (using the already-fitted exponents), instead of taking a caller-supplied fixed value.

    Returns
    -------
    dict
        ``exponents`` ``(n_models,)`` (non-negative), ``ensemble_pred`` ``(n_samples,)`` (the geometric-mean
        blend, ``exp(sum(w_i * log(p_i)))``, or the hybrid blend when ``alpha``/``fit_alpha`` is used),
        ``loss`` (the achieved ``loss_fn`` value on ``ensemble_pred``). When ``alpha`` or ``fit_alpha`` is
        used, also includes ``alpha`` (the value used, fixed or fitted) and ``geometric_pred``/
        ``arithmetic_pred`` (the two pre-blend components, for inspection).
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

    if alpha is None and not fit_alpha:
        return {"exponents": best_weights, "ensemble_pred": ensemble_pred, "loss": best_loss}

    weight_sum = float(best_weights.sum())
    arith_weights = best_weights / weight_sum if weight_sum > 0 else np.full(n_models, 1.0 / n_models)
    geometric_pred = ensemble_pred
    arithmetic_pred = np.asarray(np.tensordot(arith_weights, preds, axes=(0, 0)))

    def _hybrid(a: float) -> np.ndarray:
        return a * geometric_pred + (1.0 - a) * arithmetic_pred

    def _alpha_objective(a: np.ndarray) -> float:
        return float(loss_fn(y, _hybrid(float(a[0]))))

    if fit_alpha:
        alpha_result = minimize(_alpha_objective, np.array([0.5]), method="SLSQP", bounds=[(0.0, 1.0)], options={"maxiter": 200})
        fitted_alpha = float(np.clip(alpha_result.x[0], 0.0, 1.0))
    else:
        assert alpha is not None
        fitted_alpha = float(np.clip(alpha, 0.0, 1.0))

    hybrid_pred = _hybrid(fitted_alpha)
    hybrid_loss = float(loss_fn(y, hybrid_pred))
    return {
        "exponents": best_weights,
        "ensemble_pred": hybrid_pred,
        "loss": hybrid_loss,
        "alpha": fitted_alpha,
        "geometric_pred": geometric_pred,
        "arithmetic_pred": arithmetic_pred,
    }


__all__ = ["geometric_weight_blend"]
