"""``dual_optimizer_weight_blend``: cross-check ensemble weight search with two independent optimizers.

Source: 1st_mechanisms-of-action-moa-prediction.md -- searched CV-optimal blend weights independently with
Optuna's TPE sampler and SciPy's SLSQP against the same OOF-prediction objective, confirmed both converge to
nearly identical weights (a reliability signal that the found optimum is real, not an artifact of one
optimizer's search bias), and noted the search naturally zeroed-out two of seven candidate models (a pruning
signal for the final ensemble).

Runs ``constrained_weight_blend`` (SLSQP, this package's existing gradient-based optimizer) and an
independent Optuna TPE sampler on the SAME OOF objective, then reports the weight divergence between them --
large divergence is a red flag that the SLSQP result may be a poor local optimum (or the objective surface is
genuinely flat/multi-modal), small divergence is corroborating evidence the found weights are real. Also
surfaces near-zero-weighted models (by EITHER optimizer) as pruning candidates.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from mlframe.votenrank.constrained_weight_blend import constrained_weight_blend


def _optuna_simplex_weight_search(preds: np.ndarray, y: np.ndarray, loss_fn: Callable, n_trials: int, random_state: int) -> np.ndarray:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_models = preds.shape[0]

    def _objective(trial: "optuna.Trial") -> float:
        raw = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(n_models)])
        total = raw.sum()
        w = raw / total if total > 0 else np.full(n_models, 1.0 / n_models)
        blended = np.tensordot(w, preds, axes=(0, 0))
        return float(loss_fn(y, blended))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    raw = np.array([study.best_params[f"w{i}"] for i in range(n_models)])
    total = raw.sum()
    return np.asarray(raw / total if total > 0 else np.full(n_models, 1.0 / n_models))


def dual_optimizer_weight_blend(
    oof_preds: Sequence[np.ndarray],
    y_true: np.ndarray,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    n_restarts: int = 5,
    n_optuna_trials: int = 100,
    random_state: int = 0,
    zero_weight_threshold: float = 0.02,
) -> dict:
    """Cross-check SLSQP (``constrained_weight_blend``) against an independent Optuna TPE search.

    Parameters
    ----------
    oof_preds, y_true, loss_fn
        Same as ``constrained_weight_blend``.
    n_restarts
        SLSQP restart count (passed through to ``constrained_weight_blend``).
    n_optuna_trials
        Number of Optuna TPE trials.
    random_state
        Seed for both optimizers.
    zero_weight_threshold
        A model is flagged as a pruning candidate if its weight from EITHER optimizer falls below this.

    Returns
    -------
    dict
        ``slsqp_weights``, ``optuna_weights`` (each ``(n_models,)``), ``slsqp_loss``, ``optuna_loss``,
        ``max_weight_divergence`` (max absolute per-model weight difference between the two optimizers --
        LOW means the two independent searches corroborate each other), ``prune_candidates`` (indices of
        models with near-zero weight from either optimizer).
    """
    preds = np.stack([np.asarray(p, dtype=np.float64) for p in oof_preds], axis=0)
    y = np.asarray(y_true)

    slsqp_result = constrained_weight_blend(oof_preds, y_true, loss_fn, n_restarts=n_restarts, random_state=random_state)
    optuna_weights = _optuna_simplex_weight_search(preds, y, loss_fn, n_trials=n_optuna_trials, random_state=random_state)
    optuna_loss = float(loss_fn(y, np.tensordot(optuna_weights, preds, axes=(0, 0))))

    slsqp_weights = slsqp_result["weights"]
    max_divergence = float(np.max(np.abs(slsqp_weights - optuna_weights)))
    prune_candidates = np.flatnonzero((slsqp_weights < zero_weight_threshold) & (optuna_weights < zero_weight_threshold))

    return {
        "slsqp_weights": slsqp_weights,
        "optuna_weights": optuna_weights,
        "slsqp_loss": slsqp_result["loss"],
        "optuna_loss": optuna_loss,
        "max_weight_divergence": max_divergence,
        "prune_candidates": prune_candidates,
    }


__all__ = ["dual_optimizer_weight_blend"]
