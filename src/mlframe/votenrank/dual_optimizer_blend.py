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
surfaces near-zero-weighted models (by BOTH optimizers, i.e. corroborated) as pruning candidates.

Extension (opt-in ``include_coord_descent``): a two-optimizer check can still miss a shared blind spot -- SLSQP
(local gradient descent) and Optuna TPE (sequential Bayesian sampling) are both attracted to broad, easy-to-find
basins, so on a landscape with a wide decoy basin and a narrow true optimum they can independently converge to
the SAME wrong answer, which reads as "low divergence -- corroborated" when it is actually a correlated blind
spot. A third, mechanically distinct optimizer -- gradient-free pairwise-coordinate descent with randomized
restarts, no gradient and no probabilistic surrogate model -- triangulates against that failure mode: if it
lands on a meaningfully better loss while the other two agree with each other, that is direct evidence of a
correlated 2-way blind spot the pairwise check alone cannot see.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from mlframe.votenrank.constrained_weight_blend import constrained_weight_blend


def _coordinate_descent_simplex_search(preds: np.ndarray, y: np.ndarray, loss_fn: Callable, n_iters: int, random_state: int, n_restarts: int = 3) -> np.ndarray:
    """Independent third optimizer: gradient-free pairwise-coordinate descent on the weight simplex.

    Mechanically distinct from both SLSQP (local gradient-based) and Optuna TPE (Bayesian surrogate-model
    sampling): each step shifts a shrinking amount of mass between a random pair of weights and accepts only
    strict loss improvements, with multiple random-Dirichlet restarts. No gradients, no surrogate model -- a
    different enough search bias that it does not share the failure modes of the other two.
    """
    n_models = preds.shape[0]
    rng = np.random.default_rng(random_state)

    def _loss(w: np.ndarray) -> float:
        """Loss of the weighted blend of preds under candidate weights w."""
        blended = np.tensordot(w, preds, axes=(0, 0))
        return float(loss_fn(y, blended))

    best_w: np.ndarray | None = None
    best_loss = float("inf")
    for restart in range(n_restarts):
        w = np.full(n_models, 1.0 / n_models) if restart == 0 else rng.dirichlet(np.ones(n_models))
        cur_loss = _loss(w)
        step = 0.5
        shrink_every = max(1, n_iters // 10)
        for it in range(n_iters):
            i, j = rng.choice(n_models, size=2, replace=False)
            delta = min(step, w[i])
            w_try = w.copy()
            w_try[i] -= delta
            w_try[j] += delta
            try_loss = _loss(w_try)
            if try_loss < cur_loss:
                w, cur_loss = w_try, try_loss
            else:
                delta = min(step, w[j])
                w_try = w.copy()
                w_try[j] -= delta
                w_try[i] += delta
                try_loss = _loss(w_try)
                if try_loss < cur_loss:
                    w, cur_loss = w_try, try_loss
            if (it + 1) % shrink_every == 0:
                step *= 0.7
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_w = w

    assert best_w is not None
    return np.asarray(best_w)


def _optuna_simplex_weight_search(preds: np.ndarray, y: np.ndarray, loss_fn: Callable, n_trials: int, random_state: int) -> np.ndarray:
    """Search simplex-constrained blend weights minimizing loss_fn via Optuna trials."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_models = preds.shape[0]

    def _objective(trial: "optuna.Trial") -> float:
        """Optuna objective: loss of the blend under this trial's simplex-normalized weights."""
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
    include_coord_descent: bool = False,
    n_coord_descent_iters: int = 300,
    correlated_blind_spot_divergence_threshold: float = 0.1,
    correlated_blind_spot_loss_improvement: float = 0.02,
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
        Seed for all optimizers.
    zero_weight_threshold
        A model is flagged as a pruning candidate if its weight from every optimizer that ran falls below this.
    include_coord_descent
        Opt-in. When True, also runs a third, mechanically distinct optimizer (gradient-free pairwise-coordinate
        descent, see ``_coordinate_descent_simplex_search``) and triangulates its result against SLSQP and
        Optuna to catch a CORRELATED blind spot -- a wrong local optimum both SLSQP and Optuna independently
        converge to, which a 2-way divergence check alone reads as "corroborated" because it only ever compares
        the two most-similar optimizers to each other. Default False: with this omitted, the function is
        bit-identical to the pre-extension 2-optimizer behavior.
    n_coord_descent_iters
        Iterations per coordinate-descent restart (only used when ``include_coord_descent`` is True).
    correlated_blind_spot_divergence_threshold
        SLSQP/Optuna are considered "in apparent agreement" when their weight divergence is below this.
    correlated_blind_spot_loss_improvement
        Relative loss improvement the coordinate-descent optimizer must achieve over the better of
        SLSQP/Optuna, while they are in apparent agreement, to flag a correlated blind spot.

    Returns
    -------
    dict
        ``slsqp_weights``, ``optuna_weights`` (each ``(n_models,)``), ``slsqp_loss``, ``optuna_loss``,
        ``max_weight_divergence`` (max absolute per-model weight difference between SLSQP and Optuna -- LOW
        means the two independent searches corroborate each other), ``prune_candidates`` (indices of models
        with near-zero weight from every optimizer that ran).
        When ``include_coord_descent`` is True, also: ``coord_descent_weights``, ``coord_descent_loss``,
        ``triangulated_max_divergence`` (max pairwise weight divergence across all three optimizers),
        ``correlated_blind_spot_detected`` (bool -- SLSQP and Optuna agree with each other yet the
        coordinate-descent optimizer found a meaningfully better loss, i.e. the two apparently-corroborating
        optimizers share a blind spot the third one escapes).
    """
    preds = np.stack([np.asarray(p, dtype=np.float64) for p in oof_preds], axis=0)
    y = np.asarray(y_true)
    # F10: _coordinate_descent_simplex_search's rng.choice(n_models, size=2, replace=False) requires a
    # population of >= 2 -- a single-candidate pool is a legitimate (if degenerate) input otherwise.
    if include_coord_descent and preds.shape[0] < 2:
        raise ValueError(f"dual_optimizer_weight_blend: include_coord_descent=True requires >= 2 models, got {preds.shape[0]}.")

    slsqp_result = constrained_weight_blend(oof_preds, y_true, loss_fn, n_restarts=n_restarts, random_state=random_state)
    optuna_weights = _optuna_simplex_weight_search(preds, y, loss_fn, n_trials=n_optuna_trials, random_state=random_state)
    optuna_loss = float(loss_fn(y, np.tensordot(optuna_weights, preds, axes=(0, 0))))

    slsqp_weights = slsqp_result["weights"]
    max_divergence = float(np.max(np.abs(slsqp_weights - optuna_weights)))

    if not include_coord_descent:
        prune_candidates = np.flatnonzero((slsqp_weights < zero_weight_threshold) & (optuna_weights < zero_weight_threshold))
        return {
            "slsqp_weights": slsqp_weights,
            "optuna_weights": optuna_weights,
            "slsqp_loss": slsqp_result["loss"],
            "optuna_loss": optuna_loss,
            "max_weight_divergence": max_divergence,
            "prune_candidates": prune_candidates,
        }

    coord_descent_weights = _coordinate_descent_simplex_search(preds, y, loss_fn, n_iters=n_coord_descent_iters, random_state=random_state)
    coord_descent_loss = float(loss_fn(y, np.tensordot(coord_descent_weights, preds, axes=(0, 0))))

    prune_candidates = np.flatnonzero(
        (slsqp_weights < zero_weight_threshold) & (optuna_weights < zero_weight_threshold) & (coord_descent_weights < zero_weight_threshold)
    )

    d_slsqp_optuna = max_divergence
    d_slsqp_coord = float(np.max(np.abs(slsqp_weights - coord_descent_weights)))
    d_optuna_coord = float(np.max(np.abs(optuna_weights - coord_descent_weights)))
    triangulated_max_divergence = max(d_slsqp_optuna, d_slsqp_coord, d_optuna_coord)

    best_pair_loss = min(slsqp_result["loss"], optuna_loss)
    relative_loss_improvement = (best_pair_loss - coord_descent_loss) / abs(best_pair_loss) if best_pair_loss != 0 else 0.0
    correlated_blind_spot_detected = bool(
        d_slsqp_optuna < correlated_blind_spot_divergence_threshold and relative_loss_improvement > correlated_blind_spot_loss_improvement
    )

    return {
        "slsqp_weights": slsqp_weights,
        "optuna_weights": optuna_weights,
        "slsqp_loss": slsqp_result["loss"],
        "optuna_loss": optuna_loss,
        "max_weight_divergence": max_divergence,
        "prune_candidates": prune_candidates,
        "coord_descent_weights": coord_descent_weights,
        "coord_descent_loss": coord_descent_loss,
        "triangulated_max_divergence": triangulated_max_divergence,
        "correlated_blind_spot_detected": correlated_blind_spot_detected,
    }


__all__ = ["dual_optimizer_weight_blend"]
