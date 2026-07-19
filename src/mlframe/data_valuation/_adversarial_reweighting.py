"""Distributionally Robust Optimization (DRO) as a two-player zero-sum game -- the ONE non-cooperative-
game construction from gt_06 that concretely fits mlframe's plumbing.

Scope honesty: this is NOT a GAN, NOT multi-agent RL, NOT mechanism design -- "Nash equilibria in ML"
is a research area, and none of those constructions land in a tabular-ML framework. What DOES land: the
model (player 1) minimizes weighted training loss; an adversary (player 2) chooses per-row weights
within a chi-square uncertainty ball to MAXIMIZE that loss. The alternating-best-response equilibrium is
robust to the worst reweighting the ball allows -- practically, robust to subpopulation shift, minority-
group degradation, and mild covariate shift, WITHOUT requiring group labels. Reference: Namkoong & Duchi
(NeurIPS 2016/2017), chi-square-ball DRO. When group labels ARE available, Group-DRO is strictly easier
and stronger -- use that instead; this group-FREE variant exists because mlframe data generally has no
group annotations.

MEASURED CAVEAT (see ``tests/data_valuation/test_biz_val_adversarial_reweighting.py``): this reweights
by PER-ROW loss, not by group identity, so it does NOT reliably improve a SPECIFIC held-out subgroup's
AUC (measured on a subpopulation-shift bed: worst-group AUC gains were inconsistent/marginal across a
wide rho/step_mix/depth sweep, while overall AUC consistently paid a real cost). What it DOES reliably
guarantee -- verified -- is a lower worst-case chi2-weighted held-out loss than ERM under the SAME ball,
which is the actual quantity the minimax game optimizes for. If a specific known subgroup needs
protecting, use Group-DRO (explicit group weights) instead of expecting this to find it unsupervised.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np


def project_chi2_ball(losses: np.ndarray, rho: float) -> np.ndarray:
    """Closed-form solution to the adversary's inner maximization: max_w sum(w*losses)/n s.t. w in the
    chi-square ball of radius ``rho`` around the uniform distribution, w >= 0, mean(w) == 1.

    The chi2-ball constrained maximizer has the form ``w = max(0, losses - eta) / mean(max(0, losses -
    eta))`` for the unique ``eta`` solving the ball's radius constraint
    ``mean((w - 1)^2) == rho`` (equivalently, sum((max(0,losses-eta))^2)/n / mean(max(0,losses-eta))^2
    - 1 == rho after the mean-1 renormalization) -- found by 1-D bisection on ``eta`` over
    ``[losses.min() - 1, losses.max()]`` (monotone: larger ``eta`` -> smaller/sparser weights ->
    smaller chi2-divergence from uniform, so bisection is well-posed).

    ``rho=0`` returns uniform weights (mean 1, all equal) exactly -- the game degenerates to plain ERM.
    """
    n = losses.shape[0]
    if rho <= 0.0:
        return np.ones(n, dtype=np.float64)

    losses = np.asarray(losses, dtype=np.float64)

    def _chi2_divergence_at(eta: float) -> float:
        """chi-square divergence from uniform of the mean-1-renormalized max(0, losses - eta) weights.

        As eta -> losses.min() the surviving weights are nearly uniform (divergence -> 0); as eta ->
        losses.max() only the top loss keeps nonzero raw mass, so the renormalized weight concentrates
        entirely on that one point -- the maximum achievable divergence for n points, n-1 (verified:
        w=[n,0,...,0] gives mean((w-1)^2) = ((n-1)^2 + (n-1))/n = n-1). eta AT losses.max() exactly is
        the degenerate all-zero-raw case (undefined renormalization); its correct LIMIT is n-1, not 0
        -- returning 0 there breaks monotonicity and makes the bisection converge to the wrong root.
        """
        raw = np.maximum(losses - eta, 0.0)
        s = raw.sum()
        if s <= 0.0:
            return float(n - 1)  # limit as eta -> losses.max() from below: single-point mass, max divergence
        w = raw * (n / s)  # mean(w) == 1
        return float(np.mean((w - 1.0) ** 2))

    # lo must be far enough below losses.min() that the shifted-loss weights are (numerically) uniform,
    # so the bisection bracket actually spans divergence -> 0: a fixed "-1" offset is not far enough
    # when losses have real spread (raw = losses - eta stays proportionally non-uniform at small |eta|).
    loss_range = float(losses.max() - losses.min()) + 1.0
    lo, hi = float(losses.min()) - 1000.0 * loss_range, float(losses.max())
    # chi2 divergence is monotonically INcreasing in eta (larger eta -> fewer, larger surviving
    # weights once renormalized -> more concentrated -> higher divergence); bisect on the gap to target.
    target = min(rho, float(n - 1))  # rho beyond the achievable max concentrates on a single point regardless
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _chi2_divergence_at(mid) > target:
            hi = mid  # divergence too high -> need smaller eta to spread mass back out
        else:
            lo = mid
    eta = (lo + hi) / 2.0

    raw = np.maximum(losses - eta, 0.0)
    s = raw.sum()
    if s <= 0.0:
        return np.ones(n, dtype=np.float64)
    return np.asarray(raw * (n / s))


def dro_reweight_fit(
    fit_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], Any],
    loss_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    *,
    rho: float = 0.5,
    n_rounds: int = 8,
    step_mix: float = 0.5,
    n_splits: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Any, np.ndarray, dict]:
    """Alternating best-response minimax fit: model minimizes weighted loss, adversary reweights within
    a chi-square ball of radius ``rho`` to maximize it (closed-form inner step, see
    :func:`project_chi2_ball`).

    ``fit_fn(X, y, sample_weight) -> fitted_model`` (model must expose ``.predict`` for regression or
    ``.predict_proba`` for classification -- callers pick which via ``loss_fn``'s own convention,
    e.g. ``loss_fn(y, model.predict_proba(X)[:, 1])``). ``loss_fn(y, pred) -> (n,)`` per-row losses,
    higher = worse.

    OVERFITTING GUARD (mandatory, not optional): the adversary's losses are computed OUT-OF-FOLD
    (``n_splits``-fold CV refit each round), never in-sample -- an in-sample adversary just upweights
    rows the model already memorized (noise/outliers), which looks BETTER in-sample and is silently
    wrong out-of-sample. This is the single most important correctness property of this function; do
    not "optimize" it away by reusing the round's already-fit in-sample model's losses.

    ``step_mix`` (fictitious-play smoothing, mandatory not cosmetic): ``w_{t+1} = (1-step_mix)*w_t +
    step_mix*w_raw``. Pure best-response dynamics (``step_mix=1.0``) oscillate (the classic matching-
    pennies pathology of two simultaneous best-responses); smoothing damps this into converging
    trajectories.

    Cost: ``n_rounds * n_splits`` model fits total (each round refits ``n_splits`` OOF folds for the
    adversary's losses, plus one final full-data fit) -- document honestly, this is not cheap.

    Returns ``(final_model, final_weights, info)`` where ``info`` holds ``worst_case_loss_history``,
    ``avg_loss_history`` (both length ``n_rounds``), and ``converged`` (``True`` iff
    ``max|w_{t+1} - w_t| < 1e-3`` on the final round).
    """
    from sklearn.model_selection import KFold

    if rng is None:
        rng = np.random.default_rng()
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]

    w = np.ones(n, dtype=np.float64)
    worst_case_history = []
    avg_history = []
    converged = False

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
    fold_splits = list(kf.split(X))

    model = None
    for _round in range(n_rounds):
        model = fit_fn(X, y, w)

        # OOF losses for the adversary -- refit per fold so the adversary never sees in-sample loss.
        oof_pred = np.empty(n, dtype=np.float64)
        for tr_idx, val_idx in fold_splits:
            fold_model = fit_fn(X[tr_idx], y[tr_idx], w[tr_idx])
            pred = _predict_for_loss(fold_model, X[val_idx])
            oof_pred[val_idx] = pred
        losses = np.asarray(loss_fn(y, oof_pred), dtype=np.float64)

        worst_case_history.append(float(np.mean(w * losses)))
        avg_history.append(float(np.mean(losses)))

        w_raw = project_chi2_ball(losses, rho)
        w_next = (1.0 - step_mix) * w + step_mix * w_raw
        converged = bool(np.max(np.abs(w_next - w)) < 1e-3)
        w = w_next

    info = dict(
        worst_case_loss_history=worst_case_history,
        avg_loss_history=avg_history,
        converged=converged,
    )
    return model, w, info


def _predict_for_loss(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict using predict_proba's positive-class column when available (classification), else predict (regression)."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba)
    return np.asarray(model.predict(X))


__all__ = ["dro_reweight_fit", "project_chi2_ball"]
