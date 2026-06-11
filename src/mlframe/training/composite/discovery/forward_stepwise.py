"""Forward-stepwise multi-base selection: picks additional bases to extend a linear_residual seed via 3-fold CV-RMSE on joint-OLS predictions. Lazy-imports ``_linear_residual_multi_fit`` from composite.py to break the import cycle (composite.py re-exports this module at the bottom)."""


from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# forward_stepwise_multi_base (greedy forward-stepwise selection of additional base columns for ``linear_residual_multi``).
#
# Given a base set of candidate columns and a target y, greedily ADD bases one at a time as long as the marginal RMSE reduction from a joint OLS fit exceeds ``min_marginal_rmse_gain`` (relative; default 2%). Caller seeds with one or more anchor bases (typically the single-base ``linear_residual`` winner from discovery); the helper returns the upgraded base list + per-step diagnostics so the caller can inspect what was added and why.
#
# Use case: after Discovery returns a single-base ``linear_residual__lag_feature`` spec, the user can call this helper to find additional bases that contribute orthogonal signal. The upgraded base list feeds into ``CompositeTargetEstimator(transform_name="linear_residual_multi", base_columns=upgraded_bases)`` for production training.
#
# Not auto-integrated into Discovery.fit(): the forward-stepwise pass adds K * (n_candidates - K) extra CV-RMSE evaluations on top of the existing pipeline (e.g. on 4M rows / 25 candidate bases, 60-100 extra fits = 2-5 min). Standalone helper ships now; auto-integration is opt-in.
# ----------------------------------------------------------------------

_MULTI_BASE_DEFAULT_MAX_K: int = 3
_MULTI_BASE_DEFAULT_MIN_MARGINAL_GAIN: float = 0.02


def forward_stepwise_multi_base(
    y_train: np.ndarray,
    candidate_bases: dict[str, np.ndarray],
    *,
    seed_bases: Sequence[str] | None = None,
    max_k: int = _MULTI_BASE_DEFAULT_MAX_K,
    min_marginal_rmse_gain: float = _MULTI_BASE_DEFAULT_MIN_MARGINAL_GAIN,
    cv_folds: int = 3,
    random_state: int = 42,
    # C-P2-11: default flipped to True. The selector is downstream of every honest-OOF code path
    # in the suite (composite_target_discovery passes through a temporal split). Random-shuffle CV
    # over time-correlated rows gives optimistic gains; a base that survives KFold(shuffle=True) can
    # regress on a true forward-walk. TimeSeriesSplit on non-temporal data is harmless (just a less
    # efficient random split). Callers with confirmed non-temporal data can pass time_aware=False.
    time_aware: bool = True,
    cv_splitter: Any = None,
    groups: np.ndarray | None = None,
    cv_selector_mode: str = "mean",
    cv_selector_alpha: float = 1.0,
    cv_selector_confidence: float = 0.9,
    cv_selector_quantile_level: float = 0.9,
    cv_persist_fold_scores: bool = False,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Greedy forward-stepwise base selection for ``linear_residual_multi``.

    Parameters
    ----------
    y_train
        Target array (1-D).
    candidate_bases
        Mapping ``column_name -> 1-D base array``. All arrays must have length ``len(y_train)``. The seed_bases are removed from the candidate pool internally (no duplicate-add).
    seed_bases
        Optional starting base list -- typically the single-base winner from discovery. ``None`` starts from empty and greedily picks the best single base first.
    max_k
        Maximum total bases in the final list (including seeds). Default 3 -- beyond that, multi-collinearity dominates and gains diminish.
    min_marginal_rmse_gain
        Minimum RELATIVE RMSE reduction required to ACCEPT an addition. Default 0.02 (= 2% improvement). Set to 0.0 to add greedily without a gate.
    cv_folds
        K-fold CV for scoring each candidate base addition. Default 3 (balance between speed and noise).
    random_state
        Seed for KFold splitter.

    Returns
    -------
    (kept_bases, diagnostics):
    - ``kept_bases``: ordered list of base column names (seeds first, then greedily-added).
    - ``diagnostics``: list of per-step dicts ``{step, candidate_added, rmse_before, rmse_after, marginal_gain, accepted}`` for caller-facing audit.
    """
    from sklearn.model_selection import KFold, TimeSeriesSplit  # lazy
    # Lazy-import composite-internal transforms to break the import cycle (composite.py re-exports this module at the bottom; importing at module top would deadlock).
    from .. import (
        _linear_residual_multi_fit,
        _linear_residual_multi_forward,
        _linear_residual_multi_inverse,
    )
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if y.size < 4:
        return list(seed_bases or []), []
    # Materialise the full candidate map ONCE (no removal -- seeds and pool both read from this dict). The "available" set for the next round is simply ``candidates.keys() - kept``.
    candidates = {name: np.asarray(arr, dtype=np.float64).reshape(-1)
                  for name, arr in candidate_bases.items()}
    # A length mismatch otherwise surfaces much later as an opaque column_stack ValueError; fail at the boundary with the offending column named.
    for _n, _a in candidates.items():
        if _a.size != y.size:
            raise ValueError(
                f"forward_stepwise_multi_base: candidate '{_n}' has length {_a.size}, expected {y.size} (== len(y_train))."
            )
    # Validate seeds against the candidate map. Seeds not in candidate_bases are an API misuse; raise loudly so caller fixes their wiring.
    seeds = list(seed_bases or [])
    missing_seeds = [s for s in seeds if s not in candidates]
    if missing_seeds and candidates:
        # If candidates is empty AND seeds is non-empty (e.g. test_empty_pool case), treat as "no candidates to evaluate" -- just return seeds unchanged.
        raise ValueError(
            f"forward_stepwise_multi_base: seed_bases names {missing_seeds} must appear in candidate_bases (or candidate_bases must be empty if you only want to confirm the seeds without screening)."
        )
    if not candidates and seeds:
        # Empty candidates + seeds -> nothing to greedily add; return seeds verbatim.
        return list(seeds), []
    kept = list(seeds)
    diagnostics: list[dict[str, Any]] = []

    from ..._cv_aggregation import aggregate_fold_scores

    def _cv_rmse_with_folds(base_names: list[str]) -> tuple[float, list[float]]:
        """Return ``(aggregated_score, per_fold_rmses)`` so callers can persist the full table
        when ``cv_persist_fold_scores=True``. ``cv_selector_mode='mean'`` is bit-identical to
        the original ``float(np.mean(fold_rmses))`` path."""
        _n = int(y.size)
        _groups_eff = None
        if groups is not None:
            _g = np.asarray(groups)
            if _g.shape[0] == _n and np.unique(_g).size >= 2:
                _groups_eff = _g
        _use_groups = False
        if cv_splitter is not None:
            kf = cv_splitter
        elif time_aware:
            kf = TimeSeriesSplit(n_splits=int(cv_folds))
        elif _groups_eff is not None:
            # A19: group-aware CV so a per-group-CONSTANT base cannot show a
            # phantom gain from rows of the same group spanning train+val.
            # n_splits capped at the group count (A10: silent GroupKFold->KFold
            # when n_groups<cv_folds is avoided by clamping instead).
            from sklearn.model_selection import GroupKFold
            kf = GroupKFold(n_splits=max(2, min(int(cv_folds), int(np.unique(_groups_eff).size))))
            _use_groups = True
        else:
            kf = KFold(n_splits=int(cv_folds), shuffle=True, random_state=int(random_state))

        def _iter_splits():
            if _use_groups:
                return kf.split(np.arange(_n), groups=_groups_eff)
            return kf.split(np.arange(_n))
        if not base_names:
            # Zero-base baseline = CV-RMSE of the train-fold-MEAN predictor over
            # the SAME folds the candidates use, NOT the full-sample std. Under
            # TimeSeriesSplit on trending y the train-fold mean lags the val
            # fold, so full-sample std understated the no-base error and
            # inflated the first base's apparent gain (A29).
            fold0: list[float] = []
            _y_all_finite = y[np.isfinite(y)]
            _global_mu = float(np.mean(_y_all_finite)) if _y_all_finite.size else 0.0
            for _tr0, _va0 in _iter_splits():
                _ytr0 = y[_tr0][np.isfinite(y[_tr0])]
                _mu0 = float(np.mean(_ytr0)) if _ytr0.size else _global_mu
                _d0 = y[_va0] - _mu0
                _f0 = np.isfinite(_d0)
                if _f0.sum():
                    fold0.append(float(np.sqrt(np.mean(_d0[_f0] * _d0[_f0]))))
            if not fold0:
                sentinel = float(np.std(_y_all_finite)) if _y_all_finite.size else 0.0
                return sentinel, [sentinel]
            agg0 = aggregate_fold_scores(
                fold0, mode=cv_selector_mode, direction="min",
                alpha=cv_selector_alpha, confidence=cv_selector_confidence,
                quantile_level=cv_selector_quantile_level,
            )
            return agg0, fold0
        # All names are guaranteed to be in ``candidates`` by the validation above.
        base_matrix = np.column_stack([candidates[n] for n in base_names])
        fold_rmses: list[float] = []
        for train_idx, val_idx in _iter_splits():
            # Fit OLS y ~ base_matrix on TRAIN.
            params = _linear_residual_multi_fit(y[train_idx], base_matrix[train_idx])
            # Standard OLS prediction on VAL: y_hat = base @ alphas + beta. This is the OLS fit's val performance -- the right metric for "does adding this base reduce hold-out RMSE?" -- NOT the forward-then-inverse roundtrip which is a tautology that recovers y exactly.
            alphas = np.asarray(params["alphas"], dtype=np.float64)
            beta = float(params["beta"])
            y_pred_val = base_matrix[val_idx].astype(np.float64) @ alphas + beta
            diff = y_pred_val - y[val_idx]
            finite = np.isfinite(diff)
            if finite.sum() == 0:
                continue
            fold_rmses.append(float(np.sqrt(np.mean(diff[finite] * diff[finite]))))
        if not fold_rmses:
            return float("nan"), []
        aggregated = aggregate_fold_scores(
            fold_rmses,
            mode=cv_selector_mode,  # type: ignore[arg-type]
            direction="min",
            alpha=cv_selector_alpha,
            confidence=cv_selector_confidence,
            quantile_level=cv_selector_quantile_level,
        )
        return aggregated, fold_rmses

    def _cv_rmse(base_names: list[str]) -> float:
        return _cv_rmse_with_folds(base_names)[0]

    # Step 0: baseline RMSE with current kept bases (or 0-base if seeds empty).
    rmse_current = _cv_rmse(kept)

    # Greedy forward selection: add the best candidate each round if it clears the gate.
    while len(kept) < max_k:
        available = [n for n in candidates.keys() if n not in kept]
        if not available:
            break
        best_name = None
        best_rmse = float("inf")
        per_candidate_folds: dict[str, list[float]] = {}
        for cand_name in available:
            trial = kept + [cand_name]
            rmse_trial, fold_rmses_trial = _cv_rmse_with_folds(trial)
            if cv_persist_fold_scores:
                per_candidate_folds[cand_name] = list(fold_rmses_trial)
            if np.isfinite(rmse_trial) and rmse_trial < best_rmse:
                best_rmse = rmse_trial
                best_name = cand_name
        if best_name is None:
            break
        gain = (rmse_current - best_rmse) / max(rmse_current, 1e-12)
        accepted = gain >= float(min_marginal_rmse_gain)
        step_diag: dict[str, Any] = {
            "step": len(diagnostics) + 1,
            "candidate_added": best_name,
            "rmse_before": rmse_current,
            "rmse_after": best_rmse,
            "marginal_gain": gain,
            "accepted": bool(accepted),
        }
        if cv_persist_fold_scores:
            step_diag["fold_rmses_per_candidate"] = per_candidate_folds
        diagnostics.append(step_diag)
        if not accepted:
            break
        kept.append(best_name)
        rmse_current = best_rmse
    return kept, diagnostics
