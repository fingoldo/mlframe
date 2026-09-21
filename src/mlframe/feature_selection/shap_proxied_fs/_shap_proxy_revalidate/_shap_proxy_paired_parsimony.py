"""Paired one-standard-error parsimony pick for ``revalidate_top_n``.

The relative ``parsimony_tol`` band (default 2% of the best honest loss) is far narrower than the sampling noise of a
single-holdout loss difference: at a Brier of ~0.04 the band is ~0.0008, while swapping in a pure-noise column moves
a 375-row holdout Brier by several thousandths either way. The winner of ``top_n`` honest retrains on that one holdout
therefore tends to be the candidate whose noise columns happened to fit it best (winner's curse), and the stability term
``lambda_stab * std`` cannot help because a deterministic booster returns the same loss for every seed (std 0).

This module re-scores the winner against each SMALLER near-best candidate with out-of-fold predictions over ALL rows
(search + holdout, ``n_folds``-fold, stratified for classification): the comparison is no longer made on the holdout
that picked the winner, and it uses 4x the rows. The smallest candidate whose paired per-row OOF loss excess over the
winner is within ``n_se`` standard errors (Breiman's 1-SE rule on the paired difference) wins. A holdout-only paired
test could not separate the two cases this has to: noise-carrying winners beat the signal-only subset by 1.4-2.5 holdout
SE on a 3-signal + 9-noise bed, while dropping a genuine weak interaction pair (logit weight 2) cost only 2.1 holdout SE.
Metrics without a per-row decomposition (AUC, multiclass) keep the relative-tolerance pick unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import clone

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_loss import (
    _classification_proba,
    _expand,
    _slice_cols_to_numpy,
    _try_cap_n_estimators,
)

logger = logging.getLogger(__name__)

# Cap on how many smaller candidates are re-scored (each costs ``n_folds`` fits), smallest first.
_MAX_SMALLER_CANDIDATES = 4
# Only candidates whose holdout stable score is within this RELATIVE window of the winner's are worth re-scoring; a
# subset already far worse on the holdout cannot pass a 1-SE OOF test and would only cost fits.
_NEAR_BEST_WINDOW = 0.20


def _per_row_loss(pred, y, classification: bool, metric: str):
    """Per-row loss whose mean is the aggregate metric, or None when the metric does not decompose per row."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if classification:
        if pred.ndim != 1:
            return None
        if metric == "brier":
            return (pred - y) ** 2
        if metric == "logloss":
            p = np.clip(pred, 1e-7, 1 - 1e-7)
            return -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return None
    if metric == "mae":
        return np.abs(pred - y)
    if metric in ("rmse", "mse"):
        return (pred - y) ** 2  # paired test on squared error; RMSE is monotone in its mean
    return None


def _oof_predictions(model_template, X_parts, y_all, cols, folds, classification, seed, n_estimators_cap):
    """Out-of-fold predictions of ``model_template`` on ``cols`` over the stacked rows of ``X_parts`` (None if not 1-D)."""
    X_all = np.vstack([np.asarray(_slice_cols_to_numpy(Xp, cols)) for Xp in X_parts])
    pred = np.full(len(y_all), np.nan)
    for tr, te in folds:
        est = clone(model_template)
        if n_estimators_cap is not None:
            _try_cap_n_estimators(est, n_estimators_cap)
        if seed is not None and hasattr(est, "random_state"):
            try:
                est.set_params(random_state=int(seed))
            except (ValueError, TypeError):
                pass
        est.fit(X_all[tr], y_all[tr])
        p = _classification_proba(est, X_all[te]) if classification else est.predict(X_all[te])
        if np.ndim(p) != 1:
            return None
        pred[te] = p
    return pred


def paired_one_se_pick(
    ranked: list[dict[str, Any]],
    chosen: dict[str, Any],
    model_template: Any,
    X_search: Any,
    y_search: Any,
    X_holdout: Any,
    y_holdout: Any,
    *,
    classification: bool,
    metric: Any,
    unit_to_members: Any,
    seed: Optional[int] = None,
    n_estimators_cap: Optional[int] = None,
    n_se: float = 1.0,
    n_folds: int = 5,
) -> tuple[Any, dict[str, Any]]:
    """Return ``(features, info)``: the smallest near-best ranked candidate whose paired OOF loss excess over ``chosen``
    is within ``n_se`` standard errors, or ``chosen`` itself when none qualifies / the metric does not decompose."""
    info: Dict[str, Any] = dict(applied=False, tested=[])
    n_se = float(n_se)
    info["n_se"] = n_se
    if n_se <= 0 or not ranked:
        return chosen["features"], info
    probe = np.zeros(2) if not classification else np.array([0.5, 0.5])
    if _per_row_loss(probe, np.zeros(2), classification, metric) is None:
        info["reason"] = f"metric {metric!r} has no per-row decomposition"
        return chosen["features"], info
    window = chosen["stable_score"] + _NEAR_BEST_WINDOW * abs(chosen["stable_score"])
    smaller = sorted(
        (d for d in ranked if d["n_members"] < chosen["n_members"] and d["stable_score"] <= window), key=lambda d: (d["n_members"], d["stable_score"])
    )
    if not smaller:
        return chosen["features"], info
    try:
        from sklearn.model_selection import KFold, StratifiedKFold

        y_all = np.concatenate([np.asarray(y_search), np.asarray(y_holdout)])
        splitter = StratifiedKFold(n_folds, shuffle=True, random_state=0) if classification else KFold(n_folds, shuffle=True, random_state=0)
        folds = list(splitter.split(np.zeros(len(y_all)), y_all))
        X_parts = (X_search, X_holdout)
        ref_pred = _oof_predictions(model_template, X_parts, y_all, _expand(chosen["features"], unit_to_members), folds, classification, seed, n_estimators_cap)
        ref_loss = None if ref_pred is None else _per_row_loss(ref_pred, y_all, classification, metric)
        if ref_loss is None:
            return chosen["features"], info
        info["applied"] = True
        for d in smaller[:_MAX_SMALLER_CANDIDATES]:
            pred = _oof_predictions(model_template, X_parts, y_all, _expand(d["features"], unit_to_members), folds, classification, seed, n_estimators_cap)
            if pred is None:
                break
            diff = _per_row_loss(pred, y_all, classification, metric) - ref_loss
            finite = np.isfinite(diff)
            if finite.sum() < 3:
                continue
            diff = diff[finite]
            mean = float(diff.mean())
            se = float(diff.std(ddof=1) / np.sqrt(diff.size))
            ok = mean <= n_se * se
            info["tested"].append(dict(features=tuple(d["features"]), n_members=d["n_members"], mean_excess=mean, se=se, accepted=bool(ok)))
            if ok:
                return tuple(d["features"]), info
    except Exception as exc:  # the pick is a refinement; a failed refit must not void the revalidated winner
        logger.warning("ShapProxiedFS paired 1-SE parsimony pick failed (%s: %s); keeping the relative-tolerance winner.", type(exc).__name__, exc)
        info["error"] = f"{type(exc).__name__}: {exc}"
    return chosen["features"], info


__all__ = ["paired_one_se_pick"]
