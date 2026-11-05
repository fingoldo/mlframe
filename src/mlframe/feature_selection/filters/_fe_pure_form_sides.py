"""A pair form the pure-form retention keeps must beat its own better side.

A ``unary_binary`` form ``binary(unary_a(a), unary_b(b))`` can pass the non-separability gate on the strength of ONE side alone: on
``y = sign(x**2 - 1)`` the forms ``max(abs(x), sin(noise_k))`` correlate with the class through ``abs(x)``, the noise side only overwrites
the rows where ``|x|`` is small (where ``y`` is constant), and three of them displaced ``abs(x)`` itself from the support. The form is
compared here with each of its sides under the scorer and fold split the retention's greedy commits with (CV log-loss of a scaled
logistic model, CV MAE of OLS for regression) and kept only when it improves on the better side by the greedy's own relative margin.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _side_values(recipe: Any, X: Any) -> "list[np.ndarray] | None":
    """``[unary_a(X[a]), unary_b(X[b])]`` of a ``unary_binary`` recipe, replayed the way the recipe replays its operands; ``None`` when unavailable."""
    if getattr(recipe, "kind", None) != "unary_binary" or len(getattr(recipe, "src_names", ()) or ()) != 2:
        return None
    from .engineered_recipes._recipe_unary_binary import _apply_unary_side, _cached_unary_transformations

    funcs = _cached_unary_transformations(recipe.unary_preset)
    out = []
    for side, src, uname in zip(("a", "b"), recipe.src_names, recipe.unary_names):
        vals = np.asarray(X[src].to_numpy(), dtype=np.float64)  # Intentional float64: the retention gates score float64 columns
        out.append(np.asarray(_apply_unary_side(recipe, side, uname, vals, funcs), dtype=np.float64))  # Intentional float64: as above
    return out


def _cv_score(col: np.ndarray, y: np.ndarray, classification: bool, n_folds: int, seed: int) -> float:
    """Mean CV loss of a one-column scaled linear model on ``col`` (log-loss for classification, MAE for regression); ``inf`` when it cannot be fit."""
    from sklearn.model_selection import KFold

    from ._usability_aware_selection import _scrub, _usability_scorers

    scorers = _usability_scorers(classification, y)
    if scorers is None:
        return float("inf")
    mk, logloss, y_enc, _n_classes = scorers
    x = _scrub(np.asarray(col, dtype=np.float64)).reshape(-1, 1)  # Intentional float64: as above
    if not np.isfinite(x).all() or float(np.std(x)) <= 1e-12:
        return float("inf")
    losses = []
    for tr, va in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(x):
        model = mk()
        if classification:
            model.fit(x[tr], y_enc[tr])
            losses.append(logloss(y_enc[va], model.predict_proba(x[va])))
        else:
            model.fit(x[tr], y[tr])
            losses.append(float(np.mean(np.abs(y[va] - model.predict(x[va])))))
    return float(np.mean(losses))


def drop_forms_not_beating_their_sides(candidates: list, X: Any, y: np.ndarray, *, classification: bool, seed: int, n_folds: int = 3, improve_rel: float = 0.01) -> list:
    """``candidates`` without the pair forms whose CV loss is not below ``(1 - improve_rel)`` times the better side's; raw passthroughs and forms
    whose sides cannot be replayed are kept unchanged."""
    kept = []
    for cand in candidates:
        recipe = getattr(cand, "recipe", None)
        if recipe is None:
            kept.append(cand)
            continue
        try:
            sides = _side_values(recipe, X)
            if sides is None:
                kept.append(cand)
                continue
            form_loss = _cv_score(cand.values, y, classification, n_folds, seed)
            side_loss = min(_cv_score(s, y, classification, n_folds, seed) for s in sides)
        except Exception as exc:
            logger.debug("pure-form side check skipped for %s: %s", getattr(cand, "name", "?"), exc)
            kept.append(cand)
            continue
        if form_loss < (1.0 - improve_rel) * side_loss:
            kept.append(cand)
        else:
            logger.debug("pure-form retention: %s does not beat its better side (CV loss %.5f vs %.5f); dropped", cand.name, form_loss, side_loss)
    return kept
