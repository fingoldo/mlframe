"""A pair form the pure-form retention keeps must beat its own better side.

A ``unary_binary`` form ``binary(unary_a(a), unary_b(b))`` can pass the non-separability gate on the strength of ONE side alone: on
``y = sign(x**2 - 1)`` the forms ``max(abs(x), sin(noise_k))`` correlate with the class through ``abs(x)``, the noise side only overwrites
the rows where ``|x|`` is small (where ``y`` is constant), and three of them displaced ``abs(x)`` itself from the support. The form is
compared here with each of its sides under the scorer and fold split the retention's greedy commits with (CV log-loss of a scaled
logistic model, CV MAE of OLS for regression) and kept only when it improves on the better side by the greedy's own relative margin.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np

from ._linear_cv_loss import cv_loss
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)


def _side_values(recipe: Any, X: Any) -> "list[np.ndarray] | None":
    """``[unary_a(X[a]), unary_b(X[b])]`` of a ``unary_binary`` recipe, replayed the way the recipe replays its operands; ``None`` when unavailable."""
    if getattr(recipe, "kind", None) != "unary_binary" or len(getattr(recipe, "src_names", ()) or ()) != 2:
        return None
    from .engineered_recipes.shared import apply_unary_side as _apply_unary_side, cached_unary_transformations as _cached_unary_transformations

    funcs = _cached_unary_transformations(recipe.unary_preset)
    out = []
    for side, src, uname in zip(("a", "b"), recipe.src_names, recipe.unary_names):
        vals = np.asarray(X[src].to_numpy(), dtype=np.float64)  # Intentional float64: the retention gates score float64 columns
        out.append(np.asarray(_apply_unary_side(recipe, side, uname, vals, funcs), dtype=np.float64))  # Intentional float64: as above
    return out


def drop_forms_not_beating_their_sides(candidates: list, X: Any, y: np.ndarray, *, classification: bool, seed: int, n_folds: int = 3, improve_rel: float = 0.01) -> list:
    """``candidates`` without the pair forms whose CV loss is not below ``(1 - improve_rel)`` times the better side's; raw passthroughs and forms
    whose sides cannot be replayed are kept unchanged."""
    from ._edge_pool import map_on_edge_pool

    # candidates that share a source and a unary share a side: its loss (3 logistic fits at 100k rows) is computed once per distinct column content
    side_losses: "dict[bytes, float]" = {}

    def _side_loss(values: np.ndarray) -> float:
        """CV loss of one side column, reused across the candidates that replay the same side."""
        key = hashlib.blake2b(np.ascontiguousarray(values).tobytes(), digest_size=16).digest()
        loss = side_losses.get(key)
        if loss is None:
            loss = cv_loss(values, y, classification, n_folds, seed)
            side_losses[key] = loss
        return loss

    def _verdict(cand: Any) -> bool:
        """Whether ``cand`` stays: raw passthroughs, forms whose sides cannot be replayed and forms that cannot be judged stay; the others must beat their better side."""
        recipe = getattr(cand, "recipe", None)
        if recipe is None:
            return True
        try:
            sides = _side_values(recipe, X)
            if sides is None:
                return True
            form_loss = cv_loss(cand.values, y, classification, n_folds, seed)
            side_loss = min(_side_loss(s) for s in sides)
        except Exception as exc:
            log_throttle(logger, "fe_pure_form_sides.check_skipped", logging.WARNING, "pure-form side check skipped for %s, so the candidate is kept unchecked: %s", getattr(cand, "name", "?"), exc)
            return True
        beats = form_loss < (1.0 - improve_rel) * side_loss
        if not beats:
            logger.debug("pure-form retention: %s does not beat its better side (CV loss %.5f vs %.5f); dropped", cand.name, form_loss, side_loss)
        return beats

    # the candidates are independent (each fits its own small linear models on its own columns) and the pool returns the verdicts in candidate order
    keep = map_on_edge_pool(_verdict, candidates)
    return [cand for cand, verdict in zip(candidates, keep) if verdict]
