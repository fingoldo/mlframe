"""Final guarantee of the ``emit_both`` policy: every raw operand of a selected engineered recipe is also selected raw.

The downstream model can learn from a raw operand what the composite does not express, possibly beating the feature engineering itself. Operands are taken only from recipes that
survived the whole fit, so a noise column that no recipe uses stays out. The step runs after every pass that can drop or re-add columns, which is why it is not folded into the earlier
per-step operand re-attach: a recipe, and with it its operand, can still disappear after that step.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def retain_recipe_operands_final(self: Any) -> list[str]:
    """Append to ``support_`` the raw operands of the surviving recipes that are missing from it; returns their names. No-op under the ``drop`` policy."""
    if getattr(self, "redundancy_policy", "emit_both") == "drop":
        return []
    recipes = list(getattr(self, "_engineered_recipes_", None) or [])
    if not recipes or not hasattr(self, "support_"):
        return []
    from ._finalise import surviving_recipe_operands

    raw_names = [str(c) for c in self.feature_names_in_]
    index = {nm: i for i, nm in enumerate(raw_names)}
    operands: set[str] = set(surviving_recipe_operands(recipes, raw_names))
    for rec in recipes:
        operands.update(str(s) for s in (getattr(rec, "src_names", None) or ()) if str(s) in index)
    support = [int(v) for v in np.asarray(self.support_).ravel()]
    have = set(support)
    added = [nm for nm in raw_names if nm in operands and index[nm] not in have]
    if not added:
        return []
    self.support_ = np.array([*support, *(index[nm] for nm in added)], dtype=np.int64)
    self.n_features_ = int(self.support_.size) + len(recipes)
    self._raw_redundancy_dropped_ = set(getattr(self, "_raw_redundancy_dropped_", None) or set()) - set(added)
    if hasattr(self, "support_nonlinear_"):
        self.support_nonlinear_ = self.support_
    logger.debug("MRMR emit_both: kept the raw operand(s) %s of the selected recipes.", added)
    return added
