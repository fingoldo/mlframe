"""Additive-fusion candidates for the selection compaction.

The additive fusion runs inside an FE step and fuses the halves that step produced. A half can reach the selection later (the residual rescue admits ``mul(log(c),sin(d))`` after the
step that built ``div(sqr(a),abs(b))``), and the two then stay separate fragments of one additive compound. This module offers the finished selection to the same fusion: the
selected engineered features are replayed from their recipes, the fusion proposes ``add`` / ``sub`` compounds of disjoint halves, and each compound joins the compaction's candidate pool,
where the bias-corrected joint-information criterion decides whether it replaces the fragments.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def synthesized_fusions(self: Any, X: Any, sel_eng: "list[str]", by_recipe: dict, all_recipes: dict, raw_set: "frozenset[str]", cols: list, classes_y: np.ndarray) -> "list[tuple]":
    """``(name, recipe, codes)`` of every fused compound the additive fusion proposes for the selected engineered features; empty when fewer than two qualify or on any failure."""
    if len(sel_eng) < 2 or not bool(getattr(self, "fe_additive_fusion_enable", True)):
        return []
    try:
        from .._fe_additive_fusion import propose_additive_fusions
        from .._fe_usability_signal import _crit_np_dtype
        from ..engineered_recipes.shared import apply_recipe
        from ._selection_compaction import _replayed_codes

        values: dict = {}
        for name in sel_eng:
            recipe = by_recipe.get(name)
            if recipe is None:
                continue
            replayed = np.nan_to_num(np.asarray(apply_recipe(recipe, X), dtype=_crit_np_dtype()).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
            if replayed.shape[0] == len(X):
                values[name] = replayed
        halves = [nm for nm in sel_eng if nm in values]
        if len(halves) < 2:
            return []
        fused, _subsumed, _subsumed_raws = propose_additive_fusions(
            self,
            engineered_recipes=all_recipes,
            engineered_continuous=values,
            newly_engineered_names=halves,
            raw_name_set=set(raw_set),
            cols=list(cols),
            classes_y=np.asarray(classes_y).ravel(),
            X=X,
            nbins=int(self.quantization_nbins),
            seed=int(getattr(self, "random_seed", 0) or 0),
            verbose=0,
        )
        return [(str(f["name"]), f["recipe"], _replayed_codes(f["recipe"], X)) for f in fused]
    except Exception:
        logger.warning("MRMR compaction: the additive-fusion candidates failed; the compaction continues without them.", exc_info=True)
        return []
