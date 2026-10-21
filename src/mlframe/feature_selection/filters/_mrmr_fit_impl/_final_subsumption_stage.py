"""MRMR glue of the end-of-fit subsumption pass (``_fe_final_subsumption``): runs after every selection stage has finished.

The anchors are the engineered compounds that combine two or more raw operands; every other selected column (raw or single-operand
engineered) is dropped when it carries no information about the target beyond those compounds. Columns without fit-time values are left
alone, since they cannot be judged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype

from ._residual_rescue_stage import _raw_sources

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from ..mrmr import MRMR

logger = logging.getLogger(__name__)


def _column_values(X, name):
    """Continuous values of the raw column ``name`` (pandas or polars)."""
    col = X[name]
    return np.asarray(col.to_numpy() if hasattr(col, "to_numpy") else col, dtype=_crit_np_dtype()).ravel()


def _is_numeric_column(X, name):
    """True when the raw column ``name`` (pandas or polars) holds numbers or booleans."""
    import pandas as pd

    dtype = X[name].dtype
    if hasattr(dtype, "is_numeric"):
        return bool(dtype.is_numeric()) or str(dtype) == "Boolean"
    return bool(pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype))


def run_final_subsumption_prune(
    self: "MRMR", X: "pd.DataFrame | pl.DataFrame", selected_vars: list[int], classes_y: np.ndarray, verbose: int | bool
) -> tuple[bool, list[int]]:
    """Drop the selected columns the compounds subsume; returns ``(changed, selected_vars)``."""
    try:
        from .._fe_final_subsumption import final_subsumption_prune

        raw_names = [str(c) for c in self.feature_names_in_]
        raw_set = frozenset(raw_names)
        store = getattr(self, "_engineered_continuous_", None) or {}
        recipes = list(self._engineered_recipes_ or [])
        eng_values = {}
        for recipe in recipes:
            name = str(getattr(recipe, "name", ""))
            if name in store:
                eng_values[name] = np.asarray(store[name], dtype=_crit_np_dtype()).ravel()
        raw_selected = [raw_names[int(v)] for v in selected_vars if 0 <= int(v) < len(raw_names)]
        candidates = {}
        for name in raw_selected:
            # a string / categorical column is not a continuous candidate: the compounds cannot subsume it, and converting it raised and cancelled the whole pass
            if _is_numeric_column(X, name):
                candidates[name] = _column_values(X, name)
        candidates.update(eng_values)
        anchors = [name for name in eng_values if len(_raw_sources(name, raw_set)) >= 2]
        if not anchors:
            return False, selected_vars
        y_continuous = getattr(self, "_fe_prewarp_y_continuous_", None)
        dropped = final_subsumption_prune(
            candidates, anchors, np.asarray(classes_y).ravel(), y_continuous=y_continuous, seed=int(getattr(self, "random_seed", 0) or 0)
        )
        if not dropped:
            return False, selected_vars
        dropped_set = set(dropped)
        dropped_raw = dropped_set & set(raw_selected)
        dropped_eng = dropped_set - dropped_raw
        if dropped_raw:
            selected_vars = [v for v in selected_vars if not (0 <= int(v) < len(raw_names) and raw_names[int(v)] in dropped_raw)]
            self._raw_redundancy_dropped_ = set(getattr(self, "_raw_redundancy_dropped_", None) or set()) | dropped_raw
        if dropped_eng:
            self._engineered_recipes_ = [r for r in recipes if str(getattr(r, "name", "")) not in dropped_eng]
            self._engineered_features_ = [f for f in (self._engineered_features_ or []) if str(f) not in dropped_eng]
        if verbose:
            logger.info("MRMR final subsumption pass: dropped %d column(s) carrying no information beyond the compounds: %s", len(dropped), sorted(dropped))
        self._final_subsumption_dropped_ = sorted(dropped)
        return True, selected_vars
    except Exception:
        logger.warning("MRMR final subsumption pass failed; the selection is left as it was.", exc_info=True)
        return False, selected_vars
