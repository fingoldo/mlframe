"""Restore the raw columns the residual rescue pruned when the finished selection needs their private linear term.

The rescue prunes a raw operand when a linear model gets no held-out gain from it beyond the design of that moment (``_linearly_useless``). Later passes shrink the design (the final subsumption, the
compaction, the pruning of a helper column), so a verdict given against a design that no longer exists can leave the selection without the only column that carries a linear term of the target: a
compound that is a nonlinear function of ``b`` plus a column for ``a ** 2`` and no ``b``. This pass judges every raw the rescue pruned once more, against the FINAL design, with the rescue's own
bar (``t > z`` and a share of the target variance of at least ``fe_residual_rescue_operand_min_linear_gain``), and puts back the ones that pass. A raw that adds nothing to the final design
stays out; the pass is a no-op when the rescue pruned nothing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype

logger = logging.getLogger(__name__)


def record_pruned_raws(self: Any, names: "list[str]") -> None:
    """Remember the raw columns the rescue pruned, for :func:`restore_pruned_raws` (a fit attribute: a list of names)."""
    if names:
        held = list(getattr(self, "_residual_rescue_pruned_raw_", None) or [])
        self._residual_rescue_pruned_raw_ = held + [n for n in names if n not in held]


def _final_design(self: Any, X: Any, selected_vars: Any, cols: Any, engineered_recipes: Any) -> dict:
    """``{name: values}`` of the finished selection: the selected raw columns and every engineered column, in the criterion dtype."""
    from ..engineered_recipes.shared import apply_recipe

    design = {}
    for i in selected_vars:
        name = str(cols[i])
        if name in X.columns and name not in engineered_recipes and np.issubdtype(X[name].dtype, np.number):
            design[name] = np.asarray(X[name].to_numpy(), dtype=_crit_np_dtype())
    for name, rec in zip([str(f) for f in (self._engineered_features_ or [])], self._engineered_recipes_ or []):
        design[name] = np.nan_to_num(np.asarray(apply_recipe(rec, X), dtype=_crit_np_dtype()).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    return design


def restore_pruned_raws(self: Any, X: Any, selected_vars: Any, cols: Any, engineered_recipes: Any, verbose: Any) -> bool:
    """Put back the raws the rescue pruned that still give the finished design a significant private linear term; True when the selection changed."""
    pruned = [n for n in (getattr(self, "_residual_rescue_pruned_raw_", None) or []) if n]
    if not pruned or not getattr(self, "fe_residual_rescue_enable", False) or getattr(self, "redundancy_policy", "emit_both") != "drop":
        return False
    y_cont = getattr(self, "_fe_prewarp_y_continuous_", None)
    if y_cont is None or not hasattr(X, "columns") or not hasattr(X, "iloc"):
        return False
    col_index = {str(c): i for i, c in enumerate(cols)}
    selected = {int(v) for v in selected_vars}
    candidates = [n for n in pruned if n in col_index and col_index[n] not in selected and n in X.columns and np.issubdtype(X[n].dtype, np.number)]
    if not candidates:
        return False
    try:
        from scipy.stats import norm

        from .._pair_residual_screen import FAMILY_ALPHA
        from .._residual_rescue_fe import DEFAULT_SCAN_ROWS, MIN_ROWS, design_precompute
        from .._y_encoding import FEW_CLASSES_MAX
        from ._residual_rescue_stage import _is_group_leak_raw, _linearly_useless

        y = np.asarray(y_cont, dtype=_crit_np_dtype()).ravel()
        if len(y) != len(X) or len(y) < MIN_ROWS or np.unique(y).size <= FEW_CLASSES_MAX:
            return False
        if len(y) > DEFAULT_SCAN_ROWS:
            rows = np.linspace(0, len(y) - 1, DEFAULT_SCAN_ROWS).astype(np.int64)
            X, y = X.iloc[rows], y[rows]
        design = _final_design(self, X, selected_vars, cols, engineered_recipes)
        if not design:
            return False
        z = float(norm.isf(FAMILY_ALPHA))
        floor = float(self.fe_residual_rescue_operand_min_linear_gain)
        restored = []
        for name in candidates:
            if _is_group_leak_raw(X, name, y):
                continue
            col = np.asarray(X[name].to_numpy(), dtype=_crit_np_dtype())
            if not _linearly_useless(design, col, y, design_precompute(design, len(y)), z, floor):
                design[name] = col  # the next candidate is judged beside this one
                restored.append(name)
    except Exception:
        logger.warning("MRMR pruned-raw restoration failed; the selection is left as it was.", exc_info=True)
        return False
    for name in restored:
        selected_vars.append(col_index[name])
        if verbose:
            logger.info("MRMR pruned-raw restoration: %r gives the finished selection a significant private linear term.", name)
    if restored:
        self.support_ = np.array(selected_vars, dtype=np.int64)
        self.n_features_ = len(selected_vars) + len(self._engineered_recipes_)
    return bool(restored)
