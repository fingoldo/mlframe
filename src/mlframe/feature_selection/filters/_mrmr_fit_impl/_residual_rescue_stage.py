"""MRMR glue of the residual rescue (``_residual_rescue_fe``): builds the selected design and the candidate pool from the fitted frame and attaches what the rescue admits.

Design = the selected raw columns plus the selected engineered features (replayed from their recipes). Pool = every other numeric column of the fit frame: the unselected raws and the engineered
columns the FE stages appended but the greedy did not keep. An admitted engineered column is attached through its own recipe (``_engineered_recipes_`` / ``_engineered_features_``), an admitted
raw through ``selected_vars`` / ``support_``. Regression targets only (the continuous target is ``_fe_prewarp_y_continuous_``).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["run_residual_rescue"]


def run_residual_rescue(self, X, selected_vars, cols, engineered_recipes, retention_added_eng_names, verbose) -> "list[str]":
    """Admit the pool columns that reduce the held-out residual of the selected design; returns the admitted names (empty when disabled, not a regression fit, or nothing passes)."""
    if not getattr(self, "fe_residual_rescue_enable", False):
        return []
    y_cont = getattr(self, "_fe_prewarp_y_continuous_", None)
    if y_cont is None or not hasattr(X, "columns") or not hasattr(X, "iloc"):
        return []
    try:
        from .._residual_rescue_fe import DEFAULT_SCAN_ROWS, residual_rescue
        from ..engineered_recipes.shared import apply_recipe

        y = np.asarray(y_cont, dtype=np.float64).ravel()
        if len(y) != len(X):
            return []
        n = len(y)
        if n > DEFAULT_SCAN_ROWS:  # the scan rows are taken before any column is converted: no full-length float64 copy of the frame
            rows = np.linspace(0, n - 1, DEFAULT_SCAN_ROWS).astype(np.int64)
            X, y = X.iloc[rows], y[rows]
        sel_names = {str(cols[i]) for i in selected_vars}
        eng_names = [str(n) for n in (self._engineered_features_ or [])]
        design = {}
        for name in sel_names:
            if name in X.columns and name not in engineered_recipes and np.issubdtype(X[name].dtype, np.number):
                design[name] = np.asarray(X[name].to_numpy(), dtype=np.float64)
        for name, rec in zip(eng_names, self._engineered_recipes_ or []):
            design[name] = np.nan_to_num(np.asarray(apply_recipe(rec, X), dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
        pool = {
            str(c): np.asarray(X[c].to_numpy(), dtype=np.float64)
            for c in X.columns
            if str(c) not in design and str(c) not in sel_names and np.issubdtype(X[c].dtype, np.number)
        }
        admitted = residual_rescue(
            design, pool, y, min_relative_gain=float(self.fe_residual_rescue_min_relative_gain), max_admit=int(self.fe_residual_rescue_max_admit)
        )
    except Exception:
        logger.warning("MRMR residual rescue failed; the selection is left as it was.", exc_info=True)
        return []
    names = []
    col_index = {str(c): i for i, c in enumerate(cols)}
    for name, t_stat, rel in admitted:
        rec = engineered_recipes.get(name)
        if rec is not None:
            self._engineered_recipes_.append(rec)
            self._engineered_features_.append(name)
            retention_added_eng_names.add(name)
        elif name in col_index and col_index[name] not in selected_vars:
            selected_vars.append(col_index[name])
        else:
            continue
        names.append(name)
        if verbose:
            logger.info("MRMR residual rescue: admitted %r (t=%.1f, %.1f%% of the target variance).", name, t_stat, 100.0 * rel)
    if names:
        self.support_ = np.array(selected_vars, dtype=np.int64)
    return names
