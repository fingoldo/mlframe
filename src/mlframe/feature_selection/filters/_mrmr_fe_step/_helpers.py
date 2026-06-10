"""Small top-level helpers for the _mrmr_fe_step FE-step package.

_non_numeric_column_indices and _synergy_bootstrap_can_supply_pool were top-level
helpers of the former _mrmr_fe_step.py monolith; _step_core._run_fe_step imports them
from here, and the package __init__ re-exports both so historical
from ..._mrmr_fe_step import _non_numeric_column_indices paths keep resolving.
"""
from __future__ import annotations

import pandas as pd


def _non_numeric_column_indices(X, cols) -> set:
    """Positional indices of columns in ``X`` whose dtype is not numeric.

    The pair / synergy FE operands index positionally into ``X``; a string /
    categorical column reaching the numeric basis transforms raises, so callers
    subtract these indices from the operand pool. Returns an empty set when the
    dtype of ``X`` cannot be inspected (array input -> already all-numeric).
    """
    idx: set = set()
    try:
        schema = getattr(X, "schema", None)
        if schema is not None:  # polars DataFrame
            dtypes = [schema[c] for c in X.columns]
            for i, dt in enumerate(dtypes):
                if not dt.is_numeric():
                    idx.add(i)
            return idx
        dtypes_attr = getattr(X, "dtypes", None)
        if dtypes_attr is not None:  # pandas DataFrame
            for i, dt in enumerate(dtypes_attr):
                if not pd.api.types.is_numeric_dtype(dt):
                    idx.add(i)
    except Exception:
        return set()
    return idx


def _synergy_bootstrap_can_supply_pool(self, num_fs_steps: int, data) -> bool:
    """Whether the synergy bootstrap (below) would seed a non-empty interaction-only pair pool.

    Mirrors the bootstrap's own gating (``fe_synergy_screen_max_features`` enabled, first FE step, enough rows) so the empty-screen branch can decide to CONTINUE into the FE
    step on a pure-interaction target whose marginals all screen out -- rather than returning None and engineering nothing. The actual per-frame caps (feature count, sweep cost)
    are re-checked at the bootstrap site; this is the cheap necessary-condition probe.
    """
    if int(getattr(self, "fe_synergy_screen_max_features", 0) or 0) <= 0:
        return False
    if num_fs_steps != 0:
        return False
    _min_rows = int(getattr(self, "fe_synergy_min_rows", 300) or 0)
    _n_rows = int(data.shape[0]) if hasattr(data, "shape") else 0
    return _n_rows >= _min_rows
