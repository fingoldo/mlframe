"""``MRMR._run_fe_step`` FE-step package -- thin re-export facade.

Historical ``from mlframe.feature_selection.filters._mrmr_fe_step import _run_fe_step`` (the bind in
``mrmr.py``) and the helper imports (``_non_numeric_column_indices`` / ``_synergy_bootstrap_can_supply_pool``)
resolve from here. The irreducible single-function body lives in ``._step_core``; the two small operand-pool
helpers live in ``._helpers``. ``combinations`` is re-exported so the lazy-pair behavioural test can introspect
the symbol off the package surface.
"""
from __future__ import annotations

from itertools import combinations

from ._helpers import _non_numeric_column_indices, _synergy_bootstrap_can_supply_pool
from ._step_core import _run_fe_step

__all__ = [
    "_run_fe_step",
    "_non_numeric_column_indices",
    "_synergy_bootstrap_can_supply_pool",
    "combinations",
]
