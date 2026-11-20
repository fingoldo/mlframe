"""A high-cardinality selected factor skipped in confirmation mode must leave the running gain untouched."""

from __future__ import annotations

import numba
import numpy as np
from numba.core import types

from mlframe.feature_selection.filters.evaluation import evaluate_gain


def _call(max_confirmation_cand_nbins: int) -> tuple:
    """One confirmation-mode evaluation of candidate col0 against the single selected factor col2 (4 bins)."""
    rng = np.random.default_rng(0)
    n = 400
    col0 = rng.integers(0, 3, size=n)
    col2 = rng.integers(0, 4, size=n)
    col1 = (col0 + col2 + rng.integers(0, 2, size=n)) % 4
    data = np.column_stack([col0, col1, col2]).astype(np.int32)
    nbins = np.array([3, 4, 4], dtype=np.int32)
    return evaluate_gain(
        current_gain=5.0,
        last_checked_k=-1,
        direct_gain=0.0,
        X=[0],
        y=[1],
        nexisting=0,
        best_gain=-1.0e9,
        factors_data=data,
        factors_nbins=nbins,
        selected_vars=np.array([2], dtype=np.int64),
        max_veteranes_interactions_order=1,
        extra_knowledge_multipler=-1.0,
        entropy_cache=numba.typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64),
        cached_cond_MIs=numba.typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64),
        confidence_mode=True,
        max_confirmation_cand_nbins=max_confirmation_cand_nbins,
    )


def test_skipped_high_cardinality_factor_keeps_the_running_gain() -> None:
    """The factor is skipped (4 bins > cap 2): the gain stays at its start value instead of dropping to 0.0, and the factor still counts as visited."""
    _stopped, gain, k, _sink = _call(max_confirmation_cand_nbins=2)
    assert gain == 5.0
    assert k == 0


def test_checked_factor_still_lowers_the_gain() -> None:
    """Within the cap the redundancy is computed and pulls the minimum below the start value."""
    _stopped, gain, _k, _sink = _call(max_confirmation_cand_nbins=100)
    assert gain < 5.0
