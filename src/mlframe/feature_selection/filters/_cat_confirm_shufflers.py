"""IPF and within-strata conditional shufflers for the permutation confirmation in ``_cat_confirm_permutation`` (re-imported there by name)."""
from __future__ import annotations


from ._numba_utils import (  # noqa: F401 -- helpers hosted in _numba_utils_cross_module for the cross-module njit invariant
    _conditional_shuffle_within_strata,
    _full_conditional_shuffle_ipf,
)
