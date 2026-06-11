"""mlframe.feature_selection.filters._mrmr_fit_impl package facade.

Promoted from a ~7.5k-line flat module to a subpackage. The single giant
_fit_impl body (bound onto MRMR at the mrmr package facade via
from .._mrmr_fit_impl import _fit_impl) lives in _fit_impl_core.py;
the few small free helpers live in _helpers.py. This __init__ re-exports
every name external importers / tests pull from _mrmr_fit_impl so the split
is fully backward-compatible.
"""
from __future__ import annotations

from ._fit_impl_core import _fit_impl
from ._helpers import (
    _dispatch_default_scorer,
    _mrmr_cache_bytes_total,
    _mrmr_instance_state_size_bytes,
    _orth_fe_numeric_cols,
)

__all__ = [
    "_fit_impl",
    "_dispatch_default_scorer",
    "_mrmr_cache_bytes_total",
    "_mrmr_instance_state_size_bytes",
    "_orth_fe_numeric_cols",
]
