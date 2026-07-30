"""
Pure re-export shim: ``_RecurrentWrapperBase`` / ``RecurrentClassifierWrapper`` /
``RecurrentRegressorWrapper`` (and the ``_monitor_mode`` EarlyStopping helper) now live in
``_recurrent_wrappers.py``; re-exported here so existing
``from mlframe.training.neural.recurrent_dataset_helpers import RecurrentClassifierWrapper``
callers -- and ``recurrent.py``'s own facade re-export -- keep working unchanged.
"""

from __future__ import annotations

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# X_EFFICIENCY_ARCHITECTURE-1 fix (mrmr_audit_2026-07-22): _RecurrentWrapperBase /
# RecurrentClassifierWrapper / RecurrentRegressorWrapper (plus the _monitor_mode helper and the
# constants they use) all now live in _recurrent_wrappers.py -- moved here wholesale rather than
# split, since a prior split (base class here, subclasses there) made this module import BACK from
# _recurrent_wrappers.py for the re-export below, creating a two-module import cycle
# (test_no_import_cycles). Re-exported so existing callers keep working.
from ._recurrent_wrappers import (  # noqa: F401
    _DEFAULT_SEQ_INPUT_SIZE,
    _MONITOR_MIN_KEYS,
    _MONITOR_MAX_KEYS,
    _monitor_mode,
    _RecurrentWrapperBase,
    RecurrentClassifierWrapper,
    RecurrentRegressorWrapper,
)
