"""Shared numba @njit kwargs and parallel-dispatch thresholds for ``mlframe.metrics``.

Single source of truth. ``core.py`` and the sibling modules split out of
it (``_calibration_plot.py``, ``_classification_report.py``,
``_regression_metrics.py``, ...) import the same constants from here so a
flag change lands once.

Thresholds are crossover row-counts above which the parallel @njit
variant beats the sequential one on an 8-thread numba runtime.
"""
from __future__ import annotations

NUMBA_NJIT_PARAMS = dict(fastmath=False, cache=True, nogil=True)

_PARALLEL_REDUCTION_THRESHOLD: int = 100_000
_PARALLEL_MULTILABEL_THRESHOLD: int = 50_000
