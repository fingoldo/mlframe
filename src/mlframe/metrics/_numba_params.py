"""Shared numba @njit kwargs and parallel-dispatch thresholds for ``mlframe.metrics``.

Single source of truth. ``core.py`` and the sibling modules split out of
it (``_calibration_plot.py``, ``_classification_report.py``,
``_regression_metrics.py``, ...) import the same constants from here so a
flag change lands once.

Thresholds are crossover row-counts above which the parallel @njit
variant beats the sequential one on an 8-thread numba runtime.
"""
from __future__ import annotations

from typing import Any

NUMBA_NJIT_PARAMS = dict(fastmath=False, cache=True, nogil=True)

_PARALLEL_REDUCTION_THRESHOLD: int = 100_000
_PARALLEL_MULTILABEL_THRESHOLD: int = 50_000


def _check_equal_length(y_true: Any, y_pred: Any) -> None:
    """Raise ``ValueError`` when the two arrays differ in leading-axis length.

    The fast_* numba kernels loop on ``len(y_true)`` and index ``y_pred[i]`` with
    bounds-checking off, so a length mismatch silently reads out-of-bounds garbage
    instead of erroring the way sklearn does. This shared guard restores the
    sklearn contract for every public wrapper that takes a (y_true, y_pred) pair.
    """
    n_true = len(y_true)
    n_pred = len(y_pred)
    if n_true != n_pred:
        raise ValueError(
            f"Found input variables with inconsistent numbers of samples: "
            f"y_true has {n_true}, y_pred/y_score has {n_pred}."
        )
