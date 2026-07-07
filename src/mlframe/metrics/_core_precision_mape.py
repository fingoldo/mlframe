"""Precision / classification report + MAPE kernels for ``mlframe.metrics.core``.

Carved from ``core.py``. Public symbols are re-exported from the parent.
"""

from __future__ import annotations

import logging
from typing import Tuple

from numba import get_num_threads, get_thread_id, njit, prange  # type: ignore[attr-defined]  # numba ships no type stubs for its dynamic exports
import numpy as np

from ._numba_params import NUMBA_NJIT_PARAMS

logger = logging.getLogger(__name__)


@njit(**NUMBA_NJIT_PARAMS)
def fast_precision(y_true: np.ndarray, y_pred: np.ndarray, nclasses: int = 2, zero_division: int = 0) -> float:
    """Numba-fast precision of the last (highest-index) class: ``hits / predicted`` for that class.

    Counts predictions per class and how many were correct, returning the precision of class
    ``nclasses - 1`` (the positive class in the binary case). Out-of-range predicted labels are ignored.
    """
    # storage inits
    allpreds = np.zeros(nclasses, dtype=np.int64)
    hits = np.zeros(nclasses, dtype=np.int64)
    # count stats
    for true_class, predicted_class in zip(y_true, y_pred):
        if 0 <= predicted_class < nclasses:
            allpreds[predicted_class] += 1
            if predicted_class == true_class:
                hits[predicted_class] += 1
    precisions = np.zeros(nclasses, dtype=np.float64)
    for c in range(nclasses):
        if allpreds[c] > 0:
            precisions[c] = hits[c] / allpreds[c]
    return float(precisions[-1])


@njit(**NUMBA_NJIT_PARAMS)
def fast_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, nclasses: int = 2, zero_division: int = 0, macro_over_present: bool = True
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Custom classification report, proof of concept.

    ``macro_over_present`` (default True) averages macro precision/recall/F1 only over classes that appear in
    ``y_true`` OR ``y_pred`` -- matching ``sklearn.metrics.classification_report``, whose label set is the union of
    present classes. A class declared in ``nclasses`` but absent from both arrays would otherwise contribute a
    zeroed P/R/F1 and DEFLATE the macro averages (the same defect ``balanced_accuracy`` was already corrected for).
    Set ``macro_over_present=False`` for the legacy divide-by-``nclasses`` behaviour."""

    N_AVG_ARRAYS = 3  # precisions, recalls, f1s

    # storage inits
    weighted_averages = np.empty(N_AVG_ARRAYS, dtype=np.float64)
    macro_averages = np.empty(N_AVG_ARRAYS, dtype=np.float64)
    supports = np.zeros(nclasses, dtype=np.int64)
    allpreds = np.zeros(nclasses, dtype=np.int64)
    misses = np.zeros(nclasses, dtype=np.int64)
    hits = np.zeros(nclasses, dtype=np.int64)

    # count stats
    for true_class, predicted_class in zip(y_true, y_pred):
        # Bounds-check both labels: out-of-range values are silently dropped rather than
        # triggering a numba-level buffer overflow / segfault.
        if 0 <= true_class < nclasses:
            supports[true_class] += 1
        if 0 <= predicted_class < nclasses:
            allpreds[predicted_class] += 1
            if predicted_class == true_class:
                hits[predicted_class] += 1
            else:
                misses[predicted_class] += 1

    # main calcs
    accuracy = hits.sum() / len(y_true)

    # Balanced accuracy: classes absent from y_true (supports==0) are EXCLUDED from
    # the mean rather than contributing zero_division. sklearn.metrics.balanced_accuracy_score
    # computes mean recall over present classes only - matching that semantics.
    present_mask = supports > 0
    if present_mask.any():
        per_class_recall = np.empty(nclasses, dtype=np.float64)
        for c in range(nclasses):
            per_class_recall[c] = hits[c] / supports[c] if supports[c] > 0 else 0.0
        balanced_accuracy = per_class_recall[present_mask].mean()
    else:
        balanced_accuracy = 0.0

    recalls = np.zeros(nclasses, dtype=np.float64)
    precisions = np.zeros(nclasses, dtype=np.float64)
    f1s = np.zeros(nclasses, dtype=np.float64)
    for c in range(nclasses):
        if supports[c] > 0:
            recalls[c] = hits[c] / supports[c]
        if allpreds[c] > 0:
            precisions[c] = hits[c] / allpreds[c]
        pr_denom = precisions[c] + recalls[c]
        if pr_denom > 0:
            f1s[c] = 2.0 * (precisions[c] * recalls[c]) / pr_denom

    # Weighted averages must divide by supports.sum() (== number of labeled samples with
    # in-range class ids), NOT len(y_true): out-of-range labels were dropped above, so
    # dividing by the raw length under-reports the weighted mean proportionally to the
    # OOB fraction.
    support_total = supports.sum()
    weight_denom = support_total if support_total > 0 else 1

    # Macro denominator: classes present in y_true OR y_pred (sklearn classification_report semantics) vs the
    # legacy divide-by-nclasses. Absent declared classes (neither labeled nor predicted) carry zeroed P/R/F1 that
    # would deflate the macro mean -- the same exclusion already applied to balanced_accuracy above.
    if macro_over_present:
        present_macro = (supports > 0) | (allpreds > 0)
        macro_count = int(present_macro.sum())
    else:
        present_macro = np.ones(nclasses, dtype=np.bool_)
        macro_count = nclasses

    # fix nans & compute averages
    i = 0
    for arr in (precisions, recalls, f1s):
        np.nan_to_num(arr, copy=False, nan=zero_division)
        weighted_averages[i] = (arr * supports).sum() / weight_denom
        if macro_count > 0:
            macro_averages[i] = (arr * present_macro).sum() / macro_count
        else:
            macro_averages[i] = 0.0
        i += 1

    return hits, misses, accuracy, balanced_accuracy, supports, precisions, recalls, f1s, macro_averages, weighted_averages


@njit(**NUMBA_NJIT_PARAMS)
def _max_abs_pct_error_kernel(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, int, int]:
    """Returns (max MAPE value, count of y_true==0 entries, count of non-finite pairs).

    The zero-count is surfaced so the Python wrapper can emit a warning - silently
    swallowing y_true==0 hides the fact that the epsilon fallback dominates the ratio
    and the "percentage" becomes meaningless.

    A non-finite y_true/y_pred makes that row's error unknown; ``np.nanmax`` would
    silently drop it and report a misleadingly-finite max. We count such rows so the
    wrapper can propagate NaN (matching smape/wmape/mdape/pinball which all return NaN
    on non-finite input) instead of masking a corrupt row as a clean score.
    """
    epsilon = np.finfo(np.float64).eps
    n_zero = 0
    n_nonfinite = 0
    for i in range(len(y_true)):
        if y_true[i] == 0.0:
            n_zero += 1
        if not (np.isfinite(y_true[i]) and np.isfinite(y_pred[i])):
            n_nonfinite += 1
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    return np.nanmax(mape), n_zero, n_nonfinite


@njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _max_abs_pct_error_kernel_par(y_true: np.ndarray, y_pred: np.ndarray, nthr: int) -> Tuple[float, int, int]:
    """Parallel variant. ~2.3x faster than seq at N=1M.

    NOTE: ``if err > max_err: max_err = err`` inside ``prange`` is a
    race -- numba auto-detects ``+=`` as a reduction but NOT if-based
    max-update; concurrent threads can drop max-updates. Solution:
    per-thread max array + final reduction outside the prange.

    ``nthr`` is passed in (rather than called via numba.get_num_threads
    inside the kernel) so the @njit-cache can persist across runs.
    get_num_threads is a ctypes call that triggers the NumbaWarning
    "Cannot cache compiled function as it uses dynamic globals".
    """
    n = len(y_true)
    epsilon = np.finfo(np.float64).eps
    per_thread_max = np.zeros(nthr, dtype=np.float64)
    n_zero = 0
    n_nonfinite = 0
    for i in prange(n):
        if y_true[i] == 0.0:
            n_zero += 1
        if not (np.isfinite(y_true[i]) and np.isfinite(y_pred[i])):
            n_nonfinite += 1
        denom = abs(y_true[i])
        if denom < epsilon:
            denom = epsilon
        err = abs(y_pred[i] - y_true[i]) / denom
        # NaN guard: np.nanmax in seq variant skips NaNs; mirror that.
        if err == err:  # not NaN
            tid = get_thread_id()
            if err > per_thread_max[tid]:
                per_thread_max[tid] = err
    max_err = 0.0
    for t in range(nthr):
        if per_thread_max[t] > max_err:
            max_err = per_thread_max[t]
    return max_err, n_zero, n_nonfinite


# Module-level set: (n_zero, n_total) tuples for which the
# ``maximum_absolute_percentage_error: N of M y_true entries are zero``
# warning has already fired this process. Auto-cleared by interpreter
# shutdown. NOT thread-safe but the worst case is a duplicate warning in
# a rare race - the correctness signal is preserved.
_MAPE_ZERO_WARN_SEEN: set = set()


def maximum_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Worst-case absolute percentage error: ``max_i |y_pred_i - y_true_i| / max(|y_true_i|, eps)``.

    Auto-dispatches between serial and parallel numba kernels by input size. Returns NaN if any
    row is non-finite; warns (once per shape) when ``y_true`` contains zeros, since the epsilon
    fallback then dominates and the percentage becomes unreliable.
    """
    # Auto seq/par dispatch. Parallel only wins at large N (race-free
    # max via per-thread accumulator + final reduction; lose-band runs
    # to ~200k due to setup cost).
    if len(y_true) >= 500_000:
        value, n_zero, n_nonfinite = _max_abs_pct_error_kernel_par(y_true, y_pred, get_num_threads())
    else:
        value, n_zero, n_nonfinite = _max_abs_pct_error_kernel(y_true, y_pred)
    # A non-finite y_true/y_pred row makes the max error unknown; nanmax silently drops
    # it and returns a finite value that looks like a clean score. Propagate NaN instead.
    if n_nonfinite > 0:
        return float(np.nan)
    if n_zero > 0:
        # Rate-limit: emit the warning once per (n_zero, n_total) shape per
        # process. The metric is computed on train/val/test/OOF splits and
        # often by the per-feature ablation loop in BaselineDiagnostics, so
        # the same warning fires 4-15 times per training run with identical
        # content. Once is enough to alert the user that MAPE is mathematically
        # unreliable on their target; the rest is noise.
        _key = (int(n_zero), int(len(y_true)))
        if _key not in _MAPE_ZERO_WARN_SEEN:
            _MAPE_ZERO_WARN_SEEN.add(_key)
            logger.warning(
                "maximum_absolute_percentage_error: %d of %d y_true entries are zero; "
                "the epsilon fallback makes those ratios dominate the result. "
                "(further identical warnings suppressed this process)",
                n_zero, len(y_true),
            )
    return float(value)
