"""NaN-safe wrappers for the recurring "argmax/argmin/quantile on a
NaN-bearing array silently picks the wrong thing" bug class.

The wave-21 audit (2026-05-20) found that mlframe had no central
``safe_argmax`` / ``safe_quantile`` helper, and the codebase was
splitting roughly 32 nan-aware calls (good) against ~200 non-nan-aware
calls (silent-bug surface). This module concentrates the safe path so
new code can ``from mlframe.utils.nan_safe import argmax_classes_safe``
and stop reinventing the finite-mask boilerplate.

Helpers here are intentionally THIN -- they preserve the caller's
expected return shape, only adding:
1. A finite-mask check.
2. A WARN log (capped by row count, not per-row) so operators see when
   the safe-path actually fired.
3. A loud fallback when the entire input is NaN.

Public surface:
- ``argmax_classes_safe(probs, *, fallback_class=0, logger=None)``: per-row
  argmax for a (N, K) probability matrix; NaN rows -> ``fallback_class``
  with a single aggregate WARN.
- ``quantile_safe(arr, q, **kwargs)``: ``np.nanquantile`` with an
  all-NaN guard that returns a documented sentinel + WARN.
- ``median_safe(arr, **kwargs)``: thin ``np.nanmedian`` wrapper.

This file ships with sensors covering each helper; see
``tests/training/test_nan_propagation_fixes.py`` for the broader wave-21
contract.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

_DEFAULT_LOGGER = logging.getLogger(__name__)


def argmax_classes_safe(
    probs: np.ndarray,
    *,
    fallback_class: int = 0,
    logger: Optional[logging.Logger] = None,
    context: str = "argmax_classes_safe",
) -> np.ndarray:
    """Per-row argmax for a probability matrix, safe against NaN rows.

    Args:
        probs: (N,) or (N, K) array of probabilities (or scores). When 2D,
            argmax is taken across axis=1.
        fallback_class: class index assigned to rows that are entirely NaN.
            Default 0. Operators should pick a sentinel that's obviously wrong
            in their context if class-0 collides.
        logger: optional logger; defaults to ``mlframe.utils.nan_safe`` logger.
        context: short tag included in the WARN message so the log line
            names the caller (e.g. ``"compute_probabilistic_multiclass_error"``).

    Returns:
        (N,) int64 array of class indices.

    Behavior:
        - All-finite input: equivalent to ``np.argmax(probs, axis=1)``.
        - Some NaN rows: those rows get ``fallback_class``; finite rows
          get ``np.argmax`` over the row's finite columns
          (via ``np.nanargmax``); one aggregate WARN per call naming
          the NaN-row count.
        - Mixed finite/NaN within a single row: ``np.nanargmax`` picks
          the index of the max finite value -- the most charitable
          interpretation when only some classes have valid probs.
    """
    log = logger or _DEFAULT_LOGGER
    if probs.ndim == 1:
        # Degenerate 1D case: just argmax with a finite-mask short-circuit.
        if np.all(np.isfinite(probs)):
            return np.asarray(np.argmax(probs))
        finite_mask = np.isfinite(probs)
        if not finite_mask.any():
            log.warning(
                "%s: 1-D probs are entirely non-finite; returning "
                "fallback_class=%d.", context, fallback_class,
            )
            return np.asarray(fallback_class, dtype=np.int64)
        return np.asarray(np.nanargmax(probs), dtype=np.int64)

    if probs.ndim != 2:
        raise ValueError(f"{context}: expected 1-D or 2-D probs; got shape={probs.shape}")

    n_rows, n_cols = probs.shape
    if n_rows == 0:
        return np.empty(0, dtype=np.int64)

    if np.all(np.isfinite(probs)):
        return np.asarray(np.argmax(probs, axis=1).astype(np.int64, copy=False))

    # Identify rows that have at least one finite entry; the rest go to fallback.
    row_has_finite = np.isfinite(probs).any(axis=1)
    n_dead = int(np.sum(~row_has_finite))
    if n_dead > 0:
        log.warning(
            "%s: %d/%d rows contain NO finite probabilities (all-NaN); "
            "assigning fallback_class=%d for those rows. If this is "
            "unexpected, check upstream model for predict_proba returning "
            "NaN (common shapes: division by zero in softmax, missing "
            "features, NaN-poisoned calibration).",
            context, n_dead, n_rows, fallback_class,
        )

    preds = np.full(n_rows, fallback_class, dtype=np.int64)
    if row_has_finite.any():
        # nanargmax on rows that have at least one finite entry doesn't raise.
        preds[row_has_finite] = np.nanargmax(probs[row_has_finite], axis=1)
    return preds


def quantile_safe(
    arr: np.ndarray,
    q,
    *,
    fallback: float = float("nan"),
    logger: Optional[logging.Logger] = None,
    context: str = "quantile_safe",
    **kwargs: Any,
):
    """``np.nanquantile`` with an explicit all-NaN guard.

    Returns the quantile (or array of quantiles) computed over finite
    values. Returns ``fallback`` (default NaN) + WARN log if the entire
    input is non-finite.

    Mirrors ``np.nanquantile`` semantics for ``q`` (scalar or sequence).
    Extra kwargs forwarded to ``np.nanquantile``.
    """
    log = logger or _DEFAULT_LOGGER
    arr_np = np.asarray(arr)
    if arr_np.size == 0 or not np.any(np.isfinite(arr_np)):
        log.warning(
            "%s: input has no finite values; returning fallback=%r.",
            context, fallback,
        )
        # Mirror nanquantile's output shape: array if q is sequence, scalar otherwise.
        if hasattr(q, "__iter__"):
            return np.full(len(list(q)), fallback, dtype=np.float64)
        return float(fallback)
    return np.nanquantile(arr_np, q, **kwargs)


def median_safe(
    arr: np.ndarray,
    *,
    fallback: float = float("nan"),
    logger: Optional[logging.Logger] = None,
    context: str = "median_safe",
    **kwargs: Any,
) -> float:
    """``np.nanmedian`` with an all-NaN guard.

    Returns the median over finite values, or ``fallback`` if the input
    has no finite values (with a WARN). Use over raw ``np.median`` when
    the array can carry NaN.
    """
    log = logger or _DEFAULT_LOGGER
    arr_np = np.asarray(arr)
    if arr_np.size == 0 or not np.any(np.isfinite(arr_np)):
        log.warning(
            "%s: input has no finite values; returning fallback=%r.",
            context, fallback,
        )
        return float(fallback)
    return float(np.nanmedian(arr_np, **kwargs))


__all__ = [
    "argmax_classes_safe",
    "quantile_safe",
    "median_safe",
]
