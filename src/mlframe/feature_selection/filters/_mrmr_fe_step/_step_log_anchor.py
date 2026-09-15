"""Frozen ``smart_log`` shift anchor for a log-side FE operand, memoised within one materialise call."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def smart_log_anchor(src_name: str, nested: Any, X: Any, memo: dict) -> Optional[float]:
    """Return the fit-time ``smart_log`` shift for an operand, or ``None`` when the operand cannot be reconstructed.

    ``smart_log`` shifts non-positive inputs by ``1e-5 - nanmin(operand)``. The operand is the raw column ``src_name`` or, when ``nested`` is a
    recipe, that parent's continuous replay over ``X``. Both are fixed for one materialise call, and many sibling candidates share a parent,
    so the result is memoised on ``(src_name, id(nested))``; the memo keeps a reference to ``nested`` so its id cannot be reused meanwhile.
    """
    key = (src_name, None if nested is None else id(nested))
    hit = memo.get(key)
    if hit is not None:
        cached: Optional[float] = hit[1]
        return cached
    value: Optional[float]
    try:
        if nested is not None:
            from ..engineered_recipes import apply_recipe

            parent = nested
            if getattr(parent, "quantization", None) is not None:
                parent = dataclasses.replace(parent, quantization=None)
            operand = np.asarray(apply_recipe(parent, X), dtype=np.float64)
        else:
            col = X[src_name]
            operand = np.asarray(col.values if hasattr(col, "values") else col, dtype=np.float64)
        mn = float(np.nanmin(operand))
        value = (1e-5 - mn) if mn <= 0 else 0.0
    except Exception as e:
        logger.debug("smart_log_anchor: fit-time operand reconstruction failed, recipe replay falls back to the legacy refit path: %s", e)
        value = None
    memo[key] = (nested, value)
    return value
