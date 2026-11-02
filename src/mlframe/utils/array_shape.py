"""Element count and shape of an array, a lazy device-backed array or any sequence, without materialising a lazy one.

Lives under ``mlframe.utils`` so the low-level metrics modules can use it without importing the feature_selection package (which imports estimators, which
import the metrics: a cycle). ``mlframe.feature_selection.filters._lazy_host_codes`` re-exports both names.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["size_of", "shape_of"]


def size_of(values: Any) -> int:
    """Element count of an array, a lazy device-backed array or any sequence, without materialising a lazy one."""
    shape = getattr(values, "shape", None)
    if shape is not None:
        return int(np.prod(shape, dtype=np.int64))
    return int(np.asarray(values).size)  # length-ok: no shape attribute means a plain sequence


def shape_of(values: Any) -> tuple:
    """Shape of an array, a lazy device-backed array or any sequence, without materialising a lazy one (``np.asarray(x).shape`` copies it whole)."""
    shape = getattr(values, "shape", None)
    if shape is not None:
        return tuple(int(d) for d in shape)
    return np.asarray(values).shape  # length-ok: no shape attribute means a plain sequence
