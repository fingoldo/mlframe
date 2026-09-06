"""One column-name read for the polars/pandas frames the composite path accepts.

Three modules carried their own ``_frame_columns``. Two returned a list; the third returned a SET, because
its caller does membership tests and wants them O(1). That difference is real, so the shared helper returns
the list and the set-wanting caller wraps it -- collapsing all three into one return type would have changed
either the ordering the list callers rely on or the membership cost the set caller was written for.

What is worth sharing is the read itself: ``getattr(df, "columns", None)`` and the decision that a
frame-like object exposing no ``columns`` yields empty rather than raising.
"""

from __future__ import annotations

from typing import Any, List


def frame_columns(df: Any) -> List[str]:
    """Column names for a polars or pandas frame, or ``[]`` for a frame-like object without ``columns``."""
    cols = getattr(df, "columns", None)
    if cols is None:
        return []
    # polars ``.columns`` is a list; pandas ``.columns`` is an Index -- both iterate to the column names.
    return list(cols)
