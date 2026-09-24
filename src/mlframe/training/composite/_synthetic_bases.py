"""Synthetic interaction bases: a base column named ``<a>__<op>__<b>`` computed from its two parent columns.

Interaction discovery finds products (and sums, differences) of two features whose MI with the target beats both parents,
and discovery screens them as bases like any other column. The column does not exist in the user's frames, so every reader
of a base by name resolves such a name here: when the frame has no column of that name but has both parents, the values are
recomputed from them - at training, in the suite's target build, at predict time in a fresh process and in serving - with
the same arithmetic the discovery used. A frame that does hold a column of that name always wins.

Only ``mul`` / ``add`` / ``sub`` are resolvable: they are pure functions of the two parents. ``div`` needs a divisor floor
fitted on the train rows, so a ``div`` interaction stays a reported candidate and is not turned into a base.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

SPECABLE_OPS = ("mul", "add", "sub")
_APPLY = {"mul": np.multiply, "add": np.add, "sub": np.subtract}


def _columns(frame: Any) -> list:
    """The frame's column names (pandas or polars), or an empty list for anything else."""
    cols = getattr(frame, "columns", None)
    return list(cols) if cols is not None else []


def parse_synthetic(name: str, columns: Any) -> Optional[tuple[str, str, str]]:
    """``(parent_a, op, parent_b)`` when ``name`` is ``<a>__<op>__<b>`` with both parents among ``columns``, else None.

    Parent names may themselves contain ``__``: every split point is tried, and the one whose two sides are both columns
    is taken.
    """
    cols = set(map(str, columns))
    for op in SPECABLE_OPS:
        token = f"__{op}__"
        start = 0
        while True:
            i = name.find(token, start)
            if i < 0:
                break
            a, b = name[:i], name[i + len(token):]
            if a in cols and b in cols:
                return a, op, b
            start = i + 1
    return None


def _parent(frame: Any, col: str, rows: Optional[np.ndarray]) -> np.ndarray:
    """One parent column as float64, optionally on ``rows`` only."""
    if hasattr(frame, "get_column"):  # polars
        s = frame.get_column(col)
        if rows is not None:
            s = s.gather(rows)
        return np.asarray(s.to_numpy(), dtype=np.float64)
    s = frame[col]
    if rows is not None:
        s = s.iloc[rows]
    return np.asarray(s.to_numpy(dtype=np.float64, na_value=np.nan) if hasattr(s, "to_numpy") else s, dtype=np.float64)


def synthetic_column(frame: Any, name: str, rows: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """The values of synthetic base ``name`` on ``frame`` (float64), or None when the frame has a real column of that name
    or cannot resolve it (not a synthetic name, or a parent missing)."""
    cols = _columns(frame)
    if name in cols:
        return None
    parsed = parse_synthetic(str(name), cols)
    if parsed is None:
        return None
    a, op, b = parsed
    return np.asarray(_APPLY[op](_parent(frame, a, rows), _parent(frame, b, rows)), dtype=np.float64)


def dropped_columns(base: str, columns: Any) -> list:
    """The feature columns that carry ``base`` and leave X when X is scored without the base.

    ``[base]`` for a real feature, the two parents for a synthetic base built from features, else ``[]`` (a unary spec's
    empty base, or a base that is not among the features).
    """
    if not base:
        return []
    cols = list(columns)
    if base in cols:
        return [base]
    parsed = parse_synthetic(str(base), cols)
    return [] if parsed is None else [parsed[0], parsed[2]]


def is_resolvable(frame: Any, name: str) -> bool:
    """True when ``frame`` holds ``name`` or both parents of the synthetic base ``name``."""
    cols = _columns(frame)
    return name in cols or parse_synthetic(str(name), cols) is not None


__all__ = ["SPECABLE_OPS", "dropped_columns", "is_resolvable", "parse_synthetic", "synthetic_column"]
