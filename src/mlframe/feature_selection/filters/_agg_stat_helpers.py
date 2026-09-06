"""One stat-name -> pandas-agg / global-fallback mapping, shared by the two grouped-aggregate FE modules.

``_composite_group_agg_fe`` and ``_grouped_agg_fe`` each carried their own ``_agg_func_for_stat`` and
``_global_value_for_stat``. The copies had already diverged in substance rather than only in wording: the
composite module supports a ``count`` stat and the grouped one does not, in BOTH functions. Nothing recorded
that difference as intentional, and nothing would have caught a fix landing in one copy only.

The supported set is therefore a PARAMETER, so each module keeps exactly the stats it supports today and the
difference is visible at the call site instead of buried in two nearly-identical bodies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def agg_func_for_stat(stat: str, valid: "tuple[str, ...]", label: str) -> str:
    """The pandas groupby-agg name for ``stat``, or raise naming ``label`` and the module's own valid set."""
    if stat in valid:
        return stat
    raise ValueError(f"{label}: unknown stat {stat!r}; valid: {valid}")


def global_value_for_stat(stat: str, values: np.ndarray, label: str) -> float:
    """The whole-column fallback for ``stat``, used where a group has no usable rows.

    Mirrors the per-group aggregate on the finite subset. ``count`` is supported here whether or not a given
    caller offers it, because refusing a stat the caller never asks for costs nothing while a missing branch
    silently returns the wrong fallback.
    """
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size == 0:
        return 0.0
    if stat == "mean":
        return float(np.mean(finite))
    if stat == "std":
        return float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    if stat == "min":
        return float(np.min(finite))
    if stat == "max":
        return float(np.max(finite))
    if stat == "median":
        return float(np.median(finite))
    if stat == "nunique":
        return float(np.unique(finite).size)
    if stat == "count":
        return float(finite.size)
    if stat == "skew":
        return float(pd.Series(finite).skew()) if finite.size > 2 else 0.0
    raise ValueError(f"{label}: unknown stat {stat!r}")
