"""Category labels of a polars Categorical column, across the polars versions mlframe supports.

``Series.cat.get_categories()`` was removed in polars 2.0 (the labels are now only reachable through the values), so the labels are rebuilt
from the physical codes there: each code maps to the one string it stands for, and a code no row uses maps to an empty label.
"""

from __future__ import annotations

from typing import Any, List

import polars as pl


def categorical_labels(series: Any) -> List[str]:
    """The labels of a Categorical series indexed by physical code (``labels[code]`` is the string that code stands for)."""
    getter = getattr(series.cat, "get_categories", None)
    if getter is not None:
        return [str(v) for v in getter().to_list()]
    pairs = pl.DataFrame({"code": series.to_physical(), "label": series.cast(pl.String)}).drop_nulls().unique()
    if pairs.height == 0:
        return []
    codes = pairs["code"].to_list()
    labels = [""] * (max(codes) + 1)
    for code, label in zip(codes, pairs["label"].to_list()):
        labels[code] = label
    return labels
