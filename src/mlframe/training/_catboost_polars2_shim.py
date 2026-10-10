"""Let CatBoost build a Pool from a polars frame with Categorical columns on polars 2.

CatBoost's compiled Pool constructor reads the labels of a polars Categorical column through ``Series.cat.get_categories()``, which polars 2
removed; it raises ``AttributeRemovedError`` before any training starts. Enum and String columns do not take that path, so the Categorical
columns of the frame are declared as Enum over their own values (the same remap the pandas bridge applies) before the Pool is built.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MARKER = "_mlframe_polars2_categorical_shim"


def polars_lacks_get_categories() -> bool:
    """True when the installed polars no longer has ``Series.cat.get_categories`` (polars 2)."""
    try:
        import polars as pl
    except ImportError:
        return False
    return not hasattr(pl.Series(["a"], dtype=pl.Categorical).cat, "get_categories")


def install_catboost_polars2_shim() -> bool:
    """Wrap ``catboost.Pool.__init__`` once so a polars frame passed as ``data`` has its Categorical columns turned into Enum; returns whether it is installed."""
    if not polars_lacks_get_categories():
        return False
    try:
        import catboost
        import polars as pl
    except ImportError:
        return False
    pool_cls = getattr(catboost, "Pool", None)
    if pool_cls is None:
        return False
    if pool_cls.__dict__.get(_MARKER, False):
        return True
    orig_init = pool_cls.__init__

    def _init_with_enum_columns(self: Any, data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Pool constructor that first remaps the Categorical columns of a polars ``data`` frame."""
        if isinstance(data, pl.DataFrame):
            from mlframe.training.utils import _remap_polars_categoricals

            data = _remap_polars_categoricals(data)
        orig_init(self, data, *args, **kwargs)

    _init_with_enum_columns.__wrapped__ = orig_init  # type: ignore[attr-defined]
    pool_cls.__init__ = _init_with_enum_columns
    setattr(pool_cls, _MARKER, True)
    return True
