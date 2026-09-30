"""Null-fill for a CatBoost model's categorical columns on a polars predict frame.

CatBoost 1.2.x's polars fastpath has no dispatch overload for a Categorical / Enum / String categorical column that carries a validity bitmap:
a single null raises ``TypeError: No matching signature found`` (Categorical), ``CatBoostError: Data with nulls is not supported`` (String), or
aborts the process (Enum). Numeric, Boolean and non-nullable categorical columns of every width are accepted, so the nullable categorical
columns are the only ones that need touching -- per column, never the whole frame.

The suite already fills these with ``__MISSING__`` before fit; a frame that reaches predict through another path (selector output, a reloaded
model, a split that gained nulls) still carries them, so the same fill runs at the predict boundary.
"""

from __future__ import annotations

import logging
from typing import Any

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

MISSING_SENTINEL = "__MISSING__"


def fill_nullable_cat_columns(model: Any, X: Any) -> Any:
    """``X`` with nulls in the model's categorical columns replaced by ``__MISSING__``; ``X`` itself when there is nothing to fill."""
    import polars as pl

    if not isinstance(X, pl.DataFrame):
        return X
    from .._predict_guards import _recover_cb_feature_names
    from ._cb_pool import _polars_fill_null_in_categorical, _polars_nullable_categorical_cols

    cat_features, _ = _recover_cb_feature_names(model)
    if not cat_features:
        return X
    nullable = _polars_nullable_categorical_cols(X, cat_features=cat_features)
    if not nullable:
        return X
    log_throttle(
        logger, "cb_predict_nullable_cat_fill", logging.INFO,
        "  [predict] filled nulls with %r in %d CatBoost categorical column(s) %s before predict: CatBoost's polars fastpath has no overload "
        "for a categorical column that carries nulls",
        MISSING_SENTINEL, len(nullable), nullable[:8],
    )
    return _polars_fill_null_in_categorical(X, nullable, sentinel=MISSING_SENTINEL)
