"""Carrier-frame plumbing for PDP/ICE: keeping a manufactured frame's dtypes acceptable to the fitted model.

Carved out of ``pdp_ice.py``, which had grown past the house carve band. A partial-dependence sweep has to hand
the model rows it never saw, built by substituting one column of a real frame -- and the substitution is where
the model's own feature declarations bite: a categorical column has to stay categorical, a text column has to
stay text, and a pandas frame that loses either dtype makes the model reject or, worse, misread the batch. That
plumbing is a separate concern from the sweep itself and is what this module holds.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

def _model_text_feature_names(model: Any, carrier_columns: Optional[List[str]]) -> set:
    """Column names the model registered as native TEXT features (CatBoost's ``text_features``), so
    ``_carrier_with_categoricals`` can leave them alone.

    A CatBoost model with ``text_features=[...]`` rejects a 'category'-dtype column for one of those names at
    Pool-build time ("has dtype 'category' but is not in cat_features list") -- or, on a polars carrier, crashes
    the process with a native access violation instead of raising cleanly (caught live via a fuzz combo whose PDP
    diagnostic swept a numeric feature while a sibling text column was present). ``_carrier_with_categoricals``
    used to blanket-cast every non-numeric object column to category, which is exactly the wrong dtype for a
    text-feature column. Unwraps a Pipeline's final estimator the same way ``_training_loop.py`` does.
    """
    inner = model
    steps = getattr(model, "steps", None)
    if steps:
        inner = steps[-1][1]
    get_text = getattr(inner, "get_text_feature_indices", None)
    if not callable(get_text):
        return set()
    try:
        idx = get_text()
    except Exception as exc:
        logger.debug("text-feature index probe failed, treating as no text features: %s", exc)
        return set()
    if not idx:
        return set()
    names = getattr(inner, "feature_names_", None) or carrier_columns
    if not names:
        return set()
    return {names[i] for i in idx if 0 <= i < len(names)}


def _carrier_with_categoricals(carrier: Any, model: Any = None) -> Any:
    """Cast a pandas carrier's object/string columns to 'category' (new frame via assign -- untouched blocks reused,
    caller frame not mutated) so a categorical model can predict on it. Non-pandas / already-numeric-or-category
    carriers are returned unchanged."""
    try:
        import pandas as pd
    except ImportError:
        return carrier
    if not isinstance(carrier, pd.DataFrame):
        return carrier
    obj_cols = [c for c in carrier.columns if not (carrier[c].dtype.kind in "iufb" or isinstance(carrier[c].dtype, pd.CategoricalDtype))]
    # An object column holding non-scalar elements (e.g. a materialized embedding column reaching pandas as a
    # list per row) makes pandas' Categorical factorize() raise "unhashable type: 'numpy.ndarray'" -- lists/arrays
    # can't hash into the category-uniquing table. Drop such columns from the cast set (a single embedding vector
    # isn't a meaningful category anyway); they stay their original object dtype and the model call downstream
    # either handles them natively or fails on its own terms, same as any other unsupported dtype.
    obj_cols = [c for c in obj_cols if not any(isinstance(v, (list, tuple, np.ndarray)) for v in carrier[c])]
    text_cols = _model_text_feature_names(model, list(carrier.columns)) if model is not None else set()
    obj_cols = [c for c in obj_cols if c not in text_cols]
    return carrier.assign(**{c: carrier[c].astype("category") for c in obj_cols}) if obj_cols else carrier


def _categorical_grid(carrier: Any, col_name: Optional[str]) -> Tuple[Optional[list], Any]:
    """If ``col_name`` is a categorical column of the (pandas / polars) ``carrier``, return ``(category_labels,
    dtype)`` so the sweep can iterate the NATIVE categories and substitute native labels; else ``(None, None)``.

    Sweeping a numeric grid value into a categorical column produces an invalid model input (CatBoost:
    "cat_features must be integer or string ... =0.0" -- an outright error at Pool build for a string-category
    column, and a native-predict hang for an int-coded one). The labels come straight from the carrier's own
    category set (no float-code round-trip), so the substituted value is always a value the model saw at fit time.
    """
    if col_name is None or isinstance(carrier, np.ndarray):
        return None, None
    try:
        import pandas as pd
        if isinstance(carrier, pd.DataFrame):
            if col_name in carrier.columns and isinstance(carrier[col_name].dtype, pd.CategoricalDtype):
                return list(carrier[col_name].cat.categories), carrier[col_name].dtype
            return None, None
    except ImportError:
        pass
    if type(carrier).__module__.startswith("polars"):
        import polars as pl
        dt = carrier.schema.get(col_name) if hasattr(carrier, "schema") else None
        is_cat = dt is not None and (dt == pl.Categorical or (hasattr(pl, "Enum") and isinstance(dt, pl.Enum)))
        if is_cat and dt is not None:
            labels = carrier[col_name].cat.get_categories().to_list() if dt == pl.Categorical else list(dt.categories)
            return labels, dt
    return None, None


def _substitute_column(carrier_sample: Any, base_vals: Optional[np.ndarray], col_idx: int, value: Any,
                       col_name: Optional[str] = None, categorical_dtype: Any = None) -> Any:
    """Return a model-input block with column ``col_idx`` set to ``value`` for every row.

    For a pandas / polars ``carrier_sample`` (already the native-dtype subsampled rows), set the swept column to
    ``value`` while PRESERVING every other column's dtype (so categorical models predict) -- via ``assign`` /
    ``with_columns`` on the small (sample, n_cols) subsample, never the caller's full frame. When
    ``categorical_dtype`` is supplied the swept column is itself categorical: ``value`` is a native category label
    and is assigned back as that categorical dtype (never a bare float, which breaks categorical model predict).
    For an ndarray carrier the float ``base_vals`` block path is exact and kept.
    """
    if isinstance(carrier_sample, np.ndarray):
        assert base_vals is not None
        block = base_vals.copy()
        block[:, col_idx] = value
        return block
    if hasattr(carrier_sample, "assign"):  # pandas subsample
        import pandas as pd
        name = col_name if col_name is not None else list(carrier_sample.columns)[col_idx]
        if categorical_dtype is not None:
            arr = ([value] * len(carrier_sample)) if np.ndim(value) == 0 else list(value)
            return carrier_sample.assign(**{name: pd.Categorical(arr, dtype=categorical_dtype)})
        return carrier_sample.assign(**{name: value})
    mod = type(carrier_sample).__module__
    if mod.startswith("polars"):  # polars subsample
        import polars as pl
        name = col_name if col_name is not None else carrier_sample.columns[col_idx]
        if categorical_dtype is not None:
            expr = (pl.lit(value) if np.ndim(value) == 0 else pl.Series(name, list(value))).cast(categorical_dtype)
            return carrier_sample.with_columns(expr.alias(name))
        return carrier_sample.with_columns(pl.lit(value).alias(name))
    assert base_vals is not None
    block = base_vals.copy()
    block[:, col_idx] = value
    return block


__all__ = []
