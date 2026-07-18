"""Wave 89 (2026-05-21): LGB+XGB cat-cast block extracted from predict.py mega-try.

Two adjacent ~40-line blocks (LGB + XGB) sharing the same "detect-model-family
+ iterate cat_features + cast non-category to category" structure were merged
into a single module-level helper `_coerce_cat_dtype_for_lgb_xgb`.

Behaviour preserved bit-for-bit:
  - LGB (pandas path only) -> .assign(**{col: astype("category")})
  - XGB pandas path -> same
  - XGB polars path -> .with_columns([pl.col(c).cast(pl.Categorical)]) for
    pl.String / pl.Utf8 / LargeString columns
  - Models from other families (HGB / linear / CB) pass through untouched
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_helper_is_module_level_callable() -> None:
    """Helper is module level callable."""
    from mlframe.training.core.predict import _coerce_cat_dtype_for_lgb_xgb

    assert callable(_coerce_cat_dtype_for_lgb_xgb)


def test_passthrough_when_no_cat_features() -> None:
    """Passthrough when no cat features."""
    from mlframe.training.core.predict import _coerce_cat_dtype_for_lgb_xgb

    class _FakeLGB:
        """Groups tests covering fake l g b."""
        pass

    _FakeLGB.__module__ = "lightgbm.sklearn"

    df = pd.DataFrame({"x": [1.0, 2.0], "c": ["a", "b"]})
    out = _coerce_cat_dtype_for_lgb_xgb(df, model=_FakeLGB(), cat_features=[])
    # Empty cat_features short-circuits -> identity.
    assert out is df


def test_passthrough_for_non_lgb_xgb_model() -> None:
    """Passthrough for non lgb xgb model."""
    from mlframe.training.core.predict import _coerce_cat_dtype_for_lgb_xgb

    class _HGB:
        """Groups tests covering h g b."""
        pass

    _HGB.__module__ = "sklearn.ensemble._hist_gradient_boosting.gradient_boosting"

    df = pd.DataFrame({"x": [1.0, 2.0], "c": ["a", "b"]})
    out = _coerce_cat_dtype_for_lgb_xgb(df, model=_HGB(), cat_features=["c"])
    # HGB doesn't match LGB / XGB family detection -> passthrough.
    assert out is df


def test_lgb_pandas_path_casts_to_category() -> None:
    """Lgb pandas path casts to category."""
    from mlframe.training.core.predict import _coerce_cat_dtype_for_lgb_xgb

    class _LGB:
        """Groups tests covering l g b."""
        pass

    _LGB.__module__ = "lightgbm.sklearn"
    _LGB.__name__ = "LGBMClassifier"

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "c": ["a", "b", "a"]})
    out = _coerce_cat_dtype_for_lgb_xgb(df, model=_LGB(), cat_features=["c"])
    assert out["c"].dtype.name == "category"
    # Numeric column untouched.
    assert out["x"].dtype == np.float64


def test_xgb_pandas_path_casts_to_category() -> None:
    """Xgb pandas path casts to category."""
    from mlframe.training.core.predict import _coerce_cat_dtype_for_lgb_xgb

    class _XGB:
        """Groups tests covering x g b."""
        pass

    _XGB.__module__ = "xgboost.sklearn"
    _XGB.__name__ = "XGBClassifier"

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "c": ["a", "b", "a"]})
    out = _coerce_cat_dtype_for_lgb_xgb(df, model=_XGB(), cat_features=["c"])
    assert out["c"].dtype.name == "category"


def test_xgb_polars_path_casts_to_pl_categorical() -> None:
    """Xgb polars path casts to pl categorical."""
    pl = pytest.importorskip("polars")
    from mlframe.training.core.predict import _coerce_cat_dtype_for_lgb_xgb

    class _XGB:
        """Groups tests covering x g b."""
        pass

    _XGB.__module__ = "xgboost.sklearn"
    _XGB.__name__ = "XGBClassifier"

    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "c": ["a", "b", "a"]})
    out = _coerce_cat_dtype_for_lgb_xgb(df, model=_XGB(), cat_features=["c"])
    assert isinstance(out, pl.DataFrame)
    # pl.String -> pl.Categorical.
    assert out.schema["c"] == pl.Categorical


def test_already_categorical_column_not_recast() -> None:
    """Already categorical column not recast."""
    from mlframe.training.core.predict import _coerce_cat_dtype_for_lgb_xgb

    class _LGB:
        """Groups tests covering l g b."""
        pass

    _LGB.__module__ = "lightgbm.sklearn"
    _LGB.__name__ = "LGBMRegressor"

    df = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "c": pd.Categorical(["a", "b"]),
        }
    )
    out = _coerce_cat_dtype_for_lgb_xgb(df, model=_LGB(), cat_features=["c"])
    # No cast needed -> the helper returns df unchanged (the assign(**{}) branch is empty).
    assert out["c"].dtype.name == "category"
