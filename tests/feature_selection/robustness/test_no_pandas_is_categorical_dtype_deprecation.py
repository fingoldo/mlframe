"""Regression sensors: FE cat-detection helpers must not call the deprecated ``pd.api.types.is_categorical_dtype``.

pandas 2.1+ emits a ``DeprecationWarning`` from that function; the helpers now use ``isinstance(dtype, pd.CategoricalDtype)``.
Each test feeds a frame carrying a genuine ``category`` column so the categorical branch is exercised, and asserts no
``is_categorical_dtype`` deprecation surfaces.
"""

import warnings

import numpy as np
import pandas as pd
import pytest


def _assert_no_is_categorical_deprecation(fn):
    """Assert no is categorical deprecation."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        try:
            fn()
        except DeprecationWarning as exc:  # pragma: no cover - failure path
            if "is_categorical_dtype" in str(exc):
                pytest.fail(f"deprecated pandas is_categorical_dtype still called: {exc}")
            raise


def _cat_frame(n: int = 200) -> pd.DataFrame:
    """Cat frame."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "c": pd.Series(rng.integers(0, 6, n)).astype("category"),
            "num": rng.standard_normal(n),
        }
    )


def test_auto_detect_te_cols_no_is_categorical_deprecation():
    """Auto detect te cols no is categorical deprecation."""
    from mlframe.feature_selection.filters._target_encoding_fe import auto_detect_te_cols

    X = _cat_frame()
    _assert_no_is_categorical_deprecation(lambda: auto_detect_te_cols(X))


def test_auto_detect_cat_pair_cols_no_is_categorical_deprecation():
    """Auto detect cat pair cols no is categorical deprecation."""
    from mlframe.feature_selection.filters._cat_pair_fe import auto_detect_cat_pair_cols

    X = _cat_frame()
    _assert_no_is_categorical_deprecation(lambda: auto_detect_cat_pair_cols(X))


def test_boruta_shap_encode_categoricals_no_is_categorical_deprecation():
    """Boruta shap encode categoricals no is categorical deprecation."""
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X = _cat_frame()
    _assert_no_is_categorical_deprecation(lambda: BorutaShap._ordinal_encode_object_cols_inplace(X.copy()))
