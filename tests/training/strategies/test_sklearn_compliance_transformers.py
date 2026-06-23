"""sklearn-API-compliance regression tests for the strategy-layer transformers (SK6).

- ``_NumericOnlyTransformer.__init__`` must store ``cat_features`` VERBATIM (``None`` stays ``None``), so
  ``get_params`` / ``clone`` round-trip instead of silently turning None into ``[]``.
- All three transformers declare the ``allow_nan`` input tag, so an upstream ``check_array`` does not reject
  NaN/Inf input they are designed to pass through / convert.
"""

import numpy as np
import pytest

from sklearn.base import clone
from sklearn.impute import SimpleImputer

from mlframe.training.strategies.base import (
    _Float32CastTransformer,
    _InfToNaNTransformer,
    _NumericOnlyTransformer,
)


def _make(cls):
    if cls is _NumericOnlyTransformer:
        return cls(SimpleImputer(), None)
    return cls()


@pytest.mark.parametrize(
    "cls",
    [_Float32CastTransformer, _InfToNaNTransformer, _NumericOnlyTransformer],
    ids=["float32cast", "inf_to_nan", "numeric_only"],
)
def test_allow_nan_tag_is_set(cls):
    tags = _make(cls).__sklearn_tags__()
    assert tags.input_tags.allow_nan is True


def test_numeric_only_stores_cat_features_verbatim_and_clone_roundtrips():
    inner = SimpleImputer()
    t = _NumericOnlyTransformer(inner, None)
    # Verbatim: None must NOT become [].
    assert t.cat_features is None
    assert t.get_params(deep=False)["cat_features"] is None

    cloned = clone(t)
    assert cloned.get_params(deep=False)["cat_features"] is None


def test_numeric_only_clone_roundtrips_named_cats():
    t = _NumericOnlyTransformer(SimpleImputer(), ["a", "b"])
    cloned = clone(t)
    assert cloned.get_params(deep=False)["cat_features"] == ["a", "b"]
