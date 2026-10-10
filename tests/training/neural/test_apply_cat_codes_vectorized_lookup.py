"""``_apply_cat_codes``'s categorical lookup switched from ``Series.map(dict)`` (one Python dict
``__getitem__`` per row) to ``Index.get_indexer`` (one C-level hash-table pass over the whole column) --
see the perf-win comment in ``_base_fit_prep.py``. Both the seen-category and unseen-category (fillna)
paths, and both object and pandas-categorical dtype, must produce identical codes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from mlframe.training.neural.base._base_fit_prep import _FitPrepMixin


class _Wrapper(_FitPrepMixin):
    """Minimal host for ``_apply_cat_codes``: only the fit-time state it reads."""

    def __init__(self, cat_cols, code_maps, cardinalities):
        """Store the fit-time factorization state the mixin's ``_apply_cat_codes`` reads."""
        self._cat_cols_ = cat_cols
        self._cat_code_maps_ = code_maps
        self._cat_cardinalities_ = cardinalities


def _legacy_apply(X, cat_cols, code_maps, cardinalities):
    """The old ``Series.map`` + ``fillna`` implementation, kept here only to pin bit-identity."""
    encoded_cols = {}
    for col, card in zip(cat_cols, cardinalities):
        mapping = code_maps[col]
        mapped = X[col].astype(object).map(mapping)
        encoded_cols[col] = mapped.fillna(float(card)).astype(np.float32)
    other_cols = [c for c in X.columns if c not in cat_cols]
    ordered = [c for c in cat_cols if c in X.columns] + other_cols
    return X.assign(**encoded_cols)[ordered]


def test_matches_legacy_map_implementation_with_unseen_values():
    """An unseen category (not in the fit-time map) must map to the reserved unknown code in both implementations."""
    df = pd.DataFrame({"cat1": ["mon", "tue", "wed", "zzz_unseen", "mon"], "num1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    code_map = {"mon": 0.0, "tue": 1.0, "wed": 2.0, "thu": 3.0}
    card = 4
    wrapper = _Wrapper(["cat1"], {"cat1": code_map}, [card])

    out = wrapper._apply_cat_codes(df)
    legacy = _legacy_apply(df, ["cat1"], {"cat1": code_map}, [card])

    assert np.array_equal(out["cat1"].to_numpy(), legacy["cat1"].to_numpy())
    assert out["cat1"].iloc[3] == card  # the unseen value gets the reserved unknown code


def test_matches_legacy_map_implementation_on_categorical_dtype():
    """A pandas ``category``-dtype column (the real embedding path's usual input) must match the legacy output too."""
    df = pd.DataFrame({"cat1": pd.Categorical(["a", "b", "c", "a"])})
    code_map = {"a": 0.0, "b": 1.0, "c": 2.0}
    card = 3
    wrapper = _Wrapper(["cat1"], {"cat1": code_map}, [card])

    out = wrapper._apply_cat_codes(df)
    legacy = _legacy_apply(df, ["cat1"], {"cat1": code_map}, [card])

    assert np.array_equal(out["cat1"].to_numpy(), legacy["cat1"].to_numpy())


def test_output_dtype_is_float32():
    """Codes are cast to float32 for the embedding layer, exactly as the legacy implementation did."""
    df = pd.DataFrame({"cat1": ["a", "b"]})
    wrapper = _Wrapper(["cat1"], {"cat1": {"a": 0.0, "b": 1.0}}, [2])
    out = wrapper._apply_cat_codes(df)
    assert out["cat1"].dtype == np.float32
