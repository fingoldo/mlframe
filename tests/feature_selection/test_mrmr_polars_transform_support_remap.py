"""Regression: MRMR.transform on a polars frame must remap ``support_`` to fit-time column NAMES, not index positionally.

Pre-fix, the polars-input branch did ``X[:, support_]`` positionally. When the transform-time polars frame is narrower
(or reordered) than the fit-time feature set -- e.g. a multi-model suite reuses one fitted MRMR after an upstream step
narrowed the frame -- a fit-time support index could exceed the input width and raise ``IndexError: index N is out of
bounds for sequence of length M`` (FUZZ_SEED=11, hgb, polars-utf8, n_rows=1000), or silently select the wrong columns.
The fix mirrors the pandas-by-name branch: support -> names, validate against the input, select by name.
"""

import numpy as np
import polars as pl
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _fitted_mrmr(names, support):
    m = MRMR()
    m.feature_names_in_ = np.array(names)
    m.n_features_in_ = len(names)
    m.support_ = np.asarray(support)
    m._engineered_recipes_ = []
    return m


def test_polars_transform_narrower_input_does_not_index_out_of_range():
    # Fit-time saw 6 cols; support picks positions 5 ("f") and 3 ("d"). The polars input at transform is narrower
    # but carries the selected columns by name. Pre-fix: IndexError (5 >= width 3). Post-fix: name-selects ['f','d'].
    m = _fitted_mrmr(["a", "b", "c", "d", "e", "f"], [5, 3])
    X = pl.DataFrame({"a": [1.0, 2, 3], "d": [4.0, 5, 6], "f": [7.0, 8, 9]})
    out = m.transform(X)
    assert isinstance(out, pl.DataFrame)
    assert out.columns == ["f", "d"]
    assert out["d"].to_list() == [4.0, 5, 6]
    assert out["f"].to_list() == [7.0, 8, 9]


def test_polars_transform_reordered_input_selects_by_name_not_position():
    # Same width as fit but columns reordered. Positional indexing would return the wrong columns; name-based is right.
    m = _fitted_mrmr(["a", "b", "c", "d", "e", "f"], [5, 3])
    X = pl.DataFrame({"f": [7.0, 8, 9], "e": [0.0, 0, 0], "d": [4.0, 5, 6],
                      "c": [0.0, 0, 0], "b": [0.0, 0, 0], "a": [1.0, 2, 3]})
    out = m.transform(X)
    assert out.columns == ["f", "d"]
    assert out["f"].to_list() == [7.0, 8, 9]
    assert out["d"].to_list() == [4.0, 5, 6]


def test_polars_transform_missing_selected_column_raises_actionable():
    m = _fitted_mrmr(["a", "b", "c", "d", "e", "f"], [5, 3])
    # Width 3 (not 2) so the all-selected identity fast-path (n_selected == X.shape[1]) does not fire and the
    # name-validation branch is reached; "d" is absent so the column-drift RuntimeError must raise.
    X = pl.DataFrame({"a": [1.0, 2, 3], "f": [7.0, 8, 9], "z": [0.0, 0, 0]})  # "d" absent
    with pytest.raises(RuntimeError, match="selected columns missing from input X"):
        m.transform(X)
