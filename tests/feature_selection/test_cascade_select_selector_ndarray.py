"""CascadeSelectSelector on a bare ndarray must select the same columns as on the equivalent DataFrame.

The adapter wraps an ndarray in a frame with synthetic ``x{i}`` names for ``cascade_select``, which then returns
those NAMES, while ``_finalize`` resolves an ndarray fit's selection by POSITION. The names went straight into
``mask[list(selected)]`` and fit raised ``IndexError: only integers, slices ...``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.functional_adapters import CascadeSelectSelector


def _xy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] + 0.8 * X[:, 2] + 0.3 * rng.normal(size=300) > 0).astype(int)
    return X, y


def test_ndarray_fit_selects_same_positions_as_dataframe_fit():
    X, y = _xy()
    kw = dict(n_boruta_iterations=5, cv=3, random_state=0)

    sel_arr = CascadeSelectSelector(**kw).fit(X, y)
    sel_df = CascadeSelectSelector(**kw).fit(pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])]), y)

    assert sel_arr.support_.dtype == bool and sel_arr.support_.shape == (X.shape[1],)
    assert sel_arr.support_.any(), "ndarray fit selected nothing"
    np.testing.assert_array_equal(sel_arr.support_, sel_df.support_)
    assert list(sel_arr.get_feature_names_out()) == list(sel_df.get_feature_names_out())
    assert sel_arr.transform(X).shape[1] == int(sel_arr.support_.sum())
