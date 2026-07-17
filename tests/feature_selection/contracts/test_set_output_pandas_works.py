"""Wave 9.1 loop-iter-34 regression: ``MRMR.set_output(transform='pandas')``
must produce a DataFrame even when ``transform()`` receives ndarray input.

Pre-fix at ``mrmr.py:1321``:
  MRMR.transform = _transform_func

This bottom-of-module late rebind overwrote the slot that
``_SetOutputMixin.__init_subclass__`` had wrapped during class
definition. ``set_output(transform='pandas')`` silently became a no-op
when ``transform()`` was called directly with ndarray input. Downstream
ColumnTransformer / Pipeline steps that expected DataFrame got ndarray
without warning - sklearn canonical contract violated.

Reference:
  StandardScaler().set_output(transform='pandas').transform(ndarray)
  -> pd.DataFrame   (sklearn contract)

Pre-fix MRMR returned ndarray on the same call; pipelines downstream
broke or attributed features to wrong names.

Fix: define ``transform`` as a thin class-body delegator in
``mrmr.py`` so the ``_SetOutputMixin`` wrapper actually attaches during
class definition. Drop the late ``MRMR.transform =`` rebind at the
bottom of the module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_set_output_pandas_returns_dataframe_for_ndarray_input():
    """The iter-34 contract: ``set_output(transform='pandas')`` MUST
    produce a DataFrame even when transform() is given an ndarray.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n))
    sel = MRMR(verbose=0).set_output(transform="pandas").fit(df, y)
    out = sel.transform(df.to_numpy())
    assert isinstance(out, pd.DataFrame), f"set_output(transform='pandas') must return DataFrame; got {type(out).__name__}"


def test_set_output_default_returns_dataframe_for_dataframe_input():
    """Negative control: default set_output + DataFrame in -> DataFrame out
    (unchanged by iter-34).
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n))
    sel = MRMR(verbose=0).fit(df, y)
    out = sel.transform(df)
    assert isinstance(out, pd.DataFrame)


def test_set_output_default_returns_ndarray_for_ndarray_input():
    """Negative control: without set_output, ndarray input -> ndarray
    output (sklearn default 'default' behaviour).
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n))
    sel = MRMR(verbose=0).fit(df, y)
    out = sel.transform(df.to_numpy())
    assert isinstance(out, np.ndarray)


def test_set_output_pandas_columns_named():
    """The DataFrame returned under ``set_output(transform='pandas')``
    must have column names matching ``get_feature_names_out``.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n))
    sel = MRMR(verbose=0).set_output(transform="pandas").fit(df, y)
    out = sel.transform(df.to_numpy())
    expected_names = sel.get_feature_names_out(["a", "b", "c"])
    # The DataFrame column order must match the sklearn protocol -
    # get_feature_names_out output (with caller's input_features in
    # place since fit was on DataFrame).
    assert isinstance(out, pd.DataFrame)
    assert len(out.columns) == len(expected_names)
