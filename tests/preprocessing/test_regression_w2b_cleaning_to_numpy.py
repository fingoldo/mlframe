"""Regression sensor for w2b-percol-scattered preprocessing/cleaning.py ``.values`` -> ``.to_numpy()`` (finding #38).

Behavioural smoke check: ``is_variable_truly_continuous`` still returns a sane verdict on a numeric column after the .values -> .to_numpy()
migration; the migration must not have changed the function's output semantics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_is_variable_truly_continuous_pandas_to_numpy_preserves_verdict():
    from mlframe.preprocessing.cleaning import is_variable_truly_continuous

    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": rng.normal(size=200)})
    res = is_variable_truly_continuous(df=df, variable_name="x")
    assert res is not None


def test_is_variable_truly_continuous_tz_aware_datetimearray_no_crash():
    """A tz-aware DatetimeArray input must not raise: the tz-strip path must hand a numpy datetime64 to the
    ``.astype("datetime64[h]")`` resolution probe, else pandas >=2.0 rejects casting a DatetimeArray to that dtype."""
    from mlframe.preprocessing.cleaning import is_variable_truly_continuous

    vals = pd.Series(pd.date_range("2020-01-01", periods=200, freq="h", tz="UTC")).array
    assert getattr(vals.dtype, "tz", None) is not None
    res = is_variable_truly_continuous(values=vals)
    assert res is not None
