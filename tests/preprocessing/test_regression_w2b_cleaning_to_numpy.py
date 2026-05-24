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
