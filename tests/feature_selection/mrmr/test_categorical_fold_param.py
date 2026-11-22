"""The ``categorical_fold_strategy`` / ``categorical_fold_min_support`` constructor parameters: defaults, validation, cloning and the value that reaches ``categorize_dataset``."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from mlframe.feature_selection.filters.mrmr import MRMR


def test_defaults_keep_the_frequency_cap_only():
    """The default strategy is ``frequency`` with a support of 5 (unused until ``min_support`` is chosen); both survive ``clone`` and ``get_params``."""
    m = MRMR()
    assert m.categorical_fold_strategy == "frequency" and m.categorical_fold_min_support == 5
    c = clone(MRMR(categorical_fold_strategy="min_support", categorical_fold_min_support=8))
    assert c.get_params()["categorical_fold_strategy"] == "min_support" and c.get_params()["categorical_fold_min_support"] == 8


@pytest.mark.parametrize("kwargs", [dict(categorical_fold_strategy="target_stat"), dict(categorical_fold_strategy=None), dict(categorical_fold_min_support=1), dict(categorical_fold_min_support=2.5)])
def test_bad_values_fail_at_fit_start(kwargs):
    """An unknown strategy, ``None``, a support below 2 or a non-integer support is a ``ValueError`` before any work."""
    X = pd.DataFrame({"a": np.arange(60, dtype=float), "b": np.arange(60)[::-1].astype(float)})
    y = (np.arange(60) > 30).astype(int)
    with pytest.raises(ValueError):
        MRMR(verbose=0, **kwargs).fit(X, y)


@pytest.mark.parametrize("strategy,expected", [("frequency", None), ("min_support", 7)])
def test_categorize_dataset_receives_the_support_only_under_min_support(monkeypatch, strategy, expected):
    """``frequency`` hands ``None`` (the cap alone, bit for bit as before); ``min_support`` hands the configured support."""
    import mlframe.feature_selection.filters.mrmr as M

    seen = []
    real = M.categorize_dataset

    def spy(*args, **kwargs):
        """Record the fold support of every call."""
        seen.append(kwargs.get("categorical_fold_min_support"))
        return real(*args, **kwargs)

    monkeypatch.setattr(M, "categorize_dataset", spy)
    rng = np.random.default_rng(0)
    n = 400
    x = rng.standard_normal(n)
    df = pd.DataFrame({"x": x, "cat": pd.Series(rng.integers(0, 4, n).astype(str)).astype("category")})
    y = (x + 0.3 * rng.standard_normal(n) > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        MRMR(verbose=0, random_seed=0, fe_max_steps=0, categorical_fold_strategy=strategy, categorical_fold_min_support=7).fit(df, y)
    assert seen and set(seen) == {expected}, seen
