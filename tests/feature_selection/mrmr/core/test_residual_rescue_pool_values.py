"""The residual rescue scores an engineered pool column on its continuous values, not on the bin codes the fit frame stores for it."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._residual_rescue_stage import _pool_values
from mlframe.feature_selection.filters._residual_rescue_fe import residual_rescue


def test_pool_value_of_an_engineered_column_is_its_replayed_continuous_values():
    """The stored ten-level codes are dropped by the scorer as nominal; the replayed continuous values are scored and admitted."""
    rng = np.random.default_rng(0)
    n = 6000
    c, d = rng.uniform(0.5, 9, n), rng.normal(size=n)
    term = np.log(c) * np.sin(d)
    codes = np.searchsorted(np.quantile(term, np.linspace(0, 1, 11)[1:-1]), term).astype(np.float64)  # what the fit frame stores: 10 levels
    X = pd.DataFrame({"c": c, "d": d, "mul(log(c),sin(d))": codes})

    got = _pool_values(X, "mul(log(c),sin(d))", {"mul(log(c),sin(d))": object()}, lambda recipe, frame: np.log(frame["c"].to_numpy()) * np.sin(frame["d"].to_numpy()))
    np.testing.assert_allclose(got, term)
    assert np.unique(got).size > 1000  # not a ten-level column

    y = 3.0 * term + 0.3 * rng.normal(size=n)
    assert residual_rescue({}, {"coded": codes}, y, min_relative_gain=0.001, max_admit=1) == []  # the stored codes are dropped as nominal
    assert residual_rescue({}, {"replayed": got}, y, min_relative_gain=0.001, max_admit=1)[0][0] == "replayed"


def test_a_raw_column_and_a_failed_replay_fall_back_to_the_stored_column():
    """A column without a recipe, and one whose replay raises, are read as stored."""
    X = pd.DataFrame({"a": np.arange(10.0), "e": np.arange(10.0) * 2})
    np.testing.assert_array_equal(_pool_values(X, "a", {}, lambda r, f: None), np.arange(10.0))

    def boom(recipe, frame):
        """A recipe replay that fails."""
        raise ValueError("replay failed")

    np.testing.assert_array_equal(_pool_values(X, "e", {"e": object()}, boom), np.arange(10.0) * 2)
