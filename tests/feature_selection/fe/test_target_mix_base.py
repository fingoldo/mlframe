"""The mix base is appended as a reserved column from the raw frame, and the fixed-beta regressor reads it from there and never refits beta."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from mlframe.feature_selection.filters._target_mix_base import MIX_BASE_COLUMN, FixedBetaOffsetRegressor, with_mix_base_column


def test_no_recorded_base_leaves_the_frame_unchanged():
    """An empty provenance returns the frame object itself."""
    X = pd.DataFrame({"a": [1.0, 2.0]})
    assert with_mix_base_column(X, []) is X


def test_raw_base_is_appended_and_the_input_frame_is_not_modified():
    """A raw recipe gives a copy with the column appended; the original frame keeps its columns."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    out = with_mix_base_column(X, [{"base": "a", "beta": 2.0, "recipe": {"raw": "a"}}])
    assert MIX_BASE_COLUMN in out.columns
    assert np.allclose(out[MIX_BASE_COLUMN], [1.0, 2.0, 3.0])
    assert MIX_BASE_COLUMN not in X.columns


def test_fixed_beta_adds_back_the_base_it_removed():
    """The prediction differs from the inner regressor on the other columns by exactly beta times the base column."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(500)
    feat = rng.standard_normal(500)
    y = 2.0 * base + feat
    X = pd.DataFrame({"f": feat, MIX_BASE_COLUMN: base})
    m = FixedBetaOffsetRegressor(Ridge(alpha=1e-8), MIX_BASE_COLUMN, beta=2.0).fit(X, y)
    inner = m.regressor_.predict(X[["f"]])
    assert np.allclose(m.predict(X) - inner, 2.0 * base, atol=1e-9)
    assert m._resolve_base_columns() == [MIX_BASE_COLUMN]


def test_missing_or_non_finite_base_is_refused():
    """A missing base column raises, and a non-finite base raises rather than being zero-filled."""
    m = FixedBetaOffsetRegressor(Ridge(), MIX_BASE_COLUMN, beta=1.0)
    with pytest.raises(KeyError):
        m.fit(pd.DataFrame({"f": [1.0, 2.0]}), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        m.fit(pd.DataFrame({"f": [1.0, 2.0], MIX_BASE_COLUMN: [1.0, np.nan]}), np.array([1.0, 2.0]))
