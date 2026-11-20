"""ACE accepts a polars frame (its ``to_numpy`` takes no dtype argument)."""

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from mlframe.feature_selection.ace import ACESelector


def test_ace_selector_fits_a_polars_frame():
    """A polars frame fits and records its column names."""
    rng = np.random.default_rng(0)
    n = 300
    X = pl.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"].to_numpy() > 0).astype(int)
    sel = ACESelector(n_replicates=4, n_masking_rounds=1).fit(X, y)
    assert sel.n_features_in_ == 2 and list(sel.feature_names_in_) == ["a", "b"]
