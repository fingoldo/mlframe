"""Selectors do not deep-copy a whole input frame to read or re-encode a few columns (tracker 2026-11-02 wave 16, MEM16-1, MEM16-6)."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.functional_adapters import _numeric_view_for_selection


def test_numeric_columns_stay_shared_when_a_string_column_is_encoded():
    """One string column used to force X.copy() of every numeric column too; the caller's frame must stay unchanged and its numeric buffer shared."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(size=1000), "s": rng.choice(["x", "y", "z"], 1000)})
    before = X.copy(deep=True)
    out = _numeric_view_for_selection(X)
    assert out["s"].dtype.kind == "f"
    assert np.shares_memory(out["a"].to_numpy(), X["a"].to_numpy())
    pd.testing.assert_frame_equal(X, before)
