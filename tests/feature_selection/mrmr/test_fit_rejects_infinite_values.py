"""MRMR.fit rejects +/-inf in a numeric input with an error that names the column (AUD23-55 premise: inf is never treated as missing inside the fit).

The audit suspected that +inf was handled as a missing value at fit and as a regular value at transform. The entry validation raises before any of that code runs, so the two cannot disagree; this pins it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_fit_raises_and_names_the_column(bad):
    """A single infinite cell stops the fit with a message that names the column; a NaN in the same place does not."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((300, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] > 0).astype(int)
    X_inf = X.copy()
    X_inf.loc[5, "b"] = bad
    with pytest.raises(ValueError, match=r"inf values \(column 'b'\)"):
        MRMR(verbose=0, random_seed=0, fe_max_steps=0).fit(X_inf, y)
    X_nan = X.copy()
    X_nan.loc[5, "b"] = np.nan
    MRMR(verbose=0, random_seed=0, fe_max_steps=0).fit(X_nan, y)
