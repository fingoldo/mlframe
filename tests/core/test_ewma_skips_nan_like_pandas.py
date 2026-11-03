"""ewma skips NaN values like pandas Series.ewm instead of turning the rest of the series NaN."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.core.ewma import ewma


@pytest.mark.parametrize("adjust", [False, True])
@pytest.mark.parametrize("values", [[np.nan, 1.0, 2.0], [1.0, np.nan, 3.0, 4.0], [1.0, 2.0, np.nan, np.nan, 5.0], [1.0, 2.0, 3.0]])
def test_matches_pandas_with_leading_inner_and_trailing_nan(values, adjust):
    """Leading NaN, a single inner NaN, a gap of two and no NaN at all all agree with pandas (ignore_na=False)."""
    expected = pd.Series(values).ewm(alpha=0.4, adjust=adjust).mean().to_numpy()
    np.testing.assert_allclose(ewma(np.array(values), 0.4, adjust=adjust), expected, equal_nan=True)
