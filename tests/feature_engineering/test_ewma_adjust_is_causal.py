"""EWMA residual with adjust=True matches pandas and never reads future rows (audit 2026-11-20, finding 24-F5)."""

import numpy as np
import pandas as pd

from mlframe.feature_engineering.stationarity import ewma_residual


def test_adjusted_residual_matches_pandas_ewm():
    """The bias-corrected EWMA must equal pandas' ``ewm(adjust=True)``; the old weights were indexed from the segment end."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(60).cumsum()
    hl = 5.0
    alpha = 1.0 - 2.0 ** (-1.0 / hl)
    expected = x - pd.Series(x).ewm(alpha=alpha, adjust=True).mean().to_numpy()
    got = np.asarray(ewma_residual(x, half_life=hl, adjust=True))
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


def test_early_rows_do_not_depend_on_future_rows():
    """Appending rows must not change the residual of rows already seen."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(80)
    short = np.asarray(ewma_residual(x[:20], half_life=4.0, adjust=True))
    longer = np.asarray(ewma_residual(x, half_life=4.0, adjust=True))
    np.testing.assert_allclose(longer[:20], short, rtol=1e-12, atol=1e-12)
