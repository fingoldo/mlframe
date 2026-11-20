"""drop_near_noise_univariate_auc does not raise on a NaN cell and keeps a strongly predictive column (audit 2026-11-20, finding 24-F1)."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.drop_near_noise_univariate_auc import drop_near_noise_univariate_auc


def test_one_nan_in_a_column_does_not_crash_the_default_path():
    """The default path refused any non-finite cell, so a single missing value in any numeric column crashed the whole selection."""
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.5).astype(int)
    signal = y + 0.2 * rng.standard_normal(n)
    noise = rng.standard_normal(n)
    noise[7] = np.nan
    df = pd.DataFrame({"signal": signal, "noise": noise})
    kept = drop_near_noise_univariate_auc(df, y)
    assert "signal" not in kept  # a strongly predictive column is never flagged as noise
