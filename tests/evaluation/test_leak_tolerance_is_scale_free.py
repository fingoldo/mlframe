"""The leakage verdict must not depend on the scale of the caller's scorer.

`_LEAK_TOLERANCE = 0.02` was an absolute number in score units while `scoring` is an arbitrary sklearn scorer: on a
`neg_mean_squared_error` target of variance 1e6 an inflation of tens of thousands of MSE units reported
`leak_detected=False`, and on a 0.001-scale loss every clean feature reported a leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.evaluation import detect_expanding_window_feature_leakage


def _leaky_bed(scale: float, n: int = 4000, n_cats: int = 30, seed: int = 0):
    """A category's population FREQUENCY drives the target, so a whole-frame frequency count is a low-noise
    estimate of it while the honest per-fold count - built from few early rows - is a noisy one."""
    rng = np.random.default_rng(seed)
    cat_rate = rng.uniform(0.5, 5.0, n_cats)
    cat = rng.choice(n_cats, size=n, p=cat_rate / cat_rate.sum())
    df = pd.DataFrame({"t": np.arange(n, dtype=float), "cat": cat})
    y = (cat_rate[cat] * 3.0 + rng.normal(scale=1.0, size=n)) * scale

    def freq(fit_df, transform_df):
        """Occurrence count of each row's category WITHIN fit_df."""
        return transform_df["cat"].map(fit_df["cat"].value_counts()).fillna(0.0).to_numpy(dtype=float)

    return df, y, freq


@pytest.mark.parametrize("scale", [1.0, 1000.0])
def test_the_same_leak_is_detected_at_any_target_scale(scale):
    """The same leaky feature must trip the detector whether the target is measured in units or in thousands."""
    df, y, freq = _leaky_bed(scale)
    res = detect_expanding_window_feature_leakage(df, "t", y, freq, lambda: LinearRegression(), n_splits=5, scoring="neg_mean_squared_error")
    assert res["leak_detected"], f"scale={scale}: inflation={res['inflation']} tolerance={res['leak_tolerance']}"


@pytest.mark.parametrize("scale", [1.0, 0.001])
def test_a_clean_feature_is_not_flagged_at_any_target_scale(scale):
    """The negative control: a scale-free tolerance must not turn into a hair trigger on a large-scale target."""
    rng = np.random.default_rng(1)
    n = 1000
    df = pd.DataFrame({"t": np.arange(n, dtype=float), "cat": rng.integers(0, 10, n)})
    y = rng.normal(size=n) * scale

    def noise(fit_df, transform_df):
        """A feature carrying no information about the target, honest or not."""
        return np.random.default_rng(len(fit_df)).normal(size=len(transform_df))

    res = detect_expanding_window_feature_leakage(df, "t", y, noise, lambda: LinearRegression(), n_splits=5, scoring="neg_mean_squared_error")
    assert not res["leak_detected"], f"scale={scale}: inflation={res['inflation']} tolerance={res['leak_tolerance']}"


def test_the_band_scales_with_the_metric():
    """A thousandfold larger target gives squared-error scores a millionfold larger, and the tolerance has to follow."""

    def _run(scale):
        """Run the detector on a leaky bed built at ``scale``."""
        df, y, freq = _leaky_bed(scale)
        return detect_expanding_window_feature_leakage(df, "t", y, freq, lambda: LinearRegression(), n_splits=5, scoring="neg_mean_squared_error")

    assert _run(1000.0)["leak_tolerance"] > _run(1.0)["leak_tolerance"] * 100


def test_an_explicit_tolerance_still_overrides():
    """Deriving the band from the metric must not take the absolute override away from a caller who wants one."""
    df, y, freq = _leaky_bed(1.0)
    res = detect_expanding_window_feature_leakage(df, "t", y, freq, lambda: LinearRegression(), n_splits=5, scoring="r2", leak_tolerance=10.0)
    assert res["leak_tolerance"] == 10.0
    assert not res["leak_detected"], "an absolute gate the caller set must be honoured"
