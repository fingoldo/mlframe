"""biz_value test for ``preprocessing.gaussian_power_transform_search``.

Synthetic: a heavily right-skewed (log-normal-like) feature. The unsupervised search should pick a transform
that drives absolute skewness far below the raw feature's skewness -- no target/model involved, purely a
distribution-shape claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skew

from mlframe.preprocessing.gaussian_power_transform_search import apply_gaussian_power_transform, gaussian_power_transform_search


def test_biz_val_gaussian_power_transform_search_reduces_skew_on_lognormal_feature():
    rng = np.random.default_rng(0)
    raw = np.exp(rng.normal(loc=0.0, scale=1.0, size=5000))  # classic log-normal, heavily right-skewed
    df = pd.DataFrame({"x": raw})

    raw_abs_skew = float(abs(skew(raw)))
    result = gaussian_power_transform_search(df)

    assert "x" in result
    best_abs_skew = result["x"]["best_abs_skew"]
    assert best_abs_skew < raw_abs_skew * 0.15, f"expected the best transform to cut abs-skew by >=85% vs raw ({raw_abs_skew:.4f}), got {best_abs_skew:.4f}"
    assert result["x"]["best_transform"] != "identity", "expected a non-identity transform to win on a heavily skewed feature"

    transformed_df = apply_gaussian_power_transform(df, result)
    transformed_abs_skew = float(abs(skew(transformed_df["x"].to_numpy())))
    assert transformed_abs_skew == pytest.approx(best_abs_skew, rel=1e-6)


def test_gaussian_power_transform_search_identity_wins_on_already_gaussian_feature():
    rng = np.random.default_rng(1)
    raw = rng.normal(loc=0.0, scale=1.0, size=5000)
    df = pd.DataFrame({"x": raw})

    result = gaussian_power_transform_search(df)
    assert result["x"]["best_abs_skew"] < 0.1


def test_gaussian_power_transform_search_skips_columns_with_too_few_finite_values():
    df = pd.DataFrame({"x": [np.nan, np.nan, 1.0]})
    result = gaussian_power_transform_search(df)
    assert "x" not in result


def test_gaussian_power_transform_search_boxcox_skipped_for_non_positive_column():
    df = pd.DataFrame({"x": np.concatenate([np.array([-1.0, 0.0]), np.random.default_rng(2).normal(size=200)])})
    result = gaussian_power_transform_search(df)
    assert "boxcox" not in result["x"]["all_abs_skew"]
