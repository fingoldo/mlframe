"""String group ids in the recency rolling functions, sliding-window groups shorter than the window, and similarity_scale validation."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.feature_engineering.grouped_rank import per_group_sliding_window
from mlframe.feature_engineering.recency_weighted_rolling import recency_weighted_rolling_mean, recency_weighted_rolling_std
from mlframe.votenrank.similarity_blend import SimilarityBlendEnsemble


def test_rolling_functions_accept_string_group_ids():
    """np.diff on a string array raised TypeError; group boundaries are now found by comparing neighbours."""
    values = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    groups = np.array(["a", "a", "a", "b", "b"])
    mean_s = recency_weighted_rolling_mean(values, groups, 2)
    mean_i = recency_weighted_rolling_mean(values, np.array([0, 0, 0, 1, 1]), 2)
    np.testing.assert_allclose(mean_s, mean_i)
    assert recency_weighted_rolling_std(values, groups, 2).shape == (5,)


def test_group_shorter_than_the_window_is_skipped_not_fatal():
    """window_K=5, min_group_size=2: a 3-row group used to raise from sliding_window_view."""
    values = np.arange(8, dtype=float)
    groups = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    yielded = list(per_group_sliding_window(values, groups, 5, min_group_size=2))
    assert len(yielded) == 1


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan")])
def test_non_positive_similarity_scale_is_rejected(scale):
    """scale 0 gave NaN similarities for exact matches; a negative scale gave weights above 1."""
    est = SimilarityBlendEnsemble(LinearRegression(), LinearRegression(), similarity_scale=scale)
    with pytest.raises(ValueError, match="similarity_scale"):
        est.fit(np.zeros((4, 2)), np.zeros(4))


def test_rolling_mean_and_std_skip_nan_values():
    """values [1, NaN, 2, 3], window 3, one group: means 1, 1, 1.5, 2.5 (the NaN spread to rows 1-3 and gave NaN there)."""
    values = np.array([1.0, np.nan, 2.0, 3.0])
    groups = np.zeros(4, dtype=int)
    mean = recency_weighted_rolling_mean(values, groups, 3, param=0.0, scheme="poly")
    assert np.isfinite(mean).all()
    np.testing.assert_allclose(mean, [1.0, 1.0, 1.5, 2.5])
    assert np.isfinite(recency_weighted_rolling_std(values, groups, 3, param=0.0, scheme="poly")).all()
