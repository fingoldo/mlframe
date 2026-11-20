"""CUSUM features read only past rows (audit 2026-11-20, finding 24-F6)."""

import numpy as np
import pytest

from mlframe.feature_engineering.stationarity import cusum_features

KEYS = ("cusum_pos", "cusum_neg", "rows_since_reset", "n_resets_in_window")


@pytest.mark.parametrize("threshold", [None, 4.0])
def test_early_rows_do_not_depend_on_future_rows(threshold):
    """Appending a future level shift must leave every earlier row's features unchanged (the old global mean and MAD did not)."""
    rng = np.random.default_rng(7)
    head = rng.standard_normal(40)
    tail = rng.standard_normal(60) + 50.0
    short = cusum_features(head, threshold=threshold)
    longer = cusum_features(np.concatenate([head, tail]), threshold=threshold)
    for k in KEYS:
        np.testing.assert_array_equal(longer[k][:40], short[k], err_msg=k)


def test_first_observation_does_not_trip_a_reset_on_a_high_level_series():
    """With no past rows the reference is the row itself, so a large constant level does not reset at row 0."""
    values = np.full(10, 1000.0)
    result = cusum_features(values, threshold=1.0)
    assert result["n_resets_in_window"][0] == 0
