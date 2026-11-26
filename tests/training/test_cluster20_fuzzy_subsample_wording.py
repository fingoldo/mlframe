"""Cluster 20 regression: the fuzzy-membership docs describe the draw accurately (audit 2026-11-20, finding 20-F43).

The code draws the fit subsample with ``rng.integers``, which is with replacement, and says so in its comment (duplicates are harmless for
quantiles). The module docstring used to call it a random subsample. It now says with replacement.
"""

import numpy as np

from mlframe.reporting.charts import fuzzy_membership as fm


def _capture_fit_input(monkeypatch, values):
    """Run the fuzzy curves on ``values`` with the partition fit stubbed out, and return the sample the fit received."""
    import mlframe.feature_engineering.fuzzy_features as ff

    captured = {}

    def _fake_fit(sample, n_sets, kind):
        """Record the sample the fit was given."""
        captured["sample"] = np.asarray(sample).copy()
        return None

    def _fake_transform(grid_x, recipe):
        """Return a flat membership grid of the right shape."""
        return np.zeros((grid_x.size, 2))

    monkeypatch.setattr(ff, "fuzzy_partition_fit", _fake_fit)
    monkeypatch.setattr(ff, "fuzzy_partition_transform", _fake_transform)
    fm.fuzzy_membership_curves(values, grid=10)
    return captured["sample"]


def test_large_column_is_subsampled_with_replacement(monkeypatch):
    """A column above the cap is drawn WITH replacement: the sample has the cap's size but repeats some values (20-F43)."""
    values = np.arange(2 * fm._FIT_SUBSAMPLE_CAP, dtype=np.float64)
    sample = _capture_fit_input(monkeypatch, values)
    assert sample.size == fm._FIT_SUBSAMPLE_CAP
    assert np.unique(sample).size < sample.size


def test_column_below_the_cap_is_used_unchanged(monkeypatch):
    """Control: a column within the cap reaches the fit whole, with no repeats (20-F43)."""
    values = np.arange(100, dtype=np.float64)
    sample = _capture_fit_input(monkeypatch, values)
    assert np.array_equal(sample, values)
