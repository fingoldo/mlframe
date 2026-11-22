"""Audit 2026-11-20 finding 04-F21: a resampled probe carries the donor column's missingness, not none."""

import numpy as np

from mlframe.data.datasets.from_real import _resample_column


def test_probe_missing_fraction_matches_the_donor():
    """A donor with 30% NaN gives a probe with about 30% NaN; the old code drop left the probe with none."""
    rng = np.random.default_rng(0)
    donor = np.where(np.arange(20000) % 10 < 3, np.nan, rng.normal(size=20000))
    probe = _resample_column(donor, np.random.default_rng(1))
    assert abs(np.isnan(probe).mean() - 0.3) < 0.02
