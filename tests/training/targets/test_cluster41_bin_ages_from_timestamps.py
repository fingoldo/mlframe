"""Cluster 41 regression: monthly Timestamp bin starts give real unequal ages, not an even-spacing fallback (audit 2026-11-20, finding 41-F8)."""

from types import SimpleNamespace

import pandas as pd

from mlframe.training.targets._target_temporal_audit_from_agg import _bin_ages


def test_timestamp_bin_starts_give_real_unequal_ages():
    """Jan, Feb, Mar 2024 starts: ages (seconds from each bin to the newest) are 60 days (leap year), 29 days, 0 (41-F8)."""
    bins = [SimpleNamespace(bin_start=pd.Timestamp(t)) for t in ("2024-01-01", "2024-02-01", "2024-03-01")]
    ages = _bin_ages(bins)
    assert ages == [60 * 86400.0, 29 * 86400.0, 0.0], ages


def test_numeric_bin_starts_still_work():
    """Control: numeric starts keep their plain differences."""
    bins = [SimpleNamespace(bin_start=v) for v in (0.0, 2.0, 5.0)]
    assert _bin_ages(bins) == [5.0, 3.0, 0.0]
