"""Cluster 17 regression: the Tweedie warning-count cache is bounded (audit 2026-11-20, finding 17-F21).

Many distinct (name, invalid, total) keys used to grow the dict by one entry each. The cache now resets at its cap, so it never exceeds it.
"""

import warnings

from mlframe.metrics.regression import _regression_deviance as dev


def test_warn_cache_never_exceeds_its_cap(monkeypatch):
    """Recording more distinct keys than the cap keeps the dict at or under the cap (17-F21)."""
    monkeypatch.setattr(dev, "_TWEEDIE_WARN_MAX_KEYS", 50)
    dev._TWEEDIE_WARN_COUNTS.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(200):
            dev._maybe_warn_tweedie("tweedie", invalid=1, total=10 + i)
    assert len(dev._TWEEDIE_WARN_COUNTS) <= 50


def test_recurrence_is_still_counted_below_the_cap(monkeypatch):
    """Control: repeated identical keys still count up below the cap (17-F21)."""
    monkeypatch.setattr(dev, "_TWEEDIE_WARN_MAX_KEYS", 50)
    dev._TWEEDIE_WARN_COUNTS.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            dev._maybe_warn_tweedie("tweedie", invalid=2, total=7)
    assert dev._TWEEDIE_WARN_COUNTS[("tweedie", 2, 7)] == 3
