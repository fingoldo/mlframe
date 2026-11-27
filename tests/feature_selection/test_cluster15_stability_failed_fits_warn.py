"""Cluster 15 regression: stability selection warns when its bootstrap fits fail, even with verbose=False (audit 2026-11-20, 15-F6)."""

import logging

from mlframe.feature_selection.wrappers.rfecv import _stability_select as ss


def test_all_failed_fits_log_a_warning(caplog):
    """Every fit failed: one WARNING says none succeeded (15-F6)."""
    with caplog.at_level(logging.WARNING, logger=ss.logger.name):
        ss._warn_on_failed_bootstrap_fits(10, 10)
    assert any("none succeeded" in r.getMessage() for r in caplog.records)


def test_majority_failed_fits_warn_and_few_do_not(caplog):
    """Half the fits failing warns; one failure in ten stays quiet; zero attempts is quiet (15-F6)."""
    with caplog.at_level(logging.WARNING, logger=ss.logger.name):
        ss._warn_on_failed_bootstrap_fits(10, 1)
        ss._warn_on_failed_bootstrap_fits(0, 0)
    assert not caplog.records
    with caplog.at_level(logging.WARNING, logger=ss.logger.name):
        ss._warn_on_failed_bootstrap_fits(10, 5)
    assert len(caplog.records) == 1
