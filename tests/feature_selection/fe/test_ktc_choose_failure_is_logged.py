"""A failing kernel-tuning choose() falls back to the row floor with a WARNING, for the numeric-decompose and order-statistics device choices."""

from __future__ import annotations

import logging

from mlframe.utils.log_throttle import reset_throttle_counts


class _BrokenSpec:
    """Tuner spec whose choose() fails."""

    def choose(self, **_kw):
        """Always fails."""
        raise RuntimeError("tuning cache unreadable")


def test_numeric_decompose_choose_failure_logs_a_warning(monkeypatch, caplog):
    """decompose_device_chosen logged the choose() failure at debug only."""
    from mlframe.feature_selection.filters import _numeric_decompose_gpu as mod

    monkeypatch.setattr(mod, "_SPEC", _BrokenSpec())
    reset_throttle_counts("numeric_decompose.choose_failed")
    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        assert mod.decompose_device_chosen(1000, 2, 3) is False
    assert any("KTC choose() failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_order_stats_choose_failure_logs_a_warning(monkeypatch, caplog):
    """order_device_chosen logged the choose() failure at debug only."""
    from mlframe.feature_selection.filters import _order_stats_gpu as mod

    monkeypatch.setattr(mod, "_SPEC", _BrokenSpec())
    reset_throttle_counts("order_stats.choose_failed")
    with caplog.at_level(logging.WARNING, logger=mod.logger.name):
        assert mod.order_device_chosen(1000) is False
    assert any("KTC choose() failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
