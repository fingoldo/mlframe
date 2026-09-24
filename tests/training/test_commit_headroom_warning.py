"""Commit pressure is reported at WARNING while a run can still be stopped.

A production run logged "commit limit 344.1 GB (avail 1.0)" at INFO among a dozen other INFO lines, started anyway and
died 27 minutes later. The same kernel held 204.8 GB of private commit against 0.8 GB resident, which no code in the
run could release.
"""

from __future__ import annotations

import logging

from mlframe.training._commit_headroom import (
    COMMIT_AVAIL_WARN_GB,
    low_headroom_message,
    retained_commit_message,
    warn_on_commit_pressure,
)


class TestSystemHeadroom:
    """When the host is nearly out of commit."""

    def test_the_production_case_is_flagged(self):
        """1.0 GB left of a 344 GB limit."""
        message = low_headroom_message(344.1, 1.0)
        assert message and "1.0 GB of commit left" in message and "restart the kernel" in message

    def test_a_large_limit_is_judged_on_its_own_scale(self):
        """5% of a huge page file is more than the flat bar, so a proportionally thin margin still fires."""
        assert low_headroom_message(1000.0, 40.0) is not None

    def test_ample_headroom_is_silent(self):
        """No warning when there is room to work."""
        assert low_headroom_message(344.1, 200.0) is None
        assert low_headroom_message(100.0, COMMIT_AVAIL_WARN_GB + 1) is None

    def test_unknown_numbers_are_not_a_warning(self):
        """A probe that returned nothing must not produce a scare."""
        assert low_headroom_message(0.0, 0.0) is None


class TestRetainedCommit:
    """When this interpreter is the one holding the charge."""

    def test_the_production_shape_is_flagged(self):
        """204.8 GB committed against 0.8 GB resident: freed memory the allocator kept."""
        message = retained_commit_message(204.8, 0.8)
        assert message and "204.8 GB of private commit" in message and "restart the kernel" in message

    def test_a_working_process_is_silent(self):
        """A process actually using its commit is not asked to restart."""
        assert retained_commit_message(40.0, 30.0) is None

    def test_a_small_charge_is_silent(self):
        """Below the bar a restart buys too little to advise."""
        assert retained_commit_message(8.0, 0.5) is None

    def test_missing_numbers_are_silent(self):
        """psutil without a ``private`` field must not fabricate a verdict."""
        assert retained_commit_message(None, 1.0) is None
        assert retained_commit_message(100.0, None) is None


def test_both_conditions_are_logged_at_warning(caplog):
    """The operator sees them as warnings, not buried in the startup INFO block."""
    with caplog.at_level(logging.WARNING):
        messages = warn_on_commit_pressure(344.1, 1.0, private_commit_gb=204.8, rss_gb=0.8)
    assert len(messages) == 2
    assert len([r for r in caplog.records if "[commit-pressure]" in r.getMessage()]) == 2


def test_check_and_warn_survives_a_failing_probe(monkeypatch):
    """A diagnostic must never end a run."""
    import mlframe.training._commit_headroom as ch

    monkeypatch.setattr("mlframe.training.crash_diagnostics.windows_commit_status", lambda: (_ for _ in ()).throw(RuntimeError("probe down")))
    assert ch.check_and_warn() == []
