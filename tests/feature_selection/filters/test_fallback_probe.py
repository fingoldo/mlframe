"""``call_or_default``: a probe failure falls back to its default AND is visible (mrmr_audit_2026-09-14 NUM-32..NUM-36).

These sites wrapped an estimator call in ``except Exception`` and substituted a default at debug level, so a persistent fault silently changed
selection: operands re-attached unverified (NUM-32), a failed relevance probe tied at MI's floor of 0.0 instead of being excluded (NUM-33), the
raw floor pool emptied (NUM-34), ``uaed_auto_size`` silently disabled (NUM-35), a recipe shipped without its category code maps (NUM-36). All
now go through ``call_or_default``; relevance probes pass ``-inf`` so a failed candidate can never win or tie an argmax.
"""

from __future__ import annotations

import logging

import pytest

from mlframe.feature_selection.filters._fallback_probe import call_or_default
from mlframe.utils.log_throttle import reset_throttle_counts


def _boom():
    """A probe that always fails."""
    raise RuntimeError("synthetic probe failure")


def test_success_returns_the_value_and_stays_quiet(caplog):
    """A healthy probe returns its own value and logs nothing."""
    reset_throttle_counts("probe_test_ok")
    with caplog.at_level(logging.WARNING):
        assert call_or_default(lambda: 0.42, float("-inf"), key="probe_test_ok", message="unused") == 0.42
    assert not caplog.records


@pytest.mark.parametrize("default", [True, False, float("-inf")])
def test_failure_returns_the_default_and_warns_with_the_exception(caplog, default):
    """A failing probe returns exactly its default, and the warning names the message and the exception type."""
    key = f"probe_test_fail_{default}"
    reset_throttle_counts(key)
    with caplog.at_level(logging.WARNING):
        assert call_or_default(_boom, default, key=key, message="the probe could not run") == default
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warned and "the probe could not run" in warned[0] and "RuntimeError" in warned[0]


def test_neg_inf_default_can_never_win_an_argmax():
    """The relevance-probe contract: a failed candidate scored -inf loses to every scored one, including a genuine 0.0."""
    reset_throttle_counts("probe_test_argmax")
    scores = {"good": 0.0, "failed": call_or_default(_boom, float("-inf"), key="probe_test_argmax", message="relevance probe failed")}
    assert max(scores, key=scores.get) == "good"


def test_throttling_is_per_key(caplog):
    """Two call sites under different keys must each be reported, even inside one loop."""
    reset_throttle_counts("probe_test_a")
    reset_throttle_counts("probe_test_b")
    with caplog.at_level(logging.WARNING):
        call_or_default(_boom, True, key="probe_test_a", message="site A failed")
        call_or_default(_boom, True, key="probe_test_b", message="site B failed")
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "site A failed" in text and "site B failed" in text
