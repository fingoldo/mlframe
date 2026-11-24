"""Opt-in per-family RSS accounting: a family that allocates shows a peak above its entry, one that does not shows about nothing (PLAN-10 tooling)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_family_timing as T


@pytest.fixture(autouse=True)
def _reset_active_rss_entries():
    """Clear the RSS poller's active-entry table so a leaked entry from one test cannot raise another test's peak."""
    T._RSS_ACTIVE.clear()
    yield
    T._RSS_ACTIVE.clear()


def test_the_peak_over_entry_sees_a_transient_allocation(monkeypatch, caplog):
    """A 200 MB transient inside one family is reported as a peak of about that size, retained about nothing; the idle family reports under 50 MB."""
    monkeypatch.setenv("MLFRAME_FE_FAMILY_RSS", "1")
    T._FE_FAMILY_WALL.clear()
    T._FE_FAMILY_RSS.clear()
    with T.fe_family_timer("alloc"):
        block = np.ones(25_000_000, dtype=np.float64)  # 200 MB, touched
        T._sample_active_rss()  # the poller's own poll, taken while the block is live
        del block
    with T.fe_family_timer("idle"):
        T._sample_active_rss()
    peak_alloc, kept_alloc = T._FE_FAMILY_RSS["alloc"]
    peak_idle, _ = T._FE_FAMILY_RSS["idle"]
    assert peak_alloc > 150 * 2**20
    assert kept_alloc < 100 * 2**20
    assert peak_idle < 50 * 2**20
    with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.filters.mrmr"):
        T.log_fe_family_summary()
    assert any("per-family RSS" in r.getMessage() and "alloc=+" in r.getMessage() for r in caplog.records)


def test_off_by_default_records_nothing(monkeypatch):
    """Without the environment variable no RSS figure is recorded."""
    monkeypatch.delenv("MLFRAME_FE_FAMILY_RSS", raising=False)
    T._FE_FAMILY_RSS.clear()
    with T.fe_family_timer("x"):
        pass
    assert not T._FE_FAMILY_RSS


def test_gate_laps_are_accounted_too(monkeypatch):
    """The ``gate:<flag>`` laps (how the cascades are attributed) carry the same RSS figures."""
    monkeypatch.setenv("MLFRAME_FE_FAMILY_RSS", "1")
    T._FE_FAMILY_RSS.clear()
    timer = T.FamilyLapTimer()
    assert timer.gate("fe_demo_enable", True)
    block = np.ones(25_000_000, dtype=np.float64)
    T._sample_active_rss()  # the poller's own poll, taken while the block is live
    del block
    timer.close()
    assert T._FE_FAMILY_RSS["gate:fe_demo_enable"][0] > 150 * 2**20
