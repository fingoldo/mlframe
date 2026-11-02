"""The edge-jobs pool size: logical cores by default, an explicit count wins, MLFRAME_EDGE_JOBS overrides, never more than the columns."""

from __future__ import annotations

import os

from mlframe.feature_selection.filters._adaptive_nbins import _resolve_edge_jobs


def test_default_is_the_logical_core_count_capped_by_columns(monkeypatch):
    """No explicit count: ``os.cpu_count()`` threads, but never more than the columns to compute."""
    monkeypatch.delenv("MLFRAME_EDGE_JOBS", raising=False)
    logical = os.cpu_count() or 1
    assert _resolve_edge_jobs(None, 10_000) == logical
    assert _resolve_edge_jobs(None, 2) == min(2, logical)


def test_explicit_count_and_env_override(monkeypatch):
    """An explicit positive count is used as given; the environment variable replaces only the default."""
    monkeypatch.setenv("MLFRAME_EDGE_JOBS", "3")
    assert _resolve_edge_jobs(None, 100) == 3
    assert _resolve_edge_jobs(5, 100) == 5
    assert _resolve_edge_jobs(None, 2) == 2
