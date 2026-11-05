"""Concurrent FE stage cascades split the edge-jobs threads instead of each sizing its pool to every core."""

from mlframe.feature_selection.filters._adaptive_nbins import _resolve_edge_jobs
from mlframe.feature_selection.filters._edge_pool import concurrent_stages


def test_concurrent_stages_split_the_auto_job_count(monkeypatch):
    """Inside ``concurrent_stages(3)`` an unset ``n_jobs`` gets a third of the cores; outside, all of them; an explicit ``n_jobs`` is kept."""
    monkeypatch.delenv("MLFRAME_EDGE_JOBS", raising=False)
    full = _resolve_edge_jobs(None, 10_000)
    with concurrent_stages(3):
        shared = _resolve_edge_jobs(None, 10_000)
        explicit = _resolve_edge_jobs(5, 10_000)
    assert shared == max(1, full // 3)
    assert explicit == 5
    assert _resolve_edge_jobs(None, 10_000) == full
