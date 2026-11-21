"""The edge workers of a wave follow the cascades still running, and the shared edge pool is sized to the cores (audit 2026-11-20-perf E-F01, E-F02)."""

from mlframe.feature_selection.filters import _adaptive_nbins as nb
from mlframe.feature_selection.filters import _edge_pool as ep


def test_finished_cascade_hands_its_share_to_the_others(monkeypatch):
    """Three cascades give each a third of the cores; when one ends the other two get a half, and the count never drops below one."""
    monkeypatch.delenv("MLFRAME_EDGE_JOBS", raising=False)
    import psutil

    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 8)
    with ep.concurrent_stages(3):
        assert nb._resolve_edge_jobs(None, 100) == 2
        ep.stage_finished()
        assert nb._resolve_edge_jobs(None, 100) == 4
        ep.stage_finished()
        ep.stage_finished()
        ep.stage_finished()
        assert nb._resolve_edge_jobs(None, 100) == 8
    assert ep._ACTIVE_STAGES[0] == 1


def test_shared_edge_pool_is_created_at_core_count(monkeypatch):
    """A first caller asking for two workers gets a pool of the logical core count (its share stays the number of chunks it submits)."""
    import psutil

    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 6)
    monkeypatch.setattr(nb, "_EDGE_POOL", None)
    monkeypatch.setattr(nb, "_EDGE_POOL_WORKERS", 0)
    monkeypatch.setattr(nb, "_RETIRED_EDGE_POOLS", [])
    pool = nb._edge_pool(2)
    try:
        assert nb._EDGE_POOL_WORKERS == 6 and pool._max_workers == 6
    finally:
        pool.shutdown(wait=False)
