"""The FE cascades run side by side only when their working set fits the host RAM (PLAN-10).

Measured at 100000 x 1000 with every family: concurrent 238.0 s and 5585 MB of RSS growth, one after the other 254.1 s (+6.8 percent) and 3845 MB (-31 percent). On a host where four stages at once do not fit
(1M x 1000 on 16 GB) the slower order is the one that finishes, so the concurrent order is kept only when ``factor x float32 frame`` fits the available RAM.
"""

from __future__ import annotations

import logging

from mlframe.feature_selection.filters._mrmr_fit_impl import _fe_stage_concurrency as C

GB = 2**30


def _free(monkeypatch, gb: float) -> None:
    """Pretend ``gb`` GB of RAM are available."""
    monkeypatch.setattr(C, "host_memory", lambda: (int(gb * GB), int(32 * GB)))


def test_a_frame_that_fits_keeps_the_concurrent_order(monkeypatch):
    """100000 x 1000 (0.4 GB, 1.6 GB of working set) on 9.6 GB free: concurrent."""
    monkeypatch.delenv("MLFRAME_FE_STAGE_CONCURRENCY_MEM_FACTOR", raising=False)
    _free(monkeypatch, 9.6)
    assert C.memory_allows_concurrency(100_000, 1000) is True


def test_a_frame_that_does_not_fit_does_not_run_the_three_cascades_together(monkeypatch, caplog):
    """1M x 1000 (4 GB, 16 GB of working set) on 9.6 GB free: not the three together (the waves that fit take over, ``_fe_stage_waves``), and the decision is logged with the numbers."""
    monkeypatch.delenv("MLFRAME_FE_STAGE_CONCURRENCY_MEM_FACTOR", raising=False)
    _free(monkeypatch, 9.6)
    with caplog.at_level(logging.INFO):
        assert C.memory_allows_concurrency(1_000_000, 1000) is False
    assert any("three cascades together" in r.getMessage() and "waves that fit" in r.getMessage() and "1000000 x 1000" in r.getMessage() for r in caplog.records)


def test_the_boundary_follows_the_factor(monkeypatch):
    """500000 x 1000 (2 GB frame): concurrent at factor 4 on 9.6 GB (8 GB needed), serial at factor 6 (12 GB needed); the environment variable sets the factor."""
    _free(monkeypatch, 9.6)
    monkeypatch.delenv("MLFRAME_FE_STAGE_CONCURRENCY_MEM_FACTOR", raising=False)
    assert C.memory_allows_concurrency(500_000, 1000) is True
    monkeypatch.setenv("MLFRAME_FE_STAGE_CONCURRENCY_MEM_FACTOR", "6")
    assert C.memory_allows_concurrency(500_000, 1000) is False


def test_an_unreadable_ram_figure_keeps_the_concurrent_order(monkeypatch):
    """No RAM probe, no change of behaviour."""
    monkeypatch.delenv("MLFRAME_FE_STAGE_CONCURRENCY_MEM_FACTOR", raising=False)

    def boom():
        """A probe that fails."""
        raise OSError("no counter")

    monkeypatch.setattr(C, "host_memory", boom)
    assert C.memory_allows_concurrency(1_000_000, 1000) is True
