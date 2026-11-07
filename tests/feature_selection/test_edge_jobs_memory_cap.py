"""The edge workers are limited by memory on long frames and untouched on short ones."""

import types

import pytest

from mlframe.feature_selection.filters import _adaptive_nbins as A


@pytest.fixture(autouse=True)
def _plenty_of_free_ram(monkeypatch):
    """Make the free-RAM share larger than any cap below, so only the configured cap decides."""
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(available=10**13))


def test_short_frames_keep_every_worker(monkeypatch):
    """Up to a few hundred thousand rows the in-flight columns fit the 2 GB budget, so all 16 workers stay."""
    monkeypatch.setenv("MLFRAME_EDGE_RAM_GB", "2")
    assert A._cap_jobs_by_memory(16, 200_000) == 16


def test_long_frames_get_fewer_workers_never_fewer_than_one(monkeypatch):
    """At 10 million rows a column in flight is 800 MB, so a 2 GB budget allows two; at 100 million one."""
    monkeypatch.setenv("MLFRAME_EDGE_RAM_GB", "2")
    assert A._cap_jobs_by_memory(16, 10_000_000) == 2
    assert A._cap_jobs_by_memory(16, 100_000_000) == 1


def test_the_cap_never_raises_the_worker_count(monkeypatch):
    """A request for 3 workers stays 3 however large the budget."""
    monkeypatch.setenv("MLFRAME_EDGE_RAM_GB", "100")
    assert A._cap_jobs_by_memory(3, 1_000) == 3


def test_free_ram_share_limits_the_budget(monkeypatch):
    """With 4 GB free the budget is a quarter of it, 1 GB, which allows one 800 MB column at 10 million rows."""
    psutil = pytest.importorskip("psutil")
    monkeypatch.setenv("MLFRAME_EDGE_RAM_GB", "100")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(available=4 * 1024**3))
    assert A._cap_jobs_by_memory(16, 10_000_000) == 1
