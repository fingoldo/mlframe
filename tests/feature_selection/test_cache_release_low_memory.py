"""The stage caches are dropped at a family start only when free host RAM is low."""

import types

import pytest

from mlframe.feature_selection.filters import _cache_release as C


@pytest.fixture
def releaser(monkeypatch):
    """A registered releaser that counts its calls and reports 1 MB freed."""
    calls = []
    monkeypatch.setattr(C, "_RELEASERS", [lambda: calls.append(1) or 2**20])
    return calls


def test_plenty_of_free_ram_keeps_the_caches(releaser, monkeypatch):
    """16 GB free of 32 GB is above the floor: nothing is released."""
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(available=16 * 1024**3, total=32 * 1024**3))
    assert C.release_if_memory_low() == 0 and releaser == []


def test_low_free_ram_releases_them(releaser, monkeypatch):
    """1 GB free of 16 GB is below max(2 GB, 15%): the releasers run."""
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(available=1024**3, total=16 * 1024**3))
    assert C.release_if_memory_low() == 2**20 and releaser == [1]


def test_the_floor_is_the_larger_of_the_two(releaser, monkeypatch):
    """3 GB free of 64 GB is above 2 GB but below 15% of the total: released."""
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(available=3 * 1024**3, total=64 * 1024**3))
    assert C.release_if_memory_low() == 2**20
