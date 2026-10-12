"""The free-VRAM gate of the FE GPU default-on switches counts memory cupy's pool holds but has not handed out as free."""

import pytest

cp = pytest.importorskip("cupy")


class _Pool:
    """Stand-in for the cupy memory pool with a fixed number of idle bytes."""

    def __init__(self, idle):
        self._idle = idle

    def free_bytes(self):
        """Bytes held by the pool and not handed out."""
        return self._idle


def test_pool_held_bytes_count_as_free(monkeypatch):
    """100 MB free on the device plus 2000 MB idle in the pool passes a 1000 MB bar; without the idle bytes it fails."""
    from mlframe.feature_selection.filters import _gpu_resident_fe as gre

    monkeypatch.setattr(gre, "_gpu_min_free_vram_mb", lambda: 1000)
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (100 * 2**20, 8 * 2**30))
    monkeypatch.setattr(cp, "get_default_memory_pool", lambda: _Pool(2000 * 2**20))
    assert gre._gpu_has_free_vram() is True
    monkeypatch.setattr(cp, "get_default_memory_pool", lambda: _Pool(0))
    assert gre._gpu_has_free_vram() is False
