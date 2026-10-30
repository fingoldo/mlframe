"""The pair-search chunk width is bounded by the free VRAM budget, read at planning time.

A chunk wider than ``min(1.5 GB, free / 2)`` of int8 codes would be declined by the resident path mid-fit, so the cap
must follow the free memory the device reports, not a fixed constant.
"""

import pytest

pytest.importorskip("cupy")


@pytest.fixture
def fake_free(monkeypatch):
    """Replace the device's free-memory reading with a fixed value and return a setter for it."""
    import cupy as cp

    total = 4 * 1024**3
    state = {"free": 3 * 1024**3}
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (state["free"], total))
    return state


def test_device_chunk_cap_follows_free_vram_budget(fake_free):
    """With plenty of free VRAM the cap is the 1.5 GB ceiling over the row count; with little free VRAM it is half of what is free."""
    from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_chunks import device_chunk_cols_cap

    n_rows = 30000
    ceiling = 1536 * 1024 * 1024
    assert device_chunk_cols_cap(n_rows) == ceiling // n_rows

    fake_free["free"] = 200 * 1024**2
    assert device_chunk_cols_cap(n_rows) == (100 * 1024**2) // n_rows


def test_device_chunk_cap_releases_idle_pool_blocks_before_reading(fake_free, monkeypatch):
    """Idle blocks in the cupy pool are returned to the device before the free memory is read, so a warm pool does not shrink the budget."""
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_chunks as pch

    calls = []
    monkeypatch.setattr(pch, "flush_idle_device_pool", lambda: calls.append("flush"))
    pch.device_chunk_cols_cap(30000)
    assert calls == ["flush"]
