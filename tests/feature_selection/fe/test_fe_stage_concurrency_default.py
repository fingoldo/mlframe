"""The concurrent FE stage cascade is on by default and ``MLFRAME_FE_STAGE_CONCURRENCY=0`` turns it off."""

from __future__ import annotations

from mlframe.feature_selection.filters._mrmr_fit_impl._fe_stage_concurrency import concurrency_enabled


def test_default_on_and_off_switch(monkeypatch):
    """Unset or ``1`` -> on; ``0`` -> off."""
    monkeypatch.delenv("MLFRAME_FE_STAGE_CONCURRENCY", raising=False)
    assert concurrency_enabled()
    monkeypatch.setenv("MLFRAME_FE_STAGE_CONCURRENCY", "1")
    assert concurrency_enabled()
    monkeypatch.setenv("MLFRAME_FE_STAGE_CONCURRENCY", "0")
    assert not concurrency_enabled()


def test_vram_weights_split_memory_not_walls():
    """Placement by wall weights, memory by VRAM weights: the shortest stage with the largest memory need gets the largest budget."""
    from mlframe.feature_selection.filters._fe_device_scheduler import plan_stage_slots

    facts = [(0, 3 << 30, 6)]  # one device, 3 GiB free, 6 SMs
    by_wall = plan_stage_slots([14.0, 34.0, 18.0], facts=facts, cushion_bytes=1 << 30, min_budget_bytes=1)
    by_vram = plan_stage_slots([14.0, 34.0, 18.0], facts=facts, cushion_bytes=1 << 30, min_budget_bytes=1, vram_weights=[461.0, 274.0, 110.0])
    assert by_wall[0].budget_bytes < by_wall[1].budget_bytes
    assert by_vram[0].budget_bytes > by_vram[1].budget_bytes > by_vram[2].budget_bytes
    assert sum(s.budget_bytes for s in by_vram) <= 2 << 30
