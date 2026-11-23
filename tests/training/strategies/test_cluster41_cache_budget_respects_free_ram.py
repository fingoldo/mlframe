"""Cluster 41 regression: the pipeline-cache budget never exceeds free RAM (audit 2026-11-20, finding 41-F19)."""

from types import SimpleNamespace

import pytest

psutil = pytest.importorskip("psutil")
from mlframe.training.strategies import pipeline_cache as pc


def test_low_free_ram_gives_zero_budget_not_a_two_gib_floor(monkeypatch):
    """With 1 GiB free on a 64 GiB host the budget is 0, not the old 2 GiB floor that overrode the cap (41-F19)."""
    gib = 1024 * 1024 * 1024
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=64 * gib, available=1 * gib))
    assert pc._resolve_pipeline_cache_budget(0.5) == 0
