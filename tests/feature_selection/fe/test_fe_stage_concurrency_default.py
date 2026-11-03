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
