"""Cluster 29 regression: a failing KTC cache lookup is a WARNING and the fallback backend is used (audit 2026-11-20, finding 29-F38)."""

import logging

import pytest

from mlframe.votenrank import _confidence_gated_blend_ktc_dispatch as disp


class _BrokenCache:
    """Cache whose lookup raises, as a corrupt or unreadable KTC store would."""

    def get_or_tune(self, *a, **k):
        """Simulate a broken lookup."""
        raise RuntimeError("ktc store corrupt")


def test_failed_lookup_warns_and_returns_fallback(monkeypatch, caplog):
    """A broken KTC lookup logs at WARNING and returns the caller's fallback (29-F38)."""
    pytest.importorskip("numba")
    monkeypatch.setattr(disp, "forced_backend", lambda *a, **k: None)
    monkeypatch.setattr(disp, "get_ktc_cache", lambda: _BrokenCache())
    with caplog.at_level(logging.WARNING, logger=disp.logger.name):
        out = disp.choose_confidence_blend_backend(1000, fallback="numpy")
    assert out == "numpy"
    assert any(r.levelno == logging.WARNING and "KTC lookup failed" in r.getMessage() for r in caplog.records)
