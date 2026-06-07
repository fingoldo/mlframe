"""The post-pipeline forced 2x gc.collect must be size-gated.

The forced collect relieves Windows commit-charge pressure before composite-discovery, which only matters once the
released Polars frame is large; the collect itself costs ~0.85s (object-count driven, not frame-size), so on small
frames it is pure overhead (~17% of a 1k-row suite run). It is gated on the released-frame size.
"""
from __future__ import annotations

from mlframe.training.core import _phase_helpers as ph


def test_forced_gc_skipped_on_small_frame():
    assert ph._should_force_post_pipeline_gc(10.0) is False
    assert ph._should_force_post_pipeline_gc(None) is False
    assert ph._should_force_post_pipeline_gc(0) is False


def test_forced_gc_fires_on_large_frame():
    assert ph._should_force_post_pipeline_gc(ph._FORCED_GC_MIN_DF_MB) is True
    assert ph._should_force_post_pipeline_gc(ph._FORCED_GC_MIN_DF_MB + 1) is True
    assert ph._should_force_post_pipeline_gc(50_000.0) is True


def test_forced_gc_threshold_zero_restores_legacy_always_force(monkeypatch):
    monkeypatch.setattr(ph, "_FORCED_GC_MIN_DF_MB", 0.0)
    assert ph._should_force_post_pipeline_gc(1.0) is True
    assert ph._should_force_post_pipeline_gc(0) is True
