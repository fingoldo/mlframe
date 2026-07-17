"""Capability-gate tests for `lgb_dataset_reuse_capable`.

The shim's `lgb_dataset_reuse_capable()` is a runtime probe that decides
whether the in-process LightGBM build exposes the `Dataset.set_label` and
`Dataset.set_weight` mutators required for cross-fit Dataset reuse. The
public API is one bool predicate, but its contract has three behaviours:
(a) lgb installed and both mutators present -> True,
(b) lgb missing -> False,
(c) lgb installed but mutators removed/renamed -> False.
"""

from __future__ import annotations


import pytest

lgb = pytest.importorskip("lightgbm")

from mlframe.training import lgb_shim


def test_lgb_dataset_reuse_capable_returns_true_on_supported_build():
    # LightGBM 3.x and later carry both mutators. The shim ships only
    # with lgb >= 3 supported, so a False here is a real regression.
    """Lgb dataset reuse capable returns true on supported build."""
    assert lgb_shim.lgb_dataset_reuse_capable() is True


def test_lgb_dataset_reuse_capable_false_when_lgb_unavailable(monkeypatch):
    # Flip the module-level availability flag the shim consults; the
    # function must short-circuit to False without touching `lgb`.
    """Lgb dataset reuse capable false when lgb unavailable."""
    monkeypatch.setattr(lgb_shim, "_LGB_AVAILABLE", False)
    assert lgb_shim.lgb_dataset_reuse_capable() is False


def test_lgb_dataset_reuse_capable_false_when_set_label_missing(monkeypatch):
    # Simulate a future build that removed set_label. The probe should
    # return False, not raise. We swap in a stand-in Dataset class with
    # only set_weight so hasattr finds one mutator but not both.
    """Lgb dataset reuse capable false when set label missing."""
    class _NoSetLabelDataset:
        """Groups tests covering no set label dataset."""
        def set_weight(self, *_a, **_kw):
            """Set weight."""
            return None

    monkeypatch.setattr(lgb_shim, "_LGB_AVAILABLE", True)
    monkeypatch.setattr(lgb_shim.lgb, "Dataset", _NoSetLabelDataset)
    assert lgb_shim.lgb_dataset_reuse_capable() is False


def test_lgb_dataset_reuse_capable_false_when_set_weight_missing(monkeypatch):
    """Lgb dataset reuse capable false when set weight missing."""
    class _NoSetWeightDataset:
        """Groups tests covering no set weight dataset."""
        def set_label(self, *_a, **_kw):
            """Set label."""
            return None

    monkeypatch.setattr(lgb_shim, "_LGB_AVAILABLE", True)
    monkeypatch.setattr(lgb_shim.lgb, "Dataset", _NoSetWeightDataset)
    assert lgb_shim.lgb_dataset_reuse_capable() is False


def test_lgb_dataset_reuse_capable_repeated_calls_are_consistent():
    # The probe is stateless; repeated calls return identical bools.
    """Lgb dataset reuse capable repeated calls are consistent."""
    first = lgb_shim.lgb_dataset_reuse_capable()
    second = lgb_shim.lgb_dataset_reuse_capable()
    third = lgb_shim.lgb_dataset_reuse_capable()
    assert first is second is third
