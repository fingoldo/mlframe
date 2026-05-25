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

import sys

import pytest

lgb = pytest.importorskip("lightgbm")

from mlframe.training import lgb_shim


def test_lgb_dataset_reuse_capable_returns_true_on_supported_build():
    # LightGBM 3.x and later carry both mutators. The shim ships only
    # with lgb >= 3 supported, so a False here is a real regression.
    assert lgb_shim.lgb_dataset_reuse_capable() is True


def test_lgb_dataset_reuse_capable_false_when_lgb_unavailable(monkeypatch):
    # Flip the module-level availability flag the shim consults; the
    # function must short-circuit to False without touching `lgb`.
    monkeypatch.setattr(lgb_shim, "_LGB_AVAILABLE", False)
    assert lgb_shim.lgb_dataset_reuse_capable() is False


def test_lgb_dataset_reuse_capable_false_when_set_label_missing(monkeypatch):
    # Simulate a future build that removed set_label. The probe should
    # return False, not raise. We swap in a stand-in Dataset class with
    # only set_weight so hasattr finds one mutator but not both.
    class _NoSetLabelDataset:
        def set_weight(self, *_a, **_kw):
            return None

    monkeypatch.setattr(lgb_shim, "_LGB_AVAILABLE", True)
    monkeypatch.setattr(lgb_shim.lgb, "Dataset", _NoSetLabelDataset)
    assert lgb_shim.lgb_dataset_reuse_capable() is False


def test_lgb_dataset_reuse_capable_false_when_set_weight_missing(monkeypatch):
    class _NoSetWeightDataset:
        def set_label(self, *_a, **_kw):
            return None

    monkeypatch.setattr(lgb_shim, "_LGB_AVAILABLE", True)
    monkeypatch.setattr(lgb_shim.lgb, "Dataset", _NoSetWeightDataset)
    assert lgb_shim.lgb_dataset_reuse_capable() is False


def test_lgb_dataset_reuse_capable_repeated_calls_are_consistent():
    # The probe is stateless; repeated calls return identical bools.
    first = lgb_shim.lgb_dataset_reuse_capable()
    second = lgb_shim.lgb_dataset_reuse_capable()
    third = lgb_shim.lgb_dataset_reuse_capable()
    assert first is second is third
