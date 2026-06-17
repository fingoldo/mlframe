"""Unit tests for distribution-driven composite-estimator recommendation (E3)."""

from __future__ import annotations

from types import SimpleNamespace

from mlframe.training.composite._estimator_dispatch import recommend_composite_estimator
from mlframe.training.core._phase_finalize import _stamp_composite_estimator_recommendation


def test_heavy_tail_recommends_tail_composite():
    rec = recommend_composite_estimator(["heavy_tail(excess_kurt=12.3)"])
    assert rec["estimator"] == "TailCompositeEstimator"
    assert "extremes" in rec["module"]


def test_skew_also_recommends_tail_composite():
    rec = recommend_composite_estimator(["skewed_target(skew=3.1)"])
    assert rec["estimator"] == "TailCompositeEstimator"


def test_multimodal_recommends_distribution():
    rec = recommend_composite_estimator(["multi_modal_target(peaks=3, max_sep=4.0 stds)"])
    assert rec["estimator"] == "CompositeDistributionEstimator"


def test_heavy_tail_priority_over_multimodal():
    rec = recommend_composite_estimator(["multi_modal_target(peaks=2)", "heavy_tail(excess_kurt=9)"])
    assert rec["estimator"] == "TailCompositeEstimator"


def test_no_match_returns_none():
    assert recommend_composite_estimator(["class_imbalance(max/min=5x)"]) is None
    assert recommend_composite_estimator([]) is None


def test_finalize_stamps_recommendation():
    ctx = SimpleNamespace(
        metadata={"target_distribution_report": {"pathologies": ["heavy_tail(excess_kurt=20)"]}},
        verbose=0,
    )
    _stamp_composite_estimator_recommendation(ctx)
    assert ctx.metadata["composite_estimator_recommendation"]["estimator"] == "TailCompositeEstimator"


def test_finalize_noop_without_analyzer_report():
    ctx = SimpleNamespace(metadata={}, verbose=0)
    _stamp_composite_estimator_recommendation(ctx)
    assert "composite_estimator_recommendation" not in ctx.metadata
