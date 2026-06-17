"""Unit tests for distribution-driven composite-estimator recommendation (E3)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.training.composite._estimator_dispatch import (
    _pick_base_column,
    instantiate_recommended_estimator,
    maybe_inject_distribution_driven_estimator,
    recommend_composite_estimator,
)
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


def test_instantiate_recommended_estimator_builds_class():
    rec = recommend_composite_estimator(["heavy_tail(excess_kurt=20)"])
    est = instantiate_recommended_estimator(rec)
    assert type(est).__name__ == "TailCompositeEstimator"


def test_instantiate_none_returns_none():
    assert instantiate_recommended_estimator(None) is None


def test_pick_base_column_selects_max_corr_feature():
    rng = np.random.default_rng(0)
    n = 400
    f_signal = rng.normal(size=n)
    f_noise = rng.normal(size=n)
    y = 3.0 * f_signal + 0.1 * rng.normal(size=n)
    df = pd.DataFrame({"f_noise": f_noise, "f_signal": f_signal})
    assert _pick_base_column(df, y) == "f_signal"


def test_pick_base_column_none_when_no_numeric_or_constant():
    df = pd.DataFrame({"c": np.ones(10)})  # zero-variance column
    assert _pick_base_column(df, np.arange(10.0)) is None
    assert _pick_base_column(pd.DataFrame(index=range(5)), np.arange(5.0)) is None


def _ctx():
    return SimpleNamespace(strategy_by_model={}, sorted_mlframe_models=[], mlframe_models=["xgb"])


def test_maybe_inject_noop_when_flag_off():
    ctx = _ctx()
    out = maybe_inject_distribution_driven_estimator(
        ctx=ctx,
        metadata={"target_distribution_report": {"pathologies": ["heavy_tail(excess_kurt=20)"]}},
        mlframe_models=["xgb"],
        target_by_type=None,
        train_idx=None,
        train_df=None,
        behavior_config=SimpleNamespace(distribution_driven_estimator=False),
    )
    assert out == ["xgb"]


def test_maybe_inject_noop_when_no_recommendation():
    ctx = _ctx()
    out = maybe_inject_distribution_driven_estimator(
        ctx=ctx,
        metadata={"target_distribution_report": {"pathologies": ["class_imbalance(max/min=5x)"]}},
        mlframe_models=["xgb"],
        target_by_type=None,
        train_idx=None,
        train_df=None,
        behavior_config=SimpleNamespace(distribution_driven_estimator=True),
    )
    assert out == ["xgb"]


def test_maybe_inject_appends_estimator_and_updates_ctx():
    from mlframe.training._configs_base import TargetTypes
    from mlframe.training.composite.extremes import TailCompositeEstimator

    rng = np.random.default_rng(1)
    n = 300
    f0 = rng.normal(size=n)
    y = 2.0 * f0 + rng.normal(size=n)
    train_df = pd.DataFrame({"f0": f0, "f1": rng.normal(size=n)})
    ctx = _ctx()
    out = maybe_inject_distribution_driven_estimator(
        ctx=ctx,
        metadata={"target_distribution_report": {"pathologies": ["heavy_tail(excess_kurt=20)"]}},
        mlframe_models=["xgb"],
        target_by_type={TargetTypes.REGRESSION: {"t": y}},
        train_idx=np.arange(n),
        train_df=train_df,
        behavior_config=SimpleNamespace(distribution_driven_estimator=True),
    )
    assert len(out) == 2
    assert isinstance(out[1], TailCompositeEstimator)
    assert out[1].base_column == "f0"
    assert id(out[1]) in ctx.strategy_by_model
    assert ctx.mlframe_models is out or ctx.mlframe_models == out
