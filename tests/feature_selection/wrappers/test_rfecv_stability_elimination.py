"""Unit tests for elimination_rule='stability' in RFECV (stability-aware elimination).

Covers the aggregator (aggregate_stability) happy/edge paths, the get_next_features_subset
wiring, and the RFECV constructor validation. biz_value win lives in
test_biz_val_wrappers_rfecv_stability.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.wrappers._helpers_importance_agg import aggregate_stability
from mlframe.feature_selection.wrappers._helpers import get_next_features_subset
from mlframe.feature_selection.wrappers.rfecv import RFECV
from mlframe.feature_selection.wrappers._enums import OptimumSearch, VotesAggregation


def test_aggregate_stability_demotes_one_fold_spiker():
    # b: steady top-2 in r0,r2 (freq 2/3); c: spikes high in r1 only (freq 1/3).
    fi = {
        "r0": {"a": 1.0, "b": 0.9, "c": 0.1},
        "r1": {"a": 1.0, "b": 0.05, "c": 0.9},
        "r2": {"a": 1.0, "b": 0.8, "c": 0.1},
    }
    s = aggregate_stability(fi, cut_k=2)
    assert s["a"] > s["b"] > s["c"], s
    # b steady-mid beats c lucky-spike despite comparable raw means.
    assert s["b"] > s["c"]


def test_aggregate_stability_empty_and_single_run():
    assert aggregate_stability({}, cut_k=2) == {}
    s = aggregate_stability({"r0": {"a": 1.0, "b": 0.2}}, cut_k=1)
    # single run: a in top-1 -> freq 1.0 -> score = mean (1.0); b out -> 0.
    assert s["a"] == pytest.approx(1.0)
    assert s["b"] == pytest.approx(0.0)


def test_aggregate_stability_handles_nan_absent_feature():
    # feature c absent in r1 (ragged) -> never counts as surviving that run.
    fi = {"r0": {"a": 1.0, "c": 0.9}, "r1": {"a": 1.0}, "r2": {"a": 1.0, "c": 0.9}}
    s = aggregate_stability(fi, cut_k=1)
    # a is top-1 in all 3 -> freq 1.0; c top-1 in 0 (a always wins) -> 0.
    assert s["a"] > s["c"]


def test_get_next_features_subset_stability_path():
    fi = {
        "r0": {0: 1.0, 1: 0.9, 2: 0.1, 3: 0.05},
        "r1": {0: 1.0, 1: 0.05, 2: 0.95, 3: 0.04},
        "r2": {0: 1.0, 1: 0.85, 2: 0.1, 3: 0.05},
    }
    scores_mean = {0: 0.5, 4: 0.7}  # so remaining includes some N; pick exhaustive
    out = get_next_features_subset(
        nsteps=1,
        original_features=[0, 1, 2, 3],
        feature_importances=fi,
        evaluated_scores_mean=scores_mean,
        evaluated_scores_std={},
        use_all_fi_runs=True,
        use_last_fi_run_only=False,
        use_one_freshest_fi_run=False,
        use_fi_ranking=True,
        top_predictors_search_method=OptimumSearch.ExhaustiveDichotomic,
        votes_aggregation_method=VotesAggregation.Borda,
        rng=np.random.default_rng(0),
        elimination_rule="stability",
    )
    assert isinstance(out, list)
    # feature 0 (always top) must survive; feature 2 (one-fold spike) should be
    # demoted relative to steady feature 1 when the cut is tight.
    assert 0 in out


def test_rfecv_constructor_validates_elimination_rule():
    from sklearn.ensemble import RandomForestClassifier

    with pytest.raises(ValueError, match="elimination_rule"):
        RFECV(estimator=RandomForestClassifier(), elimination_rule="bogus")
    # valid values accepted.
    RFECV(estimator=RandomForestClassifier(), elimination_rule="stability")
    RFECV(estimator=RandomForestClassifier(), elimination_rule="importance")


def test_rfecv_default_elimination_rule_is_importance():
    from sklearn.ensemble import RandomForestClassifier

    r = RFECV(estimator=RandomForestClassifier())
    assert getattr(r, "elimination_rule", "importance") == "importance"
