"""Contract tests for the Phase-0 ``ArmResult`` interface and every fs_hybrid arm adapter.

The point of these tests is NOT that the arms select well -- it is that every arm answers with the same
shape and an HONEST ``score_kind``, so a ranking metric computed downstream is the same statistic for all
of them. Concretely: ``support`` is boolean and full-length for every arm, ``score`` is present exactly
when the declared kind says it is, a ``continuous`` arm really returns a finite full-length score, and
``all-features`` selects everything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid._arm_result import SCORED_KINDS, ArmResult
from mlframe.feature_selection._benchmarks.fs_hybrid._arms import (
    ACEArm,
    AllFeaturesArm,
    BorutaArm,
    BorutaShapArm,
    KnockoffArm,
    LarsPathArm,
    MRMRArm,
    RandomSelectionArm,
    RFECVArm,
    SelectFromModelArm,
    ShapProxiedArm,
    SklearnScoreArm,
    UnivariateMIArm,
    VarianceSortArm,
    build_arm_roster,
)

N_FEATURES = 8


@pytest.fixture(scope="module")
def tiny_bed():
    """Tiny 300x8 binary-classification frame: 3 informative columns, 5 pure noise."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, N_FEATURES)), columns=[f"f{i}" for i in range(N_FEATURES)])
    logit = 1.6 * X["f0"] + 1.2 * X["f1"] - 1.0 * X["f2"]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, y


# --------------------------------------------------------------------------------------- dataclass contract
def _ok_kwargs(**overrides):
    """Valid ``ArmResult`` constructor kwargs for a 4-feature arm, with overrides applied."""
    kwargs = dict(
        support=np.array([True, False, True, False]),
        score=None,
        score_kind="none",
        ranked_prefix=None,
        n_features_selected=2,
        selection_score=None,
        wall_time_s=0.1,
        process_time_s=0.1,
        n_model_fits=None,
        provenance={},
    )
    kwargs.update(overrides)
    return kwargs


def test_arm_result_rejects_non_boolean_support():
    """A support given as ints (a mask/indices mix-up) is refused, not coerced."""
    with pytest.raises(TypeError, match="1-D boolean"):
        ArmResult(**_ok_kwargs(support=np.array([1, 0, 1, 0])))


def test_arm_result_continuous_without_score_raises():
    """Declared ``continuous`` + ``score=None`` is FATAL, never a silent degradation to 'none'."""
    with pytest.raises(ValueError, match="score is None"):
        ArmResult(**_ok_kwargs(score_kind="continuous", score=None))


def test_arm_result_rejects_synthesised_score_on_unscored_kind():
    """A 'selection_order'/'none' arm may not publish a padded score vector."""
    with pytest.raises(ValueError, match="carries a score"):
        ArmResult(**_ok_kwargs(score_kind="none", score=np.array([1.0, 0.0, 1.0, 0.0])))


def test_arm_result_rejects_non_finite_continuous_score():
    """A continuous score with a NaN would poison the ranking metric; it is refused."""
    with pytest.raises(ValueError, match="non-finite"):
        ArmResult(**_ok_kwargs(score_kind="continuous", score=np.array([1.0, np.nan, 0.5, 0.2])))


def test_arm_result_selection_order_needs_ranked_prefix():
    """The order IS the only ranking signal a selection_order arm has, so it may not be omitted."""
    with pytest.raises(ValueError, match="ranked_prefix is None"):
        ArmResult(**_ok_kwargs(score_kind="selection_order", ranked_prefix=None))


def test_arm_result_support_count_must_match():
    """``n_features_selected`` is validated against ``support`` rather than trusted."""
    with pytest.raises(ValueError, match="disagrees with support"):
        ArmResult(**_ok_kwargs(n_features_selected=3))


# --------------------------------------------------------------------------------------- per-arm contract
def _check_arm(arm, X, y):
    """Assert the universal arm contract on a fitted result and return it."""
    result = arm.run(X, y)
    assert result.support.dtype == np.bool_
    assert result.support.shape == (X.shape[1],)
    assert result.n_features_in == X.shape[1]
    assert result.score_kind == arm.score_kind
    assert (result.score is not None) == (result.score_kind in SCORED_KINDS)
    if result.score is not None:
        assert result.score.shape == (X.shape[1],)
        assert np.all(np.isfinite(result.score))
    if result.score_kind == "selection_order":
        assert result.ranked_prefix is not None
    assert result.wall_time_s >= 0.0 and result.process_time_s >= 0.0
    assert result.provenance["arm"] == arm.name
    return result


ARMS = [
    ("all-features", lambda: AllFeaturesArm()),
    ("random-k", lambda: RandomSelectionArm(k=3)),
    ("variance-sort", lambda: VarianceSortArm(k=3)),
    ("univariate-mi", lambda: UnivariateMIArm()),
    ("skb-f", lambda: SklearnScoreArm("kbest_f", k=3)),
    ("skb-mi", lambda: SklearnScoreArm("kbest_mi", k=3)),
    ("select-fdr", lambda: SklearnScoreArm("fdr_f")),
    ("sfm-lgbm", lambda: SelectFromModelArm(n_estimators=30)),
    ("lars-order", lambda: LarsPathArm(max_features=4)),
    ("boruta", lambda: BorutaArm(n_iterations=6, n_estimators=25)),
    ("ace", lambda: ACEArm(n_replicates=3, n_masking_rounds=1, n_perm_repeats=2)),
    ("knockoffs", lambda: KnockoffArm(n_estimators=40)),
    ("mrmr", lambda: MRMRArm(max_runtime_mins=1.0)),
    ("rfecv", lambda: RFECVArm(max_refits=5, max_runtime_mins=1.0)),
    ("boruta-shap", lambda: BorutaShapArm(n_trials=8, n_estimators=25)),
    ("shap-proxied", lambda: ShapProxiedArm(top_n=5, min_features=2)),
]


# `_arm_name` is bound only so the tuple unpacks; the name reaches the report through `ids=`, not the body.
@pytest.mark.parametrize("_arm_name,factory", ARMS, ids=[n for n, _ in ARMS])
def test_arm_obeys_result_contract(_arm_name, factory, tiny_bed):
    """Every arm returns a full-length boolean support and a score consistent with its declared kind."""
    X, y = tiny_bed
    _check_arm(factory(), X, y)


def test_all_features_arm_selects_everything(tiny_bed):
    """The NULL HYPOTHESIS arm keeps every column and declares no ranking."""
    X, y = tiny_bed
    result = AllFeaturesArm().run(X, y)
    assert result.support.all()
    assert result.n_features_selected == X.shape[1]
    assert result.score is None and result.score_kind == "none"


def test_random_arm_respects_cardinality(tiny_bed):
    """The random control selects EXACTLY the caller-given cardinality (recall control)."""
    X, y = tiny_bed
    assert RandomSelectionArm(k=3, random_state=1).run(X, y).n_features_selected == 3


def test_variance_sort_ranks_by_marginal_variance(tiny_bed):
    """The varsortability tripwire's score is the columnwise variance, in input-column order."""
    X, y = tiny_bed
    result = VarianceSortArm(k=3).run(X, y)
    np.testing.assert_allclose(result.score, np.var(X.to_numpy(), axis=0), rtol=1e-9)
    assert result.n_features_selected == 3


def test_roster_builds_every_arm_unfitted():
    """``build_arm_roster`` returns fresh, distinct instances so a cell can never reuse a fitted arm."""
    roster = build_arm_roster(N_FEATURES, k=3)
    assert "all-features" in roster
    for name, factory in roster.items():
        first, second = factory(), factory()
        assert first is not second
        assert first.score_kind in ("continuous", "ordinal", "selection_order", "none"), name
