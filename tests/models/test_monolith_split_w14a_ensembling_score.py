"""Sensor: _ensembling_score.py W14A flavours carve preserves identity + facade under budget + behavioural equivalence.

The carve lifts 7 sub-bodies from ``score_ensemble`` into ``_ensembling_score_flavours.py``:
- ``build_member_tag_lists`` (tag-build, 2 lists)
- ``apply_quality_gate_kn`` (K>2 quality gate)
- ``apply_diversity_drop`` (auto-drop high-corr pairs)
- ``filter_sign_sensitive_flavours`` (regression sign-flip filter)
- ``collapse_to_single_flavour_if_identical`` (early exit when all-equal)
- ``run_stacking_aware_gate`` (composite_stacking NNLS gate)
- ``maybe_build_votenrank_leaderboard`` (tail votenrank build)

Behavioural equivalence is preserved by lifting bodies verbatim; this sensor pins
the identity of the public helpers + the parent's surviving LOC budget.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_w14a_ensembling_score_facade_under_budget():
    parent = Path(__file__).parent.parent.parent / "src" / "mlframe" / "models" / "_ensembling_score.py"
    facade_loc = sum(1 for _ in parent.open(encoding="utf-8"))
    assert facade_loc < 500, f"_ensembling_score.py LOC={facade_loc} exceeds 500 budget"


def test_w14a_ensembling_score_flavours_identity():
    from mlframe.models import _ensembling_score as parent
    from mlframe.models import _ensembling_score_flavours as flav

    # Every helper imported into the parent must be the same object exported from the sibling.
    assert parent.build_member_tag_lists is flav.build_member_tag_lists
    assert parent.apply_quality_gate_kn is flav.apply_quality_gate_kn
    assert parent.apply_diversity_drop is flav.apply_diversity_drop
    assert parent.filter_sign_sensitive_flavours is flav.filter_sign_sensitive_flavours
    assert parent.collapse_to_single_flavour_if_identical is flav.collapse_to_single_flavour_if_identical
    assert parent.run_stacking_aware_gate is flav.run_stacking_aware_gate
    assert parent.maybe_build_votenrank_leaderboard is flav.maybe_build_votenrank_leaderboard


def test_w14a_build_member_tag_lists_with_name_attr():
    """``model_name`` (TVT augmented) feeds the full-tag list; underlying class feeds the short-tag list."""
    from mlframe.models._ensembling_score_flavours import build_member_tag_lists

    class _FakeReg:
        pass

    m1 = SimpleNamespace(model_name="CatBoostRegressor TVT MTTR=11.2", model=_FakeReg())
    m2 = SimpleNamespace(model_name=None, model=_FakeReg())
    member_tags, short_tags = build_member_tag_lists([m1, m2])
    assert len(member_tags) == 2
    assert len(short_tags) == 2
    # Tag list returns strings (no crashes on either branch).
    assert all(isinstance(t, str) for t in member_tags)
    assert all(isinstance(t, str) for t in short_tags)


def test_w14a_filter_sign_sensitive_flavours_drops_harm_geo_quad_on_sign_changing():
    """Regression flavour filter must drop harm / geo / quad when preds span zero."""
    from mlframe.models._ensembling_score_flavours import filter_sign_sensitive_flavours

    m1 = SimpleNamespace(val_preds=np.array([-2.0, -1.0, 0.5, 2.0]), test_preds=None, train_preds=None)
    m2 = SimpleNamespace(val_preds=np.array([-1.5, 0.8, 1.0, 1.5]), test_preds=None, train_preds=None)
    res = filter_sign_sensitive_flavours(
        ensembling_methods=["arithm", "harm", "geo", "quad", "qube", "rrf"],
        is_regression=True,
        level_models_and_predictions=[m1, m2],
        verbose=False,
    )
    assert "harm" not in res
    assert "geo" not in res
    assert "quad" not in res
    assert "arithm" in res
    assert "qube" in res
    assert "rrf" in res


def test_w14a_filter_sign_sensitive_flavours_keeps_all_positive():
    """All-positive preds should not trip the sign-flip filter."""
    from mlframe.models._ensembling_score_flavours import filter_sign_sensitive_flavours

    m1 = SimpleNamespace(val_preds=np.array([1.0, 2.0, 3.0]), test_preds=None, train_preds=None)
    m2 = SimpleNamespace(val_preds=np.array([1.5, 2.5, 3.5]), test_preds=None, train_preds=None)
    res = filter_sign_sensitive_flavours(
        ensembling_methods=["arithm", "harm", "geo", "quad"],
        is_regression=True,
        level_models_and_predictions=[m1, m2],
        verbose=False,
    )
    assert res == ["arithm", "harm", "geo", "quad"]


def test_w14a_filter_sign_sensitive_skips_classification():
    """Classification suites bypass the sign-flip filter even with sign-changing preds."""
    from mlframe.models._ensembling_score_flavours import filter_sign_sensitive_flavours

    m1 = SimpleNamespace(val_preds=np.array([-1.0, 0.0, 1.0]), test_preds=None, train_preds=None)
    res = filter_sign_sensitive_flavours(
        ensembling_methods=["arithm", "harm"],
        is_regression=False,
        level_models_and_predictions=[m1],
        verbose=False,
    )
    assert res == ["arithm", "harm"]


def test_w14a_collapse_identical_returns_arithm_when_all_close():
    """When all member preds are byte-identical and early_exit_if_identical=True, collapse to arithm."""
    from mlframe.models._ensembling_score_flavours import collapse_to_single_flavour_if_identical

    ref = np.array([0.1, 0.2, 0.3, 0.4])
    res_dict: dict = {}
    res = collapse_to_single_flavour_if_identical(
        ensembling_methods=["arithm", "harm", "geo"],
        early_exit_if_identical=True,
        _gate_preds_for_check=[ref, ref.copy(), ref.copy()],
        level_models_and_predictions=[1, 2, 3],
        _gate_source_split="val",
        res=res_dict,
        verbose=False,
    )
    assert res == ["arithm"]
    assert res_dict["_diversity"]["all_members_identical"] is True
    assert res_dict["_diversity"]["collapsed_to_single_flavour"] == "arithm"


def test_w14a_collapse_identical_disabled_by_default():
    """When early_exit_if_identical=False, the methods list passes through untouched."""
    from mlframe.models._ensembling_score_flavours import collapse_to_single_flavour_if_identical

    ref = np.array([0.1, 0.2, 0.3])
    res = collapse_to_single_flavour_if_identical(
        ensembling_methods=["arithm", "harm"],
        early_exit_if_identical=False,
        _gate_preds_for_check=[ref, ref.copy()],
        level_models_and_predictions=[1, 2],
        _gate_source_split="val",
        res={},
        verbose=False,
    )
    assert res == ["arithm", "harm"]
