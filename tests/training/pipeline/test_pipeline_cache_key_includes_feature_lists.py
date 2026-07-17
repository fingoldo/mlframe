"""Regression test: PipelineCache key must include (cat, text, embedding) feature lists.

Without folding the feature lists into the key, a cache HIT could serve a tier frame
prepared for a different (cat/text/embedding) split between sessions, returning stale
results. The fix sorts each list and folds the resulting tuples through blake2b into a
digest suffix so the key invalidates on any membership change, stays stable across list
ordering, and (unlike a raw frozenset.__repr__) stays stable across PYTHONHASHSEED.

Imports the production helper ``_compute_pipeline_cache_key`` so any future regression
that drops the feature-digest will fail this test rather than passing against a stale
re-implementation in the test file.
"""

from __future__ import annotations

import pytest

# pytest.importorskip keeps the test runnable on lean CI without heavy training deps.
pytest.importorskip("sklearn")

from mlframe.training.core._phase_train_one_target import _compute_pipeline_cache_key


_BASE_KW = dict(
    strategy_cache_key="tree",
    pre_pipeline_name="pp",
    feature_tier=(False, False),
    supports_polars=False,
)


def test_cat_features_membership_changes_key():
    # Adding a column to cat_features must produce a different key, otherwise a stale tier frame
    # prepared for cat_features=["A"] would be served when the user later set ["A","B"].
    k1 = _compute_pipeline_cache_key(**_BASE_KW, cat_features=["A"], text_features=[], embedding_features=[])
    k2 = _compute_pipeline_cache_key(**_BASE_KW, cat_features=["A", "B"], text_features=[], embedding_features=[])
    assert k1 != k2, f"cache key did not change on cat_features mutation: {k1} == {k2}"


def test_cat_features_order_invariant():
    # Sorted-tuple folding must make column order irrelevant; otherwise upstream pipelines that
    # happen to reorder a list (sorted vs config-order) would needlessly miss cache.
    k_ab = _compute_pipeline_cache_key(**_BASE_KW, cat_features=["A", "B"], text_features=[], embedding_features=[])
    k_ba = _compute_pipeline_cache_key(**_BASE_KW, cat_features=["B", "A"], text_features=[], embedding_features=[])
    assert k_ab == k_ba, f"cache key not order-invariant for cat_features: {k_ab} != {k_ba}"


def test_text_features_membership_changes_key():
    k1 = _compute_pipeline_cache_key(**_BASE_KW, cat_features=[], text_features=["T1"], embedding_features=[])
    k2 = _compute_pipeline_cache_key(**_BASE_KW, cat_features=[], text_features=["T1", "T2"], embedding_features=[])
    assert k1 != k2


def test_text_features_order_invariant():
    k_ab = _compute_pipeline_cache_key(**_BASE_KW, cat_features=[], text_features=["T1", "T2"], embedding_features=[])
    k_ba = _compute_pipeline_cache_key(**_BASE_KW, cat_features=[], text_features=["T2", "T1"], embedding_features=[])
    assert k_ab == k_ba


def test_embedding_features_membership_changes_key():
    k1 = _compute_pipeline_cache_key(**_BASE_KW, cat_features=[], text_features=[], embedding_features=["E1"])
    k2 = _compute_pipeline_cache_key(**_BASE_KW, cat_features=[], text_features=[], embedding_features=["E1", "E2"])
    assert k1 != k2


def test_embedding_features_order_invariant():
    k_ab = _compute_pipeline_cache_key(**_BASE_KW, cat_features=[], text_features=[], embedding_features=["E1", "E2"])
    k_ba = _compute_pipeline_cache_key(**_BASE_KW, cat_features=[], text_features=[], embedding_features=["E2", "E1"])
    assert k_ab == k_ba


def test_none_and_empty_list_equivalent():
    # The production guard uses ``cat_features or ()`` so None and [] must produce the same key,
    # which matches how callers historically swap between sentinels.
    k_none = _compute_pipeline_cache_key(**_BASE_KW, cat_features=None, text_features=None, embedding_features=None)
    k_empty = _compute_pipeline_cache_key(**_BASE_KW, cat_features=[], text_features=[], embedding_features=[])
    assert k_none == k_empty


def test_cross_list_swap_changes_key():
    # Moving a column between cat/text/embedding (e.g. user promotes "X" from cat to text) must
    # invalidate the key - the per-list digests prevent the totals from masking the move.
    k_cat = _compute_pipeline_cache_key(**_BASE_KW, cat_features=["X"], text_features=[], embedding_features=[])
    k_text = _compute_pipeline_cache_key(**_BASE_KW, cat_features=[], text_features=["X"], embedding_features=[])
    k_emb = _compute_pipeline_cache_key(**_BASE_KW, cat_features=[], text_features=[], embedding_features=["X"])
    assert len({k_cat, k_text, k_emb}) == 3, f"per-list digest collapsed: {k_cat=} {k_text=} {k_emb=}"


def test_other_dimensions_still_distinguish():
    # Tier, kind, strategy_cache_key, and pre_pipeline_name must still vary the key. Without this
    # the new features-digest suffix could mask a bug where one of the older suffixes was dropped.
    base = _compute_pipeline_cache_key(
        strategy_cache_key="tree",
        pre_pipeline_name="pp",
        feature_tier=(False, False),
        supports_polars=False,
        cat_features=["A"],
        text_features=[],
        embedding_features=[],
    )
    diff_tier = _compute_pipeline_cache_key(
        strategy_cache_key="tree",
        pre_pipeline_name="pp",
        feature_tier=(True, False),
        supports_polars=False,
        cat_features=["A"],
        text_features=[],
        embedding_features=[],
    )
    diff_kind = _compute_pipeline_cache_key(
        strategy_cache_key="tree",
        pre_pipeline_name="pp",
        feature_tier=(False, False),
        supports_polars=True,
        cat_features=["A"],
        text_features=[],
        embedding_features=[],
    )
    diff_strategy = _compute_pipeline_cache_key(
        strategy_cache_key="linear",
        pre_pipeline_name="pp",
        feature_tier=(False, False),
        supports_polars=False,
        cat_features=["A"],
        text_features=[],
        embedding_features=[],
    )
    diff_pp = _compute_pipeline_cache_key(
        strategy_cache_key="tree",
        pre_pipeline_name="other",
        feature_tier=(False, False),
        supports_polars=False,
        cat_features=["A"],
        text_features=[],
        embedding_features=[],
    )
    assert len({base, diff_tier, diff_kind, diff_strategy, diff_pp}) == 5
