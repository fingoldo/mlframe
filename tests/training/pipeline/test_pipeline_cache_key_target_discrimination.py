"""Regression test: PipelineCache key must discriminate by target when a named pre_pipeline is used.

Without a target-content discriminator, ``_compute_pipeline_cache_key`` was built purely from
(strategy flags, pre_pipeline_name, tier, kind, cat/text/embedding ROLE lists, dtype) -- all of
which stay identical across DIFFERENT targets sharing the same X schema. A target-dependent
pre_pipeline (MRMR / RFECV feature selection) selects and engineers a DIFFERENT column set per
target, so a multi-target suite's second target got a cache HIT serving the first target's
transformed frame (wrong column count/set), which downstream raised a CatBoostEncoder
"Unexpected input dimension" crash (fuzz c0023).

Reproduces the exact pre-fix collision: two calls with identical everything except
``train_target`` (and ``target_name``) must now produce different keys when ``pre_pipeline_name``
is set. When there is no named pre_pipeline (pure imp/scale/encode, target-independent), the key
must stay IDENTICAL across targets to preserve the legitimate cross-target cache-sharing win.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from mlframe.training.core._phase_train_one_target import _compute_pipeline_cache_key


_BASE_KW = dict(
    strategy_cache_key="imp1_scale1_enc1",
    feature_tier=(False, False),
    supports_polars=False,
    cat_features=["cat_0"],
    text_features=[],
    embedding_features=[],
)


def test_different_targets_with_named_pre_pipeline_get_different_keys():
    """Different targets with named pre pipeline get different keys."""
    target_a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    target_b = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
    k_a = _compute_pipeline_cache_key(
        **_BASE_KW,
        pre_pipeline_name="mrmr",
        target_name="target_a",
        train_target=target_a,
    )
    k_b = _compute_pipeline_cache_key(
        **_BASE_KW,
        pre_pipeline_name="mrmr",
        target_name="target_b",
        train_target=target_b,
    )
    assert k_a != k_b, f"cache key collided across distinct targets with a named pre_pipeline: {k_a} == {k_b}"


def test_same_target_content_and_name_is_stable():
    """Same target content and name is stable."""
    target = np.array([1.0, 2.0, 3.0])
    k1 = _compute_pipeline_cache_key(**_BASE_KW, pre_pipeline_name="mrmr", target_name="t", train_target=target)
    k2 = _compute_pipeline_cache_key(**_BASE_KW, pre_pipeline_name="mrmr", target_name="t", train_target=target)
    assert k1 == k2


def test_no_pre_pipeline_stays_target_independent():
    # Pure imp/scale/encode preprocessing has no target-dependent selection, so different targets
    # must still share the cache slot -- this is the legitimate cross-target win the fix must not break.
    """No pre pipeline stays target independent."""
    target_a = np.array([0.0, 1.0, 2.0])
    target_b = np.array([9.0, 8.0, 7.0])
    k_a = _compute_pipeline_cache_key(**_BASE_KW, pre_pipeline_name=None, target_name="target_a", train_target=target_a)
    k_b = _compute_pipeline_cache_key(**_BASE_KW, pre_pipeline_name=None, target_name="target_b", train_target=target_b)
    assert k_a == k_b, "no-pre_pipeline key must stay target-independent to preserve cross-target sharing"
