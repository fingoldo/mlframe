"""Regression test for FH target-encoder cache collision.

Pre-fix ``_apply_target_encoder`` always built ``InMemoryKey(..., train_idx_token=0, ...)`` with the
literal ``0``. A multi-target suite using ``target_mean`` / ``woe`` encoders for two different y's
hit the same cache slot for the same (df_token, column, params) -- so target-2's encoder lookup
returned target-1's fitted encoder + OOF, silently leaking target-1's signal.

Post-fix ``train_idx_token`` is a blake2b-derived 63-bit fingerprint of the y content. Distinct
targets produce distinct keys; the cache holds two entries; encoders / OOF diverge.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mlframe.training.feature_handling import (
    CacheConfig,
    FeatureCache,
    TargetEncodeParams,
    current_session,
    reset_session,
)
from mlframe.training.feature_handling.apply import (
    _apply_target_encoder,
    _target_content_token,
)


@pytest.fixture(autouse=True)
def _fresh_session():
    """Each test gets a clean session token so InMemoryKey lookups don't pick up cross-test entries."""
    reset_session()
    yield
    reset_session()


def test_target_content_token_distinguishes_distinct_targets():
    """Target content token distinguishes distinct targets."""
    y_a = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    y_b = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    assert _target_content_token(y_a) != _target_content_token(y_b)


def test_target_content_token_stable_for_identical_inputs():
    """Target content token stable for identical inputs."""
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    assert _target_content_token(y) == _target_content_token(y.copy())


def test_target_content_token_handles_pandas_series():
    """Target content token handles pandas series."""
    import pandas as pd

    s = pd.Series([0, 1, 0, 1], name="target", dtype=np.int64)
    t = pd.Series([0, 1, 0, 1], name="target", dtype=np.int64)
    assert _target_content_token(s) == _target_content_token(t)
    s2 = pd.Series([1, 0, 1, 0], name="target", dtype=np.int64)
    assert _target_content_token(s) != _target_content_token(s2)


def test_target_encoder_cache_does_not_collide_across_targets():
    """End-to-end: same df + same column + same params + DIFFERENT y must produce two cache entries
    and two distinct OOF outputs. Pre-fix the second call hit the first's slot.
    """
    n = 100
    rng = np.random.RandomState(0)
    train_df = pl.DataFrame(
        {
            "country": rng.choice(["US", "UK", "DE", "FR"], size=n).tolist(),
        }
    )
    # Two different binary targets correlated with the cat column in opposite ways.
    cat_arr = np.array(train_df["country"].to_list())
    y_a = (cat_arr == "US").astype(np.int32)
    y_b = (cat_arr == "DE").astype(np.int32)

    cache = FeatureCache(CacheConfig(persistence="off"))
    sess = current_session()
    params = TargetEncodeParams(kind="target_mean", smoothing=10.0, cv=3, prior="mean", random_state=0)

    train_a, _, _ = _apply_target_encoder(
        train_df=train_df,
        val_df=None,
        test_df=None,
        column="country",
        params=params,
        train_target=y_a,
        cache=cache,
        session_id=sess.session_id,
        train_id=id(train_df),
    )
    train_b, _, _ = _apply_target_encoder(
        train_df=train_df,
        val_df=None,
        test_df=None,
        column="country",
        params=params,
        train_target=y_b,  # DIFFERENT target on the SAME df/column/params slot
        cache=cache,
        session_id=sess.session_id,
        train_id=id(train_df),
    )

    # Two distinct cache entries (pre-fix: 1).
    stats = cache.stats()
    assert stats["n_keys"] == 2, f"distinct targets must occupy distinct cache slots; cache has {stats['n_keys']} entries"

    # OOF encodings must differ -- target-1's signal must NOT leak into target-2's output.
    a_arr = train_a.data.ravel()
    b_arr = train_b.data.ravel()
    assert a_arr.shape == b_arr.shape
    assert not np.allclose(a_arr, b_arr), "target-encoder OOF must differ across distinct targets; pre-fix the second call replayed the first"
