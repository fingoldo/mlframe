"""Regression tests for the pre-pipeline LRU cache (fix P0 _pre_pipeline_cache_*).

Pre-fix: cache keyed by ``(id(train_df), id(val_df), pipeline_signature)``
collided across targets because the suite's per-target loop kept the same
``filtered_train_df`` object alive while ``train_target`` varied. Post-fix
the key carries a content fingerprint of train_df / val_df / train_target
+ a target name, so target-2 cannot serve target-1's fit-transform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.training.pipeline._pipeline_helpers import (
    _PRE_PIPELINE_CACHE,
    _pre_pipeline_cache_clear,
    _pre_pipeline_cache_get,
    _pre_pipeline_cache_set,
)


def _make_pipeline() -> Pipeline:
    """A structurally identical pipeline across both fake fits so the
    only thing that distinguishes the two cache keys is the target /
    target name. If the cache cross-confuses, fix is broken."""
    return Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="mean")),
            ("scl", StandardScaler()),
        ]
    )


def test_pre_pipeline_cache_isolates_targets_by_content():
    """Same X object, different y, same pipeline -> separate slots."""
    _pre_pipeline_cache_clear()

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(64, 4)), columns=["a", "b", "c", "d"])
    y_a = pd.Series(rng.normal(size=64), name="target_a")
    y_b = pd.Series(rng.normal(size=64) + 100.0, name="target_b")

    pipe = _make_pipeline()

    # Stash a sentinel for target_a; the second target must NOT see it.
    sentinel_a = ("train_a", "val_a")
    _pre_pipeline_cache_set(
        X,
        None,
        pipe,
        sentinel_a[0],
        sentinel_a[1],
        train_target=y_a.to_numpy(),
        target_name="target_a",
    )
    # Same X object identity, same pipeline, different target.
    hit_b = _pre_pipeline_cache_get(
        X,
        None,
        pipe,
        train_target=y_b.to_numpy(),
        target_name="target_b",
    )
    assert hit_b is None, "cache must NOT serve target_a's slot to target_b"

    # Sanity: target_a still resolves cleanly.
    hit_a = _pre_pipeline_cache_get(
        X,
        None,
        pipe,
        train_target=y_a.to_numpy(),
        target_name="target_a",
    )
    # Cache entry is 3-tuple (train_out, val_out, fitted_pipeline) since 2026-05-16.
    assert hit_a is not None
    assert hit_a[0] == sentinel_a[0]
    assert hit_a[1] == sentinel_a[1]


def test_pre_pipeline_cache_hits_when_inputs_match():
    """Same X content + same y content -> cache hit, the fast path."""
    _pre_pipeline_cache_clear()

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(32, 3)), columns=["x", "y", "z"])
    y = pd.Series(rng.normal(size=32), name="t")
    pipe = _make_pipeline()

    sentinel = ("train_out", "val_out")
    _pre_pipeline_cache_set(
        X,
        None,
        pipe,
        sentinel[0],
        sentinel[1],
        train_target=y.to_numpy(),
        target_name="t",
    )
    hit = _pre_pipeline_cache_get(
        X,
        None,
        pipe,
        train_target=y.to_numpy(),
        target_name="t",
    )
    # Cache entry is (train_out, val_out, fitted_pipeline) since the 2026-05-16
    # fit-state-transfer fix; first two slots remain the sentinel.
    assert hit is not None
    assert hit[0] == sentinel[0]
    assert hit[1] == sentinel[1]
    assert hit[2] is pipe


def test_pre_pipeline_cache_size_respects_cache_max_override():
    """``cache_max`` overrides the module default per call."""
    _pre_pipeline_cache_clear()
    pipe = _make_pipeline()
    rng = np.random.default_rng(2)
    for i in range(6):
        X = pd.DataFrame(rng.normal(size=(8, 2)), columns=["a", "b"])
        y = pd.Series(rng.normal(size=8), name=f"t{i}")
        _pre_pipeline_cache_set(
            X,
            None,
            pipe,
            f"train_{i}",
            f"val_{i}",
            train_target=y.to_numpy(),
            target_name=f"t{i}",
            cache_max=3,
        )
    assert len(_PRE_PIPELINE_CACHE) == 3
