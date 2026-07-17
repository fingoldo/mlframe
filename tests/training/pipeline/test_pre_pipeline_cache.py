"""Locks the 2026-05-12 pre-pipeline LRU cache.

Context. The TVT training log showed Linear taking ~46s on the
``SimpleImputer + StandardScaler`` fit, then MLP -- a few seconds later in
the same per-target iteration -- taking another ~18s on the SAME arithmetic
applied to the SAME train_df. Both built fresh pipeline instances so the
existing ``_is_fitted(pre_pipeline)`` short-circuit never fired.

The cache hashes the pipeline structure (step classes + their
``get_params(deep=False)``), keys by ``(id(train_df), id(val_df), sig)``,
and stores the fitted transform output. The second model in the same
per-target loop iteration gets a cache hit and skips fit+transform entirely.

Cache discipline locked here:
1. Structurally identical pipelines (same classes + same hyperparams) hit
   the cache when run on the same train_df / val_df objects.
2. A different train_df id MUST miss (no cross-target leakage).
3. A pipeline whose hyperparams differ (e.g. StandardScaler(with_mean=False)
   instead of default) MUST miss.
4. LRU eviction keeps the cache to <=2 entries.
5. ``_pre_pipeline_cache_clear`` empties the cache for explicit reset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.training.trainer import (
    _PRE_PIPELINE_CACHE,
    _pipeline_signature_for_cache,
    _pre_pipeline_cache_clear,
    _pre_pipeline_cache_get,
    _pre_pipeline_cache_set,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    """Each test starts with an empty cache."""
    _pre_pipeline_cache_clear()
    yield
    _pre_pipeline_cache_clear()


def _make_pipeline():
    """Fresh SimpleImputer + StandardScaler -- same arithmetic both times."""
    return Pipeline([("imp", SimpleImputer()), ("scaler", StandardScaler())])


def _make_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Make df."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(5, 2, n),
        }
    )


class TestSignature:
    """Groups tests covering signature."""
    def test_identical_pipelines_same_signature(self) -> None:
        """Identical pipelines same signature."""
        sig_a = _pipeline_signature_for_cache(_make_pipeline())
        sig_b = _pipeline_signature_for_cache(_make_pipeline())
        assert sig_a == sig_b

    def test_different_hyperparams_different_signature(self) -> None:
        """Different hyperparams different signature."""
        p1 = Pipeline([("imp", SimpleImputer()), ("scaler", StandardScaler())])
        p2 = Pipeline(
            [
                ("imp", SimpleImputer()),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )
        assert _pipeline_signature_for_cache(p1) != _pipeline_signature_for_cache(p2)

    def test_different_step_classes_different_signature(self) -> None:
        """Different step classes different signature."""
        from sklearn.preprocessing import MinMaxScaler

        p1 = Pipeline([("imp", SimpleImputer()), ("scaler", StandardScaler())])
        p2 = Pipeline([("imp", SimpleImputer()), ("scaler", MinMaxScaler())])
        assert _pipeline_signature_for_cache(p1) != _pipeline_signature_for_cache(p2)

    def test_none_pipeline_returns_none_string(self) -> None:
        """None pipeline returns none string."""
        assert _pipeline_signature_for_cache(None) == "None"


class TestLookupAndStore:
    """Groups tests covering lookup and store."""
    def test_miss_when_empty(self) -> None:
        """Miss when empty."""
        df = _make_df()
        assert _pre_pipeline_cache_get(df, None, _make_pipeline()) is None

    def test_set_then_get_round_trip(self) -> None:
        """Set then get round trip."""
        train, val = _make_df(seed=0), _make_df(seed=1)
        pipe = _make_pipeline()
        out_train, out_val = train.copy(), val.copy()
        _pre_pipeline_cache_set(train, val, pipe, out_train, out_val)
        hit = _pre_pipeline_cache_get(train, val, _make_pipeline())
        assert hit is not None
        assert hit[0] is out_train
        assert hit[1] is out_val

    def test_different_train_df_misses(self) -> None:
        """Different train df misses."""
        train_a, val = _make_df(seed=0), _make_df(seed=1)
        train_b = _make_df(seed=42)  # different identity
        _pre_pipeline_cache_set(train_a, val, _make_pipeline(), train_a.copy(), val.copy())
        assert _pre_pipeline_cache_get(train_b, val, _make_pipeline()) is None

    def test_different_val_df_misses(self) -> None:
        """Different val df misses."""
        train = _make_df(seed=0)
        val_a, val_b = _make_df(seed=1), _make_df(seed=2)
        _pre_pipeline_cache_set(train, val_a, _make_pipeline(), train.copy(), val_a.copy())
        assert _pre_pipeline_cache_get(train, val_b, _make_pipeline()) is None

    def test_different_hyperparams_miss(self) -> None:
        """Different hyperparams miss."""
        train, val = _make_df(seed=0), _make_df(seed=1)
        p1 = _make_pipeline()
        p2 = Pipeline([("imp", SimpleImputer()), ("scaler", StandardScaler(with_mean=False))])
        _pre_pipeline_cache_set(train, val, p1, train.copy(), val.copy())
        assert _pre_pipeline_cache_get(train, val, p2) is None


class TestLRUEviction:
    """Groups tests covering l r u eviction."""
    def test_lru_caps_at_two_entries(self) -> None:
        """Three different (train_df, pipeline) keys -> only the latest two
        survive (LRU eviction). Cap is per-call (CACHE-P1-4)."""
        train = _make_df(seed=0)
        # Three different val_df identities -> three distinct keys.
        v1, v2, v3 = _make_df(seed=1), _make_df(seed=2), _make_df(seed=3)
        pipe = _make_pipeline()
        _pre_pipeline_cache_set(train, v1, pipe, train.copy(), v1.copy(), cache_max=2)
        _pre_pipeline_cache_set(train, v2, pipe, train.copy(), v2.copy(), cache_max=2)
        assert len(_PRE_PIPELINE_CACHE) == 2
        _pre_pipeline_cache_set(train, v3, pipe, train.copy(), v3.copy(), cache_max=2)
        assert len(_PRE_PIPELINE_CACHE) == 2
        # v1 (oldest) was evicted.
        assert _pre_pipeline_cache_get(train, v1, _make_pipeline()) is None
        # v2 and v3 survive.
        assert _pre_pipeline_cache_get(train, v2, _make_pipeline()) is not None
        assert _pre_pipeline_cache_get(train, v3, _make_pipeline()) is not None

    def test_get_promotes_entry_to_most_recent(self) -> None:
        """LRU: a get on the older entry should keep it from being evicted
        next time."""
        train = _make_df(seed=0)
        v1, v2, v3 = _make_df(seed=1), _make_df(seed=2), _make_df(seed=3)
        pipe = _make_pipeline()
        _pre_pipeline_cache_set(train, v1, pipe, train.copy(), v1.copy(), cache_max=2)
        _pre_pipeline_cache_set(train, v2, pipe, train.copy(), v2.copy(), cache_max=2)
        # Bump v1 to most-recent by reading it.
        _ = _pre_pipeline_cache_get(train, v1, _make_pipeline())
        # Adding v3 should evict v2 (LRU now), NOT v1.
        _pre_pipeline_cache_set(train, v3, pipe, train.copy(), v3.copy(), cache_max=2)
        assert _pre_pipeline_cache_get(train, v1, _make_pipeline()) is not None
        assert _pre_pipeline_cache_get(train, v2, _make_pipeline()) is None
        assert _pre_pipeline_cache_get(train, v3, _make_pipeline()) is not None


class TestClear:
    """Groups tests covering clear."""
    def test_clear_empties_cache(self) -> None:
        """Clear empties cache."""
        train, val = _make_df(seed=0), _make_df(seed=1)
        _pre_pipeline_cache_set(
            train,
            val,
            _make_pipeline(),
            train.copy(),
            val.copy(),
        )
        assert len(_PRE_PIPELINE_CACHE) == 1
        _pre_pipeline_cache_clear()
        assert len(_PRE_PIPELINE_CACHE) == 0


class TestNoneInputs:
    """Groups tests covering none inputs."""
    def test_none_train_df_is_skipped(self) -> None:
        """``None`` train_df short-circuits both lookup and store."""
        assert _pre_pipeline_cache_get(None, None, _make_pipeline()) is None
        _pre_pipeline_cache_set(None, None, _make_pipeline(), None, None)
        assert len(_PRE_PIPELINE_CACHE) == 0

    def test_none_pipeline_is_skipped(self) -> None:
        """None pipeline is skipped."""
        train, val = _make_df(seed=0), _make_df(seed=1)
        assert _pre_pipeline_cache_get(train, val, None) is None
        _pre_pipeline_cache_set(train, val, None, train.copy(), val.copy())
        assert len(_PRE_PIPELINE_CACHE) == 0
