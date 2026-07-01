"""A5#4/#16 sensor: ``_get_training_configs_cached`` session memo for the per-target ``select_target`` call.

Pre-fix ``select_target`` re-invoked ``get_training_configs`` twice per (target, pre_pipeline, model) tuple even when
the underlying ``config_params`` dict (derived once per suite from ``hyperparams_config.model_dump()``) was identical
across targets. The memo collapses repeated calls with the same hashable kwargs to a single underlying call + deepcopy.

Tests pin:
1. Identical kwargs -> second call is a cache hit (counted via ``get_training_configs`` patch).
2. Different ``has_gpu`` flag -> separate cache entry (must NOT collide).
3. Unhashable kwarg present (callable scorer) -> falls through to direct call (no caching, no crash).
4. Cache cap (``_GTC_CACHE_MAX``) bounds the dict size via FIFO eviction.
5. Returned object is a deepcopy -- mutating the result must not poison subsequent hits.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from mlframe.training import _trainer_configure as tc


@pytest.fixture(autouse=True)
def _clear_gtc_cache():
    """Each test starts with an empty cache so hit / miss counts are deterministic."""
    tc._GTC_CACHE.clear()
    yield
    tc._GTC_CACHE.clear()


def test_repeated_identical_kwargs_hits_cache_after_first_call():
    """Same hashable kwargs in two back-to-back calls -> get_training_configs is invoked exactly ONCE."""
    call_count = {"n": 0}

    def _stub(**_kw):
        from types import SimpleNamespace
        call_count["n"] += 1
        return SimpleNamespace(catboost_dict=dict(iterations=100), kwargs=tuple(sorted(_kw.items())))

    with patch.object(tc, "_get_training_configs_cached", wraps=tc._get_training_configs_cached) as wrapped:
        with patch("mlframe.training.trainer.get_training_configs", side_effect=_stub):
            tc._get_training_configs_cached(has_gpu=False, iterations=100, learning_rate=0.1)
            tc._get_training_configs_cached(has_gpu=False, iterations=100, learning_rate=0.1)

    assert call_count["n"] == 1, f"second call should hit the cache; got {call_count['n']} stub invocations"


def test_different_has_gpu_flag_separate_cache_entries():
    """CPU + GPU configs must never collide on the cache key -- both should reach the stub."""
    call_count = {"n": 0}

    def _stub(**_kw):
        from types import SimpleNamespace
        call_count["n"] += 1
        return SimpleNamespace(has_gpu=_kw.get("has_gpu"))

    with patch("mlframe.training.trainer.get_training_configs", side_effect=_stub):
        tc._get_training_configs_cached(has_gpu=False, iterations=100)
        tc._get_training_configs_cached(has_gpu=None, iterations=100)

    assert call_count["n"] == 2, "CPU + GPU pair must call the stub twice (distinct cache entries)"
    assert len(tc._GTC_CACHE) == 2


def test_unhashable_kwarg_falls_through_to_direct_call():
    """A callable scorer in the kwargs short-circuits the cache; the stub fires every time, the cache stays empty."""
    call_count = {"n": 0}

    def _stub(**_kw):
        from types import SimpleNamespace
        call_count["n"] += 1
        return SimpleNamespace()

    def _scorer(y_true, y_pred):
        return 0.0

    with patch("mlframe.training.trainer.get_training_configs", side_effect=_stub):
        tc._get_training_configs_cached(has_gpu=False, default_regression_scoring=_scorer)
        tc._get_training_configs_cached(has_gpu=False, default_regression_scoring=_scorer)

    assert call_count["n"] == 2, "unhashable scorer must bypass the cache and call through twice"
    assert len(tc._GTC_CACHE) == 0


def test_cache_cap_bounds_dict_via_fifo_eviction():
    """``_GTC_CACHE_MAX`` is respected -- once the cap is hit, the oldest entry evicts."""
    def _stub(**_kw):
        from types import SimpleNamespace
        return SimpleNamespace(payload=_kw.get("iterations"))

    with patch("mlframe.training.trainer.get_training_configs", side_effect=_stub):
        for i in range(tc._GTC_CACHE_MAX + 3):
            tc._get_training_configs_cached(has_gpu=False, iterations=i)

    assert len(tc._GTC_CACHE) == tc._GTC_CACHE_MAX, (
        f"cache must be capped at {tc._GTC_CACHE_MAX}; got {len(tc._GTC_CACHE)}"
    )


def test_returned_object_is_deepcopy_so_caller_mutation_does_not_poison_hits():
    """Caller mutates the returned SimpleNamespace; subsequent hits must return a fresh copy."""
    from types import SimpleNamespace

    def _stub(**_kw):
        return SimpleNamespace(catboost_dict={"iterations": 100})

    with patch("mlframe.training.trainer.get_training_configs", side_effect=_stub):
        first = tc._get_training_configs_cached(has_gpu=False, iterations=100)
        first.catboost_dict["iterations"] = 9999
        first.poisoned_field = "MUTATED"

        second = tc._get_training_configs_cached(has_gpu=False, iterations=100)

    assert second.catboost_dict["iterations"] == 100, (
        "deepcopy on return must isolate cache from caller mutation"
    )
    assert not hasattr(second, "poisoned_field"), (
        "cache must not propagate caller-injected attributes"
    )


def test_none_subgroups_is_stable_cache_key():
    """``subgroups=None`` (the default) must produce the same key across calls so the no-fairness path still memoises."""
    call_count = {"n": 0}

    def _stub(**_kw):
        from types import SimpleNamespace
        call_count["n"] += 1
        return SimpleNamespace()

    with patch("mlframe.training.trainer.get_training_configs", side_effect=_stub):
        tc._get_training_configs_cached(has_gpu=False, subgroups=None, iterations=100)
        tc._get_training_configs_cached(has_gpu=False, subgroups=None, iterations=100)

    assert call_count["n"] == 1, "subgroups=None must hash identically across calls"
