"""Module-level caches shared by concurrent fits keep their entries consistent under thread contention."""

import threading

import mlframe.training._trainer_configure as tc
from mlframe.training.composite.discovery import _screening_tiny as st


def _hammer(fn, n_threads=8, n_iter=200):
    errors = []

    def work(t):
        try:
            for i in range(n_iter):
                fn(t, i)
        except Exception as e:  # surfaced below, not swallowed
            errors.append(e)

    threads = [threading.Thread(target=work, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    return errors


def test_training_config_cache_never_holds_a_pin_without_its_entry(monkeypatch):
    import mlframe.training.trainer as trainer

    monkeypatch.setattr(trainer, "get_training_configs", lambda **kw: {"seed": kw["random_seed"]})
    monkeypatch.setattr(tc, "_GTC_CACHE_MAX", 4)
    tc._GTC_CACHE.clear()
    tc._GTC_CACHE_SUBGROUPS_PIN.clear()

    def call(t, i):
        res = tc._get_training_configs_cached(random_seed=(t * 7 + i) % 11, subgroups={"g": [t]})
        assert res == {"seed": (t * 7 + i) % 11}

    assert not _hammer(call)
    assert tc._GTC_CACHE, "the calls must have populated the cache"
    assert set(tc._GTC_CACHE_SUBGROUPS_PIN) <= set(tc._GTC_CACHE), "an evicted entry left its subgroups pin behind"
    assert len(tc._GTC_CACHE) <= 4


def test_kfold_split_cache_serves_correct_splits_under_contention(monkeypatch):
    monkeypatch.setattr(st, "_KFOLD_SPLIT_CACHE_MAX", 3)
    st._KFOLD_SPLIT_CACHE.clear()

    def call(t, i):
        n = 20 + (t + i) % 5
        splits = st._cached_kfold_splits(n, 4, 0)
        assert sorted(int(x) for _, te in splits for x in te) == list(range(n)), "test folds must partition the rows"

    assert not _hammer(call)
    assert len(st._KFOLD_SPLIT_CACHE) <= 3
