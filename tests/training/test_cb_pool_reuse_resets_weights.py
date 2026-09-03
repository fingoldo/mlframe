"""A cached CatBoost Pool kept the previous fit's sample weights when the next fit asked for uniform ones.

On a cache hit the code re-applied the weight only `if sample_weight is not None`. The cache signature carries
no weight component, so a fit asking for uniform weights got back the Pool a previous, non-uniformly-weighted
fit had built -- and `set_weight` was never called, leaving the old weights in place.

The concrete path: an extractor supplying `sample_weights = {"recency": w, "uniform": None}`. The loop runs in
insertion order, so "recency" builds the Pool with `w`; "uniform" hits the cache and trains recency-weighted.
Its metrics are then compared against the recency model as though the two were different schemas, and the
leaderboard and the ensemble pick between duplicates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mlframe.training.cb._cb_pool as cb_pool
from mlframe.training.cb._cb_pool_build import _maybe_get_or_build_cb_pool

N = 200


class _FakePool:
    """The five Pool methods the reuse fast path touches, recording what it was told.

    A real Pool cannot be used here: this CatBoost build has no `Pool.set_label`, so `_cb_reuse_capable()` is
    False and the fast path never activates on this machine. Skipping would leave the defect untested, and the
    behaviour under test is entirely about WHICH mutators the reuse path calls.
    """

    def __init__(self, data=None, label=None, weight=None, **kw):
        """Record the construction-time label and weight."""
        self._label = np.asarray(label, dtype=np.float64) if label is not None else None
        self._n = len(self._label) if self._label is not None else N
        self._weight = np.asarray(weight, dtype=np.float64) if weight is not None else np.ones(self._n)

    def set_label(self, label):
        """Swap the label in place, as CatBoost's own Pool does."""
        self._label = np.asarray(label, dtype=np.float64)

    def get_label(self):
        """The current label."""
        return self._label

    def set_weight(self, weight):
        """Swap the weight in place."""
        self._weight = np.asarray(weight, dtype=np.float64)

    def get_weight(self):
        """The current weight."""
        return self._weight

    def num_row(self):
        """Row count, needed to build a uniform weight vector."""
        return self._n


@pytest.fixture(autouse=True)
def _fast_path(monkeypatch):
    """Force the reuse fast path on, with a fake Pool, and an empty cache per test."""
    import catboost

    monkeypatch.setattr(cb_pool, "_cb_reuse_capable", lambda: True)
    monkeypatch.setattr(catboost, "Pool", _FakePool)
    cb_pool._CB_POOL_CACHE.clear()
    yield
    cb_pool._CB_POOL_CACHE.clear()


@pytest.fixture
def frame():
    """A small numeric frame and a regression target."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(0, 1, N), "b": rng.normal(0, 1, N)}), rng.normal(0, 1, N)


def _pool(frame, target, weight):
    """One trip through the reuse fast path with the given sample weight."""
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(iterations=1, verbose=False, allow_writing_files=False)
    fit_params = {"sample_weight": weight} if weight is not None else {}
    pool = _maybe_get_or_build_cb_pool("CatBoostRegressor", model, frame, target, fit_params)
    assert pool is not None, "the reuse fast path did not activate; the test would prove nothing"
    return pool


class TestTheWeightIsResetOnACacheHit:
    """The defect, on the exact sequence that produces it."""

    def test_a_uniform_fit_after_a_weighted_one_gets_uniform_weights(self, frame):
        """The recency-then-uniform sequence a multi-schema extractor produces."""
        df, y = frame
        w = np.linspace(0.1, 2.0, N)
        _pool(df, y, w)
        second = _pool(df, y, None)
        assert np.allclose(np.asarray(second.get_weight()), 1.0), "the cached Pool kept the previous fit's weights"

    def test_it_really_was_the_same_pool(self, frame):
        """If the cache missed, the test above would pass for the wrong reason."""
        df, y = frame
        first = _pool(df, y, np.linspace(0.1, 2.0, N))
        assert _pool(df, y, None) is first

    def test_a_weighted_fit_after_a_uniform_one_gets_its_weights(self, frame):
        """The other order, which already worked; a guard against fixing one and breaking the other."""
        df, y = frame
        _pool(df, y, None)
        w = np.linspace(0.1, 2.0, N)
        assert np.allclose(np.asarray(_pool(df, y, w).get_weight()), w)

    def test_two_different_weight_schemas_do_not_bleed(self, frame):
        """Three fits in sequence, each of which must see exactly what it asked for."""
        df, y = frame
        w1 = np.linspace(0.1, 2.0, N)
        w2 = np.linspace(2.0, 0.1, N)
        _pool(df, y, w1)
        assert np.allclose(np.asarray(_pool(df, y, w2).get_weight()), w2)
        assert np.allclose(np.asarray(_pool(df, y, None).get_weight()), 1.0)
        assert np.allclose(np.asarray(_pool(df, y, w1).get_weight()), w1)

    def test_a_freshly_built_uniform_pool_is_uniform(self, frame):
        """The build path, so the reset bookkeeping starts from the right state."""
        df, y = frame
        pool = _pool(df, y, None)
        assert np.allclose(np.asarray(pool.get_weight()), 1.0)

    def test_the_label_swap_still_works(self, frame):
        """The weight reset sits next to the label swap; neither may break the other."""
        df, y = frame
        _pool(df, y, None)
        y2 = y + 100.0
        assert np.allclose(np.asarray(_pool(df, y2, None).get_label()), y2, atol=1e-3)
