"""A fold that trains on a subset of its rows goes through the locked native path, and predicts what the wrapper did.

The first fix serialised the cached full-fold construction. The production kernel then died again in the OTHER
construction site: ``LGBMRegressor.fit`` on a masked fold, which builds its dataset inside ``fit`` with no
serialisation. Masked folds now build their dataset under the same lock, cached by the exact rows they train on.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from mlframe.training.composite.discovery import _lgb_shared_fold as sf
from mlframe.training.composite.discovery._screening_tiny_perbin import _fit_fold_model

_FIT_KW = dict(
    family="lgb", n_estimators=10, num_leaves=7, learning_rate=0.1, random_state=0,
    deterministic=False, inner_n_jobs=1, n_jobs=4,
)


def test_masked_fold_predicts_exactly_like_the_sklearn_wrapper():
    """Bit-identical: the change must not move which specs the rerank keeps."""
    from mlframe.training.composite.discovery._screening_tiny import _build_tiny_model

    rng = np.random.default_rng(0)
    x = rng.normal(size=(1500, 12)).astype(np.float32)
    t = (x[:, 0] * 2 - x[:, 3] + rng.normal(size=1500)).astype(np.float64)
    train_fold = np.arange(1200)
    fit_rows = np.sort(rng.choice(1200, 800, replace=False))

    booster = _fit_fold_model(x, train_fold, fit_rows, t[fit_rows], **_FIT_KW)
    wrapper = _build_tiny_model("lgb", n_estimators=10, num_leaves=7, learning_rate=0.1, random_state=0, deterministic=False, inner_n_jobs=1)
    wrapper.set_params(n_jobs=1)
    wrapper.fit(x[fit_rows], t[fit_rows])

    np.testing.assert_array_equal(np.asarray(booster.predict(x)).ravel(), np.asarray(wrapper.predict(x)).ravel())


class _ConcurrencyProbe:
    """Stands in for ``lightgbm.Dataset``, recording how many threads construct at once."""

    def __init__(self, data, label=None, params=None, free_raw_data=True):
        self.label = label

    def construct(self):
        """Hold the construction open long enough for an unserialised sibling to overlap it."""
        with _STATE["lock"]:
            _STATE["live"] += 1
            _STATE["peak"] = max(_STATE["peak"], _STATE["live"])
        time.sleep(0.02)
        with _STATE["lock"]:
            _STATE["live"] -= 1
        return self

    def set_label(self, label):
        """Label swap between specs."""
        self.label = label


_STATE = {"lock": threading.Lock(), "live": 0, "peak": 0}


@pytest.fixture
def probe(monkeypatch):
    """Replace LightGBM's dataset and trainer so only the serialisation is measured."""
    import lightgbm as lgb

    monkeypatch.setattr(lgb, "Dataset", _ConcurrencyProbe)
    monkeypatch.setattr(lgb, "train", lambda params, ds, num_boost_round=0: object())
    _STATE["live"] = 0
    _STATE["peak"] = 0
    sf._CACHE.clear()
    return _STATE


def _drive_masked_folds(n_threads: int) -> int:
    """Fit ``n_threads`` masked folds at once, each on its own matrix; returns the construct peak."""
    barrier = threading.Barrier(n_threads)

    def _worker(seed: int) -> None:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(200, 5)).astype(np.float32)
        train_fold = np.arange(160)
        fit_rows = np.sort(rng.choice(160, 100, replace=False))
        barrier.wait()
        _fit_fold_model(x, train_fold, fit_rows, np.zeros(100), **_FIT_KW)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return int(_STATE["peak"])


def test_masked_folds_construct_one_at_a_time(probe):
    """The defect: the wrapper path constructed concurrently with other threads' training."""
    assert _drive_masked_folds(8) == 1


def test_the_probe_would_catch_concurrency_without_the_lock(probe, monkeypatch):
    """The probe is only evidence if it can see overlap."""
    import contextlib

    monkeypatch.setattr(sf, "_CONSTRUCT_LOCK", contextlib.nullcontext())
    assert _drive_masked_folds(8) > 1


def test_repeated_masks_reuse_their_dataset(probe):
    """Caching by the exact fit rows is what keeps the safe path cheaper than the unprotected wrapper was."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 5)).astype(np.float32)
    train_fold = np.arange(160)
    fit_rows = np.sort(rng.choice(160, 100, replace=False))
    for _ in range(3):  # the seed repeats of one spec see the same rows
        _fit_fold_model(x, train_fold, fit_rows, np.zeros(100), **_FIT_KW)
    assert len(sf._CACHE) == 1
