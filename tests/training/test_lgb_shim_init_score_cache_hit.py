"""TRAINING_LOOSE_C-3 (2026-08-05 audit): on a train-Dataset cache hit, lgb_shim's fit() re-applies
set_label/set_weight but never set_init_score, so a subsequent .fit(X, y, init_score=new_value) call
reusing the cached Dataset silently trained against the stale/absent init_score instead of the new one.
Fixed by re-applying set_init_score on every cache hit too, mirroring the set_label/set_weight pattern
(explicit init_score overwrites; None clears any prior init_score).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

try:
    from mlframe.training.lgb_shim import LGBMRegressorWithDatasetReuse

    SHIM_AVAILABLE = True
except ImportError:
    SHIM_AVAILABLE = False

pytestmark = pytest.mark.skipif(not SHIM_AVAILABLE, reason="lgb_shim not available")

_QUIET_LGB = {"verbose": -1}


@pytest.fixture()
def small_regression_data():
    """Small deterministic regression dataset for cache-hit tests."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    y = 10.0 + X["f0"].to_numpy() * 2.0 + rng.normal(scale=0.1, size=n)  # off-zero mean: boost_from_average must matter
    return X, y


def test_second_fit_with_new_init_score_reuses_dataset_and_updates_init_score(small_regression_data):
    """Second fit with the same X but a NEW init_score must reuse the cached Dataset (no rebuild) AND
    actually apply the new init_score -- not silently keep the first fit's stale/absent one."""
    X, y = small_regression_data
    m = LGBMRegressorWithDatasetReuse(n_estimators=3, **_QUIET_LGB)

    init_score_1 = np.zeros(len(y), dtype=np.float64)
    m.fit(X, y, init_score=init_score_1)
    id_before = id(m._cached_train_dataset)
    np.testing.assert_array_equal(m._cached_train_dataset.get_init_score(), init_score_1)

    init_score_2 = np.full(len(y), 5.0, dtype=np.float64)
    m.fit(X, y, init_score=init_score_2)
    id_after = id(m._cached_train_dataset)

    assert id_before == id_after, "second fit on identical X must reuse the cached Dataset (no rebuild)"
    np.testing.assert_array_equal(
        m._cached_train_dataset.get_init_score(),
        init_score_2,
        err_msg="cache-hit fit must re-apply the NEW init_score, not keep the first fit's stale value",
    )


def test_second_fit_without_init_score_trains_as_a_fresh_fit_without_one(small_regression_data):
    """A cache-hit fit that omits init_score must train the booster a fresh fit without one trains.

    LightGBM cannot clear an init_score from a Dataset, and explicit zeros are not the same as none: they switch off
    boost_from_average, so the reused Dataset used to yield a booster shifted by the train mean from a fresh fit's.
    """
    X, y = small_regression_data
    m = LGBMRegressorWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
    m.fit(X, y, init_score=np.full(len(y), 3.0, dtype=np.float64))
    m.fit(X, y)  # no init_score this time

    fresh = LGBMRegressorWithDatasetReuse(n_estimators=3, **_QUIET_LGB)
    fresh._init_cache()
    import os

    os.environ["MLFRAME_LGB_CACHE_DISABLE"] = "1"
    try:
        fresh.fit(X, y)
    finally:
        os.environ.pop("MLFRAME_LGB_CACHE_DISABLE", None)
    np.testing.assert_array_equal(m.predict(X), fresh.predict(X))
    assert m._cached_train_dataset.get_init_score() is None


def test_a_module_cache_hit_trains_the_same_booster_as_a_miss(small_regression_data):
    """Two instances fitting the same X: the second hits the module-level cache and must train the same booster."""
    X, y = small_regression_data
    first = LGBMRegressorWithDatasetReuse(n_estimators=5, **_QUIET_LGB)
    first.fit(X, y)
    second = LGBMRegressorWithDatasetReuse(n_estimators=5, **_QUIET_LGB)
    second.fit(X, y)
    assert second._cached_train_dataset is first._cached_train_dataset, "the second fit should reuse the cached Dataset"
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
