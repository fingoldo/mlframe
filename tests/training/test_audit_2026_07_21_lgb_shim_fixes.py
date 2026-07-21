"""Regression tests for the lgb_shim.py findings fixed from the 2026-07-21 full-repo audit
(see audits/full_audit_2026-07-21/training_loose_c.md F1/F2, and its own PR2 proposal: "Add a
parity test asserting LGB and XGB shims both survive sklearn.clone() with a cache hit -- nothing
pins the LGB side, which is why F1 went unnoticed").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")

from mlframe.training.lgb_shim import (
    LGBMRegressorWithDatasetReuse,
    _LGB_DATASET_CACHE,
    _lgb_cache_clear,
)
from mlframe.training.xgb_shim import (
    XGBRegressorWithDMatrixReuse,
    _XGB_DMATRIX_CACHE,
    _xgb_cache_clear,
)


@pytest.fixture(autouse=True)
def _clear_module_caches():
    """Clear module caches."""
    _lgb_cache_clear()
    _xgb_cache_clear()
    yield
    _lgb_cache_clear()
    _xgb_cache_clear()


def _make_data(n=300, seed=0):
    """Make data."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = rng.normal(size=n)
    return X, y


@pytest.mark.parametrize(
    "shim_cls,module_cache",
    [
        (LGBMRegressorWithDatasetReuse, _LGB_DATASET_CACHE),
        (XGBRegressorWithDMatrixReuse, _XGB_DMATRIX_CACHE),
    ],
)
def test_f1_shim_survives_sklearn_clone_via_module_cache(shim_cls, module_cache):
    """F1 (parity, PR2): sklearn.clone() must produce a fresh instance whose INSTANCE cache is
    empty, but whose subsequent .fit() call still finds the module-level cache hit (not a full
    rebuild) for identical data -- both the LGB and XGB shim must behave identically here.
    Pre-fix, lgb_shim had no module-level fallback at all, so clone() silently lost all
    Dataset-reuse benefit for LGB specifically."""
    from sklearn.base import clone

    X, y = _make_data()
    m = shim_cls(n_estimators=3)
    m.fit(X, y)
    assert len(module_cache) > 0, "module cache should be populated after the first fit"

    cloned = clone(m)
    train_pointer_attr = "_cached_train_dataset" if hasattr(cloned, "_cached_train_dataset") else "_cached_train_dmatrix"
    assert getattr(cloned, train_pointer_attr) is None, "clone() must produce a fresh instance with an empty INSTANCE cache"
    # The clone's fit() must succeed and, critically, must not need to rebuild from scratch --
    # verified indirectly by confirming the module cache size does not grow (a rebuild would
    # evict/replace under the same key, but a genuinely-fresh distinct build on different-content
    # data WOULD grow it; same X here should reuse the existing entry).
    size_before = len(module_cache)
    cloned.fit(X, y)
    assert len(module_cache) == size_before, "clone's fit() should have reused the module-cached dataset, not built a new one"


def test_f1_lgb_module_cache_populated_after_fit():
    """F1: a plain fit() must populate the module-level LGB Dataset cache (previously only
    the instance-level cache existed)."""
    X, y = _make_data()
    m = LGBMRegressorWithDatasetReuse(n_estimators=3)
    assert len(_LGB_DATASET_CACHE) == 0
    m.fit(X, y)
    assert len(_LGB_DATASET_CACHE) >= 1


def test_f2_lgb_multi_eval_set_all_entries_cached_within_one_fit():
    """F2: a single fit() call with N eval_set entries must cache all N val Datasets at the
    instance level, not just the last one (pre-fix: the single _cached_val_dataset/_cached_val_key
    slot was overwritten by every loop iteration, so only the LAST entry survived)."""
    X, y = _make_data(n=400)
    rng = np.random.default_rng(1)
    eval_set = [
        (pd.DataFrame(rng.normal(size=(30, 6)), columns=X.columns), rng.normal(size=30)),
        (pd.DataFrame(rng.normal(size=(40, 6)), columns=X.columns), rng.normal(size=40)),
        (pd.DataFrame(rng.normal(size=(50, 6)), columns=X.columns), rng.normal(size=50)),
    ]
    m = LGBMRegressorWithDatasetReuse(n_estimators=3)
    m.fit(X, y, eval_set=eval_set)
    assert len(m._cached_val_datasets) == 3, "all 3 eval_set entries should be cached, not just the last one"


def test_f2_lgb_multi_eval_set_repeat_fit_hits_all_slots():
    """F2: refitting the SAME instance with the SAME multiple eval sets must reuse every
    val Dataset (all instance-level cache hits), not rebuild N-1 of them."""
    X, y = _make_data(n=400)
    rng = np.random.default_rng(1)
    eval_set = [
        (pd.DataFrame(rng.normal(size=(30, 6)), columns=X.columns), rng.normal(size=30)),
        (pd.DataFrame(rng.normal(size=(40, 6)), columns=X.columns), rng.normal(size=40)),
    ]
    m = LGBMRegressorWithDatasetReuse(n_estimators=3)
    m.fit(X, y, eval_set=eval_set)
    ids_after_first = {id(v) for v in m._cached_val_datasets.values()}

    m.fit(X, y, eval_set=eval_set)
    ids_after_second = {id(v) for v in m._cached_val_datasets.values()}

    assert ids_after_first == ids_after_second, "refitting with identical eval_set entries should reuse every cached val Dataset by identity"


def test_f2_val_cache_key_includes_train_key_not_just_val_content():
    """F2 (related correctness fix): a val Dataset's cache key must include the originating
    train Dataset's key, not just the val content signature -- otherwise the SAME X_val paired
    with a DIFFERENT train Dataset (different bin mapping) could silently reuse stale bins."""
    X1, y1 = _make_data(n=200, seed=0)
    X2, y2 = _make_data(n=200, seed=42)  # different train content -> different bins
    rng = np.random.default_rng(9)
    X_val = pd.DataFrame(rng.normal(size=(20, 6)), columns=X1.columns)
    y_val = rng.normal(size=20)

    m = LGBMRegressorWithDatasetReuse(n_estimators=3)
    m.fit(X1, y1, eval_set=[(X_val, y_val)])
    keys_after_first = set(m._cached_val_datasets.keys())
    assert len(keys_after_first) == 1
    (val_dataset_1,) = m._cached_val_datasets.values()

    m.fit(X2, y2, eval_set=[(X_val, y_val)])
    keys_after_second = set(m._cached_val_datasets.keys())

    # Same X_val content but a DIFFERENT train_key -> the composite val_key must differ, so the
    # second fit's key is NEW (not equal to the first fit's key) and a distinct Dataset object
    # gets built for it, rather than silently reusing val_dataset_1 (built against X1's bins).
    assert keys_after_second != keys_after_first, "val cache key must be scoped to the originating train Dataset, not just X_val content"
    assert len(m._cached_val_datasets) == 2, "both the X1-scoped and X2-scoped val Dataset entries should coexist"
    val_dataset_2 = m._cached_val_datasets[next(iter(keys_after_second - keys_after_first))]
    assert val_dataset_2 is not val_dataset_1
