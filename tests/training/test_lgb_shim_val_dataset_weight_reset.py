"""TRAINING_LOOSE_C-4 (2026-08-05 audit): on a val-Dataset cache hit, lgb_shim only called
dval.set_weight() when w_val was not None -- asymmetric with the train-Dataset cache-hit path (which
explicitly resets to uniform ones when sample_weight is omitted). A later call that omits the eval-set
weight after an earlier call supplied one used to silently keep the reused Dataset's stale weight instead
of resetting to uniform. Fixed by mirroring the train path's reset in the val-Dataset branch too.

Fixing this surfaced a deeper, pre-existing bug in the train-Dataset reset itself: LightGBM's
``Dataset.set_weight(weight)`` treats an all-ones array as equivalent to "no weight" and SKIPS the
underlying field write entirely in that case, so ``set_weight(np.ones(...))`` on a Dataset that already
has a REAL prior weight silently no-ops -- confirmed live via direct LightGBM API probing. Both the train
and val reset paths now go through ``_reset_weight_to_uniform``, which uses the lower-level ``set_field``
to bypass that optimization.
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
    """Small deterministic regression dataset (train + val split) for cache-hit tests."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    y = X["f0"].to_numpy() * 2.0 + rng.normal(scale=0.1, size=n)
    n_val = 50
    return X.iloc[:-n_val], y[:-n_val], X.iloc[-n_val:], y[-n_val:]


def test_second_fit_without_eval_weight_resets_stale_val_weight(small_regression_data):
    """A second fit reusing the cached val Dataset but OMITTING eval_sample_weight must reset to
    uniform weights, not silently keep the first fit's stale weight."""
    X_train, y_train, X_val, y_val = small_regression_data
    m = LGBMRegressorWithDatasetReuse(n_estimators=3, **_QUIET_LGB)

    w_val_1 = np.linspace(0.1, 2.0, len(y_val)).astype(np.float32)
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_sample_weight=[w_val_1])

    val_key = next(iter(m._cached_val_datasets))
    dval_before = m._cached_val_datasets[val_key]
    np.testing.assert_array_equal(dval_before.get_weight(), w_val_1)

    m.fit(X_train, y_train, eval_set=[(X_val, y_val)])  # no eval_sample_weight this time

    dval_after = m._cached_val_datasets[val_key]
    assert dval_before is dval_after, "the cached val Dataset must be reused (no rebuild)"
    np.testing.assert_array_equal(
        dval_after.get_weight(),
        np.ones(len(y_val), dtype=np.float32),
        err_msg="omitting eval_sample_weight on a cache-hit fit must reset the val Dataset to uniform weight",
    )


def test_second_fit_without_train_weight_resets_stale_train_weight(small_regression_data):
    """The train-Dataset side of this same bug class: a second fit reusing the cached train Dataset
    but OMITTING sample_weight must reset it to uniform weight, not silently keep the first fit's
    stale (non-uniform) weight -- LightGBM's Dataset.set_weight(ones) alone is not sufficient once a
    real prior weight is set (see _reset_weight_to_uniform's docstring)."""
    X_train, y_train, _X_val, _y_val = small_regression_data
    m = LGBMRegressorWithDatasetReuse(n_estimators=3, **_QUIET_LGB)

    w_train_1 = np.linspace(0.1, 2.0, len(y_train)).astype(np.float32)
    m.fit(X_train, y_train, sample_weight=w_train_1)
    dtrain_before = m._cached_train_dataset
    np.testing.assert_array_equal(dtrain_before.get_weight(), w_train_1)

    m.fit(X_train, y_train)  # no sample_weight this time

    dtrain_after = m._cached_train_dataset
    assert dtrain_before is dtrain_after, "the cached train Dataset must be reused (no rebuild)"
    np.testing.assert_array_equal(
        dtrain_after.get_weight(),
        np.ones(len(y_train), dtype=np.float32),
        err_msg="omitting sample_weight on a cache-hit fit must reset the train Dataset to uniform weight",
    )
