"""Regression: ``_train_model_with_fallback`` must skip a 0-feature train frame instead of crashing the backend.

The suite-level guard at ``_trainer_train_and_evaluate`` short-circuits the common FS-empties-everything case, but any
column-dropping step between that check and the fit primitive (or a direct caller) can still hand a 0-feature frame to
``_train_model_with_fallback``. On a real CatBoost backend that raises ``CatBoostError: Input data must have at least
one feature`` and aborts the whole suite run. The guard now returns ``(None, None)`` so the caller's ``if model is None``
skip path handles it. Pre-fix this test failed with the CatBoostError; post-fix it returns a clean skip.
"""

import os

_PRIOR_CUDA_ENV = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module", autouse=True)
def _restore_cuda_env_after_module():
    """Undo this module's CUDA_VISIBLE_DEVICES override once its own tests finish -- an unrestored
    module-level setdefault poisons every later test in the same pytest-xdist worker (see the
    identical bug fixed in test_biz_val_hybrid_cooccur_clusterrep.py, which caused a ~13-test
    GPU-dispatch failure cluster in a completely different worker)."""
    yield
    _v = _PRIOR_CUDA_ENV["CUDA_VISIBLE_DEVICES"]
    if _v is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _v

from mlframe.training._training_loop import _train_model_with_fallback


def test_train_model_with_fallback_skips_zero_feature_catboost():
    """Train model with fallback skips zero feature catboost."""
    cb = pytest.importorskip("catboost")
    model = cb.CatBoostRegressor(iterations=5, verbose=0)
    train_df = pd.DataFrame(index=range(40))  # 40 rows, 0 columns
    assert train_df.shape == (40, 0)
    train_target = np.arange(40.0)

    out_model, best_iter = _train_model_with_fallback(
        model=model,
        model_obj=model,
        model_type_name="CatBoostRegressor",
        train_df=train_df,
        train_target=train_target,
        fit_params={},
        verbose=False,
    )
    assert out_model is None
    assert best_iter is None
