"""A new suite run must not inherit the LightGBM Dataset and XGBoost DMatrix caches of the previous one.

Both caches hold C++ memory for the lifetime of the process. `_lgb_cache_clear` and `_xgb_cache_clear` exist to
release it between suite invocations, and the suite setup is the boundary where that happens.
"""

from __future__ import annotations

from mlframe.training._lgb_shim_helpers import _LGB_DATASET_CACHE, _LGB_DATASET_CACHE_LOCK
from mlframe.training.core import _phase_config_setup as pcs
from mlframe.training.xgb_shim import _XGB_DMATRIX_CACHE, _XGB_DMATRIX_CACHE_LOCK

_STALE_KEY = ("stale-from-previous-suite",)


def test_suite_setup_releases_previous_suite_gbdt_dataset_caches(monkeypatch):
    """Seed both caches right before setup reaches the GBDT clear, then let setup run.

    The stub configs do not describe a whole suite, so setup may fail further down. The clear is the step right after the
    CatBoost Pool clear, so a failure later does not change what this test observes.
    """
    seeded = []

    def _seed_then_return_from_catboost_clear() -> None:
        """Stand in for the CatBoost clear: plant a stale entry in each GBDT cache, then return."""
        with _LGB_DATASET_CACHE_LOCK:
            _LGB_DATASET_CACHE[_STALE_KEY] = object()
        with _XGB_DMATRIX_CACHE_LOCK:
            _XGB_DMATRIX_CACHE[_STALE_KEY] = object()
        seeded.append(True)

    monkeypatch.setattr(pcs, "_clear_catboost_pool_caches", _seed_then_return_from_catboost_clear)
    try:
        pcs.setup_configuration(
            preprocessing_config=None,
            pipeline_config=None,
            feature_types_config=None,
            split_config=None,
            hyperparams_config=None,
            behavior_config=None,
            reporting_config=None,
            output_config=None,
            outlier_detection_config=None,
            feature_selection_config=None,
            confidence_analysis_config=None,
            baseline_diagnostics_config=None,
            dummy_baselines_config=None,
            quantile_regression_config=None,
            composite_target_discovery_config=None,
            feature_handling_config=None,
            model_name="m",
            target_name="t",
            mlframe_models=None,
            verbose=0,
        )
    except Exception:
        pass
    assert seeded, "setup never reached the CatBoost clear; the test observed nothing"
    try:
        assert _STALE_KEY not in _LGB_DATASET_CACHE
        assert _STALE_KEY not in _XGB_DMATRIX_CACHE
    finally:
        with _LGB_DATASET_CACHE_LOCK:
            _LGB_DATASET_CACHE.pop(_STALE_KEY, None)
        with _XGB_DMATRIX_CACHE_LOCK:
            _XGB_DMATRIX_CACHE.pop(_STALE_KEY, None)
