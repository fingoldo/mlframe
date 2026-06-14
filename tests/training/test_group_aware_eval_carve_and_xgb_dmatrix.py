"""Regression tests for round 5.5 follow-ups (2026-05-25):

1. ``_carve_inner_eval_split`` is group-aware: when ``group_ids`` is
   supplied, no group spans both the fit and eval slices. Catches the
   bug where the group-blind last-tail carve inflated the trained-model
   OOF RMSE by ~25% on group-aware splits, falsely triggering the AR(1)
   failsafe (TVT prod 2026-05-24).
2. ``lag_predict_failsafe_tolerance`` default rolled back to 0.10 now
   that the carve is honest.
3. ``XGBRegressorWithDMatrixReuse`` val DMatrix has a module-level
   cache fallback so sklearn.clone() in OOF refits doesn't rebuild
   from scratch each iteration.
4. ``PipelineCache`` byte budget is RAM-aware (psutil-driven) instead
   of hardcoded 2 GB.
5. Matplotlib renderer auto-enables ``constrained_layout`` when a
   suptitle is present, and ``save()`` uses ``bbox_inches='tight'`` so
   suptitles / ytick labels never get clipped. FI plot save mirrors
   this with bbox_inches='tight'.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _module_source(mod) -> str:
    return Path(mod.__file__).read_text(encoding="utf-8")


class TestCarveInnerEvalSplitGroupAware:
    def test_group_ids_keeps_groups_whole(self) -> None:
        """No group spans fit and eval slices when group_ids is given."""
        from mlframe.training.composite.ensemble import _carve_inner_eval_split

        n = 2000
        rng = np.random.default_rng(42)
        X = rng.normal(size=(n, 3))
        y = rng.normal(size=n)
        n_groups = 200
        group_ids = np.repeat(np.arange(n_groups), n // n_groups)

        X_fit, y_fit, X_eval, y_eval = _carve_inner_eval_split(
            X, y, frac=0.1, random_state=0, group_ids=group_ids,
        )
        assert X_eval is not None and y_eval is not None
        eval_mask = np.zeros(n, dtype=bool)
        eval_mask[len(y_fit):] = True
        fit_groups = set(group_ids[: len(y_fit)].tolist())
        eval_groups = set(group_ids[len(y_fit):].tolist())
        assert fit_groups.isdisjoint(eval_groups)

    def test_no_group_ids_falls_back_to_last_tail(self) -> None:
        """Backwards-compat: without group_ids the carve is the legacy
        last-``frac`` tail."""
        from mlframe.training.composite.ensemble import _carve_inner_eval_split

        n = 2000
        X = np.arange(n).reshape(-1, 1).astype(np.float64)
        y = np.arange(n).astype(np.float64)
        X_fit, y_fit, X_eval, y_eval = _carve_inner_eval_split(
            X, y, frac=0.1, random_state=0, group_ids=None,
        )
        assert y_eval is not None
        assert y_fit[0] == 0
        assert y_eval[-1] == n - 1
        assert len(y_eval) == int(0.1 * n)

    def test_group_ids_too_few_groups_skips_es(self) -> None:
        """When unique groups < 4 a group-aware carve is impossible and a group-blind tail split would
        scatter one group across fit/eval (within-group leakage -> under-stopping), so the carve
        deliberately skips ES (returns None) rather than fall through to the leaky tail split."""
        from mlframe.training.composite.ensemble import _carve_inner_eval_split

        n = 2000
        X = np.arange(n).reshape(-1, 1).astype(np.float64)
        y = np.arange(n).astype(np.float64)
        group_ids = np.repeat([0, 1, 2], n // 3 + 1)[:n]
        X_fit, y_fit, X_eval, y_eval = _carve_inner_eval_split(
            X, y, frac=0.1, random_state=0, group_ids=group_ids,
        )
        assert y_eval is None


class TestAR1FailsafeToleranceDefault:
    def test_default_tolerance_is_010(self) -> None:
        from mlframe.training._composite_target_discovery_config import (
            CompositeTargetDiscoveryConfig,
        )
        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.lag_predict_failsafe_tolerance == pytest.approx(0.10)


class TestComputeOofPassesGroupIds:
    def test_external_holdout_signature_accepts_group_ids(self) -> None:
        import inspect
        from mlframe.training.composite.ensemble import (
            _compute_oof_with_external_holdout,
            compute_oof_holdout_predictions,
        )
        params = inspect.signature(compute_oof_holdout_predictions).parameters
        assert "group_ids" in params
        params_ext = inspect.signature(_compute_oof_with_external_holdout).parameters
        assert "group_ids" in params_ext


class TestXgbValDmatrixModuleCache:
    def test_val_cache_key_includes_train_key(self) -> None:
        """Sanity check: the val cache key composition mirrors train
        content so cross-train-content reuse isn't possible."""
        from mlframe.training import xgb_shim
        src = _module_source(xgb_shim)
        assert "(_signature_of(X_val), train_key)" in src

    def test_val_cache_promotes_module_hit_to_instance(self) -> None:
        """The cache lookup chain instance -> module -> fresh mirrors
        the train block; without the module fallback sklearn.clone()
        forces a full rebuild on every refit."""
        from mlframe.training import xgb_shim
        src = _module_source(xgb_shim)
        assert "_global_val_hit = _xgb_cache_get(val_key)" in src
        assert "_xgb_cache_put(val_key, dval)" in src


class TestPipelineCacheRamAware:
    def test_resolve_returns_value_above_2gb_when_ram_free(self) -> None:
        """On any developer machine with >10 GB RAM free the dynamic
        budget should exceed the legacy 2 GB hardcoded default."""
        import os
        from mlframe.training.strategies import _resolve_pipeline_cache_budget
        prior = os.environ.pop("MLFRAME_PIPELINE_CACHE_BYTES_LIMIT", None)
        try:
            budget = _resolve_pipeline_cache_budget()
            assert budget >= 2 * 1024 * 1024 * 1024
            assert budget <= 64 * 1024 * 1024 * 1024
        finally:
            if prior is not None:
                os.environ["MLFRAME_PIPELINE_CACHE_BYTES_LIMIT"] = prior

    def test_env_var_override_takes_priority(self) -> None:
        import os
        from mlframe.training.strategies import _resolve_pipeline_cache_budget
        prior = os.environ.get("MLFRAME_PIPELINE_CACHE_BYTES_LIMIT")
        os.environ["MLFRAME_PIPELINE_CACHE_BYTES_LIMIT"] = "12345678"
        try:
            assert _resolve_pipeline_cache_budget() == 12345678
        finally:
            if prior is None:
                os.environ.pop("MLFRAME_PIPELINE_CACHE_BYTES_LIMIT", None)
            else:
                os.environ["MLFRAME_PIPELINE_CACHE_BYTES_LIMIT"] = prior

    def test_pipeline_cache_default_budget_is_dynamic(self) -> None:
        import os
        from mlframe.training.strategies import PipelineCache
        prior = os.environ.pop("MLFRAME_PIPELINE_CACHE_BYTES_LIMIT", None)
        try:
            cache = PipelineCache(verbose=False)
            assert cache._bytes_limit >= 2 * 1024 * 1024 * 1024
        finally:
            if prior is not None:
                os.environ["MLFRAME_PIPELINE_CACHE_BYTES_LIMIT"] = prior


class TestMatplotlibSuptitleNoOverlap:
    def test_renderer_forces_constrained_layout_with_suptitle(self) -> None:
        from mlframe.reporting.renderers import matplotlib as mpl_renderer
        src = _module_source(mpl_renderer)
        assert "spec.constrained_layout or spec.suptitle" in src

    def test_renderer_save_uses_bbox_tight(self) -> None:
        from mlframe.reporting.renderers import matplotlib as mpl_renderer
        src = _module_source(mpl_renderer)
        assert 'bbox_inches="tight"' in src

    def test_fi_plot_save_uses_bbox_tight(self) -> None:
        from mlframe.feature_selection import importance
        src = _module_source(importance)
        assert 'bbox_inches="tight"' in src
