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

import numpy as np
import pytest


def _spy_savefig(monkeypatch) -> list:
    """Record the kwargs of every ``Figure.savefig`` call while still saving."""
    from matplotlib.figure import Figure

    seen: list = []
    orig = Figure.savefig

    def spy(self, *args, **kwargs):
        """Record kwargs, then delegate."""
        seen.append(kwargs)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", spy)
    return seen


def _xgb_val_data():
    """Two distinct train frames and one val frame."""
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 200
    X1 = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    X2 = pd.DataFrame({"a": rng.normal(size=n) + 5.0, "b": rng.normal(size=n)})
    Xv = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    y = rng.normal(size=n)
    yv = rng.normal(size=60)
    return X1, X2, Xv, y, yv


class TestCarveInnerEvalSplitGroupAware:
    """Groups tests covering carve inner eval split group aware."""
    def test_group_ids_keeps_groups_whole(self) -> None:
        """No group spans fit and eval slices when group_ids is given."""
        from mlframe.training.composite.ensemble import _carve_inner_eval_split

        n = 2000
        rng = np.random.default_rng(42)
        X = rng.normal(size=(n, 3))
        y = rng.normal(size=n)
        n_groups = 200
        group_ids = np.repeat(np.arange(n_groups), n // n_groups)

        _X_fit, y_fit, X_eval, y_eval = _carve_inner_eval_split(
            X,
            y,
            frac=0.1,
            random_state=0,
            group_ids=group_ids,
        )
        assert X_eval is not None and y_eval is not None
        eval_mask = np.zeros(n, dtype=bool)
        eval_mask[len(y_fit) :] = True
        fit_groups = set(group_ids[: len(y_fit)].tolist())
        eval_groups = set(group_ids[len(y_fit) :].tolist())
        assert fit_groups.isdisjoint(eval_groups)

    def test_no_group_ids_falls_back_to_last_tail(self) -> None:
        """Backwards-compat: without group_ids the carve is the legacy
        last-``frac`` tail."""
        from mlframe.training.composite.ensemble import _carve_inner_eval_split

        n = 2000
        X = np.arange(n).reshape(-1, 1).astype(np.float64)
        y = np.arange(n).astype(np.float64)
        _X_fit, y_fit, _X_eval, y_eval = _carve_inner_eval_split(
            X,
            y,
            frac=0.1,
            random_state=0,
            group_ids=None,
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
        _X_fit, _y_fit, _X_eval, y_eval = _carve_inner_eval_split(
            X,
            y,
            frac=0.1,
            random_state=0,
            group_ids=group_ids,
        )
        assert y_eval is None


class TestAR1FailsafeToleranceDefault:
    """Groups tests covering a r1 failsafe tolerance default."""
    def test_default_tolerance_is_010(self) -> None:
        """Default tolerance is 010."""
        from mlframe.training._composite_target_discovery_config import (
            CompositeTargetDiscoveryConfig,
        )

        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.lag_predict_failsafe_tolerance == pytest.approx(0.10)


class TestComputeOofPassesGroupIds:
    """Groups tests covering compute oof passes group ids."""
    def test_external_holdout_signature_accepts_group_ids(self) -> None:
        """External holdout signature accepts group ids."""
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
    """Groups tests covering xgb val dmatrix module cache."""
    def test_val_cache_key_includes_train_key(self, monkeypatch) -> None:
        """The same val frame fitted against a DIFFERENT train frame must rebuild its val DMatrix (quantile cuts differ),
        while the same train frame reuses it."""
        pytest.importorskip("xgboost")
        from sklearn.base import clone
        from mlframe.training import xgb_shim

        X1, X2, Xv, y, yv = _xgb_val_data()
        builds: list = []
        orig = xgb_shim._build_quantile_dmatrix

        def counting(*args, **kwargs):
            """Record whether a val build (ref_dmatrix given) or a train build ran."""
            builds.append("val" if kwargs.get("ref_dmatrix") is not None else "train")
            return orig(*args, **kwargs)

        monkeypatch.setattr(xgb_shim, "_build_quantile_dmatrix", counting)
        xgb_shim._xgb_cache_clear()
        try:
            base = xgb_shim.XGBRegressorWithDMatrixReuse(n_estimators=2)
            base.fit(X1, y, eval_set=[(Xv, yv)])
            assert builds == ["train", "val"]
            clone(base).fit(X1, y, eval_set=[(Xv, yv)])
            assert builds == ["train", "val"], "same train + same val must reuse both DMatrix objects"
            clone(base).fit(X2, y, eval_set=[(Xv, yv)])
            assert builds[2:] == ["train", "val"], "a different train frame must not reuse the val DMatrix built against the first"
        finally:
            xgb_shim._xgb_cache_clear()

    def test_val_cache_promotes_module_hit_to_instance(self, monkeypatch) -> None:
        """A sklearn.clone() has an empty instance cache: its val DMatrix comes from the module cache (no rebuild) and is
        then promoted onto the clone so the next fit takes the instance path."""
        pytest.importorskip("xgboost")
        from sklearn.base import clone
        from mlframe.training import xgb_shim

        X1, _X2, Xv, y, yv = _xgb_val_data()
        builds: list = []
        orig = xgb_shim._build_quantile_dmatrix

        def counting(*args, **kwargs):
            """Record each DMatrix build."""
            builds.append(1)
            return orig(*args, **kwargs)

        monkeypatch.setattr(xgb_shim, "_build_quantile_dmatrix", counting)
        xgb_shim._xgb_cache_clear()
        try:
            base = xgb_shim.XGBRegressorWithDMatrixReuse(n_estimators=2)
            base.fit(X1, y, eval_set=[(Xv, yv)])
            n_built = len(builds)
            assert n_built == 2
            twin = clone(base)
            assert twin._cached_val_dmatrix is None
            twin.fit(X1, y, eval_set=[(Xv, yv)])
            assert len(builds) == n_built
            assert twin._cached_val_dmatrix is base._cached_val_dmatrix
            assert twin._cached_val_key == base._cached_val_key
            twin.fit(X1, y, eval_set=[(Xv, yv)])
            assert len(builds) == n_built
        finally:
            xgb_shim._xgb_cache_clear()


class TestPipelineCacheRamAware:
    """Groups tests covering pipeline cache ram aware."""
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
        """Env var override takes priority."""
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
        """Pipeline cache default budget is dynamic."""
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
    """Groups tests covering matplotlib suptitle no overlap."""
    def test_renderer_forces_constrained_layout_with_suptitle(self) -> None:
        """A figure with a suptitle gets a constrained layout engine without asking for one."""
        import matplotlib.pyplot as plt
        from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
        from mlframe.reporting.spec import AnnotationPanelSpec, FigureSpec

        panel = AnnotationPanelSpec(text="note", title="t")
        fig = MatplotlibRenderer().render(FigureSpec(panels=((panel,),), figsize=(6.0, 3.0), suptitle="Model identity"))
        try:
            engine = fig.get_layout_engine()
            assert engine is not None
            assert type(engine).__name__ == "ConstrainedLayoutEngine"
        finally:
            plt.close(fig)

    def test_renderer_save_uses_bbox_tight(self, monkeypatch, tmp_path) -> None:
        """The renderer's save writes the file with a tight bounding box."""
        import matplotlib.pyplot as plt
        from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
        from mlframe.reporting.spec import AnnotationPanelSpec, FigureSpec

        seen = _spy_savefig(monkeypatch)
        renderer = MatplotlibRenderer()
        fig = renderer.render(FigureSpec(panels=((AnnotationPanelSpec(text="note", title="t"),),), figsize=(6.0, 3.0), suptitle="S"))
        try:
            out = tmp_path / "fig.png"
            renderer.save(fig, str(out), "png")
        finally:
            plt.close(fig)
        assert out.is_file() and out.stat().st_size > 0
        assert len(seen) == 1
        assert seen[0]["bbox_inches"] == "tight"

    def test_fi_plot_save_uses_bbox_tight(self, monkeypatch, tmp_path) -> None:
        """The feature-importance plot is saved with a tight bounding box so long feature names are not clipped."""
        import matplotlib.pyplot as plt
        from mlframe.feature_selection.importance import plot_feature_importance

        seen = _spy_savefig(monkeypatch)
        out = tmp_path / "fi.png"
        names = [f"a_rather_long_feature_name_number_{i}" for i in range(5)]
        plot_feature_importance(np.array([0.5, 0.1, 0.3, 0.2, 0.4]), names, "kind", show_plots=False, plot_file=str(out))
        plt.close("all")
        assert out.is_file() and out.stat().st_size > 0
        assert len(seen) == 1
        assert seen[0]["bbox_inches"] == "tight"
