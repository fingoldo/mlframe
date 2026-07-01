"""display_estimator_name strips internal dataset-reuse / eval-scaling shim suffixes so they never leak into logs or
chart titles. The composite y-scale wrap-pass logs/chart title, the ``inner=`` log, and the train_eval start log all
route the estimator class name through this helper (prod TVT 2026-07: 'LGBMRegressorWithDatasetReuse' leaked back into
those five sites)."""
from __future__ import annotations

from mlframe.training.reporting._reporting import display_estimator_name, _SHIM_CLASS_SUFFIXES


def test_strips_each_shim_suffix():
    for suffix in _SHIM_CLASS_SUFFIXES:
        assert display_estimator_name(f"LGBMRegressor{suffix}") == "LGBMRegressor", suffix


def test_strips_stacked_shims():
    # Applied repeatedly so stacked shims collapse.
    assert display_estimator_name("LGBMRegressorWithDatasetReuseWithEvalSetScaling") == "LGBMRegressor"


def test_leaves_bare_and_unknown_names_unchanged():
    assert display_estimator_name("LGBMRegressor") == "LGBMRegressor"
    assert display_estimator_name("CatBoostRegressor") == "CatBoostRegressor"
    assert display_estimator_name("CompositeTargetEstimator") == "CompositeTargetEstimator"


def test_leak_sites_use_display_estimator_name():
    """The five prod leak sites (train_eval start log; composite wrap-pass chart title + inner= log; per-model-immediate
    log) must route the class name through display_estimator_name -- a new emit site with a raw type(...).__name__ would
    reintroduce the shim leak. Behavioral proxy: those modules import the stripper."""
    import mlframe.training.train_eval as te
    import mlframe.training.core._phase_composite_wrapping as cw
    for mod in (te, cw):
        src = __import__("inspect").getsource(mod)
        assert "display_estimator_name" in src, f"{mod.__name__} must use display_estimator_name for the class name"
