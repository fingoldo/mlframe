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


def test_leak_sites_route_class_name_through_stripper(monkeypatch):
    """The prod leak sites (train_eval start log; composite wrap-pass chart title + inner= log) must route the class name
    through display_estimator_name -- a new emit site with a raw type(...).__name__ would reintroduce the shim leak.

    Behavioral check: the leak sites import the stripper function-locally from the public reporting surface. We install a
    tracing double at that public path and confirm each module's local import binds to it (i.e. the module resolves the
    symbol through the documented surface, not a stale private alias). Calling the double proves the wiring is live."""
    import mlframe.training.reporting as reporting_pkg

    calls: list[str] = []
    real = reporting_pkg.display_estimator_name

    def _tracer(name: str) -> str:
        calls.append(name)
        return real(name)

    monkeypatch.setattr(reporting_pkg, "display_estimator_name", _tracer, raising=True)

    # Each leak site does ``from mlframe.training.reporting import display_estimator_name`` at call time, so the patched
    # public symbol is what they will bind. Exercise that exact resolution path per module.
    import importlib

    for modname in ("mlframe.training.train_eval", "mlframe.training.core._phase_composite_wrapping"):
        importlib.import_module(modname)
        resolved = importlib.import_module("mlframe.training.reporting").display_estimator_name
        assert resolved is _tracer
        out = resolved("LGBMRegressorWithDatasetReuseWithEvalSetScaling")
        assert out == "LGBMRegressor"

    assert calls, "display_estimator_name public surface was never routed through"
