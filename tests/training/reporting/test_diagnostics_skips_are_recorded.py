"""Dropped diagnostics are visible in charts, and only real ensemble variants lose the heavy diagnostics."""

import pytest

from mlframe.training.reporting._diagnostics_budget import DiagnosticsBudget, HeavyDiagnosticsPolicy, is_ensemble_variant_name


@pytest.mark.parametrize("name", ["sensor_lgbm", "density_model", "IntensityRegressor", "CB_prices", "lgbm"])
def test_a_primary_model_is_not_mistaken_for_an_ensemble_variant(name):
    """The bare substring test "ens" dropped SHAP / PDP for any model named for a sensor, density or intensity."""
    assert not is_ensemble_variant_name(name)


@pytest.mark.parametrize("name", ["EnsARITHM cb_lgb", "EnsHARM x", "Conf Ensemble arithm", "ensemble", "cb_ensemble"])
def test_real_ensemble_variants_are_recognised(name):
    assert is_ensemble_variant_name(name)


def test_a_budget_skip_lands_in_charts():
    """A truncated report must not look complete to a consumer reading charts."""
    charts: dict = {"saved": [], "failed": []}
    budget = DiagnosticsBudget(1e-9, charts=charts)
    budget._t0 -= 1.0  # already over budget
    assert budget.run("shap", lambda: 1) is None
    assert "shap" in charts["skipped"] and "budget" in charts["skipped"]["shap"]


def test_an_out_of_scope_skip_lands_in_charts():
    charts: dict = {}
    budget = DiagnosticsBudget(0.0, policy=HeavyDiagnosticsPolicy(mode="best", is_primary=False), charts=charts)
    heavy = next(iter(__import__("mlframe.training.reporting._diagnostics_budget", fromlist=["HEAVY_DIAGNOSTICS"]).HEAVY_DIAGNOSTICS))
    assert budget.run(heavy, lambda: 1) is None
    assert heavy in charts["skipped"]


def test_the_interaction_budget_skip_is_recorded(monkeypatch):
    """Over budget, the diagnostic returned False having recorded nothing - identical to the knob being off."""
    import numpy as np
    import pandas as pd

    from mlframe.reporting import _diagnostics_pdp as pdp

    monkeypatch.setattr(pdp, "_interaction_cost_within_budget", lambda *a, **k: False)

    class _Model:
        def predict(self, X):
            return np.zeros(len(X))

    metrics: dict = {}
    df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    ok = pdp.render_interaction_strength_diagnostic(
        model=_Model(), df=df, feature_names=["a", "b"], feature_importances=None, plot_outputs="png",
        base_path="unused", metrics_dict=metrics, max_seconds=1.0,
    )
    assert ok is False
    assert "interaction_strength" in metrics["charts"]["skipped"]
