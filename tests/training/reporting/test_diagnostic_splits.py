"""``ReportingConfig.diagnostic_splits`` limits chosen post-fit diagnostics to some splits; unset, every diagnostic runs on every split."""

import numpy as np
import pytest

from mlframe.training.configs import ReportingConfig
from mlframe.training.reporting._diagnostics_budget import DiagnosticsBudget


def _run_all(split, rules):
    """Run three named diagnostics through a budget for ``split`` and return (which ran, charts accounting)."""
    charts: dict = {}
    budget = DiagnosticsBudget(0.0, charts=charts, split=split, split_rules=rules)
    ran = [name for name in ("decile_table", "decision_curve", "interaction_strength") if budget.run(name, lambda: True)]
    return ran, charts


def test_diagnostic_restricted_to_test_is_skipped_on_val_and_recorded():
    """A diagnostic limited to the test split does not run on val, and the skip names the knob."""
    ran, charts = _run_all("val", {"decile_table": ("test",), "interaction_strength": ("test",)})
    assert ran == ["decision_curve"]
    assert set(charts["skipped"]) == {"decile_table", "interaction_strength"}
    assert "diagnostic_splits" in charts["skipped"]["decile_table"]


def test_diagnostic_restricted_to_test_still_runs_on_test():
    """The same rules let everything through on the split they allow."""
    ran, charts = _run_all("test", {"decile_table": ("test",), "interaction_strength": ("test",)})
    assert ran == ["decile_table", "decision_curve", "interaction_strength"]
    assert not charts


@pytest.mark.parametrize("split", ["val", "test", "train"])
def test_unset_rules_change_nothing(split):
    """Default (no rules): every diagnostic runs on every split."""
    ran, charts = _run_all(split, None)
    assert ran == ["decile_table", "decision_curve", "interaction_strength"]
    assert not charts


def test_split_names_are_matched_case_insensitively():
    """Config spelling like ("Test",) must not silently disable the diagnostic everywhere."""
    ran, _ = _run_all("test", {"decile_table": ("Test",)})
    assert "decile_table" in ran


def test_reporting_config_accepts_the_knob_and_defaults_to_off():
    """The knob is a plain name -> splits mapping on ReportingConfig, unset by default."""
    assert ReportingConfig().diagnostic_splits is None
    cfg = ReportingConfig(diagnostic_splits={"decile_table": ("test",)})
    assert cfg.diagnostic_splits == {"decile_table": ("test",)}


def test_post_fit_diagnostics_honour_the_split_rule(tmp_path):
    """End to end through the real diagnostics block: decile_table renders for the test report but not for the val report."""
    from mlframe.training.reporting._reporting_diagnostics import _render_post_fit_diagnostics

    rng = np.random.default_rng(0)
    n = 600
    y = (rng.random(n) > 0.5).astype(int)
    p1 = np.clip(0.5 * y + 0.25 + rng.normal(0, 0.15, n), 0.01, 0.99)
    probs = np.column_stack([1 - p1, p1])
    cfg = ReportingConfig(
        diagnostic_splits={"decile_table": ("test",)}, decision_curve=False, engineered_separability_charts=False,
        class_structure_charts=False, category_discriminability_charts=False, interaction_strength_charts=False,
        combined_html=False, async_render=False,
    )
    out = {}
    for split in ("val", "test"):
        metrics: dict = {}
        _render_post_fit_diagnostics(
            targets=y, model=None, df=None, columns=None, preds=(p1 > 0.5).astype(int), probs=probs, target_type="binary_classification",
            plot_file=str(tmp_path / f"m_{split}"), plot_outputs="matplotlib[png]", metrics=metrics, reporting_config=cfg, model_name="m",
        )
        out[split] = metrics["charts"]
    assert "decile_table" in out["test"]["saved"]
    assert "decile_table" not in out["val"].get("saved", []) and "decile_table" in out["val"]["skipped"]
