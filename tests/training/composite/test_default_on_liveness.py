"""Every default-ON corrective mechanism is shown to change something on the default path.

A default of True proves nothing when the path that ships never reaches the code: the soft base-shrink sat inert on every
wrapper ``from_fitted_inner`` built, the OOF pre-screen never ran under the default ``kfold`` source, drift detection could
not fire under the default ``eps_mi_gain``, FDR control had no p-values to correct, and the knn cost guard measured only
after the work it existed to prevent. Five estimator MoE parameters defaulted on and were never read at all.

``DEFAULT_ON_MECHANISMS`` names, for each knob, the test that asserts the mechanism's EFFECT with the knob at its default;
``INERT_BY_DESIGN`` records a knob that is legitimately inert on the default path, with the reason. A new default-True knob
matching the corrective-name pattern must land in one of the two.
"""

from __future__ import annotations

import ast
import inspect
import math
import re
import types
from pathlib import Path

import pytest

from mlframe.training.composite import CompositeTargetDiscovery, CompositeTargetEstimator
from mlframe.training.configs import CompositeTargetDiscoveryConfig

_ROOT = Path(__file__).resolve().parents[3]
_KNOB = re.compile(r"_enabled$|_control$|^soft_|^enable_|watchdog|guard")

DEFAULT_ON_MECHANISMS: dict[str, str] = {
    # Estimator.
    "soft_base_shrink": "tests/training/composite/estimator/test_biz_val_soft_base_shrink.py::test_per_row_flag_is_exact",
    # Discovery config, discovery fit.
    "interaction_base_discovery_enabled": "tests/training/composite/discovery/test_biz_val_discovery_default_on_steps.py::test_biz_val_interaction_base_default_on_fires_on_pure_interaction",
    "auto_chain_discovery_enabled": "tests/training/composite/discovery/test_biz_val_discovery_default_on_steps.py::test_biz_val_auto_chain_default_on_appends_chain_specs",
    "multi_base_enabled": "tests/training/composite/transforms/test_composite_multi_base_integration.py::test_default_on_promotes_spec_on_two_base_dgp",
    "transform_waic_validation_enabled": "tests/training/composite/discovery/test_biz_val_discovery_waic_validation.py::test_waic_scores_populated_only_when_flag_enabled",
    "yscale_holdout_gate_enabled": "tests/training/composite/discovery/test_biz_val_discovery_yscale_holdout_gate.py::test_biz_val_yscale_gate_drops_collapsing_high_alpha_spec",
    "structural_fragility_gate_enabled": "tests/training/composite/discovery/test_biz_val_discovery_yscale_holdout_gate.py::test_rejection_ledger_records_structural_gate_drops",
    "mi_gain_fdr_control": "tests/training/composite/eval/test_eval_stats_by_fdr.py::test_inactive_fdr_control_says_so",
    "honest_rmse_gate_enabled": "tests/training/composite/discovery/test_biz_val_honest_rmse_gate.py::test_biz_val_honest_rmse_gate_rejects_mi_positive_ratio_pair",
    "honest_oof_floor_reject_enabled": "tests/training/composite/test_default_on_liveness.py::test_the_honest_oof_floor_rejects_a_spec_that_loses_to_it",
    # Discovery config, suite post-processing.
    "moe_gate_enabled": "tests/training/core/test_composite_post_moe_value_report.py::test_moe_gate_and_value_report_end_to_end",
    "ood_lag_routing_enabled": "tests/training/core/test_biz_val_ood_lag_router.py::test_biz_val_router_beats_both_all_raw_and_all_lag",
    "volatility_lag_routing_enabled": "tests/training/core/test_biz_val_volatility_lag_router.py::test_biz_val_router_beats_both_all_raw_and_all_lag",
    "enable_wrap_pass_watchdog": "tests/training/composite/estimator/test_wrap_watchdog_oracle.py::test_the_watchdog_runs_when_the_metric_block_is_skipped",
    "ct_ensemble_dummy_floor_enabled": "tests/training/core/test_xt_ensemble_leaky_prescreen.py::test_the_dummy_floor_gate_drops_a_component_that_loses_to_the_dummy_by_default",
}

INERT_BY_DESIGN: dict[str, str] = {
    "enable_multiseed_early_stop": (
        "superseded on the default path: honest_oof_selection=True makes the honest-OOF prepass skip the CV seed loop the early stop "
        "would cut short; live when honest_oof_selection=False (test_multiseed_early_stop.py::TestEndToEndKeptSpecParity)"
    ),
}

# Default-on behaviours without a boolean knob, each pinned where it was fixed.
MECHANISMS_WITHOUT_A_KNOB: dict[str, str] = {
    "incremental drift detection": "tests/training/composite/discovery/test_incremental_discovery.py::test_default_config_detects_a_destroyed_base_relation",
    "knn budget guard before auto-base": "tests/training/composite/discovery/test_knn_mi_budget.py::test_the_guard_covers_auto_base_ranking",
    "OOF pre-screen under the kfold source": "tests/training/core/test_xt_ensemble_leaky_prescreen.py::test_the_validation_split_is_used_when_the_oof_path_supplies_no_frame",
    "auto-enabled discovery reaches post-processing": "tests/training/core/test_auto_enabled_discovery_reaches_postprocessing.py::test_auto_enable_publishes_the_effective_decision_for_later_phases",
}


def _default_true_knobs() -> set[str]:
    """Default-True corrective-named fields of the discovery config and constructor parameters of the estimator."""
    cfg = {n for n, f in CompositeTargetDiscoveryConfig.model_fields.items() if f.default is True and _KNOB.search(n)}
    est = {n for n, p in inspect.signature(CompositeTargetEstimator.__init__).parameters.items() if p.default is True and _KNOB.search(n)}
    return cfg | est


def _test_source(node_id: str) -> str | None:
    """Source of the test function (or class) a ``path::name`` node id names, or None when it does not exist."""
    path, name = node_id.split("::", 1)
    file = _ROOT / path
    if not file.exists():
        return None
    text = file.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(text, node)
    return None


def test_every_default_on_knob_is_accounted_for():
    """Each default-True corrective knob has a liveness test or a recorded reason it is inert by default; stale entries leave."""
    knobs = _default_true_knobs()
    listed = set(DEFAULT_ON_MECHANISMS) | set(INERT_BY_DESIGN)
    assert not set(DEFAULT_ON_MECHANISMS) & set(INERT_BY_DESIGN)
    assert not knobs - listed, f"default-on knobs with no liveness test or inert reason: {sorted(knobs - listed)}"
    assert not listed - knobs, f"entries for knobs that no longer default on: {sorted(listed - knobs)}"


def _config_calls_with_knob_off(src: str, knob: str) -> tuple[int, int]:
    """Config-building calls in ``src``: how many there are, and how many of them set ``knob=False``."""
    total = off = 0
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if "config" not in fn.lower() and "cfg" not in fn.lower():
            continue
        total += 1
        off += any(k.arg == knob and isinstance(k.value, ast.Constant) and k.value.value is False for k in node.keywords)
    return total, off


@pytest.mark.parametrize("knob", sorted(DEFAULT_ON_MECHANISMS))
def test_the_liveness_test_exists_and_keeps_the_knob_on(knob: str):
    """The named test exists and builds at least one configuration with the knob on; an OFF run is allowed only as its control."""
    src = _test_source(DEFAULT_ON_MECHANISMS[knob])
    assert src is not None, f"{knob}: {DEFAULT_ON_MECHANISMS[knob]} does not exist"
    total, off = _config_calls_with_knob_off(src, knob)
    assert not (off and off == total), f"{knob}: every configuration its liveness test builds turns the knob off"


def test_the_knob_check_sees_an_all_off_test():
    """Canary: a test whose only configuration switches the knob off is caught; an off control beside an on run is not."""
    only_off = "def t():\n    run(make_config(x_enabled=False))\n"
    with_control = "def t():\n    run(make_config(x_enabled=False))\n    run(make_config())\n"
    assert _config_calls_with_knob_off(only_off, "x_enabled") == (1, 1)
    assert _config_calls_with_knob_off(with_control, "x_enabled") == (2, 1)


@pytest.mark.parametrize("name", sorted(MECHANISMS_WITHOUT_A_KNOB))
def test_knobless_mechanisms_are_pinned(name: str):
    """Each default-on behaviour without a knob keeps its pinning test."""
    assert _test_source(MECHANISMS_WITHOUT_A_KNOB[name]) is not None, MECHANISMS_WITHOUT_A_KNOB[name]


def test_the_honest_oof_floor_rejects_a_spec_that_loses_to_it():
    """Under the default config a spec whose honest-OOF RMSE is above min(raw, lag) x 1.05 is dropped and ledgered."""
    from mlframe.training.composite.discovery._tiny_rerank_waic import apply_honest_oof_floor

    disc = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig())
    specs = [types.SimpleNamespace(name=n, base_column="b", transform_name="linear_residual") for n in ("wins", "loses")]
    kept, scores = apply_honest_oof_floor(disc, specs, [1.0, 2.0], {"wins": 9.0, "loses": 12.0}, 10.0)
    assert [s.name for s in kept] == ["wins"] and scores == [1.0]
    assert any(e["stage"] == "honest_oof_floor" and e["spec_name"] == "loses" for e in disc.rejection_ledger)
    assert math.isclose(disc._tiny_rerank_scores["wins"], 1.0)
