"""Every selection routine picks the null on data with no signal, in at least 9 of 10 seeds.

A selector that never prefers "nothing" turns noise into a shipped choice: the seasonal period picked the largest
candidate on pure noise, the power transform never had the identity on its grid, a spec found in one stability run of three
survived a truncated majority, and the drift check could not fire under the default threshold.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY

_SEEDS = range(10)


def _majority(hits: list[bool]) -> bool:
    """At least 9 of the 10 seeded runs chose the null."""
    return sum(hits) >= 9


def test_seasonal_period_is_none_on_noise_and_the_true_period_on_a_season():
    """No seasonality (period 1) on noise; 12, not a multiple of it, on a period-12 signal."""
    t = TRANSFORMS_REGISTRY["seasonal_residual"]
    noise = [t.fit(np.random.default_rng(s).normal(size=400), np.zeros(400)).get("period") in (None, 1) for s in _SEEDS]
    x = np.arange(480)
    season = [t.fit(np.sin(2 * np.pi * x / 12) + np.random.default_rng(s).normal(0.0, 0.3, 480), np.zeros(480)).get("period") == 12 for s in _SEEDS]
    assert _majority(noise) and _majority(season), (noise, season)


def test_signed_power_is_the_identity_on_a_symmetric_target():
    """``signed_power_y`` fits p == 1 on symmetric y."""
    t = TRANSFORMS_REGISTRY["signed_power_y"]
    assert _majority([float(t.fit(np.random.default_rng(s).normal(size=400), None).get("p")) == 1.0 for s in _SEEDS])


def test_the_stability_check_drops_a_spec_found_in_one_run_of_three(monkeypatch):
    """With n_bootstrap_runs=3 and the default 0.6 majority, a spec seen once is dropped and a spec seen twice kept."""
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    runs = iter([["always", "once"], ["always", "twice"], ["always", "twice"]])

    def fake_fit(self, *a, **k):
        """Each run finds the next scripted spec set."""
        self.specs_ = [types.SimpleNamespace(name=n) for n in next(runs)]
        return self

    monkeypatch.setattr(CompositeTargetDiscovery, "fit", fake_fit)
    df = pd.DataFrame({"b": np.arange(200.0), "y": np.arange(200.0)})
    disc = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(enabled=True, random_state=0))
    disc.fit_with_stability_check(df, "y", ["b"], np.arange(200), n_bootstrap_runs=3)
    assert disc.stability_counts_ == {"always": 3, "once": 1, "twice": 2}, disc.stability_counts_
    assert {s.name for s in disc.specs_} == {"always", "twice"}, disc.stability_counts_


@pytest.mark.slow
def test_default_discovery_emits_no_spec_on_pure_noise():
    """A target independent of every feature yields no spec under the default config, on every one of the 10 fixed seeds.

    The honest RMSE gate compared a composite only with the raw tiny model, which overfits noise and loses to the constant
    train mean: on seed 1 ten noise specs "beat raw" by 2-3 standard errors. The constant is now part of the null.
    """
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    empty = []
    for s in _SEEDS:
        rng = np.random.default_rng(s)
        n = 600
        df = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(4)}).assign(y=rng.normal(size=n))
        disc = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(enabled=True, random_state=s)).fit(df, "y", [f"x{i}" for i in range(4)], np.arange(n))
        empty.append(not disc.specs_)
    # The seeds are fixed, so the result is deterministic: one noise seed shipping specs is the defect, not variance.
    assert all(empty), f"noise specs emitted on seeds {[s for s, e in zip(_SEEDS, empty) if not e]}"


def test_auto_chain_proposes_nothing_when_raw_beats_every_transform():
    """A DGP whose base carries no signal: no chain clears the bar of beating raw y (DSC-20)."""
    from mlframe.training.composite.discovery._auto_chain import discover_chains

    hits = []
    for s in _SEEDS:
        rng = np.random.default_rng(s)
        x = rng.normal(size=(600, 3))
        base = rng.normal(100.0, 1.0, 600)  # unrelated to y
        y = 3.0 * x[:, 0] - 2.0 * x[:, 1] + rng.normal(0.0, 0.1, 600)
        hits.append(discover_chains(y=y, base=base, x_matrix=x, cv_folds=3, n_estimators=25, compute_mi_gain=False, random_state=s) == [])
    assert _majority(hits), hits


def test_the_change_point_scan_finds_no_break_in_a_stationary_buffer():
    """One stable regime: the streaming refit's change-point scan must not invent a break."""
    from mlframe.training.composite.streaming import _detect_change_point

    hits = []
    for s in _SEEDS:
        rng = np.random.default_rng(s)
        b = rng.uniform(0.0, 10.0, 400)
        hits.append(int(_detect_change_point(1.5 * b + rng.normal(0.0, 1.0, 400), b).get("change_point", -1)) < 0)
    assert _majority(hits), hits


def _moe_on(expert_sd: float, seed: int):
    """A MoE gate fitted on 20 groups of 30 rows where two experts have noise ``expert_sd`` and lag has noise 1."""
    from mlframe.training.composite._moe_gate import MoESelectionGate

    rng = np.random.default_rng(seed)
    n = 600
    y = rng.normal(0.0, 1.0, n)
    preds = {"lag": y + rng.normal(0.0, 1.0, n), "a": y + rng.normal(0.0, expert_sd, n), "b": y + rng.normal(0.0, expert_sd, n)}
    return MoESelectionGate().fit(y, preds, group_ids=np.arange(n) % 20)


def test_the_moe_gate_keeps_lag_when_no_expert_is_better():
    """Experts 10% noisier than lag: the per-group choice stays on lag instead of following each group's noise.

    With no significance bar (one row per group, zero shrink) these experts won half the groups and the deployed gate served
    5% worse than plain lag on fresh rows; the paired-gain z-test keeps lag in essentially every group.
    """
    hits = [np.mean([v == "lag" for v in _moe_on(1.1, s).group_choice_.values()]) >= 0.9 for s in _SEEDS]
    assert _majority(hits), hits


def test_the_moe_gate_still_picks_an_expert_that_is_really_better():
    """The control for the canary above: an expert with 40% less noise than lag wins most groups."""
    hits = [np.mean([v != "lag" for v in _moe_on(0.6, s).group_choice_.values()]) >= 0.6 for s in _SEEDS]
    assert _majority(hits), hits


def test_the_count_blend_chooses_heavy_pooling_when_entities_carry_no_signal():
    """No entity effect: cross-validation picks a large smoothing constant (trust the global model), not an entity fit on noise."""
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor

    from mlframe.training.composite.count_weighted_blend import CountWeightedBlendEnsemble

    hits = []
    for s in _SEEDS:
        rng = np.random.default_rng(s)
        X = pd.DataFrame({"ent": rng.integers(0, 40, 500), "m": rng.normal(size=500)})
        y = 2.0 * X["m"].to_numpy() + rng.normal(0.0, 1.0, 500)
        est = CountWeightedBlendEnsemble(entity_estimator=DecisionTreeRegressor(max_depth=8, random_state=0),
                                         global_estimator=LinearRegression(), entity_col="ent", metadata_cols=["m"],
                                         auto_k=True, random_state=s).fit(X, y)
        hits.append(est.k_ >= 100.0)
    assert _majority(hits), hits


def test_the_train_metric_ensemble_falls_back_when_no_component_beats_the_baseline():
    """Every component worse than the naive baseline: a single best component is returned, not a weighted ensemble of losers."""
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble

    comps = [types.SimpleNamespace(name=f"c{i}") for i in range(3)]
    out = CompositeCrossTargetEnsemble.from_train_metrics(
        component_models=comps, component_names=["c0", "c1", "c2"], component_oof_rmse=[5.0, 4.0, 6.0], baseline_oof_rmse=3.0,
    )
    assert not isinstance(out, CompositeCrossTargetEnsemble), "an ensemble was built although nothing beat the baseline"
    assert out is comps[1], "the fallback must be the best-RMSE component"


# ---------------------------------------------------------------------------
# Meta-guard: every selection routine in composite/ has a canary or a reason not to.
# ---------------------------------------------------------------------------

# Routines that choose among candidates by a noisy score, and the canary that makes each pick the null on noise.
NULL_CANARIES = {
    "_moe_gate.py::_pick_global": "test_the_moe_gate_keeps_lag_when_no_expert_is_better",
    "_moe_gate.py::_pick_per_group": "test_the_moe_gate_keeps_lag_when_no_expert_is_better",
    "_moe_gate.py::_choose_tier1": "test_the_moe_gate_keeps_lag_when_no_expert_is_better",
    "count_weighted_blend.py::_select_k_via_cv": "test_the_count_blend_chooses_heavy_pooling_when_entities_carry_no_signal",
    "discovery/_stability.py::stability_select_specs": "test_the_stability_check_drops_a_spec_found_in_one_run_of_three",
    "ensemble/_cross_target.py::from_train_metrics": "test_the_train_metric_ensemble_falls_back_when_no_component_beats_the_baseline",
    "streaming.py::_detect_change_point": "test_the_change_point_scan_finds_no_break_in_a_stationary_buffer",
}
# Functions the name/argmin heuristic matches that do not choose among candidates by a score, with the reason.
_NOT_SELECTION = {
    "_calibration_binning.py::top_label_calibration_bins": "argmax picks the top label of each probability row, not a candidate",
    "_estimator_dispatch.py::_pick_base_column": "resolves which configured column holds the base; no score involved",
    "conformal_classification.py::_aps_true_label_scores": "argmax locates the true label's rank in a sorted row",
    "conformal_classification.py::predict_set": "argmax keeps the prediction set non-empty; the set itself is the coverage rule",
    "diagnostics.py::_top_value_share": "argmax finds the modal value for a share statistic",
    "discovery/_honest_holdout.py::split_holdout_select_report": "splits rows into roles; selects nothing by score",
    "discovery/_honest_rmse_gate.py::_move_rmse_stamps_to_selection": "moves stamped numbers between fields",
    "discovery/_ktc_dispatch.py::choose_corr_backend": "picks a compute backend by hardware and size, not by a fitted score",
    "discovery/_ktc_dispatch.py::choose_collinear_backend": "picks a compute backend by hardware and size, not by a fitted score",
    "ensemble/__init__.py::_has_supervised_selection": "inspects a pipeline for a selector step",
    "estimator/_estimator_helpers.py::_carry_forward_fill": "argmax finds the last finite index to carry forward",
    "grouped_block_stacking.py::_select_group": "selects the rows of one group",
    "highlevel.py::_select_rows": "slices rows",
    "highlevel.py::_select_cols": "slices columns",
    "hpo.py::select_ensemble_from_pool": ("stepwise selection over a pool with no baseline member; averaging noisy members is a "
                                          "real gain, so there is no null to pick, and the OOF pool is its overfit guard"),
    "hpo_ensembling.py::select_oof_pool_ensemble": "the same routine as select_ensemble_from_pool, called directly",
    "pseudo_labeling.py::_select": "keeps rows above a confidence threshold; a threshold rule, not a choice among candidates",
    "quantile.py::fit": "argmin/argmax sort quantile columns to repair crossing",
    "quantile.py::predict": "argmin/argmax sort quantile columns to repair crossing",
    "segment_routed.py::_select_columns": "slices columns",
}


def _selection_candidates() -> set[str]:
    """Every function under composite/ named like a selector or computing an argmin/argmax."""
    import ast
    import re
    from pathlib import Path

    import mlframe

    root = Path(mlframe.__file__).resolve().parent / "training" / "composite"
    name_re = re.compile(r"select|choose|pick", re.I)
    out = set()
    for path in sorted(root.rglob("*.py")):
        if "_benchmarks" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(root).as_posix()
        for func in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
            argm = any(isinstance(c, ast.Call) and getattr(c.func, "attr", getattr(c.func, "id", None)) in
                       {"argmin", "argmax", "nanargmin", "nanargmax"} for c in ast.walk(func))
            if argm or name_re.search(func.name):
                out.add(f"{rel}::{func.name}")
    return out


def test_every_selection_routine_has_a_null_canary_or_a_reason():
    """A new selector must come with a canary that makes it pick the null on noise, or a reason it is not a selector."""
    found = _selection_candidates()
    assert found, "the scan found no candidate at all; it lost its subject"
    unregistered = sorted(found - set(NULL_CANARIES) - set(_NOT_SELECTION))
    stale = sorted((set(NULL_CANARIES) | set(_NOT_SELECTION)) - found)
    assert not unregistered, f"add a null canary to NULL_CANARIES or a reason to _NOT_SELECTION: {unregistered}"
    assert not stale, f"these entries name routines that no longer exist: {stale}"
    missing = sorted({t for t in NULL_CANARIES.values() if t not in globals()})
    assert not missing, f"NULL_CANARIES names canaries that do not exist: {missing}"
    assert all(str(r).strip() for r in _NOT_SELECTION.values())
