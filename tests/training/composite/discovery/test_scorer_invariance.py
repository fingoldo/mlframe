"""Scores are compared only within one unit, split and estimator.

Each defect here compared numbers that measure different things: a WAIC over T units across transforms of different T
scales, honest-holdout RMSE beside optimistic in-group CV RMSE, relative RMSE gains sorted with nats, MI statistics a
multi-base spec copied from its seed, a val-split dummy floor against train-OOF components, a de-duplicated MI(T, X)
against a full MI(y, X), and composite T-scale metrics in a table of y-scale target quality.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pandas as pd

from mlframe.training.composite.discovery import _tiny_rerank_waic
from mlframe.training.composite.discovery._tiny_rerank_honest import _put_unmeasured_on_the_honest_scale
from mlframe.training.core._phase_composite_discovery_gates import rank_pending_composites
from mlframe.training.core._phase_composite_post_xt_ensemble._prescreen import same_split_dummy_rmse
from mlframe.training.targets_performance import targets_performance_frame


def _spec(name: str, transform: str, base: str = "b"):
    """A minimal spec stand-in with the attributes the tie-break reads."""
    return types.SimpleNamespace(name=name, transform_name=transform, base_column=base, fitted_params={})


def _waic_order(monkeypatch, transforms: list[str]) -> list[int]:
    """Run the tie-break on a single near-tied band whose WAICs rank the members in reverse RMSE order."""
    from mlframe.training.composite.transforms import get_transform

    specs = [_spec(f"s{i}", t, base=f"b{i}") for i, t in enumerate(transforms)]
    n = 200
    for sp in specs:  # real fitted params, so each spec's forward runs and its WAIC is scored
        sp.fitted_params = get_transform(sp.transform_name).fit(np.linspace(2.0, 3.0, n), np.linspace(1.0, 2.0, n))
    cache = {f"b{i}": (np.linspace(1.0, 2.0, n), np.random.default_rng(i).normal(size=(n, 2))) for i in range(len(specs))}
    waics = iter(range(len(specs)))
    monkeypatch.setattr("mlframe.training.composite.discovery._eval_waic.compute_transform_waic",
                        lambda *a, **k: types.SimpleNamespace(valid=True, waic=float(next(waics))))
    self = types.SimpleNamespace(config=types.SimpleNamespace(transform_waic_n_folds=2, random_state=0, top_m_after_tiny=5))
    order = np.arange(len(specs))
    agg = [1.0 + 0.001 * i for i in range(len(specs))]  # one noise band
    out = _tiny_rerank_waic._apply_waic_tiebreak(self, order, specs, agg, [s.name for s in specs], y_screen=np.linspace(2.0, 3.0, n), per_base_cache=cache)
    return [int(i) for i in out]


def test_waic_reorders_only_bands_whose_members_share_the_y_scale(monkeypatch):
    """Two additive transforms (T in y units) may be re-ordered by WAIC; a band mixing in a log transform keeps its RMSE order."""
    assert _waic_order(monkeypatch, ["linear_residual", "diff"]) == [1, 0]
    assert _waic_order(monkeypatch, ["linear_residual", "log_y"]) == [0, 1]


def test_an_unmeasured_spec_is_put_on_the_honest_scale_before_ranking():
    """In-group CV 9 for an unmeasured spec vs honest 12: with raw honest/CV = 13.6/9 the unmeasured spec lands at 13.6."""
    specs = [_spec("honest", "linear_residual"), _spec("unmeasured", "diff")]
    scores = [12.0, 9.0]
    _put_unmeasured_on_the_honest_scale(specs, scores, {"honest": 12.0}, 13.6, 9.0)
    assert math.isclose(scores[1], 13.6) and scores[1] > scores[0]
    scores = [12.0, 9.0]
    _put_unmeasured_on_the_honest_scale(specs, scores, {"honest": 12.0}, 13.6, float("nan"))
    assert scores[1] > scores[0], "without a ratio an unmeasured spec must rank after every measured one"


def test_the_budget_ranks_rmse_gains_before_mi_gains():
    """An MI gain in nats (0.9) must not outrank a relative RMSE gain (0.1); within a tier the unit's own order holds."""
    pending = [
        {"name": "mi_big", "gain": 0.9, "rmse_gain": False},
        {"name": "rmse_small", "gain": 0.1, "rmse_gain": True},
        {"name": "rmse_big", "gain": 0.3, "rmse_gain": True},
        {"name": "mi_nan", "gain": float("nan"), "rmse_gain": False},
    ]
    assert [p["name"] for p in rank_pending_composites(pending)] == ["rmse_big", "rmse_small", "mi_big", "mi_nan"]


def test_a_multi_base_upgrade_does_not_export_its_seeds_mi():
    """The upgraded spec's MI fields are NaN with the seed named, not the seed's numbers under a different transform."""
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    rng = np.random.default_rng(4)
    n = 3000
    b1, b2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({"b1": b1, "b2": b2, "x": rng.normal(size=n), "y": b1 + b2 + 0.3 * rng.normal(size=n)})
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, base_candidates=["b1", "b2"], transforms=["linear_residual"], eps_mi_gain=-1.0,
        multi_base_enabled=True, auto_chain_discovery_enabled=False, interaction_base_discovery_enabled=False,
        auto_base_null_perms=0, tiny_model_n_estimators=10, require_beats_raw_baseline=False,
    )
    disc = CompositeTargetDiscovery(cfg).fit(df, "y", ["b1", "b2", "x"], np.arange(n))
    multi = [s for s in disc.export_specs() if s["transform_name"] == "linear_residual_multi"]
    assert multi, "the fixture no longer produces a multi-base upgrade"
    for s in multi:
        assert math.isnan(s["mi_gain"]) and s["stats_measured_for"], s


def test_the_dummy_floor_is_measured_on_the_oof_rows():
    """The floor comes from the in-pool lag column, else the strongest constant on the OOF targets, not the val split."""
    md = {"dummy_baselines": {"regression": {"y": {"strongest": "median", "primary_metric": "val_RMSE", "data": {"median": {"val_RMSE": 99.0}}}}}}
    y_oof = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert math.isclose(same_split_dummy_rmse(md, "regression", "y", ["a", "lag_predict"], [5.0, 7.5], y_oof), 7.5)
    expected = float(np.sqrt(np.mean((y_oof - np.median(y_oof)) ** 2)))
    assert math.isclose(same_split_dummy_rmse(md, "regression", "y", ["a"], [5.0], y_oof), expected)
    md_lag = {"dummy_baselines": {"regression": {"y": {"strongest": "lag_predict", "primary_metric": "val_RMSE", "data": {"lag_predict": {"val_RMSE": 9.0}}}}}}
    assert same_split_dummy_rmse(md_lag, "regression", "y", ["a"], [5.0], y_oof) == 9.0  # no same-split estimate: val value, warned


def test_mi_y_is_aggregated_over_the_same_columns_as_mi_t():
    """An exact duplicate of a high-MI feature must not inflate the MI(y, X) baseline once the dedup drops it."""
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    rng = np.random.default_rng(0)
    n = 2000
    base, x1, x2 = rng.uniform(1.0, 10.0, n), rng.normal(size=n), rng.normal(size=n)
    y = base + 2.0 * x1 + 0.1 * x2 + rng.normal(0.0, 0.2, n)
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, base_candidates=["base"], transforms=["linear_residual"], eps_mi_gain=-10.0, screening="mi",
        mi_estimator="bin", dedup_x_remaining_for_mi_baseline=True, multi_base_enabled=False, auto_chain_discovery_enabled=False,
        interaction_base_discovery_enabled=False, auto_base_null_perms=0, honest_holdout_frac=0.0, honest_rmse_gate_enabled=False,
    )
    plain = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "y": y})
    dup = plain.assign(x1_dup=x1)
    mi_plain = CompositeTargetDiscovery(cfg).fit(plain, "y", ["base", "x1", "x2"], np.arange(n)).export_specs()[0]["mi_y"]
    mi_dup = CompositeTargetDiscovery(cfg).fit(dup, "y", ["base", "x1", "x2", "x1_dup"], np.arange(n)).export_specs()[0]["mi_y"]
    assert math.isclose(mi_dup, mi_plain, rel_tol=1e-9), f"the duplicate column inflated mi_y: {mi_plain:.5f} -> {mi_dup:.5f}"


def test_the_targets_table_reports_composite_rows_on_the_y_scale():
    """A composite row comes from its y-scale metrics; without them its T-scale metrics are labelled as such."""
    entry = types.SimpleNamespace(model_name="lgb", metrics={"test": {"RMSE": 0.33, "R2": 0.77}})
    models = {"regression": {"y": [types.SimpleNamespace(model_name="lgb", metrics={"test": {"RMSE": 0.3}})], "y-linres-b": [entry], "y-diff-b": [entry]}}
    md = {
        "composite_target_specs": {"regression": {"y": [{"name": "y-linres-b"}, {"name": "y-diff-b"}]}},
        "composite_target_y_scale_metrics": {"regression": {"y-linres-b": [{"model_name": "lgb", "metrics": {"test": {"RMSE": 0.35, "R2": 0.98, "n_rows_finite": 100}}}]}},
    }
    frame = targets_performance_frame(models, md).set_index("target_name")
    assert frame.loc["y-linres-b", "scale"] == "y" and math.isclose(frame.loc["y-linres-b", "R2"], 0.98)
    assert frame.loc["y-diff-b", "scale"] == "T"
    assert frame.loc["y", "scale"] == "y"
    assert "n_rows_finite" not in frame.columns
