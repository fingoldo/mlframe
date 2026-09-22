"""The variance decomposition must recover components it was given, and must not invent one it was not.

Every assertion here is on a number the module produced from data whose true components are known by
construction, rather than on the shape of its output. The interesting direction is the second one: a
method-of-moments estimator that clamps at zero will report a small positive bed effect on data that has
none unless the level above it is de-biased, and a decomposition that always finds a bed effect is the
same as one that never ran.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid._variance_decomposition import (
    decompose_paired_variance,
    format_components,
    variance_table,
)


def _cells(
    *,
    scenario_sd: float,
    seed_sd: float,
    cv_sd: float,
    n_scenarios: int = 12,
    n_seeds: int = 20,
    n_cv: int = 4,
    seed: int = 0,
    row_variance: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build paired cells whose true variance components are exactly the three standard deviations given.

    The null arm carries a large per-cell base level that the treated arm shares. It is noise an unpaired
    analysis would drown in, and differencing must remove it entirely.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for s in range(n_scenarios):
        bed_effect = rng.normal(0.05, scenario_sd)
        for dataset_seed in range(n_seeds):
            draw_effect = bed_effect + rng.normal(0.0, seed_sd)
            for cv_seed in range(n_cv):
                delta = draw_effect + rng.normal(0.0, cv_sd)
                base = rng.normal(0.8, 0.1)
                shared = {"scenario": f"bed{s}", "dataset_seed": dataset_seed, "cv_seed": cv_seed}
                treated: Dict[str, Any] = {"arm": "treated", "value": base + delta, **shared}
                null: Dict[str, Any] = {"arm": "null", "value": base, **shared}
                if row_variance is not None:
                    treated["row_variance"] = row_variance
                    null["row_variance"] = row_variance
                rows += [treated, null]
    return rows


def test_components_recover_the_standard_deviations_they_were_built_from() -> None:
    """Each recovered component is within 15% of the standard deviation the data was generated with."""
    rows = _cells(scenario_sd=0.040, seed_sd=0.020, cv_sd=0.010)
    components = decompose_paired_variance(rows, arm="treated", null_arm="null")

    assert float(np.sqrt(components.arm_by_scenario)) == pytest.approx(0.040, rel=0.15)
    assert float(np.sqrt(components.dataset_seed)) == pytest.approx(0.020, rel=0.15)
    assert float(np.sqrt(components.cv_seed)) == pytest.approx(0.010, rel=0.15)
    assert components.n_scenarios == 12
    assert components.n_cells == 12 * 20 * 4


def test_the_bed_share_collapses_when_no_bed_effect_was_generated() -> None:
    """With every bed sharing one effect, the bed component must be a small fraction, not merely non-negative.

    This is the de-biasing check. Without subtracting the seed level from the scenario level, the spread of
    the per-bed means -- which is pure seed noise here -- is published as a real bed effect.
    """
    rows = _cells(scenario_sd=0.0, seed_sd=0.020, cv_sd=0.010)
    components = decompose_paired_variance(rows, arm="treated", null_arm="null")

    shares = components.shares()
    assert shares["arm_by_scenario"] < 0.10
    assert shares["dataset_seed"] > 0.50
    assert components.method_matters_more_than_the_draw() is False


def test_a_large_bed_effect_beats_the_data_draw() -> None:
    """The bed component dominates when the generated bed spread is twice the draw's."""
    rows = _cells(scenario_sd=0.040, seed_sd=0.020, cv_sd=0.010)
    components = decompose_paired_variance(rows, arm="treated", null_arm="null")

    assert components.method_matters_more_than_the_draw() is True
    assert components.shares()["arm_by_scenario"] > 0.55


def test_a_shared_base_level_cancels_and_does_not_reach_any_component() -> None:
    """Adding a shared per-cell bump of sd 1.0 to both arms leaves every component where it was.

    That bump is fifty times the largest real component here, so anything short of exact cancellation
    would swamp the decomposition rather than perturb it.
    """
    quiet = decompose_paired_variance(_cells(scenario_sd=0.030, seed_sd=0.015, cv_sd=0.008), arm="treated", null_arm="null")

    rng = np.random.default_rng(0)
    rows: List[Dict[str, Any]] = []
    for row in _cells(scenario_sd=0.030, seed_sd=0.015, cv_sd=0.008):
        rows.append(dict(row))
    for index in range(0, len(rows), 2):
        bump = float(rng.normal(0.0, 1.0))
        rows[index]["value"] += bump
        rows[index + 1]["value"] += bump
    loud = decompose_paired_variance(rows, arm="treated", null_arm="null")

    assert loud.arm_by_scenario == pytest.approx(quiet.arm_by_scenario, rel=1e-9)
    assert loud.dataset_seed == pytest.approx(quiet.dataset_seed, rel=1e-9)
    assert loud.cv_seed == pytest.approx(quiet.cv_seed, rel=1e-9)


def test_an_unmeasured_row_component_is_none_and_a_measured_one_is_a_number() -> None:
    """An absent row variance reads as `None`, never as zero: the two are different claims."""
    without = decompose_paired_variance(_cells(scenario_sd=0.02, seed_sd=0.02, cv_sd=0.01), arm="treated", null_arm="null")
    assert without.row is None
    assert "row" not in without.shares()
    assert "rows unmeasured" in format_components(without, "treated", "null")

    with_rows = decompose_paired_variance(_cells(scenario_sd=0.02, seed_sd=0.02, cv_sd=0.01, row_variance=1e-5), arm="treated", null_arm="null")
    assert with_rows.row == pytest.approx(1e-5)
    assert with_rows.shares()["row"] > 0.0


def test_an_unpaired_arm_raises_rather_than_reporting_an_unpaired_spread() -> None:
    """With no cell carrying both arms there is no difference to decompose, and saying so beats a number."""
    rows = [r for r in _cells(scenario_sd=0.02, seed_sd=0.02, cv_sd=0.01) if r["arm"] == "treated"]
    with pytest.raises(ValueError, match="no cell carries both"):
        decompose_paired_variance(rows, arm="treated", null_arm="null")


def test_the_report_block_names_the_arm_and_its_bed_share() -> None:
    """The rendered block carries a row per arm with the bed share the components hold."""
    records = []
    for row in _cells(scenario_sd=0.040, seed_sd=0.020, cv_sd=0.010, n_scenarios=6, n_seeds=8):
        records.append(
            {
                "status": "ok",
                "arm": "all-features" if row["arm"] == "null" else row["arm"],
                "scenario": row["scenario"],
                "dataset_seed": row["dataset_seed"],
                "cv_seed": row["cv_seed"],
                "scores": {"k=10": {"skill": {"lightgbm": row["value"]}}},
            }
        )
    lines = variance_table(records, model="lightgbm", k_label="k=10")

    assert any("`treated`" in line for line in lines)
    assert any("WHERE THE ADVANTAGE MOVES" in line for line in lines)
    assert not any("`all-features`" in line for line in lines)
