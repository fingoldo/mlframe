"""The bed that isolates tail dependence must actually isolate it, and the scorer must read it correctly.

The suite already contains a bed pair that claimed to test tail dependence and did not: the t-copula bed
and its Gaussian control separate the roster identically, which means they demonstrate non-monotonicity.
This bed exists to fix that, so the properties that make the isolation real are pinned here rather than
assumed from the construction -- the earlier bed's construction looked convincing too.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest
from scipy import stats

from mlframe.data.datasets import scenarios as scenario_registry
from mlframe.data.datasets._links import tail_gate_term
from mlframe.data.datasets.generator import generate
from mlframe.data.datasets.scenarios._tails import ISOLATION_PAIRS, ISOLATION_QUANTILE, tail_isolation_spec
from mlframe.feature_selection._benchmarks.fs_hybrid._tail_gap import TAIL_ISOLATION_BED, column_groups, tail_gap_rows, tail_gap_table

TAIL_COLUMNS = [f"c{i}{side}" for i in range(ISOLATION_PAIRS) for side in ("a", "b")]
CONTROL_COLUMNS = [f"g{i}{side}" for i in range(ISOLATION_PAIRS) for side in ("a", "b")]


@pytest.fixture(scope="module")
def generated() -> Any:
    """Generate the bed once: it is sixty thousand rows and every assertion below reads the same draw."""
    return generate(tail_isolation_spec(seed=0))


def test_the_two_groups_have_matching_rank_correlation(generated: Any) -> None:
    """If the groups differed in rank correlation, a rank statistic could separate them and the bed would prove nothing."""
    frame = generated.frame
    tail_rho = [stats.spearmanr(frame[f"c{i}a"], frame[f"c{i}b"]).statistic for i in range(ISOLATION_PAIRS)]
    control_rho = [stats.spearmanr(frame[f"g{i}a"], frame[f"g{i}b"]).statistic for i in range(ISOLATION_PAIRS)]

    assert float(np.mean(tail_rho)) == pytest.approx(float(np.mean(control_rho)), abs=0.02)


def test_the_two_groups_have_matching_linear_correlation(generated: Any) -> None:
    """Matching Spearman would not help if Pearson differed: an F-test would then find the groups apart."""
    frame = generated.frame
    tail_r = [float(np.corrcoef(frame[f"c{i}a"], frame[f"c{i}b"])[0, 1]) for i in range(ISOLATION_PAIRS)]
    control_r = [float(np.corrcoef(frame[f"g{i}a"], frame[f"g{i}b"])[0, 1]) for i in range(ISOLATION_PAIRS)]

    assert float(np.mean(tail_r)) == pytest.approx(float(np.mean(control_r)), abs=0.02)


def test_the_two_groups_have_matching_marginals(generated: Any) -> None:
    """A marginal difference would make the groups separable without any joint reasoning at all."""
    frame = generated.frame
    tail_sd = float(np.mean([frame[column].std() for column in TAIL_COLUMNS]))
    control_sd = float(np.mean([frame[column].std() for column in CONTROL_COLUMNS]))

    assert tail_sd == pytest.approx(control_sd, rel=0.05)


def test_the_tail_dependent_pairs_fire_far_more_often(generated: Any) -> None:
    """The isolation IS this ratio. Everything else about the two groups is matched by construction."""
    frame = generated.frame
    quantile = 1.0 - ISOLATION_QUANTILE
    tail_rate = float(np.mean([tail_gate_term([frame[f"c{i}a"].to_numpy(), frame[f"c{i}b"].to_numpy()], quantile, "lower").mean() for i in range(ISOLATION_PAIRS)]))
    control_rate = float(np.mean([tail_gate_term([frame[f"g{i}a"].to_numpy(), frame[f"g{i}b"].to_numpy()], quantile, "lower").mean() for i in range(ISOLATION_PAIRS)]))

    assert tail_rate > 2.0 * control_rate, f"tail pairs fire {tail_rate:.4f} against control {control_rate:.4f}: the contrast is too small to measure"


def test_the_bed_carries_enough_signal_to_measure(generated: Any) -> None:
    """A single pair produced an achievable AUC of 0.516, which is a noise bed wearing a bed's name."""
    assert generated.calibration["bayes_auc"] > 0.54


def test_the_control_columns_are_marginally_indistinguishable_from_probes(generated: Any) -> None:
    """The control group must look like noise to a univariate filter; only the tail group may stand out."""
    frame, labels = generated.frame, generated.target.to_numpy()
    probes = [column for column in frame.columns if column.startswith("n")]
    control_corr = float(np.mean([abs(np.corrcoef(frame[column], labels)[0, 1]) for column in CONTROL_COLUMNS]))
    probe_corr = float(np.mean([abs(np.corrcoef(frame[column], labels)[0, 1]) for column in probes]))
    tail_corr = float(np.mean([abs(np.corrcoef(frame[column], labels)[0, 1]) for column in TAIL_COLUMNS]))

    assert control_corr < 3.0 * probe_corr
    assert tail_corr > 2.0 * control_corr


def test_one_sided_gate_fires_only_on_the_named_tail() -> None:
    """A symmetric gate would add the upper tail, where Clayton is asymptotically independent, and halve the contrast."""
    values = np.linspace(-3.0, 3.0, 1001)
    lower = tail_gate_term([values, values], 0.95, "lower")
    upper = tail_gate_term([values, values], 0.95, "upper")
    both = tail_gate_term([values, values], 0.95, "both")

    assert lower[:10].sum() > 0 and lower[-10:].sum() == 0
    assert upper[-10:].sum() > 0 and upper[:10].sum() == 0
    np.testing.assert_allclose(both, np.clip(lower + upper, 0.0, 1.0))


def test_unknown_gate_direction_is_refused() -> None:
    """A typo'd direction must not fall back to the symmetric default, which is a different experiment."""
    with pytest.raises(ValueError, match="tail gate direction"):
        tail_gate_term([np.zeros(10), np.zeros(10)], 0.9, "sideways")


def test_every_column_of_both_groups_is_declared_a_cause() -> None:
    """Set recovery is deliberately not the measurement here, and the truth must say so by naming them all."""
    spec = tail_isolation_spec(seed=0)
    sources = {edge.source for edge in spec.edges}

    assert sources == set(TAIL_COLUMNS) | set(CONTROL_COLUMNS)


def test_column_groups_splits_by_the_declared_naming() -> None:
    """The scorer reads the group from the name, so a change in naming must break loudly here first."""
    tail, control = column_groups(TAIL_COLUMNS + CONTROL_COLUMNS + ["n000", "n001"])

    assert tail == set(TAIL_COLUMNS)
    assert control == set(CONTROL_COLUMNS)


def _cell(arm: str, seed: int, columns: List[str]) -> Dict[str, Any]:
    """Build one stored cell whose selection is ``columns``."""
    return {"scenario": TAIL_ISOLATION_BED, "arm": arm, "dataset_seed": seed, "status": "ok", "selected": {"1k": {"status": "ok", "columns": columns}}}


def test_rank_gap_is_zero_for_an_arm_that_splits_evenly() -> None:
    """Zero is the reading a rank statistic must produce, so it must not come out as anything else."""
    records = [_cell("even", seed, ["c0a", "c0b", "g0a", "g0b"]) for seed in range(4)]

    rows = tail_gap_rows(records)

    assert len(rows) == 1
    assert rows[0].gap == pytest.approx(0.0)
    assert rows[0].resolved is False


def test_rank_gap_is_positive_for_an_arm_that_prefers_the_tail_group() -> None:
    """An arm resolving the tail must score above its own seed-to-seed spread to be called resolved."""
    records = [_cell("resolver", seed, ["c0a", "c0b", "c1a", "g0a"]) for seed in range(4)]

    rows = tail_gap_rows(records)

    assert rows[0].gap == pytest.approx(0.5)
    assert rows[0].resolved is True


def test_rank_gap_ignores_cells_whose_selection_was_not_stored() -> None:
    """`all_features` and `omitted_too_large` say how many, never which, so they cannot contribute a gap."""
    records = [
        {"scenario": TAIL_ISOLATION_BED, "arm": "wide", "dataset_seed": 0, "status": "ok", "selected": {"1k": {"status": "omitted_too_large", "n": 900}}},
        _cell("narrow", 0, ["c0a", "g0a"]),
    ]

    arms = {row.arm for row in tail_gap_rows(records)}

    assert arms == {"narrow"}


def test_rank_gap_table_says_what_a_zero_means() -> None:
    """A table of numbers whose reading is left to the reader is how the earlier tail claim survived so long."""
    lines = tail_gap_table([_cell("even", seed, ["c0a", "g0a"]) for seed in range(3)])

    assert any("gap of zero" in line for line in lines)
    assert any("even" in line for line in lines)


def test_rank_gap_table_reports_an_absence_rather_than_an_empty_table() -> None:
    """No stored selections is a different statement from every arm scoring zero."""
    lines = tail_gap_table([])

    assert any("cannot be computed" in line for line in lines)


def test_the_bed_is_registered_with_a_prediction() -> None:
    """An unregistered bed cannot run, and one predicting nothing cannot be scored against its forecast."""
    assert TAIL_ISOLATION_BED in scenario_registry.names()
    assert scenario_registry.get(TAIL_ISOLATION_BED).expected_to_break
