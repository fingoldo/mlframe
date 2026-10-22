"""The four pillar beds, and the two spec extensions they needed.

Each assertion here is on a property of the generated data or of the spec's arithmetic, not on the registry's text. The
marginal facts are the ones the beds' predictions rest on, and the thresholds sit several standard errors away from the
values measured at seed 0, so a regeneration that changes the draw does not change the verdict.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from mlframe.data.datasets import scenarios
from mlframe.data.datasets._links import basis_term_value
from mlframe.data.datasets._columns import draw_family
from mlframe.data.datasets.generator import generate
from mlframe.data.datasets.spec import BasisTerm

PILLARS = ("pillar_f2_compound", "pillar_noise_marginals", "pillar_xor3", "pillar_conjunction")


@pytest.fixture(scope="module")
def pillar_datasets() -> dict:
    """Generate each pillar bed once at its own registered size."""
    return {name: generate(scenarios.get(name).build(seed=0)) for name in PILLARS}


def _univariate_auc(dataset, column: str) -> float:
    """AUC of one column against the realised label, as a marginal score would see it."""
    values = np.asarray(dataset.frame[column], dtype=float)
    return float(roc_auc_score(np.asarray(dataset.target), values))


def test_every_pillar_is_registered_with_a_predicted_arm_and_a_purpose() -> None:
    """Each pillar is a bed the registry's discipline applies to: a purpose, and a non-empty forecast."""
    for name in PILLARS:
        scenario = scenarios.get(name)
        assert scenario.purpose.strip()
        assert scenario.expected_to_break, f"{name} forecasts nothing"


def test_the_f2_compound_answer_key_is_the_four_operands_and_not_the_noise_operand(pillar_datasets: dict) -> None:
    """The published formula uses a, b, c and d; e is a pure noise column and must not be in the key."""
    members = set(pillar_datasets["pillar_f2_compound"].truth.primary_target_set().members)
    assert members == {"a", "b", "c", "d"}


def test_the_f2_operand_c_is_marginally_invisible_and_d_is_not(pillar_datasets: dict) -> None:
    """Measured at seed 0: c's univariate AUC is 0.50, d's is 0.46. Both are in the key; only c is hidden."""
    dataset = pillar_datasets["pillar_f2_compound"]
    assert abs(_univariate_auc(dataset, "c") - 0.5) < 0.04, "c should carry no marginal signal"
    assert abs(_univariate_auc(dataset, "d") - 0.5) > 0.03, "d should be weakly visible, the claim the prediction text makes"
    assert abs(_univariate_auc(dataset, "a") - 0.5) > 0.2, "a is the marginally strong operand"


def test_the_f2_bed_reaches_its_declared_ceiling(pillar_datasets: dict) -> None:
    """The bed is calibrated like the other reference beds: the declared AUC is the one the link delivers."""
    assert pillar_datasets["pillar_f2_compound"].calibration["bayes_auc"] == pytest.approx(0.85, abs=0.02)


def test_the_pure_noise_bed_is_null_with_nine_marginal_shapes_of_eight_columns(pillar_datasets: dict) -> None:
    """Nothing is in the key, the Bayes AUC is chance, and the nine shapes are all present with eight columns each."""
    dataset = pillar_datasets["pillar_noise_marginals"]
    assert dataset.truth.primary_target_set().members == ()
    assert dataset.calibration["bayes_auc"] == pytest.approx(0.5, abs=1e-9)
    columns = list(dataset.frame.columns)
    assert len(columns) == 72
    kinds = {c.rsplit("_", 1)[0] for c in columns}
    assert kinds == {"normal", "uniform", "student_t2", "lognormal", "exponential", "binary", "categorical", "near_constant", "id_like"}, kinds


def test_the_id_like_column_takes_a_distinct_value_on_every_row(pillar_datasets: dict) -> None:
    """A permutation of the row index: no two rows share a value, which a draw with replacement could not guarantee."""
    values = np.asarray(pillar_datasets["pillar_noise_marginals"].frame["id_like_0"], dtype=float)
    assert np.unique(values).size == values.size


def test_the_near_constant_column_is_zero_on_about_ninety_nine_per_cent_of_rows(pillar_datasets: dict) -> None:
    """The point mass is the shape equal-mass binning cannot split; it must actually be a point mass, not a mild tilt."""
    values = np.asarray(pillar_datasets["pillar_noise_marginals"].frame["near_constant_0"], dtype=float)
    assert (values == 0.0).mean() > 0.97


def test_the_three_way_parity_operands_have_no_marginal_signal_and_the_key_is_all_three(pillar_datasets: dict) -> None:
    """Parity of three is invisible one column at a time, which is the whole reason the pillar exists."""
    dataset = pillar_datasets["pillar_xor3"]
    assert set(dataset.truth.primary_target_set().members) == {"p0", "p1", "p2"}
    for column in ("p0", "p1", "p2"):
        assert abs(_univariate_auc(dataset, column) - 0.5) < 0.04, column


def test_the_conjunction_key_is_the_three_operands_and_a_is_marginally_strong(pillar_datasets: dict) -> None:
    """A carries most of the marginal signal; B and C are in the key but lift the marginal only slightly."""
    dataset = pillar_datasets["pillar_conjunction"]
    assert set(dataset.truth.primary_target_set().members) == {"A", "B", "C"}
    assert _univariate_auc(dataset, "A") > 0.7
    assert _univariate_auc(dataset, "B") < _univariate_auc(dataset, "A") - 0.2


def test_the_ratio_term_with_a_power_is_the_published_a_squared_over_b() -> None:
    """The numerator power turns the ratio into a**2 / b, and the default leaves every existing bed's values alone."""
    columns = {"a": np.array([2.0, 3.0]), "b": np.array([4.0, 2.0])}
    squared = basis_term_value(BasisTerm(kind="ratio", columns=("a", "b"), params={"power": 2.0}), columns)
    np.testing.assert_allclose(squared, [4.0 / 4.0, 9.0 / 2.0])
    default = basis_term_value(BasisTerm(kind="ratio", columns=("a", "b")), columns)
    np.testing.assert_allclose(default, [2.0 / 4.0, 3.0 / 2.0])


def test_the_log_abs_sin_term_is_the_published_compound_and_finite_at_zero() -> None:
    """log|c| sin(d) with the reference epsilon, so a zero c contributes a finite value rather than minus infinity."""
    columns = {"c": np.array([0.0, np.e]), "d": np.array([np.pi / 2, np.pi / 2])}
    values = basis_term_value(BasisTerm(kind="log_abs_sin", columns=("c", "d")), columns)
    assert np.all(np.isfinite(values))
    assert values[1] == pytest.approx(1.0, abs=1e-6)


def test_the_log_abs_sin_term_takes_exactly_two_columns() -> None:
    """A third operand would silently change the formula the name promises, so it is rejected at construction."""
    with pytest.raises(ValueError, match="exactly two columns"):
        BasisTerm(kind="log_abs_sin", columns=("c", "d", "e"))


def test_the_permutation_family_returns_every_index_exactly_once() -> None:
    """The id-like family is a permutation of 0..n-1, not a draw, so every value appears once."""
    values = draw_family("id", {}, 500, np.random.default_rng(0))
    assert sorted(values.tolist()) == list(range(500))
