"""The observation layer: holes, imbalance, shift, and the regional gate that could not be declared.

These beds leave the feature-target relationship alone and change what a method can see of it. That is
only true if the generator applies them in the right order, so the ordering is what these tests pin:
masking happens after the link, imbalance is reached by intercept shift rather than by dropping rows, and
the two shift beds differ in exactly one declaration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mlframe.data.datasets import scenarios as scenario_registry
from mlframe.data.datasets._columns import quantize
from mlframe.data.datasets._missing import missing_mask
from mlframe.data.datasets.generator import generate
from mlframe.data.datasets.scenarios._marginals import quantized_spec, zero_inflated_spec
from mlframe.data.datasets.scenarios._mixed_types import CARDINALITIES, id_trap_spec
from mlframe.data.datasets.scenarios._observation import RARE_MIN_ROWS, concept_shift_spec, covariate_shift_spec, missingness_trio_spec, rare_class_spec
from mlframe.data.datasets.spec import GateSpec, MissingnessSpec


def test_gate_accepts_a_fraction_declared_region() -> None:
    """Regression: the validator demanded an explicit interval, which made the fraction form unreachable.

    That was not a cosmetic gap. The generator's heteroscedastic path triggers ONLY on a region declared
    as a fraction, so a validator rejecting that form left the entire heteroscedasticity feature dead --
    present in the code, reachable by nothing, and silently absent from every bed that meant to use it.
    """
    gate = GateSpec(column="s0", fraction=0.5)

    assert gate.fraction == pytest.approx(0.5)
    assert gate.low is None and gate.high is None


def test_gate_refuses_to_mix_the_two_ways_of_naming_a_region() -> None:
    """Accepting both would leave it undefined which one applies."""
    with pytest.raises(ValueError, match="alternative ways to name a region"):
        GateSpec(column="s0", fraction=0.5, low=0.0)


def test_gate_still_refuses_a_region_declared_by_nothing() -> None:
    """The original check has to survive the fix, or a typo'd gate silently covers everything."""
    with pytest.raises(ValueError, match="needs either"):
        GateSpec(column="s0")


def test_the_regional_bed_actually_has_a_region() -> None:
    """Measured, not assumed: the effect must be present inside the region and absent outside it."""
    generated = generate(concept_shift_spec(seed=0))
    frame, labels = generated.frame, generated.target.to_numpy()
    inside = frame["s0"] >= frame["s0"].quantile(0.5)

    within = abs(float(np.corrcoef(frame["s1"][inside], labels[inside])[0, 1]))
    without = abs(float(np.corrcoef(frame["s1"][~inside], labels[~inside])[0, 1]))

    assert within > 0.15, f"the effect is not present inside the region (corr {within:.3f})"
    assert without < 0.05, f"the effect leaks outside the region (corr {without:.3f}), so it is merely weaker rather than absent"


def test_the_regional_bed_records_its_variance_drivers() -> None:
    """A column that only moves the variance is invisible to every mean-based importance, so truth must name it."""
    calibration = generate(concept_shift_spec(seed=0)).calibration

    assert calibration.get("variance_drivers") == {"s0": 1.0}


def test_the_two_shift_beds_share_everything_except_one_declaration() -> None:
    """The pair is the experiment: anything that reports the same on both is calling two problems one."""
    covariate = covariate_shift_spec(seed=0)
    concept = concept_shift_spec(seed=0)

    assert {edge.source for edge in covariate.edges} == {edge.source for edge in concept.edges}
    assert covariate.targets[0].link.coefficients == concept.targets[0].link.coefficients
    assert covariate.targets[0].link.region is None
    assert concept.targets[0].link.region is not None


def test_covariate_shift_moves_the_inputs_and_not_the_link() -> None:
    """`P(x)` must actually differ from the control's, or the bed is the control under another name."""
    frame = generate(covariate_shift_spec(seed=0)).frame

    assert float(frame["s0"].mean()) > 0.3, "the covariate bed's inputs are centred like the control's"
    assert float(frame["s0"].std()) > 1.2, "the covariate bed's inputs are as tight as the control's"


def test_missingness_leaves_the_recorded_probability_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same seed with and without masking must produce the same `true_prob`: holes change what is SEEN.

    If masking moved the recorded law, the ceiling would be the ceiling of a half-observed world and every
    regret measured against it would be wrong by an amount nobody computed.
    """
    with_holes = generate(missingness_trio_spec(seed=0))
    without = generate(missingness_trio_spec(seed=0).model_copy(update={"missingness": ()}))

    np.testing.assert_allclose(np.asarray(with_holes.truth.true_prob), np.asarray(without.truth.true_prob))


def test_missingness_is_recorded_as_a_caveat_naming_its_mechanism() -> None:
    """A ceiling that no longer applies to the emitted frame must say so, per mechanism."""
    caveats = " ".join(generate(missingness_trio_spec(seed=0)).truth.caveats)

    assert "MCAR" in caveats and "MAR" in caveats and "MNAR" in caveats
    assert "COMPLETE-data ceiling" in caveats


def test_every_mechanism_hides_the_same_share_of_rows() -> None:
    """Otherwise a comparison between mechanisms would be a comparison of how much, not of which."""
    frame = generate(missingness_trio_spec(seed=0, rate=0.3)).frame
    shares = [float(frame[column].isna().mean()) for column in ("s0", "s1", "s2")]

    for share in shares:
        assert share == pytest.approx(0.3, abs=0.03), f"realised missingness shares are {shares}, so the mechanisms differ in amount as well as in kind"


def test_mnar_hides_the_column_s_own_largest_values() -> None:
    """That is what makes its bias unidentifiable, and it is the only thing distinguishing it from MAR."""
    values = np.linspace(-3.0, 3.0, 1000)
    rng = np.random.default_rng(0)

    mask = missing_mask(MissingnessSpec(column="x", mechanism="mnar", rate=0.2), {"x": values}, 0.2, rng)

    assert values[mask].min() > values[~mask].max()


def test_mar_hides_rows_chosen_by_another_column() -> None:
    """The driver being observed is exactly what makes MAR's bias recoverable."""
    masked = np.linspace(-3.0, 3.0, 1000)
    driver = -masked
    rng = np.random.default_rng(0)

    mask = missing_mask(MissingnessSpec(column="x", mechanism="mar", rate=0.2, driver="d"), {"x": masked, "d": driver}, 0.2, rng)

    assert masked[mask].max() < masked[~mask].min(), "MAR did not follow its driver"


def test_mar_driven_by_itself_is_refused_as_mislabelled() -> None:
    """The distinction decides whether the bias is recoverable at all, so the label may not be wrong."""
    with pytest.raises(ValueError, match="which is 'mnar'"):
        MissingnessSpec(column="x", mechanism="mar", driver="x")


def test_a_masked_integer_column_is_widened_rather_than_given_a_sentinel() -> None:
    """A -999 in an integer column is a value some method treats as a very small number."""
    spec = missingness_trio_spec(seed=0).model_copy(update={"missingness": (MissingnessSpec(column="s0", mechanism="mcar", rate=0.2),)})
    frame = generate(spec.model_copy(update={"features": tuple(f.model_copy(update={"dtype": "int"}) if f.name == "s0" else f for f in spec.features)})).frame

    assert str(frame["s0"].dtype) == "Int64"
    assert frame["s0"].isna().any()
    assert not (frame["s0"].dropna() < -900).any()


def test_rare_class_reaches_its_prevalence_without_shrinking_the_bed() -> None:
    """Dropping majority rows would confound imbalance with sample size in every later comparison."""
    generated = generate(rare_class_spec(seed=0))

    assert generated.frame.shape[0] >= RARE_MIN_ROWS
    assert float(generated.target.mean()) == pytest.approx(0.01, abs=0.004)


def test_rare_class_has_enough_minority_rows_to_measure_anything() -> None:
    """An undersized rare-class fixture reads as flakiness rather than as the undersizing it is."""
    positives = int(generate(rare_class_spec(seed=0)).target.sum())

    assert positives >= 50, f"only {positives} positive rows, which is below where minority metrics stabilise"


def test_quantize_leaves_exactly_the_requested_number_of_levels() -> None:
    """A quantisation that delivered a different level count would not be the corruption the bed declares."""
    values = np.random.default_rng(0).normal(size=5000)

    assert len(np.unique(quantize(values, 8))) == 8


def test_quantize_preserves_the_range_it_was_given() -> None:
    """Shrinking the range would make the bed an outlier-removal bed as well as a quantisation one."""
    values = np.random.default_rng(0).normal(size=5000)

    quantised = quantize(values, 6)

    assert float(quantised.min()) == pytest.approx(float(values.min()))
    assert float(quantised.max()) == pytest.approx(float(values.max()))


def test_quantized_bed_gives_an_equal_mass_estimator_fewer_values_than_it_wants_bins() -> None:
    """That is the attack: ten requested bins cannot be drawn from six distinct values."""
    frame = generate(quantized_spec(seed=0, levels=6)).frame

    assert frame["s0"].nunique() <= 6


def test_zero_inflated_bed_carries_a_real_point_mass() -> None:
    """Equal-mass binning cannot split a point mass, which is the whole reason the bed exists."""
    frame = generate(zero_inflated_spec(seed=0, inflation=0.4)).frame
    most_common_share = float(frame["s0"].value_counts(normalize=True).iloc[0])

    assert most_common_share > 0.3, f"the largest point mass holds only {most_common_share:.2f} of the rows"


def test_the_id_trap_column_is_unique_and_declared_irrelevant() -> None:
    """A method selecting it has recorded the most expensive kind of false positive, so truth must not excuse it."""
    spec = id_trap_spec(seed=0)
    frame = generate(spec).frame

    assert "row_id" not in {edge.source for edge in spec.edges}
    assert frame["row_id"].nunique() > 0.9 * frame.shape[0]


def test_graded_cardinality_carries_every_level_count_among_probes_too() -> None:
    """Otherwise "prefers many levels" and "prefers informative columns" could not be told apart."""
    frame = generate(scenario_registry.get("graded_cardinality").build(seed=0)).frame

    for levels in CARDINALITIES:
        assert f"k{levels}" in frame.columns
        assert any(column.startswith(f"p{levels}_") for column in frame.columns), f"cardinality {levels} appears among the signals but not among the probes"


@pytest.mark.parametrize(
    "name",
    ["heavy_tail_t4", "outliers_020permille", "zero_inflated_40pct", "quantized_6levels", "graded_cardinality", "id_trap", "zipf_levels", "missingness_trio_30pct", "rare_class_010permille", "shift_covariate", "shift_concept"],
)
def test_every_new_bed_generates_and_reaches_its_declared_ceiling(name: str) -> None:
    """A bed that cannot reach the difficulty it declares is not the bed anybody registered."""
    generated: Any = generate(scenario_registry.get(name).build(seed=0))
    requested = generated.calibration["requested"]

    assert generated.frame.shape[0] > 0
    if requested is not None:
        assert generated.calibration["achieved_auc"] == pytest.approx(requested, abs=0.01)
