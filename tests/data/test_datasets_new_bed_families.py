"""Eleven beds whose whole value is a property that is easy to claim and easy to get wrong.

Every assertion here is on a MEASURED property of the generated data rather than on the spec that asked
for it. That distinction is the point: three of these beds were wrong on their first construction and
looked completely right in the source. The M-bias collider ended up correlated 0.35 with the target
instead of 0.00 because a second latent silently overwrote the first's reflection; the proxy bed had the
proxy weaker than the column it was supposed to dominate; and a bed built on `standardize=False` handed
its answer key to anything that sorts by variance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mlframe.data.datasets import scenarios as scenario_registry
from mlframe.data.datasets.generator import generate
from mlframe.data.datasets.scenarios._causal_extra import confounder_spec, instrument_spec, m_bias_spec, proxy_attenuation_spec
from mlframe.data.datasets.scenarios._corrupted import coarsened_label_spec, feature_dependent_flip_spec, uniform_flip_spec
from mlframe.data.datasets.scenarios._economics import CHEAP_COST, CORRELATION_LEVELS, EXPENSIVE_COST, expensive_signal_spec, graded_redundancy_spec
from mlframe.data.datasets.scenarios._structure import grouped_rows_spec, simpson_reversal_spec

NEW_BEDS = (
    "label_flip_uniform_15pct",
    "label_flip_regional_35pct",
    "label_coarsened",
    "m_bias",
    "confounder_backdoor",
    "instrument_screened_off",
    "proxy_attenuation",
    "expensive_signal",
    "redundancy_graded",
    "simpson_sign_reversal",
    "grouped_rows_40",
)


def _corr(a: Any, b: Any) -> float:
    """Pearson correlation as a plain float."""
    return float(np.corrcoef(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))[0, 1])


def _partial_corr(a: np.ndarray, b: np.ndarray, given: np.ndarray) -> float:
    """Correlation of ``a`` and ``b`` after regressing both on ``given``."""
    design = np.column_stack([np.ones_like(given, dtype=np.float64), np.asarray(given, dtype=np.float64)])
    residual_a = a - design @ np.linalg.lstsq(design, a, rcond=None)[0]
    residual_b = b - design @ np.linalg.lstsq(design, b, rcond=None)[0]
    return _corr(residual_a, residual_b)


def test_a_label_flip_makes_the_declared_ceiling_unreachable() -> None:
    """The gap between the declared ceiling and the achieved one IS the information the flip destroyed.

    A bed declaring 0.85 and then flipping fifteen per cent of its labels ships data whose real ceiling is
    lower, and no link scale recovers it. Any report quoting the declared number here is quoting a number
    the data cannot reach.
    """
    generated = generate(uniform_flip_spec(seed=0))

    requested = generated.calibration["requested"]
    achieved = generated.calibration["achieved_auc"]
    assert requested is not None and achieved is not None
    assert achieved < requested, f"a 15% label flip left the ceiling at {achieved:.3f}, at or above the declared {requested:.3f}"
    assert requested - achieved < 0.05, "the bed is so damaged that every method sits at chance, which measures nothing"


def test_the_clean_control_reaches_its_declared_ceiling() -> None:
    """Without this the test above would pass on a bed whose calibrator was simply broken."""
    generated = generate(uniform_flip_spec(seed=0, rate=0.0))

    assert generated.calibration["achieved_auc"] == pytest.approx(generated.calibration["requested"], abs=0.01)


def test_a_regional_flip_damages_only_its_region() -> None:
    """Concentrated damage is the whole difference from the uniform case; spread out, it is the uniform case."""
    generated = generate(feature_dependent_flip_spec(seed=0))
    frame, labels = generated.frame, generated.target.to_numpy().astype(np.float64)
    unreliable = frame["s0"].to_numpy() >= np.quantile(frame["s0"].to_numpy(), 0.7)

    inside = abs(_corr(frame["s1"].to_numpy()[unreliable], labels[unreliable]))
    outside = abs(_corr(frame["s1"].to_numpy()[~unreliable], labels[~unreliable]))
    assert outside > inside, f"the labels are no worse inside the unreliable region ({inside:.3f}) than outside it ({outside:.3f})"


def test_every_corrupted_bed_records_what_was_done_to_it() -> None:
    """A ceiling that no longer applies has to say so, or a reader quotes the declared number in good faith."""
    for builder in (uniform_flip_spec, feature_dependent_flip_spec, coarsened_label_spec):
        generated = generate(builder(seed=0))
        requested, achieved = generated.calibration["requested"], generated.calibration["achieved_auc"]
        assert requested is not None and achieved is not None, f"{builder.__name__} records no ceiling at all"


def test_m_bias_leaves_the_cause_and_the_target_marginally_independent() -> None:
    """This is the property, and it is the one the first construction of this bed did not have."""
    frame = generate(m_bias_spec(seed=0)).frame
    labels = generate(m_bias_spec(seed=0)).target.to_numpy().astype(np.float64)

    assert abs(_corr(frame["x_cause"].to_numpy(), labels)) < 0.06


def test_m_bias_conditioning_on_the_collider_manufactures_an_association() -> None:
    """The association does not exist in the graph; conditioning creates it. That is the whole bed."""
    generated = generate(m_bias_spec(seed=0))
    frame, labels = generated.frame, generated.target.to_numpy().astype(np.float64)

    marginal = abs(_corr(frame["x_cause"].to_numpy(), labels))
    conditioned = abs(_partial_corr(frame["x_cause"].to_numpy(), labels, frame["m_collider"].to_numpy()))
    assert conditioned > 0.10, f"conditioning on the collider produced only {conditioned:.3f}: the M has no second arm"
    assert conditioned > 5.0 * marginal, f"conditioning ({conditioned:.3f}) barely exceeds the marginal ({marginal:.3f})"


def test_the_m_bias_collider_is_connected_to_both_arms() -> None:
    """A collider with one parent is not a collider, and the first version of this bed had exactly that."""
    frame = generate(m_bias_spec(seed=0)).frame

    assert abs(_corr(frame["x_cause"].to_numpy(), frame["m_collider"].to_numpy())) > 0.3


def test_the_confounded_column_predicts_without_causing() -> None:
    """Right under prediction, wrong under causation: the bed needs both halves to be true at once."""
    generated = generate(confounder_spec(seed=0))
    frame, labels = generated.frame, generated.target.to_numpy().astype(np.float64)

    confounded = abs(_corr(frame["x_confounded"].to_numpy(), labels))
    causal = abs(_corr(frame["x_true_cause"].to_numpy(), labels))
    assert confounded > 0.1, "the confounded column predicts nothing, so no method would ever face the trade"
    assert causal > confounded, "the confounded column out-predicts the real cause, which is a different bed"
    assert "x_confounded" not in {edge.source for edge in confounder_spec(seed=0).edges}, "the truth claims the confounded column causes the target"


def test_the_instrument_is_screened_off_by_the_column_it_acts_through() -> None:
    """Structural redundancy: once the mediator is in, the instrument adds nothing."""
    generated = generate(instrument_spec(seed=0))
    frame, labels = generated.frame, generated.target.to_numpy().astype(np.float64)

    alone = abs(_corr(frame["instrument"].to_numpy(), labels))
    screened = abs(_partial_corr(frame["instrument"].to_numpy(), labels, frame["x_cause"].to_numpy()))
    assert alone > 0.2, "the instrument predicts nothing on its own, so there is nothing to screen off"
    assert screened < 0.5 * alone, f"the instrument keeps {screened:.3f} of its {alone:.3f} after conditioning: it is not screened off"


def test_the_clean_proxy_out_predicts_the_noisy_measurement() -> None:
    """The ordering IS the bed, and the first construction had it backwards."""
    generated = generate(proxy_attenuation_spec(seed=0))
    frame, labels = generated.frame, generated.target.to_numpy().astype(np.float64)

    assert abs(_corr(frame["proxy"].to_numpy(), labels)) > abs(_corr(frame["x_measured"].to_numpy(), labels))


def test_the_expensive_bed_prices_its_columns_differently() -> None:
    """A Pareto figure drawn over a bed where every column costs the same measures compute and nothing else."""
    costs = {feature.name: feature.cost for feature in expensive_signal_spec(seed=0).features}

    assert costs["lab0"] == EXPENSIVE_COST
    assert costs["free0"] == CHEAP_COST
    assert costs["n000"] == CHEAP_COST
    assert max(costs.values()) / min(costs.values()) >= 50.0, "the price range is too narrow for a cost-blind choice to be visibly wrong"


def test_the_expensive_columns_are_the_informative_ones() -> None:
    """Otherwise the trade is free and the bed poses no question."""
    generated = generate(expensive_signal_spec(seed=0))
    frame, labels = generated.frame, generated.target.to_numpy().astype(np.float64)

    assert abs(_corr(frame["lab0"].to_numpy(), labels)) > abs(_corr(frame["free0"].to_numpy(), labels))


@pytest.mark.parametrize("level", CORRELATION_LEVELS)
def test_each_redundancy_cluster_delivers_the_correlation_it_is_named_after(level: float) -> None:
    """A cluster named 0.95 that sits at 0.91 measures a different threshold than the one it claims."""
    frame = generate(graded_redundancy_spec(seed=0)).frame
    tag = f"c{round(level * 100):02d}"

    delivered = _corr(frame[f"{tag}_0"].to_numpy(), frame[f"{tag}_1"].to_numpy())
    assert delivered == pytest.approx(level, abs=0.04), f"cluster {tag} delivered {delivered:.3f} for a declared {level:.2f}"


def test_the_redundancy_levels_are_ordered_and_spread() -> None:
    """Four clusters at nearly the same correlation would be one cluster measured four times."""
    frame = generate(graded_redundancy_spec(seed=0)).frame
    delivered = [_corr(frame[f"c{round(level * 100):02d}_0"].to_numpy(), frame[f"c{round(level * 100):02d}_1"].to_numpy()) for level in CORRELATION_LEVELS]

    assert delivered == sorted(delivered)
    assert max(delivered) - min(delivered) > 0.4


def test_the_reversing_column_has_no_marginal_association() -> None:
    """Zero marginally is what makes every filter rank it with the probes."""
    generated = generate(simpson_reversal_spec(seed=0))
    frame, labels = generated.frame, generated.target.to_numpy().astype(np.float64)

    assert abs(_corr(frame["x_reversing"].to_numpy(), labels)) < 0.05


def test_the_reversing_column_is_strong_inside_each_subgroup_with_opposite_signs() -> None:
    """A column that were merely weak everywhere would be a probe, not a reversal."""
    generated = generate(simpson_reversal_spec(seed=0))
    frame, labels = generated.frame, generated.target.to_numpy().astype(np.float64)
    high = frame["group"].to_numpy() > 0

    inside_high = _corr(frame["x_reversing"].to_numpy()[high], labels[high])
    inside_low = _corr(frame["x_reversing"].to_numpy()[~high], labels[~high])
    assert abs(inside_high) > 0.3 and abs(inside_low) > 0.3
    assert inside_high * inside_low < 0, f"the two subgroups agree in sign ({inside_high:+.3f}, {inside_low:+.3f}), so nothing reverses"


def test_grouped_rows_really_are_clustered() -> None:
    """A per-group offset that did not vary between groups would leave the rows independent."""
    generated = generate(grouped_rows_spec(seed=0))
    frame, labels = generated.frame, generated.target.to_numpy().astype(np.float64)
    groups = frame["group_id"].to_numpy()

    rates = np.asarray([labels[groups == level].mean() for level in np.unique(groups)])
    assert float(np.std(rates)) > 0.05, f"per-group positive rates vary by only {float(np.std(rates)):.3f}: the rows are effectively independent"


def test_the_group_effect_is_reproducible_for_one_seed() -> None:
    """An unnamed draw would make two runs of one spec differ, which is the property the whole package rests on."""
    first = generate(grouped_rows_spec(seed=0)).target.to_numpy()
    second = generate(grouped_rows_spec(seed=0)).target.to_numpy()

    np.testing.assert_array_equal(first, second)


def test_a_group_effect_term_without_a_stream_is_refused() -> None:
    """A per-level draw from an unnamed generator would silently break reproducibility rather than fail."""
    from mlframe.data.datasets._links import basis_term_value
    from mlframe.data.datasets.spec import BasisTerm

    with pytest.raises(ValueError, match="needs a stream"):
        basis_term_value(BasisTerm(kind="group_effect", columns=("g",)), {"g": np.zeros(10)}, rng=None)


@pytest.mark.parametrize("name", NEW_BEDS)
def test_every_new_bed_is_registered_and_predicts_something(name: str) -> None:
    """A bed nobody registered cannot run, and one predicting nothing cannot be scored against its forecast."""
    assert name in scenario_registry.names()
    assert scenario_registry.get(name).expected_to_break
