"""The published reference beds, the nonlinear terms they need, and the calibration bug they exposed.

These beds are the only ones in the suite whose structure was not chosen by the author of one of the arms
being judged, which is the entire reason for having them. That makes their FIDELITY the thing to test: a
Friedman-1 that quietly became something else is worse than no reference bed at all, because it still
carries the name.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from mlframe.data.datasets import scenarios as scenario_registry
from mlframe.data.datasets._links import basis_score, basis_term_value
from mlframe.data.datasets._target import _finite_bracket, calibrate_scale
from mlframe.data.datasets.generator import generate
from mlframe.data.datasets.scenarios._reference import friedman1_spec, friedman2_spec, friedman3_spec, weston_guyon_spec
from mlframe.data.datasets.spec import BasisTerm

REFERENCE_BEDS = ("friedman1", "friedman2", "friedman3", "weston_guyon_k4_p100")


def test_sin_product_term_matches_the_published_formula() -> None:
    """`10 sin(pi x0 x1)` must be exactly that, not a product standing in for a sine."""
    x0 = np.linspace(0.0, 1.0, 101)
    x1 = np.linspace(1.0, 0.0, 101)
    term = BasisTerm(kind="sin_product", columns=("x0", "x1"), weight=10.0, params={"frequency": math.pi})

    values = basis_term_value(term, {"x0": x0, "x1": x1})

    np.testing.assert_allclose(values, 10.0 * np.sin(math.pi * x0 * x1))


def test_centered_square_is_symmetric_about_its_centre() -> None:
    """The quadratic term's whole role is being invisible to a linear statistic, which needs exact symmetry."""
    term = BasisTerm(kind="centered_square", columns=("x",), weight=20.0, params={"center": 0.5})
    below = basis_term_value(term, {"x": np.array([0.5 - 0.3])})
    above = basis_term_value(term, {"x": np.array([0.5 + 0.3])})

    assert below[0] == pytest.approx(above[0])


def test_centered_square_has_no_linear_signal_over_a_symmetric_range() -> None:
    """Measured, not asserted from theory: a correlation near zero is why this term separates method families."""
    values = np.linspace(0.0, 1.0, 20001)
    term = BasisTerm(kind="centered_square", columns=("x",), weight=20.0, params={"center": 0.5})

    squared = basis_term_value(term, {"x": values})

    assert abs(float(np.corrcoef(values, squared)[0, 1])) < 0.01


def test_ratio_term_floors_the_denominator_instead_of_producing_one_dominating_row() -> None:
    """An unfloored ratio lets a single near-zero denominator set the scale of the whole score."""
    term = BasisTerm(kind="ratio", columns=("a", "b"), weight=1.0, params={"floor": 1.0})
    numerator = np.array([1.0, 1.0, 1.0])
    denominator = np.array([1e-12, -1e-12, 4.0])

    values = basis_term_value(term, {"a": numerator, "b": denominator})

    assert np.all(np.isfinite(values))
    assert float(np.max(np.abs(values))) <= 1.0


def test_basis_terms_sum_rather_than_replace_each_other() -> None:
    """A link naming several terms must get all of them; a dispatch returning the last would be silent."""
    columns = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    terms = (BasisTerm(kind="identity", columns=("a",), weight=2.0), BasisTerm(kind="identity", columns=("b",), weight=5.0))

    total = basis_score(terms, columns, n=2)

    np.testing.assert_allclose(total, 2.0 * columns["a"] + 5.0 * columns["b"])


def test_basis_term_rejects_an_arity_its_kind_cannot_use() -> None:
    """A silently ignored extra operand would change the formula the bed's name promises."""
    with pytest.raises(ValueError, match="exactly one column"):
        BasisTerm(kind="centered_square", columns=("a", "b"))
    with pytest.raises(ValueError, match="exactly two columns"):
        BasisTerm(kind="ratio", columns=("a",))


def test_basis_term_names_an_unknown_column_loudly() -> None:
    """A term referencing a column that does not exist is a bed typo, and a zero term would hide it."""
    with pytest.raises(KeyError, match="unknown column"):
        basis_term_value(BasisTerm(kind="identity", columns=("missing",)), {"present": np.zeros(3)})


@pytest.mark.parametrize("name", REFERENCE_BEDS)
def test_reference_bed_reaches_the_ceiling_it_declares(name: str) -> None:
    """A bed whose achievable AUC does not match its spec is not the difficulty anybody declared.

    Regression test for a real defect. The calibrator abandoned the bisection whenever either bracket end
    produced a non-finite AUC -- which every published bed does at the wide end, because their coefficients
    are stated in the units of their own inputs and drive the logistic to saturation. It then returned the
    unit scale without raising: Friedman-1 shipped at an achievable AUC of 0.966 against a declared 0.85,
    and only the calibration record said so.
    """
    generated = generate(scenario_registry.get(name).build(seed=0))
    requested = generated.calibration["requested"]
    achieved = generated.calibration["achieved_auc"]

    assert requested is not None, f"{name} declares no ceiling, so there is nothing to calibrate"
    assert achieved == pytest.approx(requested, abs=0.01), f"{name} declared {requested} and achieved {achieved}"


def test_finite_bracket_walks_inward_from_a_saturating_end() -> None:
    """The rescue must find a finite bracket rather than give up, which is what the old code did."""

    def achieved(scale: float) -> float:
        """A curve that is finite only on a narrow middle band, as a saturating link's AUC is."""
        if scale > 1.0 or scale < 0.01:
            return float("nan")
        return 0.5 + 0.4 * scale

    low, lo_auc, high, hi_auc = _finite_bracket(achieved, low=1e-3, high=50.0)

    assert lo_auc is not None and hi_auc is not None
    assert 0.01 <= low <= high <= 1.0
    assert hi_auc > lo_auc


def test_calibrate_scale_hits_the_target_through_a_saturating_bracket() -> None:
    """End to end on the shape that broke it: a score wide enough to saturate at the bracket's far end."""
    rng = np.random.default_rng(0)
    score = rng.normal(0.0, 30.0, 20000)

    def probability_at(scale: float) -> np.ndarray:
        """Saturate hard at any large scale, exactly as a published bed's link does."""
        return 1.0 / (1.0 + np.exp(-np.clip(scale * score, -700, 700)))

    scale, achieved = calibrate_scale(probability_at, target_auc=0.75)

    assert achieved == pytest.approx(0.75, abs=0.01)
    assert scale > 0.0


def test_friedman1_declares_exactly_the_five_informative_columns() -> None:
    """The published design has five informative columns; an answer key with six would score every arm wrong."""
    spec = friedman1_spec(seed=0)
    sources = sorted({edge.source for edge in spec.edges})

    assert sources == ["x0", "x1", "x2", "x3", "x4"]


def test_friedman_columns_are_not_standardized_away_from_their_published_ranges() -> None:
    """Friedman's constants are calibrated to his input ranges; standardising would silently change the function."""
    frame = generate(friedman1_spec(seed=0)).frame

    for column in ("x0", "x1", "x2", "x3", "x4"):
        values = frame[column].to_numpy()
        assert 0.0 <= values.min() and values.max() <= 1.0, f"{column} left the unit interval"
        assert values.std() == pytest.approx(1.0 / math.sqrt(12.0), rel=0.1), f"{column} looks standardized"


def test_friedman2_keeps_its_published_range_imbalance() -> None:
    """The unequal column ranges ARE the bed: a method that standardises sees a different problem."""
    frame = generate(friedman2_spec(seed=0)).frame

    spans = {column: float(frame[column].max() - frame[column].min()) for column in ("x0", "x1", "x2", "x3")}
    assert spans["x1"] > 100.0 * spans["x2"], f"the range imbalance is gone: {spans}"


def test_friedman3_signal_is_the_ratio_and_the_same_four_columns() -> None:
    """Friedman-3 reuses Friedman-2's columns; a port that silently changed them would not be the named bed."""
    spec = friedman3_spec(seed=0)

    assert sorted({edge.source for edge in spec.edges}) == ["x0", "x1", "x2", "x3"]
    kinds = [term.kind for term in spec.targets[0].link.basis_terms]
    assert kinds == ["ratio"]


def test_weston_guyon_weights_every_informative_column_equally() -> None:
    """Equal weights are the design: a decaying vector would let a method score well by finding one column."""
    weights = list(weston_guyon_spec(seed=0).targets[0].link.coefficients.values())

    assert len(set(weights)) == 1, f"the informative columns carry unequal weights: {weights}"


def test_weston_guyon_probes_share_the_informative_marginal() -> None:
    """Probes drawn from a different marginal would be findable without ever looking at the target."""
    frame = generate(weston_guyon_spec(n_probes=40, seed=0)).frame
    informative_sd = float(np.mean([frame[f"s{i}"].std() for i in range(4)]))
    probe_sd = float(np.mean([frame[column].std() for column in frame.columns if column.startswith("n")]))

    assert probe_sd == pytest.approx(informative_sd, rel=0.1)


def test_every_reference_bed_is_registered_with_a_prediction() -> None:
    """A bed nobody registered cannot be run, and one predicting nothing cannot be scored."""
    registered: List[str] = list(scenario_registry.names())

    assert REFERENCE_BEDS
    for name in REFERENCE_BEDS:
        assert name in registered
        assert scenario_registry.get(name).expected_to_break, f"{name} declares no arms it expects to defeat"
