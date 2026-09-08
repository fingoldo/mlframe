"""Tests for copula dependence and the beds whose signal lives in the joint tail.

The property that matters is the one a Cholesky factor cannot express: two datasets can share a rank
correlation and differ entirely in whether extremes co-occur. If the families here do not separate on that,
the beds built from them are testing correlation under a new name.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from mlframe.data.datasets._copula import COPULA_FAMILIES, clayton_copula, sample_copula, upper_tail_coefficient
from mlframe.data.datasets._links import tail_gate_term
from mlframe.data.datasets.generator import generate
from mlframe.data.datasets.scenarios import get as get_scenario

ROWS = 60_000


def _uniforms(family: str, seed: int = 0, **kwargs: float) -> np.ndarray:
    """Draw one copula's uniform margins at a size where a tail estimate means something."""
    return sample_copula(np.random.default_rng(seed), ROWS, 2, family=family, **kwargs)


class TestFamiliesDifferWhereTheyShould:
    """Same correlation, different tails -- otherwise the families are one family."""

    def test_the_t_copula_has_more_tail_dependence_than_the_gaussian_at_equal_correlation(self) -> None:
        """This is the whole reason the module exists: a Gaussian copula has no asymptotic tail dependence."""
        gaussian = _uniforms("gaussian", rho=0.7)
        student = _uniforms("t", rho=0.7, df=4.0)
        assert abs(np.corrcoef(gaussian.T)[0, 1] - np.corrcoef(student.T)[0, 1]) < 0.05
        assert upper_tail_coefficient(student, 0.99) > upper_tail_coefficient(gaussian, 0.99) + 0.05

    def test_the_gaussian_tail_coefficient_decays_as_the_quantile_moves_out(self) -> None:
        """Asymptotic independence shows up as a coefficient that keeps shrinking; the t's does not."""
        gaussian = _uniforms("gaussian", rho=0.7)
        assert upper_tail_coefficient(gaussian, 0.99) < upper_tail_coefficient(gaussian, 0.90)

    def test_clayton_is_asymmetric_between_the_tails(self) -> None:
        """Lower-tail dependence only: a method that handles symmetry by symmetry still has work to do."""
        clayton = clayton_copula(np.random.default_rng(0), ROWS, 2, theta=2.0)
        lower = upper_tail_coefficient(1.0 - clayton, 0.95)
        upper = upper_tail_coefficient(clayton, 0.95)
        assert lower > 0.4 and upper < 0.2

    def test_margins_are_uniform_for_every_family(self) -> None:
        """Dependence and margin are separable, and mixing them is how a joint test becomes a margin test."""
        for family in COPULA_FAMILIES:
            data = _uniforms(family)
            assert 0.45 < data.mean() < 0.55, family
            assert data.min() >= 0.0 and data.max() <= 1.0, family


class TestRefusals:
    """A typo must not fall back to the control family."""

    def test_an_unknown_family_is_refused(self) -> None:
        """Silently returning the Gaussian would hand back the one family with no tail dependence."""
        with pytest.raises(ValueError, match="unknown copula family"):
            sample_copula(np.random.default_rng(0), 100, 2, family="frank")

    def test_an_impossible_equicorrelation_is_refused(self) -> None:
        """For three columns the lower bound is -1/2, not -1; asking for -0.9 asks for no matrix at all."""
        with pytest.raises(ValueError, match="equicorrelation rho"):
            sample_copula(np.random.default_rng(0), 100, 3, family="gaussian", rho=-0.9)

    def test_a_single_column_is_refused(self) -> None:
        """A copula describes dependence, and one column has none to describe."""
        with pytest.raises(ValueError, match="at least two columns"):
            sample_copula(np.random.default_rng(0), 100, 1, family="gaussian")


class TestTailGateTerm:
    """The link term that puts the signal in the joint tail."""

    def test_it_fires_where_the_operands_share_a_tail_in_either_direction(self) -> None:
        """Both-high and both-low fire; one of each does not. Symmetry is what removes the marginal channel."""
        a = np.array([-2.0, 2.0, -2.0, 2.0])
        b = np.array([-2.0, -2.0, 2.0, 2.0])
        assert tail_gate_term([a, b], quantile=0.75).tolist() == [1.0, 0.0, 0.0, 1.0]

    def test_the_gate_leaves_no_linear_marginal_signal(self) -> None:
        """The whole reason for the symmetric form: an upper-only gate left each column at +0.51 correlation.

        Being high must be no more predictive than being low, or a univariate filter recovers the pair from
        the marginal alone and the bed stops testing anything joint.
        """
        rng = np.random.default_rng(0)
        columns = [rng.normal(size=40_000), rng.normal(size=40_000)]
        gate = tail_gate_term(columns, 0.8)
        assert abs(float(np.corrcoef(columns[0], gate)[0, 1])) < 0.05

    def test_the_gate_region_stays_small(self) -> None:
        """Two independent columns at the 0.8 quantile share a tail on about 8% of rows, not 40%."""
        rng = np.random.default_rng(0)
        columns = [rng.normal(size=20_000), rng.normal(size=20_000)]
        assert 0.05 < tail_gate_term(columns, 0.8).mean() < 0.12


class TestTailBeds:
    """The registered pair."""

    def _beds(self) -> tuple:
        """Generate both tail beds at their declared sizes."""
        return generate(get_scenario("joint_tail_t4").build(seed=0)), generate(get_scenario("joint_tail_gaussian_control").build(seed=0))

    def test_the_pair_shares_a_correlation_and_differs_in_the_tail(self) -> None:
        """Only then does a failure on one and not the other identify tails as the cause."""
        student, gaussian = self._beds()
        pairs = []
        for dataset in (student, gaussian):
            columns = np.column_stack([stats.norm.cdf(dataset.frame["t0"]), stats.norm.cdf(dataset.frame["t1"])])
            pairs.append((float(np.corrcoef(columns.T)[0, 1]), upper_tail_coefficient(columns, 0.95)))
        (corr_t, tail_t), (corr_g, tail_g) = pairs
        assert abs(corr_t - corr_g) < 0.05
        assert tail_t > tail_g

    def test_the_answer_key_is_the_pair(self) -> None:
        """The probes must not enter the blanket, or the bed scores recovery of the wrong thing."""
        student, _gaussian = self._beds()
        assert set(student.truth.primary_target_set().members) == {"t0", "t1"}

    def test_the_beds_declare_no_target_ceiling_and_report_what_they_reached(self) -> None:
        """Forcing a ceiling would widen the gate region, which is the property being tested."""
        student, _gaussian = self._beds()
        assert student.calibration["requested"] is None
        assert 0.5 < student.calibration["bayes_auc"] < 0.8

    def test_neither_bed_leaves_a_marginal_shortcut(self) -> None:
        """A univariate filter must not be able to recover the pair without ever looking at the joint."""
        for dataset in self._beds():
            correlation = abs(float(np.corrcoef(dataset.frame["t0"].to_numpy(), dataset.truth.true_prob)[0, 1]))
            assert correlation < 0.06, correlation
