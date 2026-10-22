"""The time axis and the proxy decoys: the two extensions added after the first four pillar beds.

Each assertion is on a property of generated data or of the spec's arithmetic, measured at the registered seed. The
thresholds are several standard errors from the measured values, so a regeneration that changes the draw does not change
the verdict.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.data.datasets import scenarios
from mlframe.data.datasets._columns import draw_family
from mlframe.data.datasets.generator import generate
from mlframe.data.datasets.scenarios._pillars import N_PROXY, PROXY_RHO, _proxy_weight


@pytest.fixture(scope="module")
def conjunction():
    """The conjunction pillar, generated once at its registered size."""
    return generate(scenarios.get("pillar_conjunction").build(seed=0))


def test_each_proxy_correlates_with_its_operand_at_the_declared_rho(conjunction) -> None:
    """The weight is solved in closed form, so the delivered correlation is the declared 0.7, not a tuned approximation."""
    for op in ("A", "B", "C"):
        proxy = np.asarray(conjunction.frame[f"dec{op}_0"], dtype=float)
        operand = np.asarray(conjunction.frame[op], dtype=float)
        assert np.corrcoef(proxy, operand)[0, 1] == pytest.approx(PROXY_RHO, abs=0.02), op


def test_the_proxy_weight_is_the_closed_form_solution() -> None:
    """Pins the algebra: with residual scale s and operand sd 0.5, the correlation is w sd / sqrt(w^2 sd^2 + s^2)."""
    from mlframe.data.datasets._derived import DEFAULT_RESIDUAL_SD

    w = _proxy_weight()
    achieved = w * 0.5 / np.sqrt(w * w * 0.25 + DEFAULT_RESIDUAL_SD**2)
    assert achieved == pytest.approx(PROXY_RHO, abs=1e-12)


def test_the_proxies_are_decoys_and_not_part_of_the_answer_key(conjunction) -> None:
    """A proxy is a child of its operand and of nothing else, so the answer key is exactly the three operands."""
    members = set(conjunction.truth.primary_target_set().members)
    assert members == {"A", "B", "C"}
    assert all(not name.startswith("dec") for name in members)
    assert sum(1 for c in conjunction.frame.columns if c.startswith("dec")) == 3 * N_PROXY


def test_the_proxies_of_a_fill_most_of_the_fixed_budget_before_b_or_c(conjunction) -> None:
    """Measured arithmetic: at width 250 the fixed budget is 62, and fifty of the top 62 by |correlation| are copies of A."""
    y = np.asarray(conjunction.target, dtype=float)
    names = np.array(conjunction.frame.columns)
    score = np.array([abs(np.corrcoef(np.asarray(conjunction.frame[c], dtype=float), y)[0, 1]) for c in names])
    top = names[np.argsort(-score)[:62]]
    proxies_of_a = sum(1 for name in top if name.startswith("decA"))
    assert proxies_of_a >= 45, f"only {proxies_of_a} proxies of A in the top 62"


@pytest.mark.parametrize("name", ["pillar_noise_ar1", "pillar_noise_random_walk"])
def test_the_temporal_label_is_serially_dependent(name: str) -> None:
    """The target is a time series of its own: adjacent labels are correlated, which an iid target would not be."""
    dataset = generate(scenarios.get(name).build(seed=0))
    y = np.asarray(dataset.target, dtype=float)
    assert np.corrcoef(y[:-1], y[1:])[0, 1] > 0.4


@pytest.mark.parametrize("name", ["pillar_noise_ar1", "pillar_noise_random_walk"])
def test_the_temporal_beds_disclose_the_unobserved_factor_their_ceiling_counts(name: str) -> None:
    """The Bayes AUC counts the latent that drives the label, which no column carries. That must be recorded, not implied."""
    dataset = generate(scenarios.get(name).build(seed=0))
    assert dataset.calibration["bayes_auc_counts_unobserved"] == ["u"]
    assert any("unobserved latent" in caveat for caveat in dataset.truth.caveats)
    assert dataset.truth.primary_target_set().members == ()


def test_the_ar1_family_has_lag_one_autocorrelation_and_unit_variance() -> None:
    """Stationary at rho 0.9: the lag-one correlation is 0.9 and the marginal sd is 1, with no burn-in transient."""
    x = draw_family("ar1", {"rho": 0.9}, 200_000, np.random.default_rng(0))
    assert np.corrcoef(x[:-1], x[1:])[0, 1] == pytest.approx(0.9, abs=0.01)
    assert x.std() == pytest.approx(1.0, abs=0.02)


def test_the_random_walk_family_is_the_cumulative_sum_of_its_steps() -> None:
    """The variance grows with the row index, which is what makes the walk spurious against another walk."""
    rng = np.random.default_rng(1)
    walk = draw_family("random_walk", {}, 1000, rng)
    steps = np.diff(np.concatenate([[0.0], walk]))
    assert np.allclose(np.cumsum(steps), walk)
    assert abs(steps.mean()) < 0.2 and steps.std() == pytest.approx(1.0, abs=0.1)
