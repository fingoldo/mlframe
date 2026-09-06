"""Behavioural tests for the paired-design power analysis.

The point of these is that the two directions agree: a design sized by ``required_replicates`` for an effect
must actually reach the target power against that effect, and ``detectable_effect`` must be the inverse of
``required_replicates`` up to the integer rounding of the seed count. A power calculation that is merely
self-consistent in one direction is the usual way these end up off by the ``t``-versus-normal correction.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid._power import (
    DEFAULT_POWER,
    achieved_power,
    detectable_effect,
    power_report,
    required_replicates,
    tau_estimates,
)


def _record(scenario: str, arm: str, seed: int, auc: float) -> Dict[str, Any]:
    """Build a minimal ``ok`` cell record carrying one lightgbm AUC at ``k10``."""
    return {
        "status": "ok",
        "arm": arm,
        "scenario": scenario,
        "dataset_seed": seed,
        "cv_seed": 0,
        "scores": {"k10": {"models": {"lightgbm": {"roc_auc": auc}}}},
    }


class TestRequiredReplicates:
    """The planning direction."""

    def test_scales_with_the_squared_noise_to_effect_ratio(self) -> None:
        """Doubling tau at fixed effect asks for roughly four times the seeds."""
        small = required_replicates(0.01, 0.005)
        large = required_replicates(0.02, 0.005)
        assert small is not None and large is not None
        assert 3.5 < large / small < 4.5

    def test_zero_spread_needs_two_seeds(self) -> None:
        """A contrast that moves identically on every seed is established by two of them."""
        assert required_replicates(0.0, 0.005) == 2

    def test_rejects_a_nonpositive_effect(self) -> None:
        """An effect of zero is not detectable at any sample size, and says so instead of looping."""
        assert required_replicates(0.01, 0.0) is None

    @pytest.mark.parametrize("effect", [0.002, 0.005, 0.01, 0.02])
    def test_the_recommended_size_actually_reaches_the_target_power(self, effect: float) -> None:
        """The seeds it recommends hit 80% power against the effect they were sized for."""
        tau = 0.02
        m = required_replicates(tau, effect)
        assert m is not None
        power = achieved_power(tau, effect, m)
        assert power is not None
        assert power >= DEFAULT_POWER
        # And one seed fewer would not have: the recommendation is tight, not merely sufficient.
        if m > 2:
            lower = achieved_power(tau, effect, m - 1)
            assert lower is not None and lower < DEFAULT_POWER


class TestDetectableEffect:
    """The interpretive direction, which is what makes a null result readable."""

    def test_is_the_inverse_of_required_replicates(self) -> None:
        """Sizing for the MDE at m seeds returns m, up to the integer rounding of the seed count."""
        tau, m = 0.03, 20
        mde = detectable_effect(tau, m)
        assert mde is not None
        back = required_replicates(tau, mde)
        assert back is not None and abs(back - m) <= 1

    def test_shrinks_with_the_square_root_of_the_seed_count(self) -> None:
        """Four times the seeds resolve roughly half the effect."""
        at20 = detectable_effect(0.03, 20)
        at80 = detectable_effect(0.03, 80)
        assert at20 is not None and at80 is not None
        assert 1.8 < at20 / at80 < 2.2

    def test_undefined_below_two_seeds(self) -> None:
        """A single seed carries no spread, so it resolves nothing."""
        assert detectable_effect(0.03, 1) is None


class TestAchievedPower:
    """The non-central t path, which is where a normal approximation would quietly overstate power."""

    def test_power_at_the_critical_effect_is_the_target(self) -> None:
        """Evaluated at its own MDE, the design has exactly the power it was sized for."""
        tau, m = 0.02, 25
        mde = detectable_effect(tau, m)
        assert mde is not None
        power = achieved_power(tau, mde, m)
        assert power is not None
        assert math.isclose(power, DEFAULT_POWER, abs_tol=0.02)

    def test_a_normal_approximation_would_overstate_it(self) -> None:
        """At small m the exact non-central t power is below the normal-approximation value."""
        from scipy import stats

        tau, delta, m = 0.02, 0.02, 5
        exact = achieved_power(tau, delta, m)
        assert exact is not None
        normal = float(stats.norm.sf(stats.norm.ppf(0.975) - delta * math.sqrt(m) / tau))
        assert exact < normal


class TestTauEstimatesFromRecords:
    """Estimation off real cell records, including the pairing rule."""

    def test_recovers_a_planted_spread(self) -> None:
        """The estimate is the sd of the per-seed differences, not of the raw values."""
        deltas = [0.01, 0.02, 0.03, 0.04]
        records: List[Dict[str, Any]] = []
        for seed, delta in enumerate(deltas):
            # The null arm's own level drifts far more than the difference does; a method that estimated
            # spread from the levels instead of the paired differences would report roughly 0.1 here.
            base = 0.60 + 0.1 * seed
            records.append(_record("bed", "all-features", seed, base))
            records.append(_record("bed", "arm", seed, base + delta))
        ests = tau_estimates(records, model="lightgbm", k_label="k10")
        assert len(ests) == 1
        import numpy as np

        assert ests[0].tau == pytest.approx(float(np.std(deltas, ddof=1)))
        assert ests[0].mean_delta == pytest.approx(float(np.mean(deltas)))
        assert ests[0].m == len(deltas)

    def test_unpaired_seeds_do_not_contribute(self) -> None:
        """A seed where only one of the two arms produced a value is dropped, not imputed."""
        records = [_record("bed", "all-features", s, 0.6) for s in range(4)]
        records += [_record("bed", "arm", s, 0.65) for s in range(3)]
        ests = tau_estimates(records, model="lightgbm", k_label="k10")
        assert len(ests) == 1 and ests[0].m == 3

    def test_failed_cells_are_not_counted_as_values(self) -> None:
        """A cell recorded as anything but ok carries no value into the estimate."""
        records = [_record("bed", "all-features", s, 0.6) for s in range(4)]
        records += [_record("bed", "arm", s, 0.65) for s in range(4)]
        records[-1]["status"] = "error"
        ests = tau_estimates(records, model="lightgbm", k_label="k10")
        assert len(ests) == 1 and ests[0].m == 3

    def test_report_states_the_detectable_effect_at_the_declared_size(self) -> None:
        """The report carries the MDE column, which is what a null result has to be read against."""
        records: List[Dict[str, Any]] = []
        for seed in range(5):
            records.append(_record("bed", "all-features", seed, 0.60 + 0.001 * seed))
            records.append(_record("bed", "arm", seed, 0.62 + 0.002 * seed))
        text = power_report(records, models=["lightgbm"], k_labels=["k10"], declared_r=20)
        assert "detectable at R=20" in text
        assert "bed" in text and "arm" in text
