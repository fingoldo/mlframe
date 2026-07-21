"""Direct unit coverage for ``_screen_predictors_gate.compute_selection_gate`` (mrmr_audit_2026-07-20
test_coverage.md #1 / edge_cases.md #1-4). Prior to this file, the gate's Miller-Madow sign, the
running-MAX (not first-selected) relative floor, the maxT floor's use of MARGINAL (not conditional)
MI, and the order>=2 MM-skip were only reachable transitively through a full MRMR.fit() -- this file
pins each boundary directly.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._screen_predictors_gate import compute_selection_gate


def _gate(**overrides):
    """Call compute_selection_gate with sane defaults, overridden per test."""
    n = 1500
    factors_data = np.zeros((n, 5), dtype=np.int32)
    defaults = dict(
        min_relevance_gain=0.01,
        interactions_order=1,
        best_candidate=(0,),
        best_gain=0.1,
        cardinality_bias_correction=True,
        factors_data=factors_data,
        y=[4],
        factors_nbins=[10, 10, 10, 10, 2],
        min_relevance_gain_relative_to_first=0.0,
        selected_vars=[],
        predictors=[],
        fdr_gain_floor=0.0,
        cached_MIs={},
    )
    defaults.update(overrides)
    return compute_selection_gate(**defaults)


class TestMillerMadowBiasSign:
    """The MM bias must be SUBTRACTED from the raw gain, not added -- plug-in MI is biased UPWARD."""

    def test_bias_reduces_gain_not_increases(self):
        """A high-cardinality candidate's corrected gain must be strictly LESS than its raw gain."""
        _passes, corrected = _gate(
            best_candidate=(0,), best_gain=0.1,
            factors_nbins=[100, 10, 10, 10, 2],  # nbins_x=100 -> large MM bias
        )
        assert corrected < 0.1, f"MM-corrected gain ({corrected:.4f}) must be LESS than raw gain (0.1) -- bias must subtract, not add"

    def test_bias_magnitude_matches_closed_form(self):
        """The corrected gain must equal raw_gain - (nbins_x-1)*(nbins_y-1)/(2n) exactly."""
        n = 1500
        nb_x, nb_y = 20, 2
        raw_gain = 0.05
        _passes, corrected = _gate(
            best_candidate=(0,), best_gain=raw_gain,
            factors_nbins=[nb_x, 10, 10, 10, nb_y],
        )
        expected_bias = (nb_x - 1) * (nb_y - 1) / (2.0 * n)
        assert abs(corrected - (raw_gain - expected_bias)) < 1e-12


class TestJointsSkipMillerMadowCorrection:
    """Order >= 2 (joint) candidates must NOT receive the MM correction -- it over-corrects
    multiplicatively-growing joint cardinalities and can kill genuine synergy signal."""

    def test_order_2_candidate_gain_unchanged(self):
        """A 2-way joint candidate's corrected gain must equal its raw gain (no MM subtraction)."""
        raw_gain = 0.05
        _passes, corrected = _gate(
            interactions_order=2,
            best_candidate=(0, 1),
            best_gain=raw_gain,
            factors_nbins=[39, 39, 10, 10, 2],  # deliberately high joint cardinality
            min_relevance_gain=0.001,
        )
        assert corrected == raw_gain, f"order>=2 candidate gain was corrected ({corrected}) but must pass through unchanged ({raw_gain})"

    def test_order_1_candidate_gain_is_corrected(self):
        """The order==1 sibling call on the SAME cardinalities DOES get corrected (contrast case)."""
        raw_gain = 0.05
        _passes, corrected = _gate(
            interactions_order=1,
            best_candidate=(0,),
            best_gain=raw_gain,
            factors_nbins=[39, 39, 10, 10, 2],
        )
        assert corrected < raw_gain


class TestRelativeFloorUsesRunningMax:
    """The relative floor must use the MAX corrected gain over already-selected predictors, not
    just the FIRST-selected one -- a cardinality-inflated first pick's MM-collapsed gain would set
    too low a floor if only the first were used."""

    def test_relative_floor_uses_max_not_first(self):
        """Regression: first-selected predictor has a LOW corrected gain (was cardinality-inflated
        raw), a LATER-selected predictor has a HIGHER corrected gain. The relative floor must be
        derived from the later (max), not the first."""
        predictors = [
            {"gain": 0.328, "indices": (0,)},  # high-cardinality: heavily MM-corrected down
            {"gain": 0.187, "indices": (1,)},  # low-cardinality: barely corrected, corrected > first's
        ]
        # nbins for indices 0 and 1 chosen so MM correction collapses predictor 0 below predictor 1.
        factors_nbins = [500, 3, 10, 10, 2]
        n = 1500
        nb_y = 2
        corr0 = 0.328 - (500 - 1) * (nb_y - 1) / (2.0 * n)
        corr1 = 0.187 - (3 - 1) * (nb_y - 1) / (2.0 * n)
        assert corr1 > corr0, "fixture assumption: predictor 1's corrected gain must exceed predictor 0's"

        # A candidate whose corrected gain clears corr0's relative floor but NOT corr1's.
        rel_frac = 0.5
        floor_from_first = corr0 * rel_frac
        floor_from_max = corr1 * rel_frac
        assert floor_from_first < floor_from_max
        candidate_gain = (floor_from_first + floor_from_max) / 2.0  # strictly between the two floors

        passes, _corrected = _gate(
            best_candidate=(2,), best_gain=candidate_gain,
            factors_nbins=factors_nbins,
            min_relevance_gain=0.0,
            min_relevance_gain_relative_to_first=rel_frac,
            selected_vars=[0, 1],
            predictors=predictors,
        )
        assert not passes, (
            "B-regression: the relative floor used the FIRST-selected predictor's corrected gain "
            "instead of the MAX over all selected predictors -- a candidate that should have been "
            "rejected against the true (higher) max-floor was wrongly admitted."
        )


class TestMaxTFloorUsesMarginalNotConditionalMi:
    """The maxT permutation-null floor must compare the candidate's CACHED MARGINAL MI, not the
    (possibly conditioning-bias-inflated) Fleuret conditional gain passed as best_gain."""

    def test_high_conditional_gain_low_marginal_mi_fails_fdr_floor(self):
        """A candidate whose conditional gain clears the abs/rel floors but whose cached MARGINAL
        MI is below the maxT floor must be REJECTED -- proving the floor reads cached_MIs, not
        best_gain."""
        fdr_floor = 0.05
        passes, _corrected = _gate(
            best_candidate=(0,),
            best_gain=0.20,  # conditional gain: clears every other floor easily
            cached_MIs={(0,): 0.01},  # marginal MI: BELOW the fdr floor
            fdr_gain_floor=fdr_floor,
            min_relevance_gain=0.0,
            cardinality_bias_correction=False,
        )
        assert not passes, (
            "maxT floor admitted a candidate whose conditional gain (0.20) cleared it but whose "
            "cached MARGINAL MI (0.01) did not -- the floor must read cached_MIs, not best_gain."
        )

    def test_high_marginal_mi_passes_fdr_floor_even_with_moderate_conditional_gain(self):
        """The mirror case: marginal MI clears the floor -> passes, confirming the floor genuinely
        reads the marginal value rather than always failing."""
        fdr_floor = 0.05
        passes, _corrected = _gate(
            best_candidate=(0,),
            best_gain=0.06,
            cached_MIs={(0,): 0.20},  # marginal MI clears the floor comfortably
            fdr_gain_floor=fdr_floor,
            min_relevance_gain=0.0,
            cardinality_bias_correction=False,
        )
        assert passes

    def test_fdr_floor_is_noop_for_joints(self):
        """The maxT floor only applies to order-1 single candidates; a 2-way joint must never be
        gated by it (per the function's own documented contract)."""
        passes, _corrected = _gate(
            interactions_order=2,
            best_candidate=(0, 1),
            best_gain=0.06,
            cached_MIs={(0, 1): 0.0},  # would fail the floor if (wrongly) applied
            fdr_gain_floor=0.05,
            min_relevance_gain=0.0,
            cardinality_bias_correction=False,
        )
        assert passes, "maxT floor must be a no-op for interactions_order >= 2"
