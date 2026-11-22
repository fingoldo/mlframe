"""Early stop of the family permutation pass: a column is certainly passing only when its exceedance count is far below the level.

Behaviour, not formula: with no exceedances after many permutations every candidate is certain to pass; with exceedances near the level none is; the check refuses to conclude
before 512 permutations, and with no certainty margin it never concludes.
"""

from __future__ import annotations

import numpy as np

import mlframe.feature_selection.filters._family_maxt_gate as gate


def test_no_exceedances_after_many_permutations_is_certain_to_pass():
    """Three candidates with zero exceedances over 2000 of 10000 permutations are certain to pass at alpha 0.05."""
    assert gate._passes_for_certain(np.zeros(3), 2000, 10000, 0.05) is True


def test_exceedances_near_the_level_are_not_certain():
    """Candidates exceeded on most of the 2000 permutations already are not certain to pass."""
    assert gate._passes_for_certain(np.full(3, 1500.0), 2000, 10000, 0.05) is False


def test_a_single_failing_candidate_blocks_the_conclusion():
    """One candidate with many exceedances keeps the whole family open, even when the others are clean."""
    r = np.array([0.0, 0.0, 1500.0])
    assert gate._passes_for_certain(r, 2000, 10000, 0.05) is False


def test_no_conclusion_before_512_permutations():
    """Fewer than 512 permutations never conclude, however clean the candidates look."""
    assert gate._passes_for_certain(np.zeros(3), 511, 10000, 0.05) is False


def test_exactly_512_permutations_can_conclude():
    """The threshold is inclusive: at 512 permutations with no exceedances the candidates are already certain to pass."""
    assert gate._passes_for_certain(np.zeros(3), 512, 10000, 0.05) is True


def test_zero_margin_never_concludes_even_when_every_permutation_is_drawn(monkeypatch):
    """With the certainty margin at exactly zero the check refuses to conclude, even when done equals total and the bound is otherwise met."""
    monkeypatch.setattr(gate, "_CERTAINTY_DELTA", 0.0)
    assert gate._passes_for_certain(np.zeros(3), 2000, 2000, 0.05) is False


def test_the_decision_flips_exactly_at_the_oracle_boundary():
    """At the largest exceedance count the certainty bound accepts, the check passes; one more exceedance fails it.

    The bound is the Beta-quantile test written out directly, so the boundary is the one the definition gives.
    """
    from scipy.special import betaincinv

    done, total, alpha = 2000, 10000, 0.05
    limit = alpha * (total + 1.0) - 1.0

    def oracle_passes(r: int) -> bool:
        """The bound evaluated from its definition for one candidate with *r* exceedances."""
        q = betaincinv(r + 1.0, float(done) - r, 1.0 - gate._CERTAINTY_DELTA)
        return bool(r + (total - done) * q <= limit)

    boundary = max(r for r in range(0, 1200) if oracle_passes(r))
    assert oracle_passes(boundary) and not oracle_passes(boundary + 1)
    assert gate._passes_for_certain(np.array([float(boundary)]), done, total, alpha) is True
    assert gate._passes_for_certain(np.array([float(boundary + 1)]), done, total, alpha) is False


def test_the_first_alpha_shift_that_changes_the_decision_is_matched():
    """Where a shift of the Beta first parameter by 0.1 changes the decision, the check follows the unshifted bound."""
    from scipy.special import betaincinv

    done, total, alpha = 700, 10000, 0.05
    limit = alpha * (total + 1.0) - 1.0

    def passes(r: int, shift: float) -> bool:
        """The bound from its definition with the first Beta parameter moved by *shift*."""
        q = betaincinv(r + 1.0 + shift, float(done) - r, 1.0 - gate._CERTAINTY_DELTA)
        return bool(r + (total - done) * q <= limit)

    mismatch = next((r for r in range(0, 50) if passes(r, 0.0) != passes(r, 0.1)), None)
    assert mismatch is not None, "no decision separates the shifted bound from the exact one"
    expected = passes(mismatch, 0.0)
    assert gate._passes_for_certain(np.array([float(mismatch)]), done, total, alpha) is expected


def test_no_candidates_never_conclude():
    """An empty candidate set has nothing to certify."""
    assert gate._passes_for_certain(np.zeros(0), 2000, 10000, 0.05) is False


def test_zero_certainty_margin_never_concludes(monkeypatch):
    """With no certainty margin the check is disabled: it returns False even for clean candidates."""
    monkeypatch.setattr(gate, "_CERTAINTY_DELTA", 0.0)
    assert gate._passes_for_certain(np.zeros(3), 2000, 10000, 0.05) is False
