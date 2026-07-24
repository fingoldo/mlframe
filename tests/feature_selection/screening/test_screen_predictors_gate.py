"""Unit coverage for ``_screen_predictors_gate.py``'s ``compute_selection_gate`` / ``build_dcd_state``.

X_TEST_COVERAGE_QUALITY-5 fix (mrmr_audit_2026-07-22): this module had zero test references anywhere
in the suite despite implementing the core greedy screen's accept/reject decision (absolute floor,
relative-to-first floor, Miller-Madow cardinality-bias correction, maxT-FDR floor) and the Dynamic
Cluster Discovery state constructor. Both are pure functions w.r.t. their explicit arguments (per the
module's own docstring), so they are directly unit-testable without a full ``MRMR.fit()``.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._screen_predictors_gate import (
    build_dcd_state,
    compute_selection_gate,
)


def _base_kwargs(**overrides):
    """Baseline ``compute_selection_gate`` kwargs (gate features off unless overridden)."""
    kw = dict(
        min_relevance_gain=0.05,
        interactions_order=1,
        best_candidate=(0,),
        best_gain=0.2,
        cardinality_bias_correction=False,
        factors_data=np.zeros((1000, 5), dtype=np.int32),
        y=(4,),
        factors_nbins=[10, 10, 10, 10, 2],
        min_relevance_gain_relative_to_first=0.0,
        selected_vars=[],
        predictors=[],
        fdr_gain_floor=0.0,
        cached_MIs={},
    )
    kw.update(overrides)
    return kw


def test_gain_above_absolute_floor_passes():
    """A gain comfortably above ``min_relevance_gain`` passes with no other gates active."""
    passes, gain_for_gate = compute_selection_gate(**_base_kwargs(best_gain=0.2, min_relevance_gain=0.05))
    assert passes is True
    assert gain_for_gate == 0.2


def test_gain_below_absolute_floor_fails():
    """A gain below ``min_relevance_gain`` fails the absolute floor."""
    passes, _ = compute_selection_gate(**_base_kwargs(best_gain=0.01, min_relevance_gain=0.05))
    assert passes is False


def test_joint_candidate_uses_root_scaled_absolute_floor():
    """For ``interactions_order > 1`` the absolute floor is ``min_relevance_gain ** (1/(order+1))``,
    not the raw ``min_relevance_gain`` -- a joint candidate needs a smaller per-order share to pass."""
    # order=2 -> floor = 0.05**(1/3) ~= 0.368; a gain of 0.2 clears the flat 0.05 floor but NOT this one.
    passes, _ = compute_selection_gate(**_base_kwargs(best_gain=0.2, min_relevance_gain=0.05, interactions_order=2, best_candidate=(0, 1)))
    assert passes is False
    passes, _ = compute_selection_gate(**_base_kwargs(best_gain=0.5, min_relevance_gain=0.05, interactions_order=2, best_candidate=(0, 1)))
    assert passes is True


def test_cardinality_bias_correction_reduces_order1_gain():
    """Miller-Madow correction subtracts (nbins_x-1)*(nbins_y-1)/(2n) from an order-1 candidate's gain,
    and this corrected value both drives the accept/reject decision and is returned to the caller."""
    kw = _base_kwargs(
        best_gain=0.10,
        min_relevance_gain=0.05,
        cardinality_bias_correction=True,
        best_candidate=(0,),
        factors_nbins=[10, 10, 10, 10, 2],
        factors_data=np.zeros((1000, 5), dtype=np.int32),
    )
    expected_bias = (10 - 1) * (2 - 1) / (2.0 * 1000)
    passes, gain_for_gate = compute_selection_gate(**kw)
    assert gain_for_gate == 0.10 - expected_bias
    assert passes is True  # 0.10 - 0.0045 = 0.0955 still clears 0.05


def test_cardinality_bias_correction_not_applied_to_joints():
    """The MM correction is order-1-only; a joint candidate's gain passes through uncorrected."""
    _passes, gain_for_gate = compute_selection_gate(
        **_base_kwargs(
            best_gain=0.50,
            min_relevance_gain=0.05,
            cardinality_bias_correction=True,
            interactions_order=2,
            best_candidate=(0, 1),
        )
    )
    assert gain_for_gate == 0.50


def test_relative_to_first_floor_rejects_trailing_noise():
    """From the 2nd selected feature onward, a gain far below ``min_relevance_gain_relative_to_first``
    times the max corrected gain among already-accepted predictors fails, even if it clears the
    absolute floor."""
    predictors = [{"gain": 0.20, "indices": (0,)}]
    passes, _ = compute_selection_gate(
        **_base_kwargs(
            best_gain=0.06,  # clears the 0.05 absolute floor...
            min_relevance_gain=0.05,
            min_relevance_gain_relative_to_first=0.5,  # ...but needs >= 0.10 (50% of 0.20) to pass this gate
            selected_vars=[0],
            predictors=predictors,
        )
    )
    assert passes is False


def test_relative_to_first_floor_accepts_comparable_signal():
    """A candidate whose gain is comparable to the strongest already-selected predictor clears the
    relative floor."""
    predictors = [{"gain": 0.20, "indices": (0,)}]
    passes, _ = compute_selection_gate(
        **_base_kwargs(
            best_gain=0.15,
            min_relevance_gain=0.05,
            min_relevance_gain_relative_to_first=0.5,
            selected_vars=[0],
            predictors=predictors,
        )
    )
    assert passes is True


def test_relative_floor_uses_max_not_first_when_correction_active():
    """The relative floor is scaled by the MAX corrected gain across already-selected predictors, not
    just the first one -- a later, stronger predictor raises the bar for subsequent candidates."""
    predictors = [
        {"gain": 0.30, "indices": (0,)},  # weaker first pick
        {"gain": 0.60, "indices": (1,)},  # stronger second pick
    ]
    # 50% of the max (0.60) = 0.30; a candidate at 0.25 must fail even though it would pass
    # against the first predictor's own 0.30 * 0.5 = 0.15 floor.
    passes, _ = compute_selection_gate(
        **_base_kwargs(
            best_gain=0.25,
            min_relevance_gain=0.05,
            min_relevance_gain_relative_to_first=0.5,
            selected_vars=[0, 1],
            predictors=predictors,
        )
    )
    assert passes is False


def test_fdr_gain_floor_rejects_below_chance_ceiling():
    """The maxT-FDR floor compares the candidate's cached MARGINAL MI (not the conditional gain) against
    a permutation-null chance ceiling for order-1 single candidates; a marginal MI below it fails even
    when the (possibly conditioning-inflated) conditional gain would otherwise pass."""
    passes, _ = compute_selection_gate(
        **_base_kwargs(
            best_gain=0.20,  # clears the absolute floor on its own
            min_relevance_gain=0.05,
            fdr_gain_floor=0.15,
            best_candidate=(0,),
            cached_MIs={(0,): 0.05},  # marginal MI below the 0.15 FDR floor
        )
    )
    assert passes is False


def test_fdr_gain_floor_ignored_for_joint_candidates():
    """The FDR floor only applies to single-index (order-1) candidates; a multi-index candidate is
    unaffected regardless of ``cached_MIs`` contents."""
    passes, _ = compute_selection_gate(
        **_base_kwargs(
            best_gain=0.50,
            min_relevance_gain=0.05,
            fdr_gain_floor=100.0,  # would reject any single candidate
            best_candidate=(0, 1),
            interactions_order=2,
        )
    )
    assert passes is True


def test_build_dcd_state_returns_none_when_config_missing():
    """``dcd_config=None`` is a plain no-op -> ``None``, never an error."""
    assert build_dcd_state(None, np.zeros((10, 3), dtype=np.int32), [5, 5, 5], ["a", "b", "c"], (2,), None, 0) is None


def test_build_dcd_state_returns_none_when_not_enabled():
    """A config dict with ``enable`` False/absent is also a no-op."""
    assert build_dcd_state({"enable": False}, np.zeros((10, 3), dtype=np.int32), [5, 5, 5], ["a", "b", "c"], (2,), None, 0) is None
    assert build_dcd_state({}, np.zeros((10, 3), dtype=np.int32), [5, 5, 5], ["a", "b", "c"], (2,), None, 0) is None


def test_build_dcd_state_falls_back_to_none_on_init_failure(monkeypatch):
    """DCD is an opt-in accelerator: any exception constructing the state must fall back to ``None``
    (legacy path), never propagate."""
    import mlframe.feature_selection.filters._dynamic_cluster_discovery as dcd_mod

    def _raiser(**kwargs):
        """Stand in for ``make_dcd_state`` and always raise, to exercise the best-effort fallback."""
        raise RuntimeError("simulated DCD init failure")

    monkeypatch.setattr(dcd_mod, "make_dcd_state", _raiser)
    result = build_dcd_state({"enable": True}, np.zeros((10, 3), dtype=np.int32), [5, 5, 5], ["a", "b", "c"], (2,), None, 0)
    assert result is None
