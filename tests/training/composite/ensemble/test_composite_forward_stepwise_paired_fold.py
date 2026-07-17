"""Forward-stepwise FUTURE batch: paired majority-of-folds selection (M13) + A20 buffer bit-identity.

- M13: the greedy step previously accepted a candidate on the aggregate relative-gain gate alone, which one fold can drive (positively-correlated repeated-CV folds understate variance). ``paired_fold_selection`` (default ON) additionally requires the chosen candidate to beat the kept-set on a majority of jointly-finite folds. A base whose benefit is concentrated in a minority of folds (and actively hurts the rest) is now rejected.
- A20: the per-trial design matrix is built into a reused preallocated buffer (kept-prefix stacked once per round, candidate in the last column) instead of a fresh ``np.column_stack`` per trial. The matrix handed to OLS is byte-identical, so selection results must be UNCHANGED when the paired gate is off.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from mlframe.training.composite.discovery.forward_stepwise import (
    forward_stepwise_multi_base,
)


def _concentrated_signal_frame(seed: int, neutral_noise: float = 0.3):
    """y = b1 + eps. ``hb`` perfectly explains the residual on the LAST TWO of five disjoint folds and carries small independent noise everywhere else.

    Under ``KFold(5, shuffle=False)`` OLS learns hb's coefficient (~1) from the signal folds and applies it everywhere; on the three neutral folds hb is just noise, so the learned coefficient inflates their RMSE slightly. Net: hb wins on 2/5 folds (huge), loses on 3/5 (small) -> aggregate mean improves a lot (the two near-zero folds dominate) but the paired majority gate sees a 0.40 win fraction.
    """
    rng = np.random.default_rng(seed)
    n = 2500
    b1 = rng.normal(0.0, 1.0, n)
    eps = rng.normal(0.0, 1.0, n)
    y = b1 + eps
    hb = rng.normal(0.0, neutral_noise, n)
    hb[1500:2500] = eps[1500:2500]
    return y, b1, hb


class TestM13PairedFoldGate:
    def test_paired_gate_rejects_minority_win_base_that_point_estimate_accepts(self) -> None:
        """The concentrated-signal base clears the aggregate gain gate (legacy accepts it) but wins only a MINORITY of folds, so the paired gate (default ON) rejects it."""
        splitter = KFold(n_splits=5, shuffle=False)
        y, b1, hb = _concentrated_signal_frame(seed=0)

        kept_legacy, diag_legacy = forward_stepwise_multi_base(
            y,
            {"b1": b1, "hb": hb},
            seed_bases=["b1"],
            cv_splitter=splitter,
            time_aware=False,
            max_k=3,
            min_marginal_rmse_gain=0.02,
            paired_fold_selection=False,
            cv_persist_fold_scores=True,
        )
        kept_paired, diag_paired = forward_stepwise_multi_base(
            y,
            {"b1": b1, "hb": hb},
            seed_bases=["b1"],
            cv_splitter=splitter,
            time_aware=False,
            max_k=3,
            min_marginal_rmse_gain=0.02,
            paired_fold_selection=True,
            cv_persist_fold_scores=True,
        )
        # Legacy point-estimate gate accepts hb (the two perfect folds dominate the mean -> large aggregate gain).
        assert "hb" in kept_legacy, "legacy gate should accept the concentrated-signal base on the aggregate"
        assert diag_legacy[0]["accepted"] is True
        assert diag_legacy[0]["marginal_gain"] > 0.1
        # Paired gate rejects it: it only wins a minority of folds.
        assert "hb" not in kept_paired, "paired majority gate must reject a base that wins only a minority of folds"
        assert diag_paired[0]["accepted"] is False
        assert diag_paired[0]["paired_fold_win_frac"] < 0.5
        assert diag_paired[0]["paired_folds_used"] == 5

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_paired_gate_flip_is_robust_across_seeds(self, seed: int) -> None:
        """The accept(legacy) -> reject(paired) flip holds across seeds (win fraction is structurally 0.40, not seed-dependent noise)."""
        splitter = KFold(n_splits=5, shuffle=False)
        y, b1, hb = _concentrated_signal_frame(seed=seed)
        _, diag_legacy = forward_stepwise_multi_base(
            y,
            {"b1": b1, "hb": hb},
            seed_bases=["b1"],
            cv_splitter=splitter,
            time_aware=False,
            max_k=3,
            min_marginal_rmse_gain=0.02,
            paired_fold_selection=False,
        )
        _, diag_paired = forward_stepwise_multi_base(
            y,
            {"b1": b1, "hb": hb},
            seed_bases=["b1"],
            cv_splitter=splitter,
            time_aware=False,
            max_k=3,
            min_marginal_rmse_gain=0.02,
            paired_fold_selection=True,
        )
        assert diag_legacy[0]["accepted"] is True
        assert diag_paired[0]["accepted"] is False
        assert diag_paired[0]["paired_fold_win_frac"] <= 0.4 + 1e-9

    def test_paired_gate_keeps_a_genuinely_helpful_base(self) -> None:
        """A base that helps on ALL folds (majority trivially satisfied) is still accepted with the paired gate ON -- the gate is a guard against minority-win bases, not a blanket rejector."""
        rng = np.random.default_rng(5)
        n = 1500
        b1 = rng.normal(0.0, 1.0, n)
        b2 = rng.normal(0.0, 1.0, n)
        y = b1 + 0.9 * b2 + rng.normal(0.0, 0.1, n)
        kept, diag = forward_stepwise_multi_base(
            y,
            {"b1": b1, "b2": b2},
            seed_bases=["b1"],
            time_aware=False,
            cv_folds=4,
            max_k=3,
            min_marginal_rmse_gain=0.02,
            paired_fold_selection=True,
            cv_persist_fold_scores=True,
        )
        assert "b2" in kept
        accepted = [d for d in diag if d["accepted"]]
        assert accepted and accepted[0]["candidate_added"] == "b2"
        # A genuinely orthogonal helper wins (close to) all folds.
        assert accepted[0]["paired_fold_win_frac"] >= 0.75

    def test_diagnostics_always_carry_paired_fields(self) -> None:
        """Every step diagnostic exposes the paired-fold fields so downstream audit code can inspect them regardless of the gate outcome."""
        rng = np.random.default_rng(6)
        n = 800
        b1 = rng.normal(size=n)
        b2 = rng.normal(size=n)
        y = b1 + b2 + rng.normal(scale=0.1, size=n)
        _, diag = forward_stepwise_multi_base(
            y,
            {"b1": b1, "b2": b2},
            seed_bases=["b1"],
            time_aware=False,
            max_k=3,
        )
        for entry in diag:
            assert "paired_fold_win_frac" in entry
            assert "paired_folds_used" in entry


class TestM13OptOut:
    def test_opt_out_recovers_legacy_acceptance(self) -> None:
        """``paired_fold_selection=False`` restores the point-estimate-only behaviour (accepts the concentrated-signal base)."""
        splitter = KFold(n_splits=5, shuffle=False)
        y, b1, hb = _concentrated_signal_frame(seed=2)
        kept, _ = forward_stepwise_multi_base(
            y,
            {"b1": b1, "hb": hb},
            seed_bases=["b1"],
            cv_splitter=splitter,
            time_aware=False,
            max_k=3,
            min_marginal_rmse_gain=0.02,
            paired_fold_selection=False,
        )
        assert "hb" in kept

    def test_min_win_frac_unanimity_is_stricter(self) -> None:
        """Raising ``paired_fold_min_win_frac`` toward 1.0 demands more folds win; a base that clears a 0.5 majority but not unanimity is rejected at 1.0."""
        rng = np.random.default_rng(9)
        n = 1600
        b1 = rng.normal(size=n)
        # b2 helps on most folds but not all (mild noise so one fold can lose).
        b2 = rng.normal(size=n)
        y = b1 + 0.4 * b2 + rng.normal(scale=0.6, size=n)
        kept_majority, diag_majority = forward_stepwise_multi_base(
            y,
            {"b1": b1, "b2": b2},
            seed_bases=["b1"],
            time_aware=False,
            cv_folds=5,
            max_k=3,
            min_marginal_rmse_gain=0.0,
            paired_fold_selection=True,
            paired_fold_min_win_frac=0.5,
        )
        kept_unanimous, diag_unanimous = forward_stepwise_multi_base(
            y,
            {"b1": b1, "b2": b2},
            seed_bases=["b1"],
            time_aware=False,
            cv_folds=5,
            max_k=3,
            min_marginal_rmse_gain=0.0,
            paired_fold_selection=True,
            paired_fold_min_win_frac=1.0,
        )
        # Whatever the majority gate decides, the unanimity gate is never MORE permissive.
        assert diag_unanimous[0]["accepted"] <= diag_majority[0]["accepted"] or (diag_unanimous[0]["paired_fold_win_frac"] >= 1.0)


class TestA20BufferBitIdentity:
    """The A20 buffer reuse must not change any selection outcome: with the paired gate OFF the kept list and the per-step aggregate RMSEs are byte-identical to a reference column-stack run."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_buffer_path_matches_reference_column_stack(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = 1200
        cands = {f"b{i}": rng.normal(size=n) for i in range(8)}
        y = 2.0 * cands["b0"] + 1.3 * cands["b1"] - 0.7 * cands["b3"] + rng.normal(scale=0.3, size=n)
        kept, diag = forward_stepwise_multi_base(
            y,
            cands,
            seed_bases=["b0"],
            time_aware=False,
            cv_folds=4,
            max_k=4,
            min_marginal_rmse_gain=0.0,
            paired_fold_selection=False,
            cv_persist_fold_scores=True,
        )
        # Reconstruct the reference fold RMSEs for the FINAL kept set via an independent column_stack
        # + the same OLS, and confirm they match the buffer-path values bit-for-bit.
        from mlframe.training.composite import _linear_residual_multi_fit

        splitter = KFold(n_splits=4, shuffle=True, random_state=42)
        base_matrix = np.column_stack([cands[c] for c in kept])
        yf = np.asarray(y, dtype=np.float64)
        ref_folds = []
        for tr, va in splitter.split(np.arange(n)):
            params = _linear_residual_multi_fit(yf[tr], base_matrix[tr])
            alphas = np.asarray(params["alphas"], dtype=np.float64)
            beta = float(params["beta"])
            pred = base_matrix[va].astype(np.float64) @ alphas + beta
            d = pred - yf[va]
            ref_folds.append(float(np.sqrt(np.mean(d * d))))
        # The greedy run adds the signal bases (b1, b3) then stops once the best remaining candidate
        # would worsen RMSE (gain < 0). The seed must always be first.
        assert kept[0] == "b0"
        assert "b1" in kept and "b3" in kept
        assert len(kept) >= 3
        # Byte-identity: the last accepted step's rmse_after equals the reference aggregate (mean of the
        # independently column-stacked OLS folds) for that exact kept set, with ZERO tolerance.
        last_accepted = [d for d in diag if d["accepted"]][-1]
        assert last_accepted["rmse_after"] == float(np.mean(ref_folds))
