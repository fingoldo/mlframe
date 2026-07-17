"""Tests for ``forward_stepwise_multi_base`` (R10c follow-up OPEN-1; greedy forward-stepwise selection of additional base columns for linear_residual_multi)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import (
    _MULTI_BASE_DEFAULT_MAX_K,
    _MULTI_BASE_DEFAULT_MIN_MARGINAL_GAIN,
    forward_stepwise_multi_base,
)


class TestSeedBehavior:
    def test_empty_pool_returns_seeds_unchanged(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.normal(size=200)
        kept, diag = forward_stepwise_multi_base(y, candidate_bases={}, seed_bases=["b1"])
        assert kept == ["b1"]
        assert diag == []

    def test_seeds_not_re_added(self) -> None:
        """Seeds present in candidate_bases must not be re-added as duplicate candidates by the greedy loop."""
        rng = np.random.default_rng(1)
        n = 300
        b1 = rng.normal(loc=10, scale=2, size=n)
        b2 = rng.normal(size=n)
        y = b1 + b2 + rng.normal(scale=0.1, size=n)
        kept, diag = forward_stepwise_multi_base(
            y,
            candidate_bases={"b1": b1, "b2": b2},
            seed_bases=["b1"],
        )
        # b1 in seeds + b2 greedily added. No duplicate b1.
        assert "b1" in kept
        assert kept.count("b1") == 1
        # b2 should be added (gives huge gain).
        assert "b2" in kept

    def test_seed_missing_from_candidates_raises(self) -> None:
        """API contract: seeds must appear in candidate_bases when candidate_bases is non-empty."""
        rng = np.random.default_rng(10)
        n = 200
        b1 = rng.normal(size=n)
        y = b1 + rng.normal(scale=0.1, size=n)
        with pytest.raises(ValueError, match="must appear in candidate_bases"):
            forward_stepwise_multi_base(
                y,
                candidate_bases={"b1": b1},
                seed_bases=["missing_seed"],
            )


class TestGreedyAddition:
    def test_orthogonal_base_added(self) -> None:
        """When b2 carries strong orthogonal signal, the helper picks it up after b1."""
        rng = np.random.default_rng(2)
        n = 500
        b1 = rng.normal(loc=10, scale=2, size=n)
        b2 = rng.normal(loc=0, scale=3, size=n)
        b3_noise = rng.normal(size=n)  # pure noise -> should be rejected
        y = 0.9 * b1 + 0.5 * b2 + rng.normal(scale=0.2, size=n)
        kept, diag = forward_stepwise_multi_base(
            y,
            candidate_bases={"b1": b1, "b2": b2, "b3_noise": b3_noise},
            seed_bases=["b1"],
            max_k=3,
        )
        assert "b1" in kept
        assert "b2" in kept
        # b3 either rejected (gain below threshold) or added but with low gain. Acceptable either way; what matters: b2 made it in.

    def test_pure_noise_candidate_rejected(self) -> None:
        """Noise-only candidate doesn't improve RMSE; not added when min_marginal_gain > 0."""
        rng = np.random.default_rng(3)
        n = 500
        b1 = rng.normal(loc=10, scale=2, size=n)
        y = 0.9 * b1 + rng.normal(scale=0.2, size=n)
        noise_bases = {f"noise_{i}": rng.normal(size=n) for i in range(5)}
        candidates = {"b1": b1, **noise_bases}
        kept, diag = forward_stepwise_multi_base(
            y,
            candidates,
            seed_bases=["b1"],
            max_k=5,
            min_marginal_rmse_gain=0.02,
        )
        # b1 in kept; noise bases should NOT add (gain < 2%).
        assert "b1" in kept
        # At most maybe 1 noise base sneaks in if random correlation just barely beats 2%; assert <= 2 noises (room for chance).
        noise_in_kept = [k for k in kept if k.startswith("noise_")]
        assert len(noise_in_kept) <= 2

    def test_max_k_respected(self) -> None:
        rng = np.random.default_rng(4)
        n = 400
        candidates = {f"b{i}": rng.normal(size=n) for i in range(10)}
        y = sum(candidates.values()) + rng.normal(scale=0.1, size=n)
        kept, diag = forward_stepwise_multi_base(
            y,
            candidates,
            max_k=3,
            min_marginal_rmse_gain=0.0,
        )
        assert len(kept) <= 3


class TestDiagnostics:
    def test_each_step_records_gain_and_acceptance(self) -> None:
        rng = np.random.default_rng(5)
        n = 300
        b1 = rng.normal(size=n)
        b2 = rng.normal(size=n)
        y = b1 + b2 + rng.normal(scale=0.1, size=n)
        _, diag = forward_stepwise_multi_base(
            y,
            candidate_bases={"b1": b1, "b2": b2},
            seed_bases=["b1"],
            max_k=3,
        )
        # Each entry has the expected fields.
        for entry in diag:
            assert {"step", "candidate_added", "rmse_before", "rmse_after", "marginal_gain", "accepted"}.issubset(entry.keys())

    def test_no_acceptance_stops_loop_early(self) -> None:
        """First candidate fails the gate -> loop stops; no further candidates tried."""
        rng = np.random.default_rng(6)
        n = 500
        b1 = rng.normal(loc=10, scale=2, size=n)
        # Both candidates are noise; gate at 50% gain (impossible) -> neither accepted.
        noise_a = rng.normal(size=n)
        noise_b = rng.normal(size=n)
        y = 0.9 * b1 + rng.normal(scale=0.2, size=n)
        kept, diag = forward_stepwise_multi_base(
            y,
            candidate_bases={"b1": b1, "noise_a": noise_a, "noise_b": noise_b},
            seed_bases=["b1"],
            max_k=3,
            min_marginal_rmse_gain=0.5,  # 50% gain threshold
        )
        # Only b1 kept (noise gates fail).
        assert kept == ["b1"]
        # Diagnostics records exactly ONE rejected candidate (the loop stops on first rejection).
        assert len(diag) == 1
        assert diag[0]["accepted"] is False


class TestDefaultsLockedIn:
    def test_default_max_k(self) -> None:
        assert _MULTI_BASE_DEFAULT_MAX_K == 3

    def test_default_min_marginal_gain(self) -> None:
        assert _MULTI_BASE_DEFAULT_MIN_MARGINAL_GAIN == 0.02


# ===========================================================================
# Biz_value: on a 2-base DGP, helper finds the second base
# ===========================================================================


class TestBizValue:
    def test_finds_second_dgp_base(self) -> None:
        """``y = 0.9*b1 + 0.5*b2 + eps`` -- starting from seed=[b1], helper greedy-adds b2 (large RMSE gain) but skips pure noise."""
        rng = np.random.default_rng(0)
        n = 1500
        b1 = rng.normal(loc=10, scale=2, size=n)
        b2 = rng.normal(loc=0, scale=3, size=n)  # orthogonal to b1
        noise = rng.normal(size=n)
        y = 0.9 * b1 + 0.5 * b2 + rng.normal(scale=0.1, size=n)
        kept, diag = forward_stepwise_multi_base(
            y,
            candidate_bases={"b1": b1, "b2": b2, "noise": noise},
            seed_bases=["b1"],
            max_k=3,
            min_marginal_rmse_gain=0.02,
        )
        assert "b2" in kept
        # First accepted step should be b2 with significant gain.
        accepted_steps = [d for d in diag if d["accepted"]]
        assert accepted_steps, "expected at least one accepted step (b2)"
        assert accepted_steps[0]["candidate_added"] == "b2"
        assert accepted_steps[0]["marginal_gain"] > 0.1
