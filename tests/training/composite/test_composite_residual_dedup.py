"""Regression coverage for ``residual_dedup_indices`` (DX12).

``residual_dedup_indices`` is production-wired into the CompositeCrossTarget
ensemble build (``core/_phase_composite_post_xt_ensemble/__init__.py:629``,
opt-in behind ``ct_ensemble_dedup_enabled``) but previously shipped with ZERO
tests. It greedily drops near-duplicate ensemble members by honest-OOF residual
correlation, keeping the lower-RMSE member of each redundant pair, and never
dropping below ``min_keep`` survivors.

This file pins the full contract so a future refactor of the keep-best-first
logic / the ``min_keep`` floor / the NaN-row guard cannot silently regress
ensemble member selection (which tilts the downstream NNLS stacking weights).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import residual_dedup_indices
from mlframe.training.composite.ensemble.stacking import (
    residual_dedup_indices as residual_dedup_indices_direct,
)


def _redundant_block(base: np.ndarray, rng: np.random.Generator, eps: float = 1e-6) -> np.ndarray:
    """A residual column ~identical to ``base`` (Pearson corr ~ 1.0)."""
    return base + eps * rng.normal(size=base.shape[0])


class TestResidualDedupKeepBestFirst:
    def test_redundant_pair_keeps_lower_rmse_member(self) -> None:
        """Two near-duplicate members + one independent: the redundant pair is
        collapsed to its LOWER-RMSE member; the independent member survives."""
        rng = np.random.default_rng(0)
        n = 500
        base = rng.normal(size=n)
        r0 = _redundant_block(base, rng)  # corr ~ 1 with r1
        r1 = _redundant_block(base, rng)
        r2 = rng.normal(size=n)  # independent
        resid = np.column_stack([r0, r1, r2])
        # member 1 is the stronger (lower-RMSE) of the redundant pair.
        oof = np.array([0.50, 0.40, 0.55])
        keep, drop = residual_dedup_indices(resid, oof, corr_threshold=0.95, min_keep=2)
        assert keep == [1, 2], f"expected stronger redundant member (1) + independent (2) kept, got {keep}"
        assert drop == [0], f"expected weaker redundant member (0) dropped, got {drop}"

    def test_lowest_rmse_member_always_kept(self) -> None:
        """The globally best member (lowest OOF RMSE) must survive no matter how
        redundant it is with the rest."""
        rng = np.random.default_rng(1)
        n = 600
        base = rng.normal(size=n)
        resid = np.column_stack([_redundant_block(base, rng) for _ in range(4)])
        # Make member 2 the clear best.
        oof = np.array([0.50, 0.45, 0.30, 0.55])
        keep, _ = residual_dedup_indices(resid, oof, corr_threshold=0.95, min_keep=2)
        assert 2 in keep, f"best (lowest-RMSE) member must always be kept, keep={keep}"

    def test_keep_drop_partition_range(self) -> None:
        """keep_idx and dropped_idx must partition range(K) with no overlap."""
        rng = np.random.default_rng(2)
        n = 400
        base = rng.normal(size=n)
        cols = [_redundant_block(base, rng), _redundant_block(base, rng), rng.normal(size=n), rng.normal(size=n)]
        resid = np.column_stack(cols)
        oof = np.array([0.5, 0.4, 0.6, 0.55])
        keep, drop = residual_dedup_indices(resid, oof, corr_threshold=0.95, min_keep=2)
        assert sorted(keep + drop) == list(range(resid.shape[1]))
        assert set(keep).isdisjoint(set(drop))


class TestResidualDedupMinKeepFloor:
    def test_floor_never_drops_below_min_keep(self) -> None:
        """All members redundant: dedup must still leave exactly ``min_keep``."""
        rng = np.random.default_rng(3)
        n = 700
        base = rng.normal(size=n)
        resid = np.column_stack([_redundant_block(base, rng) for _ in range(5)])
        oof = np.array([0.50, 0.30, 0.40, 0.60, 0.55])
        for floor in (2, 3, 4):
            keep, drop = residual_dedup_indices(resid, oof, corr_threshold=0.95, min_keep=floor)
            assert len(keep) == floor, f"min_keep={floor}: expected exactly {floor} kept, got {len(keep)}"
            assert len(keep) + len(drop) == resid.shape[1]
            # The strongest (lowest-RMSE = index 1) is among survivors at every floor.
            assert 1 in keep, f"floor={floor}: strongest member must survive, keep={keep}"

    def test_k_at_or_below_min_keep_short_circuits(self) -> None:
        """K <= min_keep keeps everything (a stack needs >= min_keep members)."""
        rng = np.random.default_rng(4)
        n = 300
        base = rng.normal(size=n)
        # Two perfectly redundant members but K == min_keep -> no drop.
        resid = np.column_stack([_redundant_block(base, rng), _redundant_block(base, rng)])
        oof = np.array([0.5, 0.4])
        keep, drop = residual_dedup_indices(resid, oof, corr_threshold=0.95, min_keep=2)
        assert keep == [0, 1]
        assert drop == []

    def test_single_member_kept(self) -> None:
        """K=1 (1-D residuals reshaped) is below the floor -> kept."""
        rng = np.random.default_rng(5)
        resid = rng.normal(size=200)  # 1-D -> reshaped to (200, 1)
        keep, drop = residual_dedup_indices(resid, np.array([0.5]), min_keep=2)
        assert keep == [0]
        assert drop == []


class TestResidualDedupNaNRowGuard:
    def test_too_few_finite_rows_keeps_all(self) -> None:
        """When jointly-finite rows < 3, correlation is undefined -> keep all
        (rather than silently correlating on a 1-2 row sample)."""
        # Each row has exactly one NaN, so NO row is jointly finite except the last.
        resid = np.array(
            [
                [np.nan, 1.0, 2.0],
                [1.0, np.nan, 2.0],
                [1.0, 2.0, np.nan],
                [1.0, 2.0, 3.0],
            ]
        )
        oof = np.array([0.5, 0.4, 0.6])
        keep, drop = residual_dedup_indices(resid, oof, min_keep=2)
        assert keep == [0, 1, 2]
        assert drop == []

    def test_nan_rows_masked_not_member_dropping(self) -> None:
        """A handful of NaN rows must be masked, NOT cause a whole member to be
        dropped. The redundant pair is still detected on the finite rows."""
        rng = np.random.default_rng(6)
        n = 500
        base = rng.normal(size=n)
        r0 = _redundant_block(base, rng)
        r1 = _redundant_block(base, rng)
        r2 = rng.normal(size=n)
        resid = np.column_stack([r0, r1, r2])
        # Scatter a few NaNs across different members/rows (still >> 3 finite rows).
        resid[5, 0] = np.nan
        resid[17, 1] = np.nan
        resid[42, 2] = np.nan
        oof = np.array([0.50, 0.40, 0.55])
        keep, drop = residual_dedup_indices(resid, oof, corr_threshold=0.95, min_keep=2)
        # Redundant pair still collapsed despite NaN rows; independent member kept.
        assert keep == [1, 2]
        assert drop == [0]

    def test_constant_member_not_dropped(self) -> None:
        """A zero-variance member yields NaN correlations; a NaN corr is treated
        as 'not redundant' so the constant member is not spuriously dropped."""
        rng = np.random.default_rng(7)
        n = 400
        const = np.full(n, 3.0)
        r1 = rng.normal(size=n)
        r2 = rng.normal(size=n)
        resid = np.column_stack([const, r1, r2])
        oof = np.array([0.6, 0.4, 0.5])
        keep, drop = residual_dedup_indices(resid, oof, corr_threshold=0.95, min_keep=2)
        # No member is redundant (corr with const is NaN) -> nothing dropped.
        assert drop == []
        assert keep == [0, 1, 2]


class TestResidualDedupContract:
    def test_oof_length_mismatch_raises(self) -> None:
        rng = np.random.default_rng(8)
        resid = rng.normal(size=(100, 3))
        with pytest.raises(ValueError, match="oof_rmses length"):
            residual_dedup_indices(resid, np.array([0.5, 0.4]))

    def test_no_redundancy_keeps_all(self) -> None:
        """Independent members below the correlation threshold are all kept."""
        rng = np.random.default_rng(9)
        n = 800
        resid = np.column_stack([rng.normal(size=n) for _ in range(4)])
        oof = np.array([0.5, 0.4, 0.6, 0.55])
        keep, drop = residual_dedup_indices(resid, oof, corr_threshold=0.95, min_keep=2)
        assert drop == []
        assert keep == [0, 1, 2, 3]

    def test_threshold_controls_aggressiveness(self) -> None:
        """A moderately-correlated pair is dropped at a low threshold and kept at
        a high one (the gate is genuinely threshold-driven)."""
        rng = np.random.default_rng(10)
        n = 2000
        a = rng.normal(size=n)
        b = 0.7 * a + np.sqrt(1 - 0.7**2) * rng.normal(size=n)  # corr(a,b) ~ 0.7
        c = rng.normal(size=n)
        resid = np.column_stack([a, b, c])
        oof = np.array([0.4, 0.5, 0.6])
        keep_lo, drop_lo = residual_dedup_indices(resid, oof, corr_threshold=0.5, min_keep=2)
        keep_hi, drop_hi = residual_dedup_indices(resid, oof, corr_threshold=0.95, min_keep=2)
        assert drop_lo == [1], f"corr~0.7 > 0.5 -> drop weaker of (a,b); got drop={drop_lo}"
        assert drop_hi == [], f"corr~0.7 < 0.95 -> keep all; got drop={drop_hi}"

    def test_returned_indices_are_python_int(self) -> None:
        """Returned indices index Python lists in production; they must be usable
        as such. We accept numpy integers (valid list indices) but require they
        are integral and within range."""
        rng = np.random.default_rng(11)
        n = 500
        base = rng.normal(size=n)
        resid = np.column_stack([_redundant_block(base, rng), _redundant_block(base, rng), rng.normal(size=n)])
        oof = np.array([0.5, 0.4, 0.6])
        keep, drop = residual_dedup_indices(resid, oof, corr_threshold=0.95, min_keep=2)
        for idx in keep + drop:
            assert int(idx) == idx
            assert 0 <= int(idx) < resid.shape[1]

    def test_public_and_direct_imports_are_same_function(self) -> None:
        """The function is re-exported from the composite package (the import the
        production caller uses) and lives in the stacking module."""
        assert residual_dedup_indices is residual_dedup_indices_direct


class TestResidualDedupProductionShape:
    def test_pred_minus_y_residual_dedup_matches_caller(self) -> None:
        """Mirror the production call: residual = oof_pred - y_holdout, oof_rmses
        from the same predictions. A duplicated component (identical predictions)
        must be deduped, keeping the better-RMSE original."""
        rng = np.random.default_rng(12)
        n = 600
        y = rng.normal(size=n)
        # Three genuine independent components (0,1,2) + a near-clone of comp 0 (3).
        p0 = y + 0.30 * rng.normal(size=n)
        p1 = y + 0.40 * rng.normal(size=n)
        p2 = y + 0.50 * rng.normal(size=n)
        p3 = p0 + 1e-6 * rng.normal(size=n)  # near-clone of p0
        preds = np.column_stack([p0, p1, p2, p3])
        resid = preds - y[:, None]
        oof_rmses = np.sqrt(np.mean(resid**2, axis=0))
        keep, drop = residual_dedup_indices(resid, oof_rmses, corr_threshold=0.95, min_keep=2)
        # The redundant pair is {0, 3}: exactly ONE survives -- the lower-RMSE one --
        # and the other is dropped. (We don't presume which clone wins; the contract
        # is 'keep the stronger of the redundant pair'.)
        redundant_pair = {0, 3}
        kept_of_pair = redundant_pair & set(int(i) for i in keep)
        dropped_of_pair = redundant_pair & set(int(i) for i in drop)
        assert len(kept_of_pair) == 1, f"exactly one of the clone pair must survive, keep={keep} drop={drop}"
        assert len(dropped_of_pair) == 1, f"exactly one of the clone pair must be dropped, keep={keep} drop={drop}"
        survivor = next(iter(kept_of_pair))
        dropped = next(iter(dropped_of_pair))
        assert oof_rmses[survivor] <= oof_rmses[dropped], "the lower-RMSE member of the redundant pair must survive"
        # The genuine independent components must always survive.
        assert {1, 2}.issubset(set(int(i) for i in keep)), "genuine independent components must survive"
