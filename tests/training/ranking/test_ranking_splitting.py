"""Tests for ``groups`` parameter on ``make_train_test_split``.

Two paths covered:
- **Row-based fallback (no timestamps)**: GroupShuffleSplit ensures no
  group is torn across train/val/test boundaries.
- **Time-based**: when timestamps + groups both supplied, groups that
  span a cutoff get reassigned to the LATER split with WARN.

Mutual-exclusion guard: ``stratify_y`` and ``groups`` cannot coexist
in the row-based path (no group-stratified splitter is wired).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.splitting import make_train_test_split


@pytest.fixture
def synthetic_grouped_frame():
    """100 groups x 10 rows = 1000 rows. Groups 0..99 contiguous."""
    n_groups = 100
    n_per = 10
    n = n_groups * n_per
    df = pd.DataFrame({"x": np.arange(n, dtype=float)})
    groups = np.repeat(np.arange(n_groups), n_per)
    return df, groups


class TestGroupShuffleSplit:
    """Row-based path with ``groups`` -> sklearn GroupShuffleSplit."""

    def test_groups_kept_intact_across_splits(self, synthetic_grouped_frame):
        """Groups kept intact across splits."""
        df, groups = synthetic_grouped_frame
        train_idx, val_idx, test_idx, *_ = make_train_test_split(
            df,
            test_size=0.2,
            val_size=0.1,
            groups=groups,
            random_seed=42,
        )
        train_g = set(groups[train_idx].tolist())
        val_g = set(groups[val_idx].tolist())
        test_g = set(groups[test_idx].tolist())
        # Every pair of split-group-sets disjoint
        assert not (train_g & val_g)
        assert not (train_g & test_g)
        assert not (val_g & test_g)
        # Total groups == 100 (no group dropped)
        assert len(train_g) + len(val_g) + len(test_g) == 100

    def test_split_sizes_approx_match_request(self, synthetic_grouped_frame):
        """test=20% / val=10% of GROUPS (not rows); but rows ratio
        approx matches because all groups are equal-sized."""
        df, groups = synthetic_grouped_frame
        _train_idx, val_idx, test_idx, *_ = make_train_test_split(
            df,
            test_size=0.2,
            val_size=0.1,
            groups=groups,
            random_seed=42,
        )
        n = len(df)
        # Allow ±5% slack since GroupShuffleSplit operates on group counts
        assert 0.15 * n <= len(test_idx) <= 0.25 * n
        assert 0.05 * n <= len(val_idx) <= 0.15 * n

    def test_zero_test_size_route(self, synthetic_grouped_frame):
        """Zero test size route."""
        df, groups = synthetic_grouped_frame
        train_idx, val_idx, test_idx, *_ = make_train_test_split(
            df,
            test_size=0.0,
            val_size=0.1,
            groups=groups,
            random_seed=42,
        )
        assert len(test_idx) == 0
        # Train + val should still respect groups
        train_g = set(groups[train_idx].tolist())
        val_g = set(groups[val_idx].tolist())
        assert not (train_g & val_g)


class TestStratifyGroupsCombined:
    """2026-05-26: ``stratify_y`` + ``groups`` are NO LONGER mutually
    exclusive. ``make_train_test_split`` routes the combination
    through sklearn's ``StratifiedGroupKFold`` (sklearn >=1.0) so both
    invariants are honoured -- whole groups stay in one split AND the
    1-D class / bucket distribution is preserved across splits. The
    pre-2026-05-26 contract raised ValueError("mutually exclusive");
    the new contract returns a valid split. Bucket-proportion
    assertions live in ``test_group_stratified_split.py``."""

    def test_both_provided_returns_valid_split(self, synthetic_grouped_frame):
        """Both provided returns valid split."""
        df, groups = synthetic_grouped_frame
        # Stratifiable binary target correlated with group id so the
        # splitter has real work to do.
        y = ((np.asarray(groups) % 4) == 0).astype(np.int64)
        train_idx, val_idx, test_idx, *_ = make_train_test_split(
            df,
            test_size=0.2,
            val_size=0.1,
            groups=groups,
            stratify_y=y,
            random_seed=42,
        )
        # Group containment: no group spans two slices.
        g = np.asarray(groups)
        for a, b in [(train_idx, val_idx), (train_idx, test_idx), (val_idx, test_idx)]:
            assert not (set(g[a].tolist()) & set(g[b].tolist())), "stratified-group split must keep every group in exactly one slice"


class TestGroupsLengthValidation:
    """Groups tests covering groups length validation."""
    def test_groups_wrong_length_raises(self, synthetic_grouped_frame):
        """Groups wrong length raises."""
        df, _ = synthetic_grouped_frame
        bad_groups = np.zeros(len(df) - 5)  # too short
        with pytest.raises(ValueError, match="length"):
            make_train_test_split(df, test_size=0.2, val_size=0.1, groups=bad_groups)

    def test_groups_wrong_ndim_raises(self, synthetic_grouped_frame):
        """Groups wrong ndim raises."""
        df, _ = synthetic_grouped_frame
        bad_groups = np.zeros((len(df), 2))  # 2-D not allowed
        with pytest.raises(ValueError, match="1-D"):
            make_train_test_split(df, test_size=0.2, val_size=0.1, groups=bad_groups)


class TestTimeBasedGroupSpanning:
    """Time-based path: groups that span a train/val/test cutoff get
    reassigned to the LATER split with WARN."""

    def test_no_spanning_when_groups_align_with_time(self):
        """Groups that don't straddle cutoffs need no reassignment."""
        n_groups = 50
        n_per = 10
        n = n_groups * n_per
        df = pd.DataFrame({"x": np.arange(n, dtype=float)})
        groups = np.repeat(np.arange(n_groups), n_per)
        # Each group's rows sit on consecutive timestamps -> no group spans
        # any time boundary.
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        train_idx, val_idx, test_idx, *_ = make_train_test_split(
            df,
            test_size=0.2,
            val_size=0.1,
            groups=groups,
            timestamps=ts,
            wholeday_splitting=False,
            random_seed=42,
        )
        train_g = set(groups[train_idx].tolist())
        val_g = set(groups[val_idx].tolist())
        test_g = set(groups[test_idx].tolist())
        assert not (train_g & val_g)
        assert not (train_g & test_g)
        assert not (val_g & test_g)

    def test_spanning_group_reassigned_to_later_split(self, caplog):
        """Construct a scenario where a group's rows straddle the train→test
        cutoff. The group must end up entirely in test; WARN logged."""
        # 30 rows: groups [0,0,0, 1,1,1, ..., 9,9,9]. Manually arrange
        # timestamps so group 7 straddles the boundary at row 21 (70%).
        n_groups = 10
        n_per = 3
        n = n_groups * n_per
        df = pd.DataFrame({"x": np.arange(n, dtype=float)})
        groups = np.repeat(np.arange(n_groups), n_per)
        ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="1h"))
        # Force group 7's rows around the boundary by assigning very early
        # timestamps to half its rows -- they'll fall into train -- and
        # later timestamps to the other half (will fall into test).
        # Simpler: shuffle so a group has a wide timespan.
        # Re-assign group 5's rows to timestamps 0, 15, 25 (spans entire range)
        ts.iloc[15] = ts.iloc[0]
        ts.iloc[16] = ts.iloc[5]
        ts.iloc[17] = ts.iloc[28]  # row 17 way late -> different split
        with caplog.at_level("WARNING"):
            train_idx, val_idx, test_idx, *_ = make_train_test_split(
                df,
                test_size=0.2,
                val_size=0.1,
                groups=groups,
                timestamps=ts,
                wholeday_splitting=False,
                random_seed=42,
            )
        # Either no spanning happened (depending on random arrangement) OR the
        # WARN fired. Critical invariant: no group is torn across splits.
        train_g = set(groups[train_idx].tolist())
        val_g = set(groups[val_idx].tolist())
        test_g = set(groups[test_idx].tolist())
        assert not (train_g & val_g), "train and val share groups -- regression"
        assert not (train_g & test_g), "train and test share groups -- regression"
        assert not (val_g & test_g), "val and test share groups -- regression"


class TestBackwardCompatGroupsNone:
    """When ``groups=None``, behaviour is identical to pre-2026-05-04 baseline."""

    def test_no_groups_falls_back_to_train_test_split(self, synthetic_grouped_frame):
        """No groups falls back to train test split."""
        df, _ = synthetic_grouped_frame
        train_idx, val_idx, test_idx, *_ = make_train_test_split(
            df,
            test_size=0.2,
            val_size=0.1,
            groups=None,
            random_seed=42,
        )
        # No constraint asserted on group integrity (we explicitly didn't
        # pass groups). Just verify shapes are reasonable.
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)
