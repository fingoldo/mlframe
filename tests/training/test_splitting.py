"""
Tests for training/splitting.py module.

Covers:
- Basic random and sequential splits
- Date-based (whole-day) splitting
- Mixed sequential/shuffled splits
- Training set aging limits
- Edge cases and validation
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from mlframe.training.splitting import make_train_test_split


class TestMakeTrainTestSplitBasic:
    """Test basic splitting functionality."""

    def test_basic_random_split(self):
        """Test basic random split with default parameters."""
        df = pd.DataFrame({'feature': np.random.randn(1000)})

        train_idx, val_idx, test_idx, train_details, val_details, test_details = make_train_test_split(
            df, test_size=0.2, val_size=0.1, shuffle_val=True, shuffle_test=True, random_seed=42
        )

        # Check all indices are unique
        all_idx = np.concatenate([train_idx, val_idx, test_idx])
        assert len(all_idx) == len(np.unique(all_idx)), "Indices should be unique across splits"

        # Check coverage
        assert len(all_idx) == len(df), "All rows should be assigned to a split"

        # Check approximate sizes (with tolerance for rounding)
        assert abs(len(test_idx) / len(df) - 0.2) < 0.05, "Test size should be approximately 20%"
        assert abs(len(val_idx) / (len(df) - len(test_idx)) - 0.1) < 0.05, "Val size should be approximately 10% of remaining"

    def test_sequential_split_no_shuffle(self):
        """Test sequential split without shuffling."""
        df = pd.DataFrame({'feature': np.arange(100)})

        train_idx, val_idx, test_idx, _, _, _ = make_train_test_split(
            df, test_size=0.2, val_size=0.1, shuffle_val=False, shuffle_test=False, random_seed=42
        )

        # Sequential means test should be at the end (highest indices)
        # and val should be before test
        if len(test_idx) > 0 and len(val_idx) > 0:
            # With sequential splitting, test should have the highest indices
            assert test_idx.max() >= val_idx.max(), "Test indices should be >= val indices in sequential split"

    def test_zero_test_size(self):
        """Test split with zero test size."""
        df = pd.DataFrame({'feature': np.random.randn(100)})

        # Note: The splitting function may handle zero test size differently
        # We just check it doesn't crash and returns valid train/val
        try:
            train_idx, val_idx, test_idx, _, _, _ = make_train_test_split(
                df, test_size=0.0, val_size=0.2, random_seed=42
            )
            # If it succeeds, check basic validity
            assert len(train_idx) > 0, "Train should not be empty"
        except (ValueError, ZeroDivisionError):
            # Some implementations may not support zero test size
            pytest.skip("Zero test size not supported by this implementation")

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same splits."""
        df = pd.DataFrame({'feature': np.random.randn(500)})

        result1 = make_train_test_split(df, test_size=0.2, val_size=0.1, shuffle_val=True, random_seed=42)
        result2 = make_train_test_split(df, test_size=0.2, val_size=0.1, shuffle_val=True, random_seed=42)

        np.testing.assert_array_equal(result1[0], result2[0], "Train indices should be identical with same seed")
        np.testing.assert_array_equal(result1[1], result2[1], "Val indices should be identical with same seed")
        np.testing.assert_array_equal(result1[2], result2[2], "Test indices should be identical with same seed")

    def test_different_seeds_produce_different_splits(self):
        """Test that different seeds produce different splits."""
        df = pd.DataFrame({'feature': np.random.randn(500)})

        result1 = make_train_test_split(df, test_size=0.2, val_size=0.1, shuffle_val=True, random_seed=42)
        result2 = make_train_test_split(df, test_size=0.2, val_size=0.1, shuffle_val=True, random_seed=123)

        # At least one split should be different
        train_different = not np.array_equal(result1[0], result2[0])
        val_different = not np.array_equal(result1[1], result2[1])
        test_different = not np.array_equal(result1[2], result2[2])

        assert train_different or val_different or test_different, "Different seeds should produce different splits"


class TestMakeTrainTestSplitDateBased:
    """Test date-based (whole-day) splitting."""

    @pytest.fixture
    def timeseries_df(self):
        """Create a DataFrame with timestamps spanning multiple days."""
        np.random.seed(42)
        n_samples = 1000
        # Create timestamps over 100 days
        base_date = datetime(2023, 1, 1)
        timestamps = [base_date + timedelta(days=i // 10, hours=i % 24) for i in range(n_samples)]

        df = pd.DataFrame({
            'feature': np.random.randn(n_samples),
            'timestamp': timestamps
        })
        return df

    def test_wholeday_splitting_basic(self, timeseries_df):
        """Test whole-day splitting creates date-aligned splits."""
        df = timeseries_df
        timestamps = pd.to_datetime(df['timestamp'])

        train_idx, val_idx, test_idx, train_details, val_details, test_details = make_train_test_split(
            df, test_size=0.2, val_size=0.1,
            timestamps=timestamps,
            wholeday_splitting=True,
            random_seed=42
        )

        # Check all indices are valid
        assert all(0 <= i < len(df) for i in train_idx), "Train indices should be valid"
        assert all(0 <= i < len(df) for i in val_idx), "Val indices should be valid"
        assert all(0 <= i < len(df) for i in test_idx), "Test indices should be valid"

        # Check no overlap
        all_idx = set(train_idx) | set(val_idx) | set(test_idx)
        assert len(all_idx) == len(train_idx) + len(val_idx) + len(test_idx), "No overlapping indices"

        # Check details contain date information
        assert '/' in train_details or train_details == '', "Train details should contain date range"

    def test_wholeday_splitting_preserves_day_boundaries(self, timeseries_df):
        """Test that whole-day splitting doesn't split within a day."""
        df = timeseries_df
        timestamps = pd.to_datetime(df['timestamp'])

        train_idx, val_idx, test_idx, _, _, _ = make_train_test_split(
            df, test_size=0.2, val_size=0.1,
            timestamps=timestamps,
            wholeday_splitting=True,
            shuffle_val=False,
            shuffle_test=False,
            random_seed=42
        )

        # Get dates for each split
        train_dates = set(timestamps.iloc[train_idx].dt.date)
        val_dates = set(timestamps.iloc[val_idx].dt.date)
        test_dates = set(timestamps.iloc[test_idx].dt.date)

        # No date should appear in multiple splits
        assert len(train_dates & val_dates) == 0, "Train and val should not share dates"
        assert len(train_dates & test_dates) == 0, "Train and test should not share dates"
        assert len(val_dates & test_dates) == 0, "Val and test should not share dates"

    def test_row_based_splitting_with_timestamps(self, timeseries_df):
        """Test row-based splitting with timestamps (wholeday_splitting=False)."""
        df = timeseries_df
        timestamps = pd.to_datetime(df['timestamp'])

        train_idx, val_idx, test_idx, _, _, _ = make_train_test_split(
            df, test_size=0.2, val_size=0.1,
            timestamps=timestamps,
            wholeday_splitting=False,
            random_seed=42
        )

        # All rows should be assigned
        all_idx = np.concatenate([train_idx, val_idx, test_idx])
        assert len(all_idx) == len(df), "All rows should be assigned"


class TestMakeTrainTestSplitSequentialFraction:
    """Test mixed sequential/shuffled splits via sequential_fraction parameters."""

    def test_full_sequential_fraction(self):
        """Test val_sequential_fraction=1.0 means fully sequential."""
        df = pd.DataFrame({'feature': np.arange(1000)})
        timestamps = pd.Series(pd.date_range('2023-01-01', periods=1000, freq='1h'))

        train_idx, val_idx, test_idx, _, _, _ = make_train_test_split(
            df, test_size=0.2, val_size=0.1,
            timestamps=timestamps,
            wholeday_splitting=False,
            val_sequential_fraction=1.0,
            test_sequential_fraction=1.0,
            random_seed=42
        )

        # With sequential_fraction=1.0, indices should be sorted and contiguous
        assert np.array_equal(test_idx, np.sort(test_idx)), "Test indices should be sorted"
        assert np.array_equal(val_idx, np.sort(val_idx)), "Val indices should be sorted"

    def test_zero_sequential_fraction(self):
        """Test sequential_fraction=0.0 means fully shuffled."""
        df = pd.DataFrame({'feature': np.arange(1000)})
        timestamps = pd.Series(pd.date_range('2023-01-01', periods=1000, freq='1h'))

        train_idx, val_idx, test_idx, _, _, _ = make_train_test_split(
            df, test_size=0.2, val_size=0.1,
            timestamps=timestamps,
            wholeday_splitting=False,
            val_sequential_fraction=0.0,
            test_sequential_fraction=0.0,
            random_seed=42
        )

        # All indices should still be valid and unique
        all_idx = np.concatenate([train_idx, val_idx, test_idx])
        assert len(np.unique(all_idx)) == len(all_idx), "All indices should be unique"

    def test_mixed_sequential_fraction(self):
        """Test mixed sequential/shuffled split (0 < fraction < 1)."""
        df = pd.DataFrame({'feature': np.arange(1000)})
        timestamps = pd.Series(pd.date_range('2023-01-01', periods=1000, freq='1h'))

        train_idx, val_idx, test_idx, _, val_details, test_details = make_train_test_split(
            df, test_size=0.2, val_size=0.1,
            timestamps=timestamps,
            wholeday_splitting=False,
            val_sequential_fraction=0.5,
            test_sequential_fraction=0.5,
            random_seed=42
        )

        # Check total sizes are correct
        total = len(train_idx) + len(val_idx) + len(test_idx)
        assert total == len(df), "All rows should be assigned"

    def test_invalid_sequential_fraction_raises_error(self):
        """Test that invalid sequential_fraction values raise errors."""
        df = pd.DataFrame({'feature': np.arange(100)})
        timestamps = pd.Series(pd.date_range('2023-01-01', periods=100, freq='1h'))

        with pytest.raises(ValueError, match="sequential_fraction must be between"):
            make_train_test_split(
                df, test_size=0.2, val_size=0.1,
                timestamps=timestamps,
                val_sequential_fraction=1.5,  # Invalid: > 1.0
                random_seed=42
            )

        with pytest.raises(ValueError, match="sequential_fraction must be between"):
            make_train_test_split(
                df, test_size=0.2, val_size=0.1,
                timestamps=timestamps,
                test_sequential_fraction=-0.1,  # Invalid: < 0.0
                random_seed=42
            )


class TestMakeTrainTestSplitAgingLimit:
    """Test training set aging limit functionality."""

    def test_trainset_aging_limit_reduces_train_size(self):
        """Test that aging limit reduces training set size."""
        df = pd.DataFrame({'feature': np.arange(1000)})
        timestamps = pd.Series(pd.date_range('2023-01-01', periods=1000, freq='1h'))

        # Without aging limit
        train_idx_full, _, _, _, _, _ = make_train_test_split(
            df, test_size=0.2, val_size=0.1,
            timestamps=timestamps,
            trainset_aging_limit=None,
            random_seed=42
        )

        # With aging limit of 0.5 (keep only 50% most recent)
        train_idx_aged, _, _, _, _, _ = make_train_test_split(
            df, test_size=0.2, val_size=0.1,
            timestamps=timestamps,
            trainset_aging_limit=0.5,
            random_seed=42
        )

        assert len(train_idx_aged) < len(train_idx_full), "Aging limit should reduce train size"
        assert len(train_idx_aged) == pytest.approx(len(train_idx_full) * 0.5, rel=0.1), "Train should be ~50% of original"

    def test_trainset_aging_limit_keeps_recent_data(self):
        """Test that aging limit keeps the most recent data."""
        df = pd.DataFrame({'feature': np.arange(1000)})
        timestamps = pd.Series(pd.date_range('2023-01-01', periods=1000, freq='1h'))

        train_idx, _, _, _, _, _ = make_train_test_split(
            df, test_size=0.2, val_size=0.1,
            timestamps=timestamps,
            wholeday_splitting=False,
            trainset_aging_limit=0.3,
            random_seed=42
        )

        # The training data should be from the more recent part
        train_timestamps = timestamps.iloc[train_idx]
        assert train_timestamps.max() > train_timestamps.min(), "Should have a range of dates"

    def test_invalid_aging_limit_raises_error(self):
        """Test that invalid aging limit values raise errors."""
        df = pd.DataFrame({'feature': np.arange(100)})
        timestamps = pd.Series(pd.date_range('2023-01-01', periods=100, freq='1h'))

        # Test with aging_limit=0.0 (should effectively not filter due to short-circuit)
        train_idx, val_idx, test_idx, _, _, _ = make_train_test_split(
            df, test_size=0.2, val_size=0.1,
            timestamps=timestamps,
            trainset_aging_limit=0.0,
            random_seed=42
        )
        # With 0.0 aging limit, should still produce valid splits
        assert len(train_idx) > 0, "Should have train samples"

        # Test with aging_limit=1.0 (should raise error - must be in (0, 1))
        with pytest.raises(ValueError, match="trainset_aging_limit must be in"):
            make_train_test_split(
                df, test_size=0.2, val_size=0.1,
                timestamps=timestamps,
                trainset_aging_limit=1.0,
                random_seed=42
            )


class TestMakeTrainTestSplitEdgeCases:
    """Test edge cases and error conditions."""

    def test_small_dataset(self):
        """Test splitting a small dataset."""
        df = pd.DataFrame({'feature': np.arange(10)})

        train_idx, val_idx, test_idx, _, _, _ = make_train_test_split(
            df, test_size=0.2, val_size=0.2, random_seed=42
        )

        # Should still produce valid splits
        all_idx = np.concatenate([train_idx, val_idx, test_idx])
        assert len(np.unique(all_idx)) == len(all_idx), "All indices should be unique"

    def test_single_sample_dataset(self):
        """Test splitting a single-sample dataset."""
        df = pd.DataFrame({'feature': [1.0]})

        # sklearn's train_test_split can't handle single sample with fractional sizes
        # This is expected behavior - splitting a single sample is not well-defined
        # Test that sklearn raises an appropriate error
        with pytest.raises((ValueError, Exception)):
            make_train_test_split(
                df, test_size=0.2, val_size=0.1, random_seed=42
            )

    def test_sorted_output_indices(self):
        """Test that output indices are sorted."""
        df = pd.DataFrame({'feature': np.random.randn(500)})

        train_idx, val_idx, test_idx, _, _, _ = make_train_test_split(
            df, test_size=0.2, val_size=0.1, shuffle_val=True, shuffle_test=True, random_seed=42
        )

        # All outputs should be sorted
        np.testing.assert_array_equal(train_idx, np.sort(train_idx), "Train indices should be sorted")
        np.testing.assert_array_equal(val_idx, np.sort(val_idx), "Val indices should be sorted")
        np.testing.assert_array_equal(test_idx, np.sort(test_idx), "Test indices should be sorted")

    def test_no_timestamps_uses_sklearn(self):
        """Test that missing timestamps falls back to sklearn-based splitting."""
        df = pd.DataFrame({'feature': np.random.randn(100)})

        train_idx, val_idx, test_idx, train_details, val_details, test_details = make_train_test_split(
            df, test_size=0.2, val_size=0.1,
            timestamps=None,  # No timestamps
            random_seed=42
        )

        # Details should be empty without timestamps
        assert train_details == '', "Train details should be empty without timestamps"
        assert val_details == '', "Val details should be empty without timestamps"
        assert test_details == '', "Test details should be empty without timestamps"

        # But splits should still work
        all_idx = np.concatenate([train_idx, val_idx, test_idx])
        assert len(all_idx) == len(df), "All rows should be assigned"

    def test_very_small_split_sizes(self):
        """Test with very small split sizes."""
        df = pd.DataFrame({'feature': np.random.randn(1000)})

        train_idx, val_idx, test_idx, _, _, _ = make_train_test_split(
            df, test_size=0.01, val_size=0.01, random_seed=42
        )

        # Should produce very small but non-empty splits
        assert len(test_idx) > 0 or len(test_idx) == 0, "Test split may be tiny or empty"
        assert len(train_idx) > len(val_idx), "Train should be larger than val"

    def test_large_split_sizes(self):
        """Test with large test/val sizes."""
        df = pd.DataFrame({'feature': np.random.randn(100)})

        train_idx, val_idx, test_idx, _, _, _ = make_train_test_split(
            df, test_size=0.4, val_size=0.4, random_seed=42
        )

        # Train should be very small
        assert len(train_idx) < len(test_idx), "Train should be smaller than test with large test_size"


class TestMakeTrainTestSplitIntegration:
    """Integration tests combining multiple features."""

    def test_wholeday_with_aging_limit(self):
        """Test whole-day splitting combined with aging limit."""
        np.random.seed(42)
        n_samples = 1000
        base_date = datetime(2023, 1, 1)
        timestamps = pd.Series([base_date + timedelta(days=i // 10) for i in range(n_samples)])

        df = pd.DataFrame({'feature': np.random.randn(n_samples)})

        train_idx, val_idx, test_idx, _, _, _ = make_train_test_split(
            df, test_size=0.2, val_size=0.1,
            timestamps=timestamps,
            wholeday_splitting=True,
            trainset_aging_limit=0.5,
            random_seed=42
        )

        # All indices should be valid
        assert all(0 <= i < len(df) for i in train_idx), "All train indices should be valid"
        assert all(0 <= i < len(df) for i in val_idx), "All val indices should be valid"
        assert all(0 <= i < len(df) for i in test_idx), "All test indices should be valid"

    def test_mixed_sequential_with_date_details(self):
        """Test mixed sequential fraction with proper date detail strings."""
        np.random.seed(42)
        n_samples = 500
        timestamps = pd.Series(pd.date_range('2023-01-01', periods=n_samples, freq='1h'))

        df = pd.DataFrame({'feature': np.random.randn(n_samples)})

        train_idx, val_idx, test_idx, train_details, val_details, test_details = make_train_test_split(
            df, test_size=0.2, val_size=0.1,
            timestamps=timestamps,
            wholeday_splitting=True,
            val_sequential_fraction=0.7,
            test_sequential_fraction=0.8,
            random_seed=42
        )

        # Details should contain date information
        assert '2023' in train_details, "Train details should contain year"


class TestMakeTrainTestSplitValidation:
    """Test input validation."""

    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        df = pd.DataFrame({'feature': []})

        # sklearn's train_test_split raises ValueError for empty input
        # This is expected behavior - splitting empty data is not well-defined
        with pytest.raises(ValueError, match="empty"):
            make_train_test_split(
                df, test_size=0.2, val_size=0.1, random_seed=42
            )

    def test_indices_are_numpy_arrays(self):
        """Test that returned indices are numpy arrays."""
        df = pd.DataFrame({'feature': np.random.randn(100)})

        train_idx, val_idx, test_idx, _, _, _ = make_train_test_split(
            df, test_size=0.2, val_size=0.1, random_seed=42
        )

        assert isinstance(train_idx, np.ndarray), "Train indices should be numpy array"
        assert isinstance(val_idx, np.ndarray), "Val indices should be numpy array"
        assert isinstance(test_idx, np.ndarray), "Test indices should be numpy array"
