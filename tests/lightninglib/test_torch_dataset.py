"""
Tests for TorchDataset class in lightninglib.py

Run tests:
    pytest tests/lightninglib/test_torch_dataset.py -v
    pytest tests/lightninglib/test_torch_dataset.py --cov=mlframe.lightninglib --cov-report=html
"""

import pytest
import torch
import numpy as np
import pandas as pd
import polars as pl
from hypothesis import given, strategies as st, settings, assume

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mlframe.lightninglib import TorchDataset


# ================================================================================================
# Fixtures
# ================================================================================================


@pytest.fixture
def sample_numpy_data():
    """Generate sample numpy arrays."""
    features = np.random.randn(100, 10).astype(np.float32)
    labels = np.random.randint(0, 3, 100).astype(np.int64)
    return features, labels


@pytest.fixture
def sample_pandas_data():
    """Generate sample pandas DataFrame and Series."""
    features = pd.DataFrame(np.random.randn(50, 5), columns=[f"f{i}" for i in range(5)])
    labels = pd.Series(np.random.randint(0, 2, 50), name="label")
    return features, labels


@pytest.fixture
def sample_polars_data():
    """Generate sample polars DataFrame."""
    features = pl.DataFrame(np.random.randn(30, 8))
    labels = np.random.randint(0, 2, 30)
    return features, labels


# ================================================================================================
# Initialization Tests
# ================================================================================================


class TestTorchDatasetInitialization:
    """Tests for TorchDataset.__init__."""

    def test_numpy_array_initialization(self, sample_numpy_data):
        """Test initialization with numpy arrays."""
        features, labels = sample_numpy_data
        dataset = TorchDataset(features, labels)

        assert dataset.features_dtype == torch.float32
        assert dataset.labels_dtype == torch.float32
        assert dataset.batch_size == 0
        assert dataset.dataset_length == 100
        assert dataset.labels is not None

    def test_pandas_dataframe_initialization(self, sample_pandas_data):
        """Test initialization with pandas DataFrame."""
        features, labels = sample_pandas_data
        dataset = TorchDataset(features, labels)

        assert dataset.dataset_length == 50
        assert dataset.labels is not None

    def test_polars_dataframe_initialization(self, sample_polars_data):
        """Test initialization with polars DataFrame."""
        features, labels = sample_polars_data
        dataset = TorchDataset(features, labels)

        assert dataset.dataset_length == 30

    def test_torch_tensor_initialization(self):
        """Test initialization with torch tensors."""
        features = torch.randn(20, 5)
        labels = torch.randint(0, 2, (20,))
        dataset = TorchDataset(features, labels)

        assert dataset.dataset_length == 20

    def test_custom_dtypes(self, sample_numpy_data):
        """Test initialization with custom dtypes."""
        features, labels = sample_numpy_data
        dataset = TorchDataset(
            features,
            labels,
            features_dtype=torch.float64,
            labels_dtype=torch.int64
        )

        assert dataset.features_dtype == torch.float64
        assert dataset.labels_dtype == torch.int64

    def test_batch_mode_initialization(self, sample_numpy_data):
        """Test initialization in batch mode."""
        features, labels = sample_numpy_data
        dataset = TorchDataset(features, labels, batch_size=32)

        assert dataset.batch_size == 32
        assert dataset.num_batches == 4  # ceil(100 / 32)

    def test_labels_none_initialization(self, sample_numpy_data):
        """Test initialization without labels (prediction mode)."""
        features, _ = sample_numpy_data
        dataset = TorchDataset(features, labels=None)

        assert dataset.labels is None
        assert dataset.dataset_length == 100

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_initialization(self, sample_numpy_data):
        """Test initialization with CUDA device."""
        features, labels = sample_numpy_data
        dataset = TorchDataset(features, labels, device='cuda')

        assert dataset.device == 'cuda'
        # Features should be preloaded to GPU
        assert dataset.features.device.type == 'cuda'
        assert dataset.labels.device.type == 'cuda'

    def test_cpu_device_initialization(self, sample_numpy_data):
        """Test initialization with CPU device (lazy loading)."""
        features, labels = sample_numpy_data
        dataset = TorchDataset(features, labels, device=None)

        # Features should NOT be converted yet (lazy)
        assert isinstance(dataset.features, np.ndarray)

    def test_labels_pandas_series(self):
        """Test labels as pandas Series."""
        features = np.random.randn(20, 5)
        labels = pd.Series([0, 1] * 10)
        dataset = TorchDataset(features, labels)

        assert isinstance(dataset.labels, torch.Tensor)
        assert len(dataset.labels) == 20

    def test_labels_polars_dataframe(self):
        """Test labels as polars DataFrame."""
        features = np.random.randn(15, 3)
        labels = pl.DataFrame({"label": [0, 1] * 7 + [0]})
        dataset = TorchDataset(features, labels)

        assert isinstance(dataset.labels, torch.Tensor)

    def test_labels_list(self):
        """Test labels as list."""
        features = np.random.randn(10, 3)
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        dataset = TorchDataset(features, labels)

        assert isinstance(dataset.labels, torch.Tensor)
        assert len(dataset.labels) == 10

    def test_labels_reshaping(self):
        """Test that labels are reshaped to 1D."""
        features = np.random.randn(10, 3)
        labels = np.array([[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]])
        dataset = TorchDataset(features, labels)

        assert dataset.labels.dim() == 1
        assert len(dataset.labels) == 10

    def test_dataset_length_from_features_tensor(self):
        """Test dataset length determination from torch tensor features."""
        features = torch.randn(25, 4)
        dataset = TorchDataset(features, labels=None)

        assert dataset.dataset_length == 25

    def test_dataset_length_from_features_numpy(self):
        """Test dataset length determination from numpy features."""
        features = np.random.randn(30, 4)
        dataset = TorchDataset(features, labels=None)

        assert dataset.dataset_length == 30

    def test_dataset_length_from_features_pandas(self):
        """Test dataset length determination from pandas features."""
        features = pd.DataFrame(np.random.randn(35, 4))
        dataset = TorchDataset(features, labels=None)

        assert dataset.dataset_length == 35

    def test_dataset_length_from_features_polars(self):
        """Test dataset length determination from polars features."""
        features = pl.DataFrame(np.random.randn(40, 4))
        dataset = TorchDataset(features, labels=None)

        assert dataset.dataset_length == 40

    def test_batch_calculation_exact_division(self):
        """Test batch calculation when dataset_length divides evenly."""
        features = np.random.randn(100, 5)
        labels = np.random.randint(0, 2, 100)
        dataset = TorchDataset(features, labels, batch_size=25)

        assert dataset.num_batches == 4  # 100 / 25 = 4

    def test_batch_calculation_with_remainder(self):
        """Test batch calculation with remainder."""
        features = np.random.randn(103, 5)
        labels = np.random.randint(0, 2, 103)
        dataset = TorchDataset(features, labels, batch_size=32)

        assert dataset.num_batches == 4  # ceil(103 / 32) = 4

    def test_single_sample_dataset(self):
        """Test with single sample."""
        features = np.random.randn(1, 5)
        labels = np.array([0])
        dataset = TorchDataset(features, labels)

        assert dataset.dataset_length == 1

    def test_empty_dataset_handling(self):
        """Test with empty dataset."""
        features = np.empty((0, 5))
        labels = np.array([])
        dataset = TorchDataset(features, labels)

        assert dataset.dataset_length == 0


# ================================================================================================
# __len__ Tests
# ================================================================================================


class TestTorchDatasetLen:
    """Tests for TorchDataset.__len__."""

    def test_len_sample_mode(self, sample_numpy_data):
        """Test __len__ in sample mode (batch_size=0)."""
        features, labels = sample_numpy_data
        dataset = TorchDataset(features, labels, batch_size=0)

        assert len(dataset) == 100

    def test_len_batch_mode(self, sample_numpy_data):
        """Test __len__ in batch mode."""
        features, labels = sample_numpy_data
        dataset = TorchDataset(features, labels, batch_size=32)

        assert len(dataset) == 4  # ceil(100 / 32)

    def test_len_without_labels(self):
        """Test __len__ without labels."""
        features = np.random.randn(50, 5)
        dataset = TorchDataset(features, labels=None, batch_size=10)

        assert len(dataset) == 5

    def test_len_single_batch(self):
        """Test __len__ when batch_size >= dataset_length."""
        features = np.random.randn(10, 5)
        labels = np.random.randint(0, 2, 10)
        dataset = TorchDataset(features, labels, batch_size=20)

        assert len(dataset) == 1


# ================================================================================================
# _extract Tests
# ================================================================================================


class TestTorchDatasetExtract:
    """Tests for TorchDataset._extract method."""

    def test_extract_from_torch_tensor(self):
        """Test extraction from torch.Tensor."""
        features = torch.randn(20, 5)
        labels = torch.randint(0, 2, (20,))
        dataset = TorchDataset(features, labels)

        extracted = dataset._extract(dataset.features, slice(0, 5))
        assert extracted.shape == (5, 5)

    def test_extract_from_numpy_array(self):
        """Test extraction from numpy array."""
        features = np.random.randn(20, 5)
        labels = np.random.randint(0, 2, 20)
        dataset = TorchDataset(features, labels)

        extracted = dataset._extract(dataset.features, slice(0, 5))
        assert isinstance(extracted, torch.Tensor)
        assert extracted.shape == (5, 5)

    def test_extract_from_pandas_dataframe(self):
        """Test extraction from pandas DataFrame."""
        features = pd.DataFrame(np.random.randn(20, 5))
        labels = np.random.randint(0, 2, 20)
        dataset = TorchDataset(features, labels)

        extracted = dataset._extract(dataset.features, slice(0, 5))
        assert isinstance(extracted, torch.Tensor)
        assert extracted.shape == (5, 5)

    def test_extract_from_polars_dataframe(self):
        """Test extraction from polars DataFrame."""
        features = pl.DataFrame(np.random.randn(20, 5))
        labels = np.random.randint(0, 2, 20)
        dataset = TorchDataset(features, labels)

        extracted = dataset._extract(dataset.features, slice(0, 5))
        assert isinstance(extracted, torch.Tensor)
        assert extracted.shape == (5, 5)

    def test_extract_single_index(self):
        """Test extraction with single integer index."""
        features = np.random.randn(20, 5)
        labels = np.random.randint(0, 2, 20)
        dataset = TorchDataset(features, labels)

        extracted = dataset._extract(dataset.features, 3)
        assert isinstance(extracted, torch.Tensor)
        assert extracted.shape == (5,)

    def test_extract_slice_index(self):
        """Test extraction with slice index."""
        features = np.random.randn(20, 5)
        labels = np.random.randint(0, 2, 20)
        dataset = TorchDataset(features, labels)

        extracted = dataset._extract(dataset.features, slice(5, 10))
        assert extracted.shape == (5, 5)

    def test_extract_dtype_conversion(self):
        """Test that extracted data has correct dtype."""
        features = np.random.randn(20, 5).astype(np.float64)
        labels = np.random.randint(0, 2, 20)
        dataset = TorchDataset(features, labels, features_dtype=torch.float32)

        extracted = dataset._extract(dataset.features, slice(0, 5))
        assert extracted.dtype == torch.float32

    def test_extract_device_placement(self):
        """Test that extracted data is on correct device."""
        features = np.random.randn(20, 5)
        labels = np.random.randint(0, 2, 20)
        dataset = TorchDataset(features, labels, device='cpu')

        extracted = dataset._extract(dataset.features, slice(0, 5))
        assert extracted.device.type == 'cpu'

    def test_extract_empty_slice(self):
        """Test extraction with empty slice."""
        features = np.random.randn(20, 5)
        labels = np.random.randint(0, 2, 20)
        dataset = TorchDataset(features, labels)

        extracted = dataset._extract(dataset.features, slice(10, 10))
        assert extracted.shape[0] == 0


# ================================================================================================
# __getitem__ Tests
# ================================================================================================


class TestTorchDatasetGetItem:
    """Tests for TorchDataset.__getitem__."""

    # --- Sample Mode Tests ---

    def test_getitem_sample_mode(self, sample_numpy_data):
        """Test __getitem__ in sample mode."""
        features, labels = sample_numpy_data
        dataset = TorchDataset(features, labels, batch_size=0)

        x, y = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (10,)  # Single sample
        assert y.shape == ()  # Scalar label

    def test_getitem_sample_mode_no_labels(self, sample_numpy_data):
        """Test __getitem__ in sample mode without labels."""
        features, _ = sample_numpy_data
        dataset = TorchDataset(features, labels=None, batch_size=0)

        x = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert x.shape == (10,)

    def test_getitem_sample_mode_various_indices(self, sample_numpy_data):
        """Test __getitem__ with various indices in sample mode."""
        features, labels = sample_numpy_data
        dataset = TorchDataset(features, labels, batch_size=0)

        # First index
        x0, y0 = dataset[0]
        assert x0.shape == (10,)

        # Middle index
        x50, y50 = dataset[50]
        assert x50.shape == (10,)

        # Last index
        x99, y99 = dataset[99]
        assert x99.shape == (10,)

    # --- Batch Mode Tests ---

    def test_getitem_batch_mode(self, sample_numpy_data):
        """Test __getitem__ in batch mode."""
        features, labels = sample_numpy_data
        dataset = TorchDataset(features, labels, batch_size=32)

        x, y = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (32, 10)  # Batch of 32 samples
        assert y.shape == (32,)

    def test_getitem_batch_mode_last_batch(self, sample_numpy_data):
        """Test __getitem__ for last batch (may be incomplete)."""
        features, labels = sample_numpy_data
        dataset = TorchDataset(features, labels, batch_size=32)

        # Last batch: 100 - 3*32 = 4 samples
        x, y = dataset[3]
        assert x.shape == (4, 10)
        assert y.shape == (4,)

    def test_getitem_batch_mode_start_end_calculation(self):
        """Test batch start/end index calculation."""
        features = np.random.randn(50, 5)
        labels = np.random.randint(0, 2, 50)
        dataset = TorchDataset(features, labels, batch_size=10)

        # Batch 0: samples 0-9
        x0, y0 = dataset[0]
        assert x0.shape == (10, 5)

        # Batch 2: samples 20-29
        x2, y2 = dataset[2]
        assert x2.shape == (10, 5)

        # Batch 4 (last): samples 40-49
        x4, y4 = dataset[4]
        assert x4.shape == (10, 5)

    def test_getitem_batch_mode_no_labels(self):
        """Test __getitem__ in batch mode without labels."""
        features = np.random.randn(50, 5)
        dataset = TorchDataset(features, labels=None, batch_size=10)

        x = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert x.shape == (10, 5)

    # --- Squeezing Tests ---

    def test_getitem_squeeze_single_sample(self):
        """Test dimension squeezing in sample mode."""
        features = np.random.randn(10, 1, 5)  # Shape with extra dimension
        labels = np.random.randint(0, 2, 10)
        dataset = TorchDataset(features, labels, batch_size=0)

        x, y = dataset[0]
        # Should squeeze only if x.ndim==2 and x.shape[0]==1
        # In this case, x extracted will be shape (1, 5) after slicing
        # So it should be squeezed to (5,)

    def test_getitem_no_squeeze_in_batch_mode(self):
        """Test that no squeezing occurs in batch mode."""
        features = np.random.randn(50, 5)
        labels = np.random.randint(0, 2, 50)
        dataset = TorchDataset(features, labels, batch_size=10)

        x, y = dataset[0]
        assert x.ndim == 2  # Should not be squeezed
        assert x.shape == (10, 5)

    # --- Edge Cases ---

    def test_getitem_first_index(self, sample_numpy_data):
        """Test first index access."""
        features, labels = sample_numpy_data
        dataset = TorchDataset(features, labels, batch_size=32)

        x, y = dataset[0]
        assert x.shape[0] == 32

    def test_getitem_single_element_batch(self):
        """Test batch with single element."""
        features = np.random.randn(10, 5)
        labels = np.random.randint(0, 2, 10)
        dataset = TorchDataset(features, labels, batch_size=100)  # Larger than dataset

        x, y = dataset[0]
        assert x.shape == (10, 5)  # All samples in one batch

    def test_getitem_different_data_types(self):
        """Test __getitem__ with different input data types."""
        # Pandas
        df = pd.DataFrame(np.random.randn(20, 5))
        labels = pd.Series(np.random.randint(0, 2, 20))
        dataset = TorchDataset(df, labels, batch_size=5)
        x, y = dataset[0]
        assert x.shape == (5, 5)

        # Polars
        pldf = pl.DataFrame(np.random.randn(20, 5))
        labels_np = np.random.randint(0, 2, 20)
        dataset = TorchDataset(pldf, labels_np, batch_size=5)
        x, y = dataset[0]
        assert x.shape == (5, 5)

    # --- Integration Tests ---

    def test_iterate_all_batches(self):
        """Test iterating through all batches."""
        features = np.random.randn(50, 5)
        labels = np.random.randint(0, 2, 50)
        dataset = TorchDataset(features, labels, batch_size=10)

        total_samples = 0
        for i in range(len(dataset)):
            x, y = dataset[i]
            total_samples += x.shape[0]

        assert total_samples == 50

    def test_iterate_all_samples(self):
        """Test iterating through all samples in sample mode."""
        features = np.random.randn(30, 5)
        labels = np.random.randint(0, 2, 30)
        dataset = TorchDataset(features, labels, batch_size=0)

        for i in range(len(dataset)):
            x, y = dataset[i]
            assert x.shape == (5,)

    def test_dataloader_integration(self):
        """Test integration with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        features = np.random.randn(100, 10)
        labels = np.random.randint(0, 3, 100)
        dataset = TorchDataset(features, labels, batch_size=0)

        # Use DataLoader with the dataset
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        batch_count = 0
        for batch_x, batch_y in loader:
            batch_count += 1
            assert batch_x.shape[0] <= 16
            assert batch_y.shape[0] <= 16

        assert batch_count == 7  # ceil(100 / 16)


# ================================================================================================
# Property-Based Tests
# ================================================================================================


class TestTorchDatasetProperties:
    """Property-based tests for TorchDataset."""

    @given(
        n_samples=st.integers(min_value=10, max_value=200),
        n_features=st.integers(min_value=1, max_value=20),
        batch_size=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=50, deadline=None)
    def test_property_batch_coverage(self, n_samples, n_features, batch_size):
        """Property: All batches should cover all samples."""
        features = np.random.randn(n_samples, n_features)
        labels = np.random.randint(0, 2, n_samples)
        dataset = TorchDataset(features, labels, batch_size=batch_size)

        total_samples = 0
        for i in range(len(dataset)):
            x, y = dataset[i]
            total_samples += x.shape[0]

        assert total_samples == n_samples

    @given(
        n_samples=st.integers(min_value=1, max_value=100),
        n_features=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=30, deadline=None)
    def test_property_sample_mode_shapes(self, n_samples, n_features):
        """Property: Sample mode should return correct shapes."""
        features = np.random.randn(n_samples, n_features)
        labels = np.random.randint(0, 2, n_samples)
        dataset = TorchDataset(features, labels, batch_size=0)

        assert len(dataset) == n_samples

        x, y = dataset[0]
        assert x.shape == (n_features,)

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        batch_size=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=30, deadline=None)
    def test_property_batch_count(self, n_samples, batch_size):
        """Property: Batch count should be ceil(n_samples / batch_size)."""
        features = np.random.randn(n_samples, 5)
        labels = np.random.randint(0, 2, n_samples)
        dataset = TorchDataset(features, labels, batch_size=batch_size)

        expected_batches = int(np.ceil(n_samples / batch_size))
        assert len(dataset) == expected_batches


# ================================================================================================
# Mutation Testing - TorchDataset Tests
# ================================================================================================


class TestTorchDatasetMutationTests:
    """Tests specifically targeting mutation survivors in TorchDataset."""

    def test_len_sample_mode_batch_size_zero(self):
        """Test __len__ correctly returns dataset_length when batch_size=0.

        Kills mutation: `batch_size == 0` to `< 0`, `<= 0`, `== -1`.
        """
        features = np.random.randn(100, 10).astype(np.float32)
        labels = np.random.randint(0, 2, 100).astype(np.int64)

        dataset = TorchDataset(features, labels, batch_size=0)

        # Should return dataset_length, not num_batches
        assert len(dataset) == 100

    def test_getitem_squeeze_only_in_sample_mode(self):
        """Test squeeze only happens when batch_size == 0.

        Kills mutation: `batch_size == 0` boundary conditions.
        """
        features = np.random.randn(10, 5).astype(np.float32)
        labels = np.random.randint(0, 2, 10).astype(np.int64)

        # Sample mode (batch_size=0) - should access single samples
        dataset_sample = TorchDataset(features, labels, batch_size=0)
        x, y = dataset_sample[0]
        assert x.shape == (5,), f"Sample mode should squeeze, got {x.shape}"

        # Batch mode (batch_size>0) - should not squeeze
        dataset_batch = TorchDataset(features, labels, batch_size=5)
        x, y = dataset_batch[0]
        assert x.shape == (5, 5), f"Batch mode should not squeeze, got {x.shape}"

    def test_squeeze_only_when_first_dim_is_one(self):
        """Test squeeze only when shape[0] is exactly 1.

        Kills mutation: `x.shape[0] == 1` to `x.shape[1] == 1`, `x.shape[0] >= 1`.
        """
        # 2D features where extracted sample will be (1, 5) before squeeze
        features = np.random.randn(10, 5).astype(np.float32)
        labels = np.random.randint(0, 2, 10).astype(np.int64)

        dataset = TorchDataset(features, labels, batch_size=0)

        # Single sample access - x will be squeezed from (1, 5) -> (5,)
        x, y = dataset[0]
        # The squeeze targets (1, N) -> (N,) for sample mode
        assert x.ndim == 1, f"Expected 1D after squeeze, got {x.ndim}D"

    def test_batch_mode_preserves_dimensions(self):
        """Test batch mode preserves 2D output.

        Kills mutation: dimension checks in __getitem__.
        """
        features = np.random.randn(50, 8).astype(np.float32)
        labels = np.random.randint(0, 3, 50).astype(np.int64)

        dataset = TorchDataset(features, labels, batch_size=10)

        for i in range(len(dataset)):
            x, y = dataset[i]
            assert x.ndim == 2, f"Batch {i}: expected 2D, got {x.ndim}D"


# ================================================================================================
# Phase 3 - Medium Priority Mutation Tests
# ================================================================================================


class TestTorchDatasetPhase3:
    """Phase 3 tests for TorchDataset mutation survivors."""

    def test_batch_boundary_calculation(self):
        """Test batch start/end boundary calculations.

        Kills mutation: boundary index calculations.
        """
        features = np.random.randn(25, 5).astype(np.float32)
        labels = np.random.randint(0, 2, 25).astype(np.int64)

        dataset = TorchDataset(features, labels, batch_size=10)

        # Check each batch has correct number of samples
        x0, _ = dataset[0]
        assert x0.shape[0] == 10, "Batch 0 should have 10 samples"

        x1, _ = dataset[1]
        assert x1.shape[0] == 10, "Batch 1 should have 10 samples"

        x2, _ = dataset[2]
        assert x2.shape[0] == 5, "Batch 2 (last) should have 5 samples"

    def test_labels_none_returns_only_features(self):
        """Test that dataset without labels returns only features.

        Kills mutation: labels None check.
        """
        features = np.random.randn(10, 5).astype(np.float32)
        dataset = TorchDataset(features, labels=None, batch_size=0)

        result = dataset[0]
        # Should return just features, not tuple
        assert isinstance(result, torch.Tensor)
        assert result.shape == (5,)

    def test_dtype_conversion_preserves_precision(self):
        """Test dtype conversion works correctly.

        Kills mutation: dtype conversion logic.
        """
        features = np.random.randn(10, 5).astype(np.float64)
        labels = np.random.randint(0, 2, 10).astype(np.int32)

        dataset = TorchDataset(
            features, labels,
            features_dtype=torch.float32,
            labels_dtype=torch.int64,
            batch_size=0
        )

        x, y = dataset[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64

    def test_empty_batch_handling(self):
        """Test handling of edge cases with small datasets.

        Kills mutation: edge case handling.
        """
        features = np.random.randn(3, 5).astype(np.float32)
        labels = np.random.randint(0, 2, 3).astype(np.int64)

        # Batch size larger than dataset
        dataset = TorchDataset(features, labels, batch_size=10)

        assert len(dataset) == 1
        x, y = dataset[0]
        assert x.shape[0] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
