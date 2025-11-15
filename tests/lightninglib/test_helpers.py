"""
Tests for helper functions in lightninglib.py

Run tests:
    pytest tests/lightninglib/test_helpers.py -v
    pytest tests/lightninglib/test_helpers.py --cov=mlframe.lightninglib --cov-report=html
"""

import pytest
import torch
import numpy as np
import pandas as pd
import polars as pl
from hypothesis import given, strategies as st, settings

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mlframe.lightninglib import (
    custom_collate_fn,
    to_tensor_any,
    to_numpy_safe,
    get_valid_num_groups,
)


# ================================================================================================
# custom_collate_fn Tests
# ================================================================================================


class TestCustomCollateFn:
    """Tests for custom_collate_fn."""

    def test_returns_batch_unchanged(self):
        """Test that custom_collate_fn returns batch as-is."""
        batch = [1, 2, 3, 4]
        result = custom_collate_fn(batch)
        assert result == batch
        assert result is batch  # Should be same object

    def test_with_list_batch(self):
        """Test with list batch."""
        batch = [[torch.tensor([1, 2]), torch.tensor([0])],
                 [torch.tensor([3, 4]), torch.tensor([1])]]
        result = custom_collate_fn(batch)
        assert result == batch

    def test_with_tuple_batch(self):
        """Test with tuple batch."""
        batch = ((1, 2), (3, 4))
        result = custom_collate_fn(batch)
        assert result == batch

    def test_with_dict_batch(self):
        """Test with dict batch."""
        batch = [{"features": torch.randn(5), "labels": torch.tensor(1)},
                 {"features": torch.randn(5), "labels": torch.tensor(0)}]
        result = custom_collate_fn(batch)
        assert result == batch

    def test_preserves_structure(self):
        """Test that exact structure is preserved."""
        batch = {"nested": [1, 2, {"deep": 3}]}
        result = custom_collate_fn(batch)
        assert result == batch
        assert isinstance(result, dict)


# ================================================================================================
# to_tensor_any Tests
# ================================================================================================


class TestToTensorAny:
    """Tests for to_tensor_any function."""

    # --- Pandas Tests ---

    def test_pandas_dataframe_conversion(self):
        """Test conversion from pandas DataFrame."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        tensor = to_tensor_any(df, dtype=torch.float32)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 2)
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))

    def test_pandas_series_conversion(self):
        """Test conversion from pandas Series."""
        series = pd.Series([1.0, 2.0, 3.0])
        tensor = to_tensor_any(series, dtype=torch.float32)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3,)
        assert tensor.dtype == torch.float32

    def test_pandas_with_different_dtypes(self):
        """Test pandas DataFrame with various dtypes."""
        df = pd.DataFrame({"int": [1, 2, 3], "float": [1.5, 2.5, 3.5]})
        tensor = to_tensor_any(df, dtype=torch.float64)

        assert tensor.dtype == torch.float64
        assert tensor.shape == (3, 2)

    # --- Polars Tests ---

    def test_polars_dataframe_conversion(self):
        """Test conversion from polars DataFrame."""
        pldf = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        tensor = to_tensor_any(pldf, dtype=torch.float32)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 2)
        assert tensor.dtype == torch.float32

    # --- NumPy Tests ---

    def test_numpy_array_conversion(self):
        """Test conversion from numpy array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        tensor = to_tensor_any(arr, dtype=torch.float32)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 2)
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_numpy_int_array(self):
        """Test conversion from numpy int array."""
        arr = np.array([1, 2, 3, 4], dtype=np.int64)
        tensor = to_tensor_any(arr, dtype=torch.int64)

        assert tensor.dtype == torch.int64
        assert torch.equal(tensor, torch.tensor([1, 2, 3, 4], dtype=torch.int64))

    # --- Torch Tensor Tests ---

    def test_torch_tensor_passthrough(self):
        """Test that torch.Tensor is converted to correct dtype."""
        original = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        tensor = to_tensor_any(original, dtype=torch.float32)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32

    # --- Device Tests ---

    def test_device_placement_cpu(self):
        """Test device placement to CPU."""
        arr = np.array([1.0, 2.0, 3.0])
        tensor = to_tensor_any(arr, dtype=torch.float32, device='cpu')

        assert tensor.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_placement_cuda(self):
        """Test device placement to CUDA."""
        arr = np.array([1.0, 2.0, 3.0])
        tensor = to_tensor_any(arr, dtype=torch.float32, device='cuda')

        assert tensor.device.type == 'cuda'

    # --- Edge Cases ---

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        tensor = to_tensor_any(df, dtype=torch.float32)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.numel() == 0

    def test_single_value_array(self):
        """Test with single value array."""
        arr = np.array([42.0])
        tensor = to_tensor_any(arr, dtype=torch.float32)

        assert tensor.shape == (1,)
        assert tensor.item() == 42.0

    def test_multidimensional_array(self):
        """Test with 3D array."""
        arr = np.random.randn(2, 3, 4)
        tensor = to_tensor_any(arr, dtype=torch.float32)

        assert tensor.shape == (2, 3, 4)

    def test_with_nan_values(self):
        """Test handling of NaN values."""
        arr = np.array([1.0, np.nan, 3.0])
        tensor = to_tensor_any(arr, dtype=torch.float32)

        assert torch.isnan(tensor[1])
        assert tensor[0] == 1.0
        assert tensor[2] == 3.0

    def test_with_inf_values(self):
        """Test handling of inf values."""
        arr = np.array([1.0, np.inf, -np.inf])
        tensor = to_tensor_any(arr, dtype=torch.float32)

        assert torch.isinf(tensor[1])
        assert torch.isinf(tensor[2])

    # --- Dtype Tests ---

    def test_various_dtypes(self):
        """Test conversion with various dtypes."""
        arr = np.array([1, 2, 3])

        for dtype in [torch.float16, torch.float32, torch.float64, torch.int32, torch.int64]:
            tensor = to_tensor_any(arr, dtype=dtype)
            assert tensor.dtype == dtype


# ================================================================================================
# to_numpy_safe Tests
# ================================================================================================


class TestToNumpySafe:
    """Tests for to_numpy_safe function."""

    def test_basic_tensor_conversion(self):
        """Test basic tensor to numpy conversion."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        arr = to_numpy_safe(tensor)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)
        assert np.allclose(arr, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_tensor_with_gradients(self):
        """Test tensor with requires_grad=True."""
        tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        arr = to_numpy_safe(tensor)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensor_to_cpu(self):
        """Test CUDA tensor conversion (should move to CPU)."""
        tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        # to_numpy_safe should automatically move to CPU
        arr = to_numpy_safe(tensor, cpu=True)

        assert isinstance(arr, np.ndarray)
        assert np.allclose(arr, np.array([1.0, 2.0, 3.0]))

    def test_cpu_parameter_true(self):
        """Test cpu=True parameter."""
        tensor = torch.tensor([1.0, 2.0])
        arr = to_numpy_safe(tensor, cpu=True)

        assert isinstance(arr, np.ndarray)

    def test_cpu_parameter_false(self):
        """Test cpu=False parameter."""
        tensor = torch.tensor([1.0, 2.0])
        arr = to_numpy_safe(tensor, cpu=False)

        assert isinstance(arr, np.ndarray)

    # --- Unsupported Dtype Conversion Tests ---

    def test_bfloat16_conversion(self):
        """Test bfloat16 tensor conversion to float32."""
        if not hasattr(torch, 'bfloat16'):
            pytest.skip("bfloat16 not available")

        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        arr = to_numpy_safe(tensor)

        assert isinstance(arr, np.ndarray)
        # Should be converted to float32 for numpy compatibility
        assert arr.dtype in [np.float32, np.float64]

    def test_float16_conversion(self):
        """Test float16 tensor conversion."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        arr = to_numpy_safe(tensor)

        assert isinstance(arr, np.ndarray)
        # float16 might be converted to float32 for better numpy compatibility

    # --- Supported Dtypes ---

    def test_int64_tensor(self):
        """Test int64 tensor conversion."""
        tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
        arr = to_numpy_safe(tensor)

        assert arr.dtype == np.int64
        assert np.array_equal(arr, np.array([1, 2, 3]))

    def test_float32_tensor(self):
        """Test float32 tensor conversion."""
        tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
        arr = to_numpy_safe(tensor)

        assert arr.dtype == np.float32

    def test_float64_tensor(self):
        """Test float64 tensor conversion."""
        tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64)
        arr = to_numpy_safe(tensor)

        assert arr.dtype == np.float64

    # --- Edge Cases ---

    def test_scalar_tensor(self):
        """Test scalar tensor conversion."""
        tensor = torch.tensor(42.0)
        arr = to_numpy_safe(tensor)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == ()
        assert arr.item() == 42.0

    def test_empty_tensor(self):
        """Test empty tensor conversion."""
        tensor = torch.tensor([])
        arr = to_numpy_safe(tensor)

        assert isinstance(arr, np.ndarray)
        assert arr.size == 0

    def test_multidimensional_tensor(self):
        """Test 4D tensor conversion."""
        tensor = torch.randn(2, 3, 4, 5)
        arr = to_numpy_safe(tensor)

        assert arr.shape == (2, 3, 4, 5)

    # --- Error Handling ---

    def test_non_tensor_input_raises_error(self):
        """Test that non-tensor input raises TypeError."""
        with pytest.raises((AttributeError, TypeError)):
            to_numpy_safe([1, 2, 3])

    def test_non_tensor_numpy_raises_error(self):
        """Test that numpy array input raises error."""
        with pytest.raises((AttributeError, TypeError)):
            to_numpy_safe(np.array([1, 2, 3]))


# ================================================================================================
# get_valid_num_groups Tests
# ================================================================================================


class TestGetValidNumGroups:
    """Tests for get_valid_num_groups function."""

    def test_exact_divisor(self):
        """Test when num_channels is exactly divisible by preferred_num_groups."""
        result = get_valid_num_groups(num_channels=16, preferred_num_groups=4)
        assert result == 4

    def test_largest_divisor_below_preferred(self):
        """Test finding largest divisor when preferred is not exact."""
        result = get_valid_num_groups(num_channels=16, preferred_num_groups=5)
        # Divisors of 16: 1, 2, 4, 8, 16
        # Largest <= 5 is 4
        assert result == 4

    def test_preferred_larger_than_channels(self):
        """Test when preferred_num_groups > num_channels."""
        result = get_valid_num_groups(num_channels=8, preferred_num_groups=10)
        # Divisors of 8: 1, 2, 4, 8
        # Largest is 8
        assert result == 8

    def test_returns_one_as_fallback(self):
        """Test that 1 is always a valid fallback."""
        result = get_valid_num_groups(num_channels=7, preferred_num_groups=0)
        assert result == 1

    def test_prime_number_channels(self):
        """Test with prime number of channels."""
        result = get_valid_num_groups(num_channels=13, preferred_num_groups=4)
        # Only divisors are 1 and 13
        # Largest <= 4 is 1
        assert result == 1

    def test_preferred_equals_channels(self):
        """Test when preferred equals num_channels."""
        result = get_valid_num_groups(num_channels=32, preferred_num_groups=32)
        assert result == 32

    def test_single_channel(self):
        """Test with single channel."""
        result = get_valid_num_groups(num_channels=1, preferred_num_groups=4)
        assert result == 1

    def test_large_numbers(self):
        """Test with large numbers."""
        result = get_valid_num_groups(num_channels=1024, preferred_num_groups=32)
        assert result == 32
        assert 1024 % result == 0

    def test_common_scenarios(self):
        """Test common GroupNorm scenarios."""
        # 32 channels, want 8 groups
        assert get_valid_num_groups(32, 8) == 8

        # 64 channels, want 16 groups
        assert get_valid_num_groups(64, 16) == 16

        # 50 channels, want 8 groups -> should get 5
        result = get_valid_num_groups(50, 8)
        assert result == 5
        assert 50 % result == 0

    def test_zero_preferred(self):
        """Test with preferred_num_groups=0."""
        result = get_valid_num_groups(num_channels=16, preferred_num_groups=0)
        assert result == 1

    def test_negative_preferred(self):
        """Test with negative preferred (edge case)."""
        result = get_valid_num_groups(num_channels=16, preferred_num_groups=-1)
        assert result == 1

    @given(
        num_channels=st.integers(min_value=1, max_value=1024),
        preferred_num_groups=st.integers(min_value=1, max_value=128)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_always_valid_divisor(self, num_channels, preferred_num_groups):
        """Property test: result should always divide num_channels evenly."""
        result = get_valid_num_groups(num_channels, preferred_num_groups)

        assert result >= 1
        assert num_channels % result == 0
        assert result <= preferred_num_groups or result == num_channels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
