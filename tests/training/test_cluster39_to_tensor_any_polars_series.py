"""Cluster 39 regression: to_tensor_any accepts a bare polars Series (audit 2026-11-20, finding 39-F63).

A ``pl.Series`` used to fall through every branch and raise AttributeError. It now converts like a numeric array.
"""

import numpy as np
import polars as pl
import torch

from mlframe.training.neural._base_tensor_helpers import to_tensor_any


def test_polars_series_converts_to_float_tensor():
    """A pl.Series of floats becomes a torch tensor with the same values (39-F63)."""
    out = to_tensor_any(pl.Series([1.0, 2.0, 3.0]))
    assert isinstance(out, torch.Tensor)
    np.testing.assert_allclose(out.numpy(), [1.0, 2.0, 3.0])


def test_polars_dataframe_still_converts():
    """Control: the existing DataFrame branch keeps working (39-F63)."""
    out = to_tensor_any(pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
    assert out.shape == (2, 2)
