"""Tensor / dataframe conversion helpers for the PyTorch Lightning neural base.

Carved out of ``base.py`` to keep the parent below the 1k-line monolith threshold. The parent re-exports these symbols so existing imports keep working.
"""
from __future__ import annotations

import warnings as _warnings

import numpy as np
import pandas as pd
import polars as pl
import torch


def custom_collate_fn(batch):
    """Identity collate: hands the raw batch list through unchanged.

    DataLoader's default collate stacks each sample's tensors into a
    batched tensor. Some datasets in this package yield non-tensor
    objects (e.g. variable-length sequences pre-collated by their own
    helpers) where that default raises. Passing this collate selects
    "don't touch the batch" and lets the consumer handle structure.
    Equivalent to ``lambda x: x`` but defined as a top-level callable
    so it can be pickled for multi-worker DataLoaders.
    """
    return batch


def to_tensor_any(data, dtype=torch.float32, device=None, safe=True):
    """
    Converts pandas / polars / numpy / torch input to a torch.Tensor
    with minimal copies and correct dtype.

    If safe=True, ignores categorical/object columns gracefully.
    """

    # --- Pandas
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.to_numpy()
    # --- Polars
    elif isinstance(data, pl.DataFrame):
        data = data.to_torch()
    if isinstance(data, np.ndarray):
        # Pandas 2.x / PyArrow-backed Series can return read-only ndarrays via to_numpy(); torch emits
        # "The given NumPy array is not writeable" UserWarning in that case. Bench (bench_torch_from_numpy.py
        # 2026-05-21, shape=(1M, 50)): direct from_numpy on read-only is 0.009 ms/iter while .copy() to
        # silence the warning is 66 ms/iter (~7400x slower). We never write through this view: the very
        # next .to(dtype, device) either reuses the buffer (when dtype/device match) or allocates a fresh
        # writable tensor (when they don't). So the warning is informational noise; suppress it locally.
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", message=".*not writ(e)?able.*", category=UserWarning)
            data = torch.from_numpy(data)

    return data.to(dtype=dtype, device=device)


def to_numpy_safe(tensor: torch.Tensor, cpu: bool = False) -> np.ndarray:
    """Convert a torch.Tensor to a NumPy array safely and efficiently.

    - Moves tensor to CPU if needed.
    - Converts unsupported dtypes (bfloat16, float16) to float32.
    - Keeps dtype otherwise unchanged (no accidental downcast).
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    t = tensor.detach()
    if cpu and t.device.type != "cpu":
        t = t.cpu()

    # NumPy-incompatible dtypes
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.to(torch.float32)
    elif t.dtype == torch.complex32:
        t = t.to(torch.complex64)

    return t.numpy()


def _ensure_numpy(arr, dtype: np.dtype = np.float32) -> np.ndarray | None:
    """Convert DataFrame/Series/array-like to numpy array; passes None through."""
    if arr is None:
        return None
    if hasattr(arr, "to_numpy"):  # Polars DataFrame/Series
        return arr.to_numpy().astype(dtype)
    if hasattr(arr, "values"):  # Pandas DataFrame/Series
        return arr.values.astype(dtype)
    return np.asarray(arr, dtype=dtype)
