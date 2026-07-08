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


def _ensure_numpy(arr, dtype: type = np.float32) -> np.ndarray | None:
    """Convert DataFrame/Series/array-like to numpy array; passes None through."""
    if arr is None:
        return None
    if hasattr(arr, "to_numpy"):  # Polars DataFrame/Series
        return np.asarray(arr.to_numpy().astype(dtype))
    if hasattr(arr, "values"):  # Pandas DataFrame/Series
        return np.asarray(arr.values.astype(dtype))
    return np.asarray(arr, dtype=dtype)


_CUDA_PROBE_CACHE: dict[str, bool] = {}


def _probe_cuda_is_usable() -> bool:
    """Return True iff CUDA reports available AND a representative workload runs.

    ``torch.cuda.is_available()`` is necessary but not sufficient -- the host
    may carry CUDA libraries (cupy / torch CUDA wheels installed) on top of a
    broken driver / no GPU / a context the calling process can't open.
    Lightning's ``accelerator='auto'`` happily resolves to CUDA in that
    half-broken state and then dies inside ``model_to_device`` with
    ``CUDA error: an illegal memory access`` / ``CURAND_STATUS_*`` /
    ``CUDA_ERROR_NO_DEVICE``. A 1-element allocation isn't enough on every
    host -- some boxes have a CUDA context that opens but crashes on the
    first non-trivial kernel. The probe runs a small nn.Linear forward +
    backward + Adam-style sqrt op (the kernel the failure log surfaces),
    so we either confirm GPU is functional or surface the breakage HERE
    (cached, ~10ms upfront cost) rather than mid-fit every time."""
    if "usable" in _CUDA_PROBE_CACHE:
        return _CUDA_PROBE_CACHE["usable"]
    ok = False
    try:
        if torch.cuda.is_available():
            _dev = torch.device("cuda")
            # Allocation + transfer + a small matmul + foreach_sqrt mirrors
            # the ops a Lightning Adam fit drives in the first batch. The
            # final ``empty_cache`` mirrors what Lightning's CUDA accelerator
            # does FIRST in ``strategy.setup`` (lightning/pytorch/accelerators
            # /cuda.py:_clear_cuda_memory) -- some hosts have a CUDA context
            # that opens, allocates, runs kernels, syncs, BUT crashes the
            # cache cleanup with ``CUDA error: an illegal memory access``,
            # which then poisons Lightning's setup AND its teardown.
            # Probing empty_cache here catches that quirky state up front.
            _x = torch.randn(64, 8, device=_dev)
            _w = torch.randn(8, 4, device=_dev, requires_grad=True)
            _y = (_x @ _w).sum()
            _y.backward()
            _ = torch._foreach_sqrt([torch.ones(4, device=_dev)])
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            ok = True
    except Exception:
        ok = False
        # Best-effort: try empty_cache once more inside the failure handler
        # so a downstream caller isn't surprised by a half-poisoned context.
        try:
            torch.cuda.empty_cache()
        except Exception:  # nosec B110 - best-effort path
            pass
    _CUDA_PROBE_CACHE["usable"] = ok
    return ok


def safe_accelerator(requested: str | None = "auto") -> str:
    """Resolve ``requested`` to an accelerator string Lightning can use safely.

    ``'auto'`` / ``'cuda'`` / ``'gpu'`` are downgraded to ``'cpu'`` when the
    CUDA probe fails. Anything else (``'cpu'``, ``'mps'``, ``'tpu'``) passes
    through unchanged so callers can still force a specific device by name."""
    if requested is None or requested in ("auto", "cuda", "gpu"):
        return "cuda" if _probe_cuda_is_usable() else "cpu"
    return requested
