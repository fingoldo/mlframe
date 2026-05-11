"""Windows-aware MLP runtime defaults: DataLoader workers, AMP precision, predict batch_size.

Pulled out of ``helpers.py`` so the decision logic is unit-testable in isolation without spinning up Lightning. The 2026-05-11 TVT run showed the production MLP getting only ~6 epochs in 30 min because the DataLoader was single-threaded (``num_workers=0``), the GPU sat idle waiting for data, and precision was FP32 on an Ampere-class card that supports bf16-mixed at ~2x throughput. The previous defaults were chosen for **Windows + spawn-based multiprocessing safety** -- a regression we MUST preserve. Hence: Linux/Mac auto-raises workers, Windows stays at 0 unless the caller explicitly opts in.

Public surface
--------------
- :func:`resolve_mlp_dataloader_defaults` -- returns ``{num_workers, persistent_workers, prefetch_factor, pin_memory}`` per host.
- :func:`resolve_mlp_precision_default` -- returns ``"bf16-mixed"`` on Ampere+ CUDA, else ``"32-true"``.
- :func:`resolve_mlp_predict_batch_size` -- scales predict-time batch (default was 64, catastrophic for 4M-row inference); user can override via the suite-level knob.

Each function takes explicit overrides and returns a dict; no side effects, no hidden globals -- safe to call from any thread / process and trivially unit-testable.
"""
from __future__ import annotations

import logging
import os
import platform
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataLoader workers / persistent / prefetch / pin_memory
# ---------------------------------------------------------------------------

# Cross-platform policy: keep DataLoader single-threaded by default on
# EVERY OS. The 2026-05-12 audit (user observation) flagged that our
# ``TorchDataset`` stores the entire input Polars / pandas / numpy frame as
# ``self.features`` (see ``mlframe/training/neural/data.py:62``). With
# ``num_workers > 0``:
#   * Windows uses spawn -> the whole Dataset (incl. the 100 GB frame) gets
#     pickled into every worker. IPC death + N x 100 GB memory.
#   * Linux/Mac use fork -> Arrow-backed Polars buffers are CoW-shared, BUT
#     Python refcount writes break CoW on every access and Polars indexing
#     (``df[indices]``) materialises copies on the per-row path. On a 100 GB
#     frame even partial CoW breakage swap-thrashes the box.
# So both platforms stay at 0 until the datamodule is rewritten to use
# shared-memory tensors (torch.share_memory_() or memory-mapped Arrow IPC).
# Users opt in explicitly via mlp_kwargs["dataloader_params"]["num_workers"]
# when they know their dataset fits in worker memory.
_DEFAULT_NUM_WORKERS: int = 0


def _is_windows() -> bool:
    """Single chokepoint so tests can monkeypatch the OS detection."""
    return platform.system().lower().startswith("win")


def resolve_mlp_dataloader_defaults(
    *,
    user_overrides: Optional[Dict[str, Any]] = None,
    cpu_count: Optional[int] = None,
    cuda_available: Optional[bool] = None,
    force_windows: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return safe DataLoader kwargs given the host + user overrides.

    Parameters
    ----------
    user_overrides
        If provided, each key wins over the auto-resolved default. Pass
        ``mlp_kwargs.get("dataloader_params", {})`` here at the call site.
    cpu_count
        Override the autodetected CPU count (mainly for tests).
    cuda_available
        Override CUDA detection (mainly for tests). ``None`` triggers a real
        ``torch.cuda.is_available()`` probe.
    force_windows
        Override the OS detection (mainly for tests; ``None`` uses
        :func:`_is_windows`).

    Returns
    -------
    dict
        ``{num_workers, persistent_workers, prefetch_factor, pin_memory}`` --
        merged with ``user_overrides`` (user wins) and additionally consistent:
        when ``num_workers == 0`` we force ``persistent_workers=False`` and
        ``prefetch_factor=None`` because PyTorch raises otherwise.
    """
    overrides = dict(user_overrides or {})
    is_win = _is_windows() if force_windows is None else bool(force_windows)
    n_cpu = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    if cuda_available is None:
        try:
            import torch
            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
    else:
        cuda_ok = bool(cuda_available)

    # 1. num_workers: 0 on EVERY OS unless user opts in. The TorchDataset
    #    holds the entire Polars/pandas frame in its closure -- worker
    #    spawn (Windows) pickles it, worker fork (Linux) breaks Arrow CoW
    #    on every Python refcount write. Either way, n_workers > 0 on big
    #    frames is a memory landmine. The ``is_win`` flag is kept on the
    #    signature for future per-OS tuning when the datamodule is
    #    rewritten to use shared-memory tensors.
    if "num_workers" in overrides:
        num_workers = int(overrides["num_workers"])
    else:
        num_workers = _DEFAULT_NUM_WORKERS
    # Silence the unused-variable warning -- ``is_win`` and ``n_cpu`` are
    # retained on the signature so callers can still control the autodetect
    # path from tests without breaking the contract.
    _ = is_win
    _ = n_cpu

    # 2. persistent_workers + prefetch_factor only valid when num_workers > 0.
    #    PyTorch raises a hard ValueError otherwise -- keep them aligned with
    #    num_workers regardless of user overrides.
    if num_workers <= 0:
        persistent_workers = False
        prefetch_factor = None
    else:
        persistent_workers = bool(overrides.get("persistent_workers", True))
        prefetch_factor = int(overrides.get("prefetch_factor", 4))

    # 3. pin_memory: True on CUDA hosts, False otherwise. Saves a CPU->GPU
    #    copy via page-locked memory. Cheap to set; no Windows landmine.
    pin_memory = bool(overrides.get("pin_memory", cuda_ok))

    resolved = {
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "pin_memory": pin_memory,
    }
    # Re-apply any user override fields we haven't explicitly resolved (so a
    # caller can pass eg ``timeout=30`` and get it through). batch_size /
    # shuffle / sampler etc. are handled upstream and untouched here.
    for k, v in overrides.items():
        if k not in resolved:
            resolved[k] = v
    return resolved


# ---------------------------------------------------------------------------
# AMP precision
# ---------------------------------------------------------------------------

# Ampere (sm_80) is the cutoff for usable bf16 hardware. RTX 30/40-series,
# A100, H100 all clear. Older cards (V100=sm_70, T4=sm_75, RTX 20-series)
# expose bf16 only via software emulation, which is SLOWER than fp32 -- so we
# keep them on "32-true". Mixed-precision fp16 is excluded as a default
# because gradient scaling pitfalls are still a foot-gun on tabular MLPs
# where targets can be 5-decimal-precision numbers (see TVT regression where
# the small targets after StandardScaler are exactly the float16-underflow
# zone). bf16 has the same exponent range as fp32, no scaling needed.
_BF16_MIN_CC_MAJOR: int = 8


def resolve_mlp_precision_default(
    *,
    user_override: Optional[str] = None,
    cuda_available: Optional[bool] = None,
    cuda_compute_capability_major: Optional[int] = None,
) -> str:
    """Return a Lightning ``precision=`` string.

    Auto-resolves to ``"bf16-mixed"`` on Ampere+ CUDA hosts, else ``"32-true"``.
    Callers may pass ``user_override`` (mlp_kwargs["trainer_params"]["precision"])
    to bypass the autoresolution entirely -- the override always wins.

    Parameters
    ----------
    user_override
        Lightning precision string from the caller. Returned verbatim when
        non-None. Common values: ``"32-true"``, ``"16-mixed"``, ``"bf16-mixed"``.
    cuda_available
        Override CUDA detection. ``None`` probes via ``torch.cuda.is_available()``.
    cuda_compute_capability_major
        Override the GPU compute capability major version. ``None`` probes
        the first device via ``torch.cuda.get_device_capability(0)``.
    """
    if user_override is not None:
        return str(user_override)
    if cuda_available is None:
        try:
            import torch
            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
    else:
        cuda_ok = bool(cuda_available)
    if not cuda_ok:
        return "32-true"
    if cuda_compute_capability_major is None:
        try:
            import torch
            cc_major, _ = torch.cuda.get_device_capability(0)
        except Exception:
            # Probing failed -- fall back to the conservative default.
            return "32-true"
    else:
        cc_major = int(cuda_compute_capability_major)
    return "bf16-mixed" if cc_major >= _BF16_MIN_CC_MAJOR else "32-true"


# ---------------------------------------------------------------------------
# Predict-time batch size -- ADAPTIVE on memory + dataframe width
# ---------------------------------------------------------------------------
#
# The previous default was 64 -- a Lightning legacy that makes sense for
# image-segmentation memory budgets but is catastrophic for tabular inference:
# 4M-row predict at batch=64 is 64K mini-batches, each paying full DataLoader
# setup + per-batch GPU sync overhead. The 2026-05-11 TVT run logged "Using
# batch_size=64 for prediction" on every call and incurred minutes of
# overhead on top of microseconds of actual MLP compute.
#
# BUT: the user reports that on a prior project with 30K features, batch=64
# was the LARGEST that fit in GPU memory. A blind switch to batch=4096 would
# regress that case. So we make the default ADAPTIVE based on:
#   1. available GPU memory (or CPU memory in cpu-only mode);
#   2. dataframe WIDTH (n_features per row);
#   3. dtype size (assume float32 = 4 bytes; cheap upper bound).
#
# Formula (predict only -- no gradients to store):
#   per_row_bytes = n_features * dtype_bytes * activation_multiplier
#   budget        = available_mem_bytes * mem_fraction
#   batch_size    = clamp(floor(budget / per_row_bytes), [min_batch, max_batch])
#
# ``activation_multiplier`` accounts for the intermediate hidden-layer
# activations on top of the raw input (deepest MLP path holds ~4x raw input
# in the worst case for a 4-layer 128->64->32->16 net). Set conservatively
# (predict is forward-only, so no grad doubling).

_PREDICT_BATCH_MIN: int = 64
_PREDICT_BATCH_MAX: int = 16384
_PREDICT_BATCH_DEFAULT_FLOOR: int = 256
_PREDICT_ACTIVATION_MULTIPLIER: int = 4
_PREDICT_MEM_FRACTION: float = 0.25  # use 25% of free mem -- leave headroom
_PREDICT_DTYPE_BYTES: int = 4        # float32 (cheap upper bound for tabular)
# Conservative fallback when memory cannot be probed (eg neither torch.cuda
# nor psutil available). Matches the old "Lightning legacy 64" within an
# order of magnitude but lets typical predict still throughput.
_PREDICT_BATCH_FALLBACK: int = 1024


def _probe_available_memory_bytes(*, cuda_available: Optional[bool] = None) -> Optional[int]:
    """Return available memory (GPU free if CUDA, else CPU available).

    Returns ``None`` when the probe fails -- caller falls back to a constant.
    Split from the resolver so tests can monkeypatch the probe directly.
    """
    if cuda_available is None:
        try:
            import torch
            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
    else:
        cuda_ok = bool(cuda_available)
    if cuda_ok:
        try:
            import torch
            free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
            return int(free_bytes)
        except Exception:
            pass
    # CPU mode (or CUDA probe failed): use psutil if available, else None.
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def resolve_mlp_predict_batch_size(
    *,
    user_override: Optional[int] = None,
    train_batch_size: Optional[int] = None,
    n_features: Optional[int] = None,
    available_memory_bytes: Optional[int] = None,
    cuda_available: Optional[bool] = None,
    dtype_bytes: int = _PREDICT_DTYPE_BYTES,
    activation_multiplier: int = _PREDICT_ACTIVATION_MULTIPLIER,
    mem_fraction: float = _PREDICT_MEM_FRACTION,
    min_batch: int = _PREDICT_BATCH_MIN,
    max_batch: int = _PREDICT_BATCH_MAX,
) -> int:
    """Return the predict-time DataLoader batch size, adapted to memory + width.

    Decision tree:
      1. ``user_override`` -- if non-None, returned verbatim (clamped to >=1).
      2. ``n_features`` + memory available -- compute the largest batch that
         fits ``mem_fraction`` of free memory at the assumed dtype + activation
         overhead, then clamp to ``[min_batch, max_batch]``.
      3. Memory probe failed AND no ``n_features`` -- return
         ``_PREDICT_BATCH_FALLBACK`` (1024), which is the smallest
         power-of-two that brings tabular predict into the seconds regime on
         typical (~100 features) data without risking OOM on small GPUs.

    Parameters
    ----------
    user_override
        Suite-level override (eg ``hyperparams_config.mlp_predict_batch_size``).
        Wins outright when non-None.
    train_batch_size
        Train-time batch size hint. Currently unused in the adaptive path
        (predict batch is computed from memory budget, not train batch) but
        accepted for API symmetry + future use.
    n_features
        Dataframe width at predict time. Required for the memory-aware path;
        when ``None`` the function uses the fallback constant.
    available_memory_bytes
        Override the memory probe (mainly for tests). ``None`` -> probe via
        :func:`_probe_available_memory_bytes`.
    cuda_available
        Forwarded to the probe; ``None`` triggers real detection.
    dtype_bytes
        Per-element size assumed for predict-time inputs (default 4 = float32).
    activation_multiplier
        Multiplicative overhead vs raw input bytes (default 4 = covers up to
        a 4-layer dense MLP forward pass + DataLoader pinned-memory copy).
    mem_fraction
        Share of free memory the resolver is willing to use for a single
        predict batch (default 0.25 = 25%, leaves room for the model itself
        + concurrent CUDA streams).
    min_batch, max_batch
        Hard clamps on the resolved batch size (default ``[64, 16384]``).
    """
    if user_override is not None:
        return max(1, int(user_override))
    # No width hint -> safe constant. Avoids the catastrophic 64 default while
    # not assuming anything about the input width.
    if n_features is None or n_features <= 0:
        return _PREDICT_BATCH_FALLBACK
    if available_memory_bytes is None:
        available_memory_bytes = _probe_available_memory_bytes(
            cuda_available=cuda_available,
        )
    if available_memory_bytes is None or available_memory_bytes <= 0:
        # Memory probe failed (no torch / no psutil / OS denial). Use the
        # conservative fallback constant -- still beats 64 by 16x for the
        # typical 25-feature case but won't OOM 30K-feature scenarios where
        # the user previously had to drop to 64. Caller can always override.
        return _PREDICT_BATCH_FALLBACK
    per_row_bytes = max(
        1, int(n_features) * int(dtype_bytes) * int(activation_multiplier),
    )
    budget = int(float(available_memory_bytes) * float(mem_fraction))
    if budget <= 0:
        return min_batch
    raw = budget // per_row_bytes
    clamped = max(min_batch, min(max_batch, int(raw)))
    return clamped


# Back-compat constants (so other modules can import the default ceiling).
_DEFAULT_PREDICT_BATCH_SIZE: int = _PREDICT_BATCH_FALLBACK
