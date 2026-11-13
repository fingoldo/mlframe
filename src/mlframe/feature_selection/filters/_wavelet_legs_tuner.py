"""Host or device for the wavelet leg selection of one column, decided by the kernel-tuning cache on the row count.

``_wavelet_basis_fe._select_wavelet_legs`` has two bodies for the same admitted legs: the fused parallel ``@njit`` body on the host and the device-born route of
``_wavelet_basis_fe_batched`` (leg code matrices built on the card from the resident column, scored by the batched MI kernel). Which is faster depends on the row count: measured per
column (GTX 1050 Ti, one column, 5 runs): 5000 rows host 1.5 ms / device 2.4, 20000 rows 2.6 / 2.5, 100000 rows 7.5 / 5.2, 400000 rows 42.3 / 18.5, the same legs everywhere.
The switch used to be the strict-residency flag alone, whatever the row count; this entry makes the row count part of the decision. ``MLFRAME_CMI_GPU=1`` still forces the device.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from ._ktc_fallback import host_device_sweep, rows_fallback

logger = logging.getLogger(__name__)

DEVICE_FALLBACK_ROWS = 20_000  # measured crossover; the tuner replaces it per host once a sweep has run
_SWEEP_ROWS = [5_000, 20_000, 100_000, 400_000]
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """A column of ``dims["n_samples"]`` rows with one planted band and a noisy binary label, plus its bounds."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    x = rng.random(n)
    y = ((x > 0.55) & (x < 0.7)).astype(np.int64) ^ (rng.random(n) < 0.05).astype(np.int64)
    return (x, y, float(x.min()), float(x.max() - x.min()))


def _as_vector(legs: list) -> np.ndarray:
    """The admitted legs as a fixed-length vector (rank + 1 at the leg's slot), so two backends are compared on set AND order."""
    from ._wavelet_basis_fe import _WAVELET_MAX_SCALE

    out = np.zeros((1 << (int(_WAVELET_MAX_SCALE) + 1)) - 1, dtype=np.float64)  # Intentional float64: comparison vector
    for rank, (j, k) in enumerate(legs):
        out[(1 << int(j)) - 1 + int(k)] = rank + 1
    return out


def _sweep_host(x: np.ndarray, y: np.ndarray, lo: float, span: float) -> np.ndarray:
    """The fused host body, for the sweep."""
    from ._wavelet_basis_fe import _select_wavelet_legs

    return _as_vector(_select_wavelet_legs(x, y, lo, span, use_device=False))


def _sweep_device(x: np.ndarray, y: np.ndarray, lo: float, span: float) -> np.ndarray:
    """The device-born route, for the sweep; a failed launch raises so the sweep does not record a host fallback as a device time."""
    from ._wavelet_basis_fe import _WAVELET_MAX_LEGS, _WAVELET_MAX_SCALE, _WAVELET_SCALE_SIGMA
    from ._wavelet_basis_fe_batched import _select_wavelet_legs_batched_device

    legs = _select_wavelet_legs_batched_device(x, y, lo, span, max_scale=_WAVELET_MAX_SCALE, max_legs=_WAVELET_MAX_LEGS, scale_sigma=_WAVELET_SCALE_SIGMA)
    if legs is None:
        raise RuntimeError("device-born wavelet leg selection unavailable")
    return _as_vector(list(legs))


_fallback_choice = rows_fallback(__name__)

try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="fe_wavelet_legs_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-9, atol=1e-12),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_wavelet_legs_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the row-count fallback decides
    logger.warning("kernel_tuner spec build for fe_wavelet_legs_backend failed: %s", e)
    _SPEC = None


def wavelet_device_chosen(n_rows: int) -> bool:
    """Whether ``n_rows`` rows take the device-born route: forced by ``MLFRAME_CMI_GPU=1``, else the per-host kernel-tuning cache (fallback ``DEVICE_FALLBACK_ROWS``)."""
    if os.environ.get("MLFRAME_CMI_GPU", "") == "1":
        return True
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("fe_wavelet_legs_backend: KTC choose() failed, using the row-count fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
