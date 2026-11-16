"""Host or device choice of the FE batcher (kernel-tuning entry ``fe_batch_backend_choice``).

``fe_batch_mi`` scores a candidate matrix by edge-binned plug-in MI on the numba path (``_fe_cpu_batch``) or the resident device path (``_fe_gpu_batch``); the two select the same forms
(``test_fe_batch_parity``), so the choice is a speed choice. It used to read an ``fe_batch_backend`` key nobody ever wrote (the dispatcher's note: "the GTX 1050 Ti measured 0.5x for the resident
pack-and-batch pattern" was an older measurement), so AUTO always took the host. The sweep here times ``fe_batch_mi`` itself, the way production calls it, over the candidate work ``n x K``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast

import numpy as np

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

_SWEEP_WORK = [200_000, 2_000_000, 20_000_000, 80_000_000]
_SWEEP_ROWS = 20_000
_NBINS = 10
_SALT = 1
DEVICE_FALLBACK_WORK = 200_000  # before any sweep: the device from the smallest swept work (20000 x 10), where it was already ahead (and 7 to 11 times ahead from 4e6)


def _inputs(dims: dict) -> tuple:
    """``(X, y)``: ``dims["n_work"] / 20000`` float32 normal columns of 20000 rows and a 10-class target."""
    rng = np.random.default_rng(0)
    k = max(1, int(dims["n_work"]) // _SWEEP_ROWS)
    return (rng.standard_normal((_SWEEP_ROWS, k)).astype(np.float32), rng.integers(0, _NBINS, _SWEEP_ROWS).astype(np.int64))


def _host(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The numba path (``cpu_fe_batch_mi`` of ``_fe_cpu_batch``) as the dispatcher calls it."""
    from ._fe_cpu_batch import cpu_fe_batch_mi

    return np.asarray(cpu_fe_batch_mi(X, y, _NBINS), dtype=np.float64)  # Intentional float64: comparison vector


def _device(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The device path (``multi_gpu_fe_batch_mi`` of ``_fe_gpu_batch``, in the dtype the dispatcher uses) with no silent CPU fallback: a failure must fail the sweep, not time the host twice."""
    import cupy as cp

    from ._fe_gpu_batch import multi_gpu_fe_batch_mi
    from ._fe_gpu_batch._devices import fe_gpu_f32_enabled

    out = np.asarray(multi_gpu_fe_batch_mi(X, y, _NBINS, dtype=np.float32 if fe_gpu_f32_enabled() else np.float64), dtype=np.float64)  # Intentional float64: comparison vector and the dispatcher's strict default
    cp.cuda.Device().synchronize()
    return out


def _sweep() -> list:
    """Time both over the candidate work; the MI vectors must agree to 1e-9 (measured 6e-16 to 9e-16)."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return cast(list, sweep_backend_grid({"host": _host, "device": _device}, {"n_work": _SWEEP_WORK}, _inputs, reference="host", repeats=2, equiv_rtol=1e-9, equiv_atol=1e-9))


def _fallback_choice(n_work: int) -> str:
    """Pre-sweep choice from the work mark."""
    return "device" if int(n_work) >= DEVICE_FALLBACK_WORK else "host"


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="fe_batch_backend_choice",
        variant_fns=(),  # both backends are timed by the sweep callable
        tuner=_sweep,
        axes={"n_work": list(_SWEEP_WORK)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_batch_backend_choice",
    )
except Exception as e:
    logger.warning("fe_batch_backend_choice kernel_tuner registration failed: %s", e)
    _SPEC = None


def fe_batch_use_device(n_rows: int, n_cands: int) -> bool:
    """Whether the FE batcher scores an ``n_rows x n_cands`` matrix on the device (outside STRICT, which forces it): the per-host tuning cache, else the work mark. ``MLFRAME_FE_BATCH_DEVICE=0|1`` forces a side."""
    forced = os.environ.get("MLFRAME_FE_BATCH_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    work = int(n_rows) * int(n_cands)
    if _SPEC is None:
        return _fallback_choice(work) == "device"
    try:
        return str(_SPEC.choose(n_work=work)) == "device"
    except Exception as exc:
        log_throttle(logger, "_fe_batch_dispatch_ktc.fe_batch_use_device", logging.WARNING, "fe_batch_use_device: KTC choose() failed, using the work mark: %s", exc)
        return _fallback_choice(work) == "device"
