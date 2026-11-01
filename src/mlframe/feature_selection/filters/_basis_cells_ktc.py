"""Kernel-tuning-cache dispatch of the device basis-cell path (conditional routing and adaptive-degree FE).

Both families can build and score their (column, basis, degree) cells on the host (one vector per cell, stacked MI) or on the device (batched Clenshaw kernels, MI on the device). The device wins at
every size measured on the development GPU, but a faster host or a slower card moves the crossover, so the choice is looked up per host from the kernel-tuning cache (sweep over the row count, fastest
variant whose MI agrees with the host's), with the measured fallback ``n >= BASIS_CELLS_DEVICE_FALLBACK_ROWS``. ``MLFRAME_BASIS_CELLS_DEVICE=0|1`` forces the host or the device path.

GPU-only: without cupy the sweep never runs, ``use_device_cells`` returns False.
"""

from __future__ import annotations

import logging
import os
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

#: Fallback crossover when the cache has no entry: the MI binning of the device (edge selection) and of the host differ visibly on frames of a few hundred rows; speed favours the device from 2000 rows.
BASIS_CELLS_DEVICE_FALLBACK_ROWS = 5_000
_SWEEP_ROWS = [2_000, 5_000, 20_000, 100_000]
_SWEEP_COLUMNS = 6
_BASES = ("hermite", "legendre", "chebyshev", "laguerre")
_DEGREES = (2, 3)
_SALT = 1


def _make_inputs(dims: dict) -> tuple:
    """A frame of ``_SWEEP_COLUMNS`` float32 columns with a nonlinear binary target, ``dims["n_samples"]`` rows."""
    import pandas as pd

    from ._orthogonal_shared import coerce_y_classif

    n = int(dims["n_samples"])
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n).astype(np.float32) for i in range(_SWEEP_COLUMNS)})
    y = ((X["x0"] ** 2 + 0.5 * X["x1"] ** 3 + rng.normal(size=n) * 0.3) > 1).astype(np.int64).to_numpy()
    return (X, coerce_y_classif(y))


def _host_variant(X, ya):
    """Host enumeration + stacked MI of the routing cells (the CPU twin)."""
    from ._orthogonal_routing_fe import PRE_TRANSFORM_NAMES, _enumerate_routing_cells, _mi_classif_batch, _stack_candidates

    _names, values, _meta = _enumerate_routing_cells(X, list(X.columns), PRE_TRANSFORM_NAMES, _BASES, _DEGREES)
    return np.asarray(_mi_classif_batch(_stack_candidates(values), ya, nbins=10), dtype=np.float64)  # Intentional float64: MI statistic vector


def _device_variant(X, ya):
    """Device cells + MI of the same routing workload."""
    from ._orthogonal_routing_device import routing_scores_on_device
    from ._orthogonal_routing_fe import PRE_TRANSFORM_NAMES

    got = routing_scores_on_device(X, list(X.columns), PRE_TRANSFORM_NAMES, _BASES, _DEGREES, ya, 10, "corr", force=True)
    if got is None:
        raise RuntimeError("device basis cells unavailable")
    return np.asarray(got[2], dtype=np.float64)  # Intentional float64: MI statistic vector


def _run_sweep() -> list:
    """Time the host and the device variant per row count; the fastest variant whose MI agrees with the host's (atol 1e-3: the device bins by edge selection) wins each region."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return cast(
        list,
        sweep_backend_grid(
            {"host": lambda X, ya: _host_variant(X, ya), "device": lambda X, ya: _device_variant(X, ya)},
            {"n_samples": _SWEEP_ROWS},
            _make_inputs,
            reference="host",
            repeats=3,
            equiv_rtol=1e-2,
            equiv_atol=1e-3,
        ),
    )


def _fallback_choice(n_samples: int) -> str:
    """Pre-sweep fallback: the measured crossover."""
    return "device" if int(n_samples) >= BASIS_CELLS_DEVICE_FALLBACK_ROWS else "host"


try:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec, kernel_tuner

    _SPEC: "TunerSpec | None" = kernel_tuner(
        kernel_name="fe_basis_cells_device",
        variant_fns=(),  # GPU-only cupy path; covered by salt
        tuner=_run_sweep,
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_basis_cells_device",
    )
except Exception as e:
    logger.warning("kernel_tuner spec build for fe_basis_cells_device failed: %s", e)
    _SPEC = None


def use_device_cells(n_rows: int) -> bool:
    """Whether the basis cells of an ``n_rows`` frame are built on the device: env override, else the per-host cache, else the measured fallback."""
    forced = os.environ.get("MLFRAME_BASIS_CELLS_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    if _SPEC is None:
        return _fallback_choice(n_rows) == "device"
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("fe_basis_cells_device: KTC choose() failed, using the fallback: %s", exc)
        return _fallback_choice(n_rows) == "device"
