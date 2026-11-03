"""Host-or-device choice of the FE scorers that gained a device path (grouped quantile, k-fold OOF MI, CMIM), each from its OWN kernel-tuning entry.

Each scorer's host path is its own code (per-column numpy binning, a fold pool, a per-candidate CMI loop), not the plug-in MI batch kernel, so the
choice cannot borrow ``lookup_mi_classif_backend``: that entry measures the batch kernel against the device and answered "host" for every multi-column
size on this machine, which kept the device paths (3.7x-170x faster than these scorers' host code) from ever running in the fit. The sweep here times
the scorer itself with the backend forced (``forced_backend``), on synthetic inputs of the swept size.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from typing import Any, Callable, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

_FORCED: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar("mlframe_scorer_forced_backend", default=None)
#: n_rows x n_columns from which the device is the fallback choice when the tuning cache has no entry (the device won 3.7x-170x at 30000-100000 rows
#: and 150-220 columns; small inputs stay on the host).
_DEVICE_MIN_WORK = 1_000_000
_CODE_VERSION = "fe_scorer_device_backend-v1"


@contextlib.contextmanager
def forced_backend(choice: str) -> Iterator[None]:
    """Force ``"host"`` or ``"device"`` for the scorers called inside (the sweep's timing, tests)."""
    token = _FORCED.set(choice)
    try:
        yield
    finally:
        _FORCED.reset(token)


def scorer_device_choice(name: str, n_rows: int, n_cols: int, sweep: "Callable[[], list] | None" = None) -> bool:
    """True when scorer ``name`` should run on the device for this size: the forced backend, else no usable GPU -> host, else the scorer's own
    kernel-tuning entry over (n_rows, n_cols) (async sweep ``sweep``), else the work fallback."""
    forced = _FORCED.get()
    if forced is not None:
        return forced == "device"
    try:
        from ._gpu_policy import cuda_available_for_run

        if not cuda_available_for_run():
            return False
    except Exception:
        return False
    fallback = "device" if int(n_rows) * int(n_cols) >= _DEVICE_MIN_WORK else "host"
    if sweep is None:
        return fallback == "device"
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        res = KernelTuningCache.load_or_create().get_or_tune(
            f"fe_scorer_device_{name}",
            dims={"n_rows": int(n_rows), "n_cols": int(n_cols)},
            tuner=sweep,
            axes=["n_rows", "n_cols"],
            fallback={"kernel_choice": fallback},
            code_version=_CODE_VERSION,
            async_sweep=True,
        )
        choice = res if isinstance(res, str) else (str(res.get("kernel_choice", fallback)) if res else fallback)
        return choice == "device"
    except Exception as exc:
        logger.debug("scorer %s backend selection failed: %r", name, exc)
        return fallback == "device"


def sweep_scorer(run: Callable[[Any], Any], make_inputs: Callable[[dict], Any]) -> list:
    """Per-host host-vs-device sweep of one scorer over (n_rows, n_cols): ``run(inputs)`` is the scorer call, timed with each backend forced; [] when the
    sweep helper is unavailable."""
    try:
        from pyutilz.dev.benchmarking import sweep_backend_grid
    except Exception as exc:
        logger.debug("scorer sweep unavailable (%s: %s)", type(exc).__name__, exc)
        return []

    def _host(*inputs):
        """The scorer with the host forced."""
        with forced_backend("host"):
            return run(inputs)

    def _device(*inputs):
        """The scorer with the device forced."""
        with forced_backend("device"):
            return run(inputs)

    return list(
        sweep_backend_grid(
            {"host": _host, "device": _device},
            {"n_rows": [10000, 30000, 100000], "n_cols": [20, 100, 300]},
            make_inputs,
            reference="host",
            repeats=2,
            equiv_rtol=1e-9,
            equiv_atol=1e-12,
            decision_key="kernel_choice",
        )
        or []
    )


def synthetic_scorer_frame(dims: dict, seed: int = 0) -> tuple:
    """``(raw (n, 20), engineered (n, n_cols), y)`` frames for the sweeps: engineered columns named ``"{source}__e{j}"`` after the raw ones, a binary
    target driven by the first raw column."""
    import pandas as pd

    n, k = int(dims["n_rows"]), int(dims["n_cols"])
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(rng.standard_normal((n, 20)).astype(np.float32), columns=[f"r{i}" for i in range(20)])
    eng = pd.DataFrame(rng.standard_normal((n, k)).astype(np.float32), columns=[f"r{j % 20}__e{j}" for j in range(k)])
    y = (raw["r0"].to_numpy() + rng.standard_normal(n) > 0).astype(np.int64)
    return raw, eng, y
