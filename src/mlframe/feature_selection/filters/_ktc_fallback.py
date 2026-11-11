"""The pre-sweep fallback shared by the row-count-gated device kernels: take the device from a measured row count, else stay on the host."""

from __future__ import annotations

import sys


def device_from_rows(n_samples: int, min_rows: int) -> str:
    """``"device"`` when ``n_samples`` reaches ``min_rows`` (the measured crossover of the kernel), else ``"host"``."""
    return "device" if int(n_samples) >= int(min_rows) else "host"


def rows_fallback(owner: str, name: str = "DEVICE_FALLBACK_ROWS"):
    """The pre-sweep fallback of a row-count-gated kernel: ``device_from_rows`` with the threshold read from the module named ``owner`` (pass ``__name__``) at call
    time, so a test or an operator lowering the module constant takes effect."""

    def _fallback_choice(n_samples: int) -> str:
        """Pre-sweep fallback: the device from the owner module's row threshold."""
        return device_from_rows(n_samples, getattr(sys.modules[owner], name))

    return _fallback_choice


def host_device_sweep(owner: str, host, device, *, repeats: int = 3, rtol: float = 1e-9, atol: float | None = None):
    """The kernel-tuning sweep of a host-versus-device kernel: time both per row count and require the device to agree with the host.

    ``host`` and ``device`` are the two callables; ``owner`` is the module's ``__name__``, in which ``_SWEEP_ROWS`` and ``_sweep_inputs`` are looked up at call time, so the module
    can define them after the sweep is built. ``atol`` defaults to ``rtol``."""

    def _run_sweep() -> list:
        """Time the host and the device kernel per row count; the device must agree with the host."""
        from pyutilz.dev.benchmarking import sweep_backend_grid

        module = sys.modules[owner]
        return list(
            sweep_backend_grid(
                {"host": host, "device": device},
                {"n_samples": module._SWEEP_ROWS},
                module._sweep_inputs,
                reference="host",
                repeats=repeats,
                equiv_rtol=rtol,
                equiv_atol=rtol if atol is None else atol,
            )
        )

    return _run_sweep
