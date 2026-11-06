"""The pre-sweep fallback shared by the row-count-gated device kernels: take the device from a measured row count, else stay on the host."""

from __future__ import annotations


def device_from_rows(n_samples: int, min_rows: int) -> str:
    """``"device"`` when ``n_samples`` reaches ``min_rows`` (the measured crossover of the kernel), else ``"host"``."""
    return "device" if int(n_samples) >= int(min_rows) else "host"
