"""Every transform call through ``call_transform`` enters the numba parallel guard.

Discovery scores one spec per joblib thread, and the numba transforms it calls are parallel=True kernels. The gateway is the single entry
point for those calls, so it is where the guard belongs.
"""

from __future__ import annotations

import contextlib
import importlib

import numpy as np

gw = importlib.import_module("mlframe.training.composite.transforms._call_gateway")


class _Transform:
    """A minimal forward/inverse transform that records nothing; only the call path matters."""

    def forward(self, t_hat, base):
        """Identity forward."""
        return t_hat

    def inverse(self, t_hat, base):
        """Identity inverse."""
        return t_hat


def test_call_transform_enters_the_parallel_guard(monkeypatch):
    """A forward call through the gateway records at least one entry into the parallel-kernel guard."""
    entries = []

    @contextlib.contextmanager
    def _spy_guard():
        """Count one entry into the guard and yield."""
        entries.append(1)
        yield

    monkeypatch.setattr(gw, "parallel_kernel_entry", _spy_guard, raising=False)
    x = np.arange(5, dtype=np.float64)

    out = gw.call_transform(_Transform(), "forward", x, x)

    np.testing.assert_array_equal(out, x)
    assert len(entries) >= 1, "the transform call ran outside the parallel guard"
