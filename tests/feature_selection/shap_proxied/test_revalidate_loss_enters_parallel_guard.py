"""The shap-proxy revalidation loss runs its metric kernels inside the numba parallel guard.

Revalidation scores holdouts on joblib threads. At n >= 100k the size-dispatched metrics enter prange reductions, and numba's workqueue
layer aborts when several threads are inside one at once. The loss entry point therefore enters ``parallel_kernel_entry``.
"""

from __future__ import annotations

import contextlib
import importlib

import numpy as np

mod = importlib.import_module("mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_loss")


def test_revalidation_loss_enters_the_parallel_guard(monkeypatch):
    """Scoring a binary holdout records at least one entry into the parallel-kernel guard."""
    entries = []

    @contextlib.contextmanager
    def _spy_guard():
        """Count one entry into the guard and yield."""
        entries.append(1)
        yield

    monkeypatch.setattr(mod, "parallel_kernel_entry", _spy_guard, raising=False)
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400).astype(np.float64)
    p = np.clip(0.5 * y + 0.25 + rng.normal(0.0, 0.1, 400), 0.0, 1.0)

    value = mod._loss_from_predictions(p, y, True, "brier")

    assert np.isfinite(value)
    assert len(entries) >= 1, "the holdout loss ran its metric kernels outside the parallel guard"
