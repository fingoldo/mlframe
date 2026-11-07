"""The baseline diagnostics AUC runs its ranking kernel inside the numba parallel guard.

Ablation drops are scored on joblib threads, and the AUC argsort is a prange kernel at n >= 200k. Without the guard, concurrent drops enter
the same parallel region, which numba's workqueue layer aborts on.
"""

from __future__ import annotations

import contextlib
import importlib

import numpy as np

diag = importlib.import_module("mlframe.training.baselines.diagnostics")


def test_baseline_auc_enters_the_parallel_guard(monkeypatch):
    """Computing an AUC through the quick metric records at least one entry into the parallel-kernel guard."""
    entries = []

    @contextlib.contextmanager
    def _spy_guard():
        """Count one entry into the guard and yield."""
        entries.append(1)
        yield

    monkeypatch.setattr(diag, "parallel_kernel_entry", _spy_guard, raising=False)
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.4 * y + 0.3 + rng.normal(0.0, 0.2, 300), 0.0, 1.0)

    value = diag._compute_metric(y, p, "AUC")

    assert np.isfinite(value)
    assert len(entries) >= 1, "the baseline AUC ran its ranking kernel outside the parallel guard"
