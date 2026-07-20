"""GPU OOM circuit-breaker recovery (mrmr_audit_2026-07-20 edge_cases.md #143): a simulated CUDA
OutOfMemoryError on one call to adds_nonlinear_value_batch_gpu_resident must fall back to CPU
(returns None) for THAT call only, and the very next call (with cupy behaving normally again) must
run the GPU path again -- no leftover module-level 'GPU is broken' flag permanently pins later
calls to CPU. There is no persistent circuit-breaker state in this module (every cupy/device fault
is caught per-call), so this test is the regression guard against ever introducing one that doesn't
reset cleanly."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._fe_pure_form_retention_gpu_resident import (
    adds_nonlinear_value_batch_gpu_resident,
)


def _pool(n=400, seed=0):
    """A single non-separable, relevant pair candidate."""
    rng = np.random.default_rng(seed)
    xa = rng.standard_normal(n)
    xb = rng.uniform(0.5, 3.0, n)
    y = xa**2 / xb + 0.05 * rng.standard_normal(n)
    form_genuine = xa**2 / xb
    return dict(form_values=[form_genuine], src_pairs=[("a", "b")], base_names=["a", "b"], base_columns=[xa, xb], rel_y=y)


def test_simulated_oom_on_one_call_falls_back_then_next_call_uses_gpu_again(monkeypatch):
    """First call: cp.linalg.solve raises OutOfMemoryError once -> the batch call returns None
    (graceful CPU fallback). Second call: solve behaves normally again -> the SAME function call
    returns a real (non-None) verdict list, proving no permanent CPU-pinning leaked between calls."""
    pool = _pool()
    real_solve = cp.linalg.solve
    call_count = {"n": 0}

    def _flaky_solve(*args, **kwargs):
        """Raises OutOfMemoryError on the first call only, then delegates to the real cp.linalg.solve."""
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise cp.cuda.memory.OutOfMemoryError(1, 1)
        return real_solve(*args, **kwargs)

    monkeypatch.setattr(cp.linalg, "solve", _flaky_solve)

    first = adds_nonlinear_value_batch_gpu_resident(
        pool["form_values"], pool["src_pairs"], pool["base_names"], pool["base_columns"], pool["rel_y"], min_resid_frac=0.10, min_resid_corr=0.08
    )
    assert first is None, "the simulated OOM must fall back to None (CPU path), not raise or propagate"

    second = adds_nonlinear_value_batch_gpu_resident(
        pool["form_values"], pool["src_pairs"], pool["base_names"], pool["base_columns"], pool["rel_y"], min_resid_frac=0.10, min_resid_corr=0.08
    )
    assert second is not None, "the GPU path must run again on the very next call, not stay permanently pinned to CPU"
    assert call_count["n"] == 2, "both calls must have actually reached the (mocked) solve, confirming no early-exit caching hid the second attempt"
