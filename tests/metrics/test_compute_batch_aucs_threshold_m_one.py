"""iter#194 regression: compute_batch_aucs auto-dispatch must choose CPU at M=1, N>=100k.

Pre-fix: ``_GPU_BATCH_THRESHOLD_M=1`` dispatched GPU for every binary-classification
single-target call, but the GPU AUC kernels carry ~1-3 s of cupy compile + host<->device
overhead per call plus a Python loop over M columns. For M=1 on N=1M the GPU path was
~10x slower than the numba CPU fast_aucs, inflating compute_batch_aucs to ~32 s of the
~55 s suite wall on iter#194 (binary x linear,ridge,sgd x 1M rows).

Post-fix: ``_GPU_BATCH_THRESHOLD_M=5`` matches the ``gpu_multiple_*_auc_scores``
docstring guidance ("Use when N >= 100k AND M >= 5"). Single-target callers go CPU; only
multi-target / multiclass with K>=5 columns dispatch GPU.

This test asserts the auto-dispatch decision (without depending on GPU presence) and
sanity-checks the CPU numpy correctness against ``sklearn.metrics.roc_auc_score``.
"""

from __future__ import annotations

import numpy as np
import pytest

# Deterministic numerical dispatch + small-N CPU correctness checks; wall <1s.
pytestmark = [pytest.mark.fast]


def test_auto_dispatch_picks_cpu_at_m_one_n_large():
    """``_resolve_backend(None, N=1M, M=1)`` should return False (CPU)."""
    from mlframe.metrics.core import _resolve_backend, _GPU_BATCH_THRESHOLD_M

    # M=1 must NOT dispatch GPU regardless of cupy availability after the threshold fix.
    assert _GPU_BATCH_THRESHOLD_M >= 5, (
        f"_GPU_BATCH_THRESHOLD_M={_GPU_BATCH_THRESHOLD_M} would re-introduce the iter#194 "
        "regression where every binary single-target call paid cupy compile + host<->device "
        "overhead. Keep the threshold at >=5 to match the gpu_multiple_*_auc_scores "
        '"Use when N >= 100k AND M >= 5" docstring contract.'
    )
    assert _resolve_backend(None, 1_000_000, 1) is False
    assert _resolve_backend(None, 5_000_000, 1) is False


def test_compute_batch_aucs_m_one_matches_sklearn():
    """End-to-end: compute_batch_aucs at M=1 on N=200k matches sklearn ROC AUC bit-for-bit."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    from mlframe.metrics.core import compute_batch_aucs

    rng = np.random.default_rng(2026051906)
    n = 200_000
    y_true = rng.integers(0, 2, size=n).astype(np.int32)
    y_score = rng.random(size=n).astype(np.float64) + 0.2 * y_true

    roc_arr, pr_arr = compute_batch_aucs(y_true, y_score)
    assert roc_arr.shape == (1,)
    assert pr_arr.shape == (1,)

    expected_roc = roc_auc_score(y_true, y_score)
    expected_pr = average_precision_score(y_true, y_score)
    assert float(roc_arr[0]) == pytest.approx(expected_roc, abs=1e-9)
    assert float(pr_arr[0]) == pytest.approx(expected_pr, abs=1e-9)
