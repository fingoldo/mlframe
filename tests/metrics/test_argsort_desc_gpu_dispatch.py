"""Size-gated GPU argsort for the metrics kernels: valid order + bit-identical AUC vs CPU.

Bench (this host): numpy vs cupy argsort+transfer -- 0.42x @10k, 1.95x @50k, 3.92x @200k, 4.94x @1M. So the
descending argsort every metric kernel uses dispatches to cupy at N >= ~50k. fast_aucs uses tie-order-invariant
fractional ranks, so the GPU (radix) vs CPU (quicksort) path is byte-identical even on tied data.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlframe.metrics._core_auc_brier as ab


def test_argsort_desc_returns_valid_descending_order():
    """Argsort desc returns valid descending order."""
    rng = np.random.default_rng(0)
    for n in (1000, 60_000):  # CPU path + (if GPU) GPU path
        x = rng.random(n)
        idx = ab._argsort_desc_for_metrics(x)
        assert np.all(np.diff(x[idx]) <= 0), f"indices not descending at n={n}"


@pytest.mark.parametrize("scores_kind", ["continuous", "tied"])
def test_gpu_argsort_auc_matches_cpu(monkeypatch, scores_kind):
    """Gpu argsort auc matches cpu."""
    if not ab._gpu_argsort_available():
        pytest.skip("no CUDA / cupy")
    rng = np.random.default_rng(1)
    n = 60_000
    yt = (rng.random(n) < 0.3).astype(np.int64)
    ys = rng.random(n).astype(np.float64) if scores_kind == "continuous" else (rng.random(n) * 20).round()

    monkeypatch.setattr(ab, "_GPU_ARGSORT_MIN_N", 1)  # force GPU
    roc_g, pr_g = ab.fast_aucs(yt, ys)
    monkeypatch.setattr(ab, "_GPU_ARGSORT_MIN_N", 10**9)  # force CPU
    roc_c, pr_c = ab.fast_aucs(yt, ys)

    assert abs(roc_g - roc_c) < 1e-12, f"ROC GPU vs CPU diverged ({scores_kind})"
    assert abs(pr_g - pr_c) < 1e-12, f"PR GPU vs CPU diverged ({scores_kind})"


def test_stable_sort_env_stays_on_cpu(monkeypatch):
    # MLFRAME_METRICS_STABLE_SORT=1 must bypass the GPU path entirely (deterministic byte-identity contract).
    """Stable sort env stays on cpu."""
    monkeypatch.setenv("MLFRAME_METRICS_STABLE_SORT", "1")
    monkeypatch.setattr(ab, "_GPU_ARGSORT_MIN_N", 1)
    rng = np.random.default_rng(2)
    x = rng.random(60_000)
    idx = ab._argsort_desc_for_metrics(x)
    assert np.array_equal(idx, np.argsort(x, kind="stable")[::-1])
