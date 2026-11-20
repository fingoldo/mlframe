"""The silent CPU fallbacks of the hermite prewarp seed and of the polyeval threshold lookup log a warning with the numbers that decided them."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


def test_warm_start_als_seed_logs_when_the_gpu_sweep_fails(monkeypatch, caplog):
    """A device fault in the resident ALS sweep falls back to the CPU path and says so."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._gpu_strict_fe import _entry
    from mlframe.feature_selection.filters.hermite_fe import _hermite_prewarp as hp
    from mlframe.feature_selection.filters.hermite_fe import _hermite_prewarp_gpu_resident as res

    def _fault(*a, **k):
        """Stand-in for a cuSOLVER-class fault."""
        raise np.linalg.LinAlgError("simulated device linalg fault")

    monkeypatch.setattr(_entry, "fe_gpu_strict_resident_enabled", lambda: True)
    monkeypatch.setattr(res, "warm_start_als_seed_gpu", _fault)
    monkeypatch.setattr(res, "warm_start_als_seed_gpu_from_z", _fault)
    rng = np.random.default_rng(0)
    Ba, Bb, y = rng.normal(size=(300, 4)), rng.normal(size=(300, 4)), rng.normal(size=300)
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING):
        out = hp.warm_start_als_seed(Ba, Bb, y, iters=2)
    assert out[0] is not None and out[1] is not None
    assert any("GPU-resident ALS sweep failed" in r.getMessage() for r in caplog.records)


def test_polyeval_threshold_lookup_failure_is_logged(monkeypatch, caplog):
    """A broken kernel-tuning cache falls back to the source thresholds and says so."""
    import pyutilz.performance.kernel_tuning.cache as ktc
    from mlframe.feature_selection.filters.hermite_fe import _hermite_oracle as ho

    def _boom(*a, **k):
        """Stand-in for a broken tuning cache."""
        raise RuntimeError("tuning cache unavailable")

    monkeypatch.setattr(ktc.KernelTuningCache, "load_or_create", staticmethod(_boom))
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING):
        got = ho._lookup_polyeval_thresholds("hermite", 1000)
    assert got == (ho._PAR_THRESHOLD, ho._CUDA_THRESHOLD)
    assert any("kernel_tuning_cache lookup failed" in r.getMessage() for r in caplog.records)
