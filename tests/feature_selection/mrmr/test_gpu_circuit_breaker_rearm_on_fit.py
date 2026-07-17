"""Regression test for MRMR audit finding #31 (2026-07-09 fix).

The CMI / mi_direct / pair-maxT GPU circuit breakers are process-global: once ANY of them trips (a
launch fault poisons the CUDA context), every LATER call in the process permanently skips the GPU --
even fits that have nothing to do with whatever caused the original fault. In a long-lived worker
(notebook kernel, service process) this silently degrades every subsequent ``MRMR().fit()`` call forever
after one transient GPU hiccup.

The fix re-arms all three breakers at the top of ``MRMR.fit()``, bounding the cost of a genuinely-broken
GPU to one extra failed attempt per fit (the breaker still protects WITHIN a fit from repeated futile
retries) while letting a transient fault self-heal on the next fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.info_theory import _cmi_cuda
from mlframe.feature_selection.filters import permutation as _permutation_mod
from mlframe.feature_selection.filters import _permutation_null_pair_resident as _pair_resident_mod
from mlframe.feature_selection.filters.mrmr import MRMR


@pytest.fixture(autouse=True)
def _restore_breakers():
    """Never leak a tripped/reset breaker state into other test modules. Sets the module globals
    directly rather than via the reset_*() functions, which a test may have monkeypatched."""
    yield
    _cmi_cuda._CMI_GPU_FAILED = False
    _permutation_mod._MI_DIRECT_GPU_FAILED = False
    _pair_resident_mod._PAIR_MAXT_GPU_FAILED = False


def _make_data(n=300, p=5, seed=0):
    """Make data."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    y = pd.Series(((X["x0"] + X["x1"]) > 0).astype(np.int64), name="y")
    return X, y


def test_fit_rearms_cmi_gpu_breaker():
    """Fit rearms cmi gpu breaker."""
    _cmi_cuda._CMI_GPU_FAILED = True  # simulate a fault tripped by a PRIOR, unrelated fit
    X, y = _make_data(seed=1)
    MRMR(verbose=0, random_seed=42, fe_max_steps=0).fit(X, y)
    assert _cmi_cuda._CMI_GPU_FAILED is False, "fit() must re-arm the CMI GPU circuit breaker"


def test_fit_rearms_mi_direct_gpu_breaker():
    """Fit rearms mi direct gpu breaker."""
    _permutation_mod._MI_DIRECT_GPU_FAILED = True
    X, y = _make_data(seed=2)
    MRMR(verbose=0, random_seed=42, fe_max_steps=0).fit(X, y)
    assert _permutation_mod._MI_DIRECT_GPU_FAILED is False, "fit() must re-arm the mi_direct GPU circuit breaker"


def test_fit_rearms_pair_maxt_gpu_breaker():
    """Fit rearms pair maxt gpu breaker."""
    _pair_resident_mod._PAIR_MAXT_GPU_FAILED = True
    X, y = _make_data(seed=3)
    MRMR(verbose=0, random_seed=42, fe_max_steps=0).fit(X, y)
    assert _pair_resident_mod._PAIR_MAXT_GPU_FAILED is False, "fit() must re-arm the pair-maxT GPU circuit breaker"


def test_breaker_reset_is_resilient_to_missing_gpu_modules(monkeypatch):
    """If a GPU submodule import itself fails (e.g. cupy uninstalled), fit() must not raise -- the
    re-arm is a best-effort resilience nicety, never a hard fit() dependency."""

    def _boom():
        """Helper that boom."""
        raise ImportError("simulated missing GPU module")

    # Patch the re-arm targets' own reset functions to raise, proving fit() swallows the failure.
    monkeypatch.setattr(_cmi_cuda, "reset_cmi_gpu_circuit_breaker", _boom)
    X, y = _make_data(seed=4)
    MRMR(verbose=0, random_seed=42, fe_max_steps=0).fit(X, y)  # must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
