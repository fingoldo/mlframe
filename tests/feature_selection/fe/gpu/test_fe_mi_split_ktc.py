"""P-10: the fused FE MI kernel's single-vs-split-N launch choice consults the kernel_tuning_cache.

The prior code hardcoded ``K < 48 and n >= 262144`` (a 6-SM GTX 1050 Ti crossover). That is now a lookup
via ``lookup_fe_mi_split_backend`` with the constants preserved as the measurement-backed fallback + an
``MLFRAME_FE_MI_SPLIT`` env override. Both launch legs are selection-equivalent (bit-identical MI), so the
choice can never move a selection decision; this pins that invariant + the fallback / override contract.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection._benchmarks.kernel_tuning_cache.dispatch import (
    _fallback_fe_mi_split,
    lookup_fe_mi_split_backend,
)


def _cuda_ok() -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _cuda_ok(), reason="no CUDA device")


def test_fallback_preserves_hardcoded_crossover():
    # The fallback IS the old magic constants verbatim: split iff K<48 and n>=262144.
    assert _fallback_fe_mi_split(300_000, 10) == "split"
    assert _fallback_fe_mi_split(262_144, 47) == "split"
    assert _fallback_fe_mi_split(200_000, 10) == "single"   # n too small
    assert _fallback_fe_mi_split(300_000, 60) == "single"   # K too wide


def test_env_override_forces_leg(monkeypatch):
    monkeypatch.setenv("MLFRAME_FE_MI_SPLIT", "split")
    assert lookup_fe_mi_split_backend(100, 100) == "split"   # would be single by fallback
    monkeypatch.setenv("MLFRAME_FE_MI_SPLIT", "single")
    assert lookup_fe_mi_split_backend(1_000_000, 8) == "single"  # would be split by fallback


@pytest.mark.parametrize("n,k", [(500_000, 32), (300_000, 8)])
def test_single_and_split_legs_are_selection_equivalent(monkeypatch, n, k):
    from mlframe.feature_selection.filters._fe_batched_mi import binned_mi_from_values_gpu

    rng = np.random.default_rng(1)
    nbins, ky = 10, 3
    X = cp.asarray(rng.standard_normal((n, k)), dtype=cp.float32)
    E = cp.ascontiguousarray(cp.percentile(X.astype(cp.float64), cp.linspace(0, 100, nbins + 1), axis=0)[1:-1])
    y = cp.asarray(rng.integers(0, ky, size=n, dtype=np.int64))

    monkeypatch.setenv("MLFRAME_FE_MI_SPLIT", "single")
    mi_single = binned_mi_from_values_gpu(X, E, y, nbins, ky, codes_trusted=True)
    monkeypatch.setenv("MLFRAME_FE_MI_SPLIT", "split")
    mi_split = binned_mi_from_values_gpu(X, E, y, nbins, ky, codes_trusted=True)

    # Same plug-in MI + same codes -> bit-identical -> identical feature ranking (selection-equivalent).
    assert np.array_equal(np.argsort(mi_single), np.argsort(mi_split))
    assert int(np.argmax(mi_single)) == int(np.argmax(mi_split))
    assert np.max(np.abs(mi_single - mi_split)) == 0.0
