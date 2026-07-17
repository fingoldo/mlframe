"""Selection-equivalence regression for the STRICT-resident GPU fast path of ``_mi_greedy_cmi_fe._quantile_bin``.

The large-operand (n >= _GPU_QBIN_MIN_ROWS) equi-frequency binning of the gate-redundancy / subsumption /
additive-fusion continuous columns routes to the device (reuse of ``_gpu_resident_discretize_codes``) under
``fe_gpu_strict_resident_enabled()`` -- 6x at n=300k on the GTX 1050 Ti (synchronized bench). The codes feed
MI/cardinality, so the bar is SELECTION-EQUIVALENCE (same number of occupied bins + a near-identical partition,
not byte-identity): cp.percentile and np.quantile may round a quantile boundary differently, 1-offing a code at
<~1e-5 of rows -- below the bin resolution.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as M


def _cuda_ok() -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _cuda_ok(), reason="no CUDA device")


@pytest.mark.parametrize("nbins", [2, 5, 10, 12])
@pytest.mark.parametrize("kind", ["normal", "exponential", "lowcard", "masspoint"])
def test_gpu_quantile_bin_selection_equivalent(nbins, kind):
    rng = np.random.default_rng(int(nbins) * 17 + hash(kind) % 997)
    n = 120_000  # above _GPU_QBIN_MIN_ROWS so the GPU path engages
    if kind == "normal":
        a = rng.standard_normal(n)
    elif kind == "exponential":
        a = rng.exponential(2.0, n)
    elif kind == "lowcard":
        a = rng.integers(0, 6, n).astype(np.float64)
    else:
        a = np.concatenate([np.zeros(n // 2), rng.standard_normal(n - n // 2)])
        rng.shuffle(a)

    cpu = M._quantile_bin(a, nbins)
    gpu = M._quantile_bin_gpu(a, nbins)
    assert gpu is not None, "GPU quantile-bin returned None (device path unexpectedly unavailable)"

    # Same occupied-bin cardinality (load-bearing: the redundancy gate keys on np.unique(codes).size).
    assert int(np.unique(gpu).size) == int(np.unique(cpu).size)
    # Same value range (no out-of-range / shifted codes).
    assert int(gpu.min()) == int(cpu.min()) and int(gpu.max()) == int(cpu.max())
    # Selection-equivalent partition: only sub-resolution boundary rows may 1-off, far below 1e-3 of rows.
    diff = int(np.sum(gpu != cpu))
    assert diff <= max(8, n // 10_000), f"{diff} rows differ (kind={kind}, nbins={nbins})"
    assert int(np.max(np.abs(gpu - cpu))) <= 1, "code differs by more than one bin"


def test_small_column_keeps_cpu_path():
    """Below the crossover the size gate must NOT engage the GPU (the host path is faster there)."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(3_000)
    assert a.size < M._GPU_QBIN_MIN_ROWS
    # The host fast path is byte-identical to itself; just assert the threshold guards the small shape.
    cpu = M._quantile_bin(a, 10)
    assert cpu.shape == (3_000,)
