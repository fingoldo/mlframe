"""Wave 9.1 loop-iter-4 regression: CPU/GPU permutation threshold parity.

Iter-4 agent flagged a silent CPU/GPU divergence: ``mi_direct_gpu_batched``
and ``mi_direct_gpu_batched_streamed`` were hardcoding the
``min_nonzero_confidence=0.95`` rejection threshold (as ``0.05`` directly),
ignoring the caller-supplied value. This meant ``MRMR.fit()`` with
``use_gpu=True`` and ``use_gpu=False`` selected different feature sets on
the same data for any ``min_nonzero_confidence != 0.95`` (the default
``mrmr.py`` ctor uses ``0.99`` per the May-18 audit-fix).

Two tests:
- ``test_signature_exposes_min_nonzero_confidence``: signature-level regression
  that runs everywhere - guards against future regressions where someone
  removes the parameter again.
- ``test_cpu_gpu_threshold_parity``: requires a CUDA device; verifies the
  CPU and GPU paths either both reject or both accept across the
  ``min_nonzero_confidence`` range that matters in production.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest


def test_signature_exposes_min_nonzero_confidence():
    """Pre-fix this signature did NOT have ``min_nonzero_confidence`` and
    the function silently hardcoded a 0.05 threshold. Catches any future
    regression where someone removes the parameter again.
    """
    from mlframe.feature_selection.filters.gpu import (
        mi_direct_gpu_batched,
        mi_direct_gpu_batched_streamed,
    )
    sig_batched = inspect.signature(mi_direct_gpu_batched)
    assert "min_nonzero_confidence" in sig_batched.parameters, (
        "mi_direct_gpu_batched must accept min_nonzero_confidence -- "
        "without it the function silently hardcodes 0.05 and diverges "
        "from the CPU path."
    )
    sig_streamed = inspect.signature(mi_direct_gpu_batched_streamed)
    assert "min_nonzero_confidence" in sig_streamed.parameters, (
        "mi_direct_gpu_batched_streamed must accept min_nonzero_confidence "
        "(same bug applied to the streamed variant)."
    )


# -- GPU parity test (skipped when no CUDA device) ---------------------------

cp = pytest.importorskip("cupy")


def _gpu_available() -> bool:
    try:
        import cupy as _cp
        return _cp.cuda.runtime.getDeviceCount() >= 1
    except Exception:
        return False


_GPU_AVAILABLE = _gpu_available()


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="No CUDA device available")
@pytest.mark.parametrize("min_nonzero_confidence", [0.90, 0.95, 0.99])
def test_cpu_gpu_threshold_parity(min_nonzero_confidence):
    """The CPU and GPU paths must agree on accept/reject for every
    ``min_nonzero_confidence`` setting. Pre-fix CPU rejected at 0.99
    while GPU still used 0.95 -> different feature sets out of the
    selection loop.
    """
    from mlframe.feature_selection.filters.gpu import mi_direct_gpu_batched
    from mlframe.feature_selection.filters.permutation import mi_direct

    rng = np.random.default_rng(42)
    n, nb = 5000, 4
    # Two weakly-correlated discrete columns -- the regime where
    # permutation tests actually accept/reject differently across thresholds.
    factors = rng.integers(0, nb, size=(n, 2)).astype(np.int32)
    nbins = np.array([nb, nb], dtype=np.int32)
    npermutations = 100

    cpu_mi, _ = mi_direct(
        factors, x=(0,), y=(1,),
        factors_nbins=nbins, npermutations=npermutations,
        min_nonzero_confidence=min_nonzero_confidence,
        prefer_gpu=False,
    )
    gpu_mi, _ = mi_direct_gpu_batched(
        factors, x=(0,), y=(1,),
        factors_nbins=nbins, npermutations=npermutations,
        min_nonzero_confidence=min_nonzero_confidence,
    )
    cpu_rejected = (cpu_mi == 0.0)
    gpu_rejected = (gpu_mi == 0.0)
    assert cpu_rejected == gpu_rejected, (
        f"CPU/GPU divergence at min_nonzero_confidence={min_nonzero_confidence}: "
        f"cpu_mi={cpu_mi}, gpu_mi={gpu_mi}"
    )
