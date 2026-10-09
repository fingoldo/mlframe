"""A target wider than the static shared-memory cap (16 classes) is routed to the kernel that serves it, not provoked into a logged failure.

``batch_pair_mi_cuda`` keeps its class histogram in static shared memory and rejects a target with more than ``MAX_Y_BINS_CUDA`` classes. The F2 benchmark's target
quantises to 20 classes, so every strict-mode (forced CUDA) call used to raise that ``ValueError``, log "forced CUDA backend failed" and only then reach the
dynamic-shared-memory kernel that handles it. The result was right but each call looked like a failure, and a real failure would have been lost in the same log.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.feature_selection.filters import batch_pair_mi_gpu as bpm


def _inputs(n_classes_y: int, n: int = 4000, seed: int = 0):
    """Six 8-bin columns, three pairs and a target with ``n_classes_y`` classes."""
    rng = np.random.default_rng(seed)
    factors = rng.integers(0, 8, size=(n, 6)).astype(np.int32)
    nbins = np.full(6, 8, dtype=np.int32)
    pair_a = np.array([0, 1, 2], dtype=np.int64)
    pair_b = np.array([3, 4, 5], dtype=np.int64)
    y = (factors[:, 0] * 2 + rng.integers(0, 3, size=n)) % n_classes_y
    classes_y = y.astype(np.int32)
    freqs_y = np.bincount(classes_y, minlength=n_classes_y).astype(np.float64) / n
    return factors, pair_a, pair_b, nbins, classes_y, freqs_y


@pytest.mark.skipif(not bpm._CUDA_AVAIL, reason="needs numba.cuda")
@pytest.mark.parametrize("force", ["cuda", None])
def test_a_target_over_the_static_cap_is_served_without_a_failure_log(caplog, force):
    """20 classes > MAX_Y_BINS_CUDA: no 'failed' warning, a GPU backend answers, and the result equals the CPU kernel."""
    args = _inputs(n_classes_y=bpm.MAX_Y_BINS_CUDA + 4)
    with caplog.at_level(logging.WARNING, logger=bpm.logger.name):
        mi, name = bpm.dispatch_batch_pair_mi(*args, force_backend=force)
    failed = [r.getMessage() for r in caplog.records if "failed" in r.getMessage().lower()]
    assert not failed, f"a supported shape was reported as a failure: {failed}"
    reference = bpm.batch_pair_mi_njit_serial(*args)
    np.testing.assert_allclose(mi, reference, rtol=1e-6, atol=1e-9)
    if force == "cuda":
        assert name.startswith("cuda"), f"forced CUDA must be answered by a CUDA kernel, got {name!r}"


def test_the_static_cap_check_matches_the_kernels_own_guard():
    """``static_kernel_accepts`` is true exactly where ``batch_pair_mi_cuda``'s guard would not raise."""
    narrow = bpm._PairMiDispatch(*_inputs(n_classes_y=bpm.MAX_Y_BINS_CUDA))
    wide = bpm._PairMiDispatch(*_inputs(n_classes_y=bpm.MAX_Y_BINS_CUDA + 1))
    assert narrow.static_kernel_accepts() is True
    assert wide.static_kernel_accepts() is False
