"""The permutation noise gate stops counting a candidate's failed shuffles once the verdict is decided; the
early stop must never flip a verdict (``nfailed >= max_failed``) and must leave the gated MI vector unchanged."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory._batch_kernels import (
    _perm_failcount_col,
    batch_mi_with_noise_gate,
    batch_mi_with_noise_gate_v2,
)


def _dense_inputs(seed: int, n: int = 400, k: int = 60):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    cols = []
    for j in range(k):
        if j % 5 == 0:
            cols.append(np.clip(y * 3 + rng.integers(0, 4, n), 0, 9))  # genuine
        else:
            cols.append(rng.integers(0, 10, n))  # noise
    disc = np.ascontiguousarray(np.column_stack(cols).astype(np.int8))
    freqs_y = np.bincount(y, minlength=3) / n
    return disc, y.astype(np.int16), freqs_y


@pytest.mark.parametrize("npermutations", [1, 3, 10, 40])
def test_early_stop_preserves_every_verdict(npermutations):
    disc, y, freqs_y = _dense_inputs(npermutations)
    n, k = disc.shape
    rng = np.random.default_rng(1)
    locals_mat = np.stack([rng.permutation(y) for _ in range(npermutations)])
    for max_failed in (1, 2, max(1, npermutations // 3)):
        for j in range(k):
            col = disc[:, j].astype(np.int64)
            counts = np.bincount(col, minlength=10)
            nz = counts > 0
            lookup = np.cumsum(nz) - 1
            dense = np.zeros((n, k), dtype=np.int16)
            dense[:, j] = lookup[col]
            freqs_dense = np.zeros((k, 10))
            freqs_dense[j, : nz.sum()] = counts[nz] / n
            full = _perm_failcount_col(dense, j, freqs_dense, int(nz.sum()), locals_mat, freqs_y, n, npermutations, 1e-9, False, np.int32)
            stopped = _perm_failcount_col(dense, j, freqs_dense, int(nz.sum()), locals_mat, freqs_y, n, npermutations, 1e-9, False, np.int32, max_failed)
            assert (full >= max_failed) == (stopped >= max_failed)
            assert stopped == min(full, max_failed)


@pytest.mark.parametrize("kernel", [batch_mi_with_noise_gate, batch_mi_with_noise_gate_v2])
@pytest.mark.parametrize("confidence", [0.5, 0.9, 0.99])
def test_gated_mi_matches_ungated_reference(kernel, confidence):
    disc, y, freqs_y = _dense_inputs(11)
    nbins = np.full(disc.shape[1], 10, dtype=np.int64)
    npermutations = 12
    got = kernel(disc, nbins, y, y, freqs_y, npermutations, np.uint64(5), confidence, False)
    observed = kernel(disc, nbins, y, y, freqs_y, 0, np.uint64(5), confidence, False)
    # Reference verdict from a full, uncapped count of the same deterministic shuffle stream.
    max_failed = max(int(npermutations * (1.0 - confidence)), 1)
    n = disc.shape[0]
    locals_mat = np.empty((npermutations, n), dtype=y.dtype)
    for i in range(npermutations):
        state = np.uint64(5) * np.uint64(2654435761) + np.uint64(i + 1)
        local = y.copy()
        for j in range(n - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            kk = int(state >> np.uint64(33)) % (j + 1)
            local[j], local[kk] = local[kk], local[j]
        locals_mat[i] = local
    for j in range(disc.shape[1]):
        col = disc[:, j].astype(np.int64)
        counts = np.bincount(col, minlength=10)
        nz = counts > 0
        dense = np.zeros(disc.shape, dtype=np.int16)
        dense[:, j] = (np.cumsum(nz) - 1)[col]
        freqs_dense = np.zeros((disc.shape[1], 10))
        freqs_dense[j, : nz.sum()] = counts[nz] / n
        if observed[j] <= 0.0:
            assert got[j] == observed[j]
            continue
        full = _perm_failcount_col(dense, j, freqs_dense, int(nz.sum()), locals_mat, freqs_y, n, npermutations, observed[j], False, np.int32)
        assert got[j] == (0.0 if full >= max_failed else observed[j])
