"""Identity pin for the cluster-aggregate compact-stack MI optimization.

``run_cluster_aggregate_step`` scores each candidate combiner's aggregate-MI with the target. The
optimization replaces ``mi(np.column_stack([data, binned]), [data.shape[1]], target, ...)`` (a full
``(n, n_features+1)`` copy rebuilt per method) with a compact ``(n, |target|+1)`` stack of only the
target columns + the binned aggregate, remapping the x/y indices. ``mi``/``merge_vars`` read only the
x and y columns by value, so the result MUST be bit-identical.

This test pins the equivalence directly at the ``mi`` call shape so a future "simplify back to
column_stack([data, binned])" or a wrong remap is caught. It fails on a remap that changes which
columns are read (verified by perturbing y_new).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory import mi


def _old_form(data, binned, nbins, target, qnb, dtype=np.int32) -> float:
    return float(
        mi(
            np.column_stack([data, binned.astype(data.dtype)]),
            np.array([data.shape[1]], dtype=np.int64),
            target,
            np.concatenate([np.asarray(nbins), [int(qnb)]]).astype(np.int64),
            dtype=dtype,
        )
    )


def _new_form(data, binned, nbins, target, qnb, dtype=np.int32) -> float:
    tcols = np.asarray(target, dtype=np.int64)
    compact = np.column_stack([data[:, tcols], binned.astype(data.dtype)])
    n_t = tcols.shape[0]
    compact_nbins = np.concatenate([np.asarray(nbins)[tcols], [int(qnb)]]).astype(np.int64)
    return float(
        mi(compact, np.array([n_t], dtype=np.int64), np.arange(n_t, dtype=np.int64), compact_nbins, dtype=dtype)
    )


@pytest.mark.parametrize("n,n_features,nb", [(600, 20, 8), (2407, 200, 10), (5000, 50, 16)])
@pytest.mark.parametrize("target_at", [0, 5, [0]])
def test_compact_stack_mi_bit_identical(n, n_features, nb, target_at):
    rng = np.random.default_rng(n + nb + (target_at if isinstance(target_at, int) else 99))
    data = rng.integers(0, nb, size=(n, n_features), dtype=np.int32)
    # Make the binned aggregate correlate with the target so MI is nonzero.
    tcol = target_at[0] if isinstance(target_at, list) else target_at
    binned = ((data[:, tcol] + rng.integers(0, 3, size=n)) % nb).astype(np.int32)
    nbins = np.full(n_features, nb, dtype=np.int64)
    target = np.array([tcol], dtype=np.int64)

    old = _old_form(data, binned, nbins, target, nb)
    new = _new_form(data, binned, nbins, target, nb)
    assert old == new, f"compact-stack MI diverged: old={old!r} new={new!r} diff={abs(old - new)!r}"
    assert new > 0.0  # sanity: the constructed correlation gives positive MI


def test_compact_stack_handles_multi_target():
    """Multiple target columns: y maps to arange(|target|) in the compact matrix."""
    rng = np.random.default_rng(7)
    n, n_features, nb = 1500, 40, 8
    data = rng.integers(0, nb, size=(n, n_features), dtype=np.int32)
    binned = ((data[:, 0] + data[:, 1]) % nb).astype(np.int32)
    nbins = np.full(n_features, nb, dtype=np.int64)
    target = np.array([0, 1], dtype=np.int64)
    assert _old_form(data, binned, nbins, target, nb) == _new_form(data, binned, nbins, target, nb)
