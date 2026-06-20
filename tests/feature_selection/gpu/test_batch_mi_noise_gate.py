"""Bit-identity tests for ``batch_mi_with_noise_gate`` vs per-column ``mi_direct``.

The batched FE-candidate MI + permutation noise-gate kernel must reproduce the
EXACT ``fe_mi`` a per-candidate ``mi_direct`` loop produces on the default FE path
(``parallelism='outer'``, ``n_workers=1`` -> ``parallel_mi_prange``, ``base_seed=0``).
Bit-identity is non-negotiable: any drift changes which engineered features MRMR keeps.

Covered: varied n / K / nbins / npermutations in {0, 3, 10} /
min_nonzero_confidence in {0.99, 0.0}, plus tie-heavy and pure-noise (rejection-
triggering) columns that must be zeroed identically.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory import (
    batch_mi_with_noise_gate,
    merge_vars,
)
from mlframe.feature_selection.filters.permutation import mi_direct


def _make_frame(n, K, nbins, seed):
    """Build a discretized (n, K) int frame with a MIX of column types:
    informative (correlated with y), tie-heavy (near-constant), and pure noise.
    Returns (disc_2d, classes_y, freqs_y).
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n).astype(np.int32)
    cols = np.empty((n, K), dtype=np.int32)
    for k in range(K):
        kind = k % 4
        if kind == 0:
            # Informative: y plus a little noise, clipped into nbins.
            c = (y + rng.integers(0, 2, size=n)) % nbins
        elif kind == 1:
            # Pure noise.
            c = rng.integers(0, nbins, size=n)
        elif kind == 2:
            # Tie-heavy: mostly one bin, a few others.
            c = np.zeros(n, dtype=np.int64)
            idx = rng.choice(n, size=max(1, n // 20), replace=False)
            c[idx] = rng.integers(1, nbins, size=idx.size)
        else:
            # Strongly informative (y mapped, occasional flip).
            c = y.copy().astype(np.int64)
            flip = rng.choice(n, size=max(1, n // 10), replace=False)
            c[flip] = rng.integers(0, nbins, size=flip.size)
        cols[:, k] = (c % nbins).astype(np.int32)
    classes_y, freqs_y, _ = merge_vars(
        factors_data=y.reshape(-1, 1),
        vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=np.array([int(y.max()) + 1], dtype=np.int64),
        dtype=np.int32,
    )
    return cols, classes_y, freqs_y


def _per_column_reference(disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y,
                          npermutations, min_nonzero_confidence):
    """The ORIGINAL per-candidate path: loop mi_direct over columns exactly as the
    FE Phase-3 batch loop does (base_seed=0, parallelism='outer', n_workers=1,
    prefer_gpu False to keep the deterministic CPU path)."""
    K = disc_2d.shape[1]
    out = np.empty(K, dtype=np.float64)
    for ci in range(K):
        fe_mi, _ = mi_direct(
            disc_2d[:, ci].reshape(-1, 1),
            x=np.array([0], dtype=np.int64),
            y=None,
            factors_nbins=np.array([int(factors_nbins[ci])], dtype=np.int64),
            classes_y=classes_y,
            classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            min_nonzero_confidence=min_nonzero_confidence,
            npermutations=npermutations,
            prefer_gpu=False,
        )
        out[ci] = fe_mi
    return out


@pytest.mark.parametrize("n,K,nbins", [(200, 8, 4), (500, 13, 6), (1000, 20, 5)])
@pytest.mark.parametrize("npermutations", [0, 3, 10])
@pytest.mark.parametrize("min_nonzero_confidence", [0.99, 0.0])
def test_batch_bit_identical_to_mi_direct(n, K, nbins, npermutations, min_nonzero_confidence):
    disc_2d, classes_y, freqs_y = _make_frame(n, K, nbins, seed=1234 + n + K + nbins)
    classes_y_safe = classes_y.copy()
    factors_nbins = np.full(K, nbins, dtype=np.int64)

    ref = _per_column_reference(
        disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y,
        npermutations, min_nonzero_confidence,
    )
    got = batch_mi_with_noise_gate(
        disc_2d=disc_2d,
        factors_nbins=factors_nbins,
        classes_y=classes_y,
        classes_y_safe=classes_y_safe,
        freqs_y=freqs_y,
        npermutations=npermutations,
        base_seed=np.uint64(0),
        min_nonzero_confidence=float(min_nonzero_confidence),
        use_su=False,
        dtype=np.int32,
    )

    assert got.shape == ref.shape
    # EXACT float equality -- bit-identity is the contract.
    assert np.array_equal(got, ref), (
        f"mismatch n={n} K={K} nbins={nbins} nperm={npermutations} "
        f"mnc={min_nonzero_confidence}\n ref={ref}\n got={got}\n diff={got - ref}"
    )


def test_pure_noise_zeroed_identically():
    """A column of pure noise must be rejected (-> 0.0) by BOTH paths identically."""
    n = 800
    rng = np.random.default_rng(99)
    y = rng.integers(0, 4, size=n).astype(np.int32)
    noise = rng.integers(0, 6, size=n).astype(np.int32)
    disc_2d = noise.reshape(-1, 1)
    classes_y, freqs_y, _ = merge_vars(
        factors_data=y.reshape(-1, 1),
        vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=np.array([int(y.max()) + 1], dtype=np.int64),
        dtype=np.int32,
    )
    factors_nbins = np.array([6], dtype=np.int64)
    ref = _per_column_reference(disc_2d, factors_nbins, classes_y, classes_y.copy(),
                                freqs_y, npermutations=10, min_nonzero_confidence=0.99)
    got = batch_mi_with_noise_gate(
        disc_2d=disc_2d, factors_nbins=factors_nbins, classes_y=classes_y,
        classes_y_safe=classes_y.copy(), freqs_y=freqs_y, npermutations=10,
        base_seed=np.uint64(0), min_nonzero_confidence=0.99, use_su=False, dtype=np.int32,
    )
    assert np.array_equal(got, ref)


def test_su_mode_bit_identical(monkeypatch):
    """When SU normalization is active, batched use_su=True matches per-column mi_direct."""
    import mlframe.feature_selection.filters.info_theory as it
    monkeypatch.setattr(it, "use_su_normalization", lambda: True)
    import mlframe.feature_selection.filters.permutation as perm
    monkeypatch.setattr(perm, "use_su_normalization", lambda: True)

    disc_2d, classes_y, freqs_y = _make_frame(400, 10, 5, seed=7)
    factors_nbins = np.full(10, 5, dtype=np.int64)
    ref = _per_column_reference(disc_2d, factors_nbins, classes_y, classes_y.copy(),
                                freqs_y, npermutations=10, min_nonzero_confidence=0.99)
    got = batch_mi_with_noise_gate(
        disc_2d=disc_2d, factors_nbins=factors_nbins, classes_y=classes_y,
        classes_y_safe=classes_y.copy(), freqs_y=freqs_y, npermutations=10,
        base_seed=np.uint64(0), min_nonzero_confidence=0.99, use_su=True, dtype=np.int32,
    )
    assert np.array_equal(got, ref)
