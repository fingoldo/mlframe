"""Unit + biz_value tests for spectral matrix seriation (PZAD vismultivar)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.core.matrix_seriation import seriate, spectral_seriation


def _block_similarity(block_sizes, rng, noise=0.05):
    """Build a block-diagonal similarity matrix (high within-block, low across) then shuffle rows."""
    n = sum(block_sizes)
    labels = np.concatenate([np.full(s, k) for k, s in enumerate(block_sizes)])
    M = (labels[:, None] == labels[None, :]).astype(float) * 0.9 + 0.05
    M += rng.normal(0, noise, size=(n, n))
    M = 0.5 * (M + M.T)
    np.fill_diagonal(M, 1.0)
    perm = rng.permutation(n)
    return M[np.ix_(perm, perm)], labels[perm]


def _block_contiguity(labels_ordered):
    """Fraction of adjacent pairs sharing a block label (1.0 = perfectly grouped)."""
    return float(np.mean(labels_ordered[1:] == labels_ordered[:-1]))


# ---------------------------------------------------------------- unit
def test_identity_matrix_is_valid_permutation():
    """Identity matrix is valid permutation."""
    perm = spectral_seriation(np.eye(5))
    assert sorted(perm.tolist()) == list(range(5))


def test_tiny_matrices_return_trivial():
    """Tiny matrices return trivial."""
    assert spectral_seriation(np.eye(1)).tolist() == [0]
    assert sorted(spectral_seriation(np.eye(2)).tolist()) == [0, 1]


def test_non_square_raises():
    """Non square raises."""
    with pytest.raises(ValueError):
        spectral_seriation(np.ones((3, 4)))


def test_invalid_method_raises():
    """Invalid method raises."""
    with pytest.raises(ValueError):
        spectral_seriation(np.eye(4), method="nope")


def test_seriate_returns_reordered_and_perm():
    """Seriate returns reordered and perm."""
    rng = np.random.default_rng(0)
    M, _ = _block_similarity([4, 4], rng)
    R, perm = seriate(M)
    assert R.shape == M.shape
    assert np.allclose(R, M[np.ix_(perm, perm)])


def test_seriate_accepts_precomputed_perm():
    """Seriate accepts precomputed perm."""
    M = np.arange(16.0).reshape(4, 4)
    perm = np.array([3, 1, 2, 0])
    R, p = seriate(M, perm=perm)
    assert np.array_equal(p, perm)
    assert np.allclose(R, M[np.ix_(perm, perm)])


# ---------------------------------------------------------------- biz_value
def test_biz_val_fiedler_recovers_block_structure():
    """On a shuffled block-diagonal similarity matrix, Fiedler seriation groups same-block rows adjacently far
    better than the shuffled order. Measured: shuffled contiguity ~0.3-0.5, seriated ~0.85-1.0. Floor 0.8, +0.2 gain."""
    rng = np.random.default_rng(1)
    M, labels = _block_similarity([6, 6, 6], rng)
    before = _block_contiguity(labels)
    after = _block_contiguity(labels[spectral_seriation(M, method="fiedler")])
    assert after >= 0.8, f"Fiedler seriated contiguity {after:.2f} should be >=0.8"
    assert after >= before + 0.2, f"seriation should improve contiguity by >=0.2 (was {before:.2f}, now {after:.2f})"


def test_svd_method_returns_valid_permutation():
    """The lecture's leading-singular-vector order is a valid permutation (weaker block recovery than Fiedler
    on all-positive similarity — see the module docstring; not asserted to hit the Fiedler floor)."""
    rng = np.random.default_rng(1)
    M, _ = _block_similarity([6, 6, 6], rng)
    perm = spectral_seriation(M, method="svd")
    assert sorted(perm.tolist()) == list(range(M.shape[0]))


def test_biz_val_fiedler_beats_shuffle_on_two_blocks():
    """Two well-separated blocks: Fiedler order fully groups them (contiguity 1.0 up to one boundary)."""
    rng = np.random.default_rng(2)
    M, labels = _block_similarity([8, 8], rng, noise=0.02)
    perm = spectral_seriation(M, method="fiedler")
    # Two blocks perfectly seriated -> exactly one boundary crossing -> contiguity (n-2)/(n-1).
    assert _block_contiguity(labels[perm]) >= (16 - 2) / (16 - 1) - 1e-9
