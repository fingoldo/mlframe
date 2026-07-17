"""Identity-pin for the pseudo_smote SMOTE-synthesize vectorization.

``_smote_synthesize_intra`` originally drew (src, nbr, alpha) per iteration AND wrote one interpolated output
row per iteration inside the Python loop. The optimization keeps the per-iteration PCG64 draw order intact (so
the synthetic cloud is unchanged) but hoists the gather + convex-interpolation out into a single vectorized pass.

The draw order is load-bearing: ``np.random.default_rng`` interleaves the src-int / nbr-int / alpha-float draws,
so batching them would silently change WHICH neighbours interpolate and break selection downstream. This test pins
bit-identity against a self-contained reference re-implementing the OLD row-loop, including the ``cand.size == 0``
no-draw branch.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from mlframe.feature_engineering.transformer.pseudo_smote import _smote_synthesize_intra


def _old_reference(X_minority, n_synthetic, k_neighbors, seed):
    """Helper: Old reference."""
    n_min = X_minority.shape[0]
    if n_min < 2:
        return X_minority.copy() if n_min > 0 else np.zeros((0, 0), dtype=np.float32)
    from sklearn.neighbors import NearestNeighbors

    k_used = min(k_neighbors + 1, n_min)
    nn = NearestNeighbors(n_neighbors=k_used).fit(X_minority)
    _dists, ids = nn.kneighbors(X_minority)
    rng = np.random.default_rng(seed)
    out = np.zeros((n_synthetic, X_minority.shape[1]), dtype=np.float32)
    for i in range(n_synthetic):
        src_idx = rng.integers(0, n_min)
        candidates = ids[src_idx, 1:k_used]
        if candidates.size == 0:
            out[i] = X_minority[src_idx]
            continue
        nbr_idx = candidates[rng.integers(0, candidates.size)]
        alpha = rng.random()
        out[i] = X_minority[src_idx] + alpha * (X_minority[nbr_idx] - X_minority[src_idx])
    return out.astype(np.float32)


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("nrows,d,k,n_syn", [(500, 30, 5, 5000), (50, 8, 5, 400), (3, 4, 5, 20), (200, 12, 10, 2000)])
def test_smote_synthesize_bit_identical_to_row_loop(seed, nrows, d, k, n_syn):
    """Smote synthesize bit identical to row loop."""
    rng = np.random.default_rng(seed + 99)
    X = rng.standard_normal((nrows, d)).astype(np.float32)
    expected = _old_reference(X, n_syn, k, seed)
    got = _smote_synthesize_intra(X, n_synthetic=n_syn, k_neighbors=k, seed=seed)
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)
