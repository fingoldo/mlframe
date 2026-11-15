"""The LOF transform scores rows independently, in any split, and needs no memory that grows with rows x reference points."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import _lof_fe


def _fixture(n_new=700, n_ref=300, p=4, seed=0):
    """A reference set with its frozen internals and a set of new rows."""
    rng = np.random.default_rng(seed)
    X_ref = rng.standard_normal((n_ref, p)).astype(np.float32)
    X_new = rng.standard_normal((n_new, p)).astype(np.float32)
    lrd, kd, k_eff = _lof_fe._lof_fit_reference(X_ref, 10)
    return X_new, X_ref, lrd, kd, k_eff


def test_scoring_in_pieces_gives_the_scores_of_one_pass_and_keeps_the_dtype():
    """Rows are independent: three pieces concatenated equal the whole, in the same dtype."""
    X_new, X_ref, lrd, kd, k_eff = _fixture()
    whole = _lof_fe._lof_transform(X_new, X_ref, lrd, kd, k_eff)
    pieces = np.concatenate([_lof_fe._lof_transform(X_new[a:b], X_ref, lrd, kd, k_eff) for a, b in ((0, 100), (100, 450), (450, 700))])
    assert whole.shape == pieces.shape == (X_new.shape[0],)
    assert whole.dtype == pieces.dtype
    np.testing.assert_allclose(pieces, whole, rtol=1e-5, atol=1e-6)


def test_a_large_transform_needs_no_distance_matrix():
    """200000 rows against 4000 reference points: the matrix form needed 6.4 GB per (rows, n_ref) matrix of float64; the scan keeps k values per thread and finishes with finite scores."""
    X_new, X_ref, lrd, kd, k_eff = _fixture(n_new=200_000, n_ref=4000, p=3, seed=1)
    out = _lof_fe._lof_transform_host(X_new, X_ref, lrd, kd, k_eff)
    assert out.shape == (200_000,) and bool(np.isfinite(out).all())
