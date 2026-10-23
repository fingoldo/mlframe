"""The LOF transform scores rows in memory-sized blocks without changing any score."""

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


def test_blocks_give_the_scores_of_one_pass_and_keep_the_dtype(monkeypatch):
    """A one-row block and a roomy single block agree to float rounding and return the same dtype."""
    X_new, X_ref, lrd, kd, k_eff = _fixture()
    whole = _lof_fe._lof_transform(X_new, X_ref, lrd, kd, k_eff)
    monkeypatch.setattr(_lof_fe, "_lof_row_block", lambda n_ref, itemsize: 1)
    rows = _lof_fe._lof_transform(X_new, X_ref, lrd, kd, k_eff)
    monkeypatch.setattr(_lof_fe, "_lof_row_block", lambda n_ref, itemsize: 64)
    blocks = _lof_fe._lof_transform(X_new, X_ref, lrd, kd, k_eff)
    assert whole.shape == rows.shape == blocks.shape == (X_new.shape[0],)
    assert whole.dtype == rows.dtype == blocks.dtype
    np.testing.assert_allclose(rows, whole, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(blocks, whole, rtol=1e-5, atol=1e-6)


def test_row_block_shrinks_with_the_reference_and_never_reaches_zero(monkeypatch):
    """The block follows the free memory: a larger reference gives fewer rows, and the floor is one row."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _streamed_gpu_mi

    monkeypatch.setattr(_streamed_gpu_mi, "_host_free_bytes", lambda: 1 << 30)
    assert _lof_fe._lof_row_block(4000, 4) < _lof_fe._lof_row_block(2000, 4)
    monkeypatch.setattr(_streamed_gpu_mi, "_host_free_bytes", lambda: 1)
    assert _lof_fe._lof_row_block(4000, 4) == 1
