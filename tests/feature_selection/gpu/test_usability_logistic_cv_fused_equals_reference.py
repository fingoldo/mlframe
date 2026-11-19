"""The fused one-launch cross-validated logistic fit of the resident usability greedy equals the stacked-matmul reference to float32 precision and ends in the fallback signal on a non-convergent fit."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _usability_greedy_clf_batched as B


class _Fallback(Exception):
    """Stand-in for the caller's fallback error."""


def _data(seed=0, n=1500, p=30, nf=3):
    """Random design with a linear signal, the device fold index arrays and the shared fit settings."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, p))
    y = ((V[:, 0] * 1.2 - V[:, 3] + rng.standard_normal(n)) > 0).astype(np.int64)
    fold = rng.permutation(n) % nf
    folds = [(cp.asarray(np.where(fold != f)[0]), cp.asarray(np.where(fold == f)[0]), True) for f in range(nf)]
    kw = dict(dtype=cp.float32, newton_max_iter=200, tol=2e-4, lam=1.0, logloss_eps=1e-15, fallback_error=_Fallback)
    return cp.asarray(V), cp.asarray(y), folds, kw, p


@pytest.mark.parametrize("vdtype", [np.float64, np.float32])
@pytest.mark.parametrize("selected", [[], [0], [0, 3, 5]])
def test_fused_cv_equals_reference(selected, vdtype):
    """The per-fold logloss of every candidate agrees with the reference and picks the same best candidate, for a double and a float32 value matrix."""
    V, y, folds, kw, p = _data()
    V = V.astype(vdtype)
    cands = [c for c in range(p) if c not in selected][:12]
    new = B.cv_candidates_binary(cp, V, selected, cands, folds, y, **kw)
    ref = B.cv_candidates_binary_reference(cp, V, selected, cands, folds, y, **kw)
    assert new.shape == ref.shape
    np.testing.assert_allclose(new, ref, rtol=1e-5, atol=1e-6)
    assert int(np.argmin(new.mean(axis=1))) == int(np.argmin(ref.mean(axis=1)))


def test_untrainable_fold_is_inf():
    """A fold marked untrainable gives infinity for every candidate, as the reference does."""
    V, y, folds, kw, _p = _data()
    folds[1] = (folds[1][0], folds[1][1], False)
    new = B.cv_candidates_binary(cp, V, [0], [1, 2, 4], folds, y, **kw)
    assert np.isinf(new[:, 1]).all() and np.isfinite(new[:, [0, 2]]).all()


def test_non_convergence_raises_the_fallback():
    """With one Newton iteration allowed and a tight tolerance the fit does not converge: the caller's fallback error is raised."""
    V, y, folds, kw, _p = _data()
    kw = dict(kw, newton_max_iter=1, tol=1e-12)
    with pytest.raises(_Fallback):
        B.cv_candidates_binary(cp, V, [0], [1, 2], folds, y, **kw)
