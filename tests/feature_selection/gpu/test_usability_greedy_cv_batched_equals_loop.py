"""The batched regression CV of the resident usability greedy gives the per-candidate loop's fold errors and a fallback signal on a singular border."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _usability_greedy_gpu_resident as ug


def _setup(seed=0, n=600, p=9, nf=4):
    """Random design, target and fold masks of the resident greedy, and the batched closure."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, p))
    y = V[:, 0] * 1.5 - V[:, 3] + 0.3 * rng.standard_normal(n)
    fold = rng.permutation(n) % nf
    va = [cp.asarray(fold == f) for f in range(nf)]
    tr = [cp.asarray(fold != f) for f in range(nf)]
    return nf, tr, va, cp.asarray(y), cp.asarray(V)


@pytest.mark.parametrize("sel", [[], [0], [0, 3, 5]])
def test_batched_cv_equals_loop(sel):
    """Fold errors agree with the loop for an empty, a one-column and a three-column selection."""
    nf, tr, va, y, V = _setup()
    cv = ug._usability_greedy_gpu_step3_def_cv_candidates(nf, tr, va, y, V, cp)
    cands = [i for i in range(V.shape[1]) if i not in sel][:6]
    new = cv(sel, cands)
    ref = cv.reference(sel, cands)
    assert list(new) == list(ref)
    for i in cands:
        np.testing.assert_allclose(new[i], ref[i], rtol=1e-9, atol=1e-12)


def test_singular_border_is_not_silently_scored():
    """A candidate that duplicates a selected column gives a singular border: the batched form raises the fallback signal instead of returning garbage errors (the loop relies on cupy raising, which it does not)."""
    nf, tr, va, y, V = _setup()
    V2 = cp.concatenate([V, V[:, :1]], axis=1)
    cv = ug._usability_greedy_gpu_step3_def_cv_candidates(nf, tr, va, y, V2, cp)
    with pytest.raises(ug._ResidentFallbackError):
        cv([0], [V2.shape[1] - 1, 4])
