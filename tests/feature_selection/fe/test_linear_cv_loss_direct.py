"""The direct (Newton / least-squares) CV loss equals the scikit-learn pipeline's, and the pipelines still decide the cases it does not cover."""

import numpy as np
import pytest

import mlframe.feature_selection.filters._linear_cv_loss as lcv


def _data(n, k, seed, clf):
    """A design of ``k`` columns (one near-constant when k > 2) and a binary or continuous target tied to the first column."""
    r = np.random.default_rng(seed)
    x = r.standard_normal((n, k))
    if k > 2:
        x[:, 2] = 1.0 + 1e-3 * x[:, 2]
    z = 0.8 * x[:, 0] + 0.3 * r.standard_normal(n)
    y = (z > 0).astype(np.int64) if clf else z
    return x, y


@pytest.mark.parametrize("clf", [True, False])
@pytest.mark.parametrize("n,k", [(3000, 1), (1200, 3), (800, 4)])
def test_direct_equals_the_pipeline(n, k, clf, monkeypatch):
    """Same loss as the pipeline path (forced by disabling the direct one) to 1e-4 relative: lbfgs stops at its 1e-4 gradient tolerance, the Newton solve at the optimum, and every gate compares losses with a 1 percent margin."""
    x, y = _data(n, k, 0, clf)
    got = lcv.cv_loss(x, y, clf, 5, 3)
    monkeypatch.setattr(lcv, "_direct_cv_loss", lambda *a, **kw: None)
    want = lcv.cv_loss(x, y, clf, 5, 3)
    assert np.isfinite(got) and got == pytest.approx(want, rel=1e-4, abs=1e-6)


def test_multiclass_goes_to_the_pipeline(monkeypatch):
    """Three classes are not covered by the direct path: it hands back ``None``."""
    x, _ = _data(600, 2, 1, True)
    y = np.random.default_rng(1).integers(0, 3, 600)
    assert lcv._direct_cv_loss(x, y, y, True, 3, [(np.arange(500), np.arange(500, 600))]) is None
