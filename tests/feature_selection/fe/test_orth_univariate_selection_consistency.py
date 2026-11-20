"""Selection consistency of the orthogonal-univariate extra-basis FE: the held-out smoothness gate and the absolute MI floor."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_extra_basis_fe as eb
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_extra_basis_fe_generate as gen


def test_heldout_smooth_r2_slow_and_fast_twins_agree_on_tied_columns():
    """The per-column function and its precomputed-y twin rank tied values identically, so the adaptive gate does not depend on which one runs."""
    rng = np.random.default_rng(3)
    x = rng.permutation(np.repeat(np.arange(6.0), 150))
    y = x + 0.3 * rng.standard_normal(x.size)
    prep = eb._heldout_smooth_r2_prep(y)
    assert prep is not None
    slow = eb._heldout_smooth_r2(x, y)
    fast = eb._heldout_smooth_r2_fast(x, prep)
    assert abs(slow - fast) < 1e-6
    assert slow > 0.8


def test_extra_basis_hybrid_applies_noise_floor_with_few_candidates(monkeypatch):
    """With fewer than four scored candidates the noise floors still apply, so a candidate inside the noise band is not admitted."""
    import mlframe.feature_selection.filters._orthogonal_univariate_fe as pkg

    n = 100
    x = np.linspace(0.0, 1.0, n)
    X = pd.DataFrame({"a": x})
    y = (x > 0.5).astype(int)
    knots, lo, hi, _nb = eb._fit_spline_for_col(x.astype(np.float32), 3)
    engineered = pd.DataFrame({"e1": x, "e2": x**2})
    meta = {nm: {"basis": "spline", "src": "a", "knots": knots, "idx": i, "lo": lo, "hi": hi} for i, nm in enumerate(engineered.columns)}
    scores = pd.DataFrame(
        {
            "engineered_col": ["e1", "e2"], "source_col": ["a", "a"], "baseline_mi": [0.1, 0.1], "engineered_mi": [0.5, 0.05], "uplift": [5.0, 1.5],
        }
    )
    monkeypatch.setattr(gen, "generate_extra_basis_features", lambda *a, **k: (engineered, meta))
    monkeypatch.setattr(pkg, "score_features_by_mi_uplift", lambda *a, **k: scores)
    _aug, _scores, recipes = gen.hybrid_orth_extra_basis_fe_with_recipes(X, y, cols=["a"], top_k=5)
    assert [r.name for r in recipes] == ["e1"]
