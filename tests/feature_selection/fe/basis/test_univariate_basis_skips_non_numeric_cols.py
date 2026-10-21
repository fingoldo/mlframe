"""Regression: the univariate basis generator skips non-numeric columns of an explicit column list, as its docstring promises."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_univariate_fe import generate_univariate_basis_features, hybrid_orth_mi_fe


def test_an_explicit_column_list_with_a_string_column_does_not_raise():
    """A list naming a numeric and a string column expands the numeric one and ignores the string one."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.standard_normal(300), "region": rng.choice(["R_1", "R_2", "R_3"], 300)})
    out = generate_univariate_basis_features(X, cols=["x", "region"], degrees=(2, 3), basis="hermite", y=rng.standard_normal(300))
    assert len(out.columns) > 0
    assert all(name.startswith("x__") for name in out.columns)


def test_the_hybrid_scorer_skips_a_string_column_of_an_explicit_list():
    """``hybrid_orth_mi_fe`` with a numeric and a string column in ``cols`` returns a frame and engineers only from the numeric one."""
    rng = np.random.default_rng(1)
    n = 600
    x = rng.standard_normal(n)
    X = pd.DataFrame({"x": x, "region": rng.choice(["R_1", "R_2"], n)})
    X_aug, _ = hybrid_orth_mi_fe(X, (x**2 > 1).astype(int), cols=["x", "region"], degrees=(2, 3), basis="hermite")
    assert "region" in X_aug.columns and all(c.startswith("x__") for c in X_aug.columns if c not in X.columns)
