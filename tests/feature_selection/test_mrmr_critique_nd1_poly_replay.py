"""Regression (MRMR critique ND-1): a poly-FE recipe's unary name is 'poly_<coef>' (mapping to a hermite coefficient
ARRAY in the per-fit unary_transformations dict, applied via hermval at fit). The recipe did NOT persist the coef, so
recipe replay in _apply_unary_binary raised KeyError looking 'poly_<coef>' up in the static preset. The coef is now
threaded into the recipe (poly_<side>_coef in extra) and replayed via hermval, and 'poly_' is a recognised pseudo-unary.
"""

import numpy as np
import pandas as pd
import warnings


def test_fe_max_polynoms_fit_transform_no_keyerror():
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.integers(0, 6, size=(600, 10)).astype(float), columns=[f"f{i}" for i in range(10)])
    y = pd.Series((X["f0"] + X["f1"] + X["f2"] > 7).astype(int))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(interactions_max_order=2, fe_max_polynoms=4, full_npermutations=1, cv=2, run_additional_rfecv_minutes=False).fit(X, y)
        Xt = np.asarray(m.transform(X))  # pre-fix: KeyError "Unary function 'poly_[...]' not in preset"
    names = m.get_feature_names_out()
    assert Xt.shape[1] == len(names), "transform width must equal get_feature_names_out after poly replay"
    # transform must be stable across calls (pure-X replay)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xt2 = np.asarray(m.transform(X))
    assert np.allclose(np.nan_to_num(Xt), np.nan_to_num(Xt2)), "poly replay must be deterministic across calls"


def test_poly_recipe_persists_coef_and_replays_via_hermval():
    from mlframe.feature_selection.filters.engineered_recipes._recipe_unary_binary import build_unary_binary_recipe, _apply_unary_binary
    from numpy.polynomial.hermite import hermval

    coef = np.array([0.1, 1.0, 0.05, 0.02])
    r = build_unary_binary_recipe(
        name="poly_test",
        src_a_name="a",
        src_b_name="b",
        unary_a_name="poly_" + str(coef),
        unary_b_name="identity",
        binary_name="mul",
        unary_preset="medium",
        binary_preset="medium",
        quantization_nbins=None,
        quantization_method=None,
        quantization_dtype=np.int8,
        poly_a_coef=coef,
    )
    assert "poly_a_coef" in r.extra and np.allclose(r.extra["poly_a_coef"], coef)
    X = pd.DataFrame({"a": np.linspace(-2, 2, 50), "b": np.ones(50)})
    out = np.asarray(_apply_unary_binary(r, X))
    # side a = hermval(a, coef); binary mul with identity(b)=1 -> equals hermval(a, coef)
    expected = hermval(X["a"].to_numpy(dtype=float), coef)
    assert np.allclose(np.nan_to_num(out), np.nan_to_num(expected)), "poly replay must equal hermval(vals, coef)"
