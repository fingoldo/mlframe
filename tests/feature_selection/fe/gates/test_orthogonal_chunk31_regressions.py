"""Regressions for the orthogonal quadruplet / routing / auto-scorer / three-gate FE modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_three_gate_mi_fe import (
    _bin_with_train_edges,
    _bin_with_train_edges_batched,
    score_features_by_kfold_oof_mi,
)


def test_three_gate_oof_binner_keeps_signal_of_a_column_with_missing_values() -> None:
    """A raw column with NaNs keeps its finite-value signal in the OOF baseline MI (NaN edges used to erase it)."""
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.standard_normal(n)
    y = (x > 0).astype(int)
    raw = pd.DataFrame({"a": x})
    raw.loc[rng.choice(n, 150, replace=False), "a"] = np.nan
    eng = pd.DataFrame({"a__He2": np.nan_to_num(x**2)})
    df = score_features_by_kfold_oof_mi(raw, eng, y, n_folds=4, seed=0)
    assert float(df["baseline_mi_oof"].iloc[0]) > 0.3


def test_three_gate_batched_binner_matches_scalar_on_nan_columns() -> None:
    """The batched binner equals the scalar one column by column when a column holds NaN."""
    rng = np.random.default_rng(1)
    train = rng.standard_normal((400, 3))
    test = rng.standard_normal((100, 3))
    train[::7, 1] = np.nan
    test[::5, 1] = np.nan
    tb, sb = _bin_with_train_edges_batched(train, test, 10)
    for j in range(3):
        t_ref, s_ref = _bin_with_train_edges(train[:, j], test[:, j], 10)
        np.testing.assert_array_equal(tb[:, j], t_ref)
        np.testing.assert_array_equal(sb[:, j], s_ref)
    assert np.unique(tb[:, 1]).size > 3


def test_auto_scorer_plug_in_batch_bins_a_unit_interval_continuous_target() -> None:
    """The batched plug-in scorer sees the signal of a continuous y inside [0, 1) instead of truncating it to one class."""
    from mlframe.feature_selection.filters._orthogonal_scorer_auto_fe import _batch_one_scorer

    rng = np.random.default_rng(2)
    n = 2000
    x = rng.standard_normal((n, 2)).astype(np.float32)
    y = 1.0 / (1.0 + np.exp(-x[:, 0].astype(np.float64)))
    out = _batch_one_scorer("plug_in", x, y, nbins=10, random_state=0, n_neighbors=3, copula_n_bins=20, dcor_n_sample=500, call=None)
    assert out[0] > 0.3 > out[1]


def test_routing_cells_are_built_at_float64() -> None:
    """The routed cell values equal the float64 basis evaluation of the column, not a float32-narrowed one."""
    from mlframe.feature_selection.filters._orthogonal_routing_fe import _routing_cells_of_column
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _evaluate_basis_column

    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": 1.0e6 + rng.standard_normal(500)})
    _, values, _ = _routing_cells_of_column(X, "a", ("raw",), ("hermite",), (2,))
    ref = _evaluate_basis_column(X["a"].to_numpy(dtype=np.float64), "hermite", 2)
    np.testing.assert_allclose(values[0], ref, rtol=1e-9, atol=1e-9)


def test_quadruplet_recipe_params_are_finite_and_replay_matches_with_missing_values() -> None:
    """A source with NaNs yields finite frozen params and a replay equal to the fit-time column on the observed rows."""
    from mlframe.feature_selection.filters._orthogonal_quadruplet_fe import hybrid_orth_mi_quadruplet_fe_with_recipes
    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame(rng.standard_normal((n, 4)) * 3, columns=list("abcd"))
    y = (np.prod(np.sign(X.to_numpy()), axis=1) > 0).astype(int)
    X.loc[rng.choice(n, 200, replace=False), "a"] = np.nan
    X_aug, _, _, recipes = hybrid_orth_mi_quadruplet_fe_with_recipes(X, y, degrees=(2,), quadruplet_max_degree=1, top_quadruplet_count=1, min_uplift=1.0)
    quad = [r for r in recipes if r.kind == "orth_quadruplet_cross"]
    assert quad
    assert len(quad) > 0, "no quadruplet recipe was replayed"
    for r in quad:
        assert any(key.startswith("preprocess_params") for key in r.extra), f"{r.name} froze no preprocess params"
        for key, pp in r.extra.items():
            if key.startswith("preprocess_params"):
                assert all(np.isfinite(v) for v in pp.values())
        observed = X["a"].notna().to_numpy()
        np.testing.assert_allclose(apply_recipe(r, X)[observed], X_aug[r.name].to_numpy()[observed], atol=1e-4)
