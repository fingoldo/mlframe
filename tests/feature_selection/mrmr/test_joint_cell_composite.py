"""Unit tests for the joint-cell composite columns of the triplet stage and the main-effect restore that keeps a dominant operand beside the composite."""

import numpy as np
import pandas as pd


def _conjunction_frame(n=20000, seed=0):
    """Binary A, B, C and a target that is a strong main effect of A plus a conjunction A AND B AND C."""
    rng = np.random.default_rng(seed)
    A, B, C = (rng.integers(0, 2, n) for _ in range(3))
    logit = -1.5 + 2.5 * A + 3.0 * (A & B & C)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.int64)
    X = pd.DataFrame({"A": A.astype(float), "B": B.astype(float), "C": C.astype(float)})
    return X, y


def _xor_frame(n=20000, seed=0):
    """Binary x1, x2, x3 and a target that is their parity."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, 2, n) for _ in range(3)]
    y = (cols[0] ^ cols[1] ^ cols[2]).astype(np.int64)
    return pd.DataFrame({f"x{i + 1}": c.astype(float) for i, c in enumerate(cols)}), y


def _stage_product(X, y, legs):
    """``(X with the triplet stage's own product column, that column's name)`` for the three ``legs``, built by the same stage the fit runs."""
    from mlframe.feature_selection.filters._orthogonal_triplet_fe import hybrid_orth_mi_triplet_fe_with_recipes

    X_t, _, _, _ = hybrid_orth_mi_triplet_fe_with_recipes(X[list(legs)], y, cols=list(legs), top_k=0, explicit_triplets=[tuple(legs)])
    name = next(c for c in X_t.columns if c not in X.columns and c.split("__", 1)[0].count("*") == 2)
    return X.assign(**{name: X_t[name].to_numpy()}), name


def test_cell_column_is_added_for_a_conjunction():
    """A conjunction is invisible to the parity product, so the joint cell table is emitted beside it."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._hybrid_orth_family_variants._triplet_cell_columns import triplet_cell_columns

    X, y = _conjunction_frame()
    X, name = _stage_product(X, y, ("A", "B", "C"))
    columns, recipes = triplet_cell_columns(X, [name], y)
    assert list(columns) == ["cell3(A,B,C)"]
    assert set(recipes) == {"cell3(A,B,C)"}
    assert np.unique(columns["cell3(A,B,C)"]).size == 8


def test_cell_column_is_skipped_when_the_product_already_carries_the_target():
    """For a three-way XOR the parity product is the target, so the cell table adds nothing and is not emitted."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._hybrid_orth_family_variants._triplet_cell_columns import triplet_cell_columns

    X, y = _xor_frame()
    X, name = _stage_product(X, y, ("x1", "x2", "x3"))
    columns, recipes = triplet_cell_columns(X, [name], y)
    assert columns == {} and recipes == {}


def test_cell_column_is_skipped_for_a_leg_with_many_levels():
    """A continuous leg has too many levels for a cell table, so nothing is emitted."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._hybrid_orth_family_variants._triplet_cell_columns import triplet_cell_columns

    X, y = _conjunction_frame()
    X, name = _stage_product(X, y, ("A", "B", "C"))
    X = X.assign(C=np.random.default_rng(1).standard_normal(len(X)))
    columns, _ = triplet_cell_columns(X, [name], y)
    assert columns == {}


def _restore(X, y, selected_names):
    """Run the main-effect restore on a stub fitted object whose only engineered feature is the joint cell of A, B and C."""
    from types import SimpleNamespace

    from mlframe.feature_selection.filters._mrmr_fit_impl._main_effect_restore import run_main_effect_restore
    from mlframe.feature_selection.filters._mrmr_fit_impl._selection_compaction import _synthesized_cell_triple

    _, rec, _ = _synthesized_cell_triple(X, ("A", "B", "C"))
    fitted = SimpleNamespace(
        _engineered_recipes_=[rec],
        feature_names_in_=list(X.columns),
        _raw_redundancy_dropped_=set(),
        random_seed=0,
        support_=np.array([], dtype=np.int64),
    )
    cols = list(X.columns)
    data = X.to_numpy().astype(np.int64)
    selected = [cols.index(n) for n in selected_names]
    changed, out = run_main_effect_restore(fitted, X, selected, cols, data, y, False)
    return changed, [cols[i] for i in out], fitted


def test_main_effect_restore_keeps_the_dominant_operand_only():
    """A carries most of the composite's information and returns beside it; B and C only appear through the conjunction and stay out."""
    X, y = _conjunction_frame()
    changed, kept, fitted = _restore(X, y, [])
    assert changed and kept == ["A"]
    assert list(fitted.support_) == [0]


def test_main_effect_restore_leaves_a_pure_interaction_alone():
    """With no operand carrying a main effect (a three-way XOR, tested through a cell3-named recipe) nothing is restored."""
    X, y = _xor_frame()
    X = X.rename(columns={"x1": "A", "x2": "B", "x3": "C"})
    changed, kept, _ = _restore(X, y, [])
    assert not changed and kept == []
