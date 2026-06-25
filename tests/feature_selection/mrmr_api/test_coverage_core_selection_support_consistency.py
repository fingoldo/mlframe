"""Core-selection coverage: get_support / get_feature_names_out / transform input-space consistency.

The MRMR public selector exposes three views of "what got selected":
- ``support_`` / ``get_support()`` -- INPUT-SPACE: indices/mask of length ``n_features_in_`` over the RAW columns.
- ``transform(X)`` -- the materialised output matrix (raw selected columns + any engineered columns).
- ``get_feature_names_out()`` -- the names matching transform()'s output columns (raw + engineered).

These have a precise contract that must hold regardless of FE: get_support stays in input space (never counts
engineered columns), while names_out / transform width agree with each other. With FE disabled, all three collapse
to the same raw set. These behaviours had no dedicated input-space-consistency test under mrmr_api/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _no_fe(**kw):
    base = dict(
        random_seed=0, verbose=0, fe_max_steps=0, interactions_max_order=1,
        dcd_enable=False, cluster_aggregate_enable=False, build_friend_graph=False,
        cat_fe_config=None, fe_hinge_enable=False, fe_modular_enable=False,
        fe_pairwise_modular_enable=False, fe_integer_lattice_enable=False,
        fe_row_argmax_enable=False, fe_conditional_gate_enable=False,
    )
    base.update(kw)
    return MRMR(**base)


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    X = pd.DataFrame({
        "x0": x0,
        "x1": x1,
        "x0dup": x0 + rng.normal(0, 1e-6, n),   # near-duplicate of x0 (redundant)
        "noise": rng.normal(size=n),
        "noise2": rng.normal(size=n),
    })
    y = (x0 + 0.5 * x1 > 0).astype(int)
    return X, y


def test_get_support_is_input_space_mask_and_indices_agree():
    """get_support(indices=False) is a bool mask of length n_features_in_; indices=True is np.where of it."""
    X, y = _data()
    m = _no_fe().fit(X, y)
    mask = m.get_support()
    idx = m.get_support(indices=True)
    assert mask.dtype == bool
    assert mask.shape == (m.n_features_in_,)
    np.testing.assert_array_equal(np.where(mask)[0], idx)
    # support_ (raw indices) must be exactly the True positions.
    np.testing.assert_array_equal(np.sort(np.asarray(m.support_, dtype=np.intp)), idx)


def test_names_out_matches_transform_width_and_columns():
    """get_feature_names_out() names exactly the columns transform() emits, in order."""
    X, y = _data()
    m = _no_fe().fit(X, y)
    names = list(m.get_feature_names_out())
    out = m.transform(X)
    assert out.shape[1] == len(names)
    if hasattr(out, "columns"):
        assert list(out.columns) == names


def test_raw_selected_names_are_a_subset_of_names_out():
    """The raw selected columns (by support_) appear, in input order, at the head of names_out."""
    X, y = _data()
    m = _no_fe().fit(X, y)
    raw_selected = [str(c) for i, c in enumerate(m.feature_names_in_) if i in set(np.asarray(m.support_, dtype=np.intp))]
    names = list(map(str, m.get_feature_names_out()))
    assert names[: len(raw_selected)] == raw_selected


def test_no_fe_collapses_all_three_views_to_same_raw_set():
    """With every FE family off, get_support / names_out / transform describe the IDENTICAL raw set."""
    X, y = _data()
    m = _no_fe().fit(X, y)
    sup_names = {str(m.feature_names_in_[i]) for i in np.asarray(m.support_, dtype=np.intp)}
    names_out = set(map(str, m.get_feature_names_out()))
    tf = m.transform(X)
    tf_names = set(map(str, tf.columns)) if hasattr(tf, "columns") else None
    assert names_out == sup_names
    if tf_names is not None:
        assert tf_names == sup_names


def test_get_support_excludes_redundant_near_duplicate():
    """The near-duplicate of an already-selected feature is NOT in support_ (redundancy gate)."""
    X, y = _data()
    m = _no_fe().fit(X, y)
    selected = {str(m.feature_names_in_[i]) for i in np.asarray(m.support_, dtype=np.intp)}
    assert "x0" in selected
    assert "x0dup" not in selected, "near-duplicate x0dup must be rejected by the redundancy criterion"


def test_get_feature_names_out_before_fit_raises_notfitted():
    from sklearn.exceptions import NotFittedError
    m = _no_fe()
    with pytest.raises(NotFittedError):
        m.get_feature_names_out()


def test_get_support_before_fit_raises_notfitted():
    from sklearn.exceptions import NotFittedError
    m = _no_fe()
    with pytest.raises(NotFittedError):
        m.get_support()


def test_get_feature_names_out_rejects_mismatched_input_features():
    """sklearn column-drift contract: input_features != fit-time names -> ValueError (DataFrame fit)."""
    X, y = _data()
    m = _no_fe().fit(X, y)
    wrong = ["totally", "wrong", "names", "here", "now"]
    with pytest.raises(ValueError):
        m.get_feature_names_out(input_features=wrong)
    # Correct names echo back fine.
    ok = m.get_feature_names_out(input_features=list(X.columns))
    assert len(ok) == len(m.get_feature_names_out())
