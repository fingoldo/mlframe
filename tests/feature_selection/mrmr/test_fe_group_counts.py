"""``fe_family_counts_`` and ``fe_step_counts_`` say which FE groups produced how many columns, and how many each FE stage added."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._fe_group_counts import family_counts_section, group_counts, recipe_group, registry_recipes
from mlframe.feature_selection.filters.engineered_recipes._orth_basis_recipes import build_orth_univariate_recipe


def _orth(name, basis):
    """One orth-univariate recipe of ``basis``."""
    return build_orth_univariate_recipe(name=name, src_name="x", basis=basis, degree=2)


def test_the_group_is_the_kind_refined_by_the_basis():
    """Two bases of one kind are two groups; counts are over distinct recipe names, largest first."""
    recipes = [_orth("x__He2", "He"), _orth("x__He3", "He"), _orth("y__rbf2", "rbf"), _orth("x__He2", "He")]
    assert recipe_group(recipes[0]) == "orth_univariate:He"
    assert group_counts(recipes) == {"orth_univariate:He": 2, "orth_univariate:rbf": 1}


def test_registry_recipes_reads_every_dict_registry():
    """The per-family registries of a fit are dicts of name -> recipe; anything else on the store is ignored."""
    import types

    store = types.SimpleNamespace(hybrid_orth={"a": _orth("a", "He")}, mi_greedy={"b": _orth("b", "rbf")}, note="x", n=3)
    assert sorted(r.name for r in registry_recipes(store)) == ["a", "b"]


def test_a_fit_records_the_groups_and_one_row_per_fe_step():
    """After a fit: groups carry the three counts, the stage list starts with the cascades and then numbers every FE step, and the report prints both blocks."""
    from mlframe.feature_selection.filters._mrmr_fe_provenance import get_fe_report
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame(rng.standard_normal((n, 6)).astype(np.float32), columns=[f"x{i}" for i in range(6)])
    y = ((X["x0"] ** 2 - 1) + 0.5 * X["x1"] * X["x2"] + 0.3 * rng.standard_normal(n) > 0).astype(int).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(verbose=0, random_seed=0, fe_max_steps=2).fit(X, y)
    family = m.fe_family_counts_
    assert family and all(set(row) == {"emitted", "kept_for_screen", "selected"} for row in family.values())
    assert sum(row["emitted"] for row in family.values()) == m.fe_step_counts_[0]["new_columns"] > 0
    assert m.fe_step_counts_[0]["stage"] == "cascades"
    assert [row["stage"] for row in m.fe_step_counts_[1:]] == list(range(1, len(m.fe_step_counts_)))
    assert m.fe_provenance_.attrs["family_counts"] is family
    report = get_fe_report(m)
    assert "FE columns by group" in report and "FE columns by stage" in report


def test_the_section_is_empty_without_a_fit():
    """An estimator that recorded nothing adds nothing to the report."""
    import types

    assert family_counts_section(types.SimpleNamespace()) == ""
