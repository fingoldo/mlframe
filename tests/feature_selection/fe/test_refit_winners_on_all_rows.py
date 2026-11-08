"""A group-statistic family that decides on the shared subsample gets its winners computed again on ALL rows (refit_winners), not replayed from the subsample's tables."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._grouped_quantile_fe import hybrid_grouped_quantile_fe
from mlframe.feature_selection.filters._mrmr_fit_impl._helpers import _group_and_number, fe_decide_on_subsample, refit_winners


def _frame():
    """A frame with two group columns and three numeric columns, one of them depending on its group."""
    rng = np.random.default_rng(0)
    n = 6000
    g1 = rng.integers(0, 6, n)
    X = pd.DataFrame({"g1": g1.astype(float), "g2": rng.integers(0, 4, n).astype(float), "a": rng.standard_normal(n) + g1, "b": rng.standard_normal(n), "c": rng.standard_normal(n)})
    y = (X["a"] - g1 + 0.2 * rng.standard_normal(n) > 0).astype(int).to_numpy()
    return X, y


def test_refit_winners_equal_the_family_run_on_all_rows():
    """The winners' columns after the subsample decision equal the columns the family itself computes on all rows for the same names."""
    X, y = _frame()
    idx = np.sort(np.random.default_rng(1).choice(len(X), 1500, replace=False)).astype(np.int64)
    kw = {"group_cols": ["g1", "g2"], "num_cols": ["a", "b", "c"], "top_k": 4}
    refit = refit_winners(hybrid_grouped_quantile_fe, _group_and_number, top_k=10**9, min_mi=-1e9, min_uplift=-1e9)
    X_sub, appended, recipes, _scores = fe_decide_on_subsample(hybrid_grouped_quantile_fe, X, y, shared_subsample_idx=idx, recipes_at=2, refit=refit, **kw)
    assert appended and len(X_sub) == len(X)
    reference = hybrid_grouped_quantile_fe(X, y, group_cols=["g1", "g2"], num_cols=["a", "b", "c"], top_k=10**9, min_mi=-1e9, min_uplift=-1e9)[0]
    for name in appended:
        np.testing.assert_allclose(X_sub[name].to_numpy(dtype=float), reference[name].to_numpy(dtype=float), rtol=1e-9, atol=1e-12, equal_nan=True)
    assert {r.name for r in recipes} == set(appended)


def test_refit_works_for_a_family_that_returns_three_values():
    """A family returning ``(X_aug, appended, recipes)`` (the binned numeric aggregate has no scores frame) is refitted too: the recipes are at index 2 of both layouts."""
    from types import SimpleNamespace

    rng = np.random.default_rng(3)
    n = 4000
    X = pd.DataFrame({"g": rng.standard_normal(n), "a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    y = rng.integers(0, 2, n)
    calls: list = []

    def family(X_in, y_in, *, group_num_cols=None, agg_num_cols=None, **_kw):
        """One column per (group, aggregate) pair it is given; records the row count of every call."""
        calls.append(len(X_in))
        out = X_in.copy()
        recipes, names = [], []
        for g in group_num_cols or ["g"]:
            for a in agg_num_cols or ["a", "b"]:
                name = f"agg({a}|{g})"
                out[name] = X_in[a].to_numpy() * 2.0
                recipes.append(SimpleNamespace(name=name, src_names=(g, a)))
                names.append(name)
        return out, names, recipes

    idx = np.sort(rng.choice(n, 1000, replace=False)).astype(np.int64)
    refit = refit_winners(family, lambda r: {"group_num_cols": (r.src_names[0],), "agg_num_cols": (r.src_names[1],)})
    X_out, appended, recipes = fe_decide_on_subsample(family, X, y, shared_subsample_idx=idx, refit=refit)
    assert len(X_out) == n and set(appended) == {"agg(a|g)", "agg(b|g)"}
    assert calls[0] == 1000 and all(c == n for c in calls[1:])  # the decision on the draw, then one narrowed call per winner's pair on all rows
    assert {r.name for r in recipes} == set(appended)
