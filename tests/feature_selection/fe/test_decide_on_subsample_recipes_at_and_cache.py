"""``fe_decide_on_subsample`` keeps the layout of a family whose recipes are not last, and gathers the shared subsample once."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl import _helpers as H


def _family(calls):
    """A family returning ``(X, appended, recipes, scores)`` that adds one column and records the frames it was given."""

    def fam(X_in, y_in, **_kw):
        """Add ``e1`` (a copy of ``a``) and return the family's own tuple layout."""
        calls.append(X_in)
        out = X_in.copy()
        out["e1"] = out["a"] * 2.0
        return out, ["e1"], [SimpleNamespace(name="e1")], pd.DataFrame({"score": [1.0]})

    return fam


def test_recipes_at_keeps_the_family_layout(monkeypatch):
    """With ``recipes_at=2`` the result is ``(X, appended, recipes, scores)`` with the winner replayed at full n."""
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    y = rng.integers(0, 2, n)
    monkeypatch.setattr("mlframe.feature_selection.filters.engineered_recipes.apply_recipe", lambda r, X_: np.asarray(X_["a"]) * 2.0)
    out = H.fe_decide_on_subsample(_family([]), X, y, subsample_n=500, subsample_seed=0, recipes_at=2)
    X_aug, appended, recipes, scores = out
    assert len(X_aug) == n and "e1" in X_aug.columns
    assert appended == ["e1"] and recipes[0].name == "e1" and "score" in scores.columns


def test_the_shared_subsample_is_gathered_once_and_families_do_not_share_mutations(monkeypatch):
    """Two families deciding on the same index array get one gather; a column one adds to its frame is not seen by the other."""
    rng = np.random.default_rng(1)
    n = 2000
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    y = rng.integers(0, 2, n)
    idx = np.sort(rng.choice(n, 400, replace=False)).astype(np.int64)
    monkeypatch.setattr("mlframe.feature_selection.filters.engineered_recipes.apply_recipe", lambda r, X_: np.asarray(X_["a"]) * 2.0)
    gathers = []
    import mlframe.feature_selection.filters._fe_frame_ops as ops

    real = ops.fe_subsample_to_pandas
    monkeypatch.setattr(ops, "fe_subsample_to_pandas", lambda X_, i: (gathers.append(1), real(X_, i))[1])
    seen: list = []
    H.fe_decide_on_subsample(_family(seen), X, y, shared_subsample_idx=idx, recipes_at=2)
    H.fe_decide_on_subsample(_family(seen), X, y, shared_subsample_idx=idx, recipes_at=2)
    assert len(gathers) == 1
    assert all("e1" not in f.columns for f in seen)
