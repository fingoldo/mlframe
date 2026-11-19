"""The batched device binning of the cross-fold stability vote gives the codes and the verdicts of the per-job host replay."""

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _fe_stability_vote as sv
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin
from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe, apply_recipe


def _recipes():
    """Two unary_binary recipes over raw columns: both rebuilt by the device replay."""
    plain = EngineeredRecipe(name="a*b", kind="unary_binary", src_names=("a", "b"), unary_names=("identity", "identity"), binary_name="mul", extra={})
    return {"a*b": plain, "a+b": EngineeredRecipe(name="a+b", kind="unary_binary", src_names=("a", "b"), unary_names=("identity", "identity"), binary_name="add", extra={})}


def _frame(n=6000):
    """A frame with two informative raw columns, one noise column and a target."""
    r = np.random.default_rng(0)
    X = pd.DataFrame({"a": r.standard_normal(n).astype(np.float32), "b": r.standard_normal(n).astype(np.float32), "c": r.standard_normal(n).astype(np.float32)})
    return X, ((X["a"] * X["b"]) > 0).astype(int).to_numpy()


def test_batched_codes_equal_the_host_binner(monkeypatch):
    """Every (recipe, fold) code vector of the batched replay equals the host replay followed by the host quantile binner."""
    monkeypatch.setenv("MLFRAME_FE_VOTE_DEVICE", "1")
    X, _ = _frame()
    voted = _recipes()
    folds = sv._vote_folds(None, len(X), 4, np.random.default_rng(0))
    cache = {"__fold_cols__": sv._recipe_source_columns(voted.values())}
    out = sv._batched_replay_codes(voted, folds, X, 10, cache, {"a", "b", "c"}, len(X))
    checked = 0
    for (name, fold_no), codes in out.items():
        if isinstance(name, tuple):
            continue
        full = sv._slice_fold(X, folds[fold_no])
        want = _quantile_bin(np.asarray(np.asarray(apply_recipe(voted[name], full)).ravel(), dtype=_crit_np_dtype()), nbins=10)
        np.testing.assert_array_equal(codes, want)
        checked += 1
    assert checked == len(voted) * len(folds)


def test_vote_verdicts_do_not_depend_on_the_device_switch(monkeypatch):
    """The set of recipes failing the quorum is the same with the device replay forced on and forced off."""
    X, y = _frame()
    kw = dict(recipes=_recipes(), X=X, y_codes=y, feature_names_in=list(X.columns), nbins=10, k=4, quorum=0.6)
    monkeypatch.setenv("MLFRAME_FE_VOTE_DEVICE", "0")
    off = sv.confirm_recipes_cross_fold(rng=np.random.default_rng(0), **kw)
    monkeypatch.setenv("MLFRAME_FE_VOTE_DEVICE", "1")
    on = sv.confirm_recipes_cross_fold(rng=np.random.default_rng(0), **kw)
    assert on == off
