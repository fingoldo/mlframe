"""``mrmr_gains_[i]`` is the greedy gain of ``get_feature_names_out()[i]``, including after a pass re-orders the raw support.

The final alignment head-sliced the greedy log (selection order). The p >= n cap re-sorts the raw support by relevance and retention passes
re-add raws the screen never scored, so a caller zipping names with ``mrmr_gains_`` read another feature's gain.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core import _align_mrmr_gains


def _recipe(name, **extra):
    """A stand-in engineered recipe: a name and an ``extra`` dict."""
    return SimpleNamespace(name=name, extra=dict(extra))


def _fitted(support, log, n_features, recipes=(), gains=None):
    """A fitted-looking object carrying only what the alignment reads."""
    log = tuple({"name": n, "indices": tuple(ix), "gain": g} for n, ix, g in log)
    return SimpleNamespace(
        support_=np.asarray(support),
        _predictors_log_=log,
        _engineered_recipes_=list(recipes),
        n_features_=n_features,
        mrmr_gains_=np.asarray([e["gain"] for e in log] if gains is None else gains, dtype=np.float64),
    )


def test_resorted_support_gets_each_features_own_gain():
    """Support re-sorted away from selection order (the p >= n cap): every gain follows its own column."""
    log = [("c5", (5,), 0.50), ("c2", (2,), 0.30), ("c9", (9,), 0.20), ("c1", (1,), 0.10)]
    est = _fitted(support=[9, 5, 1], log=log, n_features=3)
    _align_mrmr_gains(est)
    assert est.mrmr_gains_.tolist() == [0.20, 0.50, 0.10]


def test_selection_order_support_is_unchanged():
    """Support already in selection order: the result equals the previous positional slice."""
    log = [("c5", (5,), 0.50), ("c2", (2,), 0.30), ("c9", (9,), 0.20)]
    est = _fitted(support=[5, 2, 9], log=log, n_features=3)
    _align_mrmr_gains(est)
    assert est.mrmr_gains_.tolist() == [0.50, 0.30, 0.20]


def test_unscored_raw_and_engineered_tail():
    """A raw the screen never picked gets 0.0; engineered columns take their logged gain by recipe name, else 0.0."""
    log = [("c5", (5,), 0.50), ("mul(c1,c2)", (1, 2), 0.40), ("c2", (2,), 0.30)]
    recipes = [_recipe("mul(c1,c2)"), _recipe("sqr(c7)"), _recipe("legacy", requires_refit_for_replay=True)]
    est = _fitted(support=[2, 5, 7], log=log, n_features=5, recipes=recipes)
    _align_mrmr_gains(est)
    # raw c2, c5, c7 (unscored) then advertised engineered mul(c1,c2), sqr(c7); the legacy recipe is not advertised.
    assert est.mrmr_gains_.tolist() == [0.30, 0.50, 0.0, 0.40, 0.0]
    assert est.mrmr_gains_.shape[0] == est.n_features_


def test_length_contract_with_boolean_support():
    """A boolean support mask is honoured and the result is padded or cut to n_features_."""
    log = [("c0", (0,), 0.9), ("c3", (3,), 0.4)]
    est = _fitted(support=np.array([True, False, False, True]), log=log, n_features=4)
    _align_mrmr_gains(est)
    assert est.mrmr_gains_.tolist() == [0.9, 0.4, 0.0, 0.0]
    est2 = _fitted(support=[0, 3], log=log, n_features=1)
    _align_mrmr_gains(est2)
    assert est2.mrmr_gains_.tolist() == [0.9]


def test_no_predictor_log_keeps_positional_alignment():
    """Without a log there is nothing to re-pair by, so the previous head-slice / pad applies."""
    est = SimpleNamespace(support_=np.array([4, 1]), _predictors_log_=(), _engineered_recipes_=[], n_features_=3, mrmr_gains_=np.array([0.7, 0.2]))
    _align_mrmr_gains(est)
    assert est.mrmr_gains_.tolist() == [0.7, 0.2, 0.0]


def test_p_ge_n_fit_gains_follow_names_and_provenance_matches_the_log():
    """End to end in the p >= n regime: each output name carries its own logged gain, and provenance reports the logged gain too."""
    import pandas as pd

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n, p = 60, 200
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"c{i}" for i in range(p)])
    y = ((X["c0"] + 0.8 * X["c1"] - 0.6 * X["c2"] + 0.3 * rng.normal(size=n)) > 0).astype(np.int64)
    MRMR._FIT_CACHE.clear()
    m = MRMR(random_seed=0, n_jobs=1, verbose=0, fe_max_steps=0, full_npermutations=3, baseline_npermutations=2).fit(X, y)
    log = m._predictors_log_
    assert log, "fixture precondition: the greedy screen must log picks"
    by_index = {int(e["indices"][0]): float(e["gain"]) for e in log if len(e["indices"]) == 1}
    names = list(m.get_feature_names_out())
    gains = np.asarray(m.mrmr_gains_, dtype=np.float64)
    assert gains.shape == (m.n_features_,)
    fni = list(m.feature_names_in_)
    for i, name in enumerate(names[: len(m.support_)]):
        expected = by_index.get(fni.index(name), 0.0)
        assert gains[i] == expected, f"mrmr_gains_[{i}] for {name!r} is {gains[i]}, its logged gain is {expected}"
    prov = m.fe_provenance_
    by_name = {str(e["name"]): float(e["gain"]) for e in log}
    assert list(prov.iterrows()), "the loop below must iterate at least once"
    for _, row in prov.iterrows():
        if int(row["support_rank"]) >= 0 and str(row["feature_name"]) in by_name:
            assert float(row["mrmr_gain"]) == by_name[str(row["feature_name"])], f"provenance gain for {row['feature_name']!r} differs from the log"
