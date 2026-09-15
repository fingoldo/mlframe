"""fe_provenance_ covers every FE roster, and its greedy-rank lookup simplifies the predictor log once per report rather than once per name."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.feature_selection.filters import _mrmr_fe_provenance as prov_mod
from mlframe.feature_selection.filters._mrmr_fit_impl._fe_roster_attrs import FE_ROSTER_ATTRS
from mlframe.feature_selection.filters.engineered_recipes import _recipe_name_simplify

# Replayed by dedicated protection blocks and always also present in a labelled roster, so they need no bucket of their own.
_UNLABELLED_BY_DESIGN = {"_adaptive_fourier_features_", "_hinge_features_"}


def test_every_fe_roster_has_a_provenance_origin():
    """A roster without an origin never reaches the provenance frame unless its columns also carry a recipe."""
    mapped = {a for a, _ in prov_mod._ROSTER_ATTR_TO_ORIGIN}
    missing = sorted(set(FE_ROSTER_ATTRS) - _UNLABELLED_BY_DESIGN - mapped)
    assert not missing, f"FE rosters with no provenance origin: {missing}"
    assert set(dict(prov_mod._ROSTER_ATTR_TO_ORIGIN).values()) <= set(prov_mod.FE_ORIGIN_LABELS)
    catch_alls = [a for a, _ in prov_mod._ROSTER_ATTR_TO_ORIGIN][-2:]
    assert set(catch_alls) == {"mi_greedy_features_", "hybrid_orth_features_"}, "the catch-all rosters must stay last"


def test_roster_only_survivor_is_reported_with_its_origin():
    """A surviving wavelet column with no recipe object appears in fe_provenance_ labelled wavelet_basis, not dropped."""
    est = SimpleNamespace(
        feature_names_in_=np.array(["a", "b"]),
        support_=np.array([0]),
        _engineered_recipes_=[],
        _produced_recipes_=[],
        _predictors_log_=[],
        mrmr_gains_=np.array([0.3]),
        wavelet_features_=["a__haar_j2_k1"],
    )
    frame = prov_mod.compute_fe_provenance(est)
    row = frame[frame["feature_name"] == "a__haar_j2_k1"]
    assert len(row) == 1, f"roster-only survivor missing from provenance: {frame['feature_name'].tolist()}"
    assert row["origin"].iloc[0] == "wavelet_basis"


def _fitted_stub(n_names: int):
    """A fitted-looking estimator with ``n_names`` engineered survivors, all in the predictor log."""
    names = [f"mul(x{i},x{i + 1})" for i in range(n_names)]
    return SimpleNamespace(
        feature_names_in_=np.array(["a"]),
        support_=np.array([0]),
        _engineered_recipes_=[SimpleNamespace(name=n, kind="unary_binary", src_names=(), extra={}) for n in names],
        _produced_recipes_=[],
        _predictors_log_=[{"name": n} for n in reversed(names)],
        mrmr_gains_=np.linspace(1.0, 0.1, n_names),
    )


def test_greedy_rank_lookup_is_hoisted(monkeypatch):
    """Simplifier calls grow linearly with the number of names, and the ranks equal the per-name scanning reference."""
    real = _recipe_name_simplify.simplify_fe_name
    calls = {"n": 0}

    def counting(name):
        """Count calls to the name simplifier."""
        calls["n"] += 1
        return real(name)

    monkeypatch.setattr(_recipe_name_simplify, "simplify_fe_name", counting)
    counts = {}
    for n in (20, 80):
        calls["n"] = 0
        est = _fitted_stub(n)
        frame = prov_mod.compute_fe_provenance(est)
        counts[n] = calls["n"]
        reference = [prov_mod._greedy_rank_for_name(nm, est._predictors_log_) for nm in frame["feature_name"]]
        assert frame["support_rank"].tolist() == reference
    # A per-name scan costs ~n^2 simplifier calls (6400 at n=80); a hoisted index costs a small multiple of n.
    assert counts[80] < 20 * 80, f"simplifier calls not linear in names: {counts}"
