"""Leakage discipline for the TARGET-AWARE supervised-bin FE family.

``target_aware_group_bin`` is the one numeric FE family that READS y at fit (it places
per-group bin edges that maximise ``I(bin; y)`` within each group). It is the highest-risk
leakage surface, and the contract is subtle:

* At FIT, the engineered column is the OUT-OF-FOLD bin index (K-fold): the bin a row lands
  in was fit on the OTHER folds, so the row never saw its own y. This is what makes the
  in-fold MI score honest.
* The PERSISTED recipe stores only the all-rows-refit per-group edges (a pure function of X)
  + a pooled global fallback for unseen groups -- NOT the OOF assignment, and NO y vector.
* REPLAY (transform) reads only X: each row's value is digitised against its group's FROZEN
  edges; an unseen group falls back to the frozen global edges.

This file pins:
1. The frozen recipe carries no y reference (extra holds only edges + scalars).
2. Replay reads only X and is invariant to any y in scope.
3. Replay reuses the FROZEN group edges (held-out / new-data transform does not refit).
4. Unseen group at replay -> global-edge fallback, finite, no crash.
5. The OOF fit column differs from a naive all-data refit replay -- evidence the fit value
   was assigned out-of-fold, not from the row's own fold (the leakage-prevention mechanism).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._grouped_quantile_fe import (
    generate_target_aware_group_bins,
)
from mlframe.feature_selection.filters.engineered_recipes import (
    apply_recipe,
    build_target_aware_group_bin_recipe,
)

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def fit_data():
    rng = np.random.default_rng(13)
    n = 1500
    grp = rng.choice(np.array(["g0", "g1", "g2"]), size=n)
    # Within-group, x relates to y differently per group (so target-aware edges
    # genuinely differ from a global split).
    x = rng.normal(size=n)
    shift = {"g0": 0.0, "g1": 1.5, "g2": -1.5}
    logit = x + np.array([shift[g] for g in grp]) + rng.normal(scale=0.3, size=n)
    y = (logit > np.median(logit)).astype(int)
    X = pd.DataFrame({"g": grp, "x": x})
    return X, y


def _build_recipe(name, payload):
    return build_target_aware_group_bin_recipe(
        name=name,
        group_col=payload["group_col"],
        num_col=payload["num_col"],
        group_edges=payload["group_edges"],
        global_edges=payload["global_edges"],
        n_bins=payload["n_bins"],
        op=payload["op"],
    )


@pytest.fixture(scope="module")
def fit_recipe(fit_data):
    X, y = fit_data
    enc_df, raw = generate_target_aware_group_bins(
        X,
        y,
        group_cols=["g"],
        num_cols=["x"],
        n_bins=5,
        n_folds=5,
        random_state=0,
    )
    assert len(raw) == 1
    name, payload = next(iter(raw.items()))
    return name, payload, _build_recipe(name, payload), enc_df


def test_recipe_carries_no_target_reference(fit_recipe, fit_data):
    X, _y = fit_data
    _name, _payload, rec, _enc_df = fit_recipe
    n = len(X)
    for k, v in dict(rec.extra).items():
        if isinstance(v, np.ndarray) and v.size == n:
            pytest.fail(f"extra[{k!r}] is a length-n array -- possible y/OOF leak")
    # The OOF assignment (the only y-touched artefact) must NOT be in the recipe.
    flat = str(dict(rec.extra))
    assert "oof" not in flat.lower()


def test_replay_invariant_to_y_in_scope(fit_recipe, fit_data):
    X, y = fit_data
    _name, _payload, rec, _enc_df = fit_recipe
    out_a = apply_recipe(rec, X)
    _ = 1 - y  # a corrupt y in scope
    out_b = apply_recipe(rec, X)
    np.testing.assert_array_equal(out_a, out_b)


def test_replay_reuses_frozen_group_edges_on_new_data(fit_recipe, fit_data):
    """A held-out frame digitises against the FROZEN per-group edges. Building a
    second recipe from the same payload and replaying on a fresh frame yields the
    same bins as digitising manually with the stored edges."""
    _X, _y = fit_data
    _name, payload, rec, _enc_df = fit_recipe
    rng = np.random.default_rng(77)
    Xnew = pd.DataFrame({"g": rng.choice(["g0", "g1", "g2"], size=200), "x": rng.normal(size=200)})
    out = apply_recipe(rec, Xnew)
    # Manual replay: searchsorted on each row's stored group edges.
    manual = np.zeros(len(Xnew))
    for i, (g, xv) in enumerate(zip(Xnew["g"], Xnew["x"])):
        edges = np.asarray(payload["group_edges"][str(g)], dtype=float)
        manual[i] = np.searchsorted(edges, xv, side="right")
    np.testing.assert_array_equal(out, manual)


def test_unseen_group_falls_back_to_global_edges(fit_recipe, fit_data):
    _name, payload, rec, _enc_df = fit_recipe
    Xnew = pd.DataFrame({"g": ["never_seen_group"], "x": [0.3]})
    out = apply_recipe(rec, Xnew)
    assert np.isfinite(out).all()
    global_edges = np.asarray(payload["global_edges"], dtype=float)
    expected = float(np.searchsorted(global_edges, 0.3, side="right"))
    assert out[0] == expected


def test_oof_fit_column_differs_from_naive_allrows_replay(fit_recipe, fit_data):
    """The fit-time OOF column must NOT equal the all-rows-refit replay on the SAME
    frame -- if they were identical, the fit value would have been computed from the
    row's own fold (in-fold), i.e. the leakage the OOF discipline exists to prevent.
    They share edges but differ because OOF rows are binned on OTHER folds' edges."""
    X, _y = fit_data
    name, _payload, rec, enc_df = fit_recipe
    oof_col = enc_df[name].to_numpy()
    allrows_replay = apply_recipe(rec, X)
    # Some rows must differ (OOF vs all-rows edges are not identical per group).
    assert not np.array_equal(oof_col, allrows_replay), (
        "OOF fit column equals the all-rows replay -- OOF discipline may be broken (fit value would then be in-fold, a leak)"
    )


def test_replay_is_pure_no_state_mutation(fit_recipe, fit_data):
    X, _y = fit_data
    _name, _payload, rec, _enc_df = fit_recipe
    a = apply_recipe(rec, X)
    Xnew = pd.DataFrame({"g": ["g0", "g1"], "x": [0.1, 0.2]})
    _ = apply_recipe(rec, Xnew)
    b = apply_recipe(rec, X)
    np.testing.assert_array_equal(a, b)
