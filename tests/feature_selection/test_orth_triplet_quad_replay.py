"""Orth triplet/quadruplet cross-basis recipes must replay slice-consistently (audit P0, 2026-06-13).

These recipes previously REFIT the per-leg basis preprocess (z-score mean/std, min-max lo/hi) from the
APPLY-time rows, so a row-slice / drifted test frame shifted the basis axis and silently emitted wrong
feature values (same class as the pair "BUG2 FIX 2026-06-12", never propagated to triplet/quad). The fix
freezes the fit-time preprocess params into the recipe.

Direct unit test of the recipe build+apply (not gated on the FE producing triplet features): a recipe
built WITH frozen params replays byte-exact on any slice/drift; the SAME recipe built WITHOUT frozen
params (legacy path) DRIFTS on a distribution-shifted frame -- proving both that the bug is real and
that the freeze fixes it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _evaluate_basis_column
from mlframe.feature_selection.filters._orthogonal_triplet_fe_recipes import build_orth_triplet_cross_recipe
from mlframe.feature_selection.filters._orthogonal_quadruplet_fe_recipes import build_orth_quadruplet_cross_recipe


def _frame(n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.lognormal(0.0, 1.0, n),
        "b": rng.standard_normal(n) * 3.0,
        "c": rng.random(n) * 10.0,
        "d": rng.standard_t(4, n),
    })


def test_triplet_recipe_frozen_params_replay_slice_consistent():
    df = _frame(4000, 0)
    legs = [("a", "hermite", 2), ("b", "legendre", 1), ("c", "laguerre", 2)]
    pps = [_evaluate_basis_column(df[col].to_numpy(float), basis, deg, return_params=True)[1]
           for (col, basis, deg) in legs]
    rec = build_orth_triplet_cross_recipe(
        name="tri", src_a_name="a", src_b_name="b", src_c_name="c",
        basis_i="hermite", basis_j="legendre", basis_k="laguerre",
        deg_a=2, deg_b=1, deg_c=2,
        preprocess_params_i=pps[0], preprocess_params_j=pps[1], preprocess_params_k=pps[2],
    )
    full = np.asarray(apply_recipe(rec, df), dtype=np.float64)
    sl = df.iloc[1000:1700].reset_index(drop=True)  # a slice with shifted per-column moments
    part = np.asarray(apply_recipe(rec, sl), dtype=np.float64)
    assert np.allclose(full[1000:1700], part, atol=1e-9, rtol=0.0), "frozen triplet recipe drifted on a slice"


def test_triplet_recipe_unfrozen_DRIFTS_on_slice_proving_the_bug():
    """Sanity / non-vacuity: the SAME recipe WITHOUT frozen params refits the axis from the apply-time
    rows, so a SLICE (whose sample mean/std differ NON-affinely from the full frame) replays DIFFERENT
    values than the full frame at those rows -- the exact slice-replay corruption the freeze fixes.
    (Note: z-score is affine-invariant, so an affine shift would NOT drift; a row-slice does.)"""
    df = _frame(4000, 0)
    rec = build_orth_triplet_cross_recipe(  # no preprocess_params -> legacy refit path
        name="tri", src_a_name="a", src_b_name="b", src_c_name="c",
        basis_i="hermite", basis_j="legendre", basis_k="laguerre", deg_a=2, deg_b=1, deg_c=2,
    )
    full = np.asarray(apply_recipe(rec, df), dtype=np.float64)
    sl = df.iloc[1000:1700].reset_index(drop=True)
    part = np.asarray(apply_recipe(rec, sl), dtype=np.float64)
    assert not np.allclose(full[1000:1700], part, atol=1e-6), (
        "expected the UNFROZEN recipe to drift on a slice (bug not reproduced -- test would be vacuous)"
    )


def test_quadruplet_recipe_frozen_params_replay_slice_consistent():
    df = _frame(4000, 1)
    legs = [("a", "hermite", 1), ("b", "chebyshev", 2), ("c", "legendre", 1), ("d", "hermite", 2)]
    pps = [_evaluate_basis_column(df[col].to_numpy(float), basis, deg, return_params=True)[1]
           for (col, basis, deg) in legs]
    rec = build_orth_quadruplet_cross_recipe(
        name="quad", src_a_name="a", src_b_name="b", src_c_name="c", src_d_name="d",
        basis_i="hermite", basis_j="chebyshev", basis_k="legendre", basis_l="hermite",
        deg_a=1, deg_b=2, deg_c=1, deg_d=2,
        preprocess_params_i=pps[0], preprocess_params_j=pps[1],
        preprocess_params_k=pps[2], preprocess_params_l=pps[3],
    )
    full = np.asarray(apply_recipe(rec, df), dtype=np.float64)
    sl = df.iloc[1500:2200].reset_index(drop=True)
    part = np.asarray(apply_recipe(rec, sl), dtype=np.float64)
    assert np.allclose(full[1500:2200], part, atol=1e-9, rtol=0.0), "frozen quadruplet recipe drifted on a slice"
