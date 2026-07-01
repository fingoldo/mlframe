"""Regression guard for orth-cluster-basis replay parity under drift
(audit 2026-06-03: cluster-aggregate-6/7).

The recipe replay used to RE-FIT the aggregate z-score / PC1 loading and the
basis preprocess on whatever test batch it was given, so a given input row's
engineered value depended on the rest of the batch (and shifted under train/test
distribution drift). The fix persists the fit-time member mean/std/signs +
combiner weights + basis preprocess params in the recipe and APPLIES them at
replay. These tests assert the engineered value of a row is invariant to the
batch it is scored in.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_cluster_basis_fe import (
    hybrid_orth_mi_cluster_basis_fe_with_recipes,
)
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe


def _fit_recipe(aggregator="mean_z", seed=0, n=2500):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    Xtr = pd.DataFrame({
        "a": z + 0.2 * rng.standard_normal(n),
        "b": z + 0.2 * rng.standard_normal(n),
        "c": z + 0.2 * rng.standard_normal(n),
    })
    y = (z ** 2 > 0.6).astype(int)  # nonlinear in z -> He_2 of the aggregate wins
    X_aug, scores, recipes = hybrid_orth_mi_cluster_basis_fe_with_recipes(
        Xtr, y, cluster_members={"a": ["a", "b", "c"]}, cols=list(Xtr.columns),
        basis="hermite", degrees=(2, 3), aggregator=aggregator, top_k=5,
        min_uplift=1.0, min_abs_mi_frac=0.0,
    )
    return recipes


def _batch_invariance(recipe, seed=1):
    rng = np.random.default_rng(seed)
    cols = list(recipe.src_names)
    Xte = pd.DataFrame({c: rng.standard_normal(300) for c in cols})
    # Same rows, but with extreme drifted outliers appended -> changes ANY
    # batch-refit statistic (mean/std/min/max) while leaving the first 300 rows
    # unchanged.
    drift = pd.DataFrame({c: 50.0 + 20.0 * rng.standard_normal(60) for c in cols})
    Xte_drift = pd.concat([Xte, drift], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a refit-fallback warning would mean no stats persisted
        v_a = np.asarray(apply_recipe(recipe, Xte), dtype=np.float64)
        v_b = np.asarray(apply_recipe(recipe, Xte_drift), dtype=np.float64)[:len(Xte)]
    return v_a, v_b


def test_recipe_persists_fit_time_stats():
    recipes = _fit_recipe("mean_z")
    assert recipes, "expected >=1 cluster-basis recipe"
    for r in recipes:
        assert "agg_stats" in r.extra and r.extra["agg_stats"] is not None
        assert "basis_params" in r.extra and r.extra["basis_params"] is not None


def test_mean_z_replay_invariant_to_batch_distribution():
    recipes = _fit_recipe("mean_z")
    v_a, v_b = _batch_invariance(recipes[0])
    assert np.allclose(v_a, v_b, rtol=0, atol=1e-9), (
        "replay value depends on the test batch distribution -> parity broken"
    )


def test_pc1_replay_invariant_to_batch_distribution():
    recipes = _fit_recipe("pc1")
    assert recipes, "expected >=1 pc1 cluster-basis recipe"
    v_a, v_b = _batch_invariance(recipes[0])
    assert np.allclose(v_a, v_b, rtol=0, atol=1e-9)


def test_legacy_recipe_without_stats_warns_and_still_replays():
    # A pre-fix pickle has a recipe with NO persisted stats. Build that legacy
    # shape via the recipe builder (omitting agg_stats/basis_params). Replay must
    # still produce finite output but emit the refit-fallback warning. (recipe
    # .extra is a frozen mappingproxy by design, so we build a fresh one.)
    from mlframe.feature_selection.filters.engineered_recipes import (
        build_orth_cluster_basis_recipe,
    )
    legacy = build_orth_cluster_basis_recipe(
        name="clusterbasis_legacy", members=("a", "b", "c"),
        basis="hermite", degree=2, aggregator="mean_z",
    )
    assert "agg_stats" not in legacy.extra and "basis_params" not in legacy.extra
    rng = np.random.default_rng(2)
    Xte = pd.DataFrame({c: rng.standard_normal(200) for c in ("a", "b", "c")})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = np.asarray(apply_recipe(legacy, Xte), dtype=np.float64)
        assert any("persisted fit-time stats" in str(x.message) for x in w)
    assert np.all(np.isfinite(np.nan_to_num(out)))
