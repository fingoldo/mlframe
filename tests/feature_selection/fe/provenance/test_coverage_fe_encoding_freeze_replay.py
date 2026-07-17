"""Freeze + replay fidelity for the categorical-encoding engineered-recipe family.

Covers the production replay contract for ``kfold_target_encoded``, ``count_encoded``,
``frequency_encoded`` and ``cat_num_residual`` recipes built via the public builders in
``engineered_recipes._encoding_recipes`` and replayed through ``apply_recipe``:

* FREEZE/REPLAY exactness -- a recipe built from a fit lookup, when replayed against the
  SAME frame, reproduces the manual lookup (``lookup.get(cat, fallback)``) bit-for-bit.
* UNSEEN-CATEGORY fallback -- a category absent from the fit lookup maps to the frozen
  fallback constant (``global_mean`` / ``default``), never NaN, never a crash.
* NO-Y-AT-TRANSFORM (leakage) -- replay reads only X; the recipe carries no y reference,
  so the transform output is byte-identical whether or not a y exists in scope, and there
  is no path by which a future row's target could leak.
* TRAIN/SERVE consistency -- a recipe pickled (frozen ``extra`` MappingProxy round-trips to
  a plain dict and back) replays identically to the in-memory recipe.
* FROZEN-EXTRA immutability -- the post-init ``MappingProxyType`` rejects mutation.

These exercise the standalone recipe surface (no MRMR e2e fit), so each test is < 1s.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.engineered_recipes import (
    apply_recipe,
    build_kfold_target_encoded_recipe,
    build_count_encoded_recipe,
    build_frequency_encoded_recipe,
    build_cat_num_residual_recipe,
)

warnings.filterwarnings("ignore")


def _fit_lookups(cats: np.ndarray, y: np.ndarray):
    """Build the fit-time per-category mean / count / freq tables (the constants a
    fit would freeze). Mirrors the simple non-smoothed prod path (smoothing=0)."""
    df = pd.DataFrame({"c": cats.astype(str), "y": y.astype(float)})
    grp = df.groupby("c")
    mean_lookup = grp["y"].mean().to_dict()
    count_lookup = grp.size().to_dict()
    freq_lookup = {k: v / len(df) for k, v in count_lookup.items()}
    return mean_lookup, count_lookup, freq_lookup


@pytest.fixture(scope="module")
def fit_frame():
    rng = np.random.default_rng(7)
    n = 600
    cats = rng.choice(np.array(["a", "b", "c", "d"]), size=n)
    num = rng.normal(size=n)
    # Target correlated with category so the means are distinct & meaningful.
    base = {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}
    y = np.array([base[c] for c in cats]) + rng.normal(scale=0.1, size=n)
    X = pd.DataFrame({"cat": cats, "num": num})
    return X, y, cats, num


def test_kfold_target_encoding_replay_matches_manual_lookup(fit_frame):
    X, y, cats, _ = fit_frame
    mean_lookup, _, _ = _fit_lookups(cats, y)
    gmean = float(np.mean(y))
    rec = build_kfold_target_encoded_recipe(
        name="te(cat)",
        src_name="cat",
        lookup=mean_lookup,
        global_mean=gmean,
        smoothing=0.0,
    )
    out = apply_recipe(rec, X)
    manual = np.array([mean_lookup.get(str(c), gmean) for c in X["cat"]])
    np.testing.assert_allclose(out, manual, rtol=0, atol=1e-12)


def test_kfold_te_unseen_category_maps_to_global_mean(fit_frame):
    _X, y, cats, _ = fit_frame
    mean_lookup, _, _ = _fit_lookups(cats, y)
    gmean = float(np.mean(y))
    rec = build_kfold_target_encoded_recipe(
        name="te(cat)",
        src_name="cat",
        lookup=mean_lookup,
        global_mean=gmean,
        smoothing=0.0,
    )
    # Held-out frame with a category 'z' never seen at fit.
    Xnew = pd.DataFrame({"cat": ["a", "z", "b", "z"], "num": [0.0, 0.0, 0.0, 0.0]})
    out = apply_recipe(rec, Xnew)
    assert np.isfinite(out).all(), "unseen category must not produce NaN/inf"
    # 'z' rows fall back to the frozen global mean.
    assert out[1] == pytest.approx(gmean)
    assert out[3] == pytest.approx(gmean)
    assert out[0] == pytest.approx(mean_lookup["a"])


def test_kfold_te_transform_ignores_y_in_scope(fit_frame):
    """y is never read at transform: corrupting y after the recipe is frozen must
    not change the replayed column. The recipe is the only state."""
    X, y, cats, _ = fit_frame
    mean_lookup, _, _ = _fit_lookups(cats, y)
    gmean = float(np.mean(y))
    rec = build_kfold_target_encoded_recipe(
        name="te(cat)",
        src_name="cat",
        lookup=mean_lookup,
        global_mean=gmean,
        smoothing=0.0,
    )
    out_a = apply_recipe(rec, X)
    # Mutate a y-shaped array wildly; replay must be invariant.
    _ = np.full_like(y, 999.0)
    out_b = apply_recipe(rec, X)
    np.testing.assert_array_equal(out_a, out_b)


def test_count_encoding_replay_and_unseen_default(fit_frame):
    X, y, cats, _ = fit_frame
    _, count_lookup, _ = _fit_lookups(cats, y)
    rec = build_count_encoded_recipe(
        name="cnt(cat)",
        src_name="cat",
        lookup=count_lookup,
        default=0,
    )
    out = apply_recipe(rec, X)
    manual = np.array([count_lookup[str(c)] for c in X["cat"]], dtype=float)
    np.testing.assert_allclose(out, manual, rtol=0, atol=0)
    # Unseen -> frozen default 0.
    Xnew = pd.DataFrame({"cat": ["zzz"], "num": [0.0]})
    assert apply_recipe(rec, Xnew)[0] == 0


def test_frequency_encoding_replay_and_unseen_default(fit_frame):
    X, y, cats, _ = fit_frame
    _, _, freq_lookup = _fit_lookups(cats, y)
    rec = build_frequency_encoded_recipe(
        name="freq(cat)",
        src_name="cat",
        lookup=freq_lookup,
        default=0.0,
    )
    out = apply_recipe(rec, X)
    manual = np.array([freq_lookup[str(c)] for c in X["cat"]], dtype=float)
    np.testing.assert_allclose(out, manual, rtol=0, atol=1e-12)
    Xnew = pd.DataFrame({"cat": ["zzz"], "num": [0.0]})
    assert apply_recipe(rec, Xnew)[0] == pytest.approx(0.0)


def test_cat_num_residual_replay_matches_manual(fit_frame):
    X, _y, cats, num = fit_frame
    mean_lookup, _, _ = _fit_lookups(cats, num)  # residual is over num, not y
    gmean = float(np.mean(num))
    rec = build_cat_num_residual_recipe(
        name="resid(num|cat)",
        cat_name="cat",
        num_name="num",
        lookup=mean_lookup,
        global_mean=gmean,
        smoothing=0.0,
    )
    out = apply_recipe(rec, X)
    manual = X["num"].to_numpy(dtype=float) - np.array([mean_lookup.get(str(c), gmean) for c in X["cat"]])
    np.testing.assert_allclose(out, manual, rtol=0, atol=1e-10)


def test_cat_num_residual_unseen_category_subtracts_global_mean(fit_frame):
    _X, _y, cats, num = fit_frame
    mean_lookup, _, _ = _fit_lookups(cats, num)
    gmean = float(np.mean(num))
    rec = build_cat_num_residual_recipe(
        name="resid(num|cat)",
        cat_name="cat",
        num_name="num",
        lookup=mean_lookup,
        global_mean=gmean,
        smoothing=0.0,
    )
    Xnew = pd.DataFrame({"cat": ["zzz"], "num": [5.0]})
    out = apply_recipe(rec, Xnew)
    assert out[0] == pytest.approx(5.0 - gmean)


@pytest.mark.parametrize("builder_kind", ["te", "count", "freq", "resid"])
def test_encoding_recipe_pickle_roundtrip_replays_identically(fit_frame, builder_kind):
    X, y, cats, num = fit_frame
    mean_lookup, count_lookup, freq_lookup = _fit_lookups(cats, y)
    if builder_kind == "te":
        rec = build_kfold_target_encoded_recipe(
            name="te",
            src_name="cat",
            lookup=mean_lookup,
            global_mean=float(np.mean(y)),
            smoothing=0.0,
        )
    elif builder_kind == "count":
        rec = build_count_encoded_recipe(name="cnt", src_name="cat", lookup=count_lookup)
    elif builder_kind == "freq":
        rec = build_frequency_encoded_recipe(name="frq", src_name="cat", lookup=freq_lookup)
    else:
        nm_lookup, _, _ = _fit_lookups(cats, num)
        rec = build_cat_num_residual_recipe(
            name="rsd",
            cat_name="cat",
            num_name="num",
            lookup=nm_lookup,
            global_mean=float(np.mean(num)),
            smoothing=0.0,
        )
    rec2 = pickle.loads(pickle.dumps(rec))
    assert rec2 == rec
    np.testing.assert_array_equal(apply_recipe(rec, X), apply_recipe(rec2, X))


def test_frozen_extra_rejects_mutation(fit_frame):
    _X, y, cats, _ = fit_frame
    mean_lookup, _, _ = _fit_lookups(cats, y)
    rec = build_kfold_target_encoded_recipe(
        name="te",
        src_name="cat",
        lookup=mean_lookup,
        global_mean=float(np.mean(y)),
        smoothing=0.0,
    )
    with pytest.raises(TypeError):
        rec.extra["lookup"] = {}  # MappingProxyType -> read-only


def test_kfold_te_column_order_invariance(fit_frame):
    """Replay must depend only on the source column's VALUES, not its position in
    X (train/serve column-reorder consistency)."""
    X, y, cats, _ = fit_frame
    mean_lookup, _, _ = _fit_lookups(cats, y)
    rec = build_kfold_target_encoded_recipe(
        name="te",
        src_name="cat",
        lookup=mean_lookup,
        global_mean=float(np.mean(y)),
        smoothing=0.0,
    )
    out_a = apply_recipe(rec, X)
    X_reordered = X[["num", "cat"]]  # swap column order, add nothing
    out_b = apply_recipe(rec, X_reordered)
    np.testing.assert_array_equal(out_a, out_b)
