"""Tests for the SVD co-occurrence categorical embedding (Dyakonov's code_factor).

Imports the submodule DIRECTLY (not via the package facade) per project test
convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_engineering.cat_cooccurrence_svd import (
    apply_cat_cooccurrence_svd,
    cat_cooccurrence_svd_fit,
    cat_cooccurrence_svd_with_recipes,
    engineered_name_cooccur_svd,
)


# ---------------------------------------------------------------------------
# Unit: happy path
# ---------------------------------------------------------------------------


def test_fit_returns_expected_shape_and_recipe():
    """Fit returns expected shape and recipe."""
    rng = np.random.default_rng(0)
    n = 500
    city = rng.choice(["Moscow", "London", "Paris"], size=n)
    sex = rng.choice(["M", "F"], size=n)
    X = pd.DataFrame({"city": city, "sex": sex})

    emb, recipe = cat_cooccurrence_svd_fit(X, "city", "sex", n_components=1)

    assert emb.shape == (n, 1)
    assert recipe["n_components"] == 1
    # Three cities -> three lookup keys, each a length-1 vector.
    assert set(recipe["lookup"].keys()) == {"Moscow", "London", "Paris"}
    assert all(len(v) == 1 for v in recipe["lookup"].values())
    assert recipe["default"] == [0.0]


def test_same_category_gets_same_code():
    """Same category gets same code."""
    X = pd.DataFrame(
        {
            "a": ["x", "x", "y", "y", "z", "z"],
            "b": ["p", "q", "p", "q", "p", "q"],
        }
    )
    emb, _ = cat_cooccurrence_svd_fit(X, "a", "b", n_components=1)
    # Rows 0,1 are both category x -> identical code; likewise y (2,3), z (4,5).
    assert emb[0, 0] == pytest.approx(emb[1, 0])
    assert emb[2, 0] == pytest.approx(emb[3, 0])
    assert emb[4, 0] == pytest.approx(emb[5, 0])


def test_sign_canonicalisation_is_deterministic():
    """Sign canonicalisation is deterministic."""
    X = pd.DataFrame(
        {
            "a": ["x", "y", "z", "x", "y", "z"] * 20,
            "b": ["p", "q", "r", "q", "r", "p"] * 20,
        }
    )
    e1, r1 = cat_cooccurrence_svd_fit(X, "a", "b", n_components=2)
    e2, r2 = cat_cooccurrence_svd_fit(X, "a", "b", n_components=2)
    # Bit-identical across repeat fits (sign pinned to largest-|entry| positive).
    np.testing.assert_array_equal(e1, e2)
    for k in ("lookup", "default", "n_components"):
        assert r1[k] == r2[k]
    # Largest-magnitude entry of each component is positive.
    for k in range(r1["n_components"]):
        col = e1[:, k]
        j = int(np.argmax(np.abs(col)))
        assert col[j] >= 0.0


def test_multi_component_capped_at_rank():
    # 3 x 2 contingency -> rank <= 2, so requesting 5 components yields 2.
    """Multi component capped at rank."""
    X = pd.DataFrame(
        {
            "a": ["x", "y", "z"] * 30,
            "b": ["p", "q"] * 45,
        }
    )
    emb, recipe = cat_cooccurrence_svd_fit(X, "a", "b", n_components=5)
    assert emb.shape[1] == 2
    assert recipe["n_components"] == 2


# ---------------------------------------------------------------------------
# Unit: fit/apply consistency, unseen categories, NaN
# ---------------------------------------------------------------------------


def test_apply_reproduces_fit_on_same_frame():
    """Apply reproduces fit on same frame."""
    rng = np.random.default_rng(1)
    n = 400
    X = pd.DataFrame(
        {
            "a": rng.choice(list("abcde"), size=n),
            "b": rng.choice(list("pqrs"), size=n),
        }
    )
    emb, recipe = cat_cooccurrence_svd_fit(X, "a", "b", n_components=2)
    recipe = {**recipe, "src_col": "a", "other_col": "b"}
    replayed = apply_cat_cooccurrence_svd(X, "a", recipe)
    np.testing.assert_allclose(replayed, emb, rtol=1e-12, atol=1e-12)


def test_unseen_category_maps_to_default_zero_vector():
    # Distinct partner profiles per category so the association code is non-zero:
    # x co-occurs only with p, y only with q, z with both.
    """Unseen category maps to default zero vector."""
    X = pd.DataFrame(
        {
            "a": (["x"] * 10) + (["y"] * 10) + (["z"] * 10),
            "b": (["p"] * 10) + (["q"] * 10) + (["p"] * 5 + ["q"] * 5),
        }
    )
    _, recipe = cat_cooccurrence_svd_fit(X, "a", "b", n_components=1)
    recipe = {**recipe, "src_col": "a", "other_col": "b"}
    X_test = pd.DataFrame({"a": ["x", "UNSEEN"], "b": ["p", "q"]})
    out = apply_cat_cooccurrence_svd(X_test, "a", recipe)
    assert out.shape == (2, 1)
    # Unseen category -> the zero (default) vector.
    np.testing.assert_array_equal(out[1], np.asarray(recipe["default"]))
    assert out[0, 0] != 0.0  # seen category x has a distinct partner profile -> non-zero code


def test_nan_forms_its_own_category():
    """Nan forms its own category."""
    X = pd.DataFrame(
        {
            "a": ["x", "y", np.nan, "x", np.nan],
            "b": ["p", "q", "p", "q", "q"],
        }
    )
    emb, recipe = cat_cooccurrence_svd_fit(X, "a", "b", n_components=1)
    assert "__nan__" in recipe["lookup"]
    # The two NaN rows (idx 2, 4) share the same code.
    assert emb[2, 0] == pytest.approx(emb[4, 0])


def test_int_float_dtype_drift_resolves_same_category():
    # Distinct partner profiles so codes are non-zero (raw normalize also works,
    # but ca with real association is the default path we want to exercise).
    """Int float dtype drift resolves same category."""
    X_fit = pd.DataFrame(
        {
            "a": [1, 1, 2, 2, 3, 3],
            "b": ["p", "p", "q", "q", "p", "q"],
        }
    )
    _emb_fit, recipe = cat_cooccurrence_svd_fit(X_fit, "a", "b", n_components=1)
    recipe = {**recipe, "src_col": "a", "other_col": "b"}
    # Apply with the SAME categories encoded as floats -- must hit the learned codes,
    # NOT fall through to the default (which would happen on a str('1') vs str('1.0') mismatch).
    X_test = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["p", "q", "p"]})
    out = apply_cat_cooccurrence_svd(X_test, "a", recipe)
    # Codes for categories 1 and 2 (opposite partner profiles) must differ and be non-default.
    assert out[0, 0] != out[1, 0]
    assert not (out[0, 0] == 0.0 and out[1, 0] == 0.0)


# ---------------------------------------------------------------------------
# Unit: error paths
# ---------------------------------------------------------------------------


def test_fit_empty_raises():
    """Fit empty raises."""
    with pytest.raises(ValueError, match="empty"):
        cat_cooccurrence_svd_fit(pd.DataFrame({"a": [], "b": []}), "a", "b")


def test_fit_missing_column_raises():
    """Fit missing column raises."""
    X = pd.DataFrame({"a": ["x"], "b": ["p"]})
    with pytest.raises(ValueError, match="missing"):
        cat_cooccurrence_svd_fit(X, "a", "nope")


def test_fit_bad_n_components_raises():
    """Fit bad n components raises."""
    X = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    with pytest.raises(ValueError, match="n_components"):
        cat_cooccurrence_svd_fit(X, "a", "b", n_components=0)


def test_fit_bad_normalize_raises():
    """Fit bad normalize raises."""
    X = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    with pytest.raises(ValueError, match="normalize"):
        cat_cooccurrence_svd_fit(X, "a", "b", normalize="bogus")


def test_raw_normalize_roundtrips():
    """Raw normalize roundtrips."""
    rng = np.random.default_rng(3)
    n = 300
    X = pd.DataFrame(
        {
            "a": rng.choice(list("abcd"), size=n),
            "b": rng.choice(list("pqr"), size=n),
        }
    )
    emb, recipe = cat_cooccurrence_svd_fit(X, "a", "b", n_components=2, normalize="raw")
    assert recipe["normalize"] == "raw"
    recipe = {**recipe, "src_col": "a", "other_col": "b"}
    replayed = apply_cat_cooccurrence_svd(X, "a", recipe)
    np.testing.assert_allclose(replayed, emb, rtol=1e-12, atol=1e-12)


def test_apply_missing_recipe_key_raises():
    """Apply missing recipe key raises."""
    X = pd.DataFrame({"a": ["x"]})
    with pytest.raises(KeyError):
        apply_cat_cooccurrence_svd(X, "a", {"lookup": {}})


def test_apply_non_dataframe_raises():
    """Apply non dataframe raises."""
    with pytest.raises(TypeError):
        apply_cat_cooccurrence_svd(
            np.array([1, 2]),
            "a",
            {
                "lookup": {},
                "default": [0.0],
                "n_components": 1,
            },
        )


# ---------------------------------------------------------------------------
# Unit: batch wrapper
# ---------------------------------------------------------------------------


def test_with_recipes_appends_and_skips_self_pairs():
    """With recipes appends and skips self pairs."""
    rng = np.random.default_rng(2)
    n = 300
    X = pd.DataFrame(
        {
            "city": rng.choice(["Moscow", "London", "Paris"], size=n),
            "sex": rng.choice(["M", "F"], size=n),
        }
    )
    X_aug, appended, recipes = cat_cooccurrence_svd_with_recipes(
        X,
        src_cols=["city", "sex"],
        other_cols=["city", "sex"],
        n_components=1,
    )
    # (city,sex) and (sex,city) -> 2 pairs; self-pairs skipped.
    assert len(recipes) == 2
    assert engineered_name_cooccur_svd("city", "sex", 0) in appended
    assert engineered_name_cooccur_svd("sex", "city", 0) in appended
    for name in appended:
        assert name in X_aug.columns


# ---------------------------------------------------------------------------
# biz_value: the co-occurrence SVD embedding recovers latent structure that
# predicts y, LEAKAGE-FREE (never touches y), and beats arbitrary label
# encoding for a linear model on HELD-OUT rows.
# ---------------------------------------------------------------------------


def _latent_group_cooccur_bed(seed_data: int):
    """A latent group drives y AND the co-occurrence of two categoricals.

    ``k`` source categories belong to one of two latent groups. Group-1 cats
    co-occur mostly with partner set {5..9}, group-0 cats with {0..4}. y depends
    ONLY on the latent group (plus noise); the raw src label has NO ordering
    relationship to the group. The co-occurrence embedding is the ONLY way a
    1-D-per-category encoding can recover the group without touching y.
    """
    rng = np.random.default_rng(seed_data)
    n = 2000
    k = 20
    group_of_cat = np.array([0] * (k // 2) + [1] * (k - k // 2))
    rng.shuffle(group_of_cat)

    src_code = rng.integers(0, k, size=n)
    grp = group_of_cat[src_code]
    partner = np.where(
        rng.random(n) < 0.9,
        np.where(grp == 0, rng.integers(0, 5, size=n), rng.integers(5, 10, size=n)),
        rng.integers(0, 10, size=n),  # 10% cross-group noise
    )
    logit = np.where(grp == 1, 1.4, -1.4) + rng.normal(0, 0.5, size=n)
    y = (1.0 / (1.0 + np.exp(-logit)) > rng.random(n)).astype(int)
    src = np.array([f"cat_{c:02d}" for c in src_code])
    oth = np.array([f"p_{c}" for c in partner])
    return pd.DataFrame({"src": src, "other": oth}), y


def _holdout_auc_of_embedding(X, y, normalize, n_components):
    """Helper: Holdout auc of embedding."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.4, random_state=0, stratify=y)
    emb_tr, recipe = cat_cooccurrence_svd_fit(
        X.iloc[tr],
        "src",
        "other",
        n_components=n_components,
        normalize=normalize,
    )
    recipe = {**recipe, "src_col": "src", "other_col": "other"}
    emb_te = apply_cat_cooccurrence_svd(X.iloc[te], "src", recipe)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(emb_tr, y[tr])
    auc = roc_auc_score(y[te], lr.predict_proba(emb_te)[:, 1])

    # Arbitrary lexicographic label baseline of the SAME source column.
    src = X["src"].to_numpy()
    cats_sorted = sorted(set(src[tr]))
    code_map = {c: i for i, c in enumerate(cats_sorted)}
    lab_tr = np.array([[code_map.get(c, len(code_map))] for c in src[tr]], dtype=float)
    lab_te = np.array([[code_map.get(c, len(code_map))] for c in src[te]], dtype=float)
    lr2 = LogisticRegression(max_iter=1000)
    lr2.fit(lab_tr, y[tr])
    auc_lab = roc_auc_score(y[te], lr2.predict_proba(lab_te)[:, 1])
    return auc, auc_lab


def test_biz_val_cooccur_svd_beats_label_encoding_on_holdout():
    """The leakage-free CA co-occurrence embedding recovers the latent group and
    beats arbitrary label encoding on a held-out split -- with NO y reference.

    Measured (CA, n_components=1, seed 7): embedding AUC ~0.80 vs label AUC ~0.55.
    Floor set ~7% below the measured embedding AUC and clearly above the label
    baseline (delta floor 0.15, measured ~0.25).
    """
    X, y = _latent_group_cooccur_bed(7)
    auc_svd, auc_lab = _holdout_auc_of_embedding(X, y, normalize="ca", n_components=1)
    assert auc_svd >= 0.74, f"CA co-occurrence embedding AUC {auc_svd:.3f} below floor 0.74"
    assert auc_svd >= auc_lab + 0.15, f"CA co-occurrence embedding AUC {auc_svd:.3f} should beat label AUC {auc_lab:.3f} by >=0.15"


def test_biz_val_ca_default_beats_raw_leading_vector():
    """The CA default surfaces the association axis in the LEADING component,
    where the raw count-matrix SVD's leading vector is swamped by the marginal-
    frequency size effect.

    Measured (mean over 6 seeds, n_components=1): CA holdout AUC ~0.79, raw ~0.72.
    Floor: CA beats raw by >=0.04 on the mean (measured ~0.07).
    """
    ca_aucs, raw_aucs = [], []
    for sd in range(6):
        X, y = _latent_group_cooccur_bed(sd)
        ca_aucs.append(_holdout_auc_of_embedding(X, y, normalize="ca", n_components=1)[0])
        raw_aucs.append(_holdout_auc_of_embedding(X, y, normalize="raw", n_components=1)[0])
    ca_mean, raw_mean = float(np.mean(ca_aucs)), float(np.mean(raw_aucs))
    assert ca_mean >= raw_mean + 0.04, f"CA leading-component AUC mean {ca_mean:.3f} should beat raw {raw_mean:.3f} by >=0.04"
    assert ca_mean >= 0.75, f"CA leading-component AUC mean {ca_mean:.3f} below floor 0.75"
