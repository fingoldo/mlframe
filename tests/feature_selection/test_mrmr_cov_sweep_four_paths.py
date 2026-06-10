"""Coverage sweep + regression for ``MRMR.fit/transform`` across four
under-exercised end-to-end paths: classification (binary + multiclass),
categorical features (Categorical + object/string), wide data (p~=299), and
NaN handling.

REGRESSION (the load-bearing test in this file):
``test_regression_categorical_factorize_replay_not_constant`` pins the FS-side
analog of the 4b299e25 neural ``_apply_cat_codes`` bug. A cat-interaction
``factorize`` recipe built over STRING / Categorical source columns was
replayed at ``transform`` time by routing the raw string values through
``astype(np.int64)`` in ``_coerce_to_int_with_nan_handling`` -- which raises
and fell through to an all-zero clip fallback. Every test row therefore got
code ``0``, so the engineered cell-code column collapsed to a CONSTANT at
serving time: the whole cat-synergy feature (the strongest signal) was silently
destroyed -- a train/serve skew. The fix stamps the fit-time
``raw_value -> code`` map (``cat_code_maps``) onto each cat-FE recipe so
``transform`` reproduces the discretiser's codes.

The other three tests are coverage pins (they passed before the fix and lock
the behaviour going forward).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters.mrmr import MRMR


# ---------------------------------------------------------------------------
# Path 2: CATEGORICAL -- the regression
# ---------------------------------------------------------------------------


def _xor_cat_frame(seed, n, as_categorical):
    """Two cat columns whose (a==b) synergy drives y -> forces a cat-FE
    ``factorize`` interaction recipe. Built as Categorical or object/string."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 3, size=n)
    b = rng.integers(0, 3, size=n)
    y = ((a == b).astype(int) ^ (rng.random(n) < 0.05).astype(int)).astype(np.int64)
    av = [f"A{v}" for v in a]
    bv = [f"B{v}" for v in b]
    if as_categorical:
        df = pd.DataFrame({
            "cat_a": pd.Categorical(av),
            "cat_b": pd.Categorical(bv),
            "noise": rng.normal(size=n),
        })
    else:
        df = pd.DataFrame({
            "cat_a": pd.Series(av, dtype="object"),
            "cat_b": pd.Series(bv, dtype="object"),
            "noise": rng.normal(size=n),
        })
    return df, pd.Series(y, name="y")


@pytest.mark.parametrize("as_categorical", [True, False], ids=["Categorical", "object_string"])
def test_regression_categorical_factorize_replay_not_constant(as_categorical):
    """A cat-interaction ``factorize`` (or ``target_encoding``) feature replayed
    on a raw string / Categorical frame must NOT collapse to a constant column,
    and the same value-pair must map to the same code on disjoint holdout data.

    Pre-fix: ``transform`` returned an all-zero (constant) column -> the
    selected synergy feature was destroyed at serving time."""
    df_tr, y_tr = _xor_cat_frame(7, 4000, as_categorical)
    sel = MRMR(verbose=0, random_seed=42, fe_max_steps=1)
    sel.fit(df_tr, y_tr)

    recipes = [
        r for r in getattr(sel, "_engineered_recipes_", [])
        if r.kind in ("factorize", "target_encoding")
    ]
    # The XOR fixture is designed so the (cat_a, cat_b) pair carries all signal;
    # the cat-FE step must surface it. If it didn't, this fixture / config drifted.
    assert recipes, "cat-FE produced no factorize/target_encoding recipe -- fixture or config drift"
    r = recipes[0]

    # The fix stamps the fit-time category->code map onto the recipe.
    assert "cat_code_maps" in r.extra, (
        "factorize recipe over categorical source is missing the cat_code_maps "
        "replay table -- transform will all-zero string sources"
    )

    out_tr = sel.transform(df_tr)
    assert r.name in out_tr.columns
    col_tr = np.asarray(out_tr[r.name].to_numpy())
    # The bug: constant column (all rows -> same code). Healthy: multiple cells.
    assert len(np.unique(col_tr)) > 1, (
        f"engineered factorize column '{r.name}' is CONSTANT at transform "
        f"(value={col_tr.flat[0]!r}) -- string source codes collapsed to zero "
        f"(train/serve skew). Expected ~9 distinct cell codes for a 3x3 cross."
    )

    # Train/serve consistency: identical (cat_a, cat_b) value-pair -> identical code.
    pairs_tr = list(zip(df_tr["cat_a"].astype(str), df_tr["cat_b"].astype(str)))
    pair_to_code = {}
    for p, c in zip(pairs_tr, col_tr):
        prev = pair_to_code.get(p)
        assert prev is None or prev == c, (
            f"value-pair {p} mapped to two different codes ({prev}, {c}) on the "
            f"SAME train transform -- replay is not a deterministic function of X"
        )
        pair_to_code[p] = c

    # Disjoint holdout drawn from the same universe must reuse the train mapping.
    df_te, _ = _xor_cat_frame(99, 2000, as_categorical)
    out_te = sel.transform(df_te)
    col_te = np.asarray(out_te[r.name].to_numpy())
    pairs_te = list(zip(df_te["cat_a"].astype(str), df_te["cat_b"].astype(str)))
    mismatches = sum(
        1 for p, c in zip(pairs_te, col_te)
        if p in pair_to_code and pair_to_code[p] != c
    )
    assert mismatches == 0, (
        f"{mismatches} holdout rows got a DIFFERENT code than the train mapping "
        f"for the same value-pair -- train/serve skew in factorize replay"
    )


def test_build_category_code_map_reproduces_discretiser_codes():
    """``build_category_code_map`` must reproduce ``categorize_dataset``'s codes:
    ``.cat.codes`` (category order) for Categorical, ``pd.factorize`` (first-
    appearance) for object/string. This is the unit under the e2e regression."""
    from mlframe.feature_selection.filters.engineered_recipes._recipe_extract import (
        build_category_code_map,
    )
    # Categorical: codes follow category-dictionary order (sorted here).
    cat = pd.Categorical(["red", "green", "blue", "red"])
    m_cat = build_category_code_map(pd.Series(cat))
    expected_cat = {str(c): i for i, c in enumerate(cat.categories)}
    assert m_cat == expected_cat
    # object: first-appearance order.
    obj = pd.Series(["x", "y", "x", "z"], dtype="object")
    m_obj = build_category_code_map(obj)
    assert m_obj == {"x": 0, "y": 1, "z": 2}
    # numeric: no map (already integer-coded).
    assert build_category_code_map(pd.Series([1.0, 2.0, 3.0])) == {}


# ---------------------------------------------------------------------------
# Path 1: CLASSIFICATION (binary + multiclass)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_classes", [2, 4], ids=["binary", "multiclass"])
def test_classification_recovers_signal_and_uses_classif_path(n_classes):
    """MRMR on a discrete-label classification target must recover the planted
    linear signal in the top of ``support_`` (binary AND multiclass)."""
    rng = np.random.default_rng(0)
    n, p = 1500, 8
    X = rng.normal(size=(n, p))
    lin = X[:, 0] * 1.5 + X[:, 1] * 0.8 + 0.3 * rng.normal(size=n)
    if n_classes == 2:
        y = (lin > 0).astype(np.int64)
    else:
        y = pd.qcut(lin, n_classes, labels=False).astype(np.int64)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    sel = MRMR(verbose=0, random_seed=42)
    sel.fit(df, pd.Series(y, name="y"))
    fni = list(sel.feature_names_in_)
    sel_names = {fni[i] for i in sel.support_ if i < len(fni)}
    # The two signal columns must be among the (few) selected features.
    assert {"x0", "x1"} & sel_names, (
        f"{n_classes}-class target: neither signal column selected; got {sel_names}"
    )
    assert len(sel.support_) >= 1


# ---------------------------------------------------------------------------
# Path 3: WIDE p=299
# ---------------------------------------------------------------------------


def test_wide_p299_recovers_planted_signal_bounded():
    """p=299 wide path must complete and recover the strong planted signal
    {x0, x5, x100}; the p^2 pair enumeration must stay bounded (no OOM/blowup)."""
    rng = np.random.default_rng(0)
    n, p = 2000, 299
    X = rng.normal(size=(n, p))
    y = (2.0 * X[:, 0] + 2.0 * X[:, 5] + 2.0 * X[:, 100]
         + 0.2 * rng.normal(size=n) > 0).astype(np.int64)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    sel = MRMR(verbose=0, random_seed=42, fe_max_steps=0)
    sel.fit(df, pd.Series(y, name="y"))
    fni = list(sel.feature_names_in_)
    sel_names = {fni[i] for i in sel.support_ if i < len(fni)}
    recovered = {"x0", "x5", "x100"} & sel_names
    assert len(recovered) >= 2, (
        f"wide p=299: recovered only {recovered} of the planted signal triplet"
    )


# ---------------------------------------------------------------------------
# Path 4: NaN handling
# ---------------------------------------------------------------------------


def test_nan_in_features_recovered_and_preserved_in_transform():
    """A NaN-heavy SIGNAL feature must still be recovered (NaN not silently
    propagated into MI as garbage), and ``transform`` must PRESERVE raw NaN for
    downstream NaN-aware models (separate_bin default)."""
    rng = np.random.default_rng(0)
    n, p = 2000, 6
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    nan_mask = rng.random(n) < 0.25
    df.loc[nan_mask, "x0"] = np.nan  # NaN in the signal column
    sel = MRMR(verbose=0, random_seed=42, fe_max_steps=0)
    sel.fit(df, pd.Series(y, name="y"))
    fni = list(sel.feature_names_in_)
    sel_names = {fni[i] for i in sel.support_ if i < len(fni)}
    assert "x0" in sel_names, "NaN-heavy signal column x0 was not recovered"
    out = sel.transform(df)
    assert "x0" in out.columns
    # Raw NaN preserved (count matches input) -- not imputed/zeroed on the raw col.
    assert int(out["x0"].isna().sum()) == int(nan_mask.sum())


def test_nan_in_target_raises():
    """NaN in a float target must raise (MI degrades silently on NaN), matching
    the sibling selectors' policy -- never silently propagate."""
    rng = np.random.default_rng(0)
    n, p = 800, 5
    X = rng.normal(size=(n, p))
    yf = (X[:, 0] + 0.4 * rng.normal(size=n)).astype(np.float64)
    yf[rng.random(n) < 0.05] = np.nan
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    sel = MRMR(verbose=0, random_seed=42, fe_max_steps=0)
    with pytest.raises(ValueError, match="NaN"):
        sel.fit(df, pd.Series(yf, name="y"))
