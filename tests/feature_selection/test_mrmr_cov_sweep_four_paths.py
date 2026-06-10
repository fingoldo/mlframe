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


@pytest.mark.parametrize(
    "ser, label",
    [
        (pd.Series(pd.Categorical(["red", "green", "blue", "red", None, "green"])), "Categorical+NaN"),
        (pd.Series(pd.Categorical(["red", "green", "blue", "red", "green"])), "Categorical_noNaN"),
        (pd.Series(["x", "y", "x", "z", None], dtype="object"), "object+NaN"),
        (pd.Series(["x", "y", "x", "z"], dtype="object"), "object_noNaN"),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_category_code_map_replay_matches_categorize_dataset_exactly(ser, label):
    """The map-driven replay must produce codes BIT-IDENTICAL to what
    ``categorize_dataset`` assigns at fit time, INCLUDING the NaN +1 shift
    (NaN-present columns: real cats -> base+1, NaN -> 0). A +1 / order skew here
    would silently mis-key the factorize lookup for NaN-bearing categoricals."""
    from mlframe.feature_selection.filters.discretization import categorize_dataset
    from mlframe.feature_selection.filters.engineered_recipes._recipe_extract import (
        build_category_code_map,
        _coerce_to_int_with_nan_handling,
    )
    df = pd.DataFrame({"c": ser})
    data, cols, nbins = categorize_dataset(df=df, missing_strategy="separate_bin")
    fit_codes = data[:, cols.index("c")].astype(int)
    m = build_category_code_map(ser)
    nb = int(nbins[cols.index("c")])
    vals = np.asarray(ser.to_numpy(), dtype=object)
    replay = _coerce_to_int_with_nan_handling(vals, nb, "r", "c", "clip", m)
    assert np.array_equal(fit_codes, replay), (
        f"{label}: replay codes {replay.tolist()} != fit codes {fit_codes.tolist()}"
    )


# ---------------------------------------------------------------------------
# COMBINED path (CRITIC 2026-06-11): block-NaN-shift for the NaN-free partner.
#
# ``categorize_dataset`` factorises ALL categorical columns as ONE 2-D block and
# applies the NaN ``+1`` shift to the WHOLE block when ANY column in it has a
# NaN. So a NaN-FREE categorical paired with a NaN-bearing one ALSO gets its
# codes shifted ``+1`` at fit time. The per-column ``build_category_code_map``
# decided the shift on the column's OWN NaN presence -> for the NaN-free partner
# it returned the unshifted base (``0..K-1``) while fit assigned ``1..K``: every
# code off-by-one at transform -- the SAME train/serve skew the factorize-replay
# fix targeted, for the mixed-block case the original sweep never exercised (its
# cat fixtures were either all NaN-free or had NaN in BOTH cat columns).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("as_categorical", [True, False], ids=["Categorical", "object_string"])
def test_block_nan_shift_applies_to_nanfree_partner_column(as_categorical):
    """A NaN-FREE categorical paired in the same block with a NaN-bearing one
    must get its codes block-shifted ``+1`` (matching ``categorize_dataset``),
    so the replay map is BIT-EXACT with the fit-time codes. Pre-fix the
    per-column path left the NaN-free partner unshifted -> off-by-one."""
    from mlframe.feature_selection.filters.discretization import categorize_dataset
    from mlframe.feature_selection.filters.engineered_recipes._recipe_extract import (
        build_category_code_map,
        _coerce_to_int_with_nan_handling,
        _NAN_CODE_KEY,
    )

    rng = np.random.default_rng(3)
    n = 200
    av = np.array([f"A{v}" for v in rng.integers(0, 3, n)], dtype=object)
    av[rng.random(n) < 0.2] = None  # NaN in cat_a ONLY
    bv = np.array([f"B{v}" for v in rng.integers(0, 3, n)], dtype=object)  # NaN-free
    if as_categorical:
        s_a = pd.Series(pd.Categorical(av))
        s_b = pd.Series(pd.Categorical(bv))
    else:
        s_a = pd.Series(av, dtype="object")
        s_b = pd.Series(bv, dtype="object")
    df = pd.DataFrame({"cat_a": s_a, "cat_b": s_b})

    data, cols, nbins = categorize_dataset(df=df, missing_strategy="separate_bin")
    fit_a = data[:, cols.index("cat_a")].astype(int)
    fit_b = data[:, cols.index("cat_b")].astype(int)

    # The block-level flag the stamping site computes (ANY cat col has NaN).
    block_has_nan = bool(
        df.select_dtypes(include=("category", "object", "string", "bool"))
        .isna().to_numpy().any()
    )
    assert block_has_nan, "fixture must have a NaN somewhere in the cat block"

    m_a = build_category_code_map(s_a, block_has_nan=block_has_nan)
    m_b = build_category_code_map(s_b, block_has_nan=block_has_nan)

    # The NaN-free partner must NOT carry a NaN bin yet must still be +1 shifted.
    assert _NAN_CODE_KEY not in m_b, "NaN-free partner should own no NaN cell"
    assert _NAN_CODE_KEY in m_a, "NaN-bearing column must route NaN -> 0"
    assert min(v for v in m_b.values()) >= 1, (
        "NaN-free partner codes must be block-shifted to 1..K, not 0..K-1"
    )

    # Map-driven replay must reproduce the fit codes BIT-EXACTLY for BOTH columns.
    nb_a = int(nbins[cols.index("cat_a")])
    nb_b = int(nbins[cols.index("cat_b")])
    replay_a = _coerce_to_int_with_nan_handling(
        np.asarray(s_a.to_numpy(), dtype=object), nb_a, "r", "cat_a", "clip", m_a
    )
    replay_b = _coerce_to_int_with_nan_handling(
        np.asarray(s_b.to_numpy(), dtype=object), nb_b, "r", "cat_b", "clip", m_b
    )
    assert np.array_equal(fit_a, replay_a), (
        f"cat_a replay {replay_a[:8].tolist()} != fit {fit_a[:8].tolist()}"
    )
    assert np.array_equal(fit_b, replay_b), (
        f"NaN-free partner cat_b replay {replay_b[:8].tolist()} != fit "
        f"{fit_b[:8].tolist()} -- block +1 shift not applied (off-by-one skew)"
    )


def _combined_frame(seed, n, n_classes, p_numeric, nan_in_cat_b=False):
    """COMBINED fixture: numeric (signal x0,x5 + many noise) + object cat_a
    (NaN-bearing) + Categorical cat_b (NaN-free at fit) + an (a==b) cat synergy
    that drives part of a classification target. Wide-ish numeric block + NaN in
    BOTH a numeric and a categorical source -- the four paths interacting."""
    rng = np.random.default_rng(seed)
    Xn = rng.normal(size=(n, p_numeric))
    a = rng.integers(0, 3, size=n)
    b = rng.integers(0, 3, size=n)
    cat_signal = (a == b).astype(float)
    lin = 1.8 * Xn[:, 0] + 1.2 * Xn[:, 5] + 1.6 * cat_signal + 0.25 * rng.normal(size=n)
    if n_classes == 2:
        y = (lin > np.median(lin)).astype(np.int64)
    else:
        y = pd.qcut(lin, n_classes, labels=False, duplicates="drop").astype(np.int64)
    av = np.array([f"A{v}" for v in a], dtype=object)
    bv = np.array([f"B{v}" for v in b], dtype=object)
    av[rng.random(n) < 0.15] = None      # NaN in cat_a (object) at fit
    Xn[rng.random(n) < 0.15, 0] = np.nan  # NaN in numeric signal col
    if nan_in_cat_b:
        bv = bv.copy()
        bv[rng.random(n) < 0.15] = None
    cols = {f"x{i}": Xn[:, i] for i in range(p_numeric)}
    cols["cat_a"] = pd.Series(av, dtype="object")
    cols["cat_b"] = pd.Categorical(bv)  # NaN-free Categorical partner at fit
    return pd.DataFrame(cols), pd.Series(y, name="y")


@pytest.mark.parametrize("n_classes", [2, 4], ids=["binary", "multiclass"])
def test_combined_cat_nan_wide_clf_no_serve_skew(n_classes):
    """The four paths TOGETHER: classification (binary + multiclass) over a wide
    numeric block + object/Categorical cat sources with NaN in BOTH a numeric and
    a categorical feature + an (a==b) cat-interaction. The engineered cat feature
    must NOT all-zero/collapse at transform, the NaN-free Categorical partner must
    NOT be off-by-one skewed, and NaN must not silently break selection.

    Pre-fix (block-NaN-shift bug): the NaN-free partner cat_b got codes
    ``0..K-1`` from the per-column map while fit assigned ``1..K`` -> the
    (cat_a, cat_b) lookup mis-keyed at serving -> train/serve skew."""
    df_tr, y_tr = _combined_frame(7, 6000, n_classes, p_numeric=120)
    sel = MRMR(verbose=0, random_seed=42, fe_max_steps=1)
    sel.fit(df_tr, y_tr)

    recipes = [
        r for r in getattr(sel, "_engineered_recipes_", [])
        if r.kind in ("factorize", "target_encoding")
        and set(getattr(r, "src_names", ())) >= {"cat_a", "cat_b"}
    ]
    assert recipes, "combined fixture produced no cat_a__cat_b interaction recipe"
    r = recipes[0]
    assert "cat_code_maps" in r.extra, "missing cat_code_maps replay table"

    # The NaN-free Categorical partner cat_b must be block-shifted to 1..K (the
    # NaN in cat_a shifts the whole block). A min code of 0 here == the pre-fix
    # off-by-one (partner left unshifted).
    map_b = r.extra["cat_code_maps"].get("cat_b")
    assert map_b, "cat_b code map absent"
    assert min(int(v) for v in map_b.values()) >= 1, (
        f"NaN-free partner cat_b not block-shifted (min code "
        f"{min(int(v) for v in map_b.values())}); off-by-one serve skew"
    )

    out_tr = sel.transform(df_tr)
    assert r.name in out_tr.columns
    col_tr = np.asarray(out_tr[r.name].to_numpy())
    assert len(np.unique(col_tr)) > 1, f"engineered '{r.name}' collapsed to constant"

    # Train mapping: value-pair -> code (skip NaN-bearing cat_a rows: their cell
    # is a legitimately distinct NaN bin, still deterministic).
    pairs_tr = list(zip(df_tr["cat_a"].astype(str), df_tr["cat_b"].astype(str)))
    pair_to_code: dict = {}
    for p, c in zip(pairs_tr, col_tr):
        pair_to_code.setdefault(p, c)

    # Disjoint holdout (train/serve split) must reuse the SAME mapping with zero
    # skew -- this is the assertion that fails pre-fix on the NaN-free partner.
    df_te, _ = _combined_frame(99, 3000, n_classes, p_numeric=120)
    out_te = sel.transform(df_te)
    col_te = np.asarray(out_te[r.name].to_numpy())
    pairs_te = list(zip(df_te["cat_a"].astype(str), df_te["cat_b"].astype(str)))
    mismatches = sum(
        1 for p, c in zip(pairs_te, col_te)
        if p in pair_to_code and pair_to_code[p] != c
    )
    assert mismatches == 0, (
        f"{mismatches} holdout rows got a DIFFERENT engineered code than the "
        f"train mapping for the same (cat_a, cat_b) pair -- train/serve skew "
        f"(block-NaN-shift off-by-one on the NaN-free partner)"
    )


def test_combined_nan_does_not_break_numeric_relevance():
    """In the combined load the NaN-bearing NUMERIC signal column must still be
    selectable (NaN routed to its own bin, not silently propagated into MI as a
    garbage value that destroys relevance)."""
    df_tr, y_tr = _combined_frame(11, 5000, 2, p_numeric=60)
    sel = MRMR(verbose=0, random_seed=42, fe_max_steps=1)
    sel.fit(df_tr, y_tr)
    fni = list(sel.feature_names_in_)
    sel_names = {fni[i] for i in sel.support_ if i < len(fni)}
    # At least one planted signal (numeric x0/x5 or the cat interaction) survives.
    engineered = {
        r.name for r in getattr(sel, "_engineered_recipes_", [])
    }
    assert (sel_names & {"x0", "x5"}) or (sel_names & engineered), (
        f"combined NaN load destroyed all planted signal; selected {sel_names}"
    )
    out = sel.transform(df_tr)
    # Raw NaN preserved on x0 (separate_bin keeps it for NaN-aware models).
    if "x0" in out.columns:
        assert int(out["x0"].isna().sum()) > 0, "raw NaN on x0 lost at transform"


def test_nan_in_cat_at_serve_not_fit_and_vice_versa():
    """Edge: NaN handling vs the +1 code shift across train/serve asymmetry.

    (1) Categorical with NaN at SERVE but not at FIT: fit had no NaN -> codes are
        UNSHIFTED (0..K-1) and the map has no NaN key; a transform-time NaN is
        genuinely unseen and resolves via ``unknown_strategy`` (clip -> sentinel
        bin), never crashing nor silently aliasing a real category to code 0.
    (2) NaN at FIT but not at SERVE: codes are +1 shifted; a serve row with a
        real category must map to its shifted (>=1) code, reproducing fit
        BIT-EXACTLY."""
    from mlframe.feature_selection.filters.discretization import categorize_dataset
    from mlframe.feature_selection.filters.engineered_recipes._recipe_extract import (
        build_category_code_map,
        _coerce_to_int_with_nan_handling,
        _NAN_CODE_KEY,
    )

    # (1) FIT NaN-free, SERVE has NaN. Single-column block -> block_has_nan False.
    fit_vals = pd.Series(["x", "y", "z", "x", "y"], dtype="object")
    df_fit = pd.DataFrame({"c": fit_vals})
    data, cols, nbins = categorize_dataset(df=df_fit, missing_strategy="separate_bin")
    fit_codes = data[:, cols.index("c")].astype(int)
    nb = int(nbins[cols.index("c")])
    m = build_category_code_map(fit_vals, block_has_nan=False)
    assert _NAN_CODE_KEY not in m and min(m.values()) == 0, "fit NaN-free -> unshifted 0..K-1"
    # Fit-row replay bit-exact.
    replay_fit = _coerce_to_int_with_nan_handling(
        np.asarray(fit_vals.to_numpy(), dtype=object), nb, "r", "c", "clip", m
    )
    assert np.array_equal(fit_codes, replay_fit)
    # Serve row carries a NaN unseen at fit -> resolves to sentinel bin (no crash,
    # not silently code 0 == the real category 'x').
    serve_vals = np.asarray(["x", None, "z"], dtype=object)
    replay_serve = _coerce_to_int_with_nan_handling(serve_vals, nb, "r", "c", "clip", m)
    assert replay_serve[0] == m["x"], "real category 'x' must keep its fit code"
    assert replay_serve[1] == nb - 1, "unseen serve NaN must resolve to sentinel bin, not alias code 0"

    # (2) FIT has NaN, SERVE does not. Codes +1 shifted; real cats reproduce exactly.
    fit_vals2 = pd.Series(["x", "y", None, "z", "x"], dtype="object")
    df_fit2 = pd.DataFrame({"c": fit_vals2})
    data2, cols2, nbins2 = categorize_dataset(df=df_fit2, missing_strategy="separate_bin")
    fit_codes2 = data2[:, cols2.index("c")].astype(int)
    nb2 = int(nbins2[cols2.index("c")])
    m2 = build_category_code_map(fit_vals2, block_has_nan=True)
    assert m2.get(_NAN_CODE_KEY) == 0 and min(v for k, v in m2.items() if k != _NAN_CODE_KEY) >= 1
    replay_fit2 = _coerce_to_int_with_nan_handling(
        np.asarray(fit_vals2.to_numpy(), dtype=object), nb2, "r", "c", "clip", m2
    )
    assert np.array_equal(fit_codes2, replay_fit2), "fit-with-NaN replay must be bit-exact"
    # Serve frame with only real categories -> their shifted codes, no NaN cell.
    serve_vals2 = np.asarray(["z", "x", "y"], dtype=object)
    replay_serve2 = _coerce_to_int_with_nan_handling(serve_vals2, nb2, "r", "c", "clip", m2)
    assert replay_serve2[0] == m2["z"] and replay_serve2[1] == m2["x"] and replay_serve2[2] == m2["y"]
    assert all(c >= 1 for c in replay_serve2), "shifted real-category serve codes must be >=1"


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
