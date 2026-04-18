"""Tests for mlframe.preprocessing.prepare_df_for_catboost.

Focus: dtype preservation when converting nullable/extension columns to
CatBoost-friendly numpy floats. Historical bug: everything was widened to
float64 via bare `astype(float)` (pandas) or `cast(pl.Float64)` (polars),
which silently cost memory/GPU bandwidth on users who had deliberately
chosen narrow precision.

Rules we enforce:
- Non-nullable numpy floats pass through unchanged.
- `pd.Float32Dtype`/`pd.Float64Dtype` → preserve precision (32→32, 64→64).
- `pd.Int8..Int32` / `pd.UInt8..UInt32` / `pd.BooleanDtype` → float32
  (values fit exactly, saves memory).
- `pd.Int64Dtype` / `pd.UInt64Dtype` → float64 (>~2**24 loses precision).
- Polars Float32/Float64 — untouched.
- Polars small ints with nulls → Float32; only Int64/UInt64 with nulls → Float64.
- Polars int columns WITHOUT nulls are not cast at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.preprocessing import prepare_df_for_catboost


# ---------------------------------------------------------------------------
# pandas dtype preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col_dtype, expected_out", [
    (np.float32,                   np.float32),
    (np.float64,                   np.float64),
    (pd.Float32Dtype(),            np.dtype("float32")),
    (pd.Float64Dtype(),            np.dtype("float64")),
    (pd.Int8Dtype(),               np.dtype("float32")),
    (pd.Int16Dtype(),              np.dtype("float32")),
    (pd.Int32Dtype(),              np.dtype("float32")),
    (pd.UInt8Dtype(),              np.dtype("float32")),
    (pd.UInt16Dtype(),             np.dtype("float32")),
    (pd.UInt32Dtype(),             np.dtype("float32")),
    (pd.Int64Dtype(),              np.dtype("float64")),
    (pd.UInt64Dtype(),             np.dtype("float64")),
    (pd.BooleanDtype(),            np.dtype("float32")),
])
def test_pandas_dtype_preserved_or_narrowed(col_dtype, expected_out):
    if isinstance(col_dtype, type) and issubclass(col_dtype, np.floating):
        # Non-nullable numpy floats — should pass through untouched.
        arr = np.array([1.0, 2.0, 3.0], dtype=col_dtype)
    elif isinstance(col_dtype, pd.BooleanDtype):
        arr = pd.array([True, False, None], dtype=col_dtype)
    else:
        arr = pd.array([1, 2, None], dtype=col_dtype)
    df = pd.DataFrame({"c": arr})
    out = prepare_df_for_catboost(df.copy(), cat_features=[])
    assert out.dtypes["c"] == expected_out, f"{col_dtype} → {out.dtypes['c']} (expected {expected_out})"


def test_pandas_float32_non_nullable_is_noop():
    """Sanity: bare numpy float32 must not be touched even if a bug regressed
    the extension-dtype branch.
    """
    df = pd.DataFrame({"f": np.array([1.0, 2.0, 3.0], dtype=np.float32)})
    out = prepare_df_for_catboost(df.copy(), cat_features=[])
    assert out.dtypes["f"] == np.float32


def test_pandas_nullable_float32_fills_na_but_keeps_precision():
    """End-to-end: pd.Float32Dtype column with a null must survive as
    numpy float32 with np.nan in place of the null.
    """
    df = pd.DataFrame({"f": pd.array([1.0, 2.0, None], dtype=pd.Float32Dtype())})
    out = prepare_df_for_catboost(df.copy(), cat_features=[])
    assert out.dtypes["f"] == np.float32
    # Null became np.nan (not pd.NA — CatBoost cannot handle the latter).
    assert np.isnan(out["f"].iloc[-1])


# ---------------------------------------------------------------------------
# polars dtype preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("src_dtype, values, expected_out", [
    (pl.Float32,   [1.0, 2.0, None],       pl.Float32),
    (pl.Float64,   [1.0, 2.0, None],       pl.Float64),
    (pl.Int8,      [1, 2, None],           pl.Float32),
    (pl.Int16,     [1, 2, None],           pl.Float32),
    (pl.Int32,     [1, 2, None],           pl.Float32),
    (pl.UInt8,     [1, 2, None],           pl.Float32),
    (pl.UInt16,    [1, 2, None],           pl.Float32),
    (pl.UInt32,    [1, 2, None],           pl.Float32),
    (pl.Int64,     [1, 2, None],           pl.Float64),
    (pl.UInt64,    [1, 2, None],           pl.Float64),
    (pl.Boolean,   [True, False, None],    pl.Float32),
    # No-null columns: left alone entirely (no unnecessary cast).
    (pl.Int32,     [1, 2, 3],              pl.Int32),
    (pl.Int64,     [1, 2, 3],              pl.Int64),
    (pl.Float32,   [1.0, 2.0, 3.0],        pl.Float32),
])
def test_polars_dtype_preserved_or_narrowed(src_dtype, values, expected_out):
    df = pl.DataFrame({"c": pl.Series("c", values, dtype=src_dtype)})
    out = prepare_df_for_catboost(df.clone(), cat_features=[])
    assert out.dtypes[0] == expected_out, f"{src_dtype} → {out.dtypes[0]} (expected {expected_out})"


def test_polars_no_nulls_skips_cast_entirely():
    """Micro-optimisation guard: when a non-float column has no nulls, the
    function shouldn't waste a cast. This test also catches an accidental
    regression that would cast every int column to Float*.
    """
    df = pl.DataFrame({
        "i32": pl.Series("i32", [1, 2, 3], dtype=pl.Int32),
        "i64": pl.Series("i64", [1, 2, 3], dtype=pl.Int64),
        "u16": pl.Series("u16", [1, 2, 3], dtype=pl.UInt16),
    })
    out = prepare_df_for_catboost(df.clone(), cat_features=[])
    assert out.dtypes == [pl.Int32, pl.Int64, pl.UInt16]


# ---------------------------------------------------------------------------
# Text-feature invariant (2026-04-19: prevents the production-hang scenario)
# ---------------------------------------------------------------------------
# Bug history: text columns auto-promoted from cat_features to text_features
# keep pd.Categorical dtype after polars->pandas conversion. The pandas path
# of prepare_df_for_catboost used to iterate ALL columns and, for any
# pd.Categorical one, run ``astype(str).fillna(...).astype("category")``
# and append it to cat_features. For skills_text (81k unique × 810k rows) a
# single such column took ~minutes, and the function also silently
# reclassified it as cat-feature. Invariant now enforced: columns that appear
# in the text_features argument are NEVER touched by the cat-prep loop
# — not added, not rebuilt, not timed.


def test_pandas_text_feature_categorical_not_added_to_cat_features():
    """A column declared as text_feature, even if its dtype is pd.Categorical,
    must NOT be added to cat_features by prepare_df_for_catboost."""
    df = pd.DataFrame({
        "skills_text": pd.Categorical(["a", "b", "a", "c"]),
        "true_cat":    pd.Categorical(["x", "y", "x", "y"]),
    })
    cat_features: list = []
    prepare_df_for_catboost(df, cat_features=cat_features, text_features=["skills_text"])
    assert "skills_text" not in cat_features, (
        "text-declared column must not be auto-added to cat_features"
    )
    # The companion 'true_cat' column IS pd.Categorical and NOT declared as
    # text, so it legitimately should be promoted.
    assert "true_cat" in cat_features


def test_pandas_text_feature_skips_expensive_astype_rebuild():
    """Perf-budget regression sensor.

    Before the fix a column declared as text_feature would still pass
    through the ``df[col].astype(str).fillna(...).astype("category")``
    rebuild inside the cat-prep loop. On a 50_000 × 5_000-unique-value
    column that dance dominates the function; the budget below is chosen
    so that the "not-skipped" path would blow through it by >5x but the
    correct (skipped) path finishes in a few hundred ms on a dev box.

    If this sensor ever fires, the most likely cause is a regression of
    the text-feature skip logic in ``prepare_df_for_catboost``.
    """
    import time

    rng = np.random.default_rng(42)
    n = 50_000
    pool = np.array([f"s_{i:05d}" for i in range(5_000)])
    df = pd.DataFrame({
        "skills_text": pd.Categorical(pool[rng.integers(0, len(pool), size=n)]),
        "num": rng.standard_normal(n).astype(np.float32),
    })
    # NOTE: no NaN injected. The function's text-feature branch calls
    # ``df[col].fillna("")`` which on pd.Categorical panics when "" is not a
    # category (pandas limitation: fillna on Categorical requires the fill
    # value to already be in categories). In production the caller is
    # expected to ``_decategorize_text_cols`` first; this unit test focuses
    # on the cat-loop skip invariant, not the NaN handling contract.

    cat_features: list = []
    t0 = time.perf_counter()
    prepare_df_for_catboost(df, cat_features=cat_features, text_features=["skills_text"])
    elapsed = time.perf_counter() - t0

    assert "skills_text" not in cat_features
    # Budget: comfortable for the skip path (<~0.3s on Python 3.11 + pandas
    # 2.x dev box), tight enough that the astype(str).astype("category")
    # rebuild over 5k categories would breach it easily.
    assert elapsed < 2.0, (
        f"prepare_df_for_catboost took {elapsed:.2f}s on a 50k text column — "
        "the text-feature skip likely regressed"
    )


def test_pandas_text_feature_dtype_is_not_mutated():
    """The skip must be a true no-op on dtype: a pd.Categorical text column
    exits with the same dtype it entered with (prepare_df_for_catboost is
    not responsible for text-column dtype conversion — that's handled by
    ``_decategorize_text_cols`` in trainer.py earlier in the fallback)."""
    df = pd.DataFrame({"skills_text": pd.Categorical(["a", "b", "a"])})
    src_dtype = df["skills_text"].dtype
    prepare_df_for_catboost(df, cat_features=[], text_features=["skills_text"])
    assert df["skills_text"].dtype == src_dtype


# ---------------------------------------------------------------------------
# Huge-dictionary Categorical NaN fill — production MemoryError 2026-04-19
# ---------------------------------------------------------------------------
# Production incident: CatBoost Polars fastpath rejected the data
# ("TypeError: No matching signature found" in
# _set_features_order_data_polars_categorical_column.process). Fallback
# converted polars→pandas. A Categorical column arrived with an untrimmed
# Polars global-string-pool dictionary: 3.3M unique categories, longest
# string 6133 chars. The NaN-fill path then ran
#   df[var] = df[var].astype(str).fillna(na_filler).astype("category")
# pandas' Categorical.astype(str) materialises ``categories._values`` as a
# fixed-width Unicode array: 3.3M × 6133 × 4B ≈ 75 GiB → MemoryError, the
# whole 2.5-minute pipeline died one step before fit. Fix operates on
# integer codes via ``.cat.add_categories + .fillna`` — no dict expansion.


def test_cat_nan_fill_does_not_materialize_dictionary_as_strings():
    """Functional sensor: NaN fill on a Categorical with a huge category
    dict (many categories NOT present in the row slice — simulates Polars'
    untrimmed global-string-pool behavior) must succeed without allocating
    a (n_categories × max_str_len) Unicode array.

    The test uses a Categorical whose dictionary carries 50_000 entries but
    the actual rows only reference 3 of them. Before the fix, the code
    expanded all 50_000 via ``.astype(str)`` regardless of row content. We
    assert both that the call succeeds (pre-fix would crash with MemoryError
    in prod; here we'd see a slow but finite allocation on a dev box) and
    that the NaN is filled.
    """
    # Build a Categorical with a huge dict but only a handful of unique
    # values actually present in the rows (simulates untrimmed Polars dict).
    huge_dict = [f"cat_{i:06d}" for i in range(50_000)]
    cat_type = pd.CategoricalDtype(categories=huge_dict, ordered=False)
    df = pd.DataFrame({
        "x": pd.Categorical(
            values=["cat_000001", "cat_000002", None, "cat_000001"],
            dtype=cat_type,
        ),
    })
    # Invariant pre-call: dict is huge, rows are sparse.
    assert len(df["x"].cat.categories) == 50_000
    assert df["x"].isna().any()

    cat_features: list = []
    out = prepare_df_for_catboost(df, cat_features=cat_features, text_features=[])

    # NaN was filled.
    assert not out["x"].isna().any()
    # Column was added to cat_features (it's a non-text Categorical).
    assert "x" in cat_features
    # na_filler joined the category set; pre-existing huge dict was NOT
    # thrown away (we don't trim automatically — that's caller responsibility).
    # Default na_filler is "" in prepare_df_for_catboost.
    assert "" in out["x"].cat.categories


def test_cat_nan_fill_preserves_existing_categories():
    """The fix must preserve the caller's existing category order — downstream
    CatBoost Pool indexing depends on stable codes across train/val/test."""
    cats = ["alpha", "beta", "gamma", "delta"]
    df = pd.DataFrame({
        "x": pd.Categorical(
            values=["alpha", None, "beta", "gamma"],
            categories=cats,
            ordered=False,
        ),
    })
    prepare_df_for_catboost(df, cat_features=["x"], text_features=[])
    # Original categories still present, in original order, plus na_filler.
    cats_after = list(df["x"].cat.categories)
    for c in cats:
        assert c in cats_after
    # The relative order of original cats is preserved.
    orig_indices = [cats_after.index(c) for c in cats]
    assert orig_indices == sorted(orig_indices)


def test_cat_nan_fill_idempotent_when_na_filler_already_a_category():
    """If na_filler (default "") is already in the category list, we must
    NOT try to re-add it (pandas raises ValueError on duplicate add)."""
    df = pd.DataFrame({
        "x": pd.Categorical(
            values=["a", None, "b", ""],
            categories=["a", "b", ""],  # "" already present
            ordered=False,
        ),
    })
    # Must not crash.
    prepare_df_for_catboost(df, cat_features=["x"], text_features=[])
    assert not df["x"].isna().any()


def test_cat_nan_fill_perf_budget_huge_untrimmed_dict():
    """Perf-budget sensor for the 2026-04-19 MemoryError incident.

    Simulates the production shape: 100_000-entry untrimmed dictionary,
    only 5 unique values actually in the rows, 10_000 rows, some NaN.
    The fixed code operates on int codes (O(n_rows) + O(1) dict growth);
    the buggy code allocated a (100k × max_str_len × 4B) string array
    plus copied it. Budget set so the fix passes easily and the bug
    would blow through by >10x even with small strings.
    """
    import time

    huge_dict = [f"cat_value_{i:010d}_with_moderately_long_suffix" for i in range(100_000)]
    n = 10_000
    rng = np.random.default_rng(42)
    values = rng.choice(huge_dict[:5], size=n).tolist()
    values[::100] = [None] * len(values[::100])  # sprinkle NaN
    df = pd.DataFrame({
        "x": pd.Categorical(
            values=values,
            categories=huge_dict,
            ordered=False,
        ),
    })

    t0 = time.perf_counter()
    prepare_df_for_catboost(df, cat_features=[], text_features=[])
    elapsed = time.perf_counter() - t0

    assert not df["x"].isna().any()
    assert elapsed < 2.0, (
        f"prepare_df_for_catboost took {elapsed:.2f}s on a 100k-entry "
        "untrimmed dict — regression of the 2026-04-19 MemoryError fix "
        "(likely someone restored the astype(str) path)."
    )


# ---------------------------------------------------------------------------
# Defensive None-guards (2026-04-19 proactive-exploration findings)
# ---------------------------------------------------------------------------


def test_text_features_none_does_not_crash():
    """``text_features=None`` must be accepted and treated as empty. Before
    the 2026-04-19 guard it crashed with
    ``TypeError: 'NoneType' object is not iterable`` on the
    ``for var in text_features`` loop. Callers (e.g. model paths that
    skipped text-feature auto-detection) passed None.
    """
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    out = prepare_df_for_catboost(df.copy(), cat_features=[], text_features=None)
    assert "a" in out.columns


def test_cat_features_none_does_not_crash():
    """Symmetric guard for ``cat_features=None``."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    out = prepare_df_for_catboost(df.copy(), cat_features=None, text_features=[])
    assert "a" in out.columns


def test_cat_features_both_none_does_not_crash():
    """Both at once — the most defensive call shape."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    out = prepare_df_for_catboost(df.copy(), cat_features=None, text_features=None)
    assert "a" in out.columns


# ---------------------------------------------------------------------------
# Documented behaviour: cat_features list is mutated in place (accumulated)
# ---------------------------------------------------------------------------


def test_cat_features_list_is_mutated_in_place_across_calls():
    """``prepare_df_for_catboost`` APPENDS detected categorical columns to
    the ``cat_features`` argument in place. If a caller reuses the same
    list across multiple calls the list accumulates state — a silent
    footgun. This test documents the behaviour so a future refactor that
    changes it (e.g. returns a fresh list) is visible as a test change.
    """
    shared: list = []
    df1 = pd.DataFrame({"a": pd.Categorical(["x", "y", "x"])})
    df2 = pd.DataFrame({"b": pd.Categorical(["p", "q", "p"])})
    prepare_df_for_catboost(df1.copy(), cat_features=shared, text_features=[])
    assert shared == ["a"]
    prepare_df_for_catboost(df2.copy(), cat_features=shared, text_features=[])
    # After the second call the list holds BOTH columns — the second caller
    # silently inherited 'a' from the first. Code that wants fresh state
    # must pass a new list each call.
    assert shared == ["a", "b"], (
        "cat_features list is mutated in place — if this assertion fails "
        "it means the API changed to return a fresh list; update callers "
        "that relied on mutation (core.py::_get_pipeline_components and "
        "the fit-params plumbing)."
    )
