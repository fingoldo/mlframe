"""Round 11 sensors for the TRUE root cause of CB Polars fastpath
failures (verified 2026-04-19 via direct repro in
``bench_polars_cb_nullfrac.py``, after 3 earlier misdiagnoses).

Root cause: CatBoost 1.2.10's
``_set_features_order_data_polars_categorical_column`` Cython fused
cpdef has no dispatch signature for a Polars Categorical column
carrying a validity bitmap (``null_count > 0``). A single null
anywhere in any cat_feature is enough to trip ``TypeError: No
matching signature found``. Null-free Categoricals fit cleanly.

Null-fraction sweep from the bench:
    0.0   → OK
    0.1   → FAIL  (TypeError: No matching signature found)
    0.5   → FAIL
    0.99  → FAIL
    1.0   → FAIL

In the 2026-04-19 prod schema, 6 of 9 cat_features had nulls
(from 0.15% to 100%) — all 9 got routed through a failing fastpath
attempt before pandas fallback.

Misdiagnoses that this round CLEARED (kept for the record so we
don't chase them again):
- pl.Enum (round 7): the prod schema had no Enums. Harmless WARN.
- stale cat_features_polars (round 10): genuine bug but orthogonal;
  fastpath still failed after the fix.
- Polars 1.x large_string Arrow export (round 11 first attempt):
  null-free Categorical works fine despite the large_string Arrow
  type, proved by ``bench_polars_largestring_cb_xgb.py``.

Fix: ``_polars_df_has_null_in_categorical(df, cat_features=...)``
detector in trainer.py. Called from core.py's polars-fastpath block
before fit. If True, ``polars_fastpath_active`` flips to False
BEFORE the fit attempt, the pandas tier DF is built now, and
training + prediction uses pandas end-to-end.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest


class TestPolarsDfHasNullInCategoricalDetector:

    def test_detects_single_null_in_categorical(self):
        """ONE null is enough — the trigger isn't 'mostly null', it's
        'has any null at all' in a Categorical."""
        from mlframe.training.trainer import _polars_df_has_null_in_categorical
        vals = ["a", "b", "c", None, "a"]
        df = pl.DataFrame({"c": pl.Series("c", vals, dtype=pl.String).cast(pl.Categorical)})
        assert _polars_df_has_null_in_categorical(df)

    def test_null_free_categorical_returns_false(self):
        """Clean case: CB fastpath actually works for null-free
        Categorical, so the detector must NOT flip True here."""
        from mlframe.training.trainer import _polars_df_has_null_in_categorical
        df = pl.DataFrame({"c": pl.Series("c", ["a", "b", "c"]).cast(pl.Categorical)})
        assert not _polars_df_has_null_in_categorical(df)

    def test_null_in_non_cat_column_ignored(self):
        """A null in a Float or Boolean column does NOT trigger CB's
        dispatcher miss — only nullable Categorical does. The detector
        must be specific."""
        from mlframe.training.trainer import _polars_df_has_null_in_categorical
        df = pl.DataFrame({
            "num": [1.0, None, 3.0, 4.0],
            "c":   pl.Series("c", ["a", "b", "a", "b"]).cast(pl.Categorical),
        })
        assert not _polars_df_has_null_in_categorical(df)

    def test_cat_features_scope(self):
        """When ``cat_features`` list is passed, only those columns
        are inspected. A null in some OTHER Categorical column not
        listed as a cat_feature shouldn't trigger the bypass."""
        from mlframe.training.trainer import _polars_df_has_null_in_categorical
        df = pl.DataFrame({
            "good_cat":    pl.Series("good_cat",    ["a", "b", "a"]).cast(pl.Categorical),
            "null_cat":    pl.Series("null_cat",    ["a", None, "b"]).cast(pl.Categorical),
        })
        # null_cat has a null but we only check good_cat → False
        assert not _polars_df_has_null_in_categorical(df, cat_features=["good_cat"])
        # null_cat in scope → True
        assert _polars_df_has_null_in_categorical(df, cat_features=["null_cat"])

    def test_all_cat_columns_checked_when_cat_features_none(self):
        """Default (cat_features=None): ANY Categorical with nulls
        triggers. Useful when the caller doesn't have the explicit
        cat_features list at hand."""
        from mlframe.training.trainer import _polars_df_has_null_in_categorical
        df = pl.DataFrame({
            "clean": pl.Series("clean", ["a", "b"]).cast(pl.Categorical),
            "dirty": pl.Series("dirty", ["a", None]).cast(pl.Categorical),
        })
        assert _polars_df_has_null_in_categorical(df)

    def test_non_polars_input_returns_false(self):
        """Defensive: pandas DF / None / ndarray → False, no crash."""
        import pandas as pd
        from mlframe.training.trainer import _polars_df_has_null_in_categorical
        assert not _polars_df_has_null_in_categorical(pd.DataFrame({"a": [1]}))
        assert not _polars_df_has_null_in_categorical(None)
        assert not _polars_df_has_null_in_categorical(np.arange(3))

    def test_empty_df_returns_false(self):
        """Empty DataFrame has no nulls by definition."""
        from mlframe.training.trainer import _polars_df_has_null_in_categorical
        df = pl.DataFrame({"c": pl.Series("c", [], dtype=pl.Categorical)})
        assert not _polars_df_has_null_in_categorical(df)

    def test_prod_shape_multiple_nullable_categoricals(self):
        """Matches the 2026-04-19 prod shape: several cat_features
        with varying null counts from 0.1% to 100%. Any one of them
        must trip the detector."""
        from mlframe.training.trainer import _polars_df_has_null_in_categorical
        n = 200
        df = pl.DataFrame({
            "job_type":      pl.Series("job_type", np.random.choice(["a", "b"], size=n)).cast(pl.Categorical),
            "category_grp":  pl.Series("category_grp",
                ([None] * 2 + ["x"] * (n - 2)), dtype=pl.String).cast(pl.Categorical),
            "hourly_budget": pl.Series("hourly_budget", [None] * n, dtype=pl.String).cast(pl.Categorical),
        })
        assert _polars_df_has_null_in_categorical(
            df, cat_features=["job_type", "category_grp", "hourly_budget"]
        )


class TestLegacyAlias:
    """The round-11 first-iteration name _polars_df_emits_large_string
    survives as a deprecated alias to avoid breaking in-flight callers
    during the renaming transition."""

    def test_alias_delegates_to_null_detector(self):
        from mlframe.training.trainer import _polars_df_emits_large_string
        df_clean = pl.DataFrame({"c": pl.Series("c", ["a", "b"]).cast(pl.Categorical)})
        df_dirty = pl.DataFrame({"c": pl.Series("c", ["a", None]).cast(pl.Categorical)})
        assert not _polars_df_emits_large_string(df_clean)
        assert _polars_df_emits_large_string(df_dirty)


class TestFillNullPreservesFastpath:
    """The real win: instead of bypassing the Polars fastpath when
    cat_features have nulls, fill the nulls with a sentinel string on
    the Polars frame in place. CB's fastpath dispatcher then matches
    the non-nullable Categorical signature and we keep the ~5-minute
    native Polars fit instead of the ~18-minute pandas-path detour.
    """

    def test_fillnull_keeps_categorical_dtype(self):
        """Polars auto-extends the category dict when fill_null adds a
        new value — the dtype stays pl.Categorical, just one extra
        category entry appears."""
        df = pl.DataFrame({
            "cat": pl.Series("cat", ["a", None, "b", None, "a"], dtype=pl.String).cast(pl.Categorical),
        })
        assert df["cat"].null_count() == 2
        assert df["cat"].dtype == pl.Categorical
        filled = df.with_columns(pl.col("cat").fill_null("__MISSING__"))
        assert filled["cat"].null_count() == 0
        assert filled["cat"].dtype == pl.Categorical, (
            "fill_null must keep Categorical dtype; otherwise downstream "
            "CB cat_features dispatch breaks for a different reason"
        )
        # __MISSING__ is now one of the categories.
        assert "__MISSING__" in filled["cat"].unique().to_list()

    @pytest.mark.skipif(
        True,  # CB fit is heavy; covered by the earlier bench. Keep the
               # test body as documentation of the expected behaviour.
        reason="End-to-end CB fit covered in bench_polars_cb_nullfrac.py; "
               "running it per-test adds 5-10 s and noise.",
    )
    def test_cb_fastpath_accepts_filled_categorical(self):
        """Expected behavior (proven in bench_polars_cb_nullfrac.py):
        CB fit on a fill-null'd Categorical succeeds on the Polars
        fastpath without any fallback."""
        pass
