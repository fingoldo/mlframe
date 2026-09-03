"""Regression tests for TC30 (splitter empty-val raise) and TC29
(polars<->pandas categorical category-list alignment on roundtrip).
"""

import pandas as pd
import polars as pl
import pytest

from mlframe.training.splitting import make_train_test_split
from mlframe.training.utils import get_pandas_view_of_polars_df
from mlframe.training._eval_helpers import _align_xgb_cat_categories

# ---------------------------------------------------------------------------
# TC30: sequential / aging split producing 0 val (or test) rows must RAISE
#       at the splitter (the source), not silently return an empty split.
# ---------------------------------------------------------------------------


def test_tc30_sequential_empty_val_raises():
    """val_size floored to 0 rows in the timestamp/sequential path must raise
    a ValueError naming the offending config (pre-fix: silent empty val)."""
    n = 20
    df = pd.DataFrame({"a": range(n)})
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="h"))
    with pytest.raises(ValueError, match="0 validation rows"):
        make_train_test_split(
            df,
            test_size=0.1,
            val_size=0.04,
            timestamps=ts,
            wholeday_splitting=False,
            shuffle_val=False,
            shuffle_test=False,
        )


def test_tc30_sequential_aging_empty_val_raises():
    """Same defect under trainset_aging_limit -- still the splitter's source."""
    n = 20
    df = pd.DataFrame({"a": range(n)})
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="h"))
    with pytest.raises(ValueError, match="0 validation rows"):
        make_train_test_split(
            df,
            test_size=0.1,
            val_size=0.04,
            timestamps=ts,
            wholeday_splitting=False,
            trainset_aging_limit=0.5,
            shuffle_val=False,
            shuffle_test=False,
        )


def test_tc30_empty_test_raises():
    """Tc30 empty test raises."""
    n = 20
    df = pd.DataFrame({"a": range(n)})
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="h"))
    with pytest.raises(ValueError, match="0 test rows"):
        make_train_test_split(
            df,
            test_size=0.02,
            val_size=0.0,
            timestamps=ts,
            wholeday_splitting=False,
            shuffle_val=False,
            shuffle_test=False,
        )


def test_tc30_healthy_split_does_not_raise():
    """Sanity: a normal split with adequate n is unaffected."""
    n = 200
    df = pd.DataFrame({"a": range(n)})
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="h"))
    tr, va, te, *_ = make_train_test_split(
        df,
        test_size=0.1,
        val_size=0.1,
        timestamps=ts,
        wholeday_splitting=False,
        shuffle_val=False,
        shuffle_test=False,
    )
    assert len(va) > 0 and len(te) > 0 and len(tr) > 0


# ---------------------------------------------------------------------------
# TC29: separate polars->pandas conversions diverge the Categorical category
#       lists; _align_xgb_cat_categories realigns to the train+val union so
#       a given string maps to the SAME integer code across train/val.
# ---------------------------------------------------------------------------


def test_tc29_roundtrip_diverges_without_alignment():
    """Document the underlying divergence the fix must repair."""
    train = pl.DataFrame({"k": ["a", "b", "c", "a"]}).with_columns(pl.col("k").cast(pl.Categorical))
    val = pl.DataFrame({"k": ["b", "c", "d", "b"]}).with_columns(pl.col("k").cast(pl.Categorical))
    pt = get_pandas_view_of_polars_df(train)
    pv = get_pandas_view_of_polars_df(val)
    assert isinstance(pt["k"].dtype, pd.CategoricalDtype)
    # Same string, different code -> the latent silent mis-encode.
    assert pt["k"].cat.categories.get_loc("b") != pv["k"].cat.categories.get_loc("b")


def test_tc29_alignment_makes_codes_agree():
    """After _align_xgb_cat_categories the train+val category lists are the
    union, so the same string has the same code in train and val."""
    train = get_pandas_view_of_polars_df(pl.DataFrame({"k": ["a", "b", "c", "a"]}).with_columns(pl.col("k").cast(pl.Categorical)))
    val = get_pandas_view_of_polars_df(pl.DataFrame({"k": ["b", "c", "d", "b"]}).with_columns(pl.col("k").cast(pl.Categorical)))
    train2, val2, _ = _align_xgb_cat_categories("CatBoostClassifier", train, val_df=val, test_df=None)
    # Category lists now identical (train+val union).
    assert list(train2["k"].cat.categories) == list(val2["k"].cat.categories)
    for s in ("b", "c"):
        assert train2["k"].cat.categories.get_loc(s) == val2["k"].cat.categories.get_loc(s)
    # train2 retains 'a', val2 union retains 'd' -- both present in the shared list.
    assert "a" in train2["k"].cat.categories and "d" in train2["k"].cat.categories


def test_tc29_test_categories_excluded_from_union():
    """TRAINING_LOOSE_A-2: a category present ONLY in test must NOT be folded into the train/val union --
    the union is deliberately train+val ONLY (leak-safe); a test-only category surfaces as NaN under
    pandas' Categorical semantics rather than being silently absorbed into the fit-time cat universe."""
    train = get_pandas_view_of_polars_df(pl.DataFrame({"k": ["a", "b", "c", "a"]}).with_columns(pl.col("k").cast(pl.Categorical)))
    val = get_pandas_view_of_polars_df(pl.DataFrame({"k": ["b", "c", "a", "b"]}).with_columns(pl.col("k").cast(pl.Categorical)))
    test = get_pandas_view_of_polars_df(pl.DataFrame({"k": ["z", "a", "z", "b"]}).with_columns(pl.col("k").cast(pl.Categorical)))

    train2, val2, test2 = _align_xgb_cat_categories("XGBClassifier", train, val_df=val, test_df=test)

    assert "z" not in train2["k"].cat.categories
    assert "z" not in val2["k"].cat.categories
    # 'z' was never fed back into train/val's fit-time universe -- must not leak into their category lists.
    assert set(train2["k"].cat.categories) == set(val2["k"].cat.categories) == {"a", "b", "c"}
    # test2 itself is untouched by the train/val union -- its own 'z' rows still carry their own category,
    # not silently absorbed into (or rejected against) the train+val universe.
    assert "z" in test2["k"].cat.categories


def test_get_pandas_view_of_polars_df_memo_no_stale_cross_call_category():
    """Regression: ``get_pandas_view_of_polars_df``'s single-entry memo (keyed on id()+shape+columns)
    once cached the WRONG object's id -- the Categorical->Enum remap rebinds its internal ``df`` local
    to a short-lived intermediate before the memo-populate code runs, so the cached key never matched
    a genuine same-frame repeat call, AND (worse) that intermediate's refcount hits zero and its address
    gets freed almost immediately once the function returns. CPython's small-object allocator commonly
    reuses a just-freed address for the very next similarly-sized allocation, so a LATER, unrelated call's
    brand-new small polars frame (same column name/shape -- realistic across many small test fixtures
    that all use a column named "k") can land at that exact freed address and false-hit the memo,
    silently returning a stale pandas view carrying a PRIOR call's categories. Exactly this flaked
    test_tc29_test_categories_excluded_from_union on CI with a 'd' category that belongs only to
    this file's earlier test_tc29_alignment_makes_codes_agree fixture. Simulates the same call pattern
    (a 'd'-bearing frame, then unrelated 'a'/'b'/'c'-only frames) across repeated GC cycles to catch a
    regression even when a given run doesn't happen to reuse the exact freed address."""
    import gc

    def _run_d_bearing_call():
        """Helper: convert a short-lived 'd'-bearing frame, mimicking an earlier, unrelated test."""
        get_pandas_view_of_polars_df(pl.DataFrame({"k": ["b", "c", "d", "b"]}).with_columns(pl.col("k").cast(pl.Categorical)))

    def _run_unrelated_call():
        """Helper: convert two fresh 'd'-free frames and return the union of their observed categories."""
        train = get_pandas_view_of_polars_df(pl.DataFrame({"k": ["a", "b", "c", "a"]}).with_columns(pl.col("k").cast(pl.Categorical)))
        val = get_pandas_view_of_polars_df(pl.DataFrame({"k": ["b", "c", "a", "b"]}).with_columns(pl.col("k").cast(pl.Categorical)))
        return set(train["k"].cat.categories) | set(val["k"].cat.categories)

    for _trial in range(20):
        _run_d_bearing_call()
        gc.collect()
        union = _run_unrelated_call()
        assert "d" not in union, f"trial {_trial}: memo leaked a prior call's 'd' category into an unrelated frame: {union}"
