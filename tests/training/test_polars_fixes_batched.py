"""Regression sensor for S44: ``apply_polars_categorical_fixes`` must batch per-cat-column
``unique()`` collects into a single ``.collect()`` per frame (train + val), not one per column.

The old code called ``train_df_polars.select(pl.col(col).drop_nulls().unique())`` inside a loop
over ``cat_features``, paying 2*N sync collects for N cat columns. The fix batches these into
one collect per frame via ``.lazy().select([...]).collect()`` with ``implode()``.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import polars as pl

from mlframe.training.core._phase_polars_fixes import apply_polars_categorical_fixes


def _make_frame(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(8):
        cats = [f"c{i}_{k}" for k in range(20)]
        data[f"cat_{i}"] = pl.Series(rng.choice(cats, size=n).tolist(), dtype=pl.Utf8)
    return pl.DataFrame(data)


def test_S44_apply_polars_categorical_fixes_batches_unique_collects():
    """``apply_polars_categorical_fixes`` must NOT call ``DataFrame.select`` per cat column for
    the union compute step. A single batched ``LazyFrame.collect()`` covers all cat cols.
    """
    train = _make_frame(2000, 1)
    val = _make_frame(500, 2)
    test = _make_frame(500, 3)
    cat_features = list(train.columns)

    # Count select() calls on train_df_polars. The per-col path issues two selects per cat col
    # (one for train, one for val); the batched path issues none (uses lazy().select(...).collect()
    # instead). N cat columns => >= 2*N selects on the unbatched code.
    select_count = {"train": 0, "val": 0}
    orig_train_select = pl.DataFrame.select

    def _counting_select(self, *args, **kwargs):
        # Identify whether the caller hits train, val, or test by object identity below.
        if self is train:
            select_count["train"] += 1
        elif self is val:
            select_count["val"] += 1
        return orig_train_select(self, *args, **kwargs)

    with patch.object(pl.DataFrame, "select", _counting_select):
        res = apply_polars_categorical_fixes(
            train_df_polars=train,
            val_df_polars=val,
            test_df_polars=test,
            train_df_pd=train,
            val_df_pd=val,
            test_df_pd=test,
            filtered_train_df=train,
            filtered_val_df=val,
            cat_features=cat_features,
            align_polars_categorical_dicts=True,
            defer_pandas_conv=True,
            was_polars_input=True,
            verbose=False,
        )

    # Output schema sanity check: each cat col cast to Enum.
    out_train = res.train_df_polars
    for c in cat_features:
        dt = out_train.schema[c]
        assert hasattr(pl, "Enum") and isinstance(dt, pl.Enum), f"{c} should be cast to pl.Enum after alignment; got {dt}."

    # Per-col select-on-DataFrame must be 0 for both train and val (batched path uses .lazy()).
    # The two original per-col selects per cat col would have produced >= 2*N selects each. We
    # allow a small budget for unrelated select() invocations elsewhere in the function (e.g.
    # the Step 4 _cast_utf8_cats_to_categorical helper does NOT call DataFrame.select), but
    # the union-compute step must not loop selects over cat cols.
    assert select_count["train"] <= 2, (
        f"train_df_polars.select() called {select_count['train']} times; the batched union path "
        f"must avoid per-col DataFrame.select() (use .lazy().select(...).collect() instead)."
    )
    assert select_count["val"] <= 2, (
        f"val_df_polars.select() called {select_count['val']} times; the batched union path "
        f"must avoid per-col DataFrame.select() (use .lazy().select(...).collect() instead)."
    )


def test_S44_output_semantically_equivalent_to_per_col_path():
    """Behavioural correctness: the batched path must produce the same per-col Enum domains as the
    legacy per-col path. We verify by comparing the per-col category set after alignment.
    """
    train = _make_frame(1500, 5)
    val = _make_frame(400, 6)

    cat_features = list(train.columns)
    res = apply_polars_categorical_fixes(
        train_df_polars=train,
        val_df_polars=val,
        test_df_polars=None,
        train_df_pd=train,
        val_df_pd=val,
        test_df_pd=None,
        filtered_train_df=train,
        filtered_val_df=val,
        cat_features=cat_features,
        align_polars_categorical_dicts=True,
        defer_pandas_conv=True,
        was_polars_input=True,
        verbose=False,
    )
    out_train = res.train_df_polars
    out_val = res.val_df_polars

    # Per col: the Enum domain MUST be the union of train.unique() + val.unique().
    for c in cat_features:
        expected_union = set(train[c].drop_nulls().unique().to_list()) | set(val[c].drop_nulls().unique().to_list())
        got_train_domain = set(out_train.schema[c].categories)
        got_val_domain = set(out_val.schema[c].categories)
        assert got_train_domain == expected_union, f"train col {c}: expected Enum domain {expected_union}, got {got_train_domain}"
        assert got_val_domain == expected_union, f"val col {c}: expected Enum domain {expected_union}, got {got_val_domain}"
