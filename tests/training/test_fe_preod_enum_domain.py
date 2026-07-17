"""Regression: ``apply_polars_categorical_fixes`` builds the Enum domain
from POST-outlier-detection train+val. If outlier detection (OD) filters
out the only row in train carrying category 'rare_X', the level vanishes
from the Enum domain - and val rows with 'rare_X' silently cast to null.

Fix (option b in the task): the caller threads a ``precomputed_category_union``
mapping (computed at frame-load time, BEFORE OD) into
``apply_polars_categorical_fixes``. If provided, it wins over the
post-OD train+val recomputation.
"""

from __future__ import annotations

import polars as pl


def test_precomputed_union_keeps_rare_val_category_alive():
    """OD-style filter drops the only train row of 'rare_X' before fix is
    called; precomputed_category_union (built pre-OD) must keep 'rare_X'
    in the Enum so val's 'rare_X' rows don't go null."""
    from mlframe.training.core._phase_polars_fixes import apply_polars_categorical_fixes

    # Original train (pre-OD): has 'rare_X' once. Post-OD train: rare_X dropped.
    train_pre = pl.DataFrame({"cat": ["A", "B", "rare_X", "A"]}).with_columns(
        pl.col("cat").cast(pl.Categorical),
    )
    train_post = train_pre.head(2).vstack(train_pre.slice(3, 1))  # drops the rare row
    val = pl.DataFrame({"cat": ["A", "rare_X", "B"]}).with_columns(
        pl.col("cat").cast(pl.Categorical),
    )

    # Precomputed union (pre-OD) - this is what fix (b) threads in.
    precomputed = {"cat": ["A", "B", "rare_X"]}

    _result = apply_polars_categorical_fixes(
        train_df_polars=train_post,
        val_df_polars=val,
        test_df_polars=None,
        train_df_pd=train_post,
        val_df_pd=val,
        test_df_pd=None,
        filtered_train_df=train_post,
        filtered_val_df=val,
        cat_features=["cat"],
        align_polars_categorical_dicts=True,
        defer_pandas_conv=False,
        was_polars_input=True,
        verbose=False,
        precomputed_category_union=precomputed,
    )
    train_out, val_out = _result.train_df_polars, _result.val_df_polars
    # val's 'rare_X' must NOT become null after Enum cast.
    val_cat_col = val_out["cat"]
    null_count = int(val_cat_col.is_null().sum())
    assert null_count == 0, (
        f"val rows lost {null_count} cell(s) to null - 'rare_X' was dropped from the Enum domain. Pre-OD union not honoured. val_out: {val_out.to_dict()}"
    )
    # Confirm the Enum domain includes rare_X.
    dt = train_out.schema["cat"]
    assert "rare_X" in list(dt.categories), f"Enum domain {list(dt.categories)} is missing rare_X"
