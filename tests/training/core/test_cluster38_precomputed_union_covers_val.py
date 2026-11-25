"""Cluster 38 regression: a pre-OD category union also covers validation categories (audit 2026-11-20, finding 38-F22).

Before the fix, a val category missing from the supplied union made the strict val cast fail outside the per-column try. Val categories now
join the union, so the cast is total and the val values stay non-null.
"""

import polars as pl

from mlframe.training.core._phase_polars_fixes import _apply_polars_categ_t_step4_col_eligible_cols


def test_val_category_absent_from_precomputed_union_is_kept():
    """A val-only category 'z' is added to the Enum and survives as a non-null value (38-F22)."""
    train = pl.DataFrame({"c": ["a", "b"]})
    val = pl.DataFrame({"c": ["a", "z"]})
    train_out, val_out = _apply_polars_categ_t_step4_col_eligible_cols(
        ["c"],
        {"c": ["a", "b"]},
        None,
        train,
        val,
        None,
        set(),
        [],
        {},
        [],
        [],
        None,
        [],
        [],
    )
    assert val_out is not None
    assert val_out["c"].null_count() == 0
    assert set(val_out["c"].to_list()) == {"a", "z"}
    assert train_out["c"].null_count() == 0
