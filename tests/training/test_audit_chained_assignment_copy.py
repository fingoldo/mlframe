"""Wave-33 sensor: cleaning.py:93 unreliable _is_view gate -> unconditional copy.

The wave-33 audit found 1 residual P1 in a codebase that already swept
this bug class. The cleaning.py site previously gated a value-refresh
on an unreliable ``values.base`` probe:

Pre-fix:
    vals = sub_df[col].values if col in sub_df else None
    is_view = vals is not None and getattr(vals, "base", None) is not None
    if not is_view:
        sub_df[col] = df.loc[analyse_mask, col]

Problem:
- pandas <2 (copy-on-write OFF default): values.base could be None even
  for a view -> gate fired and SettingWithCopyWarning emitted.
- pandas >=2 (copy-on-write ON default in 2.1+): values always returns
  a fresh ndarray -> base is ALWAYS non-None -> gate INVERTS:
  the refresh NEVER fires -> sub_df[col] stays stale -> value_counts
  reflects pre-mutation state.

Post-fix: unconditional ``sub_df.copy()`` + ``sub_df[col] = df.loc[...]``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_update_sub_df_col_refreshes_from_df_loc_not_stale_subdf():
    """``_update_sub_df_col`` must value_count the FRESH ``df.loc[analyse_mask]``
    values, not a stale copy already living in ``sub_df``. Pre-fix the
    ``values.base`` view-heuristic inverted under pandas 2.x CoW and the refresh
    silently skipped, so value_counts reflected pre-mutation state."""
    from mlframe.preprocessing.cleaning import _update_sub_df_col

    df = pd.DataFrame({"c": ["a", "a", "b", "b", "b"]})
    # sub_df carries a STALE version of the column (all 'z').
    sub_df = pd.DataFrame({"c": ["z", "z", "z", "z", "z"]})
    mask = np.array([True, True, True, False, False])  # selects the 3 fresh rows

    counts, nunique = _update_sub_df_col(
        df, sub_df, "c", col_unique_values=None, nunique=0, analyse_mask=mask,
    )
    # Fresh df.loc[mask] -> ['a','a','b'] (unmasked rows become NaN via index
    # alignment); the stale 'z' must NOT appear at all.
    assert "z" not in counts.index, f"stale sub_df value leaked: {counts.to_dict()}"
    assert counts.get("a") == 2 and counts.get("b") == 1


def test_update_sub_df_col_does_not_mutate_caller_subdf():
    """The defensive copy must shield the caller's sub_df from the column write."""
    from mlframe.preprocessing.cleaning import _update_sub_df_col

    df = pd.DataFrame({"c": ["a", "b", "c"]})
    sub_df = pd.DataFrame({"c": ["x", "y", "z"]})
    mask = np.array([True, True, True])
    _update_sub_df_col(df, sub_df, "c", None, 0, analyse_mask=mask)
    # Caller's sub_df keeps its original values (copy-on-write semantics).
    assert list(sub_df["c"]) == ["x", "y", "z"]
