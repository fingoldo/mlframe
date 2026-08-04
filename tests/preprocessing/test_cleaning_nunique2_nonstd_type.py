"""Regression test: analyse_and_clean_features's nunique==2 NaN-replacement branch must not
crash on a real value that is neither str, numeric, nor boolean (e.g. a decimal.Decimal or
pd.Timestamp held in an object/category column).

Pre-fix, the branch only assigned ``repl_value`` for isinstance(real_val, str), col_is_numeric,
or col_is_boolean, with no final else -- any other type left ``repl_value`` unbound, raising
UnboundLocalError at ``repl_instructions = {na_val: repl_value}``.
"""

from __future__ import annotations

import decimal

import numpy as np
import pandas as pd
import pytest


def test_nunique2_decimal_categorical_column_does_not_crash():
    """A 2-valued categorical column holding a Decimal + NaN must not raise UnboundLocalError."""
    pytest.importorskip("psutil")
    pytest.importorskip("pyutilz")
    from mlframe.preprocessing.cleaning import analyse_and_clean_features

    rng = np.random.default_rng(0)
    n = 300
    # A roughly balanced 2-value split (not a rare NaN sliver) so the rare-category merge step
    # (which would otherwise drop this column as "constant" before reaching the nunique==2
    # NaN-replacement branch) does not fire; this genuinely exercises the crashing branch.
    values = [decimal.Decimal("1.5")] * (n // 2) + [None] * (n - n // 2)
    df = pd.DataFrame(
        {
            "dec_col": pd.Categorical(values),
            "filler": rng.normal(size=n).astype(np.float32),
        }
    )
    result = analyse_and_clean_features(df, update_data=True, verbose=False)  # pre-fix: UnboundLocalError
    assert isinstance(result, dict)
    assert "dec_col" in df.columns
    assert not df["dec_col"].isna().any(), "NaN should have been replaced, not left in place"
    assert df["dec_col"].nunique() == 2
    assert decimal.Decimal("-1.5") in set(df["dec_col"].unique())


def test_nunique2_non_negatable_type_falls_back_to_string_sentinel():
    """A type without unary negation (e.g. a tuple) must fall back to a string sentinel, not crash."""
    pytest.importorskip("psutil")
    pytest.importorskip("pyutilz")
    from mlframe.preprocessing.cleaning import analyse_and_clean_features

    rng = np.random.default_rng(1)
    n = 300
    values = [("a", "b")] * (n // 2) + [None] * (n - n // 2)
    df = pd.DataFrame(
        {
            "tup_col": pd.Categorical(values),
            "filler": rng.normal(size=n).astype(np.float32),
        }
    )
    result = analyse_and_clean_features(df, update_data=True, verbose=False)  # pre-fix: UnboundLocalError
    assert isinstance(result, dict)
    assert "tup_col" in df.columns
    assert not df["tup_col"].isna().any()
    assert df["tup_col"].nunique() == 2
