"""Edge-case sensors for the unsupervised pre-screen (polars non-numeric / boundary / 1-row).

Complements ``test_pre_screen_unsupervised.py`` (which pins the pandas happy-path + the pandas
strict-``>`` null boundary) by covering the POLARS-side edges of
``mlframe.feature_selection.pre_screen.compute_unsupervised_drops``:

  (a) polars string (Utf8) and categorical columns are never dropped by the variance rule
      (variance is undefined on non-numeric -- only a constant NUMERIC column is dropped);
  (b) the polars null-fraction boundary is strict ``>`` -- a column whose null fraction equals
      the threshold is KEPT, one strictly above is dropped (the pandas half is already pinned);
  (c) the CURRENT 1-row-frame contract (pandas + polars): every numeric column is dropped as
      "constant" (single-value variance is None / NaN), strings are kept -- pinned as the
      documented contract so a future change to single-row handling is a conscious decision.

All frames are tiny (n<=100) so every test is sub-second; fixed seeds throughout. Polars is a
hard dep of the suite, but importorskip keeps the file collectable on a polars-less checkout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.pre_screen import compute_unsupervised_drops

pl = pytest.importorskip("polars")


# ---------------------------------------------------------------------------------------------
# (a) polars non-numeric columns survive; only the constant numeric column is dropped
# ---------------------------------------------------------------------------------------------


def test_polars_string_and_categorical_not_dropped():
    """Utf8 + Categorical columns must not be dropped by the variance rule (variance is undefined
    on non-numeric dtypes). Only the zero-variance NUMERIC column is dropped; a well-varied numeric
    column and both non-numeric columns survive."""
    df = pl.DataFrame(
        {
            "str_col": ["a", "b", "c", "d"],
            "cat_col": pl.Series(["x", "y", "x", "y"], dtype=pl.Categorical),
            "const_num": [5.0, 5.0, 5.0, 5.0],
            "good_num": [1.0, 2.0, 3.0, 4.0],
        }
    )
    drops = compute_unsupervised_drops(df)
    assert drops == ["const_num"], drops
    assert "str_col" not in drops
    assert "cat_col" not in drops
    assert "good_num" not in drops


def test_polars_constant_string_not_dropped_by_variance_rule():
    """A literally-constant STRING column is still not dropped by the variance rule -- the rule is
    numeric-only by design (constant strings are caught downstream by per-target FS, not here)."""
    df = pl.DataFrame(
        {
            "const_str": pl.Series(["same"] * 8, dtype=pl.Utf8),
            "const_cat": pl.Series(["c"] * 8, dtype=pl.Categorical),
            "good_num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    drops = compute_unsupervised_drops(df)
    assert drops == [], drops


# ---------------------------------------------------------------------------------------------
# (b) polars null-fraction boundary is strict ``>``
# ---------------------------------------------------------------------------------------------


def test_polars_null_fraction_boundary_strict_greater_string():
    """POLARS half of the strict-``>`` null boundary at the DEFAULT 0.99 threshold, isolated on a
    STRING column so the numeric variance rule cannot confound the result.

    n=100: 99 nulls -> fraction exactly 0.99 -> NOT strictly greater -> KEPT.
           100 nulls -> fraction 1.0 -> strictly greater -> DROPPED.
    """
    n = 100
    kept = pl.DataFrame(
        {
            "s": pl.Series(["v"] + [None] * (n - 1), dtype=pl.Utf8),  # 99 nulls
            "pad": list(range(n)),
        }
    )
    dropped = pl.DataFrame(
        {
            "s": pl.Series([None] * n, dtype=pl.Utf8),  # 100 nulls
            "pad": list(range(n)),
        }
    )
    assert "s" not in compute_unsupervised_drops(kept)
    assert "s" in compute_unsupervised_drops(dropped)


def test_polars_null_fraction_boundary_strict_greater_numeric():
    """POLARS strict-``>`` null boundary on a NUMERIC column, isolated from the variance rule by
    using a lowered threshold (0.49) and a var-safe non-null tail (51 / 50 distinct values).

    n=100, threshold 0.49: 49 nulls -> fraction 0.49 -> NOT strictly greater -> KEPT.
                           50 nulls -> fraction 0.50 -> strictly greater -> DROPPED.
    """
    n = 100
    kept_vals = [None] * 49 + list(np.linspace(0.0, 1.0, n - 49))
    dropped_vals = [None] * 50 + list(np.linspace(0.0, 1.0, n - 50))
    kept = pl.DataFrame({"a": pl.Series(kept_vals, dtype=pl.Float64), "pad": list(range(n))})
    dropped = pl.DataFrame({"a": pl.Series(dropped_vals, dtype=pl.Float64), "pad": list(range(n))})
    assert "a" not in compute_unsupervised_drops(kept, null_fraction_threshold=0.49)
    assert "a" in compute_unsupervised_drops(dropped, null_fraction_threshold=0.49)


def test_polars_numeric_99_null_dropped_via_variance_not_null_rule():
    """PINNED CONTRACT (subtle): a NUMERIC column with 99 nulls + 1 value at the default 0.99
    threshold IS dropped -- but via the VARIANCE rule, not the null rule. The null fraction is
    exactly 0.99 (not strictly greater, so the null rule does not fire), yet the single surviving
    non-null value yields ``var()==None`` (ddof=1 on n=1), which the pre-screen treats as constant.

    This is the documented contract: with 99/100 nulls a numeric column cannot have a defined
    variance, so it is correctly removed as no-information. A reader must NOT mistake this drop for
    a null-rule drop -- the null rule's strict-``>`` boundary is exercised on the string column in
    ``test_polars_null_fraction_boundary_strict_greater_string`` instead.
    """
    n = 100
    df = pl.DataFrame(
        {
            "a": pl.Series([1.0] + [None] * (n - 1), dtype=pl.Float64),  # 99 nulls, 1 value
            "pad": list(range(n)),
        }
    )
    assert "a" in compute_unsupervised_drops(df)


# ---------------------------------------------------------------------------------------------
# (c) single-row frame contract: numeric dropped as constant, strings kept (pandas + polars)
# ---------------------------------------------------------------------------------------------


def test_single_row_frame_drops_all_numeric_documented():
    """PINNED CONTRACT for the 1-row frame on both backends.

    A single-row frame has ``var()`` undefined for every numeric column (polars returns None,
    pandas returns NaN under ddof=1), and the pre-screen treats undefined variance as "constant"
    -> drops every numeric column. Non-numeric (string / categorical) columns are kept because the
    variance rule never applies to them and a single value is below the null threshold.

    This is the CURRENT, intentional behavior -- pinned so any future change to single-row handling
    (e.g. keeping numeric columns when n==1) is a conscious decision that updates this test.
    """
    # pandas: numeric dropped, string kept
    pdf = pd.DataFrame({"num_a": [3.0], "num_b": [7], "str_col": ["hello"]})
    pdrops = compute_unsupervised_drops(pdf)
    assert "num_a" in pdrops
    assert "num_b" in pdrops
    assert "str_col" not in pdrops

    # polars: numeric dropped, string + categorical kept
    pldf = pl.DataFrame(
        {
            "num_a": [3.0],
            "num_b": [7],
            "str_col": ["hello"],
            "cat_col": pl.Series(["k"], dtype=pl.Categorical),
        }
    )
    pldrops = compute_unsupervised_drops(pldf)
    assert "num_a" in pldrops
    assert "num_b" in pldrops
    assert "str_col" not in pldrops
    assert "cat_col" not in pldrops


def test_single_row_protected_numeric_survives():
    """Even on a 1-row frame, a protected numeric column is never dropped (protection outranks the
    constant-variance rule)."""
    pdf = pd.DataFrame({"num_a": [3.0], "target": [1]})
    drops = compute_unsupervised_drops(pdf, protected_columns={"target"})
    assert "num_a" in drops
    assert "target" not in drops
