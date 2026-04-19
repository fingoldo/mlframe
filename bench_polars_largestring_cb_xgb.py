"""Standalone verification: is the CB/XGB Polars-fastpath failure really
about pa.large_string() vs pa.string()?

Approach: isolate the Arrow string variant as the ONLY variable.
Build the same data two ways — once with pa.string() dictionary values,
once with pa.large_string() dictionary values — and fit CatBoost / XGBoost
on each. If my hypothesis is correct, pa.string() succeeds and
pa.large_string() fails with the specific errors observed in production.
"""
from __future__ import annotations

import sys
import traceback
import numpy as np
import polars as pl
import pyarrow as pa
import pandas as pd

# Make output utf-8 safe on Windows cp1251 console.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def build_pa_table_with_string_type(string_type):
    """Build a pa.Table where one column is a Dictionary wrapping the
    specified string type (either pa.string() or pa.large_string()).

    Three test columns:
      - num: float64 (plain numeric)
      - cat: Dictionary<int32, {string|large_string}> — what Polars
        would emit for a pl.Categorical
      - s:   the raw string type (plain, not dictionary)
    """
    n = 100
    nums = pa.array(np.random.randn(n).astype(np.float64))
    cat_values = pa.array(["alpha", "beta", "gamma"], type=string_type)
    cat_indices = pa.array(np.random.randint(0, 3, size=n).astype(np.int32))
    cat_dict = pa.DictionaryArray.from_arrays(cat_indices, cat_values)
    s = pa.array(["alpha", "beta", "gamma"] * (n // 3) + ["alpha"], type=string_type)
    # Truncate to n rows (the list comprehension may overshoot).
    s = s.slice(0, n)
    return pa.table({"num": nums, "cat": cat_dict, "s": s})


def pa_table_to_polars(tbl):
    """pyarrow.Table → Polars DataFrame. Use pl.from_arrow which
    preserves the original Arrow type variant. If we use to_arrow()
    afterwards, we round-trip the type."""
    return pl.from_arrow(tbl)


def fit_cb(df_or_table, enable_text=False):
    """Try to fit a CatBoostClassifier. Return (ok_bool, error_repr)."""
    from catboost import CatBoostClassifier
    y = np.random.randint(0, 2, size=len(df_or_table)) if hasattr(df_or_table, "__len__") else np.random.randint(0, 2, size=df_or_table.num_rows)
    try:
        m = CatBoostClassifier(iterations=2, verbose=0, allow_writing_files=False, thread_count=1)
        if enable_text:
            m.fit(df_or_table, y, cat_features=["cat"], text_features=["s"])
        else:
            m.fit(df_or_table, y, cat_features=["cat"])
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"


def fit_xgb(df_or_table):
    """Try to fit an XGBClassifier. Return (ok_bool, error_repr)."""
    from xgboost import XGBClassifier
    n = df_or_table.height if hasattr(df_or_table, "height") else (
        df_or_table.num_rows if hasattr(df_or_table, "num_rows") else len(df_or_table)
    )
    y = np.random.randint(0, 2, size=n)
    try:
        m = XGBClassifier(
            enable_categorical=True, tree_method="hist", n_estimators=2,
            max_depth=2, verbosity=0,
        )
        m.fit(df_or_table, y)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"


def probe(case_name, df, cb_enable_text=False):
    """Fit CB and XGB, print results."""
    cb_ok, cb_err = fit_cb(df, enable_text=cb_enable_text)
    xgb_ok, xgb_err = fit_xgb(df)
    print(f"  [{case_name}]")
    print(f"    CB  fit: {'OK' if cb_ok else 'FAIL'} {cb_err or ''}")
    print(f"    XGB fit: {'OK' if xgb_ok else 'FAIL'} {xgb_err or ''}")


def main():
    np.random.seed(42)
    import catboost, xgboost
    print(f"catboost: {catboost.__version__}")
    print(f"xgboost:  {xgboost.__version__}")
    print(f"polars:   {pl.__version__}")
    print(f"pyarrow:  {pa.__version__}")
    print()

    # ======================================================================
    # Experiment 1: construct the SAME data two ways via pyarrow, then
    # convert to Polars DataFrames, and check which one the Polars
    # fastpath accepts.
    # ======================================================================
    print("=" * 70)
    print("EXPERIMENT 1: pa.string() vs pa.large_string() in a Polars DataFrame")
    print("=" * 70)

    tbl_small = build_pa_table_with_string_type(pa.string())
    tbl_large = build_pa_table_with_string_type(pa.large_string())

    print("\nTable built with pa.string() values:")
    for i, name in enumerate(tbl_small.column_names):
        print(f"  {name}: {tbl_small.column(i).type}")

    print("\nTable built with pa.large_string() values:")
    for i, name in enumerate(tbl_large.column_names):
        print(f"  {name}: {tbl_large.column(i).type}")

    # Convert both to Polars DataFrames — check if the Arrow type
    # survives the round-trip.
    df_small = pa_table_to_polars(tbl_small)
    df_large = pa_table_to_polars(tbl_large)

    print("\nPolars DF from pa.string() table; re-exported arrow types:")
    rt_small = df_small.to_arrow()
    for i, name in enumerate(rt_small.column_names):
        print(f"  {name}: {rt_small.column(i).type}")

    print("\nPolars DF from pa.large_string() table; re-exported arrow types:")
    rt_large = df_large.to_arrow()
    for i, name in enumerate(rt_large.column_names):
        print(f"  {name}: {rt_large.column(i).type}")

    # ======================================================================
    # Now the actual fit tests: CB and XGB on each variant.
    # ======================================================================
    print("\n" + "=" * 70)
    print("FIT TESTS on Polars DataFrames")
    print("=" * 70)

    # Drop text column 's' for XGB (XGB's Polars fastpath doesn't handle
    # plain-string columns even in the "small" case; our interest is the
    # Categorical Dictionary<*, string|large_string> dispatch).
    print("\n-- Without 's' (plain-string) column, just num + cat Dictionary --")
    print("Polars DF from pa.string():")
    probe("pl.DF, pa.string() inner", df_small.drop("s"))
    print("Polars DF from pa.large_string():")
    probe("pl.DF, pa.large_string() inner", df_large.drop("s"))

    print("\n-- Raw pyarrow.Table (no Polars wrapper) --")
    print("pa.Table from pa.string():")
    probe("pa.Table, pa.string() inner", tbl_small.drop(["s"]))
    print("pa.Table from pa.large_string():")
    probe("pa.Table, pa.large_string() inner", tbl_large.drop(["s"]))

    # ======================================================================
    # Experiment 2: what does a pl.Categorical natively produce?
    # ======================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: native pl.Categorical")
    print("=" * 70)
    native = pl.DataFrame({
        "num": np.random.randn(100),
        "cat": pl.Series("cat", np.random.choice(["a", "b", "c"], size=100)).cast(pl.Categorical),
    })
    native_arrow = native.to_arrow()
    print("Native pl.Categorical → arrow export:")
    for i, name in enumerate(native_arrow.column_names):
        print(f"  {name}: {native_arrow.column(i).type}")
    probe("native pl.Categorical DF", native)

    # ======================================================================
    # Experiment 3: pandas path as control (no Polars fastpath)
    # ======================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: pandas control (no Polars fastpath)")
    print("=" * 70)
    pd_df = pd.DataFrame({
        "num": np.random.randn(100),
        "cat": pd.Categorical(np.random.choice(["a", "b", "c"], size=100)),
    })
    print("pandas DataFrame columns:")
    for c in pd_df.columns:
        print(f"  {c}: {pd_df[c].dtype}")
    probe("pandas DataFrame", pd_df)


if __name__ == "__main__":
    main()
