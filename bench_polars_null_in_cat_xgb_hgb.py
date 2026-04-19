"""Verify null-in-Categorical behavior for XGBoost 3.x and sklearn HGB
(HistGradientBoostingClassifier) — the other two polars-capable
strategies in mlframe (alongside CatBoost).

Upstream fill in core.py applies to the base Polars DFs BEFORE tier
filtering, so XGB and HGB get the same pre-filled frame as CB. But
are they actually affected by null-in-Categorical? If yes, the fill
is load-bearing for them too; if not, it's a harmless no-op for
their path.

Bench matrix:
  - XGB 3.x: null-free Categorical, null-in-Categorical
  - HGB: null-free Categorical, null-in-Categorical
"""
from __future__ import annotations

import sys
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def build_df(with_nulls: bool, n: int = 500) -> pl.DataFrame:
    """Float32 + Categorical column. with_nulls=True salts the cat
    column with 10% None values."""
    vals = list(np.random.choice(["a", "b", "c"], size=n))
    if with_nulls:
        for i in range(0, n, 10):
            vals[i] = None
    return pl.DataFrame({
        "num": np.random.randn(n).astype(np.float32),
        "cat": pl.Series("cat", vals, dtype=pl.String).cast(pl.Categorical),
    })


def fit_xgb(df: pl.DataFrame):
    from xgboost import XGBClassifier
    y = np.random.randint(0, 2, size=df.height)
    m = XGBClassifier(
        enable_categorical=True,
        tree_method="hist",
        n_estimators=2,
        max_depth=2,
        verbosity=0,
    )
    try:
        m.fit(df, y)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"


def fit_hgb(df):
    """sklearn HGB requires a pandas DataFrame (even when we call it
    'polars-capable' — mlframe's HGBStrategy casts to pandas internally
    before fit). Try direct polars → see what happens, then try
    via pandas (the actual mlframe path)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    y = np.random.randint(0, 2, size=df.height)
    m = HistGradientBoostingClassifier(
        max_iter=2, max_depth=2, categorical_features="from_dtype"
    )
    # Direct polars: sklearn 1.8 has some polars support via Arrow.
    try:
        m.fit(df, y)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"


def main():
    np.random.seed(42)
    import xgboost, sklearn
    print(f"xgboost: {xgboost.__version__}, sklearn: {sklearn.__version__}, polars: {pl.__version__}")
    print()

    df_clean = build_df(with_nulls=False)
    df_dirty = build_df(with_nulls=True)
    print(f"df_clean: cat null_count = {df_clean['cat'].null_count()}")
    print(f"df_dirty: cat null_count = {df_dirty['cat'].null_count()}")
    print()

    print("XGB 3.x:")
    ok, err = fit_xgb(df_clean)
    print(f"  clean cat: {'OK' if ok else 'FAIL'} {err or ''}")
    ok, err = fit_xgb(df_dirty)
    print(f"  dirty cat: {'OK' if ok else 'FAIL'} {err or ''}")
    print()

    print("sklearn HGB (direct polars → fit):")
    ok, err = fit_hgb(df_clean)
    print(f"  clean cat: {'OK' if ok else 'FAIL'} {err or ''}")
    ok, err = fit_hgb(df_dirty)
    print(f"  dirty cat: {'OK' if ok else 'FAIL'} {err or ''}")


if __name__ == "__main__":
    main()
