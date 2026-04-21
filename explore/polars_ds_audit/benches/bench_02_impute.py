"""API2: impute median/linear — polars_ds vs sklearn SimpleImputer vs pandas."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import polars as pl
from polars_ds.pipeline import Blueprint

from _common import make_numeric_data, time_and_mem, save_result

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "bench_02_impute.json")


def pds_impute(df: pl.DataFrame) -> pl.DataFrame:
    cols = [c for c in df.columns if c != "y"]
    bp = Blueprint(df, name="i").impute(cols=cols, method="median")
    return bp.materialize().transform(df)


def pandas_fillna(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c != "y"]
    medians = df[cols].median()
    out = df.copy()
    out[cols] = df[cols].fillna(medians)
    return out


def sklearn_impute(df: pd.DataFrame):
    from sklearn.impute import SimpleImputer
    cols = [c for c in df.columns if c != "y"]
    return SimpleImputer(strategy="median").fit_transform(df[cols].values)


def main():
    out = {"api": "impute median", "rows": {}}
    for n in (10_000, 50_000, 200_000):
        pl_df = make_numeric_data(n=n, n_features=20, missing_rate=0.1)
        pd_df = pl_df.to_pandas()
        r_pds = time_and_mem(pds_impute, "polars_ds.impute", pl_df, repeats=3)
        r_pd = time_and_mem(pandas_fillna, "pandas.fillna", pd_df, repeats=3)
        r_sk = time_and_mem(sklearn_impute, "sklearn.SimpleImputer", pd_df, repeats=3)
        out["rows"][str(n)] = {
            "polars_ds": r_pds.dict(),
            "pandas": r_pd.dict(),
            "sklearn": r_sk.dict(),
            "speedup_vs_pandas": r_pd.seconds / r_pds.seconds,
            "speedup_vs_sklearn": r_sk.seconds / r_pds.seconds,
        }
        print(f"n={n:>7}: pds={r_pds.seconds:.3f}s  pandas={r_pd.seconds:.3f}s  sk={r_sk.seconds:.3f}s")
    save_result(RESULTS, out)
    print("saved:", RESULTS)


if __name__ == "__main__":
    main()
