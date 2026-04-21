"""API1: winsorize (outlier clipping) — polars_ds vs pandas vs sklearn RobustScaler."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import polars as pl
from polars_ds.pipeline import Blueprint

from _common import make_numeric_data, time_and_mem, save_result

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "bench_01_outliers.json")


def pds_winsorize(df: pl.DataFrame) -> pl.DataFrame:
    cols = [c for c in df.columns if c != "y"]
    bp = Blueprint(df, name="w").winsorize(cols=cols, q_low=0.01, q_high=0.99)
    return bp.materialize().transform(df)


def pandas_clip(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c != "y"]
    lo = df[cols].quantile(0.01)
    hi = df[cols].quantile(0.99)
    out = df.copy()
    out[cols] = df[cols].clip(lower=lo, upper=hi, axis=1)
    return out


def sklearn_robust(df: pd.DataFrame) -> np.ndarray:
    from sklearn.preprocessing import RobustScaler
    cols = [c for c in df.columns if c != "y"]
    return RobustScaler(quantile_range=(1, 99)).fit_transform(df[cols].values)


def main():
    out = {"api": "winsorize / outlier clipping", "rows": {}}
    for n in (10_000, 50_000, 200_000):
        pl_df = make_numeric_data(n=n, n_features=20, missing_rate=0.0)
        pd_df = pl_df.to_pandas()
        r_pds = time_and_mem(pds_winsorize, "polars_ds.winsorize", pl_df, repeats=3)
        r_pd = time_and_mem(pandas_clip, "pandas.clip", pd_df, repeats=3)
        r_sk = time_and_mem(sklearn_robust, "sklearn.RobustScaler", pd_df, repeats=3)
        out["rows"][str(n)] = {
            "polars_ds": r_pds.dict(),
            "pandas": r_pd.dict(),
            "sklearn_robust": r_sk.dict(),
            "speedup_vs_pandas": r_pd.seconds / r_pds.seconds,
            "speedup_vs_sklearn": r_sk.seconds / r_pds.seconds,
        }
        print(f"n={n:>7}: pds={r_pds.seconds:.3f}s  pandas={r_pd.seconds:.3f}s  sk={r_sk.seconds:.3f}s")
    save_result(RESULTS, out)
    print("saved:", RESULTS)


if __name__ == "__main__":
    main()
