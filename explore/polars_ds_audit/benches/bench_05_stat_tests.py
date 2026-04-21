"""API5: t-test / chi2 — polars_ds vs scipy + sklearn.feature_selection."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
import polars_ds as pds

from _common import make_numeric_data, time_and_mem, save_result

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "bench_05_stat_tests.json")


def pds_ttest(df: pl.DataFrame) -> list:
    feats = [c for c in df.columns if c != "y"]
    df2 = df.with_columns((pl.col("y") == 1).alias("_mask"))
    results = []
    for f in feats:
        res = df2.select(pds.ttest_ind(pl.col(f).filter(pl.col("_mask")),
                                        pl.col(f).filter(~pl.col("_mask")),
                                        equal_var=False).alias("t"))
        results.append(res)
    return results


def scipy_ttest(df: pl.DataFrame) -> list:
    from scipy import stats
    pdf = df.to_pandas()
    pos = pdf["y"] == 1
    return [stats.ttest_ind(pdf.loc[pos, f], pdf.loc[~pos, f], equal_var=False) for f in pdf.columns if f != "y"]


def sk_fclassif(df: pl.DataFrame):
    from sklearn.feature_selection import f_classif
    pdf = df.to_pandas()
    X = pdf.drop(columns=["y"]).fillna(0).values
    return f_classif(X, pdf["y"].values)


def main():
    out = {"api": "t-test / f_classif", "rows": {}}
    for n in (10_000, 50_000, 200_000):
        df = make_numeric_data(n=n, n_features=20, missing_rate=0.0)
        r_pds = time_and_mem(pds_ttest, "polars_ds.ttest_ind (per-feat loop)", df, repeats=3)
        r_sp = time_and_mem(scipy_ttest, "scipy.stats.ttest_ind", df, repeats=3)
        r_sk = time_and_mem(sk_fclassif, "sklearn.f_classif (vectorized)", df, repeats=3)
        out["rows"][str(n)] = {
            "polars_ds": r_pds.dict(),
            "scipy": r_sp.dict(),
            "sklearn": r_sk.dict(),
            "speedup_vs_scipy": r_sp.seconds / r_pds.seconds,
            "speedup_vs_sklearn": r_sk.seconds / r_pds.seconds,
        }
        print(f"n={n:>7}: pds={r_pds.seconds:.3f}s  scipy={r_sp.seconds:.3f}s  sk={r_sk.seconds:.3f}s")
    save_result(RESULTS, out)
    print("saved:", RESULTS)


if __name__ == "__main__":
    main()
