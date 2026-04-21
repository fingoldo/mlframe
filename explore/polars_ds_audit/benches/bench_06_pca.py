"""API6: PCA — polars_ds vs sklearn.PCA."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
import polars_ds as pds

from _common import make_numeric_data, time_and_mem, save_result

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "bench_06_pca.json")


def pds_pca(df: pl.DataFrame, k=5):
    feats = [c for c in df.columns if c != "y"]
    return df.select(pds.principal_components(*[pl.col(f) for f in feats], k=k, center=True))


def sklearn_pca(df: pl.DataFrame, k=5):
    from sklearn.decomposition import PCA
    X = df.drop("y").fill_null(0.0).to_numpy()
    return PCA(n_components=k).fit_transform(X)


def main():
    out = {"api": "PCA", "rows": {}}
    for n in (10_000, 50_000, 200_000):
        df = make_numeric_data(n=n, n_features=20, missing_rate=0.0, outlier_rate=0.0)
        try:
            r_pds = time_and_mem(pds_pca, "polars_ds.pca", df, k=5, repeats=3)
        except Exception as e:
            print("pds.pca failed:", e); r_pds = None
        r_sk = time_and_mem(sklearn_pca, "sklearn.PCA", df, k=5, repeats=3)
        out["rows"][str(n)] = {
            "polars_ds": r_pds.dict() if r_pds else None,
            "sklearn": r_sk.dict(),
            "speedup_vs_sklearn": (r_sk.seconds / r_pds.seconds) if r_pds else None,
        }
        print(f"n={n:>7}: pds={(r_pds.seconds if r_pds else 'skip')}  sk={r_sk.seconds:.3f}s")
    save_result(RESULTS, out)
    print("saved:", RESULTS)


if __name__ == "__main__":
    main()
