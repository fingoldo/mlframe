"""API8: power transforms (yeo-johnson / box-cox) — polars_ds НЕ поддерживает → сравниваем только sklearn vs scipy.

Вывод этого бенча — в REPORT.md: в polars_ds 0.10.3 нет first-class power transforms; upstream-кандидат.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl

from _common import make_numeric_data, time_and_mem, save_result

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "bench_08_power_transform.json")


def sklearn_yj(df):
    from sklearn.preprocessing import PowerTransformer
    X = df.drop("y").fill_null(0.0).to_numpy()
    X = np.abs(X) + 1e-3
    return PowerTransformer(method="yeo-johnson").fit_transform(X)


def scipy_yj(df):
    from scipy.stats import yeojohnson
    X = df.drop("y").fill_null(0.0).to_numpy()
    return np.column_stack([yeojohnson(X[:, i])[0] for i in range(X.shape[1])])


def main():
    out = {"api": "power transforms", "note": "polars_ds 0.10.3 НЕ имеет first-class yeo-johnson/box-cox — kandидat на upstream", "rows": {}}
    for n in (10_000, 100_000):
        df = make_numeric_data(n=n, n_features=10, missing_rate=0.0, outlier_rate=0.0)
        r_sk = time_and_mem(sklearn_yj, "sklearn.PowerTransformer", df, repeats=3)
        r_sp = time_and_mem(scipy_yj, "scipy.stats.yeojohnson (per-col)", df, repeats=3)
        out["rows"][str(n)] = {"sklearn": r_sk.dict(), "scipy": r_sp.dict(), "polars_ds": "not implemented upstream"}
        print(f"n={n:>7}: sklearn={r_sk.seconds:.3f}s  scipy={r_sp.seconds:.3f}s  (pds: skip)")
    save_result(RESULTS, out)
    print("saved:", RESULTS)


if __name__ == "__main__":
    main()
