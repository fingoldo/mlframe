"""iter120 bench: prepare_df_for_catboost polars numeric loop -- gate dtype-membership BEFORE the full-column is_null().any() scan.

OLD shape (git HEAD): for every non-cat/text column, compute df[var].is_null().any() (a full-column scan) FIRST, then check whether the
dtype is in the castable set. For Float32/Float64 columns -- the overwhelmingly common 10M ML-feature case -- the null scan is computed and
immediately discarded (the dtype is not in _INT_BOOL_TO_F32 / _INT_TO_F64, so nothing is appended regardless of null status).

NEW shape: check the (free) dtype-membership dict lookup FIRST; only run the full-column is_null().any() scan for columns whose dtype is
actually castable. Bit-identical by construction (the OLD code never appended for non-castable dtypes anyway).

Run:
    PYTHONPATH=src CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 python src/mlframe/preprocessing/_benchmarks/bench_cb_prep_null_scan_iter120.py
"""
import sys
sys.modules["cupy"] = None
import time

import numpy as np
import polars as pl

N = 10_000_000
RNG = np.random.default_rng(0)


def make_df():
    # Common ML-feature frame: many Float64 columns (NOT castable -> OLD wastes a null scan on each) + a couple castable Int columns.
    data = {}
    for i in range(8):
        data[f"f{i}"] = RNG.standard_normal(N)
    data["i0"] = RNG.integers(0, 100, N).astype(np.int32)
    data["i1"] = RNG.integers(0, 2, N).astype(np.int8)
    return pl.DataFrame(data)


_INT_BOOL_TO_F32 = (pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32, pl.Boolean)
_INT_TO_F64 = (pl.Int64, pl.UInt64)


def old_numeric_loop(df, cat_features, text_features):
    numeric_exprs = []
    for var in df.columns:
        if var not in cat_features and var not in text_features:
            dtype = df[var].dtype
            if df[var].is_null().any():
                if dtype in _INT_BOOL_TO_F32:
                    numeric_exprs.append(pl.col(var).cast(pl.Float32))
                elif dtype in _INT_TO_F64:
                    numeric_exprs.append(pl.col(var).cast(pl.Float64))
    return numeric_exprs


def new_numeric_loop(df, cat_features, text_features):
    numeric_exprs = []
    for var in df.columns:
        if var not in cat_features and var not in text_features:
            dtype = df[var].dtype
            if dtype in _INT_BOOL_TO_F32:
                if df[var].is_null().any():
                    numeric_exprs.append(pl.col(var).cast(pl.Float32))
            elif dtype in _INT_TO_F64:
                if df[var].is_null().any():
                    numeric_exprs.append(pl.col(var).cast(pl.Float64))
    return numeric_exprs


def bestof(fn, df, reps=9):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(df, [], [])
        ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts))


def main():
    df = make_df()
    # warm
    old_numeric_loop(df, [], [])
    new_numeric_loop(df, [], [])
    # identity
    o = [str(e) for e in old_numeric_loop(df, [], [])]
    n = [str(e) for e in new_numeric_loop(df, [], [])]
    assert o == n, (o, n)
    o_min, o_med = bestof(old_numeric_loop, df)
    n_min, n_med = bestof(new_numeric_loop, df)
    print(f"OLD  min={o_min*1e3:.2f}ms med={o_med*1e3:.2f}ms")
    print(f"NEW  min={n_min*1e3:.2f}ms med={n_med*1e3:.2f}ms")
    print(f"speedup min={o_min/n_min:.2f}x med={o_med/n_med:.2f}x")
    print(f"identity OK ({len(o)} exprs)")


if __name__ == "__main__":
    main()
