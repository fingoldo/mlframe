"""Reproduce the exact production CB Polars fastpath TypeError.

The bench_polars_largestring_cb_xgb.py experiment DISPROVED the
large_string hypothesis — CB handles Dictionary<uint32, large_string>
fine in the simple case. The prod shape is different:
  - 9 Categorical cat_features (with varying null counts)
  - 15 Boolean
  - 40 Float32
  - 30 Int16
  - 4 String (text_features, declared via text_features=[...])
  - Some Categorical cols are nearly 100% null (hourly_budget_type:
    n_unique=1, nulls=810000 of 810000).

Probe dimensions that could trigger "No matching signature found":
  - Int16 columns (CB may expect Int32/Int64)
  - Boolean columns
  - text_features (pl.String) passed alongside cat_features
  - all-null or near-all-null Categorical
"""
from __future__ import annotations

import sys
import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def fit_cb(df, cat_features=None, text_features=None, label="?"):
    from catboost import CatBoostClassifier
    y = np.random.randint(0, 2, size=df.height)
    try:
        m = CatBoostClassifier(iterations=2, verbose=0, allow_writing_files=False, thread_count=1)
        kw = {}
        if cat_features:
            kw["cat_features"] = cat_features
        if text_features:
            kw["text_features"] = text_features
        m.fit(df, y, **kw)
        print(f"  [{label}] CB OK")
        return True
    except Exception as e:
        msg = str(e)[:250].replace("\n", " ")
        print(f"  [{label}] CB FAIL {type(e).__name__}: {msg}")
        return False


def main():
    np.random.seed(42)
    import catboost
    print(f"catboost: {catboost.__version__}, polars: {pl.__version__}")

    n = 500

    # ===============================================================
    # Case A: baseline — just one Categorical + one Float
    # ===============================================================
    df_a = pl.DataFrame({
        "num": np.random.randn(n).astype(np.float32),
        "cat": pl.Series("cat", np.random.choice(["x", "y", "z"], size=n)).cast(pl.Categorical),
    })
    fit_cb(df_a, cat_features=["cat"], label="A: 1 cat + 1 f32")

    # ===============================================================
    # Case B: add Int16 columns
    # ===============================================================
    df_b = df_a.with_columns(
        pl.Series("i16_1", np.random.randint(-100, 100, size=n).astype(np.int16)),
        pl.Series("i16_2", np.random.randint(-100, 100, size=n).astype(np.int16)),
    )
    fit_cb(df_b, cat_features=["cat"], label="B: +2 Int16")

    # ===============================================================
    # Case C: add Boolean columns
    # ===============================================================
    df_c = df_a.with_columns(
        pl.Series("b1", np.random.choice([True, False], size=n)),
        pl.Series("b2", np.random.choice([True, False], size=n)),
    )
    fit_cb(df_c, cat_features=["cat"], label="C: +2 Boolean")

    # ===============================================================
    # Case D: multiple Categorical (9 like prod)
    # ===============================================================
    df_d = pl.DataFrame({
        "num": np.random.randn(n).astype(np.float32),
        **{
            f"cat{i}": pl.Series(f"cat{i}", np.random.choice(["a", "b", "c"], size=n)).cast(pl.Categorical)
            for i in range(9)
        },
    })
    fit_cb(df_d, cat_features=[f"cat{i}" for i in range(9)], label="D: 9 Categorical")

    # ===============================================================
    # Case E: Categorical with many nulls (near 100%)
    # ===============================================================
    nonnull_count = n // 100
    vals = ["a"] * nonnull_count + [None] * (n - nonnull_count)
    np.random.shuffle(vals)
    df_e = pl.DataFrame({
        "num": np.random.randn(n).astype(np.float32),
        "null_cat": pl.Series("null_cat", vals, dtype=pl.String).cast(pl.Categorical),
    })
    fit_cb(df_e, cat_features=["null_cat"], label="E: near-100%-null Categorical")

    # ===============================================================
    # Case F: add text_features (pl.String)
    # ===============================================================
    df_f = df_a.with_columns(
        pl.Series("txt1", ["hello world", "data science"] * (n // 2)),
        pl.Series("txt2", ["alpha beta", "gamma delta"] * (n // 2)),
    )
    fit_cb(df_f, cat_features=["cat"], text_features=["txt1", "txt2"],
           label="F: +text_features")

    # ===============================================================
    # Case G: full production mix — 9 cat + 15 bool + 40 f32 + 30 i16 + 4 text
    # ===============================================================
    cols = {
        **{
            f"cat{i}": pl.Series(f"cat{i}", np.random.choice(["a", "b", "c"], size=n)).cast(pl.Categorical)
            for i in range(9)
        },
        **{
            f"b{i}": np.random.choice([True, False], size=n)
            for i in range(15)
        },
        **{
            f"f{i}": np.random.randn(n).astype(np.float32)
            for i in range(40)
        },
        **{
            f"i{i}": np.random.randint(-100, 100, size=n).astype(np.int16)
            for i in range(30)
        },
        **{
            f"txt{i}": np.random.choice(["hello world", "data science", "alpha beta"], size=n).tolist()
            for i in range(4)
        },
    }
    df_g = pl.DataFrame(cols)
    fit_cb(
        df_g,
        cat_features=[f"cat{i}" for i in range(9)],
        text_features=[f"txt{i}" for i in range(4)],
        label="G: full prod mix",
    )

    # ===============================================================
    # Case H: G but with near-100%-null Categoricals (matching prod)
    # ===============================================================
    def make_nullish(nullfrac=0.99):
        nonnull_count = int(n * (1 - nullfrac))
        vals_obj = ["a"] * nonnull_count + [None] * (n - nonnull_count)
        np.random.shuffle(vals_obj)
        return pl.Series(vals_obj, dtype=pl.String).cast(pl.Categorical)

    cols_h = {
        **{
            f"cat{i}": make_nullish(nullfrac=0.99) if i in (4, 5, 6) else
                pl.Series(f"cat{i}", np.random.choice(["a", "b", "c"], size=n)).cast(pl.Categorical)
            for i in range(9)
        },
        **{f"b{i}": np.random.choice([True, False], size=n) for i in range(15)},
        **{f"f{i}": np.random.randn(n).astype(np.float32) for i in range(40)},
        **{f"i{i}": np.random.randint(-100, 100, size=n).astype(np.int16) for i in range(30)},
        **{f"txt{i}": np.random.choice(["hello world", "data science"], size=n).tolist()
           for i in range(4)},
    }
    df_h = pl.DataFrame(cols_h)
    fit_cb(
        df_h,
        cat_features=[f"cat{i}" for i in range(9)],
        text_features=[f"txt{i}" for i in range(4)],
        label="H: prod mix + 3 near-100%-null cats",
    )


if __name__ == "__main__":
    main()
