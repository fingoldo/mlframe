"""Binary-search the null-fraction threshold at which CB 1.2.10's
Polars fastpath dispatcher breaks on Categorical columns."""
from __future__ import annotations

import sys
import numpy as np
import polars as pl
from catboost import CatBoostClassifier

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def fit_probe(null_frac, n=500, n_unique=3):
    """Build a Polars DF with 1 Categorical column at the given null
    fraction, try to fit CB, return (ok, error_kind)."""
    nonnull_count = int(n * (1 - null_frac))
    pool = np.random.choice(["a", "b", "c"][:n_unique], size=nonnull_count).tolist()
    vals = pool + [None] * (n - nonnull_count)
    np.random.shuffle(vals)
    df = pl.DataFrame({
        "num": np.random.randn(n).astype(np.float32),
        "cat": pl.Series("cat", vals, dtype=pl.String).cast(pl.Categorical),
    })
    y = np.random.randint(0, 2, size=n)
    try:
        m = CatBoostClassifier(iterations=2, verbose=0, allow_writing_files=False, thread_count=1)
        m.fit(df, y, cat_features=["cat"])
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:100]}"


def main():
    np.random.seed(42)
    print("CB Polars fastpath vs Categorical null_frac:")
    # Test a range of null fractions
    for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0]:
        ok, err = fit_probe(frac)
        status = "OK  " if ok else "FAIL"
        print(f"  null_frac={frac:<6}  {status}  {err or ''}")


if __name__ == "__main__":
    main()
