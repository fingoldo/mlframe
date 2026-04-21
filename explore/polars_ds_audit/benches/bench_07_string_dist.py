"""API7: string distances — polars_ds vs rapidfuzz."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
import polars_ds as pds

from _common import time_and_mem, save_result

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "bench_07_string_dist.json")


def make_strings(n: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    alph = "abcdefghijklmnopqrstuvwxyz"
    a = ["".join(rng.choice(list(alph), size=rng.integers(5, 15))) for _ in range(n)]
    b = ["".join(rng.choice(list(alph), size=rng.integers(5, 15))) for _ in range(n)]
    return pl.DataFrame({"a": a, "b": b})


def pds_leven(df: pl.DataFrame):
    return df.select(pds.str_leven("a", "b").alias("d"))


def pds_jaro(df: pl.DataFrame):
    return df.select(pds.str_jaro("a", "b").alias("d"))


def rf_leven(df: pl.DataFrame):
    from rapidfuzz.distance import Levenshtein
    a = df["a"].to_list(); b = df["b"].to_list()
    return [Levenshtein.distance(x, y) for x, y in zip(a, b)]


def rf_jaro(df: pl.DataFrame):
    from rapidfuzz.distance import JaroWinkler
    a = df["a"].to_list(); b = df["b"].to_list()
    return [JaroWinkler.distance(x, y) for x, y in zip(a, b)]


def main():
    out = {"api": "string distances", "rows": {}}
    for n in (10_000, 50_000, 200_000):
        df = make_strings(n)
        r_pds_l = time_and_mem(pds_leven, "polars_ds.str_leven", df, repeats=3)
        r_pds_j = time_and_mem(pds_jaro, "polars_ds.str_jaro", df, repeats=3)
        r_rf_l = time_and_mem(rf_leven, "rapidfuzz.Levenshtein", df, repeats=3)
        r_rf_j = time_and_mem(rf_jaro, "rapidfuzz.JaroWinkler", df, repeats=3)
        out["rows"][str(n)] = {
            "polars_ds_leven": r_pds_l.dict(),
            "rapidfuzz_leven": r_rf_l.dict(),
            "polars_ds_jaro": r_pds_j.dict(),
            "rapidfuzz_jaro": r_rf_j.dict(),
            "speedup_leven_vs_rf": r_rf_l.seconds / r_pds_l.seconds,
            "speedup_jaro_vs_rf": r_rf_j.seconds / r_pds_j.seconds,
        }
        print(f"n={n:>7}: pds_leven={r_pds_l.seconds:.3f}s  rf_leven={r_rf_l.seconds:.3f}s  pds_jaro={r_pds_j.seconds:.3f}s  rf_jaro={r_rf_j.seconds:.3f}s")
    save_result(RESULTS, out)
    print("saved:", RESULTS)


if __name__ == "__main__":
    main()
