"""API3: WoE encode — polars_ds vs category_encoders."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from polars_ds.pipeline import Blueprint

from _common import make_high_card_cat, time_and_mem, save_result

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "bench_03_woe.json")


def pds_woe(df: pl.DataFrame, cat_cols: list[str]) -> pl.DataFrame:
    bp = Blueprint(df, name="w", target="y").woe_encode(cols=cat_cols)
    return bp.materialize().transform(df)


def ce_woe(df, cat_cols):
    from category_encoders.woe import WOEEncoder
    pdf = df.to_pandas() if isinstance(df, pl.DataFrame) else df
    enc = WOEEncoder(cols=cat_cols)
    return enc.fit_transform(pdf[cat_cols], pdf["y"])


def main():
    out = {"api": "WoE encode", "rows": {}}
    for n in (10_000, 50_000, 200_000):
        df = make_high_card_cat(n=n, n_cat_cols=3, cardinality=500, seed=1)
        cat_cols = [c for c in df.columns if c.startswith("c")]
        r_pds = time_and_mem(pds_woe, "polars_ds.woe_encode", df, cat_cols, repeats=3)
        try:
            r_ce = time_and_mem(ce_woe, "category_encoders.WOEEncoder", df, cat_cols, repeats=3)
        except Exception as e:
            r_ce = None
            print("category_encoders failed:", e)
        out["rows"][str(n)] = {
            "polars_ds": r_pds.dict(),
            "category_encoders": r_ce.dict() if r_ce else None,
            "speedup_vs_ce": (r_ce.seconds / r_pds.seconds) if r_ce else None,
        }
        print(f"n={n:>7}: pds={r_pds.seconds:.3f}s  ce={(r_ce.seconds if r_ce else 'skip'):>}")
    save_result(RESULTS, out)
    print("saved:", RESULTS)


if __name__ == "__main__":
    main()
