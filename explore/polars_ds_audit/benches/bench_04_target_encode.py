"""API4: target encode — polars_ds vs sklearn.TargetEncoder (с OOF) vs category_encoders."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from polars_ds.pipeline import Blueprint

from _common import make_high_card_cat, time_and_mem, save_result

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "bench_04_target_encode.json")


def pds_te(df, cat_cols):
    bp = Blueprint(df, name="te", target="y").target_encode(
        cols=cat_cols, min_samples_leaf=20, smoothing=10.0
    )
    return bp.materialize().transform(df)


def sk_te(df, cat_cols):
    # sklearn 1.3+ TargetEncoder — cross-fitting встроен
    from sklearn.preprocessing import TargetEncoder
    pdf = df.to_pandas()
    enc = TargetEncoder(target_type="binary", smooth="auto", cv=5)
    return enc.fit_transform(pdf[cat_cols], pdf["y"])


def ce_te(df, cat_cols):
    from category_encoders.target_encoder import TargetEncoder
    pdf = df.to_pandas()
    return TargetEncoder(cols=cat_cols, smoothing=10.0).fit_transform(pdf[cat_cols], pdf["y"])


def main():
    out = {"api": "target encode", "note": "polars_ds БЕЗ OOF; sklearn С встроенным cross-fit cv=5; category_encoders БЕЗ OOF (базовая смесь с prior)", "rows": {}}
    for n in (10_000, 50_000, 200_000):
        df = make_high_card_cat(n=n, n_cat_cols=3, cardinality=500, seed=1)
        cat_cols = [c for c in df.columns if c.startswith("c")]
        r_pds = time_and_mem(pds_te, "polars_ds.target_encode (no OOF)", df, cat_cols, repeats=3)
        try:
            r_sk = time_and_mem(sk_te, "sklearn.TargetEncoder (cv=5)", df, cat_cols, repeats=3)
        except Exception as e:
            r_sk = None; print("sklearn TE failed:", e)
        try:
            r_ce = time_and_mem(ce_te, "category_encoders.TargetEncoder", df, cat_cols, repeats=3)
        except Exception as e:
            r_ce = None; print("category_encoders failed:", e)
        row = {"polars_ds_no_oof": r_pds.dict(),
               "sklearn_cv5": r_sk.dict() if r_sk else None,
               "category_encoders": r_ce.dict() if r_ce else None}
        out["rows"][str(n)] = row
        print(f"n={n:>7}: pds_no_oof={r_pds.seconds:.3f}s  sk_cv5={(r_sk.seconds if r_sk else 'skip')}  ce={(r_ce.seconds if r_ce else 'skip')}")
    save_result(RESULTS, out)
    print("saved:", RESULTS)


if __name__ == "__main__":
    main()
