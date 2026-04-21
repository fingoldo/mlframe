"""Benchmark: Rust-native OOF target_encode vs Python OOFEncoder vs sklearn.TargetEncoder(cv=5).

Measures both correctness (leak gap) and speed.
Requires polars_ds built from fork with pl_target_encode_oof / pl_woe_discrete_oof.
"""
from __future__ import annotations
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import TargetEncoder as SkTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from _common import make_high_card_cat, train_test_split_frame, save_result
from oof_encoders import OOFTargetEncoder, OOFWOEEncoder

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "bench_rust_vs_python_oof.json")

N_REPEATS = 5


def _fit_predict(X_tr, y_tr, X_te, y_te):
    m = LogisticRegression(max_iter=500, solver="lbfgs").fit(X_tr, y_tr)
    return {
        "train_auc": float(roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1])),
        "test_auc": float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])),
    }


def _to_np(df, cols):
    return np.nan_to_num(np.column_stack([df[c].cast(pl.Float64).to_numpy() for c in cols]), nan=0.0)


def rust_oof_te(tr, cat_cols, y_tr, cv=5, seed=0, smoothing=10.0, min_samples_leaf=20):
    """Call the Rust pl_target_encode_oof directly via polars plugin."""
    import polars_ds.exprs.num as pds_num

    n = len(tr)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    fold_arr = np.zeros(n, dtype=np.uint32)
    for fold_k, (_, in_idx) in enumerate(splitter.split(np.zeros(n), y_tr.astype(int))):
        fold_arr[in_idx] = fold_k

    target_mean = float(y_tr.mean())
    tr_with = tr.with_columns([
        pl.Series("__y__", y_tr.astype(np.float64)),
        pl.Series("__fold__", fold_arr),
    ])

    exprs = []
    for c in cat_cols:
        expr = pds_num.pl_plugin(
            symbol="pl_target_encode_oof",
            args=[pl.col(c).cast(pl.String), pl.col("__y__"), pl.col("__fold__")],
            kwargs={
                "min_samples_leaf": float(min_samples_leaf),
                "smoothing": smoothing,
                "n_folds": int(cv),
                "default": target_mean,
            },
        ).alias(c)
        exprs.append(expr)

    result = tr_with.select(exprs + [pl.col(nc) for nc in tr.columns if nc not in cat_cols])
    return result


def rust_oof_woe(tr, cat_cols, y_tr, cv=5, seed=0):
    """Call the Rust pl_woe_discrete_oof directly via polars plugin."""
    import polars_ds.exprs.num as pds_num

    n = len(tr)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    fold_arr = np.zeros(n, dtype=np.uint32)
    for fold_k, (_, in_idx) in enumerate(splitter.split(np.zeros(n), y_tr.astype(int))):
        fold_arr[in_idx] = fold_k

    tr_with = tr.with_columns([
        pl.Series("__y__", y_tr.astype(np.float64)),
        pl.Series("__fold__", fold_arr),
    ])

    exprs = []
    for c in cat_cols:
        expr = pds_num.pl_plugin(
            symbol="pl_woe_discrete_oof",
            args=[pl.col(c).cast(pl.String), pl.col("__y__"), pl.col("__fold__")],
            kwargs={
                "n_folds": int(cv),
                "default": 0.0,
            },
        ).alias(c)
        exprs.append(expr)

    result = tr_with.select(exprs + [pl.col(nc) for nc in tr.columns if nc not in cat_cols])
    return result


def run_one(n, seed, cardinality=500, n_cat=3, signal=0.3):
    df = make_high_card_cat(n=n, n_cat_cols=n_cat, cardinality=cardinality,
                            signal_strength=signal, seed=seed)
    tr, te = train_test_split_frame(df, frac=0.7, seed=seed)
    cat_cols = [c for c in tr.columns if c.startswith("c")]
    num_cols = [c for c in tr.columns if c.startswith("n")]
    all_cols = cat_cols + num_cols
    y_tr = tr["y"].to_numpy().astype(int)
    y_te = te["y"].to_numpy().astype(int)

    results = {}

    # --- TE ---
    te_enc_by_cv = {}
    for cv in (3, 5):
        # Python OOF
        t0 = time.perf_counter()
        enc = OOFTargetEncoder(cols=cat_cols, cv=cv, min_samples_leaf=20, smoothing=10.0, random_state=seed)
        tr_enc = enc.fit_transform(tr.drop("y"), y_tr)
        te_enc = enc.transform(te.drop("y"))
        t_py = time.perf_counter() - t0
        r = _fit_predict(_to_np(tr_enc, all_cols), y_tr, _to_np(te_enc, all_cols), y_te)
        results[f"python_OOF_TE_cv{cv}"] = {**r, "time_s": t_py}
        te_enc_by_cv[cv] = te_enc

        # Rust OOF
        t0 = time.perf_counter()
        tr_rust = rust_oof_te(tr.drop("y"), cat_cols, y_tr.astype(np.float64), cv=cv, seed=seed)
        t_rust = time.perf_counter() - t0
        r = _fit_predict(_to_np(tr_rust, all_cols), y_tr, _to_np(te_enc, all_cols), y_te)
        results[f"rust_OOF_TE_cv{cv}"] = {**r, "time_s": t_rust}

    # sklearn cv=5 (reference)
    t0 = time.perf_counter()
    tr_pd = tr.to_pandas(); te_pd = te.to_pandas()
    skte = SkTE(target_type="binary", smooth="auto", cv=5)
    tr_cat = skte.fit_transform(tr_pd[cat_cols], y_tr)
    te_cat = skte.transform(te_pd[cat_cols])
    t_sk = time.perf_counter() - t0
    X_tr = np.hstack([tr_cat, tr_pd[num_cols].fillna(0).values])
    X_te = np.hstack([te_cat, te_pd[num_cols].fillna(0).values])
    r = _fit_predict(X_tr, y_tr, X_te, y_te)
    results["sklearn_TE_cv5"] = {**r, "time_s": t_sk}

    # --- WoE ---
    for cv in (3, 5):
        # Python OOF
        t0 = time.perf_counter()
        enc_w = OOFWOEEncoder(cols=cat_cols, cv=cv, random_state=seed)
        tr_enc_w = enc_w.fit_transform(tr.drop("y"), y_tr)
        te_enc_w = enc_w.transform(te.drop("y"))
        t_py_w = time.perf_counter() - t0
        r = _fit_predict(_to_np(tr_enc_w, all_cols), y_tr, _to_np(te_enc_w, all_cols), y_te)
        results[f"python_OOF_WoE_cv{cv}"] = {**r, "time_s": t_py_w}

        # Rust OOF WoE
        t0 = time.perf_counter()
        tr_rust_w = rust_oof_woe(tr.drop("y"), cat_cols, y_tr.astype(np.float64), cv=cv, seed=seed)
        t_rust_w = time.perf_counter() - t0
        r = _fit_predict(_to_np(tr_rust_w, all_cols), y_tr, _to_np(te_enc_w, all_cols), y_te)
        results[f"rust_OOF_WoE_cv{cv}"] = {**r, "time_s": t_rust_w}

    return results


def main():
    all_results = {}
    for n in (20_000, 50_000, 100_000):
        print(f"\n{'='*80}")
        print(f"  n = {n}")
        print(f"{'='*80}")
        runs = []
        for rep in range(N_REPEATS):
            seed = 7 + rep * 13
            res = run_one(n, seed)
            runs.append(res)
            for name, r in res.items():
                gap = r["train_auc"] - r["test_auc"]
                print(f"  rep={rep+1} {name:<20} gap={gap:+.3f}  t={r['time_s']:.4f}s")

        agg = {}
        for vn in runs[0]:
            times = [r[vn]["time_s"] for r in runs]
            gaps = [r[vn]["train_auc"] - r[vn]["test_auc"] for r in runs]
            agg[vn] = {
                "time_median": float(np.median(times)),
                "time_std": float(np.std(times)),
                "gap_median": float(np.median(gaps)),
            }
        all_results[str(n)] = agg

        print(f"\n  {'variant':<30} {'time median':>12} {'gap':>8}")
        for vn, m in agg.items():
            print(f"  {vn:<30} {m['time_median']:>12.4f}s {m['gap_median']:>+8.3f}")
        for cv in (3, 5):
            py_key = f"python_OOF_TE_cv{cv}"
            rs_key = f"rust_OOF_TE_cv{cv}"
            if py_key in agg and rs_key in agg:
                speedup = agg[py_key]["time_median"] / agg[rs_key]["time_median"]
                print(f"  -> Rust TE cv={cv} speedup vs Python: {speedup:.1f}x")
            py_key = f"python_OOF_WoE_cv{cv}"
            rs_key = f"rust_OOF_WoE_cv{cv}"
            if py_key in agg and rs_key in agg:
                speedup = agg[py_key]["time_median"] / agg[rs_key]["time_median"]
                print(f"  -> Rust WoE cv={cv} speedup vs Python: {speedup:.1f}x")

    save_result(RESULTS, all_results)
    print(f"\nsaved: {RESULTS}")


if __name__ == "__main__":
    main()
