"""Extended leak experiment: TE + WoE × {polars_ds, OOF, sklearn/category_encoders}, 5 repeats.

Extends original demo_oof_leak.py with:
- category_encoders.TargetEncoder (plain, no OOF) — to prove it leaks same as polars_ds
- category_encoders.WOEEncoder (plain + randomized) vs OOFWOEEncoder(cv=5)
- polars_ds.woe_encode (plain) vs OOFWOEEncoder(cv=5)
- 5 independent repeats (different seeds), report median ± std
"""
from __future__ import annotations
import json, os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import polars as pl
from polars_ds.pipeline import Blueprint
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import TargetEncoder as SkTE
from sklearn.metrics import roc_auc_score

from _common import make_high_card_cat, train_test_split_frame, save_result
from oof_encoders import OOFTargetEncoder, OOFWOEEncoder

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "demo_oof_leak_extended.json")

N_REPEATS = 5
N, CARDINALITY, N_CAT, SIGNAL = 20_000, 500, 3, 0.3


def _fit_predict(X_tr, y_tr, X_te, y_te):
    m = LogisticRegression(max_iter=500, solver="lbfgs").fit(X_tr, y_tr)
    return {
        "train_auc": float(roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1])),
        "test_auc": float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])),
    }


def _to_np(df, cols):
    return np.nan_to_num(np.column_stack([df[c].cast(pl.Float64).to_numpy() for c in cols]), nan=0.0)


def run_one(seed: int):
    df = make_high_card_cat(n=N, n_cat_cols=N_CAT, cardinality=CARDINALITY,
                            signal_strength=SIGNAL, seed=seed)
    tr, te = train_test_split_frame(df, frac=0.7, seed=seed)
    cat_cols = [c for c in tr.columns if c.startswith("c")]
    num_cols = [c for c in tr.columns if c.startswith("n")]
    all_cols = cat_cols + num_cols
    y_tr = tr["y"].to_numpy().astype(int)
    y_te = te["y"].to_numpy().astype(int)

    results = {}

    # ---------- TARGET ENCODING ----------

    # 1) plain polars_ds TE
    t0 = time.perf_counter()
    bp = Blueprint(tr, name="te", target="y").target_encode(cols=cat_cols, min_samples_leaf=20, smoothing=10.0)
    pipe = bp.materialize()
    tr_enc = pipe.transform(tr); te_enc = pipe.transform(te)
    t1 = time.perf_counter() - t0
    r = _fit_predict(_to_np(tr_enc, all_cols), y_tr, _to_np(te_enc, all_cols), y_te)
    results["pds_TE_plain"] = {**r, "fit_transform_s": t1}

    # 2a) OOFTargetEncoder cv=3
    t0 = time.perf_counter()
    enc = OOFTargetEncoder(cols=cat_cols, cv=3, min_samples_leaf=20, smoothing=10.0, random_state=seed)
    tr_enc = enc.fit_transform(tr.drop("y"), y_tr); te_enc = enc.transform(te.drop("y"))
    t1 = time.perf_counter() - t0
    r = _fit_predict(_to_np(tr_enc, all_cols), y_tr, _to_np(te_enc, all_cols), y_te)
    results["pds_OOF_TE_cv3"] = {**r, "fit_transform_s": t1}

    # 2b) OOFTargetEncoder cv=5
    t0 = time.perf_counter()
    enc = OOFTargetEncoder(cols=cat_cols, cv=5, min_samples_leaf=20, smoothing=10.0, random_state=seed)
    tr_enc = enc.fit_transform(tr.drop("y"), y_tr); te_enc = enc.transform(te.drop("y"))
    t1 = time.perf_counter() - t0
    r = _fit_predict(_to_np(tr_enc, all_cols), y_tr, _to_np(te_enc, all_cols), y_te)
    results["pds_OOF_TE_cv5"] = {**r, "fit_transform_s": t1}

    # 3a) sklearn.TargetEncoder cv=3
    t0 = time.perf_counter()
    tr_pd = tr.to_pandas(); te_pd = te.to_pandas()
    skte3 = SkTE(target_type="binary", smooth="auto", cv=3)
    tr_cat3 = skte3.fit_transform(tr_pd[cat_cols], y_tr)
    te_cat3 = skte3.transform(te_pd[cat_cols])
    t1 = time.perf_counter() - t0
    X_tr = np.hstack([tr_cat3, tr_pd[num_cols].fillna(0).values])
    X_te = np.hstack([te_cat3, te_pd[num_cols].fillna(0).values])
    r = _fit_predict(X_tr, y_tr, X_te, y_te)
    results["sklearn_TE_cv3"] = {**r, "fit_transform_s": t1}

    # 3b) sklearn.TargetEncoder cv=5
    t0 = time.perf_counter()
    skte = SkTE(target_type="binary", smooth="auto", cv=5)
    tr_cat = skte.fit_transform(tr_pd[cat_cols], y_tr)
    te_cat = skte.transform(te_pd[cat_cols])
    t1 = time.perf_counter() - t0
    X_tr = np.hstack([tr_cat, tr_pd[num_cols].fillna(0).values])
    X_te = np.hstack([te_cat, te_pd[num_cols].fillna(0).values])
    r = _fit_predict(X_tr, y_tr, X_te, y_te)
    results["sklearn_TE_cv5"] = {**r, "fit_transform_s": t1}

    # 4) category_encoders.TargetEncoder (plain, no OOF)
    try:
        import category_encoders as ce
        t0 = time.perf_counter()
        ce_te = ce.TargetEncoder(cols=cat_cols, smoothing=10.0, min_samples_leaf=20)
        tr_ce = ce_te.fit_transform(tr_pd[cat_cols], y_tr)
        te_ce = ce_te.transform(te_pd[cat_cols])
        t1 = time.perf_counter() - t0
        X_tr = np.hstack([tr_ce.values, tr_pd[num_cols].fillna(0).values])
        X_te = np.hstack([te_ce.values, te_pd[num_cols].fillna(0).values])
        r = _fit_predict(X_tr, y_tr, X_te, y_te)
        results["catenc_TE_plain"] = {**r, "fit_transform_s": t1}
    except ImportError:
        results["catenc_TE_plain"] = "not installed"

    # 5) OOF randomized (noise-based, cheap)
    t0 = time.perf_counter()
    enc2 = OOFTargetEncoder(cols=cat_cols, cv=0, randomized=True, sigma=0.05,
                             min_samples_leaf=20, smoothing=10.0, random_state=seed)
    tr_enc = enc2.fit_transform(tr.drop("y"), y_tr); te_enc = enc2.transform(te.drop("y"))
    t1 = time.perf_counter() - t0
    r = _fit_predict(_to_np(tr_enc, all_cols), y_tr, _to_np(te_enc, all_cols), y_te)
    results["pds_TE_randomized"] = {**r, "fit_transform_s": t1}

    # ---------- WOE ENCODING ----------

    # 6) plain polars_ds WoE
    t0 = time.perf_counter()
    bp = Blueprint(tr, name="woe", target="y").woe_encode(cols=cat_cols)
    pipe = bp.materialize()
    tr_enc = pipe.transform(tr); te_enc = pipe.transform(te)
    t1 = time.perf_counter() - t0
    r = _fit_predict(_to_np(tr_enc, all_cols), y_tr, _to_np(te_enc, all_cols), y_te)
    results["pds_WoE_plain"] = {**r, "fit_transform_s": t1}

    # 7a) OOFWOEEncoder cv=3
    t0 = time.perf_counter()
    enc = OOFWOEEncoder(cols=cat_cols, cv=3, random_state=seed)
    tr_enc = enc.fit_transform(tr.drop("y"), y_tr); te_enc = enc.transform(te.drop("y"))
    t1 = time.perf_counter() - t0
    r = _fit_predict(_to_np(tr_enc, all_cols), y_tr, _to_np(te_enc, all_cols), y_te)
    results["pds_OOF_WoE_cv3"] = {**r, "fit_transform_s": t1}

    # 7b) OOFWOEEncoder cv=5
    t0 = time.perf_counter()
    enc = OOFWOEEncoder(cols=cat_cols, cv=5, random_state=seed)
    tr_enc = enc.fit_transform(tr.drop("y"), y_tr); te_enc = enc.transform(te.drop("y"))
    t1 = time.perf_counter() - t0
    r = _fit_predict(_to_np(tr_enc, all_cols), y_tr, _to_np(te_enc, all_cols), y_te)
    results["pds_OOF_WoE_cv5"] = {**r, "fit_transform_s": t1}

    # 8) category_encoders.WOEEncoder plain
    try:
        import category_encoders as ce
        t0 = time.perf_counter()
        ce_woe = ce.WOEEncoder(cols=cat_cols, randomized=False)
        tr_ce = ce_woe.fit_transform(tr_pd[cat_cols], y_tr)
        te_ce = ce_woe.transform(te_pd[cat_cols])
        t1 = time.perf_counter() - t0
        X_tr = np.hstack([tr_ce.values, tr_pd[num_cols].fillna(0).values])
        X_te = np.hstack([te_ce.values, te_pd[num_cols].fillna(0).values])
        r = _fit_predict(X_tr, y_tr, X_te, y_te)
        results["catenc_WoE_plain"] = {**r, "fit_transform_s": t1}
    except ImportError:
        results["catenc_WoE_plain"] = "not installed"

    # 9) category_encoders.WOEEncoder randomized
    try:
        import category_encoders as ce
        t0 = time.perf_counter()
        ce_woe2 = ce.WOEEncoder(cols=cat_cols, randomized=True, sigma=0.05)
        tr_ce = ce_woe2.fit_transform(tr_pd[cat_cols], y_tr)
        te_ce = ce_woe2.transform(te_pd[cat_cols])
        t1 = time.perf_counter() - t0
        X_tr = np.hstack([tr_ce.values, tr_pd[num_cols].fillna(0).values])
        X_te = np.hstack([te_ce.values, te_pd[num_cols].fillna(0).values])
        r = _fit_predict(X_tr, y_tr, X_te, y_te)
        results["catenc_WoE_randomized"] = {**r, "fit_transform_s": t1}
    except ImportError:
        results["catenc_WoE_randomized"] = "not installed"

    # 10) OOFWOEEncoder randomized (no OOF, just noise)
    t0 = time.perf_counter()
    enc3 = OOFWOEEncoder(cols=cat_cols, cv=0, randomized=True, sigma=0.05, random_state=seed)
    tr_enc = enc3.fit_transform(tr.drop("y"), y_tr); te_enc = enc3.transform(te.drop("y"))
    t1 = time.perf_counter() - t0
    r = _fit_predict(_to_np(tr_enc, all_cols), y_tr, _to_np(te_enc, all_cols), y_te)
    results["pds_WoE_randomized"] = {**r, "fit_transform_s": t1}

    return results


def aggregate(all_runs: list[dict]) -> dict:
    """Aggregate N_REPEATS runs into median ± std."""
    agg = {}
    variant_names = [k for k in all_runs[0] if isinstance(all_runs[0][k], dict)]
    for vn in variant_names:
        metrics = {}
        for metric in ("train_auc", "test_auc", "fit_transform_s"):
            vals = [r[vn][metric] for r in all_runs if isinstance(r.get(vn), dict)]
            if vals:
                metrics[f"{metric}_median"] = float(np.median(vals))
                metrics[f"{metric}_std"] = float(np.std(vals))
        if metrics:
            metrics["gap_median"] = metrics["train_auc_median"] - metrics["test_auc_median"]
        agg[vn] = metrics
    return agg


def main():
    all_runs = []
    for rep in range(N_REPEATS):
        seed = 7 + rep * 13
        print(f"\n--- Repeat {rep+1}/{N_REPEATS} (seed={seed}) ---")
        res = run_one(seed)
        all_runs.append(res)
        for name, r in res.items():
            if isinstance(r, dict):
                gap = r["train_auc"] - r["test_auc"]
                print(f"  {name:<30} train={r['train_auc']:.3f}  test={r['test_auc']:.3f}  gap={gap:+.3f}  t={r['fit_transform_s']:.3f}s")

    agg = aggregate(all_runs)
    print(f"\n{'='*90}")
    print(f"{'variant':<30}  {'train AUC':>10}  {'test AUC':>9}  {'gap':>7}  {'time,s':>7}  {'± std gap':>10}")
    print(f"{'-'*90}")
    for name, m in agg.items():
        print(f"{name:<30}  {m['train_auc_median']:>10.3f}  {m['test_auc_median']:>9.3f}"
              f"  {m['gap_median']:>+7.3f}  {m['fit_transform_s_median']:>7.3f}"
              f"  {m.get('test_auc_std', 0):>10.3f}")
    print(f"{'='*90}")

    out = {
        "config": {"n": N, "cardinality": CARDINALITY, "n_cat": N_CAT,
                    "signal": SIGNAL, "n_repeats": N_REPEATS},
        "per_run": all_runs,
        "aggregated": agg,
    }
    save_result(RESULTS, out)
    print(f"\nsaved: {RESULTS}")


if __name__ == "__main__":
    main()
