"""Демонстрация: target leakage в голом polars_ds TE/WoE vs. наш OOFEncoder vs. sklearn.TargetEncoder(cv=5).

Методика:
- синтетический датасет: 3 категориальных признака с кардинальностью 500, слабый истинный сигнал.
- 4 ветки:
  (a) no-encoding baseline: LogReg на численных признаках + OHE cat (для reference)
  (b) plain polars_ds target_encode (БЕЗ OOF) → fit_transform на train
  (c) OOFTargetEncoder(cv=5) → fit_transform на train с OOF
  (d) sklearn.TargetEncoder(cv=5, target_type='binary') — референс industry-standard

Для каждой ветки: train AUC, test AUC, gap. Наличие большого gap для (b) и маленького для (c)/(d)
— эмпирическое подтверждение leakage без OOF.

Также замер времени fit_transform.
"""
from __future__ import annotations
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import polars as pl
from polars_ds.pipeline import Blueprint
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import TargetEncoder as SkTE
from sklearn.metrics import roc_auc_score

from _common import make_high_card_cat, train_test_split_frame, auc, save_result
from oof_encoders import OOFTargetEncoder

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "demo_oof_leak.json")


def to_numpy_feats(df: pl.DataFrame, cat_cols: list[str], num_cols: list[str]) -> np.ndarray:
    arrs = [df[c].cast(pl.Float64).to_numpy().reshape(-1, 1) for c in cat_cols]
    arrs += [df[c].cast(pl.Float64).to_numpy().reshape(-1, 1) for c in num_cols]
    X = np.hstack(arrs)
    return np.nan_to_num(X, nan=0.0)


def run(n=20000, cardinality=500, n_cat=3, signal=0.3, seed=7):
    df = make_high_card_cat(n=n, n_cat_cols=n_cat, cardinality=cardinality, signal_strength=signal, seed=seed)
    tr, te = train_test_split_frame(df, frac=0.7, seed=seed)
    cat_cols = [c for c in tr.columns if c.startswith("c")]
    num_cols = [c for c in tr.columns if c.startswith("n")]
    y_tr = tr["y"].to_numpy().astype(int)
    y_te = te["y"].to_numpy().astype(int)

    results = {}

    # (b) plain polars_ds TE
    t0 = time.perf_counter()
    bp = Blueprint(tr, name="te", target="y").target_encode(cols=cat_cols, min_samples_leaf=20, smoothing=10.0)
    pipe = bp.materialize()
    tr_b = pipe.transform(tr)
    te_b = pipe.transform(te)
    t_b = time.perf_counter() - t0
    X_tr = to_numpy_feats(tr_b, cat_cols, num_cols)
    X_te = to_numpy_feats(te_b, cat_cols, num_cols)
    m = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
    results["plain_polars_ds_TE"] = {
        "train_auc": float(roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1])),
        "test_auc":  float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])),
        "fit_transform_s": t_b,
    }

    # (c) OOFTargetEncoder cv=5
    t0 = time.perf_counter()
    enc = OOFTargetEncoder(cols=cat_cols, cv=5, min_samples_leaf=20, smoothing=10.0)
    tr_c = enc.fit_transform(tr.drop("y"), y_tr)
    te_c = enc.transform(te.drop("y"))
    t_c = time.perf_counter() - t0
    X_tr = to_numpy_feats(tr_c, cat_cols, num_cols)
    X_te = to_numpy_feats(te_c, cat_cols, num_cols)
    m = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
    results["OOFTargetEncoder_cv5"] = {
        "train_auc": float(roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1])),
        "test_auc":  float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])),
        "fit_transform_s": t_c,
    }

    # (d) sklearn.TargetEncoder cv=5
    t0 = time.perf_counter()
    tr_pd = tr.to_pandas(); te_pd = te.to_pandas()
    skte = SkTE(target_type="binary", smooth="auto", cv=5)
    tr_d_cat = skte.fit_transform(tr_pd[cat_cols], y_tr)
    te_d_cat = skte.transform(te_pd[cat_cols])
    t_d = time.perf_counter() - t0
    X_tr = np.hstack([tr_d_cat, tr_pd[num_cols].fillna(0).values])
    X_te = np.hstack([te_d_cat, te_pd[num_cols].fillna(0).values])
    m = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
    results["sklearn_TargetEncoder_cv5"] = {
        "train_auc": float(roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1])),
        "test_auc":  float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])),
        "fit_transform_s": t_d,
    }

    # (e) randomized=True — дешёвая регуляризация (WOEEncoder-style)
    t0 = time.perf_counter()
    enc2 = OOFTargetEncoder(cols=cat_cols, cv=0, randomized=True, sigma=0.05,
                             min_samples_leaf=20, smoothing=10.0, random_state=1)
    tr_e = enc2.fit_transform(tr.drop("y"), y_tr)
    te_e = enc2.transform(te.drop("y"))
    t_e = time.perf_counter() - t0
    X_tr = to_numpy_feats(tr_e, cat_cols, num_cols)
    X_te = to_numpy_feats(te_e, cat_cols, num_cols)
    m = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
    results["OOFTargetEncoder_randomized"] = {
        "train_auc": float(roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1])),
        "test_auc":  float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])),
        "fit_transform_s": t_e,
    }

    return results


def main():
    out = {"synthesis": {"n": 20000, "cardinality": 500, "signal": 0.3}, "variants": {}}
    print("=" * 76)
    print(f"{'variant':<35}  {'train AUC':>10}  {'test AUC':>9}  {'gap':>7}  {'fit,s':>7}")
    print("-" * 76)
    res = run()
    for name, r in res.items():
        gap = r["train_auc"] - r["test_auc"]
        print(f"{name:<35}  {r['train_auc']:>10.3f}  {r['test_auc']:>9.3f}  {gap:>7.3f}  {r['fit_transform_s']:>7.3f}")
        out["variants"][name] = {**r, "gap": gap}
    print("=" * 76)
    print("Интерпретация:")
    print(" * plain_polars_ds_TE: train_AUC >> test_AUC → target leakage без OOF")
    print(" * OOFTargetEncoder_cv5: train_AUC ≈ test_AUC → OOF защищает от leakage")
    print(" * sklearn_TargetEncoder_cv5: industry reference")
    print(" * OOFTargetEncoder_randomized: дёшево и работает частично (как WOEEncoder.randomized)")
    save_result(RESULTS, out)
    print(f"\nsaved: {RESULTS}")


if __name__ == "__main__":
    main()
