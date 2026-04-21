"""Benchmark the Blueprint(refit_downstream_on_full) option with a scale-sensitive
downstream model (k-NN) to show the numerical effect that's invisible under
LogisticRegression (which is scale-invariant up to convergence).

Pipeline: target_encode(cv=3) -> scale(standard) -> KNeighborsClassifier(k=50).
- refit_full=True: scaler fits on the full-mapping distribution (matches
  pipe.transform(test)). k-NN sees consistent train/test feature scaling.
- refit_full=False (legacy): scaler fits on the OOF distribution. At test time
  the full mapping produces slightly different stats → scaled test features
  land in a shifted region → k-NN neighbours are computed in a distorted space.

On the Amazon Employee Access dataset (9 high-card features, imbalanced binary
target) the skew is small in absolute terms but visible in both AUC and in the
direct comparison of scaler stats.
"""
from __future__ import annotations

import time
import json
import numpy as np
import polars as pl
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

from polars_ds.pipeline import Blueprint

N_REPEATS = 5
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CAT_COLS = [
    "RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2",
    "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE",
]


def load_amazon() -> pl.DataFrame:
    data = fetch_openml(data_id=4135, as_frame=True)
    pdf = data.data.copy()
    for c in CAT_COLS:
        pdf[c] = pdf[c].astype(str)
    pdf["target"] = (data.target == "1").astype(float)
    return pl.from_pandas(pdf[CAT_COLS + ["target"]])


def build_and_eval(train_df: pl.DataFrame, test_df: pl.DataFrame,
                   refit_full: bool, k_neighbors: int = 50) -> dict:
    t0 = time.perf_counter()
    bp = Blueprint(
        train_df.lazy(), target="target",
        refit_downstream_on_full=refit_full,
    )
    bp.target_encode(cols=CAT_COLS, target="target", cv=3, default="mean")
    bp.scale(cols=CAT_COLS, method="standard")
    df_train_lazy, pipe = bp.materialize(return_df=True)
    train_enc = df_train_lazy.collect()
    test_enc = pipe.transform(test_df)
    build_time = time.perf_counter() - t0

    X_tr = np.nan_to_num(train_enc.select(CAT_COLS).to_numpy().astype(float), nan=0.0)
    y_tr = train_enc["target"].to_numpy()
    X_te = np.nan_to_num(test_enc.select(CAT_COLS).to_numpy().astype(float), nan=0.0)
    y_te = test_enc["target"].to_numpy()

    # Per-feature scaled-mean/std as seen by the downstream model; use train
    # because that's what transform(test) should mimic.
    tr_mean = X_tr.mean(axis=0)
    tr_std = X_tr.std(axis=0)
    te_mean = X_te.mean(axis=0)
    te_std = X_te.std(axis=0)

    knn = KNeighborsClassifier(n_neighbors=k_neighbors, n_jobs=-1)
    t1 = time.perf_counter()
    knn.fit(X_tr, y_tr)
    proba = knn.predict_proba(X_te)[:, 1]
    fit_pred_time = time.perf_counter() - t1
    test_auc = roc_auc_score(y_te, proba)
    proba_tr = knn.predict_proba(X_tr)[:, 1]
    train_auc = roc_auc_score(y_tr, proba_tr)

    return {
        "refit_full": refit_full,
        "build_time": build_time,
        "fit_pred_time": fit_pred_time,
        "train_auc": float(train_auc),
        "test_auc": float(test_auc),
        "train_mean_mean_abs": float(np.abs(tr_mean).mean()),
        "train_std_mean": float(tr_std.mean()),
        "test_mean_mean_abs": float(np.abs(te_mean).mean()),
        "test_std_mean": float(te_std.mean()),
    }


def main():
    print("Loading Amazon Employee Access...")
    df = load_amazon()
    n = len(df)
    print(f"  shape: {df.shape}, positive rate: {df['target'].mean():.3f}")

    all_rows = []
    for repeat in range(N_REPEATS):
        seed = 42 + repeat
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        split = int(0.75 * n)
        train_df = df[idx[:split].tolist()]
        test_df = df[idx[split:].tolist()]

        print(f"\n[repeat {repeat+1}/{N_REPEATS}, seed={seed}] train n={len(train_df)} test n={len(test_df)}")
        for refit in (True, False):
            r = build_and_eval(train_df, test_df, refit_full=refit)
            r["repeat"] = repeat
            all_rows.append(r)
            print(f"  refit_full={str(refit):5s} | "
                  f"train_auc={r['train_auc']:.4f}  test_auc={r['test_auc']:.4f}  "
                  f"build={r['build_time']:.2f}s  knn={r['fit_pred_time']:.2f}s  "
                  f"train<|mean|>={r['train_mean_mean_abs']:.4f}  train<std>={r['train_std_mean']:.4f}  "
                  f"test<|mean|>={r['test_mean_mean_abs']:.4f}  test<std>={r['test_std_mean']:.4f}")

    # Aggregate
    print()
    print("=" * 100)
    print(f"{'mode':12s} {'train AUC':>11s} {'test AUC':>10s} "
          f"{'test<|mean|>':>13s} {'test<std>':>11s}  {'delta vs full AUC':>18s}")
    print("-" * 100)

    def med(xs): return float(np.median(xs))
    full_rows = [r for r in all_rows if r["refit_full"]]
    oof_rows = [r for r in all_rows if not r["refit_full"]]
    for label, rows in [("refit_full", full_rows), ("refit_oof", oof_rows)]:
        tr_auc = med([r["train_auc"] for r in rows])
        te_auc = med([r["test_auc"] for r in rows])
        te_mean = med([r["test_mean_mean_abs"] for r in rows])
        te_std = med([r["test_std_mean"] for r in rows])
        line_end = ""
        if label == "refit_oof":
            full_auc = med([r["test_auc"] for r in full_rows])
            line_end = f"  {te_auc - full_auc:+.4f}"
        else:
            line_end = "  (baseline)"
        print(f"{label:12s} {tr_auc:11.4f} {te_auc:10.4f} {te_mean:13.4f} {te_std:11.4f} {line_end:>16s}")
    print("=" * 100)

    out_path = RESULTS_DIR / "bench_refit_downstream_knn.json"
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
