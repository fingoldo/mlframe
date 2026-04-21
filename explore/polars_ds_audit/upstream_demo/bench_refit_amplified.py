"""Show the refit_downstream_on_full effect in AUC, not just in scaler stats.

Strategy: amplify the OOF-vs-full distribution shift by shrinking the training
set so each OOF fold removes a larger fraction of per-category data. With
small train + many categories, the OOF mapping diverges meaningfully from the
full mapping; scale-sensitive downstream models then react in AUC.

Three downstream models tested (decreasing scale-sensitivity):
  1. k-NN k=5, weights='distance' — distances dominate, k small
  2. RBF-SVM C=1, gamma='scale' — kernel width depends on training std
  3. MLP (16,), max_iter=200 — small NN, weight inits sensitive to scale
"""
from __future__ import annotations
import time, json
import numpy as np
import polars as pl
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

from polars_ds.pipeline import Blueprint

N_REPEATS = 5
TRAIN_N = 2000  # amplification: small train -> larger OOF/full divergence
TEST_N = 8000
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


def encode(train_df, test_df, refit_full):
    bp = Blueprint(
        train_df.lazy(), target="target",
        refit_downstream_on_full=refit_full,
    )
    bp.target_encode(cols=CAT_COLS, target="target", cv=3, default="mean")
    bp.scale(cols=CAT_COLS, method="standard")
    df_train_lazy, pipe = bp.materialize(return_df=True)
    train_enc = df_train_lazy.collect()
    test_enc = pipe.transform(test_df)
    X_tr = np.nan_to_num(train_enc.select(CAT_COLS).to_numpy().astype(float), nan=0.0)
    y_tr = train_enc["target"].to_numpy()
    X_te = np.nan_to_num(test_enc.select(CAT_COLS).to_numpy().astype(float), nan=0.0)
    y_te = test_enc["target"].to_numpy()
    return X_tr, y_tr, X_te, y_te


def eval_model(name, model_factory, X_tr, y_tr, X_te, y_te):
    m = model_factory()
    t0 = time.perf_counter()
    m.fit(X_tr, y_tr)
    if hasattr(m, "predict_proba"):
        proba = m.predict_proba(X_te)[:, 1]
    else:
        proba = m.decision_function(X_te)
    elapsed = time.perf_counter() - t0
    auc = roc_auc_score(y_te, proba)
    return float(auc), elapsed


def main():
    print("Loading Amazon...")
    df = load_amazon()
    n = len(df)
    print(f"  shape: {df.shape}, positive rate: {df['target'].mean():.3f}")
    print(f"  amplification: TRAIN_N={TRAIN_N} (vs full {n}), TEST_N={TEST_N}")

    models = {
        "kNN k=5 weighted":  lambda: KNeighborsClassifier(
            n_neighbors=5, weights="distance", n_jobs=-1
        ),
        "RBF-SVM C=1":       lambda: SVC(
            C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42
        ),
        "MLP (16,)":         lambda: MLPClassifier(
            hidden_layer_sizes=(16,), max_iter=200, random_state=42, early_stopping=False
        ),
    }

    rows = []
    for repeat in range(N_REPEATS):
        seed = 100 + repeat
        rng = np.random.default_rng(seed)
        idx = np.arange(n); rng.shuffle(idx)
        train_df = df[idx[:TRAIN_N].tolist()]
        test_df = df[idx[TRAIN_N:TRAIN_N + TEST_N].tolist()]
        print(f"\n[rep {repeat+1}/{N_REPEATS}, seed={seed}]")

        encoded = {}
        for refit in (True, False):
            X_tr, y_tr, X_te, y_te = encode(train_df, test_df, refit_full=refit)
            encoded[refit] = (X_tr, y_tr, X_te, y_te)
            tr_std = float(X_tr.std(axis=0).mean())
            te_std = float(X_te.std(axis=0).mean())
            print(f"  refit_full={str(refit):5s}  train<std>={tr_std:.4f}  test<std>={te_std:.4f}")

        for mname, mfact in models.items():
            for refit in (True, False):
                X_tr, y_tr, X_te, y_te = encoded[refit]
                auc, elapsed = eval_model(mname, mfact, X_tr, y_tr, X_te, y_te)
                rows.append({
                    "repeat": repeat, "model": mname,
                    "refit_full": refit, "auc": auc, "elapsed_s": elapsed,
                })
            full = [r for r in rows if r["repeat"] == repeat and r["model"] == mname and r["refit_full"]][0]
            oof  = [r for r in rows if r["repeat"] == repeat and r["model"] == mname and not r["refit_full"]][0]
            delta = full["auc"] - oof["auc"]
            print(f"    {mname:18s}  full={full['auc']:.4f}  oof={oof['auc']:.4f}  "
                  f"diff(full-oof)={delta:+.4f}  ({elapsed:.1f}s)")

    print()
    print("=" * 90)
    print(f"{'model':20s} {'mode':12s} {'AUC median':>11s} {'AUC mean':>10s} {'+- std':>9s}  {'wins':>5s}")
    print("-" * 90)

    def med(xs): return float(np.median(xs))
    def mean(xs): return float(np.mean(xs))
    def std(xs): return float(np.std(xs))

    summary = []
    for mname in models:
        full_aucs = [r["auc"] for r in rows if r["model"] == mname and r["refit_full"]]
        oof_aucs  = [r["auc"] for r in rows if r["model"] == mname and not r["refit_full"]]
        # paired diffs
        full_wins = sum(1 for f, o in zip(full_aucs, oof_aucs) if f > o)
        diff_mean = mean([f - o for f, o in zip(full_aucs, oof_aucs)])
        for label, aucs in [("refit_full", full_aucs), ("refit_oof", oof_aucs)]:
            print(f"{mname:20s} {label:12s} {med(aucs):11.4f} {mean(aucs):10.4f} {std(aucs):9.4f}  "
                  f"{(full_wins if label=='refit_full' else N_REPEATS - full_wins):>5d}")
        print(f"{mname:20s} {'mean diff':12s} {'':>11s} {diff_mean:+10.4f}  (full minus oof; "
              f"full wins {full_wins}/{N_REPEATS})")
        summary.append({
            "model": mname,
            "median_auc_full": med(full_aucs),
            "median_auc_oof": med(oof_aucs),
            "mean_diff_full_minus_oof": diff_mean,
            "full_wins_count": full_wins,
            "n_repeats": N_REPEATS,
        })
    print("=" * 90)

    out = {"summary": summary, "raw": rows, "settings": {
        "train_n": TRAIN_N, "test_n": TEST_N, "n_repeats": N_REPEATS,
    }}
    out_path = RESULTS_DIR / "bench_refit_amplified.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
