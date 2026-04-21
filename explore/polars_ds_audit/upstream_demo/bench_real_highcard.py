"""
Benchmark: polars_ds OOF vs sklearn vs category_encoders on high-cardinality real dataset.

Dataset: Amazon Employee Access (OpenML #4135, Kaggle competition).
  32769 rows, 9 categorical features, max cardinality 7518 (RESOURCE).
  Target: access granted (1) vs denied (0), imbalanced (94% positive).

This is the key test: high-cardinality features cause severe target leakage
with naive target/WoE encoding. OOF should show clear gap reduction.
"""
import time
import json
import numpy as np
import polars as pl
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder as SklearnTE
import category_encoders as ce

import polars_ds.exprs.num as pds_num

N_REPEATS = 5
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CAT_COLS = ["RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2",
            "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE"]


def load_amazon():
    data = fetch_openml(data_id=4135, as_frame=True)
    pdf = data.data.copy()
    for c in CAT_COLS:
        pdf[c] = pdf[c].astype(str)
    pdf["target"] = (data.target == "1").astype(float)
    return pl.from_pandas(pdf[CAT_COLS + ["target"]])


def encode_pds_plain(train_df, test_df, cols, target="target"):
    t0 = time.perf_counter()
    mappings = {}
    for c in cols:
        mapping = train_df.select(
            pds_num.target_encode(c, target, min_samples_leaf=20, smoothing=10.0).implode()
        ).to_series()[0]
        mappings[c] = (mapping.struct.field("value"), mapping.struct.field("to"))
    train_enc = train_df.with_columns([
        pl.col(c).replace_strict(old=mappings[c][0], new=mappings[c][1], default=None).alias(c)
        for c in cols
    ])
    test_enc = test_df.with_columns([
        pl.col(c).replace_strict(old=mappings[c][0], new=mappings[c][1], default=None).alias(c)
        for c in cols
    ])
    return train_enc, test_enc, time.perf_counter() - t0


def _make_stratified_fold_ids(y, cv, seed=42):
    """Stratified fold assignment: preserves class ratio (matches sklearn StratifiedKFold)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    fold_ids = np.empty(n, dtype=np.uint32)
    unique_vals = np.unique(y[~np.isnan(y)])
    if len(unique_vals) <= 20:
        for val in unique_vals:
            class_idx = np.where(y == val)[0]
            perm = rng.permutation(len(class_idx))
            shuffled = class_idx[perm]
            for k, chunk in enumerate(np.array_split(shuffled, cv)):
                fold_ids[chunk] = k
    else:
        indices = rng.permutation(n)
        for k, chunk in enumerate(np.array_split(indices, cv)):
            fold_ids[chunk] = k
    return fold_ids


def encode_pds_oof(train_df, test_df, cols, cv=3, target="target", smooth_auto=False,
                   bayes_variant="classical"):
    t0 = time.perf_counter()
    y = train_df[target].to_numpy()
    fold_ids = _make_stratified_fold_ids(y, cv)
    fold_series = pl.Series("__fold__", fold_ids, dtype=pl.UInt32)

    oof_exprs = [
        pds_num.target_encode_oof(c, target, "__fold__", n_folds=cv,
                                  min_samples_leaf=20, smoothing=10.0, default=0.0,
                                  smooth_auto=smooth_auto,
                                  bayes_variant=bayes_variant).alias(c)
        for c in cols
    ]
    train_enc = train_df.with_columns(fold_series).with_columns(oof_exprs).drop("__fold__")

    # For unseen categories in test, use train target mean (matching sklearn)
    train_target_mean = train_df[target].mean()
    if smooth_auto:
        # Use Bayes encoding for test too
        mappings = {}
        for c in cols:
            mapping = train_df.select(
                pds_num.target_encode_bayes(c, target, bayes_variant=bayes_variant).implode()
            ).to_series()[0]
            mappings[c] = (mapping.struct.field("value"), mapping.struct.field("to"))
    else:
        mappings = {}
        for c in cols:
            mapping = train_df.select(
                pds_num.target_encode(c, target, min_samples_leaf=20, smoothing=10.0).implode()
            ).to_series()[0]
            mappings[c] = (mapping.struct.field("value"), mapping.struct.field("to"))
    test_enc = test_df.with_columns([
        pl.col(c).replace_strict(old=mappings[c][0], new=mappings[c][1], default=train_target_mean).alias(c)
        for c in cols
    ])
    return train_enc, test_enc, time.perf_counter() - t0


def encode_pds_woe_plain(train_df, test_df, cols, target="target"):
    t0 = time.perf_counter()
    mappings = {}
    for c in cols:
        mapping = train_df.select(
            pds_num.woe_discrete(c, target).implode()
        ).to_series()[0]
        mappings[c] = (mapping.struct.field("value"), mapping.struct.field("woe"))
    train_enc = train_df.with_columns([
        pl.col(c).replace_strict(old=mappings[c][0], new=mappings[c][1], default=0.0).alias(c)
        for c in cols
    ])
    test_enc = test_df.with_columns([
        pl.col(c).replace_strict(old=mappings[c][0], new=mappings[c][1], default=0.0).alias(c)
        for c in cols
    ])
    return train_enc, test_enc, time.perf_counter() - t0


def encode_pds_woe_oof(train_df, test_df, cols, cv=3, target="target"):
    t0 = time.perf_counter()
    y = train_df[target].to_numpy()
    fold_ids = _make_stratified_fold_ids(y, cv)
    fold_series = pl.Series("__fold__", fold_ids, dtype=pl.UInt32)

    oof_exprs = [
        pds_num.woe_encode_oof(c, target, "__fold__", n_folds=cv, default=0.0).alias(c)
        for c in cols
    ]
    train_enc = train_df.with_columns(fold_series).with_columns(oof_exprs).drop("__fold__")

    mappings = {}
    for c in cols:
        mapping = train_df.select(
            pds_num.woe_discrete(c, target).implode()
        ).to_series()[0]
        mappings[c] = (mapping.struct.field("value"), mapping.struct.field("woe"))
    test_enc = test_df.with_columns([
        pl.col(c).replace_strict(old=mappings[c][0], new=mappings[c][1], default=0.0).alias(c)
        for c in cols
    ])
    return train_enc, test_enc, time.perf_counter() - t0


def encode_sklearn_te(train_pdf, test_pdf, cols, cv=5):
    t0 = time.perf_counter()
    enc = SklearnTE(categories="auto", target_type="binary", cv=cv, random_state=42)
    X_train_enc = enc.fit_transform(train_pdf[cols], train_pdf["target"].values)
    X_test_enc = enc.transform(test_pdf[cols])
    elapsed = time.perf_counter() - t0

    train_pdf_enc = train_pdf.copy()
    test_pdf_enc = test_pdf.copy()
    for i, c in enumerate(cols):
        train_pdf_enc[c] = X_train_enc[:, i]
        test_pdf_enc[c] = X_test_enc[:, i]
    return train_pdf_enc, test_pdf_enc, elapsed


def encode_catenc_te(train_pdf, test_pdf, cols):
    t0 = time.perf_counter()
    enc = ce.TargetEncoder(cols=cols, return_df=True)
    train_enc = enc.fit_transform(train_pdf[cols + ["target"]], train_pdf["target"])
    test_enc = enc.transform(test_pdf[cols + ["target"]])
    elapsed = time.perf_counter() - t0
    train_enc["target"] = train_pdf["target"].values
    test_enc["target"] = test_pdf["target"].values
    return train_enc, test_enc, elapsed


def encode_catenc_woe(train_pdf, test_pdf, cols):
    t0 = time.perf_counter()
    enc = ce.WOEEncoder(cols=cols, return_df=True)
    train_enc = enc.fit_transform(train_pdf[cols + ["target"]], train_pdf["target"])
    test_enc = enc.transform(test_pdf[cols + ["target"]])
    elapsed = time.perf_counter() - t0
    train_enc["target"] = train_pdf["target"].values
    test_enc["target"] = test_pdf["target"].values
    return train_enc, test_enc, elapsed


def encode_blueprint_cv3(train_df, test_df, cols):
    from polars_ds.pipeline import Blueprint
    t0 = time.perf_counter()
    bp = Blueprint(train_df.lazy(), target="target")
    bp.target_encode(cols=cols, target="target", cv=3, default="mean")
    df_train_lazy, pipe = bp.materialize(return_df=True)
    train_enc = df_train_lazy.collect()
    test_enc = pipe.transform(test_df)
    return train_enc, test_enc, time.perf_counter() - t0


def encode_blueprint_with_scaler(train_df, test_df, cols, refit_downstream_on_full=True):
    """Blueprint with TE(cv=3) followed by standard scaler on the encoded columns.

    Used to measure the effect of ``refit_downstream_on_full``: when True the
    scaler fits on the full-mapping distribution (matches transform(test));
    when False (legacy) it fits on the OOF distribution — causing train/serve skew.
    """
    from polars_ds.pipeline import Blueprint
    t0 = time.perf_counter()
    bp = Blueprint(
        train_df.lazy(), target="target",
        refit_downstream_on_full=refit_downstream_on_full,
    )
    bp.target_encode(cols=cols, target="target", cv=3, default="mean")
    bp.scale(cols=cols, method="standard")
    df_train_lazy, pipe = bp.materialize(return_df=True)
    train_enc = df_train_lazy.collect()
    test_enc = pipe.transform(test_df)
    return train_enc, test_enc, time.perf_counter() - t0


def evaluate(train_enc, test_enc, cols, is_pandas=False):
    if is_pandas:
        X_train = train_enc[cols].values.astype(float)
        X_test = test_enc[cols].values.astype(float)
        y_train = train_enc["target"].values
        y_test = test_enc["target"].values
    else:
        X_train = train_enc.select(cols).to_numpy().astype(float)
        X_test = test_enc.select(cols).to_numpy().astype(float)
        y_train = train_enc["target"].to_numpy()
        y_test = test_enc["target"].to_numpy()

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, lr.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    return train_auc, test_auc


def run_all():
    print("Loading Amazon Employee Access dataset (high-cardinality)...")
    df = load_amazon()
    print(f"  shape: {df.shape}, positive rate: {df['target'].mean():.3f}")
    for c in CAT_COLS:
        print(f"  {c}: nunique={df[c].n_unique()}")

    all_results = []

    for repeat in range(N_REPEATS):
        seed = 42 + repeat
        # Stratified-ish split
        n = len(df)
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        split = int(0.75 * n)
        train_df = df[idx[:split].tolist()]
        test_df = df[idx[split:].tolist()]

        train_pdf = train_df.to_pandas()
        test_pdf = test_df.to_pandas()

        variants = {}

        # 1. pds_TE_plain
        tr, te, t = encode_pds_plain(train_df, test_df, CAT_COLS)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS)
        variants["pds_TE_plain"] = (auc_tr, auc_te, t)

        # 2. pds_OOF_TE_cv3
        tr, te, t = encode_pds_oof(train_df, test_df, CAT_COLS, cv=3)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS)
        variants["pds_OOF_TE_cv3"] = (auc_tr, auc_te, t)

        # 3. pds_OOF_TE_cv5
        tr, te, t = encode_pds_oof(train_df, test_df, CAT_COLS, cv=5)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS)
        variants["pds_OOF_TE_cv5"] = (auc_tr, auc_te, t)

        # 4. pds_OOF_Bayes_cv3 (classical Micci-Barreca)
        tr, te, t = encode_pds_oof(train_df, test_df, CAT_COLS, cv=3, smooth_auto=True,
                                   bayes_variant="classical")
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS)
        variants["pds_OOF_Bayes_classical_cv3"] = (auc_tr, auc_te, t)

        # 5. pds_OOF_Bayes_cv5 (classical)
        tr, te, t = encode_pds_oof(train_df, test_df, CAT_COLS, cv=5, smooth_auto=True,
                                   bayes_variant="classical")
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS)
        variants["pds_OOF_Bayes_classical_cv5"] = (auc_tr, auc_te, t)

        # 5b. pds_OOF_Bayes pooled (sklearn-equivalent within-variance)
        tr, te, t = encode_pds_oof(train_df, test_df, CAT_COLS, cv=3, smooth_auto=True,
                                   bayes_variant="pooled")
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS)
        variants["pds_OOF_Bayes_pooled_cv3"] = (auc_tr, auc_te, t)

        tr, te, t = encode_pds_oof(train_df, test_df, CAT_COLS, cv=5, smooth_auto=True,
                                   bayes_variant="pooled")
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS)
        variants["pds_OOF_Bayes_pooled_cv5"] = (auc_tr, auc_te, t)

        # 6. sklearn_TE_cv5
        tr, te, t = encode_sklearn_te(train_pdf, test_pdf, CAT_COLS, cv=5)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS, is_pandas=True)
        variants["sklearn_TE_cv5"] = (auc_tr, auc_te, t)

        # 7. sklearn_TE_cv3
        tr, te, t = encode_sklearn_te(train_pdf, test_pdf, CAT_COLS, cv=3)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS, is_pandas=True)
        variants["sklearn_TE_cv3"] = (auc_tr, auc_te, t)

        # 8. catenc_TE_plain
        tr, te, t = encode_catenc_te(train_pdf, test_pdf, CAT_COLS)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS, is_pandas=True)
        variants["catenc_TE_plain"] = (auc_tr, auc_te, t)

        # 9. pds_WoE_plain
        tr, te, t = encode_pds_woe_plain(train_df, test_df, CAT_COLS)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS)
        variants["pds_WoE_plain"] = (auc_tr, auc_te, t)

        # 10. pds_OOF_WoE_cv3
        tr, te, t = encode_pds_woe_oof(train_df, test_df, CAT_COLS, cv=3)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS)
        variants["pds_OOF_WoE_cv3"] = (auc_tr, auc_te, t)

        # 11. catenc_WoE_plain
        tr, te, t = encode_catenc_woe(train_pdf, test_pdf, CAT_COLS)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS, is_pandas=True)
        variants["catenc_WoE_plain"] = (auc_tr, auc_te, t)

        # 12. Blueprint cv=3
        tr, te, t = encode_blueprint_cv3(train_df, test_df, CAT_COLS)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS)
        variants["pds_Blueprint_cv3"] = (auc_tr, auc_te, t)

        # 13. Blueprint TE(cv=3) + scale, refit_downstream_on_full=True (default)
        tr, te, t = encode_blueprint_with_scaler(train_df, test_df, CAT_COLS,
                                                  refit_downstream_on_full=True)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS)
        variants["pds_Blueprint_TE_scale_refit_full"] = (auc_tr, auc_te, t)

        # 14. Blueprint TE(cv=3) + scale, refit_downstream_on_full=False (legacy)
        tr, te, t = encode_blueprint_with_scaler(train_df, test_df, CAT_COLS,
                                                  refit_downstream_on_full=False)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS)
        variants["pds_Blueprint_TE_scale_refit_oof"] = (auc_tr, auc_te, t)

        for name, (auc_tr, auc_te, elapsed) in variants.items():
            gap = auc_tr - auc_te
            print(f"  [{repeat+1}/{N_REPEATS}] {name:30s} train={auc_tr:.4f}  test={auc_te:.4f}  gap={gap:+.4f}  t={elapsed:.3f}s")
            all_results.append({
                "variant": name, "repeat": repeat,
                "train_auc": round(auc_tr, 6), "test_auc": round(auc_te, 6),
                "gap": round(gap, 6), "time_s": round(elapsed, 4),
            })

    # Aggregate
    print()
    print("=" * 105)
    header = f"{'variant':35s} {'train AUC':>10s} {'test AUC':>10s} {'gap':>8s} {'time,s':>8s}  {'std gap':>8s}"
    print(header)
    print("-" * 105)

    variant_names = list(dict.fromkeys(r["variant"] for r in all_results))
    summary = []
    for v in variant_names:
        rows = [r for r in all_results if r["variant"] == v]
        train_aucs = [r["train_auc"] for r in rows]
        test_aucs = [r["test_auc"] for r in rows]
        gaps = [r["gap"] for r in rows]
        times = [r["time_s"] for r in rows]
        med = lambda xs: sorted(xs)[len(xs)//2]
        std_gap = np.std(gaps)
        line = {
            "variant": v,
            "train_auc_median": round(med(train_aucs), 6),
            "test_auc_median": round(med(test_aucs), 6),
            "gap_median": round(med(gaps), 6),
            "time_median": round(med(times), 4),
            "gap_std": round(float(std_gap), 4),
        }
        summary.append(line)
        print(f"{v:35s} {line['train_auc_median']:10.4f} {line['test_auc_median']:10.4f} {line['gap_median']:+8.4f} {line['time_median']:8.3f}  +/- {line['gap_std']:.4f}")

    print("=" * 105)

    out_path = RESULTS_DIR / "bench_real_amazon.json"
    with open(out_path, "w") as f:
        json.dump({"raw": all_results, "summary": summary, "dataset": "Amazon_employee_access",
                    "n_rows": len(df), "max_cardinality": 7518}, f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    run_all()
