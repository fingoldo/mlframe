"""
Benchmark: polars_ds OOF vs sklearn.TargetEncoder vs category_encoders on Adult Census dataset.

Compares ML predictive power (AUC) and encoding speed on a real-world dataset from sklearn docs.
Dataset: OpenML "adult" (Census Income, 48842 rows, 8 categorical features).

Variants tested:
  1. pds_TE_plain         — polars_ds target_encode, no OOF (leaky baseline)
  2. pds_OOF_TE_cv3       — polars_ds Rust OOF target_encode, cv=3
  3. pds_OOF_TE_cv5       — polars_ds Rust OOF target_encode, cv=5
  4. sklearn_TE_cv5        — sklearn.TargetEncoder(cv=5) (built-in cross-fitting)
  5. sklearn_TE_cv3        — sklearn.TargetEncoder(cv=3)
  6. catenc_TE_plain       — category_encoders.TargetEncoder (no OOF, leaky)
  7. pds_WoE_plain         — polars_ds woe_encode, no OOF
  8. pds_OOF_WoE_cv3       — polars_ds Rust OOF woe_encode, cv=3
  9. catenc_WoE_plain      — category_encoders.WOEEncoder (no OOF)
  10. pds_Blueprint_cv3    — Blueprint.target_encode(cv=3) end-to-end

Results: train AUC, test AUC, gap, encode_time_s (median of N_REPEATS).
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

CAT_COLS = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"]


def load_adult():
    """Load Adult dataset, return polars DataFrame with binary target."""
    data = fetch_openml("adult", version=2, as_frame=True)
    pdf = data.data.copy()
    pdf["target"] = (data.target == ">50K").astype(float)
    # Keep only categorical + target
    pdf = pdf[CAT_COLS + ["target"]]
    return pl.from_pandas(pdf)


def encode_pds_plain(train_df, test_df, cols, target="target"):
    """Plain polars_ds target_encode (fit on train, apply mapping to both)."""
    t0 = time.perf_counter()
    # Build mapping from train
    mappings = {}
    for c in cols:
        mapping = train_df.select(
            pds_num.target_encode(c, target, min_samples_leaf=20, smoothing=10.0).implode()
        ).to_series()[0]
        old = mapping.struct.field("value")
        new = mapping.struct.field("to")
        mappings[c] = (old, new)

    train_enc = train_df.with_columns([
        pl.col(c).replace_strict(old=mappings[c][0], new=mappings[c][1], default=None).alias(c)
        for c in cols
    ])
    test_enc = test_df.with_columns([
        pl.col(c).replace_strict(old=mappings[c][0], new=mappings[c][1], default=None).alias(c)
        for c in cols
    ])
    elapsed = time.perf_counter() - t0
    return train_enc, test_enc, elapsed


def encode_pds_oof(train_df, test_df, cols, cv=3, target="target"):
    """OOF polars_ds target_encode (Rust, leak-safe on train)."""
    t0 = time.perf_counter()
    n = len(train_df)
    rng = np.random.default_rng(42)
    fold_ids = rng.integers(0, cv, size=n).astype(np.uint32)
    fold_series = pl.Series("__fold__", fold_ids, dtype=pl.UInt32)

    # OOF on train
    oof_exprs = [
        pds_num.target_encode_oof(c, target, "__fold__", n_folds=cv,
                                  min_samples_leaf=20, smoothing=10.0, default=0.0).alias(c)
        for c in cols
    ]
    train_enc = train_df.with_columns(fold_series).with_columns(oof_exprs).drop("__fold__")

    # Full-train mapping for test
    mappings = {}
    for c in cols:
        mapping = train_df.select(
            pds_num.target_encode(c, target, min_samples_leaf=20, smoothing=10.0).implode()
        ).to_series()[0]
        old = mapping.struct.field("value")
        new = mapping.struct.field("to")
        mappings[c] = (old, new)

    test_enc = test_df.with_columns([
        pl.col(c).replace_strict(old=mappings[c][0], new=mappings[c][1], default=None).alias(c)
        for c in cols
    ])
    elapsed = time.perf_counter() - t0
    return train_enc, test_enc, elapsed


def encode_pds_woe_plain(train_df, test_df, cols, target="target"):
    """Plain polars_ds woe_encode."""
    t0 = time.perf_counter()
    mappings = {}
    for c in cols:
        mapping = train_df.select(
            pds_num.woe_discrete(c, target).implode()
        ).to_series()[0]
        old = mapping.struct.field("value")
        new = mapping.struct.field("woe")
        mappings[c] = (old, new)

    train_enc = train_df.with_columns([
        pl.col(c).replace_strict(old=mappings[c][0], new=mappings[c][1], default=0.0).alias(c)
        for c in cols
    ])
    test_enc = test_df.with_columns([
        pl.col(c).replace_strict(old=mappings[c][0], new=mappings[c][1], default=0.0).alias(c)
        for c in cols
    ])
    elapsed = time.perf_counter() - t0
    return train_enc, test_enc, elapsed


def encode_pds_woe_oof(train_df, test_df, cols, cv=3, target="target"):
    """OOF polars_ds woe_encode (Rust)."""
    t0 = time.perf_counter()
    n = len(train_df)
    rng = np.random.default_rng(42)
    fold_ids = rng.integers(0, cv, size=n).astype(np.uint32)
    fold_series = pl.Series("__fold__", fold_ids, dtype=pl.UInt32)

    oof_exprs = [
        pds_num.woe_discrete_oof(c, target, "__fold__", n_folds=cv, default=0.0).alias(c)
        for c in cols
    ]
    train_enc = train_df.with_columns(fold_series).with_columns(oof_exprs).drop("__fold__")

    mappings = {}
    for c in cols:
        mapping = train_df.select(
            pds_num.woe_discrete(c, target).implode()
        ).to_series()[0]
        old = mapping.struct.field("value")
        new = mapping.struct.field("woe")
        mappings[c] = (old, new)

    test_enc = test_df.with_columns([
        pl.col(c).replace_strict(old=mappings[c][0], new=mappings[c][1], default=0.0).alias(c)
        for c in cols
    ])
    elapsed = time.perf_counter() - t0
    return train_enc, test_enc, elapsed


def encode_sklearn_te(train_pdf, test_pdf, cols, cv=5):
    """sklearn.TargetEncoder with built-in cross-fitting."""
    t0 = time.perf_counter()
    enc = SklearnTE(categories="auto", target_type="binary", cv=cv, random_state=42)
    X_train = train_pdf[cols].copy()
    y_train = train_pdf["target"].values
    X_test = test_pdf[cols].copy()

    X_train_enc = enc.fit_transform(X_train, y_train)
    X_test_enc = enc.transform(X_test)
    elapsed = time.perf_counter() - t0

    train_pdf_enc = train_pdf.copy()
    test_pdf_enc = test_pdf.copy()
    for i, c in enumerate(cols):
        train_pdf_enc[c] = X_train_enc[:, i]
        test_pdf_enc[c] = X_test_enc[:, i]
    return train_pdf_enc, test_pdf_enc, elapsed


def encode_catenc_te(train_pdf, test_pdf, cols):
    """category_encoders.TargetEncoder (no OOF)."""
    t0 = time.perf_counter()
    enc = ce.TargetEncoder(cols=cols, return_df=True)
    train_enc = enc.fit_transform(train_pdf[cols + ["target"]], train_pdf["target"])
    test_enc = enc.transform(test_pdf[cols + ["target"]])
    elapsed = time.perf_counter() - t0
    train_enc["target"] = train_pdf["target"].values
    test_enc["target"] = test_pdf["target"].values
    return train_enc, test_enc, elapsed


def encode_catenc_woe(train_pdf, test_pdf, cols):
    """category_encoders.WOEEncoder (no OOF)."""
    t0 = time.perf_counter()
    enc = ce.WOEEncoder(cols=cols, return_df=True)
    train_enc = enc.fit_transform(train_pdf[cols + ["target"]], train_pdf["target"])
    test_enc = enc.transform(test_pdf[cols + ["target"]])
    elapsed = time.perf_counter() - t0
    train_enc["target"] = train_pdf["target"].values
    test_enc["target"] = test_pdf["target"].values
    return train_enc, test_enc, elapsed


def encode_blueprint_cv3(train_df, test_df, cols):
    """Blueprint.target_encode(cv=3) end-to-end."""
    from polars_ds.pipeline import Blueprint
    t0 = time.perf_counter()

    bp = Blueprint(train_df.lazy(), target="target")
    bp.target_encode(cols=cols, target="target", cv=3)
    df_train_lazy, pipe = bp.materialize(return_df=True)
    train_enc = df_train_lazy.collect()
    test_enc = pipe.transform(test_df)
    elapsed = time.perf_counter() - t0
    return train_enc, test_enc, elapsed


def evaluate(train_enc, test_enc, cols, is_pandas=False):
    """Train LogReg on encoded features, return (train_auc, test_auc)."""
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

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, lr.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    return train_auc, test_auc


def run_all():
    print("Loading Adult Census dataset...")
    df = load_adult()
    print(f"  shape: {df.shape}, positive rate: {df['target'].mean():.3f}")

    all_results = []

    for repeat in range(N_REPEATS):
        seed = 42 + repeat
        train_df, test_df = df[:36000], df[36000:]
        # Shuffle train for fold diversity
        train_df = train_df.sample(fraction=1.0, seed=seed)

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

        # 4. sklearn_TE_cv5
        tr, te, t = encode_sklearn_te(train_pdf, test_pdf, CAT_COLS, cv=5)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS, is_pandas=True)
        variants["sklearn_TE_cv5"] = (auc_tr, auc_te, t)

        # 5. sklearn_TE_cv3
        tr, te, t = encode_sklearn_te(train_pdf, test_pdf, CAT_COLS, cv=3)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS, is_pandas=True)
        variants["sklearn_TE_cv3"] = (auc_tr, auc_te, t)

        # 6. catenc_TE_plain
        tr, te, t = encode_catenc_te(train_pdf, test_pdf, CAT_COLS)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS, is_pandas=True)
        variants["catenc_TE_plain"] = (auc_tr, auc_te, t)

        # 7. pds_WoE_plain
        tr, te, t = encode_pds_woe_plain(train_df, test_df, CAT_COLS)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS)
        variants["pds_WoE_plain"] = (auc_tr, auc_te, t)

        # 8. pds_OOF_WoE_cv3
        tr, te, t = encode_pds_woe_oof(train_df, test_df, CAT_COLS, cv=3)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS)
        variants["pds_OOF_WoE_cv3"] = (auc_tr, auc_te, t)

        # 9. catenc_WoE_plain
        tr, te, t = encode_catenc_woe(train_pdf, test_pdf, CAT_COLS)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS, is_pandas=True)
        variants["catenc_WoE_plain"] = (auc_tr, auc_te, t)

        # 10. Blueprint cv=3
        tr, te, t = encode_blueprint_cv3(train_df, test_df, CAT_COLS)
        auc_tr, auc_te = evaluate(tr, te, CAT_COLS)
        variants["pds_Blueprint_cv3"] = (auc_tr, auc_te, t)

        for name, (auc_tr, auc_te, elapsed) in variants.items():
            gap = auc_tr - auc_te
            print(f"  [{repeat+1}/{N_REPEATS}] {name:30s} train={auc_tr:.4f}  test={auc_te:.4f}  gap={gap:+.4f}  t={elapsed:.3f}s")
            all_results.append({
                "variant": name, "repeat": repeat,
                "train_auc": round(auc_tr, 6), "test_auc": round(auc_te, 6),
                "gap": round(gap, 6), "time_s": round(elapsed, 4),
            })

    # Aggregate: median over repeats
    print()
    print("=" * 100)
    header = f"{'variant':35s} {'train AUC':>10s} {'test AUC':>10s} {'gap':>8s} {'time,s':>8s}  {'std gap':>8s}"
    print(header)
    print("-" * 100)

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
        print(f"{v:35s} {line['train_auc_median']:10.4f} {line['test_auc_median']:10.4f} {line['gap_median']:+8.4f} {line['time_median']:8.3f}  ± {line['gap_std']:.4f}")

    print("=" * 100)

    out_path = RESULTS_DIR / "bench_real_adult.json"
    with open(out_path, "w") as f:
        json.dump({"raw": all_results, "summary": summary}, f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    run_all()
