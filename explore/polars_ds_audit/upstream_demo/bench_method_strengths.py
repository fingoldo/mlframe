"""Three synthetic scenarios designed to make each encoder variant win.

Each scenario constructs a dataset with characteristics that play to one
method's strengths. We then evaluate four variants on each scenario:

  1. plain TE (no OOF, leaky baseline)
  2. OOF sigmoid (cv=5, min_samples_leaf=20, smoothing=10) -- fixed smoothing
  3. OOF Bayes classical (per-category within-variance)
  4. OOF Bayes pooled (sklearn-style global within-variance)

A 5th column (group OOF) is shown only in scenario C, where row clustering
matters.

Goal: demonstrate that no single method dominates -- the right choice depends
on the cardinality / signal / clustering structure of the data.
"""
from __future__ import annotations
import time, json
import numpy as np
import polars as pl
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import polars_ds.exprs.num as pds_num

N_REPEATS = 5
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# Scenario builders
# --------------------------------------------------------------------------

def build_scenario_A(seed):
    """Sigmoid wins: many rare categories, imbalanced binary target.

    2000 categories with power-law occurrence (many singletons), n=10k rows,
    90% positive class. Adaptive Bayes over-shrinks rare cats; fixed sigmoid
    smoothing keeps a balanced floor of regularisation.
    """
    rng = np.random.default_rng(seed)
    n = 10_000
    n_cats = 2000
    # Power-law category frequencies; rank-1 cat is most common, rank-2000 has ~1 row
    ranks = np.arange(1, n_cats + 1)
    weights = 1.0 / ranks
    weights = weights / weights.sum()
    cats = rng.choice([f"c{i}" for i in range(n_cats)], size=n, p=weights)
    # Per-cat probability strongly correlated with rank (top cats more positive)
    cat_p = {f"c{i}": 0.6 + 0.35 * (1 - i / n_cats) for i in range(n_cats)}
    target = np.array([float(rng.random() < cat_p[c]) for c in cats])
    return pl.DataFrame({"cat": cats, "target": target})


def build_scenario_B(seed):
    """Bayes classical wins: heterogeneous within-category noise.

    100 categories, n=5000 rows. Half the categories are 'tight' (target is
    almost always one value) and half are 'noisy' (target close to 50/50).
    Per-category within-variance estimation correctly identifies which cats
    deserve trust and which need shrinkage; uniform sigmoid cannot.
    """
    rng = np.random.default_rng(seed)
    n = 5000
    n_cats = 100
    cats = rng.integers(0, n_cats, size=n)
    cat_p = {}
    for i in range(n_cats):
        if i % 2 == 0:
            cat_p[i] = 0.95 if rng.random() < 0.5 else 0.05  # tight, signal
        else:
            cat_p[i] = 0.45 + 0.10 * rng.random()           # noisy, near 50/50
    target = np.array([float(rng.random() < cat_p[c]) for c in cats])
    cat_str = np.array([f"c{c}" for c in cats])
    return pl.DataFrame({"cat": cat_str, "target": target})


def build_scenario_C(seed):
    """GroupKFold wins when train/test itself is split by group (unseen users
    at test time) -- which is the realistic deployment case for user-level
    features. Each user appears ONLY in train OR test, never both. A regular
    K-fold on train lets the model learn per-user target info that cannot
    generalise to unseen users; GroupKFold forces the OOF encoder to behave
    like the transform(test) path, where the user is always unseen.

    200 users, each has 5-50 rows, with a latent per-user Beta(2,2) rate plus
    moderate row noise. Returns the full DF plus a user array for grouping.
    """
    rng = np.random.default_rng(seed)
    n_users = 200
    user_sizes = rng.integers(5, 50, size=n_users)
    users = np.repeat(np.arange(n_users), user_sizes)
    user_p = rng.beta(2, 2, size=n_users)
    target = np.array([float(rng.random() < user_p[u]) for u in users])
    cat_str = np.array([f"u{u}" for u in users])
    df = pl.DataFrame({"cat": cat_str, "target": target,
                        "user": users.astype(np.int64)})
    return df


def split_by_group(df, frac=0.7, seed=0):
    """Train/test split where each group (user) is entirely on one side.

    This is the realistic setting for user-level categorical features:
    production sees users never observed at training time.
    """
    rng = np.random.default_rng(seed)
    unique_groups = df["user"].unique().to_numpy()
    rng.shuffle(unique_groups)
    cut = int(frac * len(unique_groups))
    train_groups = set(unique_groups[:cut].tolist())
    train_mask = df["user"].is_in(list(train_groups)).to_numpy()
    train_df = df.filter(pl.Series(train_mask))
    test_df = df.filter(pl.Series(~train_mask))
    return train_df, test_df


# --------------------------------------------------------------------------
# Encoders (return train_enc, test_enc)
# --------------------------------------------------------------------------

def make_kfold_ids(n, cv, seed=42):
    """Random K-fold (no stratification, fast)."""
    rng = np.random.default_rng(seed)
    fold_ids = np.zeros(n, dtype=np.uint32)
    indices = rng.permutation(n)
    for k, chunk in enumerate(np.array_split(indices, cv)):
        fold_ids[chunk] = k
    return fold_ids


def make_groupkfold_ids(groups, cv, seed=42):
    """sklearn GroupKFold-style: every row of a group in the same fold."""
    sk_gkf = GroupKFold(n_splits=cv)
    fold_ids = np.zeros(len(groups), dtype=np.uint32)
    for k, (_, val_idx) in enumerate(sk_gkf.split(np.zeros(len(groups)), groups=groups)):
        fold_ids[val_idx] = k
    return fold_ids


def encode_plain(train_df, test_df, target="target"):
    mapping = train_df.select(
        pds_num.target_encode("cat", target, min_samples_leaf=20, smoothing=10.0).implode()
    ).to_series()[0]
    old = mapping.struct.field("value"); new = mapping.struct.field("to")
    train_enc = train_df.with_columns(
        pl.col("cat").replace_strict(old=old, new=new, default=float(train_df[target].mean())).alias("cat_enc")
    )
    test_enc = test_df.with_columns(
        pl.col("cat").replace_strict(old=old, new=new, default=float(train_df[target].mean())).alias("cat_enc")
    )
    return train_enc, test_enc


def encode_oof(train_df, test_df, mode, fold_ids=None, target="target", cv=5):
    """mode in {'sigmoid', 'bayes_classical', 'bayes_pooled'}."""
    n = len(train_df)
    if fold_ids is None:
        fold_ids = make_kfold_ids(n, cv)
    fold_series = pl.Series("__fold__", fold_ids, dtype=pl.UInt32)
    n_folds = int(fold_series.max()) + 1

    smooth_auto = (mode in ("bayes_classical", "bayes_pooled"))
    bayes_variant = "pooled" if mode == "bayes_pooled" else "classical"
    train_target_mean = float(train_df[target].mean())

    oof_expr = pds_num.target_encode_oof(
        "cat", target, "__fold__",
        n_folds=n_folds,
        min_samples_leaf=20, smoothing=10.0,
        default=train_target_mean,
        smooth_auto=smooth_auto,
        bayes_variant=bayes_variant,
    )
    train_enc = train_df.with_columns(fold_series).with_columns(
        oof_expr.alias("cat_enc")
    ).drop("__fold__")

    # Build full-train mapping for test
    if smooth_auto:
        full_expr = pds_num.target_encode_bayes("cat", target, bayes_variant=bayes_variant)
    else:
        full_expr = pds_num.target_encode("cat", target, min_samples_leaf=20, smoothing=10.0)
    mapping = train_df.select(full_expr.implode()).to_series()[0]
    old = mapping.struct.field("value"); new = mapping.struct.field("to")
    test_enc = test_df.with_columns(
        pl.col("cat").replace_strict(old=old, new=new, default=train_target_mean).alias("cat_enc")
    )
    return train_enc, test_enc


def evaluate(train_enc, test_enc, target="target"):
    X_tr = train_enc["cat_enc"].to_numpy().reshape(-1, 1).astype(float)
    X_te = test_enc["cat_enc"].to_numpy().reshape(-1, 1).astype(float)
    y_tr = train_enc[target].to_numpy()
    y_te = test_enc[target].to_numpy()
    X_tr = np.nan_to_num(X_tr, nan=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr, y_tr)
    return (
        float(roc_auc_score(y_tr, lr.predict_proba(X_tr)[:, 1])),
        float(roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])),
    )


# --------------------------------------------------------------------------
# Run all scenarios
# --------------------------------------------------------------------------

def split(df, frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(frac * len(df))
    return df[idx[:cut].tolist()], df[idx[cut:].tolist()]


def run_scenario(name, builder, with_group=False, use_group_split=False):
    print(f"\n=== {name} ===")
    rows = []
    for r in range(N_REPEATS):
        df = builder(seed=r)
        if use_group_split:
            train_df, test_df = split_by_group(df, frac=0.7, seed=r)
        else:
            train_df, test_df = split(df, frac=0.7, seed=r)
        n_cats_train = train_df["cat"].n_unique()
        print(f"  rep {r+1}/{N_REPEATS}  train n={len(train_df)}  test n={len(test_df)}  "
              f"train cats={n_cats_train}")

        # plain
        tr, te = encode_plain(train_df, test_df)
        atr, ate = evaluate(tr, te)
        rows.append({"rep": r, "method": "plain TE", "train_auc": atr, "test_auc": ate})

        # OOF sigmoid
        tr, te = encode_oof(train_df, test_df, mode="sigmoid", cv=5)
        atr, ate = evaluate(tr, te)
        rows.append({"rep": r, "method": "OOF sigmoid cv=5", "train_auc": atr, "test_auc": ate})

        # OOF Bayes classical
        tr, te = encode_oof(train_df, test_df, mode="bayes_classical", cv=5)
        atr, ate = evaluate(tr, te)
        rows.append({"rep": r, "method": "OOF Bayes classical cv=5", "train_auc": atr, "test_auc": ate})

        # OOF Bayes pooled
        tr, te = encode_oof(train_df, test_df, mode="bayes_pooled", cv=5)
        atr, ate = evaluate(tr, te)
        rows.append({"rep": r, "method": "OOF Bayes pooled cv=5", "train_auc": atr, "test_auc": ate})

        if with_group:
            # Random KFold OOF (default behavior, leaks group info)
            # vs GroupKFold OOF via fold_col
            train_groups = train_df["user"].to_numpy()
            gk_folds = make_groupkfold_ids(train_groups, cv=5)
            train_df_with_fold = train_df.with_columns(
                pl.Series("__fold__", gk_folds, dtype=pl.UInt32)
            )
            # use sigmoid OOF with custom fold_col
            n_folds = int(gk_folds.max()) + 1
            train_target_mean = float(train_df["target"].mean())
            oof_expr = pds_num.target_encode_oof(
                "cat", "target", "__fold__",
                n_folds=n_folds,
                min_samples_leaf=20, smoothing=10.0,
                default=train_target_mean,
                smooth_auto=False,
            )
            tr_g = train_df_with_fold.with_columns(oof_expr.alias("cat_enc")).drop("__fold__")
            full_expr = pds_num.target_encode("cat", "target", min_samples_leaf=20, smoothing=10.0)
            mapping = train_df.select(full_expr.implode()).to_series()[0]
            old = mapping.struct.field("value"); new = mapping.struct.field("to")
            te_g = test_df.with_columns(
                pl.col("cat").replace_strict(old=old, new=new, default=train_target_mean).alias("cat_enc")
            )
            atr, ate = evaluate(tr_g, te_g)
            rows.append({"rep": r, "method": "OOF GroupKFold (fold_col)", "train_auc": atr, "test_auc": ate})

    # Aggregate
    methods = list(dict.fromkeys(r["method"] for r in rows))
    print()
    print(f"  {'method':30s} {'train AUC':>11s} {'test AUC':>10s} {'gap':>9s}")
    print(f"  {'-'*30} {'-'*11} {'-'*10} {'-'*9}")
    summary = []
    best_test = -1.0
    best_method = None
    for m in methods:
        ms = [r for r in rows if r["method"] == m]
        med_tr = float(np.median([r["train_auc"] for r in ms]))
        med_te = float(np.median([r["test_auc"] for r in ms]))
        gap = med_tr - med_te
        line = {"method": m, "train_auc_med": med_tr, "test_auc_med": med_te, "gap_med": gap}
        summary.append(line)
        if med_te > best_test:
            best_test = med_te; best_method = m
        print(f"  {m:30s} {med_tr:11.4f} {med_te:10.4f} {gap:+9.4f}")
    print(f"  >>> winner (test AUC): {best_method}")
    return {"name": name, "rows": rows, "summary": summary, "winner": best_method}


def main():
    print("=" * 80)
    print("Method-strengths benchmark: which encoder wins on which data?")
    print("=" * 80)
    results = []
    results.append(run_scenario("A. Power-law cardinality + imbalanced binary",
                                  build_scenario_A))
    results.append(run_scenario("B. Heterogeneous within-cat noise (tight + noisy mix)",
                                  build_scenario_B))
    results.append(run_scenario(
        "C. Group-split (unseen users at test time, 5-50 rows per user)",
        build_scenario_C, with_group=True, use_group_split=True,
    ))
    print()
    print("=" * 80)
    print("Per-scenario winner:")
    for r in results:
        print(f"  {r['name'][:55]:55s}  -> {r['winner']}")
    print("=" * 80)
    out_path = RESULTS_DIR / "bench_method_strengths.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
